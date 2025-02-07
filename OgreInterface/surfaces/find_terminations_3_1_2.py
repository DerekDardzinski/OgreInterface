import argparse
from copy import deepcopy
from itertools import combinations
import math
import os
import sys
from typing import Dict, List, Tuple, TypeVar, Union
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Colormap

from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import JmolNN
from pymatgen.core.lattice import Lattice
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.structure import Molecule, Structure
from pymatgen.transformations.site_transformations import TranslateSitesTransformation
from pymatgen.transformations.standard_transformations import RotationTransformation
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage

from OgreInterface.surfaces.oriented_bulk import OrientedBulk
from OgreInterface.utils import get_unique_miller_indices

import inspect
import time

SelfSurfacePrism = TypeVar("SelfSurfacePrism", bound="SurfacePrism")


class SurfacePrism:
    """A surface-orthogonal parallelipiped with unbounded bases.

    The SurfacePrism class performs the analytical geometry needed to exclude points that are
    located outside of a surface-orthogonal parallelipiped with unbounded bases.

    Examples:
        Creating a SurfacePrism object from a numpy ndarray consisting of the Cartesian coordinates
            of three consecutive adjacent vertices of the bounding surface parallelogram:
        >>> from OgreInterface.surfaces.terminations import SurfacePrism
        >>> import numpy as np
        >>> vertex_1 = np.array([[0], [1], [2]])
        >>> vertex_2 = np.array([[3], [4], [5]])
        >>> vertex_3 = np.array([[6], [7], [8]])
        >>> vertices = np.hstack((vertex_1, vertex_2, vertex_3))
        >>> surface_prism = SurfacePrism(vertices=vertices)

        Creating a SurfacePrism object from the lattice matrix of a unit cell via the from_matrix()
            class method:
        >>> from OgreInterface.surfaces.terminations import SurfacePrism
        >>> from pymatgen.core.structure import Structure
        >>> structure = Structure.from_file('structure.cif')
        >>> matrix = structure.lattice.matrix
        >>> surface_prism = SurfacePrism.from_matrix(matrix=matrix)

        Creating a SurfacePrism object from a pymatgen.core.structure Structure object via the
            from_structure() class method:
        >>> from OgreInterface.surfaces.terminations import SurfacePrism
        >>> from pymatgen.core.structure import Structure
        >>> structure = Surface.from_file('structure.cif')
        >>> surface_prism = SurfacePrism.from_structure(structure=structure)

    Arguments:
        vertices (numpy.ndarray): Column-wise array of three or four consecutive adjacent
            vertices of the bounding surface parallelogram in Cartesian coordinate space.
        surface_normal (numpy.ndarray): A normal vector of the surface in Cartesian coordinate
            space. If not provided, then it will be computed from the passed vertices. Passing a
            surface_normal that is not derived from the same unit cell from which vertices is
            derived may lead to unexpected behavior.

    Attributes:
        vertices (numpy.ndarray): Column-wise 3x4 array of the four consecutive adjacent vertices
            of the bounding surface parallelogram in Cartesian coordinate space.
        surface_normal (numpy.ndarray): Unit normal vector of the surface in Cartesian coordinate
            space.

    Methods:
        mask_points(points: numpy.ndarray) -> numpy.ndarray: Mask the passed array of Cartesian
            points.
        mask_structure(structure: Structure, in_place: bool = False) -> Structure: Mask the passed
            pymatgen Structure's sites.

    Class methods:
        from_matrix(matrix: numpy.ndarray, surface_normal: numpy.ndarray = None) -> SurfacePrism:
            Create a SurfacePrism object from the passed lattice matrix.

    Protected attributes:
        _structure (pymatgen.core.structure Structure):
        _face_plane_equations ((4,4) numpy.ndarray):
        _inside_sides ((4,) numpy.ndarray):
        _inward_face_normals (() numpy.ndarray):

    """

    def __init__(self, vertices: np.ndarray, surface_normal: np.ndarray = None):
        self._vertices = (
            self._complete_vertices(vertices) if vertices.shape[1] == 3 else vertices.copy()
        )
        self._surface_normal = (
            self._calc_surface_normal()
            if surface_normal is None
            else surface_normal.copy() / np.linalg.norm(surface_normal.copy())
        )
        self._structure = None
        self._face_plane_equations = self._calc_face_plane_equations()
        self._inside_sides = self._calc_inside_sides()
        self._inward_face_normals = self._calc_inward_face_normals()

    @property
    def vertices(self) -> np.ndarray:
        return self._vertices.copy()

    @property
    def surface_normal(self) -> np.ndarray:
        return self._surface_normal.copy()

    @property
    def structure(self) -> Structure:
        return self._structure.copy() if self._structure else None

    @property
    def face_plane_equations(self) -> np.ndarray:
        return self._face_plane_equations.copy()

    @property
    def inside_sides(self) -> np.ndarray:
        return self._inside_sides.copy()

    @property
    def inward_face_normals(self) -> np.ndarray:
        return self._inward_face_normals.copy()

    def _complete_vertices(self, vertices: np.ndarray) -> np.ndarray:
        """Append the missing fourth vertex."""

        return np.hstack(vertices, vertices[:, 0] - vertices[:, 1] + vertices[:, 1])

    def _calc_surface_normal(self) -> np.ndarray:
        """Compute the unit normal vector to the surface."""

        AB = self._vertices[:, 1] - self._vertices[:, 0]
        AD = self._vertices[:, 3] - self._vertices[:, 0]
        surface_normal = np.cross(AB, AD)
        surface_normal /= np.linalg.norm(surface_normal)
        return surface_normal

    def _calc_face_plane_equations(self) -> np.ndarray:
        """Calculate the coefficients of the plane equations of the faces of the parallelipiped."""

        next_vertices = np.roll(self._vertices, -1, axis=1)
        face_plane_equations = np.zeros((4, 4))
        for i in range(4):
            normal_vector = np.cross(
                self._vertices[:, i] - next_vertices[:, i], self._surface_normal
            )
            normal_vector = normal_vector / np.linalg.norm(normal_vector)
            D = -1.0 * normal_vector @ self._vertices[:, i]
            face_plane_equations[i, :] = np.append(normal_vector, D)
        return np.array(face_plane_equations)

    def _calc_inside_sides(self) -> np.ndarray:
        """Compute example outputs of a point on the inside side of each face."""

        homogeneous_vertices = homogenize(self._vertices)

        return np.array(
            [
                self._face_plane_equations[0] @ homogeneous_vertices[:, 2],
                self._face_plane_equations[1] @ homogeneous_vertices[:, 3],
                self._face_plane_equations[2] @ homogeneous_vertices[:, 0],
                self._face_plane_equations[3] @ homogeneous_vertices[:, 1],
            ]
        )

    def _inside(self, homogenenous_point: np.ndarray) -> bool:
        """Determine whether a point is inside/on the parallelipiped."""

        return np.all((self._face_plane_equations @ homogenenous_point) * self._inside_sides >= 0)

    def _calc_inward_face_normals(self) -> np.ndarray:
        inward_face_normals = []

        for i in range(4):
            inward_face_normals.append(
                normalize(
                    np.append(
                        self._face_plane_equations[i, :2], 0
                    )  # Maybe do self._face_plane_equations[i, :-1] instead!
                    * math.copysign(1, self._inside_sides[i])
                )
            )

        return np.array(inward_face_normals)

    def _infiltrates_any_face(self, site: PeriodicSite) -> bool:
        innermost_vdW_corners = (
            site.specie.van_der_waals_radius * self._inward_face_normals + site.coords
        ).T
        return np.any(self._mask(points=innermost_vdW_corners))

    def _mask(self, points: np.ndarray) -> np.ndarray:
        """Compute the mask of the passed Cartesian points."""
        pts = homogenize(points) if points.shape[0] == 3 else points
        return np.apply_along_axis(func1d=self._inside, axis=0, arr=pts)

    def mask_points(self, points: np.ndarray) -> np.ndarray:
        """Mask the passed array of Cartesian points.

        Mask the passed column-wise array of Cartesian points to exclude all points located outside
        of the surface-orthogonal parallelipiped with unbounded bases. Points located exactly on
        the surface(s) of one or two of the parallelipiped's faces are included in addition to
        points located inside of the parallelipiped.

        Arguments:
            points (numpy.ndarray): Column-wise array of Cartesian points to be masked.
        """

        return points[:, self._mask(points=points)]

    def mask_structure(
        self, structure: Structure = None, in_place: bool = False, snug_corner=False
    ) -> Structure:
        """Mask the passed pymatgen Structure.

        Mask the passed pymatgen Structure's sites to exclude all sites located outside of the
        surface-orthogonal parallelipiped with unbounded bases. A 3x3x1 supercell or a nonperiodic
        cell rather than a periodic unit cell is recommended, as a periodic unit cell may undo the
        masking by applying periodic boundary conditions.

        Arguments:
            structure (pymatgen.core.structure Structure): Structure whose sites are to be masked. A
                3x3x1 supercell or a nonperiodic cell rather than a periodic unit cell is
                recommended, as a periodic unit cell may undo the masking by applying periodic
                boundary conditions. Default is None, in which case the ._structure attribute will
                be used. In that case, the SurfacePrism object must have been instantiated via the
                SurfacePrism.from_structure() class method or the ._structure attribute must have
                been set manually.
            in_place (bool): Whether to edit the passed Structure in-place. If False,
                mask_structure() will also return a masked copy of the passed Structure without
                affecting the passed Structure. Default is False.
        """

        structure = self._structure.copy() if structure is None else structure
        if structure is None:
            raise ValueError(
                "kwarg 'structure' is required unless SurfacePrism object was instantiated via \
                    SurfacePrism.from_structure()"
            )

        if in_place and not snug_corner:
            structure.remove_sites(
                indices=np.where(self._mask(points=structure.cart_coords.T) == False)[0]
            )
            return structure

        masked_structure = structure.copy()

        masked_structure.remove_sites(
            indices=np.where(self._mask(points=structure.cart_coords.T) == False)[0]
        )

        if not snug_corner:
            return masked_structure

        translation_vector = -1.0 * np.append(masked_structure.lattice.matrix[2, :2], 0.0)
        snug_coords = masked_structure.cart_coords + translation_vector
        snug_matrix = masked_structure.lattice.matrix.copy()
        snug_matrix[-1, :2] = 0.0
        snug_lattice = Lattice(matrix=snug_matrix, pbc=(False, False, False))
        struc = Structure(
            lattice=snug_lattice,
            species=masked_structure.species,
            coords=snug_coords,
            coords_are_cartesian=True,
        )
        return struc

    def buffer_mask_supercell(
        self,
        supercell: Structure,
        in_place: bool = False,
    ) -> Structure:
        to_remove = []
        for i, site in enumerate(supercell):
            if site.coords[1] < 0:
                sys.stdout.write(f"{site.coords[1]} < 0!\n")
            if not self._infiltrates_any_face(site=site):
                to_remove.append(i)

        if in_place:
            supercell.remove_sites(indices=to_remove)
            return supercell

        masked_supercell = supercell.copy()
        masked_supercell.remove_sites(indices=to_remove)
        return masked_supercell

    @classmethod
    def from_matrix(cls, matrix: np.ndarray, surface_normal: np.ndarray = None) -> SelfSurfacePrism:
        """Create a SurfacePrism object from the passed lattice matrix.

        Create a SurfacePrism object from the passed lattice matrix (i.e., row-wise numpy
        ndarray of lattice vectors). If a normal vector of the surface in Cartesian coordinate space
        is not provided, then it will be computed from the passed lattice matrix.

        Arguments:
            matrix (numpy.ndarray): The lattice matrix, i.e., row-wise numpy ndarray of lattice
                vectors, from which the SurfacePrism object is to be constructed. The "surface,"
                according to the SurfacePrism object, is coplanar with the top (in the c-direction)
                ab-face of the unit cell that this lattice matrix describes.
            surface_normal (numpy.ndarray): A normal vector of the surface in Cartesian coordinate
                space. If not provided, then it will be computed from the passed lattice matrix.
                Passing a surface_normal that is not derived from the same unit cell from which
                matrix is derived may lead to unexpected behavior.
        """

        a, b, c = matrix
        vertices = np.vstack((c, c + b, c + b + a, c + a)).T

        if surface_normal is None:
            surface_norm = np.cross(a, b)
            surface_norm /= np.linalg.norm(surface_norm)
        else:
            surface_norm = surface_normal.copy()

        return cls(vertices=vertices, surface_normal=surface_norm)

    @classmethod
    def from_structure(
        cls, structure: Structure, surface_normal: np.ndarray = None
    ) -> SelfSurfacePrism:
        """Create a SurfacePrism object from the passed pymatgen Structure.

        Create a SurfacePrism object from the passed pymatgen.core.structure Structure object. If a
        normal vector of the surface in Cartesian coordinate space is not provided, then it will be
        computed from the lattice matrix of the passed Structure.

        Arguments:
            structure (pymatgen.core.structure Structure): The pymatgen Structure object from which
                the SurfacePrism object is to be constructed. The "surface," according to the
                SurfacePrism object, is coplanar with the top (in the c-direction) ab-face of the
                unit cell that this Structure represents.
            surface_normal (numpy.ndarray): A normal vector of the surface in Cartesian coordinate
                space. If not provided, then it will be computed from the lattice matrix of the
                passed Structure. Passing a surface_normal that is not derived from the passed
                Structure may lead to unexpected behavior.
        """

        surface_prism = cls.from_matrix(
            matrix=structure.lattice.matrix, surface_normal=surface_normal
        )
        surface_prism._structure = structure.copy()
        return surface_prism


class SurfaceCell:
    def __init__(
        self,
        original_cell: Structure,
        surface_normal: np.ndarray = None,
        molecule_width: float = None,
    ):
        self._original_cell = original_cell
        self._surface_normal = (
            surface_normal if not surface_normal is None else self._calc_surface_normal()
        )
        self._molecule_width = (
            molecule_width if not molecule_width is None else molecular_width(self._original_cell)
        )
        self._unit_height = self._original_cell.lattice.matrix[-1] @ self._surface_normal
        self._c_scale = math.ceil(self._molecule_width / self._unit_height)
        self._structure, self._height = self._derive_structure()
        self._bounds = self._calc_bounds()
        if debug:
            self._structure.to(os.path.join(os.getcwd(), "buffered_structure.cif"))

    @property
    def original_cell(self) -> Structure:
        return self._original_cell.copy()

    @property
    def surface_normal(self) -> np.ndarray:
        return self._surface_normal.copy()

    @property
    def molecule_width(self) -> float:
        return self._molecule_width

    @property
    def unit_height(self) -> float:
        return self._unit_height

    @property
    def c_scale(self) -> int:
        return self._c_scale

    @property
    def structure(self) -> Structure:
        return self._structure.copy()

    @property
    def height(self) -> float:
        return self._height

    @property
    def bounds(self) -> np.ndarray:
        return self._bounds.copy()

    def _calc_surface_normal(self) -> np.ndarray:
        a, b, _ = self._original_cell.lattice.matrix
        return normalize(np.cross(normalize(a), normalize(b)))

    def _derive_structure(self) -> Structure:

        ceiling = self._c_scale * self._unit_height
        start_timer("raw_supercell")
        # Create the raw supercell
        raw_supercell = self._original_cell.make_supercell(
            scaling_matrix=[3, 3, self._c_scale], to_unit_cell=True, in_place=False
        )
        stop_timer("raw_supercell")
        start_timer("shifted_supercell")
        # Translate the supercell
        shifted_supercell = Structure(
            lattice=raw_supercell.lattice,
            species=raw_supercell.species,
            coords=raw_supercell.cart_coords
            - np.sum(self._original_cell.lattice.matrix[:2], axis=0),
            coords_are_cartesian=True,
        )
        stop_timer("shifted_supercell")
        start_timer("surface_prism")
        # Create the surface prism and mask the structure
        surface_prism = SurfacePrism.from_matrix(
            self._original_cell.lattice.matrix, self._surface_normal
        )
        stop_timer("surface_prism")
        start_timer("supercell_slug")
        supercell_slug = surface_prism.mask_structure(
            shifted_supercell, in_place=False, snug_corner=True
        )
        stop_timer("supercell_slug")

        # Collect coordinates and species to keep, and compute max head
        coords_to_keep, species_to_keep, max_head = [], [], 0.0
        start_timer("molecule_graphs")
        molecule_graphs = get_molecule_graphs(supercell_slug)
        stop_timer("molecule_graphs")
        start_timer("slug planing")
        for molecule_graph in molecule_graphs:  # get_molecule_graphs(supercell_slug):
            keep_molecule, local_max_head = True, 0.0

            for node in molecule_graph.nodes:
                site = supercell_slug[node]
                top = site.coords[-1] + site.specie.van_der_waals_radius

                if top > ceiling:
                    keep_molecule = False
                    break

                local_max_head = max(local_max_head, top)

            if keep_molecule:
                max_head = max(max_head, local_max_head)
                for node in molecule_graph.nodes:
                    site = supercell_slug[node]
                    coords_to_keep.append(site.coords)
                    species_to_keep.append(site.specie)
        stop_timer("slug planing")

        # Update lattice matrix with the maximum head
        matrix = self._original_cell.lattice.matrix.copy()
        matrix[-1] = np.array([0.0, 0.0, max_head])
        start_timer("structure")
        # Create the structure
        structure = Structure(
            lattice=Lattice(matrix=matrix),
            species=species_to_keep,
            coords=coords_to_keep,
            coords_are_cartesian=True,
        )
        stop_timer("structure")

        # Make the supercell and buffer it
        start_timer("buffered_prism")
        buffered_prism = SurfacePrism.from_matrix(structure.lattice.matrix, self._surface_normal)
        stop_timer("buffered_prism")
        start_timer("buffered_supercell")
        structure.make_supercell([3, 3, 1], to_unit_cell=True, in_place=True)
        stop_timer("buffered_supercell")
        start_timer("buffered mask")
        buffered_prism.buffer_mask_supercell(structure, in_place=True)
        stop_timer("buffered mask")

        return structure, max_head

    def _calc_bounds(self) -> np.ndarray:
        a, bx, by = (
            self._original_cell.lattice.matrix[0, 0],
            self._original_cell.lattice.matrix[1, 0],
            self._original_cell.lattice.matrix[1, 1],
        )
        x_bounds = [bx, a] if bx < 0 else [0, bx + a]
        y_bounds = [by, 0] if by < 0 else [0, by]
        z_bounds = [0, self._height]
        return np.vstack((x_bounds, y_bounds, z_bounds))


class SurfaceVoxels:
    def __init__(
        self,
        unit_cell: Structure,
        attrv_adj: float = 0.0,
        precision: int = 26,
        scan_step: float = 0.1,
        delta_z: float = 0.0,
        molecule_width: float = None,
    ):
        self._unit_cell = unit_cell
        self._attrv_adj = attrv_adj
        self._precision = precision
        self._scan_step = scan_step
        self._molecule_width = (
            molecule_width if not molecule_width is None else molecular_width(self._unit_cell)
        )
        self._delta_z = delta_z if delta_z != 0.0 else self._molecule_width

        self._surface_normal = self._calc_surface_normal()
        start_timer("SurfaceVoxels.surface_cell")
        self._surface_cell = SurfaceCell(
            original_cell=self.unit_cell,
            surface_normal=self.surface_normal,
            molecule_width=self.molecule_width,
        )
        self._bounds = self._surface_cell.bounds
        stop_timer("SurfaceVoxels.surface_cell")
        start_timer("SurfaceVoxels._voxelize()")
        self._voxel_surface, self._masked_points = self._voxelize()
        stop_timer("SurfaceVoxels._voxelize()")
        start_timer("SurfaceVoxels._roughnesses()")
        self._roughnesses = self._calc_roughnesses()
        self._average_roughness = self._calc_average_roughness()
        stop_timer("SurfaceVoxels._roughnesses()")

    @property
    def unit_cell(self) -> Structure:
        return self._unit_cell.copy()

    @property
    def attrv_adj(self) -> float:
        return self._attrv_adj

    @property
    def precision(self) -> int:
        return self._precision

    @property
    def scan_step(self) -> float:
        return self._scan_step

    @property
    def molecule_width(self) -> float:
        return self._molecule_width

    @property
    def delta_z(self) -> float:
        return self._delta_z

    @property
    def surface_normal(self) -> np.ndarray:
        return self._surface_normal.copy()

    @property
    def surface_cell(self) -> SurfaceCell:
        return deepcopy(self._surface_cell)

    @property
    def bounds(self) -> np.ndarray:
        return self._bounds.copy()

    @property
    def voxel_surface(self) -> np.ndarray:
        return self._voxel_surface.copy()

    @property
    def masked_points(self) -> np.ndarray:
        return self._masked_points.copy()

    @property
    def roughnesses(self) -> np.ndarray:
        return self._roughnesses.copy()

    @property
    def average_roughness(self) -> float:
        return self._average_roughness

    def _calc_surface_normal(self) -> np.ndarray:
        a, b, _ = self._unit_cell.lattice.matrix
        return normalize(np.cross(normalize(a), normalize(b)))

    def _voxelize(self) -> Tuple[np.ndarray, np.ndarray]:
        max_z = self._bounds[2, 1]

        target_sites = [
            site
            for site in self._surface_cell.structure
            if max_z - site.coords[-1] - site.specie.van_der_waals_radius < self._delta_z
        ]

        xi = np.arange(self._bounds[0, 0], self._bounds[0, 1], self._scan_step)
        yi = np.arange(self._bounds[1, 0], self._bounds[1, 1], self._scan_step)
        zi = np.arange(self._bounds[2, 0], self._bounds[2, 1], self._scan_step)

        xn = len(xi)
        yn = len(yi)
        zn = len(zi)

        voxel_array = np.zeros((xn, yn, zn))

        thetas = np.linspace(0.0, math.pi / 2, self._precision)
        phis_template = np.linspace(0.0, math.pi * 2, 4 * self._precision)

        for site in tqdm(target_sites, desc="Voxelizing sites..."):
            x, y, z = site.coords
            rad = site.specie.van_der_waals_radius + self._attrv_adj

            for theta in thetas:
                z_i = int((z + rad * np.cos(theta) - self._bounds[2, 0]) / self._scan_step) - 1
                if z_i < 0 or z_i > zn:
                    break

                sin_theta = np.sin(theta)
                phis = phis_template[: len(thetas) * 4]
                sin_phis = np.sin(phis)
                cos_phis = np.cos(phis)

                for cos_phi, sin_phi in zip(cos_phis, sin_phis):
                    x_i = (
                        int((x + rad * sin_theta * cos_phi - self._bounds[0, 0]) / self._scan_step)
                        - 1
                    )
                    y_i = (
                        int((y + rad * sin_theta * sin_phi - self._bounds[1, 0]) / self._scan_step)
                        - 1
                    )

                    if 0 <= x_i < xn and 0 <= y_i < yn:
                        voxel_array[x_i, y_i, z_i] = 1

        voxel_surface = np.zeros((xn, yn))
        points = []
        for x_index in range(xn):
            x_coord = x_index * self._scan_step
            for y_index in range(yn):
                true_z_indices = np.where(voxel_array[x_index, y_index, :] == 1)[0]
                height = true_z_indices[-1] * self._scan_step if len(true_z_indices) > 0 else 0.0
                voxel_surface[x_index, y_index] = height
                points.append([x_coord, y_index * self._scan_step, height])

        if len(points) == 0:
            sys.stdout.write("\nThere are no occupied points!\n\n")

        return voxel_surface, np.array(points).T

    def _calc_roughnesses(self) -> np.ndarray:
        """Compute roughnesses from masked voxel surface heights"""
        return -1.0 * self._masked_points[-1] + self._surface_cell.height

    def _calc_average_roughness(self) -> float:
        return np.mean(self._roughnesses)

    def visualize(self, save_to: str = "", cmap: Union[str, Colormap] = "viridis") -> None:
        fig, ax = plt.subplots()
        im = ax.imshow(
            self._voxel_surface.T,
            origin="lower",
            extent=(
                self._bounds[0][0],
                self._bounds[0][1],
                self._bounds[1][0],
                self._bounds[1][1],
            ),
            cmap=cmap,
            interpolation="nearest",
            aspect="equal",
        )

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label("Surface Height ($\AA$)", fontsize=16, fontweight="bold", rotation=90.0)
        ax.set_xlabel("x-Coordinate ($\AA$)", fontsize=16, fontweight="bold")
        ax.set_ylabel("y-Coordinate ($\AA$)", fontsize=16, fontweight="bold", rotation=90.0)
        ax.set_title("Surface Height Map", fontsize=20, fontweight="bold")
        cbar_tick_labels = cbar.ax.get_yticklabels()
        for label in cbar_tick_labels:
            label.set_fontsize(12)
            label.set_fontweight("bold")
        xticklabels = ax.get_xticklabels()
        yticklabels = ax.get_yticklabels()
        for label in xticklabels:
            label.set_fontsize(12)
            label.set_fontweight("bold")
        for label in yticklabels:
            label.set_fontsize(12)
            label.set_fontweight("bold")

        if save_to:
            plt.savefig(save_to)
        plt.close()


class Termination:
    def __init__(
        self,
        rotated_dummy_obs: Structure,
        shift: float,
        average_roughness: float,
        attrv_adj: float,
        precision: int,
        scan_step: float,
        delta_z: float,
        smoothest: bool,
        undummify: bool = False,
        unrotate: bool = False,
        unrotation_transformations: Tuple[RotationTransformation] = None,
        unrotated_dummy_structure: Structure = None,
        rotated_molecular_structure: Structure = None,
        unrotated_molecular_structure: Structure = None,
    ):
        self._rotated_dummy_obs = deepcopy(rotated_dummy_obs)
        self._structures = {
            "rotated": {"dummy": self._rotated_dummy_obs.oriented_bulk_structure, "molecular": rotated_molecular_structure},
            "unrotated": {
                "dummy": unrotated_dummy_structure,
                "molecular": unrotated_molecular_structure,
            },
        }
        self._shift = shift
        self._average_roughnesses = [average_roughness]
        self._roughness_calc_params = [
            {
                "attrv_adj": attrv_adj,
                "precision": precision,
                "scan_step": scan_step,
                "delta_z": delta_z,
            }
        ]
        self._smoothest = [smoothest]
        self._unrotation_transformations = unrotation_transformations

        if unrotate and unrotated_dummy_structure is None:
            if unrotation_transformations is None:
                print("unrotation_transformations must be supplied in order to unrotate.")
            else:
                self._structures["unrotated"]["dummy"] = self._unrotate(
                    self._structures["rotated"]["dummy"]
                )

        if undummify:
            self._structures["rotated"]["molecular"] = add_molecules(
                structure=self._structures["rotated"]["dummy"]
            )
            if self._structures["unrotated"]["dummy"]:
                self._structures["unrotated"]["molecular"] = add_molecules(
                    structure=self._structures["unrotated"]["dummy"]
                )

    @property
    def rotated_dummy_obs(self) -> OrientedBulk:
        return deepcopy(self._rotated_dummy_obs)

    def _unrotate(
        self,
        structure: Structure,
        unrotation_transformations: Tuple[RotationTransformation] = None,
        overwrite: bool = False,
    ) -> Structure:
        transformations = (
            unrotation_transformations
            if unrotation_transformations
            else self._unrotation_transformations
        )
        if unrotation_transformations and overwrite:
            self._unrotation_transformations = unrotation_transformations

        intermediate = transformations[0].apply_transformation(structure)
        return transformations[1].apply_transformation(intermediate)

    def append_roughness_calc(
        self,
        average_roughness: float,
        attrv_adj: float,
        precision: int,
        scan_step: float,
        delta_z: float,
        smoothest: bool,
    ) -> None:

        self._average_roughnesses.append(average_roughness)
        self._roughness_calc_params.append(
            {
                "attrv_adj": attrv_adj,
                "precision": precision,
                "scan_step": scan_step,
                "delta_z": delta_z,
            }
        )
        self._smoothest.append(smoothest)

    @property
    def structures(self) -> Dict:
        return deepcopy(self._structures)

    def structure(
        self,
        unrotate: bool = True,
        dummy: bool = False,
        unrotation_transformations: Tuple[RotationTransformation] = None,
        overwrite: bool = False,
    ) -> Structure:
        if unrotate:
            return (
                self.unrotated_dummy_structure(
                    unrotation_transformations=unrotation_transformations, overwrite=overwrite
                )
                if dummy
                else self.unrotated_molecular_structure(
                    unrotation_transformations=unrotation_transformations, overwrite=overwrite
                )
            )
        return self.rotated_dummy_structure if dummy else self.rotated_molecular_structure

    @property
    def rotated_dummy_structure(self) -> Structure:
        return self._structures["rotated"]["dummy"].copy()

    def unrotated_dummy_structure(
        self,
        unrotation_transformations: Tuple[RotationTransformation] = None,
        overwrite: bool = False,
    ) -> Structure:
        if self._structures["unrotated"]["dummy"]:
            return self._structures["unrotated"]["dummy"].copy()

        unrotated = (
            self._unrotate(
                structure=self._structures["rotated"]["dummy"],
                unrotation_transformations=unrotation_transformations,
                overwrite=overwrite,
            )
            if unrotation_transformations
            else self._unrotate(
                structure=self._structures["rotated"]["dummy"],
                unrotation_transformations=self._unrotation_transformations,
                overwrite=False,
            )
        )
        if overwrite:
            self._structures["unrotated"]["dummy"] = unrotated
            return unrotated.copy()
        return unrotated

    def rotated_molecular_structure(self, overwrite: bool = False) -> Structure:
        if self._structures["rotated"]["molecular"]:
            return self._structures["rotated"]["molecular"].copy()

        molecular = add_molecules(structure=self._rotated_dummy_obs)
        if overwrite:
            self._structures["rotated"]["molecular"] = molecular
            return molecular.copy()
        return molecular

    def unrotated_molecular_structure(
        self,
        unrotation_transformations: Tuple[RotationTransformation] = None,
        overwrite: bool = False,
    ) -> Structure:
        if self._structures["unrotated"]["molecular"]:
            return self._structures["unrotated"]["molecular"].copy()
        
        if self._structures["rotated"]["molecular"] is None:
            molecular = add_molecules(self._structures["rotated"]["dummy"])
            if overwrite:
                self._structures["rotated"]["molecular"] = molecular
        else:
            molecular = self._structures["rotated"]["molecular"]

        unrotated = (
            self._unrotate(
                structure=molecular,
                unrotation_transformations=unrotation_transformations,
                overwrite=overwrite,
            )
            if unrotation_transformations
            else self._unrotate(
                structure=molecular,
                unrotation_transformations=self._unrotation_transformations,
                overwrite=False,
            )
        )  

        if overwrite:
            self._structures["unrotated"]["molecular"] = unrotated
            return unrotated.copy()
        return molecular

    @property
    def shift(self) -> float:
        return self._shift

    @property
    def average_roughness(self) -> Union[float, List[float]]:
        return (
            self._average_roughnesses[0]
            if len(self._average_roughnesses) == 1
            else self._average_roughnesses.copy()
        )

    @property
    def attrv_adj(self) -> Union[float, List[float]]:
        attrv_adj_list = [params["attrv_adj"] for params in self._roughness_calc_params]
        return attrv_adj_list[0] if len(attrv_adj_list) == 1 else attrv_adj_list

    @property
    def precision(self) -> Union[int, List[int]]:
        precision_list = [params["precision"] for params in self._roughness_calc_params]
        return precision_list[0] if len(precision_list) == 1 else precision_list

    @property
    def scan_step(self) -> Union[float, List[float]]:
        scan_step_list = [params["scan_step"] for params in self._roughness_calc_params]
        return scan_step_list[0] if len(scan_step_list) == 1 else scan_step_list

    @property
    def delta_z(self) -> Union[float, List[float]]:
        delta_z_list = [params["delta_z"] for params in self._roughness_calc_params]
        return delta_z_list[0] if len(delta_z_list) == 1 else delta_z_list

    @property
    def roughness_calc_params(self) -> Union[Dict, List[Dict]]:
        return (
            self._roughness_calc_params[0].copy()
            if len(self._roughness_calc_params) == 1
            else deepcopy(self._roughness_calc_params)
        )

    @property
    def is_smoothest(self) -> Union[bool, List[bool]]:
        return self._smoothest[0] if len(self._smoothest) == 1 else self._smoothest.copy()

    @property
    def unrotation_transformations(self) -> Tuple[RotationTransformation]:
        return self._unrotation_transformations

    @unrotation_transformations.setter
    def unrotation_transformations(
        self, unrotation_transformations: Tuple[RotationTransformation]
    ) -> None:
        self._unrotation_transformations = unrotation_transformations


class Terminator:
    def __init__(
        self,
        bulk: Structure,
        plane: List[int],
        clustering_tolerance_scale: float = 0.1,
        attrv_adj: float = 0.0,
        precision: int = 26,
        scan_step: float = 0.1,
        delta_z: float = 0.0,
        out_dir: str = os.getcwd(),
        species: str = "SPECIES",
        molecule_width: float = None,
        terminations: List[Termination] = None,
    ):
        self._bulk = bulk
        self._plane = plane
        self._obs = OrientedBulk(bulk=bulk, miller_index=plane, make_planar=False)

        self._clustering_tolerance_scale = clustering_tolerance_scale

        self._attrv_adj = attrv_adj
        self._precision = precision
        self._scan_step = scan_step
        self._delta_z = delta_z if delta_z <= 0.0 else self._obs.layer_thickness

        self.out_dir = out_dir
        self.species = species
        self._plane_name = miller_name(plane)
        self._molecule_width = (
            molecule_width if not molecule_width is None else molecular_width(self._bulk)
        )

        self._rotated_obs, self._rotation_params = self._rotate_obs()
        self._unrotation_transformations = self._calc_unrotation_transformations()

        self._dummy_obs, self._raw_altitudes, self._clustering_tolerance = self._make_dummy_obs()
        start_timer("Terminator._calculate_possible_shifts()")
        self._shifts = self._calc_shifts()
        stop_timer("Terminator._calculate_possible_shifts()")
        start_timer("Terminator._apply_possible_shifts()")
        self._shifted_dummy_obses = self._apply_shifts()
        stop_timer("Terminator._apply_possible_shifts()")
        start_timer("Terminator._undummify()")
        self._shifted_cells = self._undummify()
        stop_timer("Terminator._undummify()")

        self._surface_voxels = None
        self._average_roughnesses = None
        self._terminations = (
            terminations if terminations else [None] * len(self._shifted_dummy_obses)
        )

    @property
    def bulk(self) -> Structure:
        return self._bulk.copy()

    @property
    def plane(self) -> np.ndarray:
        return self._plane.copy()

    @property
    def plane_name(self) -> str:
        return self._plane_name

    @property
    def molecule_width(self) -> float:
        return self._molecule_width

    def obs(
        self, unrotate: bool = True, dummy: bool = False, structure_only: bool = True
    ) -> Union[OrientedBulk, Structure]:
        if dummy:
            return (
                self._unrotate_structure(structure=self._dummy_obs)
                if unrotate
                else self._dummy_obs.copy()
            )

        if unrotate:
            return (
                self._obs.oriented_bulk_structure.copy() if structure_only else deepcopy(self._obs)
            )
        else:
            return (
                self._rotated_obs.copy()
                if structure_only
                else OrientedBulk(bulk=self._rotated_obs, miller_index=[0, 0, 1], make_planar=False)
            )

    @property
    def rotation_params(self) -> Dict:
        return deepcopy(self._rotation_params)

    @property
    def unrotation_transformations(self) -> Tuple[RotationTransformation]:
        # RotationTransformation inherits from AbstractTransformation, which inherits from MSONable, which should be immutable.
        return self._unrotation_transformations

    @property
    def raw_altitudes(self) -> List[float]:
        return self._raw_altitudes.copy()

    @property
    def clustering_tolerance(self) -> float:
        return self._clustering_tolerance

    @property
    def shifts(self) -> List[float]:
        return self._shifts.copy()

    @property
    def surface_voxels(self) -> List[SurfaceVoxels]:
        if self._surface_voxels is None:
            self._surface_voxels = self._make_surface_voxels()
        return deepcopy(self._surface_voxels)

    @property
    def clustering_tolerance_scale(self) -> float:
        return self._clustering_tolerance_scale

    @property
    def average_roughnesses(self) -> List[float]:
        if self._average_roughnesses is None:
            self._average_roughnesses = self._calc_average_roughnesses()
        return self._average_roughnesses.copy()

    def terminated_structures(self, unrotate: bool = True, dummy: bool = False) -> List[Structure]:
        cells = (
            [obs.oriented_bulk_structure for obs in self._shifted_dummy_obses]
            if dummy
            else self._shifted_cells
        )

        if unrotate:
            terminations = []
            for cell in cells:
                terminations.append(self._unrotate_structure(cell))
            return terminations

        return deepcopy(cells) if dummy else cells

    @property
    def terminations(self) -> List[Termination]:
        if self._terminations[0] is None:
            self._average_roughnesses = self._calc_average_roughnesses()
        return deepcopy(self._terminations)

    @property
    def smoothest_terminated_surface(self) -> Structure:
        return self.smoothest_terminated_structure(
            unrotate=True, dummy=False, all_calcs=False, force_roughness_calc=False, abbreviate=True
        )

    def smoothest_terminated_structure(
        self,
        unrotate: bool = True,
        dummy: bool = False,
        all_calcs: bool = False,
        force_roughness_calc: bool = False,
        abbreviate: bool = True,
    ) -> Union[Structure, List[Structure], List[List[Structure]]]:
        smoothest_termination = self.smoothest_termination(
            all_calcs=all_calcs, force_roughness_calc=force_roughness_calc, abbreviate=False
        )
        result = [
            [term.structure(unrotate=unrotate, dummy=dummy) for term in calc]
            for calc in smoothest_termination
        ]
        if abbreviate:
            if all_calcs:
                return (
                    deepcopy([calc[0] for calc in result])
                    if all([len(calc) == 1 for calc in result])
                    else deepcopy(result)
                )
            return (
                deepcopy(result[0][0])
                if all([len(calc) == 1 for calc in result])
                else deepcopy(result[0])
            )
        return deepcopy(result)

    def smoothest_termination(
        self,
        all_calcs: bool = False,
        force_roughness_calc: bool = False,
        abbreviate: bool = True,
    ) -> Union[Termination, List[Termination], List[List[Termination]]]:
        if self._terminations[0] is None or force_roughness_calc:
            self._average_roughnesses = self._calc_average_roughnesses()

        num_calcs = len(self._terminations[0]._smoothest)

        if len(self._terminations) > 1:
            result = (
                [
                    [term for term in self._terminations if term._smoothest[calc_i]]
                    for calc_i in range(num_calcs)
                ]
                if all_calcs
                else [[term for term in self._terminations if term._smoothest[-1]]]
            )

        else:
            result = (
                [[self._terminations[0]]] * num_calcs if all_calcs else [[self._terminations[0]]]
            )

        if abbreviate:
            if all_calcs:
                return (
                    deepcopy([calc[0] for calc in result])
                    if all([len(calc) == 1 for calc in result])
                    else deepcopy(result)
                )
            return (
                deepcopy(result[0][0])
                if all([len(calc) == 1 for calc in result])
                else deepcopy(result[0])
            )
        return deepcopy(result)

    # TODO: Implement valid_terminations()

    def valid_terminated_structures(
        self,
        unrotate: bool = True,
        dummy: bool = False,
        threshold_type: str = "proportional",
        threshold: float = 1.1,
    ) -> List[Structure]:
        # TODO: Account for multiple roughness calculations
        if self._average_roughnesses is None:
            self._average_roughnesses = self._calc_average_roughnesses()

        valid_terms = []
        cells = (
            [obs.oriented_bulk_structure for obs in self._shifted_dummy_obses]
            if dummy
            else self._shifted_cells
        )
        min_roughness = min(self._average_roughnesses)
        smoothest_index = self._average_roughnesses.index(min_roughness)
        if unrotate:
            valid_terms.append(self._unrotate_structure(cells[smoothest_index]))
        else:
            valid_terms.append(cells[smoothest_index].copy())

        if threshold_type == "proportional":
            threshold = max([threshold, 1.0]) * min_roughness
        elif threshold_type == "fixed":
            threshold = abs(threshold)

        for i, (roughness, cell) in enumerate(zip(self._average_roughnesses, cells)):
            if i == smoothest_index:
                continue

            if roughness <= threshold:
                if unrotate:
                    valid_terms.append(self._unrotate_structure(cell))
                else:
                    valid_terms.append(cell.copy())

        return valid_terms

    def calculate_roughnesses(
        self,
        attrv_adj: float = None,
        precision: int = None,
        scan_step: float = None,
        delta_z: float = None,
    ) -> List[float]:
        if attrv_adj:
            self._attrv_adj = attrv_adj
        if precision:
            self._precision = precision
        if scan_step:
            self._scan_step = scan_step
        if delta_z:
            self._delta_z = self._obs.layer_thickness if delta_z == 0.0 else delta_z

        self._average_roughnesses = self._calc_average_roughnesses()
        return self.average_roughnesses

    def _rotate_obs(self) -> Structure:
        orig_a, orig_b, _ = self._obs.oriented_bulk_structure.lattice.matrix
        orig_surface_normal = np.cross(orig_a, orig_b)
        rot_axis_surf, rot_angle_surf = find_rotation_matrix(orig_v=orig_surface_normal, dest_v="z")

        obs_surf = RotationTransformation(
            axis=rot_axis_surf, angle=rot_angle_surf, angle_in_radians=True
        ).apply_transformation(self._obs.oriented_bulk_structure)
        a_surf = obs_surf.lattice.matrix[0]
        rot_axis_ax, rot_angle_ax = find_rotation_matrix(orig_v=a_surf, dest_v="x")

        obs_surf_ax = RotationTransformation(
            axis=rot_axis_ax, angle=rot_angle_ax, angle_in_radians=True
        ).apply_transformation(obs_surf)
        return obs_surf_ax, {
            "axes": [rot_axis_surf, rot_axis_ax],
            "angles": [rot_angle_surf, rot_angle_ax],
        }

    def _unrotate_structure(self, structure: Structure) -> Structure:
        intermediate = self._unrotation_transformations[0].apply_transformation(structure)
        return self._unrotation_transformations[1].apply_transformation(intermediate)

    def _calc_unrotation_transformations(self) -> Tuple[RotationTransformation]:
        u1 = RotationTransformation(
            self._rotation_params["axes"][1],
            -1.0 * self._rotation_params["angles"][1],
            angle_in_radians=True,
        )
        u2 = RotationTransformation(
            self._rotation_params["axes"][0],
            -1.0 * self._rotation_params["angles"][0],
            angle_in_radians=True,
        )
        return u1, u2

    def _make_dummy_obs(self) -> Tuple[OrientedBulk, List[float], float]:
        start_timer("Terminator._make_dummy_obs()")
        # Rotate the OBS
        structure = self._rotated_obs
        if debug:
            print_structure(structure, "rotated_obs")
        # Create a structure graph so we can extract the molecules
        struc_graph = StructureGraph.from_local_env_strategy(
            structure,
            JmolNN(),
        )

        # Find the center of masses of all the molecules in the unit cell
        # We can do this similar to how the get_subgraphs_as_molecules()
        # function works by creating a 3x3 supercell and only keeping the
        # molecules that don't intersect the boundary of the unit cell
        struc_graph *= (3, 3, 3)
        supercell_g = nx.Graph(struc_graph.graph)

        # Extract all molecule subgraphs
        all_subgraphs = [supercell_g.subgraph(c) for c in nx.connected_components(supercell_g)]

        # Only keep that molecules that are completely contained in the 3x3 supercell
        molecule_graphs = []
        for molecule_graph in all_subgraphs:
            intersects_boundary = any(
                d["to_jimage"] != (0, 0, 0) for u, v, d in molecule_graph.edges(data=True)
            )
            if not intersects_boundary:
                molecule_graphs.append(nx.MultiDiGraph(molecule_graph))

        # Get the center of mass and the molecule index
        molecule_top_centers = []
        molecule_top_tops = []
        molecule_statures = []
        site_props = list(structure.site_properties.keys())
        props = {p: [] for p in site_props}
        for molecule_graph in molecule_graphs:
            cart_coords = np.vstack([struc_graph.structure[node].coords for node in molecule_graph])

            z_coords = np.array(
                [struc_graph.structure[node].coords[-1] for node in molecule_graph]
            )  # struc_graph.structure[node].coords for node in molecule_graph])
            vdW_radii = np.array(
                [struc_graph.structure[node].specie.van_der_waals_radius for node in molecule_graph]
            )
            altitudes = z_coords + vdW_radii
            soles = z_coords - vdW_radii
            top_ind = np.argmax(altitudes)
            bottom = np.min(soles)

            top_position = cart_coords[top_ind]
            is_top = np.zeros(len(cart_coords)).astype(bool)
            is_top[top_ind] = True

            for t, n in zip(is_top, molecule_graph):
                struc_graph.structure[n].properties["is_top"] = t

            for p in props:
                ind = list(molecule_graph.nodes.keys())[0]
                props[p].append(struc_graph.structure[ind].properties[p])

            molecule_top_centers.append(np.round(top_position, 6))
            molecule_top_tops.append(np.round(altitudes[top_ind], 6))
            molecule_statures.append(np.round(altitudes[top_ind] - bottom, 6))

        molecule_top_centers = np.vstack(molecule_top_centers)
        molecule_top_tops = np.vstack(molecule_top_tops)

        # Now we can find which center of masses are contained in the original
        # unit cell. First we can shift the center of masses by the [1, 1, 1]
        # vector of the original unit cell so the center unit cell of the 3x3
        # supercell is positioned at (0, 0, 0)
        shift = structure.lattice.get_cartesian_coords([1, 1, 1])
        inv_matrix = structure.lattice.inv_matrix

        # Shift the center of masses
        molecule_top_centers -= shift

        # Convert to fractional coordinates of the original unit cell
        frac_top_center = molecule_top_centers.dot(inv_matrix)
        frac_top_top = wrap_frac(molecule_top_tops / self._obs.layer_thickness)

        # The real tops of the reference atoms in the unit cell should have fractional
        # coordinates on [0, 1)
        in_original_cell = np.logical_and(
            0 <= np.round(frac_top_top, 6),
            np.round(frac_top_top, 6) < 1,
        ).all(axis=1)

        # Extract the fractional coordinates in the original cell
        frac_coords_in_cell = frac_top_center[in_original_cell]

        # Extract the molecules that have the reference atom in the unit cell
        m_graphs_in_cell = [molecule_graphs[i] for i in np.where(in_original_cell)[0]]

        # Initiate a list of pymatgen.Molecule objects
        molecules = []

        # Initialize a list of van der Waals radii of top atoms
        vdWs = []

        # Initial a new site property dict for the dummy atom structure
        props_in_cell = {}

        # Extract the molecules who's reference atom is in the original cell
        for i, m_graph in enumerate(m_graphs_in_cell):
            # Get the cartesian coordinates of the molecule from the graph
            coords = np.vstack([struc_graph.structure[n].coords for n in m_graph.nodes()])

            # Get the species of the molecule from the graph
            species = [struc_graph.structure[n].specie for n in m_graph.nodes()]

            # Get the is_top site properties of the molecule from the graph
            # This is used to find the reference atom to shift the molecule
            is_top = [struc_graph.structure[n].properties["is_top"] for n in m_graph.nodes()]
            top_ind = int(np.where(is_top)[0][0])

            # Get the site properties of all the atoms in the molecules
            site_props = [struc_graph.structure[n].properties for n in m_graph.nodes()]

            # Extract the properties of the reference atom to be used as the
            # site propeties of the dummy atom in the dummy atom structure
            top_props = site_props[top_ind]

            # Add these properties to the props in cell dict
            for k, v in top_props.items():
                if k in props_in_cell:
                    props_in_cell[k].append(v)
                else:
                    props_in_cell[k] = [v]

            # Get the coordinates of the reference atom
            top_coord = coords[top_ind]

            # Create a Molecule with the reference atom shifted to (0, 0, 0)
            molecule = Molecule(species, coords - top_coord)

            # Add to the list of molecules
            molecules.append(molecule)

            # Add to the list of top-atom van der Waals radii
            vdWs.append(species[top_ind].van_der_waals_radius)

        # Now we will compare molecules to see if any are identically oriented
        combos = combinations(range(len(molecules)), 2)

        # Create an graph and add the indices from the molecules list as the
        # nodes of the graph
        mol_id_graph = nx.Graph()
        mol_id_graph.add_nodes_from(list(range(len(molecules))))

        # Loop through each combination and see if they are the same
        for i, j in combos:
            is_same = compare_molecules(
                mol_i=molecules[i],
                mol_j=molecules[j],
            )

            # If they are oriented the same, then connect their node id's
            # with an edge
            if is_same:
                mol_id_graph.add_edge(i, j)

        # Extract all the connected components from the graph to find all the
        # identical molecules so they can be given the same dummy bulk equiv.
        connected_components = [list(c) for c in nx.connected_components(mol_id_graph)]

        # Map the molecule node id to a dummy bulk equivalent
        bulk_equiv_mapping = {}
        for i, comps in enumerate(connected_components):
            for c in comps:
                bulk_equiv_mapping[c] = i

        # Remove the is_top site property because that is no longer needed
        props_in_cell.pop("is_top")

        # Replace the oriented bulk equivalent for the dummy structure
        props_in_cell["oriented_bulk_equivalent"] = list(
            range(len(props_in_cell["oriented_bulk_equivalent"]))
        )

        # Replace the bulk equivalent for the dummy structure
        # This is needed to filer equivalent surfaces
        props_in_cell["bulk_equivalent"] = [bulk_equiv_mapping[i] for i in range(len(molecules))]

        # Get the atomic numbers for the dummy species
        # (22 is just for nicer colors in vesta)
        species = [i + 22 for i in range(len(molecules))]
        props_in_cell["dummy_species"] = species

        # Get the vdW radii for the top atoms from which the dummy atoms were derived
        props_in_cell["vdW_radii"] = vdWs

        # Create the dummy obs structure
        frac_coords = frac_coords_in_cell
        # sys.stdout.write(f'\n\n{frac_coords=}\n\n')
        struc_props = {
            "molecules": molecules,
        }
        struc_props.update(props_in_cell)

        dummy_struc = Structure(
            lattice=structure.lattice,
            coords=frac_coords,
            species=species,
            site_properties=struc_props,
            to_unit_cell=True,
        )

        dummy_obs = OrientedBulk(bulk=dummy_struc, miller_index=[0, 0, 1], make_planar=True)
        if debug:
            print_structure(add_molecules(dummy_obs), "planar_obs")
        raw_altitudes = [molecule_top_tops[i] for i in np.where(in_original_cell)[0]]
        statures = [molecule_statures[i] for i in np.where(in_original_cell)[0]]
        clustering_tolerance = min(statures) * self._clustering_tolerance_scale
        stop_timer("Terminator._make_dummy_obs()")
        return dummy_obs, raw_altitudes, clustering_tolerance

    def _calc_shifts(self) -> List[float]:
        h = self._dummy_obs.layer_thickness
        headrooms = [h - raw_altitude for raw_altitude in self._raw_altitudes]
        frac_shifts = np.sort(wrap_frac(np.unique(headrooms) / h))
        # Note that these are not actually fractional coordinates, but coordinates as fractions of the height of the cell in the z-dimension.

        n = len(frac_shifts)
        dist_matrix = np.zeros((n, n))

        for i, j in combinations(list(range(n)), 2):
            if i != j:
                cdist = frac_shifts[i] - frac_shifts[j]
                cdist = abs(cdist - round(cdist)) * h
                dist_matrix[i, j] = cdist
                dist_matrix[j, i] = cdist

        condensed_m = squareform(dist_matrix)
        z = linkage(condensed_m)
        clusters = fcluster(z, self._clustering_tolerance, criterion="distance")

        # Generate dict of cluster# to c val - doesn't matter what the c is.
        c_loc = {c: frac_shifts[i] for i, c in enumerate(clusters)}

        # Put all shifts into the unit cell.
        possible_shifts = [wrap_frac(frac_shift) for frac_shift in sorted(c_loc.values())]

        return [h * possible_shift for possible_shift in possible_shifts]

    def _apply_shifts(self) -> List[OrientedBulk]:
        shifted_obses = []
        for possible_shift in self._shifts:

            slab_base = deepcopy(self._dummy_obs)
            slab_base.translate_sites(
                vector=[0, 0, possible_shift],
                frac_coords=False,
            )
            slab_base.round(tol=6)
            shifted_obses.append(slab_base)

        return shifted_obses

    def _undummify(self) -> List[Structure]:
        undummified = []
        for dummy in self._shifted_dummy_obses:
            undummified.append(add_molecules(structure=dummy))
        if debug:
            print_structure(undummified[0], "shifted_obs")
        return undummified

    def _make_surface_voxels(self) -> List[SurfaceVoxels]:
        surface_voxels = []
        for i, shifted_cell in enumerate(self._shifted_cells):
            if debug:
                shifted_cell.to(os.path.join(os.getcwd(), f"shifted_cell{i}.cif"))
            surface_voxels.append(
                SurfaceVoxels(
                    unit_cell=shifted_cell.copy(),
                    attrv_adj=self._attrv_adj,
                    precision=self._precision,
                    scan_step=self._scan_step,
                    delta_z=self._delta_z,
                    molecule_width=self._molecule_width,
                )
            )

        return surface_voxels

    def _calc_average_roughnesses(self) -> List[float]:
        if self._surface_voxels is None:
            self._surface_voxels = self._make_surface_voxels()

        average_roughnesses = []
        for surface_voxels in self._surface_voxels:
            average_roughnesses.append(surface_voxels.average_roughness)

        for i, (termination, roughness) in enumerate(zip(self._terminations, average_roughnesses)):
            smoothest = True if roughness == min(average_roughnesses) else False
            if termination is None:
                self._terminations[i] = Termination(
                    rotated_dummy_obs=self._shifted_dummy_obses[i],
                    shift=self._shifts[i],
                    average_roughness=roughness,
                    attrv_adj=self._attrv_adj,
                    precision=self._precision,
                    scan_step=self._scan_step,
                    delta_z=self._delta_z,
                    smoothest=smoothest,
                    unrotation_transformations=self._unrotation_transformations,
                    rotated_molecular_structure=self._shifted_cells[i],
                )
            else:
                termination.append_roughness_calc(
                    average_roughness=roughness,
                    attrv_adj=self._attrv_adj,
                    precision=self._precision,
                    scan_step=self._scan_step,
                    delta_z=self._delta_z,
                    smoothest=smoothest,
                )

        return average_roughnesses

    def output_files(
        self, typ: str = "smoothest", unrotate: bool = True, dummy: bool = False
    ) -> None:
        if typ == "all":
            rng = self._shifted_cells
        elif typ == "valid":
            rng = self.valid_terminated_structures(
                unrotate=unrotate, dummy=dummy, threshold_type="proportional", threshold=1.1
            )
        else:
            rng = self.smoothest_terminated_structure(
                unrotate=unrotate,
                dummy=dummy,
                all_calcs=False,
                force_roughness_calc=False,
                abbreviate=True,
            )

        if not os.path.isdir(self.out_dir):
            os.mkdir(self.out_dir)
        species_dir = os.path.join(self.out_dir, self.species)
        if not os.path.isdir(species_dir):
            os.mkdir(species_dir)
        plane_dir = os.path.join(species_dir, self._plane_name)
        if not os.path.isdir(plane_dir):
            os.mkdir(plane_dir)

        for shift, roughness, struc in zip(self._shifts, self._average_roughnesses, rng):
            # accepted = "accepted" if struc in self.valid_terminated_structures else "rejected"
            # sys.stdout.write(f"{plane_dir=}\n")
            struc.to(os.path.join(plane_dir, f"shift_{shift:.3f}_roughness_{roughness:.3f}.cif"))

    def visualize(self, save_dir: str = None, cmap: Union[str, Colormap] = "viridis") -> None:
        if self._surface_voxels is None:
            self._surface_voxels = self._make_surface_voxels()

        if save_dir is None:
            for surface_voxels in self._surface_voxels:
                surface_voxels.visualize(cmap=cmap)
        else:
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            species_dir = os.path.join(save_dir, self.species)
            if not os.path.isdir(species_dir):
                os.mkdir(species_dir)
            plane_dir = os.path.join(species_dir, self._plane_name)
            if not os.path.isdir(plane_dir):
                os.mkdir(plane_dir)

            if self._average_roughnesses is None:
                self._average_roughnesses = self._calc_average_roughnesses()

            for surface_voxels, shift, average_roughness in zip(
                self._surface_voxels, self._shifts, self._average_roughnesses
            ):
                surface_voxels.visualize(
                    save_to=os.path.join(
                        plane_dir,
                        f"surface_voxels_shifted_{shift:.3f}_roughness_{average_roughness:.3f}.png",
                    ),
                    cmap=cmap,
                )


def print_structure(structure, name):
    structure.to(os.path.join(os.getcwd(), f"{name}.cif"))


timers = {}
times = {}


def start_timer(timer_name):
    """Start a timer with a given name."""
    timers[timer_name] = time.time()


def stop_timer(timer_name):
    """Stop a timer with a given name and print the elapsed time."""
    if timer_name in timers:
        start_time = timers.pop(timer_name)
        elapsed_time = time.time() - start_time
        print(f"\tTimer '{timer_name}' elapsed time: {elapsed_time:.2f} seconds")
        if not timer_name in times.keys():
            times[timer_name] = [elapsed_time]
        else:
            times[timer_name].append(elapsed_time)
            if len(times[timer_name]) % 4 == 0:
                time_strings = [f"{elapsed:.2f}" for elapsed in times[timer_name]]
                print(f"\tTimer '{timer_name}' elapsed times: {time_strings}")
    else:
        print(f"\tTimer '{timer_name}' was not started")


def caller_line_number():
    result = ""
    stack = inspect.stack()
    for i in range(len(stack) - 1, 0, -1):
        result += f"{stack[i].lineno}:"
    return result


def compare_molecules(mol_i: Molecule, mol_j: Molecule) -> bool:
    # Check if they are the same length
    if len(mol_i) == len(mol_j):
        # Get the cartesian coordinates for each molecule
        coords_i = mol_i.cart_coords
        coords_j = mol_j.cart_coords

        # Get the atomic numbers for each molecule
        atomic_numbers_i = np.array(mol_i.atomic_numbers)
        atomic_numbers_j = np.array(mol_j.atomic_numbers)

        # Concatenate the coords and atomic numbers into a (N, 4) array
        # That needs to be sorted to compare the molecules
        sort_array_i = np.round(np.c_[coords_i, atomic_numbers_i], 5)
        sort_array_j = np.round(np.c_[coords_j, atomic_numbers_j], 5)

        # Refactor the sort array into a list of tuples (easier to sort)
        sort_data_i = list(map(tuple, sort_array_i))

        # Sort by x, then y, then z, then atomic number
        sort_data_i.sort(key=lambda x: (x[0], x[1], x[2], x[3]))

        # Refactor the sort array into a list of tuples (easier to sort)
        sort_data_j = list(map(tuple, sort_array_j))

        # Sort by x, then y, then z, then atomic number
        sort_data_j.sort(key=lambda x: (x[0], x[1], x[2], x[3]))

        # Check if the molecules have the exact same orientation & species
        is_same = np.allclose(
            np.array(sort_data_i),
            np.array(sort_data_j),
            atol=1e-5,
        )

        return is_same
    else:
        return False


def normalize(vector):
    return vector / np.linalg.norm(vector)


def homogenize(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 4:
        return points.copy()
    if points.shape[0] == 3:
        return np.vstack((points, np.ones((1, points.shape[1]))))
    if len(points.shape) == 2:
        if points.shape[1] == 3:
            return points.T
        if points.shape[1] == 3:
            return np.vstack((points.T, np.ones((1, points.shape[0]))))

    sys.stdout.write(
        f'\nA "points" array with an unexpected shape was passed to homogenize()! points.shape = {points.shape}\n\n'
    )
    return points.copy()


def wrap_frac(frac_coord: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return (
        frac_coord - math.floor(frac_coord)
        if type(frac_coord) == float
        else frac_coord - np.floor(frac_coord)
    )


def find_rotation_matrix(
    orig_v: np.ndarray, dest_v: Union[np.ndarray, str]
) -> Tuple[np.ndarray, float]:
    orig_v = normalize(orig_v)

    if type(dest_v) == str:
        str_to_vector = {
            "x": np.array([1, 0, 0]),
            "y": np.array([0, 1, 0]),
            "z": np.array([0, 0, 1]),
        }
        dest_v = str_to_vector[dest_v]
    dest_v = normalize(dest_v)

    rot_axis = np.cross(orig_v, dest_v)
    if np.linalg.norm(rot_axis) == 0:
        if np.allclose(orig_v, dest_v):
            return np.eye(3)
        else:
            sys.stdout.write("Vectors are opposite to one another!\n")
            rot_axis = np.cross(orig_v, np.random.rand(3))

    rot_axis = normalize(rot_axis)
    rot_angle = np.arccos(orig_v @ dest_v)

    return rot_axis, rot_angle


def plane_from_name(plane_name: str) -> List[int]:
    miller_index = []
    i = 0
    while i < len(plane_name):
        if plane_name[i] == "-":
            miller_index.append(int(plane_name[i : i + 2]))
            i += 1
        else:
            miller_index.append(int(plane_name[i]))
        i += 1
    return miller_index


def miller_name(miller_index: List[int]) -> str:
    name = ""
    for hkl in miller_index:
        name += str(hkl)
    return name


def molecular_width(structure: Structure) -> float:
    molecules = get_molecules_from_structure(structure)
    return max(
        [
            squareform(pdist([site.coords for site in molecule.sites])).max()
            for molecule in molecules
        ]
    )


def add_molecules(structure: OrientedBulk) -> Structure:
    mol_coords = []
    mol_atom_nums = []

    properties = list(structure.site_properties.keys())
    mols = structure.site_properties["molecules"]
    if "molecules" in properties:
        properties.remove("molecules")
    site_props = {p: [] for p in properties}
    site_props["molecule_index"] = []

    for i, site in enumerate(structure):
        site_mol = mols[i]  # site.properties["molecules"]
        mol_coords.append(site_mol.cart_coords + site.coords)
        mol_atom_nums.extend(site_mol.atomic_numbers)

        site_props["molecule_index"].extend([i] * len(site_mol))

        for p in properties:
            site_props[p].extend([site.properties[p]] * len(site_mol))

    mol_layer_struc = Structure(
        lattice=structure.oriented_bulk_structure.lattice,
        species=mol_atom_nums,
        coords=np.vstack(mol_coords),
        to_unit_cell=True,
        coords_are_cartesian=True,
        site_properties=site_props,
    )
    mol_layer_struc.sort()

    return mol_layer_struc


def subgraph_to_molecule(structure: Structure, subgraph: nx.Graph) -> Molecule:
    # Get the list of node indices (atom indices) in the subgraph
    atom_indices = list(subgraph.nodes())
    # sys.stdout.write(f'number of atoms in molecule = {len(atom_indices)}\n')

    # Get the species and coordinates of these atoms
    species = [structure[i].specie for i in atom_indices]
    coords = [structure[i].coords for i in atom_indices]

    # Create and return a pymatgen Molecule object
    return Molecule(species, coords)


def get_molecules_from_structure(structure: Structure) -> List[Molecule]:
    molecule_graphs = get_molecule_graphs(structure)
    molecules = [subgraph_to_molecule(structure, graph) for graph in molecule_graphs]
    return molecules


def get_molecule_graphs(structure: Structure) -> List[nx.Graph]:
    struc_graph = StructureGraph.from_local_env_strategy(structure, JmolNN())
    cell_graph = nx.Graph(struc_graph.graph)
    return [cell_graph.subgraph(c) for c in nx.connected_components(cell_graph)]


def compute_maximum_width(structure: Structure) -> float:
    max_distance = 0.0
    jmol_nn = JmolNN()
    struc = structure.copy().get_supercell([3, 3, 3])

    # Loop through each site in the structure
    for i, site_i in enumerate(struc):
        visited = set()
        stack = [i]
        site_stack = [site_i]

        while stack:
            current_i = stack.pop()
            current_site = site_stack.pop()
            if current_site not in visited:
                visited.add(current_site)

                # Get neighbor information using JmolNN
                neighbors_info = jmol_nn.get_nn_info(struc, current_i)

                for neighbor_info in neighbors_info:
                    neighbor_site = neighbor_info["site"]

                    if neighbor_site not in visited:  # and neighbor_site in structure:

                        neighbor_i = next(
                            (
                                idx
                                for idx, s in enumerate(struc)
                                if s.is_periodic_image(neighbor_site)
                            ),
                            None,
                        )  # neighbor_i = structure.index(neighbor_site)
                        if not neighbor_i is None:
                            site_stack.append(neighbor_site)
                            stack.append(neighbor_i)

                            # Calculate distance between current_site and neighbor_site
                            distance = structure.get_distance(current_i, neighbor_i)
                            if distance > max_distance:
                                max_distance = distance
                        else:
                            sys.stdout.write("\n\n\nsite not found :(\n\n\n")

    return max_distance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find surface terminations for a given surface.")
    parser.add_argument(
        "--structure_path",
        type=str,
        default=os.path.join("obs_files", "ASPIRIN", "unrefined", "nonplanar", "OBS_100.cif"),
        help="Relative path to the surface-oriented structure file whose terminations are to be found",
    )
    parser.add_argument(
        "--bulk_path",
        type=str,
        default="ASPIRIN.cif",
        help="Relative path to the bulk unit cell structure file",
    )
    parser.add_argument(
        "--attrv_adj",
        type=float,
        default=0.0,
        help="Probe attractive radius adjustment in Angstroms",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=26,
        help="Precision",
    )
    parser.add_argument(
        "--scan_step",
        type=float,
        default=0.1,
        help="Scan step size",
    )
    parser.add_argument(
        "--delta_z",
        type=float,
        default=0.0,
        help="Maximum distance from maximum z-coordinate at which surface atoms should be considered",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Whether to visualize the voxelized surface terminations",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Whether to write the shifted cells to files",
    )
    parser.add_argument(
        "--struct_out_dir",
        type=str,
        default="term_files",
        help="Relative path to shifted structure output directory",
    )
    parser.add_argument(
        "--viz_out_dir",
        type=str,
        default="visualizations",
        help="Relative path to visualization output directory",
    )
    args = parser.parse_args()
    bulk = Structure.from_file(os.path.join(os.getcwd(), args.bulk_path))
    species = os.path.basename(args.bulk_path).split(".")[0]

    molecule_width = molecular_width(bulk)
    sys.stdout.write(f"\n\n\n{molecule_width=}\n\n\n")

    max_index = 2 if "TETCEN" in species else 1

    for plane in tqdm(
        get_unique_miller_indices(structure=bulk, max_index=max_index), desc="Terminating facets..."
    ):
        # for plane in [[0, 1, 0]]:
        sys.stdout.write(f"{plane=}\n")
        debug = False
        if plane[0] == 0 and plane[1] == 1 and plane[2] == 0:
            debug = True

        terminator = Terminator(
            bulk=bulk,
            plane=plane,
            clustering_tolerance_scale=0.1,
            attrv_adj=args.attrv_adj,
            precision=args.precision,
            scan_step=args.scan_step,
            delta_z=args.delta_z,
            out_dir=os.path.join(os.getcwd(), args.struct_out_dir),
            species=species,
            molecule_width=molecule_width,
        )

        # terminator.output_termination_files(typ="all")
        smoothest_termination = terminator.smoothest_terminated_surface
        smoothest_termination.to(
            os.path.join(
                os.getcwd(), "smoothest_terminations", species, f"{miller_name(plane)}.cif"
            )
        )
        if args.visualize:
            terminator.visualize(save_dir=os.path.join(os.getcwd(), args.viz_out_dir))
