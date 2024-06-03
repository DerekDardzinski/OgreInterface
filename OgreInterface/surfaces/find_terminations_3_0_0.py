import argparse
from copy import deepcopy
from itertools import combinations
import math
import os
import sys
from typing import Dict, List, Tuple, TypeVar, Union

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from mpl_toolkits.axes_grid1 import make_axes_locatable
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import JmolNN
from pymatgen.core.lattice import Lattice
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.structure import Molecule, Structure
from pymatgen.transformations.site_transformations import (
    TranslateSitesTransformation,
)  # , RemoveSitesTransformation
from pymatgen.transformations.standard_transformations import RotationTransformation
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage

from OgreInterface.surfaces.oriented_bulk import OrientedBulk

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
    """

    def __init__(self, vertices: np.ndarray, surface_normal: np.ndarray = None):
        sys.stdout.write("creating SurfacePrism\n")
        self.vertices = self._complete_vertices(vertices) if vertices.shape[1] == 3 else vertices
        self.surface_normal = (
            self._surface_normal()
            if surface_normal is None
            else surface_normal / np.linalg.norm(surface_normal)
        )
        self._structure = None
        self._face_plane_equations = self._calc_face_plane_equations()
        self._inside_sides = self._calc_inside_sides()
        self._inward_face_normals = self._calc_inward_face_normals()
        sys.stdout.write("created SurfacePrism\n")

    def _complete_vertices(self, vertices: np.ndarray) -> np.ndarray:
        """Append the missing fourth vertex."""

        return np.hstack(vertices, vertices[:, 0] - vertices[:, 1] + vertices[:, 1])

    def _surface_normal(self) -> np.ndarray:
        """Compute the unit normal vector to the surface."""

        AB = self.vertices[:, 1] - self.vertices[:, 0]
        AD = self.vertices[:, 3] - self.vertices[:, 0]
        surface_normal = np.cross(AB, AD)
        surface_normal /= np.linalg.norm(surface_normal)
        return surface_normal

    def _calc_face_plane_equations(self) -> np.ndarray:
        """Calculate the coefficients of the plane equations of the faces of the parallelipiped."""

        next_vertices = np.roll(self.vertices, -1, axis=1)
        face_plane_equations = np.zeros((4, 4))
        for i in range(4):
            normal_vector = np.cross(self.vertices[:, i] - next_vertices[:, i], self.surface_normal)
            normal_vector = normal_vector / np.linalg.norm(normal_vector)
            D = -1.0 * normal_vector @ self.vertices[:, i]
            face_plane_equations[i, :] = np.append(normal_vector, D)
        return np.array(face_plane_equations)

    def _calc_inside_sides(self) -> np.ndarray:
        """Compute example outputs of a point on the inside side of each face."""

        homogeneous_vertices = homogenize(self.vertices)
        return np.vstack(
            (
                self._face_plane_equations[0] @ homogeneous_vertices[:, 2],
                self._face_plane_equations[1] @ homogeneous_vertices[:, 3],
                self._face_plane_equations[2] @ homogeneous_vertices[:, 0],
                self._face_plane_equations[3] @ homogeneous_vertices[:, 1],
            )
        )

    def _inside(self, homogenenous_point: np.ndarray) -> bool:
        """Determine whether a point is inside/on the parallelipiped."""

        return np.all((self._face_plane_equations @ homogenenous_point) * self._inside_sides >= 0)

    def _calc_inward_face_normals(self) -> np.ndarray:
        inward_face_normals = []
        for i in range(4):
            inward_face_normals.append(
                normalize(
                    self._face_plane_equations[i, :2] * math.copysign(1, self._inside_sides[i])
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

        if points.shape[0] == 3:
            points = homogenize(points)
        return np.apply_along_axis(func1d=self._inside, axis=0, arr=points)

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

    def mask_structure(self, structure: Structure = None, in_place: bool = False) -> Structure:
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

        structure = self._structure if structure is None else structure
        if structure is None:
            raise ValueError(
                "kwarg 'structure' is required unless SurfacePrism object was instantiated via \
                    SurfacePrism.from_structure() or a pymatgen.core.structure Structure object \
                    was manually assigned to the SurfacePrism object's ._structure attribute."
            )

        if in_place:
            structure.remove_sites(
                indices=np.where(self._mask(points=structure.coords.T) == False)[0]
            )
            return structure

        masked_structure = structure.copy()
        masked_structure.remove_sites(
            indices=np.where(self._mask(points=structure.coords.T) == False)[0]
        )
        return masked_structure

    def buffer_mask_supercell(
        self,
        supercell: Structure,
        translation_vector_to_unit_cell: np.ndarray,
        in_place: bool = False,
    ) -> Structure:
        translated_points = (supercell.coords + translation_vector_to_unit_cell).T
        mask = self._mask(points=translated_points)

        to_remove = []
        for i, site in enumerate(supercell):
            if not self._infiltrates_any_face(site=site) and not mask[i]:
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
            surface_normal = np.cross(a, b)
            surface_normal /= np.linalg.norm(surface_normal)

        return cls(vertices=vertices, surface_normal=surface_normal)

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
        surface_prism._structure = structure
        return surface_prism


class SurfaceCell:
    def __init__(
        self, original_cell: Structure, surface_normal: np.ndarray = None, c_scale: int = 1
    ):
        sys.stdout.write("creating SurfaceCell\n")
        self.original_cell = original_cell
        self.surface_normal = (
            surface_normal if not surface_normal is None else self._surface_normal()
        )
        self.c_scale = c_scale
        self.raw_supercell = self._raw_supercell()
        self.planed_supercell = self._planed_supercell()
        self.surface_prism = SurfacePrism.from_matrix(
            matrix=self.original_cell.lattice.matrix, surface_normal=self.surface_normal
        )
        self.supercell_slug = self._supercell_slug()
        self.structure = self._structure()
        self.vdW_radii = self._vdW_radii()
        self.buffered_structure = self._buffered_structure()
        self.buffered_prism = SurfacePrism.from_matrix(
            matrix=self.buffered_structure.lattice.matrix
            + np.sum(self.original_cell.lattice.matrix[:2], axis=0),
            surface_normal=self.surface_normal,
        )
        self.bounds = self._buffered_bounds()
        sys.stdout.write("created SurfaceCell\n")

    def _surface_normal(self) -> np.ndarray:
        a, b, _ = self.original_cell.lattice.matrix
        return normalize(np.cross(normalize(a), normalize(b)))

    def _raw_supercell(self) -> Structure:
        return self.original_cell.make_supercell(
            scaling_matrix=[3, 3, self.c_scale], to_unit_cell=True, in_place=False
        )

    def _intersects_surface_plane(self, molecule_graph, ceiling):

        # TODO: toggle periodicity of sites off?
        for node in molecule_graph.nodes:
            site = self.raw_supercell[node]
            z_coord = site.coords[-1]
            vdW_radius = site.specie.van_der_waals_radius
            if z_coord + vdW_radius > ceiling:
                return True
            if z_coord > ceiling and z_coord - vdW_radius < ceiling:
                return True
        return False

    def _planed_supercell(self) -> Structure:
        ceiling = self.c_scale * self.original_cell.lattice.matrix[-1] @ self.surface_normal
        sys.stdout.write("getting moleule graphs\n")
        # upper_
        for site in self.raw_supercell:
            if (
                site.coords @ self.surface_normal
                > 0.8 * self.original_cell.lattice.matrix[-1] @ self.surface_normal
            ):
                pass

        # molecule_graphs = get_molecule_graphs(self.raw_supercell.)
        sys.stdout.write("got moleule graphs\n")
        graphs_to_keep = []
        for molecule_graph in molecule_graphs:
            if not self._intersects_surface_plane(molecule_graph, ceiling):
                graphs_to_keep.append(molecule_graph)

        new_sites = []
        for subgraph in graphs_to_keep:
            for node in subgraph.nodes:
                new_sites.append(self.raw_supercell[node])

        # TODO: toggle periodicity of sites off?
        return Structure.from_sites(new_sites)

    def _supercell_slug(self) -> Structure:
        # transform_to_unit_cell = np.hstack((np.ones(3), np.sum(self.original_cell.lattice.matrix[:2].T)))
        # transform_to_unit_cell[:, -1] += self.c_scale * self.original_cell.lattice.matrix[-1].T
        translation_vector = np.sum(
            self.original_cell.lattice.matrix[:2], axis=1
        ).T + self.c_scale * self.original_cell.lattice.matrix[-1].reshape(-1, 1)

        translate_sites_transformation = TranslateSitesTransformation(
            indicies_to_move=range(len(self.planed_supercell)),
            translation_vector=translation_vector,
            vector_in_frac_coords=False,
        )

        # unit_cell = transform_to_unit_cell @ self.planed_supercell
        shifted_supercell = translate_sites_transformation.apply_transformation(
            structure=self.planed_supercell.copy()
        )
        return self.surface_prism.mask_structure(structure=shifted_supercell, in_place=False)

    def _structure(self) -> Structure:
        a, b, old_c = self.original_cell.lattice.matrix
        c = old_c @ self.surface_normal
        return Structure(
            lattice=Lattice(matrix=np.hstack((a, b, c))),
            species=self.supercell_slug.species,
            coords=self.supercell_slug.coords,
            coords_are_cartesian=True,
        )

    def _vdW_radii(self) -> Dict:
        """Retrieve all needed van der Waal radii"""

        all_species = list(set(self.structure.species))
        return {
            species.symbol: species.van_der_waals_radius + self.attrv_adj for species in all_species
        }

    def _buffered_structure(self) -> Structure:
        supercell = self.structure.make_supercell(
            scaling_matrix=[3, 3, 1], to_unit_cell=True, in_place=False
        )
        translation_vector_to_unit_cell = -1.0 * np.sum(self.structure.lattice.matrix[:2], axis=0)
        masked_supercell = self.surface_prism.buffer_mask_supercell(
            supercell=supercell,
            translation_vector_to_unit_cell=translation_vector_to_unit_cell,
            in_place=False,
        )
        return masked_supercell  # masked_supercell.translate_sites(indices=range(len(masked_supercell)), vector=translation_vector_to_unit_cell, frac_coords=False, to_unit_cell=False)

    def _buffered_bounds(self) -> np.ndarray:
        a_plus_b = np.sum(self.structure.lattice.matrix[:2], axis=0).reshape(-1, 1)
        return np.hstack(
            (a_plus_b, a_plus_b + np.sum(self.structure.lattice.matrix, axis=0).reshape(-1, 1))
        )


class SurfaceVoxels:
    def __init__(
        self,
        unit_cell: Structure,
        probe_rad: float = 1.2,
        attrv_adj: float = 0.0,
        precision: int = 30,
        scan_step: float = 0.1,
        delta_z: float = 0.0,
        raw_z_adj: float = 0.0,
        z_adj_mode: str = "sub_min_z",
    ):
        sys.stdout.write("creating SurfaceVoxels\n")
        self.unit_cell = unit_cell
        self.probe_rad = probe_rad
        self.attrv_adj = attrv_adj
        self.precision = precision
        self.scan_step = scan_step
        self.delta_z = delta_z if delta_z != 0.0 else molecular_width(structure=unit_cell)
        # self.net_z_adj = self._net_z_adj(raw_z_adj=raw_z_adj, z_adj_mode=z_adj_mode)
        # self.vdW_radii = self._vdW_radii()

        self.xn = 0
        self.yn = 0
        self.zn = 0

        self.surface_normal = self._surface_normal()
        self.surface_cell = SurfaceCell(
            original_cell=self.unit_cell, surface_normal=self.surface_normal
        )
        self.surface_prism = SurfacePrism.from_matrix(
            matrix=unit_cell.lattice.matrix, surface_normal=self.surface_normal
        )
        self.net_z_adj = self._net_z_adj(raw_z_adj=raw_z_adj, z_adj_mode=z_adj_mode)
        self.voxels = self._voxelize()  # , self.voxel_cart_value_ranges = self._voxelize()
        self.voxel_surface, self.masked_points = self._voxel_surface()
        self.roughnesses = self._roughnesses()
        self.average_roughness = self._average_roughness()
        sys.stdout.write("created SurfaceVoxels\n")

    def _surface_normal(self) -> np.ndarray:
        a, b, _ = self.unit_cell.lattice.matrix
        return normalize(np.cross(normalize(a), normalize(b)))

    def _net_z_adj(self, raw_z_adj: float = 0.0, z_adj_mode: str = "sub_min_z") -> float:
        if "min_z" in z_adj_mode:
            mode_adj = np.min(self.surface_cell.structure.cart_coords[:, 2])
        elif "max_z" in z_adj_mode:
            mode_adj = np.max(self.surface_cell.structure.cart_coords[:, 2])
        else:
            mode_adj = 0.0

        if "sub" in z_adj_mode:
            mode_adj *= -1.0

        return raw_z_adj + mode_adj

    def _voxelize(self) -> Tuple[np.ndarray, np.ndarray]:
        # Get the Cartesian coordinates and species of the atoms in self.structure
        structure = self.surface_cell.buffered_structure

        # Determine the Cartesian z-range of the surface atoms
        max_z = structure.lattice.matrix[-1, -1]  # np.max(cart_coords[:, 2])

        # Adjust Cartesian z-coordinates to align the structure for voxelization
        # cart_coords[:, 2] += self.net_z_adj  # -= min_z  # += 20 - min_z

        # Select surface atoms within delta_z of the max_z
        target_coords = []
        target_sites = []

        # Find all atoms whose adjusted Cartesian z-coordinates are less than self.delta_z less than the maximum unadjusted Cartesian z-coordinate/are greater than the maximum unadjusted Cartesian z-coordinate minus self.delta_z
        for site in structure:
            if max_z - site.coords[-1] - site.specie.van_der_waals_radius < self.delta_z:
                target_sites.append(site)

        target_coords = np.array(target_coords)

        # Discretize the voxel grid in steps of length self.scan_step, but to 0.01 Angstroms beyond the maximum extent in all three Cartesian directions.
        xi = np.arange(
            self.surface_cell.bounds[0, 0],
            self.surface_cell.bounds[0, 1],
            self.scan_step,
        )
        yi = np.arange(
            self.surface_cell.bounds[1, 0],
            self.surface_cell.bounds[1, 1],
            self.scan_step,
        )
        zi = np.arange(
            self.surface_cell.bounds[2, 0],
            self.surface_cell.bounds[2, 1],
            self.scan_step,
        )

        self.xn = len(xi)
        self.yn = len(yi)
        self.zn = len(zi)

        # Zero-initialize an array of points that correspond to gridpoints in the voxel grid
        voxel_array = np.zeros((self.xn, self.yn, self.zn))

        for site in target_sites:
            thetas = np.linspace(
                0.0, math.pi / 2, self.precision
            )  # math.ceil(math.pi * rad / (self.scan_step * 2)))
            rad = site.specie.van_der_waals_radius + self.attrv_adj
            angle_counter = 0
            x, y, z = site.coords

            for theta in thetas:
                phis = np.linspace(
                    0.0, math.pi * 2, angle_counter * 4
                )  # math.ceil(2 * math.pi * rad * np.sin(theta) / self.scan_step))
                for phi in phis:
                    x_i = (
                        int(
                            (x + rad * np.sin(theta) * np.cos(phi) - self.surface_cell.bounds[0, 0])
                            / self.scan_step
                        )
                        - 1
                    )
                    y_i = (
                        int(
                            (y + rad * np.sin(theta) * np.sin(phi) - self.surface_cell.bounds[1, 0])
                            / self.scan_step
                        )
                        - 1
                    )
                    z_i = (
                        int(
                            (z + rad * np.cos(theta) - self.surface_cell.bounds[2, 0])
                            / self.scan_step
                        )
                        - 1
                    )

                    if 0 <= x_i and x_i < self.xn and 0 <= y_i and y_i < self.yn:
                        voxel_array[x_i, y_i, min(z_i, self.zn - 1)] = 1
                angle_counter += 1

        return voxel_array

    def _voxel_surface(self) -> Tuple[np.ndarray, np.ndarray]:
        voxel_surface = np.zeros((self.voxels.shape[0], self.voxels.shape[1]))
        points = []  # = np.full(self.voxels.shape, np.nan)
        for x_index in range(self.xn):
            x_coord = x_index * self.scan_step
            for y_index in range(self.yn):
                height = 0.0
                occupied_z_indices = np.where(self.voxels[x_index, y_index, :] == 1)[0]
                if len(occupied_z_indices) > 0:
                    height = occupied_z_indices[-1] * self.scan_step
                voxel_surface[x_index, y_index] = height
                points.append([x_coord, y_index * self.scan_step, height])

        # The rectangular voxel grid extends beyond the bounds of the surface cell.
        # Make a surface prism from the surface cell's lattice vectors to mask the out-of-bounds points.
        masked_points = self.surface_cell.buffered_prism.mask_points(points=np.array(points).T)

        return voxel_surface, masked_points

    def _roughnesses(self) -> np.ndarray:
        """Compute roughnesses from masked voxel surface heights"""

        return -1.0 * self.masked_points[-1] + self.surface_cell.structure.lattice.matrix[-1][-1]

    def _average_roughness(self) -> float:
        return np.mean(self.roughnesses)

    def visualize(self, save_to: str = "") -> None:
        fig, ax = plt.subplots()  # plt.figure()#figsize=(10, 8))
        im = ax.imshow(
            self.voxel_surface.T,
            origin="lower",
            extent=(
                self.surface_cell.bounds[0][0],
                self.surface_cell.bounds[0][1],
                self.surface_cell.bounds[1][0],
                self.surface_cell.bounds[1][1],
            ),
            cmap="viridis",
            interpolation="nearest",
            aspect="equal",
        )  # aspect='auto')

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

        if len(save_to) > 0:
            sys.stdout.write(f"{save_to=}\n")
            plt.savefig(save_to)


class Terminator:
    def __init__(
        self,
        bulk: Structure,
        obs: OrientedBulk,
        plane: List[int],
        make_planar: bool = False,
        clustering_tolerance_scale: float = 0.1,
        surface_buffer: float = 0.01,
        probe_rad: float = 1.2,
        attrv_adj: float = 0.0,
        precision: int = 30,
        scan_step: float = 0.1,
        delta_z: float = 0.0,
        raw_z_adj: float = 0.0,
        z_adj_mode: str = "sub_min_z",
        out_dir: str = os.getcwd(),
        species: str = "SPECIES",
        all_terminations: bool = True,
        max_roughness_factor: float = 1.5,
    ):
        sys.stdout.write("creating Terminator\n")
        self.bulk = bulk
        self.obs = obs
        self.plane = plane
        self.make_planar = make_planar
        self.clustering_tolerance_scale = clustering_tolerance_scale
        self.surface_buffer = surface_buffer
        self.refined = "refined" if refined else "unrefined"
        self.probe_rad = probe_rad
        self.attrv_adj = attrv_adj
        self.precision = precision
        self.scan_step = scan_step
        self.delta_z = delta_z
        self.raw_z_adj = raw_z_adj
        self.z_adj_mode = z_adj_mode
        self.out_dir = out_dir
        self.species = species
        self.plane_name = miller_name(plane)
        self.all_terminations = all_terminations
        self.max_roughness_factor = max_roughness_factor

        self.rotation_matrix, self.rotation_transformation = self._rotation_matrix()
        self.dummy_obs, self.raw_altitudes, self.clustering_tolerance = self._make_dummy_obs()
        sys.stdout.write("Made dummy OBS\n")
        self.possible_shifts = self._calculate_possible_shifts()
        sys.stdout.write("Calculated possible shifts\n")
        self.shifted_dummy_cells = self._apply_possible_shifts()
        sys.stdout.write("Applied possible shifts\n")
        self.shifted_cells = self._undummify()
        sys.stdout.write("Undummified shifted cells\n")
        self.surface_voxels = self._surface_voxels()
        # self.dummy_roughnesses = self._dummy_roughnesses()
        self.average_roughnesses = self._average_roughnesses()
        self.maximum_acceptable_average_roughness = self._maximum_acceptable_average_roughness()
        sys.stdout.write("Calculated roughnesses\n")
        self.valid_terminations = self._valid_terminations()
        sys.stdout.write("created Terminator\n")

    def _rotation_matrix(self):
        orig_a, orig_b, _ = self.obs.oriented_bulk_structure.lattice.matrix
        orig_surface_normal = np.cross(orig_a, orig_b)
        Rsurf, rot_axis_surf, rot_angle_surf = find_rotation_matrix(
            orig_v=orig_surface_normal, dest_v="z"
        )
        Rax, rot_axis_ax, rot_angle_ax = find_rotation_matrix(orig_v=orig_a, dest_v="x")
        R = Rsurf @ Rax
        rot_axis = normalize(np.cross(normalize(rot_axis_surf), normalize(rot_axis_ax)))
        rot_angle = np.sqrt(rot_angle_surf**2 + (rot_angle_ax - rot_angle_surf) ** 2)
        return R, RotationTransformation(axis=rot_axis, angle=rot_angle, angle_in_radians=True)

    def _compare_molecules(self, mol_i: Molecule, mol_j: Molecule) -> bool:
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

    def _make_dummy_obs(self) -> Tuple[OrientedBulk, List[float]]:
        # Rotate the OBS
        structure = self.rotation_transformation.apply_transformation(
            self.obs.oriented_bulk_structure
        )
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
        # site_props.remove("molecule_index")
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
            sys.stdout.write(f"{altitudes=}\n")
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
            sys.stdout.write(f"{altitudes[top_ind]=}\n")
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
        frac_top_top = wrap_frac(molecule_top_tops / self.obs.layer_thickness)

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

        sys.stdout.write(f"{molecule_top_tops=}\n")
        sys.stdout.write(f"{frac_top_top=}\n")
        sys.stdout.write(f"{in_original_cell=}\n")
        sys.stdout.write(f"{len(molecule_graphs)=}\n")
        sys.stdout.write(f"{np.where(in_original_cell)[0]=}\n")
        sys.stdout.write(f"{len(m_graphs_in_cell)=}\n")

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
            sys.stdout.write(f"{is_top=}\n")

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
            is_same = self._compare_molecules(
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
        struc_props = {
            "molecules": molecules,
        }
        struc_props.update(props_in_cell)

        dummy_struc = Structure(
            lattice=structure.lattice,
            coords=frac_coords,
            species=species,
            site_properties=struc_props,
        )
        sys.stdout.write(f'{dummy_struc.site_properties["molecules"]=}\n')
        # dummy_struc.sort()

        dummy_obs = OrientedBulk(dummy_struc, [0, 0, 1], self.make_planar)
        sys.stdout.write(f'{dummy_obs.site_properties["molecules"]=}\n')
        raw_altitudes = [molecule_top_tops[i] for i in np.where(in_original_cell)[0]]
        statures = [molecule_statures[i] for i in np.where(in_original_cell)[0]]
        clustering_tolerance = min(statures) * self.clustering_tolerance_scale
        return dummy_obs, raw_altitudes, clustering_tolerance

    def _calculate_possible_shifts(self) -> List[float]:
        h = self.dummy_obs.layer_thickness
        headrooms = [h - raw_altitude for raw_altitude in self.raw_altitudes]
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
        clusters = fcluster(z, self.clustering_tolerance, criterion="distance")

        # Generate dict of cluster# to c val - doesn't matter what the c is.
        c_loc = {c: frac_shifts[i] for i, c in enumerate(clusters)}

        # Put all shifts into the unit cell.
        possible_shifts = [wrap_frac(frac_shift) for frac_shift in sorted(c_loc.values())]

        return [h * possible_shift for possible_shift in possible_shifts]

    def _apply_possible_shifts(self) -> List[Structure]:
        shifted_obses = []
        for possible_shift in self.possible_shifts:

            slab_base = deepcopy(self.dummy_obs)
            # sys.stdout.write(f'{slab_base.oriented_bulk_structure.site_properties["molecules"]=}\n')
            sys.stdout.write(f'{self.dummy_obs.site_properties["molecules"]=}\n')
            slab_base.translate_sites(
                vector=[0, 0, possible_shift - self.surface_buffer],
                frac_coords=False,
            )
            slab_base.round(tol=6)
            shifted_obses.append(slab_base)  # .oriented_bulk_structure)  # _

        return shifted_obses

    def _undummify(self) -> List[Structure]:
        undummified = []
        for dummy in self.shifted_dummy_cells:
            undummified.append(add_molecules(structure=dummy))
        return undummified

    def _surface_voxels(self) -> List[SurfaceVoxels]:
        surface_voxels = []
        for shifted_cell in self.shifted_cells:
            surface_voxels.append(
                SurfaceVoxels(
                    unit_cell=shifted_cell,
                    probe_rad=self.probe_rad,
                    attrv_adj=self.attrv_adj,
                    precision=self.precision,
                    scan_step=self.scan_step,
                    delta_z=self.delta_z,
                    raw_z_adj=self.raw_z_adj,
                    z_adj_mode=self.z_adj_mode,
                )
            )
        return surface_voxels

    def _average_roughnesses(self) -> List[float]:
        average_roughnesses = []
        for surface_voxels in self.surface_voxels:
            average_roughnesses.append(surface_voxels.average_roughness)
        return average_roughnesses

    def _maximum_acceptable_average_roughness(self) -> float:
        return self.max_roughness_factor * min(self.average_roughnesses)

    def _valid_terminations(self) -> List[Structure]:
        valid_terminations = []
        minimum_average_roughness = min(self.average_roughnesses)
        valid_terminations.append(
            self.shifted_cells[self.average_roughnesses.index(minimum_average_roughness)]
        )

        for shifted_cell, average_roughness in zip(self.shifted_cells, self.average_roughnesses):
            if (
                average_roughness <= self.maximum_acceptable_average_roughness
                and average_roughness > minimum_average_roughness
            ):
                valid_terminations.append(shifted_cell)

        return valid_terminations

    def all_terms(self) -> List[Structure]:
        return self.shifted_cells

    def terminations(self, return_all: bool = True):
        if return_all:
            return self.shifted_cells
        return self.valid_terminations

    def visualize(self, save_dir: str = None) -> None:
        if save_dir is None:
            for surface_voxels in self.surface_voxels:
                surface_voxels.visualize()
        else:
            planar = "planar" if self.make_planar else "nonplanar"
            for surface_voxels, shift, average_roughness in zip(
                self.surface_voxels, self.possible_shifts, self.average_roughnesses
            ):
                surface_voxels.visualize(
                    save_to=os.path.join(
                        save_dir,
                        self.species,
                        self.refined,
                        planar,
                        self.plane_name,
                        f"surface_voxels_shifted_{shift:.3f}_roughness_{average_roughness:.3f}.png",
                    )
                )


def normalize(vector):
    return vector / np.linalg.norm(vector)


def wrap_frac(frac_coord: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    if type(frac_coord) == float:
        return frac_coord - math.floor(frac_coord)
    return frac_coord - np.floor(frac_coord)


def find_rotation_matrix(orig_v: np.ndarray, dest_v: Union[np.ndarray, str]):
    orig_v = normalize(orig_v)

    if type(dest_v) == str:
        str_to_vector = {
            "x": np.array([1, 0, 0]),
            "y": np.array([0, 1, 0]),
            "z": np.array([0, 0, 1]),
        }
        dest_v = str_to_vector[dest_v]
    dest_v = normalize(dest_v)

    # sys.stdout.write(f"{orig_v=}\n")
    # sys.stdout.write(f"{dest_v=}\n")
    rot_axis = np.cross(orig_v, dest_v)
    if np.linalg.norm(rot_axis) == 0:
        if np.allclose(orig_v, dest_v):
            return np.eye(3)
        else:
            sys.stdout.write("Vectors are opposite to one another!!\n")
            rot_axis = np.cross(orig_v, np.random.rand(3))

    rot_axis = normalize(rot_axis)
    a_x, a_y, a_z = rot_axis
    K = np.array([[0, -a_z, a_y], [a_z, 0, -a_x], [-a_y, a_x, 0]])

    rot_angle = np.arccos(orig_v @ dest_v)
    R = np.eye(3) + np.sin(rot_angle) * K + (1 - np.cos(rot_angle)) * (K @ K)
    sys.stdout.write(f"\n\n{R=}\n\n")
    return R, rot_axis, rot_angle


def get_molecule_graphs(structure: Structure) -> List[nx.Graph]:
    struc_graph = StructureGraph.from_local_env_strategy(structure, JmolNN())
    sys.stdout.write("made struc_graph\n")
    cell_graph = nx.Graph(struc_graph.graph)
    sys.stdout.write("made cell_graph\n")
    return [cell_graph.subgraph(c) for c in nx.connected_components(cell_graph)]


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
    return squareform(pdist([site.coords for site in structure.sites])).max()


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


def homogenize(points: np.ndarray) -> np.ndarray:
    return np.vstack((points, np.ones((1, points.shape[1]))))


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
        "--probe_rad",
        type=float,
        default=1.2,
        help="Probe radius",
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
        default=40,
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
        "--raw_z_adj",
        type=float,
        default=0.0,
        help="Flat addition to z-coordinates of shifted unit cells",
    )
    parser.add_argument(
        "--z_adj_mode",
        type=str,
        default="sub_min_z",
        help="Shifted unit cell z-coordinate adjustment mode",
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
    # obs = Structure.from_file(os.path.join(os.getcwd(), args.structure_path))

    split_structure_path = args.structure_path.split(os.sep)
    plane_name = split_structure_path[-1].split("_")[-1].split(".")[0]
    plane = plane_from_name(plane_name)
    planar = True if split_structure_path[-2] == "planar" else False
    refined = True if split_structure_path[-3] == "refined" else False
    species = split_structure_path[-4]

    # obs = OrientedBulk(bulk=obs, miller_index=[0, 0, 1], make_planar=planar)
    obs = OrientedBulk(bulk=bulk, miller_index=plane, make_planar=planar)

    delta_z = args.delta_z if args.delta_z != 0.0 else molecular_width(obs.oriented_bulk_structure)

    terminator = Terminator(
        bulk=bulk,
        obs=obs,
        plane=plane,
        make_planar=planar,
        probe_rad=args.probe_rad,
        attrv_adj=args.attrv_adj,
        precision=args.precision,
        scan_step=args.scan_step,
        delta_z=delta_z,
        raw_z_adj=args.raw_z_adj,
        z_adj_mode=args.z_adj_mode,
        out_dir=os.path.join(os.getcwd(), args.struct_out_dir),
        species=species,
    )

    if args.visualize:
        terminator.visualize(save_dir=os.path.join(os.getcwd(), args.viz_out_dir))
