import argparse
from copy import deepcopy
from itertools import combinations, groupby
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

from OgreInterface.surfaces.surface import Surface
from OgreInterface.surfaces.molecular_surface import MolecularSurface
from OgreInterface.surfaces.oriented_bulk import OrientedBulk
from OgreInterface.utils import (
    get_unique_miller_indices,
    get_rounded_structure,
    sort_slab,
    get_layer_supercell,
)

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

        Creating a SurfacePrism object from a pymatgen Structure via the from_structure() class
            method:
        >>> from OgreInterface.surfaces.terminations import SurfacePrism
        >>> from pymatgen.core.structure import Structure
        >>> structure = Structure.from_file('structure.cif')
        >>> surface_prism = SurfacePrism.from_structure(structure=structure)

    Arguments:
        vertices (np.ndarray): Column-wise array of three or four consecutive adjacent
            vertices of the bounding surface parallelogram in Cartesian coordinate space.
        surface_normal (np.ndarray): A normal vector of the surface in Cartesian coordinate
            space. If not provided, then it will be computed from the passed vertices. Passing a
            surface_normal that is not derived from the same unit cell from which vertices is
            derived may lead to unexpected behavior.
        shift (np.ndarray): 3-element vector of ints representing the number of a-, b-, and c-
            lattice vectors by which to shift the parallelipiped. Useful for masking supercells.

    Attributes:
        vertices (np.ndarray): Column-wise 3x4 array of the four consecutive adjacent vertices
            of the bounding surface parallelogram in Cartesian coordinate space.
        surface_normal (np.ndarray): Unit normal vector of the surface in Cartesian coordinate
            space.
        shift (np.ndarray): 3-element vector of ints representing the number of a-, b-, and c-
            lattice vectors by which to shift the parallelipiped. Useful for masking supercells.
        structure (Structure): pymatgen Structure from which the parallelipiped was constructed. If
            no structure kwarg is passed to .mask_structure(), then this Structure is masked
            instead.
        face_plane_equations (np.ndarray): Row-wise 4x4 array of the parallelipiped's faces' plane
            equation coefficients (Ax + By + Cz + D = 0). Rows ordered such that the vertex at
            column index i of ._vertices and that at column index i+1 are both on the plane
            described by the plane equation with coefficients at row index i of
            ._face_plane_equations.
        inside_sides (np.ndarray): 4-element vector of values of Ax + By + Cz + D for an example
            point (x, y, z) that is on the same side of the index-respective face of the
            parallelipiped as the rest of the parallelipiped. The signs of the elements can be
            used to determine whether an arbitrary point is on the inside side of the respective
            face of the parallelipiped.
        inward_face_normals (np.ndarray): Row-wise 4x3 array of inward-facing unit normal vectors
            of the row index-corresponding faces of the parallelipiped.

    Methods:
        mask_points(points: np.ndarray) -> np.ndarray: Mask the passed array of Cartesian
            points.
        mask_structure(structure: Structure,
                       in_place: bool = False,
                       snug_corner: bool = False,
                       orthogonalize_c: bool = False,
                       unshift: bool = True) -> Structure: Mask the passed Structure's sites.
        buffer_mask_supercell(supercell: Structure, in_place: bool = False) -> Structure: Mask the
            passed supercell's sites, keeping any exterior atoms that intersect with the
            parallelipiped.

    Class methods:
        from_matrix(matrix: np.ndarray, surface_normal: np.ndarray = None) -> SurfacePrism:
            Create a SurfacePrism object from the passed lattice matrix.
        from_structure(structure: Structure, surface_normal: np.ndarray = None) -> SurfacePrism:
            Create a SurfacePrism object from the passed pymatgen Structure.
    """

    def __init__(
        self, vertices: np.ndarray, surface_normal: np.ndarray = None, shift: np.ndarray = None
    ) -> None:
        self._vertices = (
            self._complete_vertices(vertices) if vertices.shape[1] == 3 else vertices.copy()
        )
        self._surface_normal = (
            surface_normal.copy() / np.linalg.norm(surface_normal.copy())
            if not surface_normal is None
            else self._calc_surface_normal()
        )
        self._shift = shift
        self._structure = None
        self._face_plane_equations = self._calc_face_plane_equations()
        self._inside_sides = self._calc_inside_sides()
        self._inward_face_normals = self._calc_inward_face_normals()

    @property
    def vertices(self) -> np.ndarray:
        """Column-wise 3x4 array of the four consecutive adjacent vertices of the bounding surface
        parallelogram in Cartesian coordinate space.
        """

        return self._vertices.copy()

    @property
    def surface_normal(self) -> np.ndarray:
        """Unit normal vector of the surface in Cartesian coordinate space."""
        return self._surface_normal.copy()

    @property
    def shift(self) -> np.ndarray:
        """3-element vector of ints representing the number of a-, b-, and c-lattice vectors by
        which to shift the parallelipiped. Useful for masking supercells.
        """

        return self._shift.copy()

    @property
    def structure(self) -> Structure:
        """pymatgen Structure from which the parallelipiped was constructed. If no structure kwarg
        is passed to SurfacePrism.mask_structure(), then this Structure is masked instead.
        """

        return self._structure.copy() if self._structure else None

    @property
    def face_plane_equations(self) -> np.ndarray:
        """Row-wise 4x4 array of the parallelipiped's faces' plane equation coefficients.

        Row-wise 4x4 array of the parallelipiped's faces' plane equation coefficients
        (Ax + By + Cz + D = 0). Rows ordered such that the vertex at column index i of
        SurfacePrism.vertices and that at column index i+1 are both on the plane described by the
        plane equation with coefficients at row index i of SurfacePrism.face_plane_equations.
        """

        return self._face_plane_equations.copy()

    @property
    def inside_sides(self) -> np.ndarray:
        """4-element vector of values of Ax + By + Cz + D for an example point (x, y, z) that is on
        the same side of the index-respective face of the parallelipiped as the rest of the
        parallelipiped. The signs of the elements can be used to determine whether an arbitrary
        point is on the inside side of the respective face of the parallelipiped.
        """

        return self._inside_sides.copy()

    @property
    def inward_face_normals(self) -> np.ndarray:
        """Row-wise 4x3 array of inward-facing unit normal vectors of the row index-corresponding
        face of the parallelipiped.
        """

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
        """Calculate inward-facing unit normal vectors of the faces of the parallelipiped."""

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
        """Return whether any part of the vdW sphere of the passed site is inside the
        parallelipiped.
        """

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
            points (np.ndarray): Column-wise array of Cartesian points to be masked.
        """

        return points[:, self._mask(points=points)]

    def mask_structure(
        self,
        structure: Structure = None,
        in_place: bool = False,
        snug_corner: bool = False,
        orthogonalize_c: bool = False,
        unshift: bool = True,
    ) -> Structure:
        """Mask the passed pymatgen Structure.

        Mask the passed pymatgen Structure's sites to exclude all sites located outside of the
        surface-orthogonal parallelipiped with unbounded bases. A 3x3x1 supercell or a nonperiodic
        cell rather than a periodic unit cell is recommended, as a periodic unit cell may undo the
        masking by applying periodic boundary conditions.

        Keyword Arguments:
            structure (Structure): Structure whose sites are to be masked. A 3x3x1 supercell or a
                nonperiodic cell rather than a periodic unit cell is recommended, as a periodic
                unit cell may undo the masking by applying periodic boundary conditions. Default is
                None, in which case the ._structure attribute will be used. In that case, the
                SurfacePrism object must have been instantiated via the
                SurfacePrism.from_structure() class method or the ._structure attribute must have
                been set manually.
            in_place (bool): Whether to edit the passed Structure in-place. If False,
                mask_structure() will also return a masked copy of the passed Structure without
                affecting the passed Structure. Default is False.
            snug_corner (bool): Whether to shift the sites into the corner formed by the lattice
                vectors. Default is False.
            orthogonalize_c (bool): Whether to force the c-vector to be colinear with the Cartesian
                z-axis. Default is False.
            unshift (bool): Whether to un-shift the masked sites according to the ._shift arrtibute.
                Default is True.
        """

        structure = self._structure.copy() if structure is None else structure
        if structure is None:
            raise ValueError(
                "kwarg 'structure' is required unless SurfacePrism object was instantiated via \
                    SurfacePrism.from_structure()"
            )

        if in_place and not snug_corner and not orthogonalize_c and not unshift:
            structure.remove_sites(
                indices=np.where(self._mask(points=structure.cart_coords.T) == False)[0]
            )
            return structure

        masked_structure = structure.copy()
        masked_structure.remove_sites(
            indices=np.where(self._mask(points=structure.cart_coords.T) == False)[0]
        )

        if orthogonalize_c:
            ortho_c_matrix = masked_structure.lattice.matrix.copy()
            ortho_c_matrix[-1, :2] = 0.0
            lattice = Lattice(matrix=ortho_c_matrix, pbc=(True, True, True))
        else:
            lattice = masked_structure.lattice

        coords = (
            masked_structure.cart_coords - np.append(masked_structure.lattice.matrix[2, :2], 0.0)
            if snug_corner
            else masked_structure.cart_coords
        )

        if unshift and not self._shift is None:
            coords -= masked_structure.lattice.matrix.T @ self._shift

        return Structure(
            lattice=lattice,
            species=masked_structure.species,
            coords=coords,
            coords_are_cartesian=True,
            to_unit_cell=False,
        )

    def buffer_mask_supercell(
        self,
        supercell: Structure,
        in_place: bool = False,
    ) -> Structure:
        """Mask the passed supercell's sites, keeping any exterior atoms that intersect with the
        parallelipiped.

        Arguments:
            supercell (Structure): The supercell whose sites are to be buffer-masked.

        Keyword Arguments:
            in_place (bool): Whether to edit the passed supercell in-place. If False,
                buffer_mask_supercell() will also return a buffer-masked copy of the passed
                supercell without affecting the passed supercell. Default is False.
        """

        to_remove = []
        for i, site in enumerate(supercell):
            if not self._infiltrates_any_face(site=site):
                to_remove.append(i)

        if in_place:
            supercell.remove_sites(indices=to_remove)
            return supercell

        masked_supercell = supercell.copy()
        masked_supercell.remove_sites(indices=to_remove)
        return masked_supercell

    @classmethod
    def from_matrix(
        cls, matrix: np.ndarray, surface_normal: np.ndarray = None, shift: np.ndarray = None
    ) -> SelfSurfacePrism:
        """Create a SurfacePrism object from the passed lattice matrix.

        Create a SurfacePrism object from the passed lattice matrix (i.e., row-wise numpy
        ndarray of lattice vectors). If a normal vector of the surface in Cartesian coordinate space
        is not provided, then it will be computed from the passed lattice matrix.

        Arguments:
            matrix (np.ndarray): The lattice matrix, i.e., row-wise numpy ndarray of lattice
                vectors, from which the SurfacePrism object is to be constructed. The "surface,"
                according to the SurfacePrism object, is coplanar with the top (in the c-direction)
                ab-face of the unit cell that this lattice matrix describes.
            surface_normal (np.ndarray): A normal vector of the surface in Cartesian coordinate
                space. If not provided, then it will be computed from the passed lattice matrix.
                Passing a surface_normal that is not derived from the same unit cell from which
                matrix is derived may lead to unexpected behavior.
            shift (np.ndarray): 3-element vector of ints representing the number of a-, b-, and
                c-lattice vectors by which to shift the parallelipiped. Useful for masking
                supercells.
        """

        a, b, c = matrix
        vertices = np.vstack((c, c + b, c + b + a, c + a)).T
        if not shift is None:
            translation_vector = matrix.T @ shift
            vertices = vertices + translation_vector[:, np.newaxis]  # matrix.T @ shift

        if surface_normal is None:
            surface_norm = np.cross(a, b)
            surface_norm /= np.linalg.norm(surface_norm)
        else:
            surface_norm = surface_normal.copy()

        return cls(vertices=vertices, surface_normal=surface_norm, shift=shift)

    @classmethod
    def from_structure(
        cls, structure: Structure, surface_normal: np.ndarray = None
    ) -> SelfSurfacePrism:
        """Create a SurfacePrism object from the passed pymatgen Structure.

        Create a SurfacePrism object from the passed pymatgen Structure. If a
        normal vector of the surface in Cartesian coordinate space is not provided, then it will be
        computed from the lattice matrix of the passed Structure.

        Arguments:
            structure (Structure): The pymatgen Structure object from which
                the SurfacePrism object is to be constructed. The "surface," according to the
                SurfacePrism object, is coplanar with the top (in the c-direction) ab-face of the
                unit cell that this Structure represents.
            surface_normal (np.ndarray): A normal vector of the surface in Cartesian coordinate
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
    """_summary_"""

    def __init__(
        self,
        original_cell: Structure,
        surface_normal: np.ndarray = None,
        molecule_width: float = None,
        atom_count: int = None,
    ) -> None:
        """_summary_

        Arguments:
            original_cell -- _description_

        Keyword Arguments:
            surface_normal -- _description_ (default: {None})
            molecule_width -- _description_ (default: {None})
            atom_count -- _description_ (default: {None})
        """
        self._original_cell = Structure(
            lattice=deepcopy(original_cell.lattice),
            species=original_cell.species.copy(),
            coords=original_cell.cart_coords.copy(),
            coords_are_cartesian=True,
            to_unit_cell=True,
        )
        self._surface_normal = (
            surface_normal.copy() if not surface_normal is None else self._calc_surface_normal()
        )
        self._molecule_width, self._atom_count = self._molecule_data(
            molecule_width=molecule_width, atom_count=atom_count
        )
        self._unit_height = self._original_cell.lattice.matrix[-1] @ self._surface_normal
        self._c_scale = math.ceil(self._molecule_width / self._unit_height)
        self._structure, self._height = self._derive_structure()
        self._bounds = self._calc_bounds()
        if debug:
            self._structure.to(os.path.join(os.getcwd(), "buffered_structure.cif"))

    def _molecule_data(
        self, molecule_width: float = None, atom_count: int = None
    ) -> Tuple[float, int]:
        """_summary_

        Keyword Arguments:
            molecule_width -- _description_ (default: {None})
            atom_count -- _description_ (default: {None})

        Returns:
            _description_
        """
        if molecule_width is None:
            if atom_count is None:
                return molecule_data(structure=self._original_cell)
            return molecular_width(structure=self._original_cell), atom_count
        return molecule_width, count_atoms(structure=self._original_cell)

    @property
    def atom_count(self) -> int:
        """_summary_

        Returns:
            _description_
        """
        return self._atom_count

    @property
    def original_cell(self) -> Structure:
        """_summary_

        Returns:
            _description_
        """
        return self._original_cell.copy()

    @property
    def surface_normal(self) -> np.ndarray:
        """_summary_

        Returns:
            _description_
        """
        return self._surface_normal.copy()

    @property
    def molecule_width(self) -> float:
        """_summary_

        Returns:
            _description_
        """
        return self._molecule_width

    @property
    def unit_height(self) -> float:
        """_summary_

        Returns:
            _description_
        """
        return self._unit_height

    @property
    def c_scale(self) -> int:
        """_summary_

        Returns:
            _description_
        """
        return self._c_scale

    @property
    def structure(self) -> Structure:
        """_summary_

        Returns:
            _description_
        """
        return self._structure.copy()

    @property
    def height(self) -> float:
        """_summary_

        Returns:
            _description_
        """
        return self._height

    @property
    def bounds(self) -> np.ndarray:
        """_summary_

        Returns:
            _description_
        """
        return self._bounds.copy()

    def _calc_surface_normal(self) -> np.ndarray:
        """_summary_

        Returns:
            _description_
        """
        a, b, _ = self._original_cell.lattice.matrix
        return normalize(np.cross(normalize(a), normalize(b)))

    def _orig_cell(self) -> Structure:
        """_summary_

        Returns:
            _description_
        """
        coords_to_keep, species_to_keep, max_head = [], [], 0.0
        start_timer("molecule_graphs")
        mol_graphs = get_molecule_graphs(structure=self._original_cell)
        stop_timer("molecule_graphs")
        start_timer("slug planing")
        for mol_graph in mol_graphs:  # get_molecule_graphs(supercell_slug):
            keep_molecule, local_max_head = True, 0.0

            for node in mol_graph.nodes:
                site = self._original_cell[node]
                top = site.coords[-1] + site.specie.van_der_waals_radius

                if top > self._unit_height:
                    keep_molecule = False
                    break

                local_max_head = max(local_max_head, top)

            if keep_molecule:
                max_head = max(max_head, local_max_head)
                for node in mol_graph.nodes:
                    site = self._original_cell[node]
                    coords_to_keep.append(site.coords)
                    species_to_keep.append(site.specie)
        return Structure(
            lattice=deepcopy(self._original_cell.lattice),
            species=species_to_keep,
            coords=coords_to_keep,
            coords_are_cartesian=True,
        )

    def _planed_slug(self, supercell_slug: Structure) -> Tuple[Structure, float]:
        """_summary_

        Arguments:
            supercell_slug -- _description_

        Returns:
            _description_
        """
        ceiling = (
            supercell_slug.lattice.matrix[-1, -1] + 0.000002
        )  # 1# (empirically 1.7E-6 Angstroms?)This tolerance comes from when the altitudes and then the shifted slabs are rounded to 6 decimal places; the Cartesian z-coordinate of the intended top is on the interval (t - 0.000001 Angstroms, t + 0.000001 Angstroms], where t is the actual height of the cell. Due to the aforementioned intended top placement tolerance of 1.5E-6 Angstroms, the distance between the top of the cell and an atom's highest point that should have been shifted to infinitesimally below the top of the overall top atom (from which the shift was derived) will be placed at most 1E-6 Angstroms below where the top of the overall top atom will be placed; roughness tolerance for an arbitrary point (not voxel) is 1E-6 Angstroms, for differential 1E-6 Angstroms.#self._c_scale * self._unit_height
        # Collect coordinates and species to keep, and compute max head
        coords_to_keep, species_to_keep, max_head = [], [], 0.0
        mol_graphs = get_molecule_graphs(structure=supercell_slug)
        molecules_removed = 0
        underground_atoms_removed = 0
        local_max_heads = []
        for mol_graph in mol_graphs:
            keep_molecule, local_max_head = True, 0.0

            for node in mol_graph.nodes:
                site = supercell_slug[node]
                top = site.coords[-1] + site.specie.van_der_waals_radius

                if top > ceiling:
                    keep_molecule = False
                    molecules_removed += 1
                    break

                local_max_head = max(local_max_head, top)
                local_max_heads.append(local_max_head)

            if keep_molecule:
                max_head = max(max_head, local_max_head)
                for node in mol_graph.nodes:
                    site = supercell_slug[node]
                    if site.coords[-1] >= 0.0:
                        coords_to_keep.append(site.coords)
                        species_to_keep.append(site.specie)
                    else:
                        underground_atoms_removed += 1
        if molecules_removed > 0 or underground_atoms_removed > 0:
            sys.stdout.write(" (!!!)\n")

        # Create the structure
        planed_slug = Structure(
            lattice=supercell_slug.lattice,
            species=species_to_keep,
            coords=np.array(coords_to_keep),
            coords_are_cartesian=True,
            to_unit_cell=False,
        )

        return planed_slug, max_head

    def _derive_structure(self) -> Structure:
        """_summary_

        Returns:
            _description_
        """
        original_cell = self._original_cell
        buffer = 1.0
        matrix = original_cell.lattice.matrix.copy()
        translation_vector = (buffer / matrix[-1, -1]) * matrix[-1]
        matrix[-1] += translation_vector
        cell_with_buns = Structure(
            lattice=Lattice(matrix=matrix),
            species=original_cell.species.copy(),
            coords=original_cell.cart_coords + (translation_vector / 2.0),
            coords_are_cartesian=True,
            to_unit_cell=False,
        )

        coords_to_keep, species_to_keep = [], []
        mol_graphs = get_molecule_graphs(structure=cell_with_buns)
        for mol_graph in mol_graphs:
            keep_molecule = True

            for node in mol_graph.nodes:
                site = cell_with_buns[node]
                top = site.coords[-1] + site.specie.van_der_waals_radius - buffer

                if top > self._unit_height:
                    keep_molecule = False
                    break

            if keep_molecule:
                for node in mol_graph.nodes:
                    site = cell_with_buns[node]
                    if site.coords[-1] >= buffer:
                        coords_to_keep.append(site.coords - (translation_vector / 2.0))
                        species_to_keep.append(site.specie)

        planed_unit_cell = Structure(
            lattice=deepcopy(self._original_cell.lattice),
            species=species_to_keep,
            coords=coords_to_keep,
            coords_are_cartesian=True,
            to_unit_cell=False,
        )

        if len(planed_unit_cell) == 0:
            sys.stdout.write(f"\nNo sites in planed_unit_cell!!!\n")
            print_structure(structure=cell_with_buns, name="preceding_cell_with_buns")
            print_structure(structure=original_cell, name="preceding_original_cell")

        surface_layer = planed_unit_cell.make_supercell(
            scaling_matrix=[3, 3, 1], to_unit_cell=True, in_place=False
        )

        if self._c_scale > 1:
            underground_layers = original_cell.make_supercell(
                scaling_matrix=[3, 3, self._c_scale - 1], to_unit_cell=True, in_place=False
            )
            supercell_matrix = underground_layers.lattice.matrix.copy()
            supercell_matrix[-1] += planed_unit_cell.lattice.matrix[-1]
            supercell_species = surface_layer.species + underground_layers.species
            surface_coords = surface_layer.cart_coords + original_cell.lattice.matrix[-1] * (
                self._c_scale - 1
            )
            supercell_coords = np.append(surface_coords, underground_layers.cart_coords, axis=0)

            raw_supercell = Structure(
                lattice=Lattice(matrix=supercell_matrix),
                species=supercell_species,
                coords=supercell_coords,
                coords_are_cartesian=True,
                to_unit_cell=True,
            )
        else:
            raw_supercell = surface_layer

        # Create the surface prism and mask the structure
        surface_prism = SurfacePrism.from_matrix(
            matrix=self._original_cell.lattice.matrix,
            surface_normal=self._surface_normal,
            shift=np.array([1, 1, 0]),
        )

        supercell_slug = surface_prism.mask_structure(
            raw_supercell, in_place=False, snug_corner=False, orthogonalize_c=True, unshift=True
        )

        matrix = self._original_cell.lattice.matrix.copy()
        matrix[-1] = supercell_slug.lattice.matrix[-1]
        supercell_slug = Structure(
            lattice=Lattice(matrix=matrix),
            species=supercell_slug.species,
            coords=supercell_slug.cart_coords,
            coords_are_cartesian=True,
            to_unit_cell=True,
        )
        if len(supercell_slug) == 0:
            sys.stdout.write(f"\nNo sites in supercell_slug!!!\n")

        planed_slug, max_head = self._planed_slug(supercell_slug=supercell_slug)
        if len(planed_slug) == 0:
            sys.stdout.write(f"\nNo sites in planed_slug!!!\n")
            print_structure(structure=supercell_slug, name="preceding_supercell_slug")
            print_structure(structure=raw_supercell, name="preceding_raw_supercell")
            print_structure(structure=surface_layer, name="preceding_surface_layer")
            print_structure(structure=underground_layers, name="preceding_underground_layers")
            print_structure(structure=planed_unit_cell, name="preceding_planed_unit_cell")
            print_structure(structure=cell_with_buns, name="preceding_cell_with_buns")
            print_structure(structure=original_cell, name="preceding_original_cell")

        # Make the supercell and buffer it
        buffered_prism = SurfacePrism.from_matrix(planed_slug.lattice.matrix, self._surface_normal)
        planed_slug.make_supercell([3, 3, 1], to_unit_cell=True, in_place=True)
        buffered_prism.buffer_mask_supercell(planed_slug, in_place=True)
        counter["count"] += 1
        return planed_slug, max_head

    def _calc_bounds(self) -> np.ndarray:
        """_summary_

        Returns:
            _description_
        """
        a, bx, by = (
            self._original_cell.lattice.matrix[0, 0],
            self._original_cell.lattice.matrix[1, 0],
            self._original_cell.lattice.matrix[1, 1],
        )
        x_bounds = [bx, a] if bx < 0 else [0, bx + a]
        y_bounds = [by, 0] if by < 0 else [0, by]
        z_bounds = [0, self._height]
        return np.vstack((x_bounds, y_bounds, z_bounds))


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
    ) -> None:
        self._rotated_dummy_obs = deepcopy(rotated_dummy_obs)
        self._structures = {
            "rotated": {
                "dummy": self._rotated_dummy_obs,  # .oriented_bulk_structure,
                "molecular": rotated_molecular_structure,
            },
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

        if unrotate:
            if unrotation_transformations:
                if unrotated_dummy_structure is None:
                    self._structures["unrotated"]["dummy"] = self._unrotate(
                        structure=self._structures["rotated"]["dummy"]
                    )
                if (
                    self._structures["rotated"]["molecular"]
                    and unrotated_molecular_structure is None
                ):
                    self._structures["unrotated"]["molecular"] = self._unrotate(
                        structure=self._structures["rotated"]["molecular"]
                    )
            else:
                print("unrotation_transformations must be supplied in order to unrotate.")

        if undummify:
            if rotated_molecular_structure is None:
                self._structures["rotated"]["molecular"] = add_molecules(
                    structure=self._rotated_dummy_obs
                )
            if self._structures["unrotated"]["dummy"] and unrotated_molecular_structure is None:
                if unrotation_transformations:
                    self._structures["unrotated"]["molecular"] = self._unrotate(
                        structure=self._structures["rotated"]["molecular"]
                    )
                else:
                    print(
                        "unrotation_transformations must be supplied in order to obtain the \
                        unrotated molecular structure."
                    )

    @property
    def rotated_dummy_obs(self) -> Structure:  # OrientedBulk:
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
        self,
        unrotation_transformations: Tuple[RotationTransformation],
        undummify: bool = False,
        unrotate: bool = False,
    ) -> None:
        self._unrotation_transformations = unrotation_transformations

        if unrotate:
            self._structures["unrotated"]["dummy"] = self._unrotate(
                structure=self._structures["rotated"]["dummy"]
            )
            if self._structures["rotated"]["molecular"]:
                self._structures["unrotated"]["molecular"] = self._unrotate(
                    structure=self._structures["rotated"]["molecular"]
                )

        if undummify:
            self._structures["rotated"]["molecular"] = add_molecules(
                structure=self._rotated_dummy_obs
            )
            if self._structures["unrotated"]["dummy"]:
                self._structures["unrotated"]["molecular"] = self._unrotate(
                    structure=self._structures["rotated"]["molecular"]
                )


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
        molecular: bool = True,
        generate_all: bool = True,
        molecule_width: float = None,
        atom_count: int = None,
        terminations: List[Termination] = None,
        num_layers: int = None,
        vacuum: float = 60.0,
    ) -> None:
        self._bulk = bulk
        self._plane = plane
        plane_counter["plane"] = miller_name(plane)
        self._obs = OrientedBulk(bulk=bulk, miller_index=plane, make_planar=False)

        self._clustering_tolerance_scale = clustering_tolerance_scale

        self._attrv_adj = attrv_adj
        self._precision = precision
        self._scan_step = scan_step

        self._molecule_width, self._atom_count = self._molecule_data(
            molecule_width=molecule_width, atom_count=atom_count
        )
        self._delta_z = delta_z if delta_z <= 0.0 else self._molecule_width

        self.out_dir = out_dir
        self.species = species
        self._plane_name = miller_name(plane)
        self.molecular = molecular

        self.generate_all = generate_all
        self.num_layers = 1 if num_layers is None else num_layers
        self.vacuum = vacuum

        self._rotated_obs, self._rotation_params = self._rotate_obs()
        self._unrotation_transformations = self._calc_unrotation_transformations()

        self._dummy_obs, self._raw_altitudes, self._clustering_tolerance = self._make_dummy_obs()
        start_timer("Terminator._calculate_possible_shifts()")
        self._shifts = self._calc_shifts()
        stop_timer("Terminator._calculate_possible_shifts()")
        start_timer("Terminator._apply_possible_shifts()")
        self._shifted_dummy_obses, self._surfaces = self._apply_shifts()
        stop_timer("Terminator._apply_possible_shifts()")
        start_timer("Terminator._undummify()")
        self._shifted_cells = self._undummify()
        stop_timer("Terminator._undummify()")

        self._surface_voxels = None
        self._average_roughnesses = None
        self._terminations = (
            terminations if terminations else [None] * len(self._shifted_dummy_obses)
        )
        self._smoothest_surface = None

    def _molecule_data(
        self, molecule_width: float = None, atom_count: int = None
    ) -> Tuple[float, int]:
        if molecule_width is None:
            if atom_count is None:
                return molecule_data(structure=self._bulk)
            return molecular_width(structure=self._bulk), atom_count
        return molecule_width, count_atoms(structure=self._bulk)

    @property
    def atom_count(self) -> int:
        return self._atom_count

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
            if unrotate:
                return self._unrotate_structure(structure=self._dummy_obs)
            return (
                self._dummy_obs.copy()
            )  # .oriented_bulk_structure.copy() if structure_only else deepcopy(self._dummy_obs)

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
    def surfaces(self) -> List[Surface]:
        return deepcopy(self._surfaces)

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
            self._shifted_dummy_obses  # [obs.oriented_bulk_structure for obs in self._shifted_dummy_obses]
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

    @property
    def smoothest_surface(self) -> Surface:
        if self._smoothest_surface is None:
            self._average_roughnesses = self._calc_average_roughnesses()
        return deepcopy(self._smoothest_surface)

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
            self._shifted_dummy_obses  # [obs.oriented_bulk_structure for obs in self._shifted_dummy_obses]
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
            self._delta_z = self._molecule_width if delta_z == 0.0 else delta_z

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
        # NOTE: maybe this could be accomplished by counting the number of atoms in the molecule?
        mol_graphs = []
        for mol_graph in all_subgraphs:
            intersects_boundary = any(
                d["to_jimage"] != (0, 0, 0) for u, v, d in mol_graph.edges(data=True)
            )
            if not intersects_boundary:
                mol_graphs.append(nx.MultiDiGraph(mol_graph))

        # Get the center of mass and the molecule index
        molecule_top_centers = []
        molecule_top_tops = []
        molecule_statures = []
        site_props = list(structure.site_properties.keys())
        props = {p: [] for p in site_props}
        for mol_graph in mol_graphs:
            cart_coords = np.vstack([struc_graph.structure[node].coords for node in mol_graph])

            z_coords = np.array([struc_graph.structure[node].coords[-1] for node in mol_graph])
            vdW_radii = np.array(
                [struc_graph.structure[node].specie.van_der_waals_radius for node in mol_graph]
            )
            altitudes = z_coords + vdW_radii
            soles = z_coords - vdW_radii
            top_ind = np.argmax(altitudes)
            bottom = np.min(soles)

            top_position = cart_coords[top_ind]
            is_top = np.zeros(len(cart_coords)).astype(bool)
            is_top[top_ind] = True

            for t, n in zip(is_top, mol_graph):
                struc_graph.structure[n].properties["is_top"] = t

            for p in props:
                ind = list(mol_graph.nodes.keys())[0]
                props[p].append(struc_graph.structure[ind].properties[p])

            molecule_top_centers.append(np.round(top_position, 6))
            molecule_top_tops.append(np.round(altitudes[top_ind], 6))
            molecule_statures.append(np.round(altitudes[top_ind] - bottom, 6))

        molecule_top_centers = np.vstack(molecule_top_centers)
        alts = molecule_top_tops.copy()
        molecule_top_tops = np.hstack((molecule_top_centers[:, :2], np.vstack(molecule_top_tops)))

        # Now we can find which center of masses are contained in the original
        # unit cell. First we can shift the center of masses by the [1, 1, 1]
        # vector of the original unit cell so the center unit cell of the 3x3
        # supercell is positioned at (0, 0, 0)
        shift = structure.lattice.get_cartesian_coords([1, 1, 1])
        inv_matrix = structure.lattice.inv_matrix

        # Shift the center of masses
        molecule_top_centers -= shift
        molecule_top_tops -= shift

        # Convert to fractional coordinates of the original unit cell
        frac_top_center = molecule_top_centers.dot(inv_matrix)
        frac_top_top = molecule_top_tops.dot(inv_matrix)[:, -1].reshape(-1, 1)

        # The real tops of the reference atoms in the unit cell should have fractional
        # coordinates on [0, 1)
        in_original_cell = np.logical_and(
            0 <= np.round(frac_top_top, 6),
            np.round(frac_top_top, 6) < 1,
        ).all(axis=1)

        # Extract the fractional coordinates in the original cell
        frac_coords_in_cell = frac_top_center[in_original_cell]

        # Extract the molecules that have the reference atom in the unit cell
        m_graphs_in_cell = [mol_graphs[i] for i in np.where(in_original_cell)[0]]

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
        struc_props = {"molecules": molecules}
        struc_props.update(props_in_cell)

        dummy_struc = Structure(
            lattice=deepcopy(structure.lattice),
            coords=frac_coords,
            species=species,
            site_properties=struc_props,
            to_unit_cell=True,
        )

        dummy_obs = get_rounded_structure(structure=dummy_struc, tol=6)
        raw_altitudes = [alts[i] for i in np.where(in_original_cell)[0]]
        statures = [molecule_statures[i] for i in np.where(in_original_cell)[0]]
        clustering_tolerance = min(statures) * self._clustering_tolerance_scale
        return dummy_obs, raw_altitudes, clustering_tolerance

    def _calc_shifts(self, slab_base: Structure = None) -> List[float]:
        slab_base = self._dummy_obs if slab_base is None else slab_base
        a, b, _ = slab_base.lattice.matrix
        surface_normal = np.cross(a, b)
        surface_normal /= np.linalg.norm(surface_normal)

        h = slab_base.lattice.matrix[-1] @ surface_normal  # .layer_thickness
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

        # Generate dict of cluster# to maximum z-coordinate in cluster
        c_loc = {}
        for cluster in np.unique(clusters):
            c_loc[cluster] = min(frac_shifts[clusters == cluster])

        # Put all shifts into the unit cell.
        possible_shifts = [wrap_frac(frac_shift) for frac_shift in sorted(c_loc.values())]
        # wrapping again is probably unnecessary

        return [h * possible_shift for possible_shift in possible_shifts]

    def _apply_shifts(self) -> List[Tuple[OrientedBulk, OrientedBulk, float, Tuple[int, ...]]]:
        a, b, _ = self._dummy_obs.lattice.matrix
        surface_normal = np.cross(a, b)
        surface_normal /= np.linalg.norm(surface_normal)
        h = self._dummy_obs.lattice.matrix[-1] @ surface_normal
        vacuum_scale = self.vacuum // h

        # Ensure even vacuum scale for symmetric centering
        if vacuum_scale % 2:
            vacuum_scale += 1

        if vacuum_scale == 0:
            vacuum_scale = 2

        vacuum = h * vacuum_scale

        shifted_slab_bases = []
        non_orthogonal_slabs = []

        # Iterate over the possible shifts
        if self.generate_all:
            for i, possible_shift in enumerate(self._shifts):
                slab_base = self._dummy_obs.copy()

                # Apply the shift to the slab_base structure
                slab_base.translate_sites(
                    indices=range(len(slab_base)),
                    vector=[0, 0, possible_shift],
                    frac_coords=False,
                    to_unit_cell=True,
                )

                # Round and mod the slab structure
                slab_base = get_rounded_structure(slab_base, tol=6)
                slab_obs = OrientedBulk(bulk=slab_base, miller_index=[0, 0, 1], make_planar=False)

                # Extract fractional c-coordinates
                c_coords = slab_base.frac_coords[:, -1]

                # Calculate possible shifts for atomic layer grouping
                shifts = self._calc_shifts(slab_base=slab_base)
                shifts += [1.0]  # To close the last layer

                # Define atomic layer bounds
                atomic_layer_bounds = [(shifts[i], shifts[i + 1]) for i in range(len(shifts) - 1)]

                # Initialize atomic layer array
                atomic_layers = -np.ones(len(c_coords))

                # Determine atomic layers based on the shifts
                for i, (bottom_bound, top_bound) in enumerate(atomic_layer_bounds):
                    layer_mask = (c_coords > bottom_bound) & (c_coords < top_bound)
                    atomic_layers[layer_mask] = i

                # Assign atomic layer indices as site properties
                slab_obs.add_site_property(
                    "atomic_layer_index", np.round(atomic_layers).astype(int).tolist()
                )

                # Compute top and bottom c-coords and horizontal shift for centering
                top_c = c_coords.max()
                max_c_inds = np.where(np.isclose(top_c, c_coords))[0]

                dists = [
                    slab_obs[i].distance_and_image_from_frac_coords(fcoords=[0.0, 0.0, 0.0])[0]
                    for i in max_c_inds
                ]
                horiz_shift_ind = max_c_inds[np.argmin(dists)]
                horiz_shift = -slab_obs[horiz_shift_ind].frac_coords
                horiz_shift[-1] = 0  # Keep the c-direction shift at zero

                # Apply the horizontal shift for centering
                slab_obs.translate_sites(vector=horiz_shift, frac_coords=True)
                slab_obs.round(tol=6)

                # Create non-orthogonal slab
                non_orthogonal_slab = get_layer_supercell(
                    structure=slab_obs._oriented_bulk_structure,
                    layers=self.num_layers,
                    vacuum_scale=vacuum_scale,
                )
                sort_slab(non_orthogonal_slab)

                # Center the slab in the vacuum region
                center_shift = 0.5 * (vacuum_scale / (vacuum_scale + self.num_layers))
                non_orthogonal_slab.translate_sites(
                    indices=range(len(non_orthogonal_slab)),
                    vector=[0, 0, center_shift],
                    frac_coords=True,
                    to_unit_cell=True,
                )

                non_orthogonal_slab.sort_index = i
                shifted_slab_bases.append(slab_obs)
                non_orthogonal_slabs.append(non_orthogonal_slab)
        else:
            slab_base = self._dummy_obs.copy()

            # Apply the shift to the slab_base structure
            slab_base.translate_sites(
                indices=range(len(slab_base)),
                vector=[0, 0, self._shifts[0]],
                frac_coords=False,
                to_unit_cell=True,
            )

            # Round and mod the slab structure
            slab_base = get_rounded_structure(slab_base, tol=6)
            slab_obs = OrientedBulk(bulk=slab_base, miller_index=[0, 0, 1], make_planar=False)

            # Extract fractional c-coordinates
            c_coords = slab_base.frac_coords[:, -1]

            # Calculate possible shifts for atomic layer grouping
            shifts = self._calc_shifts(slab_base=slab_base)
            shifts += [1.0]  # To close the last layer

            # Define atomic layer bounds
            atomic_layer_bounds = [(shifts[i], shifts[i + 1]) for i in range(len(shifts) - 1)]

            # Initialize atomic layer array
            atomic_layers = -np.ones(len(c_coords))

            # Determine atomic layers based on the shifts
            for i, (bottom_bound, top_bound) in enumerate(atomic_layer_bounds):
                layer_mask = (c_coords > bottom_bound) & (c_coords < top_bound)
                atomic_layers[layer_mask] = i

            # Assign atomic layer indices as site properties
            slab_obs.add_site_property(
                "atomic_layer_index", np.round(atomic_layers).astype(int).tolist()
            )

            # Compute top and bottom c-coords and horizontal shift for centering
            top_c = c_coords.max()
            max_c_inds = np.where(np.isclose(top_c, c_coords))[0]

            dists = [
                slab_obs[i].distance_and_image_from_frac_coords(fcoords=[0.0, 0.0, 0.0])[0]
                for i in max_c_inds
            ]
            horiz_shift_ind = max_c_inds[np.argmin(dists)]
            horiz_shift = -slab_obs[horiz_shift_ind].frac_coords
            horiz_shift[-1] = 0  # Keep the c-direction shift at zero

            # Apply the horizontal shift for centering
            slab_obs.translate_sites(vector=horiz_shift, frac_coords=True)
            slab_obs.round(tol=6)

            # Create non-orthogonal slab
            non_orthogonal_slab = get_layer_supercell(
                structure=slab_obs._oriented_bulk_structure,
                layers=self.num_layers,
                vacuum_scale=vacuum_scale,
            )
            sort_slab(non_orthogonal_slab)

            # Center the slab in the vacuum region
            center_shift = 0.5 * (vacuum_scale / (vacuum_scale + self.num_layers))
            non_orthogonal_slab.translate_sites(
                indices=range(len(non_orthogonal_slab)),
                vector=[0, 0, center_shift],
                frac_coords=True,
                to_unit_cell=True,
            )

            non_orthogonal_slab.sort_index = 0
            shifted_slab_bases.append(slab_obs)
            non_orthogonal_slabs.append(non_orthogonal_slab)

        surfaces = []

        # Loop through slabs to ensure that they are all properly oriented and reduced
        # Return Surface objects
        for i in range(len(self._shifts)):
            # Create the Surface object
            surface = (
                MolecularSurface(
                    slab=non_orthogonal_slabs[i],  # KEEP
                    oriented_bulk=shifted_slab_bases[i],  # KEEP
                    miller_index=self.plane,  # KEEP
                    layers=self.num_layers,  # KEEP
                    vacuum=vacuum,  # KEEP
                    termination_index=i,  # KEEP
                )
                if self.molecular
                else Surface(
                    slab=non_orthogonal_slabs[i],  # KEEP
                    oriented_bulk=shifted_slab_bases[i],  # KEEP
                    miller_index=self.plane,  # KEEP
                    layers=self.num_layers,  # KEEP
                    vacuum=vacuum,  # KEEP
                    termination_index=i,  # KEEP
                )
            )
            surfaces.append(surface)

        return [
            shifted_slab_base.oriented_bulk_structure for shifted_slab_base in shifted_slab_bases
        ], surfaces

    def _undummify(self) -> List[Structure]:
        undummified = []
        for dummy in self._shifted_dummy_obses:
            undummy = (
                add_molecules(structure=dummy)
                if "molecules" in dummy.site_properties.keys()
                else dummy
            )
            undummified.append(undummy)

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
                    atom_count=self._atom_count,
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
            if smoothest:
                print(len(self._surfaces))
                print(len(self._terminations))
                self._smoothest_surface = self._surfaces[i]

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


def normalize(vector: np.ndarray) -> np.ndarray:
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


def wrapped_z(site: PeriodicSite, structure: Structure) -> float:
    """
    Wrap the z-coordinate of a site back into the unit cell via periodic boundary conditions.

    Parameters:
    site (PeriodicSite): A site in the pymatgen structure.
    structure (Structure): A pymatgen structure containing the site.

    Returns:
    float: Wrapped z-coordinate.
    """
    # Get fractional coordinates of the site
    frac_coords = site.frac_coords

    # Wrap the fractional coordinates
    wrapped_frac_coords = frac_coords % 1.0

    # Convert back to Cartesian coordinates
    wrapped_cartesian_coords = structure.lattice.get_cartesian_coords(wrapped_frac_coords)

    # Return the wrapped z-coordinate
    return wrapped_cartesian_coords[2]


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


def count_atoms(structure: Structure) -> int:
    return max([len(list(graph.nodes())) for graph in get_molecule_graphs(structure)])


def molecule_data(structure: Structure) -> Tuple[float, int]:
    mol_graphs = get_molecule_graphs(structure)
    atom_counts = []
    molecules = []
    for graph in mol_graphs:
        atom_indices = list(graph.nodes())
        atom_counts.append(len(atom_indices))
        species = [structure[i].specie for i in atom_indices]
        coords = [structure[i].coords for i in atom_indices]
        molecules.append(Molecule(species, coords))

    width = max(
        [
            squareform(pdist([site.coords for site in molecule.sites])).max()
            for molecule in molecules
        ]
    )

    atom_count = max(atom_counts)

    return width, atom_count


def molecular_width(structure: Structure) -> float:
    molecules = get_molecules_from_structure(structure)
    return max(
        [
            squareform(pdist([site.coords for site in molecule.sites])).max()
            for molecule in molecules
        ]
    )


def add_molecules(structure: Structure) -> Structure:
    mol_coords = []
    mol_atom_nums = []

    properties = list(structure.site_properties.keys())
    mols = structure.site_properties["molecules"]
    if "molecules" in properties:
        properties.remove("molecules")
    site_props = {p: [] for p in properties}
    site_props["molecule_index"] = []

    for i, site in enumerate(structure):
        site_mol = mols[i]
        mol_coords.append(site_mol.cart_coords + site.coords)
        mol_atom_nums.extend(site_mol.atomic_numbers)

        site_props["molecule_index"].extend([i] * len(site_mol))

        for p in properties:
            site_props[p].extend([site.properties[p]] * len(site_mol))

    mol_layer_struc = Structure(
        lattice=structure.lattice,
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

    # Get the species and coordinates of these atoms
    species = [structure[i].specie for i in atom_indices]
    coords = [structure[i].coords for i in atom_indices]

    # Create and return a pymatgen Molecule object
    return Molecule(species, coords)


def get_molecules_from_structure(structure: Structure) -> List[Molecule]:
    mol_graphs = get_molecule_graphs(structure)
    molecules = [subgraph_to_molecule(structure, graph) for graph in mol_graphs]
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
                        )
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

    molecule_width, atom_count = molecule_data(bulk)
    sys.stdout.write(f"\n\n\n{molecule_width=}\n\n\n")

    max_index = 2 if "TETCEN" in species else 1

    for plane in tqdm(
        get_unique_miller_indices(structure=bulk, max_index=max_index), desc="Terminating facets..."
    ):
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

        if debug:
            shifted_cells = terminator.terminated_structures(unrotate=False, dummy=False)
            dummy_cells = terminator.terminated_structures(unrotate=False, dummy=True)
            shifts = terminator.shifts

        smoothest_termination = terminator.smoothest_terminated_surface
        smoothest_termination.to(
            os.path.join(
                os.getcwd(), "smoothest_terminations", species, f"{miller_name(plane)}.cif"
            )
        )
        if args.visualize:
            terminator.visualize(save_dir=os.path.join(os.getcwd(), args.viz_out_dir))
