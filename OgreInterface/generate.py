"""
This module will be used to construct the surfaces and interfaces used in this package.
"""
from OgreInterface.surfaces import Surface, Interface
from OgreInterface import utils
from OgreInterface.lattice_match import ZurMcGill, OgreMatch

from pymatgen.core.structure import Structure, Molecule
from pymatgen.core.lattice import Lattice
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.operations import SymmOp
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import JmolNN
from pymatgen.analysis.molecule_matcher import (
    IsomorphismMolAtomMapper,
)
import networkx as nx
import numpy as np


from tqdm import tqdm
import numpy as np
import math
from copy import deepcopy
from typing import Union, List, TypeVar
from itertools import combinations, product, groupby
from ase import Atoms
from multiprocessing import Pool, cpu_count
import time
from collections.abc import Sequence

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

SelfSurfaceGenerator = TypeVar(
    "SelfSurfaceGenerator", bound="SurfaceGenerator"
)

SelfOrganicSurfaceGenerator = TypeVar(
    "SelfOrganicSurfaceGenerator", bound="OrganicSurfaceGenerator"
)

SelfInterfaceGenerator = TypeVar(
    "SelfInterfaceGenerator", bound="InterfaceGenerator"
)


class TolarenceError(RuntimeError):
    """Class to handle errors when no interfaces are found for a given tolarence setting."""

    pass


class SurfaceGenerator(Sequence):
    """Class for generating surfaces from a given bulk structure.

    The SurfaceGenerator classes generates surfaces with all possible terminations and contains
    information pertinent to generating interfaces with the InterfaceGenerator.

    Examples:
        Creating a SurfaceGenerator object using PyMatGen to load the structure:
        >>> from OgreInterface.generate import SurfaceGenerator
        >>> from pymatgen.core.structure import Structure
        >>> bulk = Structure.from_file("POSCAR_bulk")
        >>> surfaces = SurfaceGenerator(bulk=bulk, miller_index=[1, 1, 1], layers=5, vacuum=60)
        >>> surface = surfaces.slabs[0] # OgreInterface.Surface object

        Creating a SurfaceGenerator object using the build in from_file() method:
        >>> from OgreInterface.generate import SurfaceGenerator
        >>> surfaces = SurfaceGenerator.from_file(filename="POSCAR_bulk", miller_index=[1, 1, 1], layers=5, vacuum=60)
        >>> surface = surfaces.slabs[0] # OgreInterface.Surface object

    Args:
        bulk: Bulk crystal structure used to create the surface
        miller_index: Miller index of the surface
        layers: Number of layers to include in the surface
        vacuum: Size of the vacuum to include over the surface in Angstroms
        generate_all: Determines if all possible surface terminations are generated.
        filter_ionic_slab: Determines if the terminations of ionic crystals should be filtered out based on their
            predicted stability calculated using the IonicScoreFunction
        lazy: Determines if the surfaces are actually generated, or if only the surface basis vectors are found.
            (this is used for the MillerIndex search to make things faster)

    Attributes:
        slabs (list): List of OgreInterface Surface objects with different surface terminations.
        bulk_structure (Structure): Pymatgen Structure class for the conventional cell of the input bulk structure
        bulk_atoms (Atoms): ASE Atoms class for the conventional cell of the input bulk structure
        primitive_structure (Structure): Pymatgen Structure class for the primitive cell of the input bulk structure
        primitive_atoms (Atoms): ASE Atoms class for the primitive cell of the input bulk structure
        miller_index (list): Miller index of the surface
        layers (int): Number of layers to include in the surface
        vacuum (float): Size of the vacuum to include over the surface in Angstroms
        generate_all (bool): Determines if all possible surface terminations are generated.
        filter_ionic_slab (bool): Determines if the terminations of ionic crystals should be filtered out based on their
            predicted stability calculated using the IonicScoreFunction
        lazy (bool): Determines if the surfaces are actually generated, or if only the surface basis vectors are found.
            (this is used for the MillerIndex search to make things faster)
        oriented_bulk_structure (Structure): Pymatgen Structure class of the smallest building block of the slab,
            which will eventually be used to build the slab supercell
        oriented_bulk_atoms (Atoms): Pymatgen Atoms class of the smallest building block of the slab,
            which will eventually be used to build the slab supercell
        uvw_basis (list): The miller indices of the slab lattice vectors.
        transformation_matrix: Transformation matrix used to convert from the bulk basis to the slab basis
            (usefull for band unfolding calculations)
        inplane_vectors (list): The cartesian vectors of the in-plane lattice vectors.
        surface_normal (list): The normal vector of the surface
        c_projection (float): The projections of the c-lattice vector onto the surface normal
    """

    def __init__(
        self,
        bulk: Union[Structure, Atoms],
        miller_index: List[int],
        layers: int,
        vacuum: float,
        convert_to_conventional: bool = True,
        generate_all: bool = True,
        filter_ionic_slabs: bool = False,
        lazy: bool = False,
    ) -> None:
        super().__init__()
        self.convert_to_conventional = convert_to_conventional

        (
            self.bulk_structure,
            self.bulk_atoms,
            self.primitive_structure,
            self.primitive_atoms,
        ) = self._get_bulk(atoms_or_struc=bulk)

        self._use_prim = len(self.bulk_structure) != len(
            self.primitive_structure
        )

        self._point_group_operations = self._get_point_group_operations()

        self.miller_index = miller_index
        self.layers = layers
        self.vacuum = vacuum
        self.generate_all = generate_all
        self.filter_ionic_slabs = filter_ionic_slabs
        self.lazy = lazy
        (
            self.oriented_bulk_structure,
            self.oriented_bulk_atoms,
            self.uvw_basis,
            self.transformation_matrix,
            self.inplane_vectors,
            self.surface_normal,
            self.c_projection,
        ) = self._get_oriented_bulk_structure()

        if not self.lazy:
            self._slabs = self._generate_slabs()
        else:
            self._slabs = None

    def __getitem__(self, i):
        if self._slabs:
            return self._slabs[i]
        else:
            print(
                "The slabs have not been generated yet, please use the generate_slabs() function to create them."
            )

    def __len__(self):
        return len(self._slabs)

    def generate_slabs(self) -> None:
        """Used to generate list of Surface objects if lazy=True"""
        if self.lazy:
            self._slabs = self._generate_slabs()
        else:
            print(
                "The slabs are already generated upon initialization. This function is only needed if lazy=True"
            )

    @classmethod
    def from_file(
        cls,
        filename: str,
        miller_index: List[int],
        layers: int,
        vacuum: float,
        convert_to_conventional: bool = True,
        generate_all: bool = True,
        filter_ionic_slabs: bool = False,
        lazy: bool = False,
    ) -> SelfSurfaceGenerator:
        """Creating a SurfaceGenerator from a file (i.e. POSCAR, cif, etc)

        Args:
            filename: File path to the structure file
            miller_index: Miller index of the surface
            layers: Number of layers to include in the surface
            vacuum: Size of the vacuum to include over the surface in Angstroms
            generate_all: Determines if all possible surface terminations are generated
            filter_ionic_slab: Determines if the terminations of ionic crystals should be filtered out based on their
                predicted stability calculated using the IonicScoreFunction
            lazy: Determines if the surfaces are actually generated, or if only the surface basis vectors are found.
                (this is used for the MillerIndex search to make things faster)

        Returns:
            SurfaceGenerator
        """
        structure = Structure.from_file(filename=filename)

        return cls(
            structure,
            miller_index,
            layers,
            vacuum,
            convert_to_conventional,
            generate_all,
            filter_ionic_slabs,
            lazy,
        )

    def _get_bulk(self, atoms_or_struc):
        if type(atoms_or_struc) == Atoms:
            init_structure = AseAtomsAdaptor.get_structure(atoms_or_struc)
        elif type(atoms_or_struc) == Structure:
            init_structure = atoms_or_struc
        else:
            raise TypeError(
                f"structure accepts 'pymatgen.core.structure.Structure' or 'ase.Atoms' not '{type(atoms_or_struc).__name__}'"
            )

        if self.convert_to_conventional:
            sg = SpacegroupAnalyzer(init_structure)
            conventional_structure = sg.get_conventional_standard_structure()
            prim_structure = self._add_symmetry_info(
                conventional_structure, return_primitive=True
            )
            # prim_structure = sg.get_primitive_standard_structure()
            prim_atoms = AseAtomsAdaptor.get_atoms(prim_structure)
            conventional_atoms = AseAtomsAdaptor.get_atoms(
                conventional_structure
            )

            return (
                conventional_structure,
                conventional_atoms,
                prim_structure,
                prim_atoms,
            )
        else:
            if "molecule_index" in init_structure.site_properties:
                self._add_symmetry_info_molecule(init_structure)
            else:
                self._add_symmetry_info(init_structure, return_primitive=False)

            init_atoms = AseAtomsAdaptor().get_atoms(init_structure)

            prim_structure = init_structure.get_primitive_structure()
            prim_atoms = AseAtomsAdaptor().get_atoms(prim_structure)

            return init_structure, init_atoms, prim_structure, prim_atoms

    def _get_point_group_operations(self):
        sg = SpacegroupAnalyzer(self.bulk_structure)
        point_group_operations = sg.get_point_group_operations(cartesian=False)
        operation_array = np.round(
            np.array([p.rotation_matrix for p in point_group_operations])
        ).astype(np.int8)
        unique_operations = np.unique(operation_array, axis=0)

        return unique_operations

    def _add_symmetry_info(self, struc, return_primitive=False):
        sg = SpacegroupAnalyzer(struc)
        dataset = sg.get_symmetry_dataset()
        struc.add_site_property("bulk_wyckoff", dataset["wyckoffs"])
        struc.add_site_property(
            "bulk_equivalent",
            dataset["equivalent_atoms"].tolist(),
        )

        if return_primitive:
            prim_mapping = dataset["mapping_to_primitive"]
            _, prim_inds = np.unique(prim_mapping, return_index=True)
            prim_bulk = sg.get_primitive_standard_structure()

            prim_bulk.add_site_property(
                "bulk_wyckoff",
                [dataset["wyckoffs"][i] for i in prim_inds],
            )
            prim_bulk.add_site_property(
                "bulk_equivalent",
                dataset["equivalent_atoms"][prim_inds].tolist(),
            )

            return prim_bulk

    def _add_symmetry_info_molecule(self, struc):
        sg = SpacegroupAnalyzer(struc)
        dataset = sg.get_symmetry_dataset()
        wyckoffs = dataset["wyckoffs"]
        equivalent_atoms = dataset["equivalent_atoms"]
        molecule_index = struc.site_properties["molecule_index"]
        equivalent_molecules = [molecule_index[i] for i in equivalent_atoms]
        struc.add_site_property("bulk_wyckoff", wyckoffs)
        struc.add_site_property("bulk_equivalent", equivalent_molecules)

    def _get_oriented_bulk_structure(self):
        bulk = self.bulk_structure
        prim_bulk = self.primitive_structure

        lattice = bulk.lattice
        prim_lattice = prim_bulk.lattice

        recip_lattice = lattice.reciprocal_lattice_crystallographic

        miller_index = self.miller_index

        d_hkl = lattice.d_hkl(miller_index)

        normal_vector = lattice.get_cartesian_coords(
            np.array(miller_index).dot(recip_lattice.metric_tensor)
        )
        prim_normal_vector = prim_lattice.get_fractional_coords(normal_vector)
        prim_miller_index = prim_normal_vector.dot(prim_lattice.metric_tensor)
        prim_miller_index = prim_miller_index
        prim_miller_index = utils._get_reduced_vector(
            prim_miller_index
        ).astype(int)

        normal_vector /= np.linalg.norm(normal_vector)

        if not self._use_prim:
            intercepts = np.array(
                [1 / i if i != 0 else 0 for i in miller_index]
            )
            non_zero_points = np.where(intercepts != 0)[0]
            lattice_for_slab = lattice
            struc_for_slab = bulk
        else:
            intercepts = np.array(
                [1 / i if i != 0 else 0 for i in prim_miller_index]
            )
            non_zero_points = np.where(intercepts != 0)[0]
            d_hkl = lattice.d_hkl(miller_index)
            lattice_for_slab = prim_lattice
            struc_for_slab = prim_bulk

        if len(non_zero_points) == 1:
            basis = np.eye(3)
            basis[non_zero_points[0], non_zero_points[0]] *= intercepts[
                non_zero_points[0]
            ]
            dot_products = basis.dot(normal_vector)
            sort_inds = np.argsort(dot_products)
            basis = basis[sort_inds]

            if np.linalg.det(basis) < 0:
                basis = basis[[1, 0, 2]]

            basis = basis

        if len(non_zero_points) == 2:
            points = intercepts * np.eye(3)
            vec1 = points[non_zero_points[1]] - points[non_zero_points[0]]
            vec2 = np.eye(3)[intercepts == 0]

            basis = np.vstack([vec1, vec2])

        if len(non_zero_points) == 3:
            points = intercepts * np.eye(3)
            possible_vecs = []
            for center_inds in [[0, 1, 2], [1, 0, 2], [2, 0, 1]]:
                vec1 = (
                    points[non_zero_points[center_inds[1]]]
                    - points[non_zero_points[center_inds[0]]]
                )
                vec2 = (
                    points[non_zero_points[center_inds[2]]]
                    - points[non_zero_points[center_inds[0]]]
                )
                cart_vec1 = lattice_for_slab.get_cartesian_coords(vec1)
                cart_vec2 = lattice_for_slab.get_cartesian_coords(vec2)
                angle = np.arccos(
                    np.dot(cart_vec1, cart_vec2)
                    / (np.linalg.norm(cart_vec1) * np.linalg.norm(cart_vec2))
                )
                possible_vecs.append((vec1, vec2, angle))

            chosen_vec1, chosen_vec2, angle = min(
                possible_vecs, key=lambda x: abs(x[-1])
            )

            basis = np.vstack([chosen_vec1, chosen_vec2])

        basis = utils.get_reduced_basis(basis)

        if len(basis) == 2:
            max_normal_search = 2

            index_range = sorted(
                reversed(range(-max_normal_search, max_normal_search + 1)),
                key=lambda x: abs(x),
            )
            candidates = []
            for uvw in product(index_range, index_range, index_range):
                if (not any(uvw)) or abs(
                    np.linalg.det(np.vstack([basis, uvw]))
                ) < 1e-8:
                    continue

                vec = lattice_for_slab.get_cartesian_coords(uvw)
                proj = np.abs(np.dot(vec, normal_vector) - d_hkl)
                vec_length = np.linalg.norm(vec)
                cosine = np.dot(vec / vec_length, normal_vector)
                candidates.append(
                    (
                        uvw,
                        np.round(cosine, 5),
                        np.round(vec_length, 5),
                        np.round(proj, 5),
                    )
                )
                if abs(abs(cosine) - 1) < 1e-8:
                    # If cosine of 1 is found, no need to search further.
                    break
            # We want the indices with the maximum absolute cosine,
            # but smallest possible length.
            uvw, cosine, l, diff = max(
                candidates,
                key=lambda x: (-x[3], x[1], -x[2]),
            )
            basis = np.vstack([basis, uvw])

        init_oriented_struc = struc_for_slab.copy()
        init_oriented_struc.make_supercell(basis)

        cart_basis = init_oriented_struc.lattice.matrix

        if np.linalg.det(cart_basis) < 0:
            ab_switch = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
            init_oriented_struc.make_supercell(ab_switch)
            basis = ab_switch.dot(basis)
            cart_basis = init_oriented_struc.lattice.matrix

        cross_ab = np.cross(cart_basis[0], cart_basis[1])
        cross_ab /= np.linalg.norm(cross_ab)
        cross_ac = np.cross(cart_basis[0], cross_ab)
        cross_ac /= np.linalg.norm(cross_ac)

        ortho_basis = np.vstack(
            [
                cart_basis[0] / np.linalg.norm(cart_basis[0]),
                cross_ac,
                cross_ab,
            ]
        )

        to_planar_operation = SymmOp.from_rotation_and_translation(
            ortho_basis, translation_vec=np.zeros(3)
        )

        # if "molecules" in init_oriented_struc.site_properties:
        #     for site in init_oriented_struc:
        #         mol = site.properties["molecules"]
        #         planar_mol = mol.copy()
        #         planar_mol.translate_sites(range(len(mol)), site.coords)
        #         planar_mol.apply_operation(to_planar_operation)
        #         centered_mol = planar_mol.get_centered_molecule()
        #         site.properties["molecules"] = centered_mol

        planar_oriented_struc = init_oriented_struc.copy()
        planar_oriented_struc.apply_operation(to_planar_operation)

        planar_matrix = deepcopy(planar_oriented_struc.lattice.matrix)

        new_a, new_b, mat = utils.reduce_vectors_zur_and_mcgill(
            planar_matrix[0, :2], planar_matrix[1, :2]
        )

        planar_oriented_struc.make_supercell(mat)

        a_norm = (
            planar_oriented_struc.lattice.matrix[0]
            / planar_oriented_struc.lattice.a
        )
        a_to_i = np.array(
            [[a_norm[0], -a_norm[1], 0], [a_norm[1], a_norm[0], 0], [0, 0, 1]]
        )

        a_to_i_operation = SymmOp.from_rotation_and_translation(
            a_to_i.T, translation_vec=np.zeros(3)
        )

        # if "molecules" in planar_oriented_struc.site_properties:
        #     for site in planar_oriented_struc:
        #         mol = site.properties["molecules"]
        #         a_to_i_mol = mol.copy()
        #         a_to_i_mol.translate_sites(range(len(mol)), site.coords)
        #         a_to_i_mol.apply_operation(a_to_i_operation)
        #         centered_mol = a_to_i_mol.get_centered_molecule()
        #         site.properties["molecules"] = centered_mol

        planar_oriented_struc.apply_operation(a_to_i_operation)

        if "molecule_index" not in planar_oriented_struc.site_properties:
            planar_oriented_struc.add_site_property(
                "oriented_bulk_equivalent",
                list(range(len(planar_oriented_struc))),
            )

        planar_oriented_struc.sort()

        planar_oriented_atoms = AseAtomsAdaptor().get_atoms(
            planar_oriented_struc
        )

        final_matrix = deepcopy(planar_oriented_struc.lattice.matrix)

        final_basis = mat.dot(basis)
        final_basis = utils.get_reduced_basis(final_basis).astype(int)

        transformation_matrix = np.copy(final_basis)

        if self._use_prim:
            for i, b in enumerate(final_basis):
                cart_coords = prim_lattice.get_cartesian_coords(b)
                conv_frac_coords = lattice.get_fractional_coords(cart_coords)
                conv_frac_coords = utils._get_reduced_vector(conv_frac_coords)
                final_basis[i] = conv_frac_coords

        inplane_vectors = final_matrix[:2]

        norm = np.cross(final_matrix[0], final_matrix[1])
        norm /= np.linalg.norm(norm)

        if np.dot(norm, final_matrix[-1]) < 0:
            norm *= -1

        norm_proj = np.dot(norm, final_matrix[-1])

        return (
            planar_oriented_struc,
            planar_oriented_atoms,
            final_basis,
            transformation_matrix,
            inplane_vectors,
            norm,
            norm_proj,
        )

    def _calculate_possible_shifts(self, tol: float = 0.1):
        frac_coords = self.oriented_bulk_structure.frac_coords
        n = len(frac_coords)

        if n == 1:
            # Clustering does not work when there is only one data point.
            shift = frac_coords[0][2] + 0.5
            return [shift - math.floor(shift)]

        # We cluster the sites according to the c coordinates. But we need to
        # take into account PBC. Let's compute a fractional c-coordinate
        # distance matrix that accounts for PBC.
        dist_matrix = np.zeros((n, n))
        # h = self.oriented_bulk_structure.lattice.matrix[-1, -1]
        h = self.c_projection
        # Projection of c lattice vector in
        # direction of surface normal.
        for i, j in combinations(list(range(n)), 2):
            if i != j:
                cdist = frac_coords[i][2] - frac_coords[j][2]
                cdist = abs(cdist - round(cdist)) * h
                dist_matrix[i, j] = cdist
                dist_matrix[j, i] = cdist

        condensed_m = squareform(dist_matrix)
        z = linkage(condensed_m)
        clusters = fcluster(z, tol, criterion="distance")

        # Generate dict of cluster# to c val - doesn't matter what the c is.
        c_loc = {c: frac_coords[i][2] for i, c in enumerate(clusters)}

        # Put all c into the unit cell.
        possible_c = [c - math.floor(c) for c in sorted(c_loc.values())]

        # Calculate the shifts
        nshifts = len(possible_c)
        shifts = []
        for i in range(nshifts):
            if i == nshifts - 1:
                # There is an additional shift between the first and last c
                # coordinate. But this needs special handling because of PBC.
                shift = (possible_c[0] + 1 + possible_c[i]) * 0.5
                if shift > 1:
                    shift -= 1
            else:
                shift = (possible_c[i] + possible_c[i + 1]) * 0.5
            shifts.append(shift - math.floor(shift))

        shifts = sorted(shifts)

        return shifts

    def _get_slab(self, shift=0, tol: float = 0.1, energy=None):
        """
        This method takes in shift value for the c lattice direction and
        generates a slab based on the given shift. You should rarely use this
        method. Instead, it is used by other generation algorithms to obtain
        all slabs.

        Args:
            shift (float): A shift value in Angstrom that determines how much a
                slab should be shifted.
            tol (float): Tolerance to determine primitive cell.
            energy (float): An energy to assign to the slab.

        Returns:
            (Slab) A Slab object with a particular shifted oriented unit cell.
        """
        init_matrix = deepcopy(self.oriented_bulk_structure.lattice.matrix)
        slab_base = self.oriented_bulk_structure.copy()
        slab_base.translate_sites(
            indices=range(len(slab_base)),
            vector=[0, 0, -shift],
            frac_coords=True,
            to_unit_cell=True,
        )

        z_coords = slab_base.frac_coords[:, -1]
        bot_z = z_coords.min()
        top_z = z_coords.max()

        max_z_inds = np.where(np.isclose(top_z, z_coords))[0]

        dists = []
        for i in max_z_inds:
            dist, image = slab_base[i].distance_and_image_from_frac_coords(
                fcoords=[0.0, 0.0, 0.0]
            )
            dists.append(dist)

        horiz_shift_ind = max_z_inds[np.argmin(dists)]
        horiz_shift = -slab_base[horiz_shift_ind].frac_coords
        horiz_shift[-1] = 0
        slab_base.translate_sites(
            indices=range(len(slab_base)),
            vector=horiz_shift,
            frac_coords=True,
            to_unit_cell=True,
        )

        bottom_layer_dist = np.abs(bot_z - (top_z - 1)) * init_matrix[-1, -1]
        top_layer_dist = np.abs((bot_z + 1) - top_z) * init_matrix[-1, -1]

        vacuum_scale = self.vacuum // self.c_projection

        if vacuum_scale % 2:
            vacuum_scale += 1

        if vacuum_scale == 0:
            vacuum_scale = 1

        non_orthogonal_slab = utils.get_layer_supercelll(
            structure=slab_base, layers=self.layers, vacuum_scale=vacuum_scale
        )
        non_orthogonal_slab.sort()

        a, b, c = non_orthogonal_slab.lattice.matrix
        new_c = np.dot(c, self.surface_normal) * self.surface_normal
        vacuum = self.oriented_bulk_structure.lattice.c * vacuum_scale

        orthogonal_matrix = np.vstack([a, b, new_c])
        orthogonal_slab = Structure(
            lattice=Lattice(matrix=orthogonal_matrix),
            species=non_orthogonal_slab.species,
            coords=non_orthogonal_slab.cart_coords,
            coords_are_cartesian=True,
            to_unit_cell=True,
            site_properties=non_orthogonal_slab.site_properties,
        )
        orthogonal_slab.sort()

        shift = 0.5 * (vacuum_scale / (vacuum_scale + self.layers))
        non_orthogonal_slab.translate_sites(
            indices=range(len(non_orthogonal_slab)),
            vector=[0, 0, shift],
            frac_coords=True,
            to_unit_cell=True,
        )
        orthogonal_slab.translate_sites(
            indices=range(len(orthogonal_slab)),
            vector=[0, 0, shift],
            frac_coords=True,
            to_unit_cell=True,
        )

        if "molecules" in slab_base.site_properties:
            slab_base = self._add_molecules(slab_base)
            orthogonal_slab = self._add_molecules(orthogonal_slab)
            non_orthogonal_slab = self._add_molecules(non_orthogonal_slab)

        return (
            slab_base,
            orthogonal_slab,
            non_orthogonal_slab,
            bottom_layer_dist,
            top_layer_dist,
            vacuum,
        )

    def _add_molecules(self, struc):
        mol_coords = []
        mol_atom_nums = []

        properties = list(struc.site_properties.keys())
        properties.remove("molecules")
        # properties.remove("basis")
        site_props = {p: [] for p in properties}
        site_props["molecule_index"] = []

        for i, site in enumerate(struc):
            site_mol = site.properties["molecules"]
            mol_coords.append(site_mol.cart_coords + site.coords)
            mol_atom_nums.extend(site_mol.atomic_numbers)

            site_props["molecule_index"].extend([i] * len(site_mol))

            for p in properties:
                site_props[p].extend([site.properties[p]] * len(site_mol))

        mol_layer_struc = Structure(
            lattice=struc.lattice,
            species=mol_atom_nums,
            coords=np.vstack(mol_coords),
            to_unit_cell=True,
            coords_are_cartesian=True,
            site_properties=site_props,
        )
        mol_layer_struc.sort()

        return mol_layer_struc

    def _generate_slabs(self):
        """
        This function is used to generate slab structures with all unique
        surface terminations.

        Returns:
            A list of Surface classes
        """
        # Determine if all possible terminations are generated
        possible_shifts = self._calculate_possible_shifts()
        shifted_slab_bases = []
        orthogonal_slabs = []
        non_orthogonal_slabs = []
        bottom_layer_dists = []
        top_layer_dists = []
        if not self.generate_all:
            (
                shifted_slab_base,
                orthogonal_slab,
                non_orthogonal_slab,
                bottom_layer_dist,
                top_layer_dist,
                actual_vacuum,
            ) = self._get_slab(shift=possible_shifts[0])
            orthogonal_slab.sort_index = 0
            non_orthogonal_slab.sort_index = 0
            shifted_slab_bases.append(shifted_slab_base)
            orthogonal_slabs.append(orthogonal_slab)
            non_orthogonal_slabs.append(non_orthogonal_slab)
            bottom_layer_dists.append(bottom_layer_dist)
            top_layer_dists.append(top_layer_dist)
        else:
            for i, possible_shift in enumerate(possible_shifts):
                (
                    shifted_slab_base,
                    orthogonal_slab,
                    non_orthogonal_slab,
                    bottom_layer_dist,
                    top_layer_dist,
                    actual_vacuum,
                ) = self._get_slab(shift=possible_shift)
                orthogonal_slab.sort_index = i
                non_orthogonal_slab.sort_index = i
                shifted_slab_bases.append(shifted_slab_base)
                orthogonal_slabs.append(orthogonal_slab)
                non_orthogonal_slabs.append(non_orthogonal_slab)
                bottom_layer_dists.append(bottom_layer_dist)
                top_layer_dists.append(top_layer_dist)

        surfaces = []

        if self._use_prim:
            base_structure = self.primitive_structure
        else:
            base_structure = self.bulk_structure

        # Loop through slabs to ensure that they are all properly oriented and reduced
        # Return Surface objects
        for i, slab in enumerate(orthogonal_slabs):
            # Create the Surface object
            surface = Surface(
                orthogonal_slab=slab,
                non_orthogonal_slab=non_orthogonal_slabs[i],
                oriented_bulk=shifted_slab_bases[i],
                bulk=base_structure,
                transformation_matrix=self.transformation_matrix,
                miller_index=self.miller_index,
                layers=self.layers,
                vacuum=actual_vacuum,
                uvw_basis=self.uvw_basis,
                point_group_operations=self._point_group_operations,
                bottom_layer_dist=bottom_layer_dists[i],
                top_layer_dist=top_layer_dists[i],
                termination_index=i,
                surface_normal=self.surface_normal,
                c_projection=self.c_projection,
            )
            surfaces.append(surface)

        return surfaces

    def __len__(self):
        return len(self._slabs)

    @property
    def nslabs(self):
        """
        Return the number of slabs generated by the SurfaceGenerator
        """
        return self.__len__()

    @property
    def terminations(self):
        """
        Return the terminations of each slab generated by the SurfaceGenerator
        """
        return {
            i: slab.get_termination() for i, slab in enumerate(self._slabs)
        }


class OrganicSurfaceGenerator(SurfaceGenerator):
    """Class for generating surfaces from a given bulk structure.

    The SurfaceGenerator classes generates surfaces with all possible terminations and contains
    information pertinent to generating interfaces with the InterfaceGenerator.

    Examples:
        Creating a SurfaceGenerator object using PyMatGen to load the structure:
        >>> from OgreInterface.generate import SurfaceGenerator
        >>> from pymatgen.core.structure import Structure
        >>> bulk = Structure.from_file("POSCAR_bulk")
        >>> surfaces = SurfaceGenerator(bulk=bulk, miller_index=[1, 1, 1], layers=5, vacuum=60)
        >>> surface = surfaces.slabs[0] # OgreInterface.Surface object

        Creating a SurfaceGenerator object using the build in from_file() method:
        >>> from OgreInterface.generate import SurfaceGenerator
        >>> surfaces = SurfaceGenerator.from_file(filename="POSCAR_bulk", miller_index=[1, 1, 1], layers=5, vacuum=60)
        >>> surface = surfaces.slabs[0] # OgreInterface.Surface object

    Args:
        bulk: Bulk crystal structure used to create the surface
        miller_index: Miller index of the surface
        layers: Number of layers to include in the surface
        vacuum: Size of the vacuum to include over the surface in Angstroms
        generate_all: Determines if all possible surface terminations are generated.
        lazy: Determines if the surfaces are actually generated, or if only the surface basis vectors are found.
            (this is used for the MillerIndex search to make things faster)

    Attributes:
        slabs (list): List of OgreInterface Surface objects with different surface terminations.
        bulk_structure (Structure): Pymatgen Structure class for the conventional cell of the input bulk structure
        bulk_atoms (Atoms): ASE Atoms class for the conventional cell of the input bulk structure
        primitive_structure (Structure): Pymatgen Structure class for the primitive cell of the input bulk structure
        primitive_atoms (Atoms): ASE Atoms class for the primitive cell of the input bulk structure
        miller_index (list): Miller index of the surface
        layers (int): Number of layers to include in the surface
        vacuum (float): Size of the vacuum to include over the surface in Angstroms
        generate_all (bool): Determines if all possible surface terminations are generated.
        filter_ionic_slab (bool): Determines if the terminations of ionic crystals should be filtered out based on their
            predicted stability calculated using the IonicScoreFunction
        lazy (bool): Determines if the surfaces are actually generated, or if only the surface basis vectors are found.
            (this is used for the MillerIndex search to make things faster)
        oriented_bulk_structure (Structure): Pymatgen Structure class of the smallest building block of the slab,
            which will eventually be used to build the slab supercell
        oriented_bulk_atoms (Atoms): Pymatgen Atoms class of the smallest building block of the slab,
            which will eventually be used to build the slab supercell
        uvw_basis (list): The miller indices of the slab lattice vectors.
        transformation_matrix: Transformation matrix used to convert from the bulk basis to the slab basis
            (usefull for band unfolding calculations)
        inplane_vectors (list): The cartesian vectors of the in-plane lattice vectors.
        surface_normal (list): The normal vector of the surface
        c_projection (float): The projections of the c-lattice vector onto the surface normal
    """

    def __init__(
        self,
        bulk: Union[Structure, Atoms],
        miller_index: List[int],
        layers: int,
        vacuum: float,
        generate_all: bool = True,
        lazy: bool = False,
    ) -> None:
        dummy_bulk = self._get_dummy_bulk(bulk)
        labeled_bulk = self._add_molecules(dummy_bulk)
        super().__init__(
            bulk=labeled_bulk,
            miller_index=miller_index,
            layers=layers,
            vacuum=vacuum,
            convert_to_conventional=False,
            generate_all=generate_all,
            filter_ionic_slabs=False,
            lazy=True,
        )

        obs = self.oriented_bulk_structure
        dummy_obs = self._get_dummy_bulk(obs)
        dummy_obs.add_site_property(
            "oriented_bulk_equivalent", range(len(dummy_obs))
        )
        self.oriented_bulk_structure = dummy_obs

        if not lazy:
            self.generate_slabs()

    def _get_dummy_bulk(self, s) -> Structure:
        # Create a structure graph so we can extract the molecules
        struc_graph = StructureGraph.with_local_env_strategy(s, JmolNN())

        # Extract a list of the unique molecules on the structure
        mols = struc_graph.get_subgraphs_as_molecules()

        # Get all the molecults from the initial structure
        g = nx.Graph(struc_graph.graph)
        sub_graphs = [g.subgraph(c) for c in nx.connected_components(g)]

        # Assign a molecule index to each atom in each molecule
        mol_index = np.zeros(len(s)).astype(int)
        for i, sub_g in enumerate(sub_graphs):
            for site in sub_g.nodes:
                mol_index[site] = i

        # Add the molecule index as a site propery for each atom
        s.add_site_property("molecule_index", mol_index.tolist())

        # Find the center of masses of all the molecules in the unit cell
        # We can do this similar to how the get_subgraphs_as_molecules()
        # function works by creating a 3x3 supercell and only keeping the
        # molecules that don't intersect the boundary of the unit cell
        supercell = s.copy()
        supercell_sg = StructureGraph.with_local_env_strategy(
            supercell,
            JmolNN(),
        )

        # Create supercell of the graph
        supercell_sg *= (3, 3, 3)
        supercell_g = nx.Graph(supercell_sg.graph)

        # Extract all molecule subgraphs
        all_subgraphs = [
            supercell_g.subgraph(c)
            for c in nx.connected_components(supercell_g)
        ]

        # Only keep that molecules that are completely contained in the 3x3 supercell
        molecule_subgraphs = []
        for subgraph in all_subgraphs:
            intersects_boundary = any(
                d["to_jimage"] != (0, 0, 0)
                for u, v, d in subgraph.edges(data=True)
            )
            if not intersects_boundary:
                molecule_subgraphs.append(nx.MultiDiGraph(subgraph))

        # Get the center of mass and the molecule index
        center_of_masses = []
        site_props = list(s.site_properties.keys())
        site_props.remove("molecule_index")
        props = {p: [] for p in site_props}
        # bulk_equivalents = []
        # bulk_wyckoffs = []
        for subgraph in molecule_subgraphs:
            cart_coords = np.vstack(
                [supercell_sg.structure[n].coords for n in subgraph]
            )
            weights = np.array(
                [supercell_sg.structure[n].species.weight for n in subgraph]
            )
            mol_ind = [
                supercell_sg.structure[n].properties["molecule_index"]
                for n in subgraph
            ]

            for p in props:
                ind = list(subgraph.nodes.keys())[0]
                props[p].append(supercell_sg.structure[ind].properties[p])

            center_of_mass = (
                np.sum(cart_coords * weights[:, None], axis=0) / weights.sum()
            )
            center_of_masses.append(np.round(center_of_mass, 6))

        center_of_masses = np.vstack(center_of_masses)

        # Now we can find which center of masses are contained in the original unit cell
        # First we can shift the center of masses by the [1, 1, 1] vector of the original unit cell
        # so the center unit cell of the 3x3 supercell is positioned at (0, 0, 0)
        shift = s.lattice.get_cartesian_coords([1, 1, 1])
        inv_matrix = s.lattice.inv_matrix

        # Shift the center of masses
        center_of_masses -= shift

        # Convert to fractional coordinates in the basis of the original unit cell
        frac_com = center_of_masses.dot(inv_matrix)

        # The center of masses in the unit cell should have fractional coordinates between [0, 1)
        in_original_cell = np.logical_and(
            0 <= np.round(frac_com, 6), np.round(frac_com, 6) < 1
        ).all(axis=1)

        # Extract the fractional coordinates in the original cell
        frac_coords_in_cell = frac_com[in_original_cell]
        props_in_cell = {
            p: [l[i] for i in in_original_cell] for p, l in props.items()
        }

        # Extract the molecules who's center of mass is in the original cell
        molecules = []
        for i in np.where(in_original_cell)[0]:
            m_graph = molecule_subgraphs[i]
            coords = [
                supercell_sg.structure[n].coords for n in m_graph.nodes()
            ]
            species = [
                supercell_sg.structure[n].specie for n in m_graph.nodes()
            ]
            molecule = Molecule(species, coords)
            molecule = molecule.get_centered_molecule()
            molecules.append(molecule)

        # To identify the unique orientations of the molecules an orthogonal basis can be created
        # by using 3 equivalent atomic positions in each molecule.
        mol_basis = []
        for mol in molecules:
            # Find equivalent indices between the two molecules in reference to the first molecule in the list
            mol_match = IsomorphismMolAtomMapper()
            l1, l2 = mol_match.uniform_labels(molecules[0], mol)

            # Extract the first three atom indices to create the basis
            a, b, c = l2[:3]

            # Using atom-b as the center atom find the vector from b-a and b-c
            ba_vec = mol.cart_coords[a] - mol.cart_coords[b]
            ba_vec /= np.linalg.norm(ba_vec)
            bc_vec = mol.cart_coords[c] - mol.cart_coords[b]
            bc_vec /= np.linalg.norm(bc_vec)

            # Get an orthogonal vector to the b-a and b-c by taking the cross product
            cross1 = np.cross(ba_vec, bc_vec)
            cross1 /= np.linalg.norm(cross1)

            # Get and orthogonal vector from b-a and the previous cross product to create and orthogonal basis
            cross2 = np.cross(ba_vec, cross1)
            cross2 /= np.linalg.norm(cross2)

            # Create the basis matrix
            basis = np.vstack([ba_vec, cross1, cross2])
            mol_basis.append(np.round(basis, 5))

        # Find the unique basis sets
        comp_basis = np.vstack([m.ravel() for m in mol_basis])
        unique_basis, unique_inds, inv_inds = np.unique(
            comp_basis, axis=0, return_index=True, return_inverse=True
        )

        # # Get the neccesary data to create the center-of-mass structure with dummy atoms at the center of mass
        # struc_data = []
        # struc_props = {p: [] for p in props_in_cell}
        # for i in inv_inds:
        #     mol_ind = unique_inds[i]
        #     mol = molecules[mol_ind]
        #     basis = unique_basis[i].reshape(3, 3)
        #     com = frac_coords_in_cell[mol_ind]
        #     struc_data.append((mol_ind + 3, com, basis, mol))

        # # Create the structure with the center of mass
        # species, frac_coords, bases, mols = list(zip(*struc_data))
        species = [i + 3 for i in range(len(molecules))]
        frac_coords = frac_coords_in_cell
        struc_props = {
            # "basis": [c.reshape(3, 3) for c in comp_basis],
            "molecules": molecules,
        }
        struc_props.update(props_in_cell)

        dummy_struc = Structure(
            lattice=s.lattice,
            coords=frac_coords,
            species=species,
            site_properties=struc_props,
        )

        print(dummy_struc)

        return dummy_struc


class InterfaceGenerator:
    """Class for generating interfaces from two bulk structures

    This class will use the lattice matching algorithm from Zur and McGill to generate
    commensurate interface structures between two inorganic crystalline materials.

    Examples:
        >>> from OgreInterface.generate import SurfaceGenerator, InterfaceGenerator
        >>> subs = SurfaceGenerator.from_file(filename="POSCAR_sub", miller_index=[1,1,1], layers=5)
        >>> films = SurfaceGenerator.from_file(filename="POSCAR_film", miller_index=[1,1,1], layers=5)
        >>> interface_generator = InterfaceGenerator(substrate=subs.slabs[0], film=films.slabs[0])
        >>> interfaces = interface_generator.generate_interfaces() # List of OgreInterface Interface objects

    Args:
        substrate: Surface class of the substrate material
        film: Surface class of the film materials
        max_area_mismatch: Tolarance of the area mismatch (eq. 2.1 in Zur and McGill)
        max_angle_strain: Tolarence of the angle mismatch between the film and substrate lattice vectors
        max_linear_strain: Tolarence of the length mismatch between the film and substrate lattice vectors
        max_area: Maximum area of the interface unit cell cross section
        interfacial_distance: Distance between the top atom in the substrate to the bottom atom of the film
            If None, the interfacial distance will be predicted based on the average distance of the interlayer
            spacing between the film and substrate materials.
        vacuum: Size of the vacuum in Angstroms
        center: Determines of the interface should be centered in the vacuum

    Attributes:
        substrate (Surface): Surface class of the substrate material
        film (Surface): Surface class of the film materials
        max_area_mismatch (float): Tolarance of the area mismatch (eq. 2.1 in Zur and McGill)
        max_angle_strain (float): Tolarence of the angle mismatch between the film and substrate lattice vectors
        max_linear_strain (float): Tolarence of the length mismatch between the film and substrate lattice vectors
        max_area (float): Maximum area of the interface unit cell cross section
        interfacial_distance (Union[float, None]): Distance between the top atom in the substrate to the bottom atom of the film
            If None, the interfacial distance will be predicted based on the average distance of the interlayer
            spacing between the film and substrate materials.
        vacuum (float): Size of the vacuum in Angstroms
        center: Determines of the interface should be centered in the vacuum
        match_list (List[OgreMatch]): List of OgreMatch objects for each interface generated
    """

    def __init__(
        self,
        substrate: Surface,
        film: Surface,
        max_area_mismatch: float = 0.01,
        max_angle_strain: float = 0.01,
        max_linear_strain: float = 0.01,
        max_area: float = 500.0,
        interfacial_distance: Union[float, None] = 2.0,
        vacuum: float = 40.0,
        center: bool = False,
    ):
        if type(substrate) == Surface:
            self.substrate = substrate
        else:
            raise TypeError(
                f"InterfaceGenerator accepts 'ogre.core.Surface' not '{type(substrate).__name__}'"
            )

        if type(film) == Surface:
            self.film = film
        else:
            raise TypeError(
                f"InterfaceGenerator accepts 'ogre.core.Surface' not '{type(film).__name__}'"
            )

        self.center = center
        self.max_area_mismatch = max_area_mismatch
        self.max_angle_strain = max_angle_strain
        self.max_linear_strain = max_linear_strain
        self.max_area = max_area
        self.interfacial_distance = interfacial_distance
        self.vacuum = vacuum
        self.match_list = self._generate_interface_props()

    def _generate_interface_props(self):
        zm = ZurMcGill(
            film_vectors=self.film.inplane_vectors,
            substrate_vectors=self.substrate.inplane_vectors,
            film_basis=self.film.uvw_basis,
            substrate_basis=self.substrate.uvw_basis,
            max_area=self.max_area,
            max_linear_strain=self.max_linear_strain,
            max_angle_strain=self.max_angle_strain,
            max_area_mismatch=self.max_area_mismatch,
        )
        match_list = zm.run(return_all=True)

        if len(match_list) == 0:
            raise TolarenceError(
                "No interfaces were found, please increase the tolarences."
            )
        elif len(match_list) == 1:
            return match_list
        else:
            film_basis_vectors = []
            sub_basis_vectors = []
            film_scale_factors = []
            sub_scale_factors = []
            for i, match in enumerate(match_list):
                film_basis_vectors.append(match.film_sl_basis)
                sub_basis_vectors.append(match.substrate_sl_basis)
                film_scale_factors.append(match.film_sl_scale_factors)
                sub_scale_factors.append(match.substrate_sl_scale_factors)

            film_basis_vectors = np.round(
                np.vstack(film_basis_vectors)
            ).astype(np.int8)
            sub_basis_vectors = np.round(np.vstack(sub_basis_vectors)).astype(
                np.int8
            )
            film_scale_factors = np.round(
                np.concatenate(film_scale_factors)
            ).astype(np.int8)
            sub_scale_factors = np.round(
                np.concatenate(sub_scale_factors)
            ).astype(np.int8)

            film_map = self._get_miller_index_map(
                self.film.point_group_operations, film_basis_vectors
            )
            sub_map = self._get_miller_index_map(
                self.substrate.point_group_operations, sub_basis_vectors
            )

            split_film_basis_vectors = np.vsplit(
                film_basis_vectors, len(match_list)
            )
            split_sub_basis_vectors = np.vsplit(
                sub_basis_vectors, len(match_list)
            )
            split_film_scale_factors = np.split(
                film_scale_factors, len(match_list)
            )
            split_sub_scale_factors = np.split(
                sub_scale_factors, len(match_list)
            )

            sort_vecs = []

            for i in range(len(split_film_basis_vectors)):
                fb = split_film_basis_vectors[i]
                sb = split_sub_basis_vectors[i]
                fs = split_film_scale_factors[i]
                ss = split_sub_scale_factors[i]
                sort_vec = np.concatenate(
                    [
                        [ss[0]],
                        sub_map[tuple(sb[0])],
                        [ss[1]],
                        sub_map[tuple(sb[1])],
                        [fs[0]],
                        film_map[tuple(fb[0])],
                        [fs[1]],
                        film_map[tuple(fb[1])],
                    ]
                )
                sort_vecs.append(sort_vec)

            sort_vecs = np.vstack(sort_vecs)
            unique_sort_vecs, unique_sort_inds = np.unique(
                sort_vecs, axis=0, return_index=True
            )
            unique_matches = [match_list[i] for i in unique_sort_inds]

            sorted_matches = sorted(
                unique_matches,
                key=lambda x: (
                    x.area,
                    np.max(np.abs(x.linear_strain)),
                    np.abs(x.angle_strain),
                ),
            )

            return sorted_matches

    def _get_miller_index_map(self, operations, miller_indices):
        miller_indices = np.unique(miller_indices, axis=0)
        not_used = np.ones(miller_indices.shape[0]).astype(bool)
        op = np.einsum("...ij,jk", operations, miller_indices.T)
        op = op.transpose(2, 0, 1)
        unique_vecs = {}

        for i, vec in enumerate(miller_indices):
            if not_used[i]:
                same_inds = (op == vec).all(axis=2).sum(axis=1) > 0

                if not_used[same_inds].all():
                    same_vecs = miller_indices[same_inds]
                    optimal_vec = self._get_optimal_miller_index(same_vecs)
                    unique_vecs[tuple(optimal_vec)] = list(
                        map(tuple, same_vecs)
                    )
                    not_used[same_inds] = False

        mapping = {}
        for key, value in unique_vecs.items():
            for v in value:
                mapping[v] = key

        return mapping

    def _get_optimal_miller_index(self, vecs):
        diff = np.abs(np.sum(np.sign(vecs), axis=1))
        like_signs = vecs[diff == np.max(diff)]
        if len(like_signs) == 1:
            return like_signs[0]
        else:
            first_max = like_signs[
                np.abs(like_signs)[:, 0] == np.max(np.abs(like_signs)[:, 0])
            ]
            if len(first_max) == 1:
                return first_max[0]
            else:
                second_max = first_max[
                    np.abs(first_max)[:, 1] == np.max(np.abs(first_max)[:, 1])
                ]
                if len(second_max) == 1:
                    return second_max[0]
                else:
                    return second_max[
                        np.argmax(np.sign(second_max).sum(axis=1))
                    ]

    def _build_interface(self, match):
        if self.interfacial_distance is None:
            i_dist = (
                self.substrate.top_layer_dist + self.film.bottom_layer_dist
            ) / 2
        else:
            i_dist = self.interfacial_distance

        interface = Interface(
            substrate=self.substrate,
            film=self.film,
            interfacial_distance=i_dist,
            match=match,
            vacuum=self.vacuum,
            center=self.center,
        )
        return interface

    def generate_interfaces(self):
        """Generates a list of Interface objects from that matches found using the Zur and McGill lattice matching algorithm"""
        if self.interfacial_distance is None:
            i_dist = (
                self.substrate.top_layer_dist + self.film.bottom_layer_dist
            ) / 2
        else:
            i_dist = self.interfacial_distance

        interfaces = []

        print("Generating Interfaces:")
        for match in tqdm(self.match_list, dynamic_ncols=True):
            interface = Interface(
                substrate=self.substrate,
                film=self.film,
                interfacial_distance=i_dist,
                match=match,
                vacuum=self.vacuum,
                center=self.center,
            )
            interfaces.append(interface)

        return interfaces
