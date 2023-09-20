"""
This module will be used to construct the surfaces and interfaces used in this package.
"""
from OgreInterface.surfaces import (
    Surface,
    Interface,
)
from OgreInterface import utils
from OgreInterface.lattice_match import ZurMcGill, OgreMatch

from pymatgen.core.structure import Structure, Molecule
from pymatgen.core.periodic_table import Element, DummySpecies
from pymatgen.core.lattice import Lattice
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.operations import SymmOp
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import JmolNN, CrystalNN
import networkx as nx
import numpy as np
import spglib


from tqdm import tqdm
import numpy as np
import math
from copy import deepcopy
from typing import Union, List, TypeVar, Tuple, Dict
from itertools import combinations, product, groupby
from ase import Atoms
from ase.data import chemical_symbols
from multiprocessing import Pool, cpu_count
import time
from collections.abc import Sequence

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

SelfSurfaceGenerator = TypeVar(
    "SelfSurfaceGenerator", bound="SurfaceGenerator"
)

SelfMolecularSurfaceGenerator = TypeVar(
    "SelfMolecularSurfaceGenerator", bound="MolecularSurfaceGenerator"
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
        >>> surface = surfaces[0] # OgreInterface.Surface object

        Creating a SurfaceGenerator object using the build in from_file() method:
        >>> from OgreInterface.generate import SurfaceGenerator
        >>> surfaces = SurfaceGenerator.from_file(filename="POSCAR_bulk", miller_index=[1, 1, 1], layers=5, vacuum=60)
        >>> surface = surfaces[0] # OgreInterface.Surface object

    Args:
        bulk: Bulk crystal structure used to create the surface
        miller_index: Miller index of the surface
        layers: Number of layers to include in the surface
        vacuum: Size of the vacuum to include over the surface in Angstroms
        refine_structure: Determines if the structure is first refined to it's standard settings according to it's spacegroup.
            This is done using spglib.standardize_cell(cell, to_primitive=False, no_idealize=False). Mainly this is usefull if
            users want to input a primitive cell of a structure instead of generating a conventional cell because most DFT people
            work exclusively with the primitive sturcture so we always have it on hand.
        generate_all: Determines if all possible surface terminations are generated.
        lazy: Determines if the surfaces are actually generated, or if only the surface basis vectors are found.
            (this is used for the MillerIndex search to make things faster)
        suppress_warnings: This gives the user the option to suppress warnings if they know what they are doing and don't need to see the warning messages

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
        refine_structure: bool = True,
        generate_all: bool = True,
        lazy: bool = False,
        suppress_warnings: bool = False,
    ) -> None:
        super().__init__()
        self.refine_structure = refine_structure
        self._suppress_warnings = suppress_warnings

        (
            self.bulk_structure,
            self.bulk_atoms,
            self.primitive_structure,
            self.primitive_atoms,
        ) = self._get_bulk(atoms_or_struc=bulk)

        self._use_prim = len(self.bulk_structure) != len(
            self.primitive_structure
        )
        # self._use_prim = False

        self._point_group_operations = self._get_point_group_operations()

        self.miller_index = miller_index
        self.layers = layers
        self.vacuum = vacuum
        self.generate_all = generate_all
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

    def __getitem__(self, i) -> Surface:
        if self._slabs:
            return self._slabs[i]
        else:
            print(
                "The slabs have not been generated yet, please use the generate_slabs() function to create them."
            )

    def __len__(self) -> int:
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
        refine_structure: bool = True,
        generate_all: bool = True,
        lazy: bool = False,
        suppress_warnings: bool = False,
    ) -> SelfSurfaceGenerator:
        """Creating a SurfaceGenerator from a file (i.e. POSCAR, cif, etc)

        Args:
            filename: File path to the structure file
            miller_index: Miller index of the surface
            layers: Number of layers to include in the surface
            vacuum: Size of the vacuum to include over the surface in Angstroms
            generate_all: Determines if all possible surface terminations are generated
            refine_structure: Determines if the structure is first refined to it's standard settings according to it's spacegroup.
                This is done using spglib.standardize_cell(cell, to_primitive=False, no_idealize=False). Mainly this is usefull if
                users want to input a primitive cell of a structure instead of generating a conventional cell because most DFT people
                work exclusively with the primitive structure so we always have it on hand.
            lazy: Determines if the surfaces are actually generated, or if only the surface basis vectors are found.
                (this is used for the MillerIndex search to make things faster)
            suppress_warnings: This gives the user the option to suppress warnings if they know what they are doing and don't need to see the warning messages

        Returns:
            SurfaceGenerator
        """
        structure = Structure.from_file(filename=filename)

        return cls(
            structure,
            miller_index,
            layers,
            vacuum,
            refine_structure,
            generate_all,
            lazy,
            suppress_warnings,
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

        if self.refine_structure:
            conventional_structure = utils.spglib_standardize(
                init_structure,
                to_primitive=False,
                no_idealize=False,
            )

            init_angles = init_structure.lattice.angles
            init_lengths = init_structure.lattice.lengths
            init_length_and_angles = np.concatenate(
                [list(init_lengths), list(init_angles)]
            )

            conv_angles = conventional_structure.lattice.angles
            conv_lengths = conventional_structure.lattice.lengths
            conv_length_and_angles = np.concatenate(
                [list(conv_lengths), list(conv_angles)]
            )

            if not np.isclose(
                conv_length_and_angles - init_length_and_angles, 0
            ).all():
                if not self._suppress_warnings:
                    labels = ["a", "b", "c", "alpha", "beta", "gamma"]
                    init_cell_str = ", ".join(
                        [
                            f"{label} = {val:.3f}"
                            for label, val in zip(
                                labels, init_length_and_angles
                            )
                        ]
                    )
                    conv_cell_str = ", ".join(
                        [
                            f"{label} = {val:.3f}"
                            for label, val in zip(
                                labels, conv_length_and_angles
                            )
                        ]
                    )
                    warning_str = "\n".join(
                        [
                            "----------------------------------------------------------",
                            "WARNING: The refined cell is different from the input cell",
                            f"Initial: {init_cell_str}",
                            f"Refined: {conv_cell_str}",
                            "Make sure the input miller index is for the refined structure, otherwise set refine_structure=False",
                            "To turn off this warning set suppress_warnings=True",
                            "----------------------------------------------------------",
                            "",
                        ]
                    )
                    print(warning_str)

            prim_structure = self._add_symmetry_info(
                conventional_structure, return_primitive=True
            )
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
                # TODO: Primitive structure with spglib for molecular crystals
                prim_structure = init_structure.get_primitive_structure()
            else:
                prim_structure = self._add_symmetry_info(
                    init_structure, return_primitive=True
                )

            init_atoms = AseAtomsAdaptor().get_atoms(init_structure)
            prim_atoms = AseAtomsAdaptor().get_atoms(prim_structure)

            # utils._get_colored_molecules(
            #     prim_structure, "vis/POSCAR_mol_prim_color"
            # )

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
        init_lattice = struc.lattice.matrix
        init_positions = struc.frac_coords
        init_numbers = np.array(struc.atomic_numbers)
        init_cell = (init_lattice, init_positions, init_numbers)

        init_dataset = spglib.get_symmetry_dataset(init_cell)

        struc.add_site_property(
            "bulk_wyckoff",
            init_dataset["wyckoffs"],
        )

        struc.add_site_property(
            "bulk_equivalent",
            init_dataset["equivalent_atoms"].tolist(),
        )

        if return_primitive:
            prim_mapping = init_dataset["mapping_to_primitive"]
            _, prim_inds = np.unique(prim_mapping, return_index=True)

            prim_bulk = utils.spglib_standardize(
                structure=struc,
                to_primitive=True,
                no_idealize=True,
            )

            prim_bulk.add_site_property(
                "bulk_wyckoff",
                [init_dataset["wyckoffs"][i] for i in prim_inds],
            )
            prim_bulk.add_site_property(
                "bulk_equivalent",
                init_dataset["equivalent_atoms"][prim_inds].tolist(),
            )

            return prim_bulk

    def _add_symmetry_info_old(self, struc, return_primitive=False):
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
        lattice = struc.lattice.matrix
        positions = struc.frac_coords
        numbers = np.array(struc.atomic_numbers)
        cell = (lattice, positions, numbers)
        dataset = spglib.get_symmetry_dataset(cell)
        wyckoffs = dataset["wyckoffs"]
        equivalent_atoms = dataset["equivalent_atoms"]
        molecule_index = struc.site_properties["molecule_index"]
        equivalent_molecules = [molecule_index[i] for i in equivalent_atoms]
        struc.add_site_property("bulk_wyckoff", wyckoffs)
        struc.add_site_property("bulk_equivalent", equivalent_molecules)

    def _add_symmetry_info_molecule_old(self, struc):
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
        # Poscar(init_oriented_struc).write_file("POSCAR_obs_001")

        planar_oriented_struc = init_oriented_struc.copy()
        planar_oriented_struc.apply_operation(to_planar_operation)

        planar_matrix = deepcopy(planar_oriented_struc.lattice.matrix)

        new_a, new_b, mat = utils.reduce_vectors_zur_and_mcgill(
            planar_matrix[0],
            planar_matrix[1],
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

    def _get_neighborhood(self, cutoff: float = 5.0) -> None:
        obs = self.oriented_bulk_structure.copy()
        obs.add_oxidation_state_by_guess()
        charges = [s._oxi_state for s in obs.species]
        self.oriented_bulk_structure.add_site_property("charge", charges)

        cnn = CrystalNN(search_cutoff=cutoff, cation_anion=True)

        bond_dict = {}
        for i, site in enumerate(obs):
            info_dict = cnn.get_nn_info(obs, i)
            bonds = []
            for neighbor in info_dict:
                bond = neighbor["site"].coords - site.coords
                bonds.append(
                    {
                        "from_Z": site.specie.Z,
                        "to_Z": neighbor["site"].specie.Z,
                        "bond": bond,
                    }
                )

            bond_dict[site.properties["oriented_bulk_equivalent"]] = bonds

        return bond_dict

    def _get_surface_atoms(
        self,
        obs: Structure,
        neighbor_info: Dict[int, List[Dict[str, Union[int, np.ndarray]]]],
    ):
        oriented_bulk_equivalents = np.array(list(neighbor_info.keys()))
        obs_oriented_bulk_equivalent = obs.site_properties[
            "oriented_bulk_equivalent"
        ]

        bottom_surf = []
        top_surf = []
        bottom_breaks = 0
        top_breaks = 0
        for obe in oriented_bulk_equivalents:
            if len(neighbor_info[obe]) > 0:
                ind = np.where(obs_oriented_bulk_equivalent == obe)[0][0]
                center_coord = obs[ind].coords
                bonds = np.vstack(
                    [bond["bond"] for bond in neighbor_info[obe]]
                )
                bonds += center_coord
                frac_bonds = bonds.dot(obs.lattice.inv_matrix)
                images = np.round(frac_bonds - np.mod(frac_bonds, 1)).astype(
                    int
                )

                bottom_broken_bonds = (images[:, -1] < 0).sum()
                top_broken_bonds = (images[:, -1] > 0).sum()

                if bottom_broken_bonds > 0:
                    bottom_breaks += bottom_broken_bonds
                    bottom_surf.append(obe)

                if top_broken_bonds > 0:
                    top_breaks += top_broken_bonds
                    top_surf.append(obe)

        is_top = np.isin(obs_oriented_bulk_equivalent, top_surf)
        is_bot = np.isin(obs_oriented_bulk_equivalent, bottom_surf)
        obs.add_site_property("is_top", is_top.tolist())
        obs.add_site_property("is_bottom", is_bot.tolist())

    def _get_surface_atoms_old(
        self, struc: Structure, cutoff: float = 4.0
    ) -> None:
        obs = struc.copy()
        obs.add_oxidation_state_by_guess()

        cnn = CrystalNN(search_cutoff=cutoff)
        top_neighborhood = []
        bottom_neighborhood = []
        for i in range(len(obs)):
            info_dict = cnn.get_nn_info(obs, i)
            top_broken_bonds = 0
            bottom_broken_bonds = 0
            for neighbor in info_dict:
                if neighbor["image"][-1] > 0:
                    top_broken_bonds += 1
                if neighbor["image"][-1] < 0:
                    bottom_broken_bonds += 1

            if top_broken_bonds > 0:
                top_neighborhood.append(i)

            if bottom_broken_bonds > 0:
                bottom_neighborhood.append(i)

        # is_surf = np.zeros(len(obs)).astype(bool)
        is_top = np.zeros(len(obs)).astype(bool)
        is_bottom = np.zeros(len(obs)).astype(bool)

        # is_surf[top_neighborhood + bottom_neighborhood] = True
        is_top[top_neighborhood] = True
        is_bottom[bottom_neighborhood] = True

        # struc.add_site_property("is_surf", is_surf.tolist())
        struc.add_site_property("is_top", is_top.tolist())
        struc.add_site_property("is_bottom", is_bottom.tolist())

    def _get_slab(self, neighbor_info, shift=0, tol: float = 0.1, energy=None):
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
        self._get_surface_atoms(slab_base, neighbor_info)

        is_top = np.array(slab_base.site_properties["is_top"])
        is_bottom = np.array(slab_base.site_properties["is_bottom"])
        bulk_equiv = np.array(slab_base.site_properties["bulk_equivalent"])

        bottom_bulk_equiv = bulk_equiv[is_bottom]
        top_bulk_equiv = bulk_equiv[is_top]

        surf_key = tuple(
            np.concatenate(
                [
                    np.sort(bottom_bulk_equiv),
                    np.sort(top_bulk_equiv),
                    np.zeros(bottom_bulk_equiv.shape),
                    np.ones(top_bulk_equiv.shape),
                ]
            ).astype(int)
        )

        z_coords = slab_base.frac_coords[:, -1]
        bot_z = z_coords.min()
        top_z = z_coords.max()
        unique_z = np.unique(np.round(z_coords, 5))
        z_groups = [
            (i, np.where(np.isclose(z_coords, z))[0])
            for i, z in enumerate(unique_z)
        ]
        atomic_layers = np.zeros(len(z_coords))

        for layer_num, inds in z_groups:
            atomic_layers[inds] = layer_num

        slab_base.add_site_property(
            "atomic_layer_index",
            atomic_layers.tolist(),
        )

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

        slab_base = Structure(
            lattice=slab_base.lattice,
            species=slab_base.species,
            coords=np.round(slab_base.frac_coords, 6),
            to_unit_cell=True,
            coords_are_cartesian=False,
            site_properties=slab_base.site_properties,
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

        center_shift = 0.5 * (vacuum_scale / (vacuum_scale + self.layers))
        non_orthogonal_slab.translate_sites(
            indices=range(len(non_orthogonal_slab)),
            vector=[0, 0, center_shift],
            frac_coords=True,
            to_unit_cell=True,
        )
        orthogonal_slab.translate_sites(
            indices=range(len(orthogonal_slab)),
            vector=[0, 0, center_shift],
            frac_coords=True,
            to_unit_cell=True,
        )
        top_z = non_orthogonal_slab.frac_coords[:, -1].max()
        top_cart = non_orthogonal_slab.lattice.matrix[-1] * top_z
        top_cart[-1] = 0.0
        orthogonal_slab.translate_sites(
            indices=range(len(orthogonal_slab)),
            vector=-top_cart,
            frac_coords=False,
            to_unit_cell=True,
        )

        orthogonal_slab = Structure(
            lattice=orthogonal_slab.lattice,
            species=orthogonal_slab.species,
            coords=np.round(orthogonal_slab.frac_coords, 6),
            to_unit_cell=True,
            coords_are_cartesian=False,
            site_properties=orthogonal_slab.site_properties,
        )

        return (
            slab_base,
            orthogonal_slab,
            non_orthogonal_slab,
            bottom_layer_dist,
            top_layer_dist,
            vacuum,
            surf_key,
        )

    def _generate_slabs(self) -> List[Surface]:
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
        surface_keys = []
        neighbor_info = self._get_neighborhood()

        if not self.generate_all:
            (
                shifted_slab_base,
                orthogonal_slab,
                non_orthogonal_slab,
                bottom_layer_dist,
                top_layer_dist,
                actual_vacuum,
                surf_key,
            ) = self._get_slab(
                shift=possible_shifts[0],
                neighbor_info=neighbor_info,
            )
            orthogonal_slab.sort_index = 0
            non_orthogonal_slab.sort_index = 0
            shifted_slab_bases.append(shifted_slab_base)
            orthogonal_slabs.append(orthogonal_slab)
            non_orthogonal_slabs.append(non_orthogonal_slab)
            bottom_layer_dists.append(bottom_layer_dist)
            top_layer_dists.append(top_layer_dist)
            surface_keys.append((surf_key, 0))
        else:
            for i, possible_shift in enumerate(possible_shifts):
                (
                    shifted_slab_base,
                    orthogonal_slab,
                    non_orthogonal_slab,
                    bottom_layer_dist,
                    top_layer_dist,
                    actual_vacuum,
                    surf_key,
                ) = self._get_slab(
                    shift=possible_shift,
                    neighbor_info=neighbor_info,
                )
                orthogonal_slab.sort_index = i
                non_orthogonal_slab.sort_index = i
                shifted_slab_bases.append(shifted_slab_base)
                orthogonal_slabs.append(orthogonal_slab)
                non_orthogonal_slabs.append(non_orthogonal_slab)
                bottom_layer_dists.append(bottom_layer_dist)
                top_layer_dists.append(top_layer_dist)
                surface_keys.append((surf_key, i))

        surfaces = []

        if self._use_prim:
            base_structure = self.primitive_structure
        else:
            base_structure = self.bulk_structure

        sorted_surface_keys = sorted(surface_keys, key=lambda x: x[0])
        groups = groupby(sorted_surface_keys, key=lambda x: x[0])

        unique_inds = []
        for group_key, group in groups:
            _, inds = list(zip(*group))
            unique_inds.append(min(inds))

        unique_inds.sort()

        # Loop through slabs to ensure that they are all properly oriented and reduced
        # Return Surface objects
        for i in unique_inds:
            # Create the Surface object
            surface = Surface(
                orthogonal_slab=orthogonal_slabs[i],
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


class MolecularSurfaceGenerator(SurfaceGenerator):
    """Class for generating surfaces from a given bulk molecular crystal structure.

    The MolecularSurfaceGenerator class generates surfaces with all possible terminations and contains
    information pertinent to generating interfaces with the InterfaceGenerator.

    Examples:
        Creating a SurfaceGenerator object using PyMatGen to load the structure:
        >>> from OgreInterface.generate import SurfaceGenerator
        >>> from pymatgen.core.structure import Structure
        >>> bulk = Structure.from_file("POSCAR_bulk")
        >>> surfaces = SurfaceGenerator(bulk=bulk, miller_index=[1, 1, 1], layers=5, vacuum=60)
        >>> surface = surfaces[0] # OgreInterface.Surface object

        Creating a SurfaceGenerator object using the build in from_file() method:
        >>> from OgreInterface.generate import SurfaceGenerator
        >>> surfaces = SurfaceGenerator.from_file(filename="POSCAR_bulk", miller_index=[1, 1, 1], layers=5, vacuum=60)
        >>> surface = surfaces[0] # OgreInterface.Surface object

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
        # Convert bulk to Structure if it is Atoms
        if type(bulk) == Atoms:
            bulk_structure = AseAtomsAdaptor.get_structure(bulk)
        else:
            bulk_structure = bulk
            
        # Get the primitive structure to label the molecules
        prim_bulk = utils.spglib_standardize(
            bulk_structure,
            to_primitive=True,
            no_idealize=True,
        )

        # Get the primitive to conventional transformation matrix
        prim_to_conv = np.round(utils.conv_a_to_b(prim_bulk, bulk_structure), 3)

        if not np.isclose(
            (np.round(prim_to_conv).astype(int) - prim_to_conv).sum(), 0.0
        ):
            print(
                "WARNING: Something went wrong with reducing the structure to its primitive cell"
            )

        # Label the molecules with dummy atoms
        self._label_molecules(prim_bulk)

        # Revert back to the supercell for the input into the SurfaceGenerator
        labeled_bulk = prim_bulk.copy()
        labeled_bulk.make_supercell(prim_to_conv)

        super().__init__(
            bulk=labeled_bulk,
            miller_index=miller_index,
            layers=layers,
            vacuum=vacuum,
            refine_structure=False,
            generate_all=generate_all,
            lazy=True,
        )

        # Extract the oriented bulk structure
        obs = self.oriented_bulk_structure

        # Replace the molecules with their corresponding dummy atoms at their centers of mass
        dummy_obs = utils.replace_molecules_with_atoms(obs)

        # Add oriented_bulk_equivalent site property
        dummy_obs.add_site_property(
            "oriented_bulk_equivalent", range(len(dummy_obs))
        )

        # Replace the oriented bulk structure from the SurfaceGenerator class with the dummy atoms
        self.oriented_bulk_structure = dummy_obs

        # Generate the surfaces
        if not lazy:
            self.generate_slabs()

    @classmethod
    def from_file(
        cls,
        filename: str,
        miller_index: List[int],
        layers: int,
        vacuum: float,
        generate_all: bool = True,
        lazy: bool = False,
    ) -> SelfMolecularSurfaceGenerator:
        """Creating a MolecularSurfaceGenerator from a file (i.e. POSCAR, cif, etc)

        Args:
            filename: File path to the structure file
            miller_index: Miller index of the surface
            layers: Number of layers to include in the surface
            vacuum: Size of the vacuum to include over the surface in Angstroms
            generate_all: Determines if all possible surface terminations are generated
            lazy: Determines if the surfaces are actually generated, or if only the surface basis vectors are found.
                (this is used for the MillerIndex search to make things faster)

        Returns:
            MolecularSurfaceGenerator
        """
        structure = Structure.from_file(filename=filename)

        return cls(
            structure,
            miller_index,
            layers,
            vacuum,
            generate_all,
            lazy,
        )

    def _label_molecules(self, struc):
        # Create a structure graph so we can extract the molecules
        struc_graph = StructureGraph.with_local_env_strategy(struc, JmolNN())
        graph = nx.Graph(struc_graph.graph)

        # Extract all molecule subgraphs
        subgraphs = [graph.subgraph(c) for c in nx.connected_components(graph)]

        labels = np.zeros(len(struc))
        for i, subgraph in enumerate(subgraphs):
            for n in subgraph:
                labels[n] = i + 22

        struc.add_site_property("dummy_species", labels.astype(int).tolist())
        struc.add_site_property(
            "molecule_index", (labels - 22).astype(int).tolist()
        )


class InterfaceGenerator:
    """Class for generating interfaces from two bulk structures

    This class will use the lattice matching algorithm from Zur and McGill to generate
    commensurate interface structures between two inorganic crystalline materials.

    Examples:
        >>> from OgreInterface.generate import SurfaceGenerator, InterfaceGenerator
        >>> subs = SurfaceGenerator.from_file(filename="POSCAR_sub", miller_index=[1,1,1], layers=5)
        >>> films = SurfaceGenerator.from_file(filename="POSCAR_film", miller_index=[1,1,1], layers=5)
        >>> interface_generator = InterfaceGenerator(substrate=subs[0], film=films[0])
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
        substrate: Union[Surface, Interface],
        film: Union[Surface, Interface],
        max_area_mismatch: float = 0.01,
        max_angle_strain: float = 0.01,
        max_linear_strain: float = 0.01,
        max_area: float = 500.0,
        interfacial_distance: Union[float, None] = 2.0,
        vacuum: float = 40.0,
        center: bool = False,
    ):
        if type(substrate) == Surface or type(substrate) == Interface:
            self.substrate = substrate
        else:
            raise TypeError(
                f"InterfaceGenerator accepts 'ogre.core.Surface' or 'ogre.core.Interface' not '{type(substrate).__name__}'"
            )

        if type(film) == Surface or type(film) == Interface:
            self.film = film
        else:
            raise TypeError(
                f"InterfaceGenerator accepts 'ogre.core.Surface' or 'ogre.core.Interface' not '{type(film).__name__}'"
            )

        self.center = center
        self.max_area_mismatch = max_area_mismatch
        self.max_angle_strain = max_angle_strain
        self.max_linear_strain = max_linear_strain
        self.max_area = max_area
        self.interfacial_distance = interfacial_distance
        self.vacuum = vacuum
        self.match_list = self._generate_interface_props()

    def _get_point_group_operations(self, struc: Structure) -> np.ndarray:
        sg = SpacegroupAnalyzer(struc)
        point_group_operations = sg.get_point_group_operations(cartesian=False)
        operation_array = np.round(
            np.array([p.rotation_matrix for p in point_group_operations])
        ).astype(np.int8)
        unique_operations = np.unique(operation_array, axis=0)

        return unique_operations

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
