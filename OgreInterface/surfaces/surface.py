"""
This module will be used to construct the surfaces and interfaces used in this package.
"""
from typing import Dict, Union, Iterable, List, Tuple, TypeVar
from itertools import combinations, groupby
from copy import deepcopy
from functools import reduce
import warnings

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import Element, Species
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SymmOp, SpacegroupAnalyzer
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pymatgen.analysis.local_env import CrystalNN, BrunnerNN_real
import numpy as np
from ase import Atoms

from OgreInterface import utils
from OgreInterface.lattice_match import OgreMatch
from OgreInterface.plotting_tools import plot_match
from OgreInterface.surfaces.oriented_bulk import OrientedBulk
from OgreInterface.surfaces.base_surface import BaseSurface


# suppress warning from CrystallNN when ionic radii are not found.
warnings.filterwarnings("ignore", module=r"pymatgen.analysis.local_env")

SelfSurface = TypeVar("SelfSurface", bound="Surface")


class Surface(BaseSurface):
    """Container for surfaces generated with the SurfaceGenerator

    The Surface class and will be used as an input to the InterfaceGenerator class,
    and it should be create exclusively using the SurfaceGenerator.

    Examples:
        Generating a surface with pseudo-hydrogen passivation where the atomic positions of the hydrogens need to be relaxed using DFT.
        >>> from OgreInterface.generate import SurfaceGenerator
        >>> surfaces = SurfaceGenerator.from_file(filename="POSCAR_bulk", miller_index=[1, 1, 1], layers=5, vacuum=60)
        >>> surface = surfaces[0] # OgreInterface.Surface object
        >>> surface.passivate(bot=True, top=True)
        >>> surface.write_file(output="POSCAR_slab", orthogonal=True, relax=True) # relax=True will automatically set selective dynamics=True for all passivating hydrogens

        Generating a surface with pseudo-hydrogen passivation that comes from a structure with pre-relaxed pseudo-hydrogens.
        >>> from OgreInterface.generate import SurfaceGenerator
        >>> surfaces = SurfaceGenerator.from_file(filename="POSCAR_bulk", miller_index=[1, 1, 1], layers=20, vacuum=60)
        >>> surface = surfaces[0] # OgreInterface.Surface object
        >>> surface.passivate(bot=True, top=True, passivated_struc="CONTCAR") # CONTCAR is the output of the structural relaxation
        >>> surface.write_file(output="POSCAR_slab", orthogonal=True, relax=False)

    Args:
        orthogonal_slab: Slab structure that is forced to have an c lattice vector that is orthogonal
            to the inplane lattice vectors
        non_orthogonal_slab: Slab structure that is not gaurunteed to have an orthogonal c lattice vector,
            and assumes the same basis as the primitive_oriented_bulk structure.
        oriented_bulk: Structure of the smallest building block of the slab, which was used to
            construct the non_orthogonal_slab supercell by creating a (1x1xN) supercell where N in the number
            of layers.
        bulk: Bulk structure that can be transformed into the slab basis using the transformation_matrix
        transformation_matrix: 3x3 integer matrix that used to change from the bulk basis to the slab basis.
        miller_index: Miller indices of the surface, with respect to the conventional bulk structure.
        layers: Number of unit cell layers in the surface
        vacuum: Size of the vacuum in Angstroms
        uvw_basis: Miller indices corresponding to the lattice vector directions of the slab
        point_group_operations: List of unique point group operations that will eventually be used to efficiently
            filter out symmetrically equivalent interfaces found using the lattice matching algorithm.
        bottom_layer_dist: z-distance of where the next atom should be if the slab structure were to continue downwards
            (This is used to automatically approximate the interfacial distance in interfacial_distance is set to None in the InterfaceGenerator)
        top_layer_dist: z-distance of where the next atom should be if the slab structure were to continue upwards
            (This is used to automatically approximate the interfacial distance in interfacial_distance is set to None in the InterfaceGenerator)
        termination_index: Index of the Surface in the list of Surfaces produced by the SurfaceGenerator
        surface_normal (np.ndarray): The normal vector of the surface
        c_projection (float): The projections of the c-lattice vector onto the surface normal

    Attributes:
        oriented_bulk_structure: Pymatgen Structure of the smallest building block of the slab, which was used to
            construct the non_orthogonal_slab supercell by creating a (1x1xN) supercell where N in the number
            of layers.
        oriented_bulk_atoms (Atoms): ASE Atoms of the smallest building block of the slab, which was used to
            construct the non_orthogonal_slab supercell by creating a (1x1xN) supercell where N in the number
            of layers.
        bulk_structure (Structure): Bulk Pymatgen Structure that can be transformed into the slab basis using the transformation_matrix
        bulk_atoms (Atoms): Bulk ASE Atoms that can be transformed into the slab basis using the transformation_matrix
        transformation_matrix (np.ndarray): 3x3 integer matrix that used to change from the bulk basis to the slab basis.
        miller_index (list): Miller indices of the surface, with respect to the conventional bulk structure.
        layers (int): Number of unit cell layers in the surface
        vacuum (float): Size of the vacuum in Angstroms
        uvw_basis (np.ndarray): Miller indices corresponding to the lattice vector directions of the slab
        point_group_operations (np.ndarray): List of unique point group operations that will eventually be used to efficiently
            filter out symmetrically equivalent interfaces found using the lattice matching algorithm.
        bottom_layer_dist (float): z-distance of where the next atom should be if the slab structure were to continue downwards
            (This is used to automatically approximate the interfacial distance in interfacial_distance is set to None in the InterfaceGenerator)
        top_layer_dist (float): z-distance of where the next atom should be if the slab structure were to continue upwards
            (This is used to automatically approximate the interfacial distance in interfacial_distance is set to None in the InterfaceGenerator)
        termination_index (int): Index of the Surface in the list of Surfaces produced by the SurfaceGenerator
        surface_normal (np.ndarray): The normal vector of the surface
        c_projection (float): The projections of the c-lattice vector onto the surface normal
    """

    def __init__(
        self,
        slab: Structure,
        oriented_bulk: OrientedBulk,
        miller_index: list,
        layers: int,
        vacuum: float,
        termination_index: int,
    ) -> None:
        super().__init__(
            slab=slab,
            oriented_bulk=oriented_bulk,
            miller_index=miller_index,
            layers=layers,
            vacuum=vacuum,
            termination_index=termination_index,
        )

    @property
    def top_surface_charge(self) -> float:
        frac_coords = self.oriented_bulk_structure.frac_coords
        mod_frac_coords = np.mod(np.round(frac_coords[:, -1], 5), 1.0)
        mod_frac_coords += 1.0 - mod_frac_coords.max()

        charges = np.array(
            self.oriented_bulk_structure.site_properties["charge"]
        )

        z_frac = 1 - mod_frac_coords

        return np.round((charges * z_frac).sum(), 4)

    @property
    def bottom_surface_charge(self) -> float:
        frac_coords = self.oriented_bulk_structure.frac_coords

        mod_frac_coords = np.mod(np.round(frac_coords[:, -1], 5), 1.0)
        mod_frac_coords -= mod_frac_coords.min()

        charges = np.array(
            self.oriented_bulk_structure.site_properties["charge"]
        )
        z_frac = mod_frac_coords

        return np.round((charges * z_frac).sum(), 4)

    def _get_surface_atoms(self, cutoff: float) -> Tuple[Structure, List]:
        obs = self.oriented_bulk_structure.copy()
        obs.add_oxidation_state_by_site(obs.site_properties["charge"])

        layer_struc = utils.get_layer_supercell(structure=obs, layers=3)
        layer_struc.sort()

        layer_inds = np.array(layer_struc.site_properties["layer_index"])

        bottom_inds = np.where(layer_inds == 0)[0]
        top_inds = np.where(layer_inds == np.max(layer_inds))[0]

        cnn = CrystalNN(search_cutoff=cutoff)
        top_neighborhood = []
        for i in top_inds:
            info_dict = cnn.get_nn_info(layer_struc, i)
            for neighbor in info_dict:
                if neighbor["image"][-1] > 0:
                    top_neighborhood.append((i, info_dict))
                    break

        bottom_neighborhood = []
        for i in bottom_inds:
            info_dict = cnn.get_nn_info(layer_struc, i)
            for neighbor in info_dict:
                if neighbor["image"][-1] < 0:
                    bottom_neighborhood.append((i, info_dict))
                    break

        neighborhool_list = [bottom_neighborhood, top_neighborhood]

        return layer_struc, neighborhool_list

    def _get_pseudohydrogen_charge(
        self,
        site,
        coordination,
        include_d_valence: bool = True,
        manual_oxidation_states: Union[Dict[str, float], None] = None,
    ) -> float:
        electronic_struc = site.specie.electronic_structure.split(".")[1:]

        # TODO automate anion/cation determination
        if manual_oxidation_states:
            species_str = str(site.specie._el)
            oxi_state = manual_oxidation_states[species_str]
        else:
            oxi_state = site.specie.oxi_state

        valence = 0
        for orb in electronic_struc:
            if include_d_valence:
                if orb[1] == "d":
                    if int(orb[2:]) < 10:
                        valence += int(orb[2:])
                else:
                    valence += int(orb[2:])
            else:
                if orb[1] != "d":
                    valence += int(orb[2:])

        if oxi_state < 0:
            charge = (8 - valence) / coordination
        else:
            charge = ((2 * coordination) - valence) / coordination

        available_charges = np.array(
            [
                0.25,
                0.33,
                0.42,
                0.5,
                0.58,
                0.66,
                0.75,
                1.00,
                1.25,
                1.33,
                1.50,
                1.66,
                1.75,
            ]
        )

        closest_charge = np.abs(charge - available_charges)
        min_diff = np.isclose(closest_charge, closest_charge.min())
        charge = np.min(available_charges[min_diff])

        return charge

    def _get_bond_dict(
        self,
        cutoff: float,
        include_d_valence: bool,
        manual_oxidation_states,
    ) -> Dict[str, Dict[int, Dict[str, Union[np.ndarray, float, str]]]]:
        image_map = {1: "+", 0: "=", -1: "-"}
        (
            layer_struc,
            surface_neighborhoods,
        ) = self._get_surface_atoms(cutoff)

        labels = ["bottom", "top"]
        bond_dict = {"bottom": {}, "top": {}}
        H_len = 0.31

        for i, neighborhood in enumerate(surface_neighborhoods):
            for surface_atom in neighborhood:
                atom_index = surface_atom[0]
                center_atom_equiv_index = layer_struc[atom_index].properties[
                    "oriented_bulk_equivalent"
                ]

                try:
                    center_len = CovalentRadius.radius[
                        layer_struc[atom_index].specie.symbol
                    ]
                except KeyError:
                    center_len = layer_struc[atom_index].specie.atomic_radius

                oriented_bulk_equivalent = layer_struc[atom_index].properties[
                    "oriented_bulk_equivalent"
                ]
                neighbor_info = surface_atom[1]
                coordination = len(neighbor_info)
                charge = self._get_pseudohydrogen_charge(
                    layer_struc[atom_index],
                    coordination,
                    include_d_valence,
                    manual_oxidation_states,
                )
                broken_atoms = [
                    neighbor
                    for neighbor in neighbor_info
                    if neighbor["image"][-1] != 0
                ]

                bonds = []
                bond_strs = []
                for atom in broken_atoms:
                    broken_site = atom["site"]
                    broken_atom_equiv_index = broken_site.properties[
                        "oriented_bulk_equivalent"
                    ]
                    broken_image = np.array(broken_site.image).astype(int)
                    broken_atom_cart_coords = broken_site.coords
                    center_atom_cart_coords = layer_struc[atom_index].coords
                    bond_vector = (
                        broken_atom_cart_coords - center_atom_cart_coords
                    )
                    norm_vector = bond_vector / np.linalg.norm(bond_vector)
                    H_vector = (H_len + center_len) * norm_vector

                    H_str = ",".join(
                        [
                            str(center_atom_equiv_index),
                            str(broken_atom_equiv_index),
                            "".join([image_map[i] for i in broken_image]),
                            str(i),  # top or bottom bottom=0, top=1
                        ]
                    )

                    bonds.append(H_vector)
                    bond_strs.append(H_str)

                bond_dict[labels[i]][oriented_bulk_equivalent] = {
                    "bonds": np.vstack(bonds),
                    "bond_strings": bond_strs,
                    "charge": charge,
                }

        return bond_dict

    def _get_passivation_atom_index(
        self, struc, bulk_equivalent, top=False
    ) -> int:
        struc_layer_index = np.array(struc.site_properties["layer_index"])
        struc_bulk_equiv = np.array(
            struc.site_properties["oriented_bulk_equivalent"]
        )

        if top:
            layer_number = np.max(struc_layer_index)
        else:
            layer_number = 0

        atom_index = np.where(
            np.logical_and(
                struc_layer_index == layer_number,
                struc_bulk_equiv == bulk_equivalent,
            )
        )[0][0]

        return atom_index

    def _passivate(self, struc, index, bond, bond_str, charge) -> None:
        position = struc[index].coords + bond
        frac_coords = np.mod(
            np.round(struc.lattice.get_fractional_coords(position), 6), 1
        )
        props = {k: -1 for k in struc[index].properties}
        props["hydrogen_str"] = f"{index}," + bond_str

        struc.append(
            Species("H", oxidation_state=charge),
            coords=frac_coords,
            coords_are_cartesian=False,
            properties=props,
        )

    def _get_passivated_bond_dict(
        self,
        bond_dict: Dict[
            str, Dict[int, Dict[str, Union[np.ndarray, float, str]]]
        ],
        relaxed_structure_file: str,
    ) -> Dict[str, Dict[int, Dict[str, Union[np.ndarray, float, str]]]]:
        # Load in the relaxed structure file to get the description string
        with open(relaxed_structure_file, "r") as f:
            poscar_str = f.read().split("\n")

        # Get the description string at the top of the POSCAR/CONTCAR
        desc_str = poscar_str[0].split("|")

        # Extract the number of layers
        layers = int(desc_str[0].split("=")[1])

        # Extract the termination index
        termination_index = int(desc_str[1].split("=")[1])

        # If the termination index is the same the proceed with passivation
        if termination_index == self.termination_index:
            # Extract the structure
            structure = Structure.from_file(relaxed_structure_file)

            # Make a copy of the oriented bulk structure
            obs = self.oriented_bulk_structure.copy()

            # Add oxidation states for the passivation
            obs.add_oxidation_state_by_guess()

            # If the OBS is left handed make it right handed like the pymatgen Poscar class does
            is_negative = np.linalg.det(obs.lattice.matrix) < 0

            if is_negative:
                structure = Structure(
                    lattice=Lattice(structure.lattice.matrix * -1),
                    species=structure.species,
                    coords=structure.frac_coords,
                )

            # Reproduce the passivated structure
            vacuum_scale = 4
            layer_struc = utils.get_layer_supercell(
                structure=obs, layers=layers, vacuum_scale=vacuum_scale
            )

            center_shift = 0.5 * (vacuum_scale / (vacuum_scale + layers))
            layer_struc.translate_sites(
                indices=range(len(layer_struc)),
                vector=[0, 0, center_shift],
                frac_coords=True,
                to_unit_cell=True,
            )

            layer_struc.sort()

            # Add hydrogen_str propery. This avoids the PyMatGen warning
            layer_struc.add_site_property(
                "hydrogen_str", [-1] * len(layer_struc)
            )

            # Add a site propery indexing each atom before the passivation is applied
            layer_struc.add_site_property(
                "pre_passivation_index", list(range(len(layer_struc)))
            )

            # Get top and bottom species to determine if the layer_struc should be
            # passivated on the top or bottom of the structure
            atomic_numbers = structure.atomic_numbers
            top_species = atomic_numbers[
                np.argmax(structure.frac_coords[:, -1])
            ]
            bot_species = atomic_numbers[
                np.argmin(structure.frac_coords[:, -1])
            ]

            # If the top species is a Hydrogen then passivate the top
            if top_species == 1:
                for bulk_equiv, bonds in bond_dict["top"].items():
                    ortho_index = self._get_passivation_atom_index(
                        struc=layer_struc, bulk_equivalent=bulk_equiv, top=True
                    )

                    for bond, bond_str in zip(
                        bonds["bonds"], bonds["bond_strings"]
                    ):
                        self._passivate(
                            layer_struc,
                            ortho_index,
                            bond,
                            bond_str,
                            bonds["charge"],
                        )

            # If the bottom species is a Hydrogen then passivate the bottom
            if bot_species == 1:
                for bulk_equiv, bonds in bond_dict["bottom"].items():
                    ortho_index = self._get_passivation_atom_index(
                        struc=layer_struc,
                        bulk_equivalent=bulk_equiv,
                        top=False,
                    )

                    for bond, bond_str in zip(
                        bonds["bonds"], bonds["bond_strings"]
                    ):
                        self._passivate(
                            layer_struc,
                            ortho_index,
                            bond,
                            bond_str,
                            bonds["charge"],
                        )

            layer_struc.sort()

            shifts = np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [1, 1, 0],
                    [-1, 1, 0],
                    [1, -1, 0],
                    [-1, -1, 0],
                    [-1, 0, 0],
                    [0, -1, 0],
                ]
            ).dot(structure.lattice.matrix)

            # Get the index if the hydrogens
            hydrogen_index = np.where(np.array(structure.atomic_numbers) == 1)[
                0
            ]

            # Get the bond strings from the passivated structure
            bond_strs = layer_struc.site_properties["hydrogen_str"]

            # Get the index of sites before passivation
            pre_pas_inds = layer_struc.site_properties["pre_passivation_index"]

            # The bond center of the hydrogens are the first element of the bond string
            pre_pas_bond_centers = [
                int(bond_strs[i].split(",")[0]) for i in hydrogen_index
            ]

            # Map the pre-passivation bond index to the actual index in the passivated structure
            post_pas_bond_centers = [
                pre_pas_inds.index(i) for i in pre_pas_bond_centers
            ]

            # Get the coordinates of the bond centers in the actual relaxed structure
            # and the recreated ideal passivated structure
            relaxed_bond_centers = structure.cart_coords[post_pas_bond_centers]
            ideal_bond_centers = layer_struc.cart_coords[post_pas_bond_centers]

            # Get the coordinates of the hydrogens in the actual relaxed structure
            # and the recreated ideal passivated structure
            relaxed_hydrogens = structure.cart_coords[hydrogen_index]
            ideal_hydrogens = layer_struc.cart_coords[hydrogen_index]

            # Substract the bond center positions from the hydrogen positions to get only the bond vector
            relaxed_hydrogens -= relaxed_bond_centers
            ideal_hydrogens -= ideal_bond_centers

            # Mapping to accessing the bond_dict
            top_bot_dict = {1: "top", 0: "bottom"}

            # Lopp through the matching hydrogens and indices to get the difference between the bond vectors
            for H_ind, H_ideal, H_relaxed in zip(
                hydrogen_index, ideal_hydrogens, relaxed_hydrogens
            ):
                # Find all periodic shifts of the relaxed hydrogens
                relaxed_shifts = H_relaxed + shifts

                # Find the difference between the ideal hydrogens and all 3x3 periodic images of the relaxed hydrogen
                diffs = relaxed_shifts - H_ideal

                # Find the length of the bond difference vectors
                norm_diffs = np.linalg.norm(diffs, axis=1)

                # Find the difference vector between the ideal hydrogen and the closest relaxed hydrogen image
                bond_diff = diffs[np.argmin(norm_diffs)]

                # Get the bond string of the hydrogen
                bond_str = bond_strs[H_ind].split(",")

                # Extract the side from the bond string (the last element)
                side = top_bot_dict[int(bond_str[-1])]

                # Get the center index
                center_ind = int(bond_str[1])

                # Extract the bond info from the bond_dict
                bond_info = bond_dict[side][center_ind]

                # Find which bond this hydrogen corresponds to
                bond_ind = bond_info["bond_strings"].index(
                    ",".join(bond_str[1:])
                )

                # Add the bond diff to the bond to get the relaxed position
                bond_dict[side][center_ind]["bonds"][bond_ind] += bond_diff

            return bond_dict
        else:
            raise ValueError(
                f"This is not the same termination. The passivated structure has termination={termination_index}, and the current surface has termination={self.termination_index}"
            )

    def passivate(
        self,
        bottom: bool = True,
        top: bool = True,
        cutoff: float = 4.0,
        passivated_struc: Union[str, None] = None,
        include_d_valence: bool = True,
        manual_oxidation_states: Union[Dict[str, float], None] = None,
    ) -> None:
        """
        This function will apply pseudohydrogen passivation to all broken bonds on the surface and assign charges to the pseudo-hydrogens based
        on the equations provided in https://doi.org/10.1103/PhysRevB.85.195328. The identification of the local coordination environments is
        provided using CrystalNN in Pymatgen which is based on https://doi.org/10.1021/acs.inorgchem.0c02996.

        Examples:
            Initial passivation:
            >>> surface.passivate(bottom=True, top=True)

            Relaxed passivation from a CONTCAR file:
            >>> surface.passivate(bottom=True, top=True, passivated_struc="CONTCAR")

        Args:
            bottom: Determines if the bottom of the structure should be passivated
            top: Determines of the top of the structure should be passivated
            cutoff: Determines the cutoff in Angstroms for the nearest neighbor search. 3.0 seems to give reasonalble reasults.
            passivated_struc: File path to the CONTCAR/POSCAR file that contains the relaxed atomic positions of the pseudo-hydrogens.
                This structure must have the same miller index and termination index.
            include_d_valence: (DO NOT CHANGE FROM DEFAULT, THIS IS ONLY FOR DEBUGING) Determines if the d-orbital electrons are included the calculation of the pseudohydrogen charge.
            manual_oxidation_states:  (DO NOT CHANGE FROM DEFAULT, THIS IS ONLY FOR DEBUGING) Option to pass in a dictionary determining which elements are anions vs cations.
                This will be automated hopefully at some point.
                (i.e {"Ti": 1, "Mn": 1, "In": -1} would mean Ti and Mn are cations and In is an anion)
        """
        bond_dict = self._get_bond_dict(
            cutoff, include_d_valence, manual_oxidation_states
        )

        if passivated_struc is not None:
            bond_dict = self._get_passivated_bond_dict(
                bond_dict=bond_dict, relaxed_structure_file=passivated_struc
            )

        ortho_slab = self._orthogonal_slab_structure.copy()
        non_ortho_slab = self._non_orthogonal_slab_structure.copy()
        ortho_slab.add_site_property("hydrogen_str", [-1] * len(ortho_slab))
        non_ortho_slab.add_site_property(
            "hydrogen_str", [-1] * len(non_ortho_slab)
        )

        if top:
            for bulk_equiv, bonds in bond_dict["top"].items():
                ortho_index = self._get_passivation_atom_index(
                    struc=ortho_slab, bulk_equivalent=bulk_equiv, top=True
                )
                non_ortho_index = self._get_passivation_atom_index(
                    struc=non_ortho_slab, bulk_equivalent=bulk_equiv, top=True
                )

                for bond, bond_str in zip(
                    bonds["bonds"], bonds["bond_strings"]
                ):
                    self._passivate(
                        ortho_slab,
                        ortho_index,
                        bond,
                        bond_str,
                        bonds["charge"],
                    )
                    self._passivate(
                        non_ortho_slab,
                        non_ortho_index,
                        bond,
                        bond_str,
                        bonds["charge"],
                    )

        if bottom:
            for bulk_equiv, bonds in bond_dict["bottom"].items():
                ortho_index = self._get_passivation_atom_index(
                    struc=ortho_slab, bulk_equivalent=bulk_equiv, top=False
                )
                non_ortho_index = self._get_passivation_atom_index(
                    struc=non_ortho_slab, bulk_equivalent=bulk_equiv, top=False
                )

                for bond, bond_str in zip(
                    bonds["bonds"], bonds["bond_strings"]
                ):
                    self._passivate(
                        ortho_slab,
                        ortho_index,
                        bond,
                        bond_str,
                        bonds["charge"],
                    )
                    self._passivate(
                        non_ortho_slab,
                        non_ortho_index,
                        bond,
                        bond_str,
                        bonds["charge"],
                    )

        ortho_slab.sort()
        non_ortho_slab.sort()

        ortho_slab.remove_site_property("hydrogen_str")
        non_ortho_slab.remove_site_property("hydrogen_str")

        self._passivated = True
        self._orthogonal_slab_structure = ortho_slab
        self._non_orthogonal_slab_structure = non_ortho_slab