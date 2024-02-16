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
from OgreInterface.surfaces.surface import Surface
from OgreInterface.surfaces.base_surface import BaseSurface
from OgreInterface.surfaces.molecular_surface import MolecularSurface


# suppress warning from CrystallNN when ionic radii are not found.
warnings.filterwarnings("ignore", module=r"pymatgen.analysis.local_env")

SelfInterface = TypeVar("SelfInterface", bound="Interface")


class Interface:
    """Container of Interfaces generated using the InterfaceGenerator

    The Surface class and will be used as an input to the InterfaceGenerator class,
    and it should be create exclusively using the SurfaceGenerator.

    Args:
        substrate: Surface class of the substrate material
        film: Surface class of the film material
        match: OgreMatch class of the matching interface
        interfacial_distance: Distance between the top atom of the substrate and the bottom atom of the film
        vacuum: Size of the vacuum in Angstroms
        center: Determines if the interface is centered in the vacuum

    Attributes:
        substrate (Surface): Surface class of the substrate material
        film (Surface): Surface class of the film material
        match (OgreMatch): OgreMatch class of the matching interface
        interfacial_distance (float): Distance between the top atom of the substrate and the bottom atom of the film
        vacuum (float): Size of the vacuum in Angstroms
        center (bool): Determines if the interface is centered in the vacuum
    """

    def __init__(
        self,
        substrate: Union[Surface, MolecularSurface, SelfInterface],
        film: Union[Surface, MolecularSurface, SelfInterface],
        match: OgreMatch,
        interfacial_distance: float,
        vacuum: float,
        center: bool = True,
        substrate_strain_fraction: float = 0.0,
    ) -> None:
        self.center = center
        self.substrate = substrate
        self.film = film
        self.match = match
        self.vacuum = vacuum
        self._substrate_strain_fraction = substrate_strain_fraction
        (
            self._substrate_supercell,
            self._substrate_obs_supercell,
            self._substrate_supercell_uvw,
            self._substrate_supercell_scale_factors,
        ) = self._create_supercell(substrate=True)
        (
            self._film_supercell,
            self._film_obs_supercell,
            self._film_supercell_uvw,
            self._film_supercell_scale_factors,
        ) = self._create_supercell(substrate=False)

        self._substrate_a_to_i = self._orient_supercell(
            supercell=self._substrate_supercell,
            substrate=True,
        )
        self._film_a_to_i = self._orient_supercell(
            supercell=self._film_supercell,
            substrate=False,
        )

        self._avg_sc_inplane_lattice = self._get_average_inplane_lattice()

        self.interfacial_distance = interfacial_distance
        (
            self._strained_sub,
            self._substrate_strain_matrix,
        ) = self._prepare_substrate()
        self._strained_film, self._film_strain_matrix = self._prepare_film()

        (
            self._M_matrix,
            self._non_orthogonal_structure,
            self._non_orthogonal_substrate_structure,
            self._non_orthogonal_film_structure,
            self._non_orthogonal_atoms,
            self._non_orthogonal_substrate_atoms,
            self._non_orthogonal_film_atoms,
            self._orthogonal_structure,
            self._orthogonal_substrate_structure,
            self._orthogonal_film_structure,
            self._orthogonal_atoms,
            self._orthogonal_substrate_atoms,
            self._orthogonal_film_atoms,
        ) = self._stack_interface()
        self._a_shift = 0.0
        self._b_shift = 0.0
        # self.bottom_layer_dist = self.substrate.bottom_layer_dist
        # self.top_layer_dist = self.film.top_layer_dist

    def _get_average_inplane_lattice(self):
        film_lattice = self._film_supercell.lattice.matrix[:2]
        substrate_lattice = self._substrate_supercell.lattice.matrix[:2]

        film_frac = self._substrate_strain_fraction
        sub_frac = 1 - film_frac

        avg_lattice = (film_frac * film_lattice) + (
            sub_frac * substrate_lattice
        )

        return avg_lattice

    @property
    def inplane_vectors(self) -> np.ndarray:
        """
        In-plane cartesian vectors of the interface structure

        Examples:
            >>> interface.inplane_vectors
            >>> [[4.0 0.0 0.0]
            ...  [2.0 2.0 0.0]]

        Returns:
            (2, 3) numpy array containing the cartesian coordinates of the in-place lattice vectors
        """
        matrix = deepcopy(self._orthogonal_structure.lattice.matrix)
        return matrix[:2]

    @property
    def uvw_basis(self) -> np.ndarray:
        return np.eye(3).astype(int)

    @property
    def oriented_bulk_structure(self) -> Structure:
        return self.substrate.oriented_bulk_structure

    @property
    def substrate_oriented_bulk_supercell(self) -> Structure:
        if self._substrate_obs_supercell is not None:
            obs_supercell = utils.apply_strain_matrix(
                structure=self._substrate_obs_supercell,
                strain_matrix=self._substrate_strain_matrix,
            )
            return obs_supercell
        else:
            raise "substrate_oriented_bulk_supercell is not applicable when an Interface is used as the substrate"

    @property
    def film_oriented_bulk_supercell(self) -> Structure:
        if self._film_obs_supercell is not None:
            obs_supercell = utils.apply_strain_matrix(
                structure=self._film_obs_supercell,
                strain_matrix=self._film_strain_matrix,
            )
            return obs_supercell
        else:
            raise "film_oriented_bulk_supercell is not applicable when an Interface is used as the film"

    @property
    def substrate_oriented_bulk_structure(self) -> Structure:
        if issubclass(type(self.substrate), BaseSurface):
            obs_structure = utils.apply_strain_matrix(
                structure=self.substrate.oriented_bulk_structure,
                strain_matrix=self._substrate_strain_matrix,
            )
            return obs_structure
        else:
            raise "substrate_oriented_bulk_structure is not applicable when an Interface is used as the substrate"

    @property
    def film_oriented_bulk_structure(self) -> Structure:
        if issubclass(type(self.film), BaseSurface):
            obs_structure = utils.apply_strain_matrix(
                structure=self.film.oriented_bulk_structure,
                strain_matrix=self._film_strain_matrix,
            )
            return obs_structure
        else:
            raise "film_oriented_bulk_structure is not applicable when an Interface is used as the film"

    @property
    def layer_thickness(self) -> float:
        return self.substrate.oriented_bulk.layer_thickness

    def _passivated(self) -> bool:
        return self.substrate._passivated or self.film._passivated

    @property
    def bulk_transformation_matrix(self) -> np.ndarray:
        return self.substrate.bulk_transformation_matrix

    @property
    def surface_normal(self) -> np.ndarray:
        return self.substrate.surface_normal

    @property
    def layers(self) -> int:
        return self.substrate.layers + self.film.layers

    @property
    def atomic_layers(self) -> int:
        return self.substrate.atomic_layers + self.film.atomic_layers

    @property
    def termination_index(self) -> int:
        return 0

    @property
    def point_group_operations(self) -> np.ndarray:
        sg = SpacegroupAnalyzer(self._orthogonal_structure)
        point_group_operations = sg.get_point_group_operations(cartesian=False)
        operation_array = np.round(
            np.array([p.rotation_matrix for p in point_group_operations])
        ).astype(np.int8)
        unique_operations = np.unique(operation_array, axis=0)

        return unique_operations

    @property
    def transformation_matrix(self):
        """
        Transformation matrix to convert the primitive bulk lattice vectors of the substrate material to the
        interface supercell lattice vectors (including the vacuum region)

        Examples:
            >>> interface.transformation_matrix
            >>> [[ -2   2   0]
            ...  [  0   0   2]
            ...  [ 15  15 -15]]
        """
        return self._M_matrix.astype(int)

    @property
    def interface_height(self):
        """
        This returns the z-height of the interface (average between the top film atom z and bottom substrate z)
        """
        sub_z_coords = self._orthogonal_substrate_structure.cart_coords[:, -1]
        film_z_coords = self._orthogonal_film_structure.cart_coords[:, -1]

        return (sub_z_coords.min() + film_z_coords.max()) / 2

    def get_interface(
        self,
        orthogonal: bool = True,
        return_atoms: bool = False,
    ) -> Union[Atoms, Structure]:
        """
        This is a simple function for easier access to the interface structure generated from the OgreMatch

        Args:
            orthogonal: Determines if the orthogonalized structure is returned
            return_atoms: Determines if the ASE Atoms object is returned instead of a Pymatgen Structure object (default)

        Returns:
            Either a Pymatgen Structure of ASE Atoms object of the interface structure
        """
        if orthogonal:
            if return_atoms:
                return self._orthogonal_atoms
            else:
                return self._orthogonal_structure
        else:
            if return_atoms:
                return self._non_orthogonal_atoms
            else:
                return self._non_orthogonal_structure

    def get_substrate_supercell(
        self,
        orthogonal: bool = True,
        return_atoms: bool = False,
    ) -> Union[Atoms, Structure]:
        """
        This is a simple function for easier access to the substrate supercell generated from the OgreMatch
        (i.e. the interface structure with the film atoms removed)

        Args:
            orthogonal: Determines if the orthogonalized structure is returned
            return_atoms: Determines if the ASE Atoms object is returned instead of a Pymatgen Structure object (default)

        Returns:
            Either a Pymatgen Structure of ASE Atoms object of the substrate supercell structure
        """
        if orthogonal:
            if return_atoms:
                return self._orthogonal_substrate_atoms
            else:
                return self._orthogonal_substrate_structure
        else:
            if return_atoms:
                return self._non_orthogonal_substrate_atoms
            else:
                return self._non_orthogonal_substrate_structure

    def get_film_supercell(
        self,
        orthogonal: bool = True,
        return_atoms: bool = False,
    ) -> Union[Atoms, Structure]:
        """
        This is a simple function for easier access to the film supercell generated from the OgreMatch
        (i.e. the interface structure with the substrate atoms removed)

        Args:
            orthogonal: Determines if the orthogonalized structure is returned
            return_atoms: Determines if the ASE Atoms object is returned instead of a Pymatgen Structure object (default)

        Returns:
            Either a Pymatgen Structure of ASE Atoms object of the film supercell structure
        """
        if orthogonal:
            if return_atoms:
                return self._orthogonal_film_atoms
            else:
                return self._orthogonal_film_structure
        else:
            if return_atoms:
                return self._non_orthogonal_film_atoms
            else:
                return self._non_orthogonal_film_structure

    def get_substrate_layer_indices(
        self,
        layer_from_interface: int,
        atomic_layers: bool = True,
    ) -> np.ndarray:
        """
        This function is used to extract the atom-indicies of specific layers of the substrate part of the interface.

        Examples:
            >>> interface.get_substrate_layer_indices(layer_from_interface=0)
            >>> [234 235 236 237 254 255 256 257]


        Args:
            layer_from_interface: The layer number of the substrate which you would like to get
                atom-indices for. The layer number is reference from the interface, so layer_from_interface=0
                would be the layer of the substrate that is at the interface.

        Returns:
            A numpy array of integer indices corresponding to the atom index of the interface structure
        """
        if atomic_layers:
            layer_key = "atomic_layer_index"
        else:
            layer_key = "layer_index"

        interface = self._non_orthogonal_structure
        site_props = interface.site_properties
        is_sub = np.array(site_props["is_sub"])
        layer_index = np.array(site_props[layer_key])
        sub_n_layers = layer_index[is_sub].max()
        rel_layer_index = sub_n_layers - layer_index
        is_layer = rel_layer_index == layer_from_interface

        return np.where(np.logical_and(is_sub, is_layer))[0]

    def get_film_layer_indices(
        self,
        layer_from_interface: int,
        atomic_layers: bool = True,
    ) -> np.ndarray:
        """
        This function is used to extract the atom-indicies of specific layers of the film part of the interface.

        Examples:
            >>> interface.get_substrate_layer_indices(layer_from_interface=0)
            >>> [0 1 2 3 4 5 6 7 8 9 10 11 12]

        Args:
            layer_from_interface: The layer number of the film which you would like to get atom-indices for.
            The layer number is reference from the interface, so layer_from_interface=0
            would be the layer of the film that is at the interface.

        Returns:
            A numpy array of integer indices corresponding to the atom index of the interface structure
        """
        if atomic_layers:
            layer_key = "atomic_layer_index"
        else:
            layer_key = "layer_index"

        interface = self._non_orthogonal_structure
        site_props = interface.site_properties
        is_film = np.array(site_props["is_film"])
        layer_index = np.array(site_props[layer_key])
        is_layer = layer_index == layer_from_interface

        return np.where(np.logical_and(is_film, is_layer))[0]

    def replace_species(
        self, site_index: int, species_mapping: Dict[str, str]
    ) -> None:
        """
        This function can be used to replace the species at a given site in the interface structure

        Examples:
            >>> interface.replace_species(site_index=42, species_mapping={"In": "Zn", "As": "Te"})

        Args:
            site_index: Index of the site to be replaced
            species_mapping: Dictionary showing the mapping between species.
                For example if you wanted to replace InAs with ZnTe then the species mapping would
                be as shown in the example above.
        """
        species_str = self._orthogonal_structure[site_index].species_string

        if species_str in species_mapping:
            is_sub = self._non_orthogonal_structure[site_index].properties[
                "is_sub"
            ]
            self._non_orthogonal_structure[site_index].species = Element(
                species_mapping[species_str]
            )
            self._orthogonal_structure[site_index].species = Element(
                species_mapping[species_str]
            )

            if is_sub:
                sub_iface_equiv = np.array(
                    self._orthogonal_substrate_structure.site_properties[
                        "interface_equivalent"
                    ]
                )
                sub_site_ind = np.where(sub_iface_equiv == site_index)[0][0]
                self._non_orthogonal_substrate_structure[
                    sub_site_ind
                ].species = Element(species_mapping[species_str])
                self._orthogonal_substrate_structure[
                    sub_site_ind
                ].species = Element(species_mapping[species_str])
            else:
                film_iface_equiv = np.array(
                    self._orthogonal_film_structure.site_properties[
                        "interface_equivalent"
                    ]
                )
                film_site_ind = np.where(film_iface_equiv == site_index)[0][0]
                self._non_orthogonal_film_structure[
                    film_site_ind
                ].species = Element(species_mapping[species_str])
                self._orthogonal_film_structure[
                    film_site_ind
                ].species = Element(species_mapping[species_str])
        else:
            raise ValueError(
                f"Species: {species_str} is not is species mapping"
            )

    @property
    def area(self) -> float:
        """
        Cross section area of the interface in Angstroms^2

        Examples:
            >>> interface.area
            >>> 205.123456

        Returns:
            Cross-section area in Angstroms^2
        """
        return self.match.area

    @property
    def _structure_volume(self) -> float:
        matrix = deepcopy(self._orthogonal_structure.lattice.matrix)
        vac_matrix = np.vstack(
            [
                matrix[:2],
                self.vacuum * (matrix[-1] / np.linalg.norm(matrix[-1])),
            ]
        )

        total_volume = np.abs(np.linalg.det(matrix))
        vacuum_volume = np.abs(np.linalg.det(vac_matrix))

        return total_volume - vacuum_volume

    @property
    def substrate_basis(self) -> np.ndarray:
        """
        Returns the miller indices of the basis vectors of the substrate supercell

        Examples:
            >>> interface.substrate_basis
            >>> [[3 1 0]
            ...  [-1 3 0]
            ...  [0 0 1]]

        Returns:
            (3, 3) numpy array containing the miller indices of each lattice vector
        """
        return self._substrate_supercell_uvw

    @property
    def substrate_a(self) -> np.ndarray:
        """
        Returns the miller indices of the a basis vector of the substrate supercell

        Examples:
            >>> interface.substrate_a
            >>> [3 1 0]

        Returns:
            (3,) numpy array containing the miller indices of the a lattice vector
        """
        return self._substrate_supercell_uvw[0]

    @property
    def substrate_b(self) -> np.ndarray:
        """
        Returns the miller indices of the b basis vector of the substrate supercell

        Examples:
            >>> interface.substrate_b
            >>> [-1 3 0]

        Returns:
            (3,) numpy array containing the miller indices of the b lattice vector
        """
        return self._substrate_supercell_uvw[1]

    @property
    def film_basis(self) -> np.ndarray:
        """
        Returns the miller indices of the basis vectors of the film supercell

        Examples:
            >>> interface.film_basis
            >>> [[1 -1 0]
            ...  [0 1 0]
            ...  [0 0 1]]

        Returns:
            (3, 3) numpy array containing the miller indices of each lattice vector
        """
        return self._film_supercell_uvw

    @property
    def film_a(self) -> np.ndarray:
        """
        Returns the miller indices of the a basis vector of the film supercell

        Examples:
            >>> interface.film_a
            >>> [1 -1 0]

        Returns:
            (3,) numpy array containing the miller indices of the a lattice vector
        """
        return self._film_supercell_uvw[0]

    @property
    def film_b(self) -> np.ndarray:
        """
        Returns the miller indices of the a basis vector of the film supercell

        Examples:
            >>> interface.film_b
            >>> [0 1 0]

        Returns:
            (3,) numpy array containing the miller indices of the b lattice vector
        """
        return self._film_supercell_uvw[1]

    def __str__(self):
        fm = self.film.miller_index
        sm = self.substrate.miller_index
        film_str = f"{self.film.formula}({fm[0]} {fm[1]} {fm[2]})"
        sub_str = f"{self.substrate.formula}({sm[0]} {sm[1]} {sm[2]})"
        s_uvw = self._substrate_supercell_uvw
        s_sf = self._substrate_supercell_scale_factors
        f_uvw = self._film_supercell_uvw
        f_sf = self._film_supercell_scale_factors
        match_a_film = (
            f"{f_sf[0]}*[{f_uvw[0][0]:2d} {f_uvw[0][1]:2d} {f_uvw[0][2]:2d}]"
        )
        match_a_sub = (
            f"{s_sf[0]}*[{s_uvw[0][0]:2d} {s_uvw[0][1]:2d} {s_uvw[0][2]:2d}]"
        )
        match_b_film = (
            f"{f_sf[1]}*[{f_uvw[1][0]:2d} {f_uvw[1][1]:2d} {f_uvw[1][2]:2d}]"
        )
        match_b_sub = (
            f"{s_sf[1]}*[{s_uvw[1][0]:2d} {s_uvw[1][1]:2d} {s_uvw[1][2]:2d}]"
        )
        return_info = [
            "Film: " + film_str,
            "Substrate: " + sub_str,
            "Epitaxial Match Along \\vec{a} (film || sub): "
            + f"({match_a_film} || {match_a_sub})",
            "Epitaxial Match Along \\vec{b} (film || sub): "
            + f"({match_b_film} || {match_b_sub})",
            "Strain (%): " + f"{100*self.match.strain:.3f}",
            "Cross Section Area (Ang^2): " + f"{self.area:.3f}",
        ]
        return_str = "\n".join(return_info)

        return return_str

    def _load_relaxed_structure(
        self, relaxed_structure_file: str
    ) -> np.ndarray:
        with open(relaxed_structure_file, "r") as f:
            poscar_str = f.read().split("\n")

        desc_str = poscar_str[0].split("|")

        layers = desc_str[0].split("=")[1].split(",")
        termination_index = desc_str[1].split("=")[1].split(",")
        ortho = bool(int(desc_str[2].split("=")[1]))
        d_int = desc_str[3].split("=")[1]
        layers_to_relax = desc_str[4].split("=")[1].split(",")

        film_layers = int(layers[0])
        sub_layers = int(layers[1])

        film_termination_index = int(termination_index[0])
        sub_termination_index = int(termination_index[1])

        N_film_layers_to_relax = int(layers_to_relax[0])
        N_sub_layers_to_relax = int(layers_to_relax[1])

        if (
            d_int == f"{self.interfacial_distance:.3f}"
            and film_termination_index == self.film.termination_index
            and sub_termination_index == self.substrate.termination_index
        ):
            relaxed_structure = Structure.from_file(relaxed_structure_file)

            if ortho:
                unrelaxed_structure = self._orthogonal_structure.copy()
            else:
                unrelaxed_structure = self._non_orthogonal_structure.copy()

            unrelaxed_structure.add_site_property(
                "orig_ind", list(range(len(unrelaxed_structure)))
            )

            unrelaxed_hydrogen_inds = np.where(
                np.array(unrelaxed_structure.atomic_numbers) == 1
            )[0]

            unrelaxed_structure.remove_sites(unrelaxed_hydrogen_inds)

            unrelaxed_structure.add_site_property(
                "orig_ind", list(range(len(unrelaxed_structure)))
            )

            is_negative = np.linalg.det(unrelaxed_structure.lattice.matrix) < 0

            if is_negative:
                relaxed_structure = Structure(
                    lattice=Lattice(relaxed_structure.lattice.matrix * -1),
                    species=relaxed_structure.species,
                    coords=relaxed_structure.frac_coords,
                )

            hydrogen_inds = np.where(
                np.array(relaxed_structure.atomic_numbers) == 1
            )[0]

            relaxed_structure.remove_sites(hydrogen_inds)

            relaxation_shifts = np.zeros((len(unrelaxed_structure), 3))

            is_film_full = np.array(
                unrelaxed_structure.site_properties["is_film"]
            )
            is_sub_full = np.array(
                unrelaxed_structure.site_properties["is_sub"]
            )
            layer_index_full = np.array(
                unrelaxed_structure.site_properties["layer_index"]
            )
            sub_to_delete = np.logical_and(
                is_sub_full,
                layer_index_full < self.substrate.layers - sub_layers,
            )

            film_to_delete = np.logical_and(
                is_film_full, layer_index_full >= film_layers
            )

            to_delete = np.where(np.logical_or(sub_to_delete, film_to_delete))[
                0
            ]

            unrelaxed_structure.remove_sites(to_delete)

            is_film_small = np.array(
                unrelaxed_structure.site_properties["is_film"]
            )
            is_sub_small = np.array(
                unrelaxed_structure.site_properties["is_sub"]
            )
            layer_index_small = np.array(
                unrelaxed_structure.site_properties["layer_index"]
            )

            film_layers_to_relax = np.arange(N_film_layers_to_relax)

            sub_layers_to_relax = np.arange(
                self.substrate.layers - N_sub_layers_to_relax,
                self.substrate.layers,
            )

            film_to_relax = np.logical_and(
                is_film_small, np.isin(layer_index_small, film_layers_to_relax)
            )
            sub_to_relax = np.logical_and(
                is_sub_small, np.isin(layer_index_small, sub_layers_to_relax)
            )

            relaxed_inds = np.where(
                np.logical_or(film_to_relax, sub_to_relax)
            )[0]

            periodic_shifts = np.array(
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
            ).dot(unrelaxed_structure.lattice.matrix)

            ref_ind = np.min(np.where(is_sub_small)[0])
            unrelaxed_ref = unrelaxed_structure[ref_ind].coords
            relaxed_ref = relaxed_structure[ref_ind].coords

            for i in relaxed_inds:
                init_ind = unrelaxed_structure[i].properties["orig_ind"]
                relaxed_coords = relaxed_structure[i].coords
                relaxed_coords -= relaxed_ref
                unrelaxed_coords = unrelaxed_structure[i].coords
                unrelaxed_coords -= unrelaxed_ref

                all_relaxed_coords = periodic_shifts + relaxed_coords
                dists = np.linalg.norm(
                    all_relaxed_coords - unrelaxed_coords, axis=1
                )
                center_ind = np.argmin(dists)
                bond = all_relaxed_coords[center_ind] - unrelaxed_coords
                relaxation_shifts[init_ind] = bond

            return relaxation_shifts
        else:
            raise ValueError(
                "The surface terminations and interfacial distances must be the same"
            )

    def relax_interface(self, relaxed_structure_file: str) -> None:
        """
        This function will shift the positions of the atoms near the interface coorresponding to the
        atomic positions from a relaxed interface. This especially usefull when running DFT on large interface
        structures because the atomic positions can be relaxed using an interface with less layers, and
        then the relax positions can be applied to a much larger interface for a static DFT calculation.

        Examples:
            >>> interface.relax_interface(relax_structure_file="CONTCAR")

        Args:
            relaxed_structure_file: File path to the relax structure (CONTCAR/POSCAR for now)
        """
        relaxation_shifts = self._load_relaxed_structure(
            relaxed_structure_file
        )
        init_ortho_structure = self._orthogonal_structure
        non_hydrogen_inds = np.array(init_ortho_structure.atomic_numbers) != 1
        new_coords = init_ortho_structure.cart_coords
        new_coords[non_hydrogen_inds] += relaxation_shifts
        relaxed_ortho_structure = Structure(
            lattice=init_ortho_structure.lattice,
            species=init_ortho_structure.species,
            coords=new_coords,
            to_unit_cell=True,
            coords_are_cartesian=True,
            site_properties=init_ortho_structure.site_properties,
        )
        relaxed_ortho_atoms = utils.get_atoms(relaxed_ortho_structure)
        (
            relaxed_ortho_film_structure,
            relaxed_ortho_film_atoms,
            relaxed_ortho_sub_structure,
            relaxed_ortho_sub_atoms,
        ) = self._get_film_and_substrate_parts(relaxed_ortho_structure)

        init_non_ortho_structure = self._non_orthogonal_structure
        non_hydrogen_inds = (
            np.array(init_non_ortho_structure.atomic_numbers) != 1
        )
        new_coords = init_non_ortho_structure.cart_coords
        new_coords[non_hydrogen_inds] += relaxation_shifts
        relaxed_non_ortho_structure = Structure(
            lattice=init_non_ortho_structure.lattice,
            species=init_non_ortho_structure.species,
            coords=new_coords,
            to_unit_cell=True,
            coords_are_cartesian=True,
            site_properties=init_non_ortho_structure.site_properties,
        )
        relaxed_non_ortho_atoms = utils.get_atoms(relaxed_non_ortho_structure)
        (
            relaxed_non_ortho_film_structure,
            relaxed_non_ortho_film_atoms,
            relaxed_non_ortho_sub_structure,
            relaxed_non_ortho_sub_atoms,
        ) = self._get_film_and_substrate_parts(relaxed_non_ortho_structure)

        self._orthogonal_structure = relaxed_ortho_structure
        self._orthogonal_film_structure = relaxed_ortho_film_structure
        self._orthogonal_substrate_structure = relaxed_ortho_sub_structure
        self._non_orthogonal_structure = relaxed_non_ortho_structure
        self._non_orthogonal_film_structure = relaxed_non_ortho_film_structure
        self._non_orthogonal_substrate_structure = (
            relaxed_non_ortho_sub_structure
        )
        self._orthogonal_atoms = relaxed_ortho_atoms
        self._orthogonal_film_atoms = relaxed_ortho_film_atoms
        self._orthogonal_substrate_atoms = relaxed_ortho_sub_atoms
        self._non_orthogonal_atoms = relaxed_non_ortho_atoms
        self._non_orthogonal_film_atoms = relaxed_non_ortho_film_atoms
        self._non_orthogonal_substrate_atoms = relaxed_non_ortho_sub_atoms

    def write_file(
        self,
        output: str = "POSCAR_interface",
        orthogonal: bool = True,
        relax: bool = False,
        film_layers_to_relax: int = 1,
        substrate_layers_to_relax: int = 1,
        atomic_layers: bool = False,
        relax_z_only: bool = False,
    ) -> None:
        """
        Write the POSCAR of the interface

        Args:
            output: File path of the output POSCAR
            orthogonal: Determines of the orthogonal structure is written
            relax: Determines if selective dynamics is applied to the atoms at the interface
            film_layers_to_relax: Number of unit cell layers near the interface to relax
            substrate_layers_to_relax: Number of unit cell layers near the interface to relax
            atomic_layers: Determines if the number of layer is atomic layers or unit cell layers
            relax_z_only: Determines if the relaxation is in the z-direction only
        """
        if orthogonal:
            slab = self._orthogonal_structure
        else:
            slab = self._non_orthogonal_structure

        comment = "|".join(
            [
                f"L={self.film.layers},{self.substrate.layers}",
                f"T={self.film.termination_index},{self.substrate.termination_index}",
                f"O={int(orthogonal)}",
                f"d={self.interfacial_distance:.3f}",
            ]
        )

        if relax:
            comment += "|" + "|".join(
                [
                    f"R={film_layers_to_relax},{substrate_layers_to_relax}",
                ]
            )
            film_layers = np.arange(film_layers_to_relax)

            if atomic_layers:
                layer_key = "atomic_layer_index"
                sub_layers = np.arange(
                    self.substrate.atomic_layers - substrate_layers_to_relax,
                    self.substrate.atomic_layers,
                )
            else:
                layer_key = "layer_index"
                sub_layers = np.arange(
                    self.substrate.layers - substrate_layers_to_relax,
                    self.substrate.layers,
                )

            layer_index = np.array(slab.site_properties[layer_key])
            is_sub = np.array(slab.site_properties["is_sub"])
            is_film = np.array(slab.site_properties["is_film"])
            film_to_relax = np.logical_and(
                is_film, np.isin(layer_index, film_layers)
            )
            sub_to_relax = np.logical_and(
                is_sub, np.isin(layer_index, sub_layers)
            )

            to_relax = np.repeat(
                np.logical_or(sub_to_relax, film_to_relax).reshape(-1, 1),
                repeats=3,
                axis=1,
            )

            if relax_z_only:
                to_relax[:, :2] = False

        comment += "|" + "|".join(
            [
                f"a={self._a_shift:.4f}",
                f"b={self._b_shift:.4f}",
            ]
        )

        if not self.substrate._passivated and not self.film._passivated:
            poscar = Poscar(slab, comment=comment)

            if relax:
                poscar.selective_dynamics = to_relax

            poscar_str = poscar.get_str()

        else:
            syms = [site.specie.symbol for site in slab]

            syms = []
            for site in slab:
                if site.specie.symbol == "H":
                    if hasattr(site.specie, "oxi_state"):
                        oxi = site.specie.oxi_state

                        if oxi < 1.0 and oxi != 0.5:
                            H_str = "H" + f"{oxi:.2f}"[1:]
                        elif oxi == 0.5:
                            H_str = "H.5"
                        elif oxi > 1.0 and oxi != 1.5:
                            H_str = "H" + f"{oxi:.2f}"
                        elif oxi == 1.5:
                            H_str = "H1.5"
                        else:
                            H_str = "H"

                        syms.append(H_str)
                else:
                    syms.append(site.specie.symbol)

            comp_list = [(a[0], len(list(a[1]))) for a in groupby(syms)]
            atom_types, n_atoms = zip(*comp_list)

            new_atom_types = []
            for atom in atom_types:
                if "H" == atom[0] and atom not in ["Hf", "Hs", "Hg", "He"]:
                    new_atom_types.append("H")
                else:
                    new_atom_types.append(atom)

            comment += "|potcar=" + " ".join(atom_types)

            poscar = Poscar(slab, comment=comment)

            if relax:
                poscar.selective_dynamics = to_relax

            poscar_str = poscar.get_str().split("\n")
            poscar_str[5] = " ".join(new_atom_types)
            poscar_str[6] = " ".join(list(map(str, n_atoms)))
            poscar_str = "\n".join(poscar_str)

        with open(output, "w") as f:
            f.write(poscar_str)

    def _shift_film(
        self, interface: Structure, shift: Iterable, fractional: bool
    ) -> Tuple[Structure, Atoms, Structure, Atoms]:
        shifted_interface_structure = interface.copy()
        film_ind = np.where(
            shifted_interface_structure.site_properties["is_film"]
        )[0]
        shifted_interface_structure.translate_sites(
            indices=film_ind,
            vector=shift,
            frac_coords=fractional,
            to_unit_cell=True,
        )
        shifted_interface_atoms = utils.get_atoms(shifted_interface_structure)
        (
            shifted_film_structure,
            shifted_film_atoms,
            _,
            _,
        ) = self._get_film_and_substrate_parts(shifted_interface_structure)

        return (
            shifted_interface_structure,
            shifted_interface_atoms,
            shifted_film_structure,
            shifted_film_atoms,
        )

    def set_interfacial_distance(self, interfacial_distance: float) -> None:
        """
        Sets a new interfacial distance for the interface by shifting the film in the z-direction

        Examples:
            >>> interface.set_interfacial_distance(interfacial_distance=2.0)

        Args:
            interfacial_distance: New interfacial distance for the interface
        """
        shift = np.array(
            [0.0, 0.0, interfacial_distance - self.interfacial_distance]
        )
        self.interfacial_distance = interfacial_distance
        (
            self._orthogonal_structure,
            self._orthogonal_atoms,
            self._orthogonal_film_structure,
            self._orthogonal_film_atoms,
        ) = self._shift_film(
            interface=self._orthogonal_structure,
            shift=shift,
            fractional=False,
        )
        (
            self._non_orthogonal_structure,
            self._non_orthogonal_atoms,
            self._non_orthogonal_film_structure,
            self._non_orthogonal_film_atoms,
        ) = self._shift_film(
            interface=self._non_orthogonal_structure,
            shift=shift,
            fractional=False,
        )

    def shift_film_inplane(
        self,
        x_shift: float,
        y_shift: float,
        fractional: bool = False,
    ) -> None:
        """
        Shifts the film in-place over the substrate within the plane of the interface by a given shift vector.

        Examples:
            Shift using fractional coordinates:
            >>> interface.shift_film(shift=[0.5, 0.25], fractional=True)

            Shift using cartesian coordinates:
            >>> interface.shift_film(shift=[4.5, 0.0], fractional=False)

        Args:
            x_shift: Shift in the x or a-vector directions depending on if fractional=True
            y_shift: Shift in the y or b-vector directions depending on if fractional=True
            fractional: Determines if the shift is defined in fractional coordinates
        """
        shift_array = np.array([x_shift, y_shift, 0.0])

        if fractional:
            frac_shift = shift_array
        else:
            frac_shift = (
                self._orthogonal_structure.lattice.get_fractional_coords(
                    shift_array
                )
            )

        self._a_shift += shift_array[0]
        self._b_shift += shift_array[1]

        (
            self._orthogonal_structure,
            self._orthogonal_atoms,
            self._orthogonal_film_structure,
            self._orthogonal_film_atoms,
        ) = self._shift_film(
            interface=self._orthogonal_structure,
            shift=frac_shift,
            fractional=True,
        )
        (
            self._non_orthogonal_structure,
            self._non_orthogonal_atoms,
            self._non_orthogonal_film_structure,
            self._non_orthogonal_film_atoms,
        ) = self._shift_film(
            interface=self._non_orthogonal_structure,
            shift=frac_shift,
            fractional=True,
        )

    def _create_supercell(
        self, substrate: bool = True
    ) -> Tuple[Structure, Structure, np.ndarray, np.ndarray]:
        if substrate:
            matrix = self.match.substrate_sl_transform

            if issubclass(type(self.substrate), BaseSurface):
                supercell = (
                    self.substrate._non_orthogonal_slab_structure.copy()
                )
                obs_supercell = self.substrate.oriented_bulk_structure.copy()
            elif type(self.substrate) is Interface:
                supercell = self.substrate._non_orthogonal_structure.copy()
                obs_supercell = None

                layer_keys = ["layer_index", "atomic_layer_index"]

                for layer_key in layer_keys:
                    layer_index = np.array(
                        supercell.site_properties[layer_key]
                    )
                    not_hydrogen = layer_index != -1
                    is_film = supercell.site_properties["is_film"]
                    is_sub = supercell.site_properties["is_sub"]
                    layer_index[(is_film & not_hydrogen)] += (
                        layer_index[is_sub].max() + 1
                    )
                    supercell.add_site_property(layer_key, layer_index)

            basis = self.substrate.crystallographic_basis
        else:
            matrix = self.match.film_sl_transform

            if issubclass(type(self.film), BaseSurface):
                supercell = self.film._non_orthogonal_slab_structure.copy()
                obs_supercell = self.film.oriented_bulk_structure.copy()
            elif type(self.film) is Interface:
                supercell = self.film._non_orthogonal_structure.copy()
                obs_supercell = None

                layer_keys = ["layer_index", "atomic_layer_index"]

                for layer_key in layer_keys:
                    layer_index = np.array(
                        supercell.site_properties[layer_key]
                    )
                    is_film = supercell.site_properties["is_film"]
                    is_sub = supercell.site_properties["is_sub"]
                    layer_index[is_film] += layer_index[is_sub].max() + 1
                    supercell.add_site_property(layer_key, layer_index)

            basis = self.film.crystallographic_basis

        supercell.make_supercell(scaling_matrix=matrix)

        if obs_supercell is not None:
            obs_supercell.make_supercell(scaling_matrix=matrix)

        uvw_supercell = matrix @ basis
        scale_factors = []
        for i, b in enumerate(uvw_supercell):
            scale = np.abs(reduce(utils._float_gcd, b))
            uvw_supercell[i] = uvw_supercell[i] / scale
            scale_factors.append(scale)

        return supercell, obs_supercell, uvw_supercell, scale_factors

    def _orient_supercell(
        self,
        supercell: Structure,
        substrate: bool,
    ) -> np.ndarray:
        if substrate:
            a_to_i = self.match.substrate_align_transform.T
        else:
            a_to_i = self.match.film_align_transform.T

        a_to_i_operation = SymmOp.from_rotation_and_translation(
            a_to_i, translation_vec=np.zeros(3)
        )

        if "molecules" in supercell.site_properties:
            utils.apply_op_to_mols(supercell, a_to_i_operation)

        supercell.apply_operation(a_to_i_operation)

        return a_to_i

    def _prepare_film(self) -> Structure:
        supercell_slab = self._film_supercell
        sc_matrix = supercell_slab.lattice.matrix
        sub_sc_matrix = self._strained_sub.lattice.matrix

        inplane_strain_transformation = (
            np.linalg.inv(sc_matrix[:2, :2]) @ sub_sc_matrix[:2, :2]
        )
        inplane_strained_matrix = (
            sc_matrix[:, :2] @ inplane_strain_transformation
        )

        strained_matrix = np.c_[inplane_strained_matrix, sc_matrix[:, -1]]

        init_volume = supercell_slab.volume
        strain_volume = np.abs(np.linalg.det(strained_matrix))
        scale_factor = init_volume / strain_volume

        # Maintain constant volume
        strained_matrix[-1, -1] *= scale_factor
        const_volume_strain_transformation = (
            np.linalg.inv(sc_matrix) @ strained_matrix
        )
        strained_film = utils.apply_strain_matrix(
            structure=supercell_slab,
            strain_matrix=const_volume_strain_transformation,
        )

        sub_non_orth_c_vec = self._strained_sub.lattice.matrix[-1]
        sub_non_orth_c_norm = sub_non_orth_c_vec / np.linalg.norm(
            sub_non_orth_c_vec
        )

        norm = self.film.surface_normal
        proj = np.dot(norm, sub_non_orth_c_norm)
        scale = strained_film.lattice.c / proj

        new_c_matrix = np.vstack(
            [sub_sc_matrix[:2], sub_non_orth_c_norm * scale]
        )

        oriented_film = Structure(
            lattice=Lattice(new_c_matrix),
            species=strained_film.species,
            coords=strained_film.cart_coords,
            coords_are_cartesian=True,
            to_unit_cell=True,
            site_properties=strained_film.site_properties,
        )

        return oriented_film, const_volume_strain_transformation

    def _prepare_substrate(self) -> Structure:
        supercell_slab = self._substrate_supercell
        sc_matrix = supercell_slab.lattice.matrix
        avg_sc_matrix = self._avg_sc_inplane_lattice

        inplane_strain_transformation = (
            np.linalg.inv(sc_matrix[:2, :2]) @ avg_sc_matrix[:2, :2]
        )

        inplane_strained_matrix = (
            sc_matrix[:, :2] @ inplane_strain_transformation
        )

        strained_matrix = np.c_[inplane_strained_matrix, sc_matrix[:, -1]]

        init_volume = supercell_slab.volume
        strain_volume = np.abs(np.linalg.det(strained_matrix))
        scale_factor = init_volume / strain_volume

        # Maintain constant volume
        strained_matrix[-1, -1] *= scale_factor
        const_volume_strain_transformation = (
            np.linalg.inv(sc_matrix) @ strained_matrix
        )
        strained_substrate = utils.apply_strain_matrix(
            structure=supercell_slab,
            strain_matrix=const_volume_strain_transformation,
        )

        return strained_substrate, const_volume_strain_transformation

    def _stack_interface(
        self,
    ) -> Tuple[
        np.ndarray,
        Structure,
        Structure,
        Structure,
        Atoms,
        Atoms,
        Atoms,
        Structure,
        Structure,
        Structure,
        Atoms,
        Atoms,
        Atoms,
    ]:
        # Get the strained substrate and film
        strained_sub = self._strained_sub
        strained_film = self._strained_film

        if "molecules" in strained_sub.site_properties:
            strained_sub = utils.add_molecules(strained_sub)

        if "molecules" in strained_film.site_properties:
            strained_film = utils.add_molecules(strained_film)

        # Get the oriented bulk structure of the substrate
        oriented_bulk_c = self.substrate.oriented_bulk_structure.lattice.c

        # Get the normalized projection of the substrate c-vector onto the normal vector,
        # This is used to determine the length of the non-orthogonal c-vector in order to get
        # the correct vacuum size.
        c_norm_proj = self.substrate.layer_thickness / oriented_bulk_c

        # Get the substrate matrix and c-vector
        sub_matrix = strained_sub.lattice.matrix
        sub_c = deepcopy(sub_matrix[-1])

        # Get the fractional and cartesian coordinates of the substrate and film
        strained_sub_coords = deepcopy(strained_sub.cart_coords)
        strained_film_coords = deepcopy(strained_film.cart_coords)
        strained_sub_frac_coords = deepcopy(strained_sub.frac_coords)
        strained_film_frac_coords = deepcopy(strained_film.frac_coords)

        # Find the min and max coordinates of the substrate and film
        min_sub_coords = np.min(strained_sub_frac_coords[:, -1])
        max_sub_coords = np.max(strained_sub_frac_coords[:, -1])
        min_film_coords = np.min(strained_film_frac_coords[:, -1])
        max_film_coords = np.max(strained_film_frac_coords[:, -1])

        # Get the lengths of the c-vetors of the substrate and film
        sub_c_len = strained_sub.lattice.c
        film_c_len = strained_film.lattice.c

        # Find the total length of the interface structure including the interfacial distance
        interface_structure_len = np.sum(
            [
                (max_sub_coords - min_sub_coords) * sub_c_len,
                (max_film_coords - min_film_coords) * film_c_len,
                self.interfacial_distance / c_norm_proj,
            ]
        )

        # Find the length of the vacuum region in the non-orthogonal basis
        interface_vacuum_len = self.vacuum / c_norm_proj

        # The total length of the interface c-vector should be the length of the structure + length of the vacuum
        # This will get changed in the next line to be exactly an integer multiple of the
        # oriented bulk cell of the substrate
        init_interface_c_len = interface_structure_len + interface_vacuum_len

        # Find the closest integer multiple of the substrate oriented bulk c-vector length
        n_unit_cell = int(np.ceil(init_interface_c_len / oriented_bulk_c))

        # Make the new interface c-vector length an integer multiple of the oriented bulk c-vector
        interface_c_len = oriented_bulk_c * n_unit_cell

        # Create the transformation matrix from the primtive bulk structure to the interface unit cell
        # this is only needed for band unfolding purposes
        sub_M = self.substrate.bulk_transformation_matrix
        layer_M = np.eye(3).astype(int)
        layer_M[-1, -1] = n_unit_cell
        interface_M = layer_M @ self.match.substrate_sl_transform @ sub_M

        # Create the new interface lattice vectors
        interface_matrix = np.vstack(
            [sub_matrix[:2], interface_c_len * (sub_c / sub_c_len)]
        )
        interface_lattice = Lattice(matrix=interface_matrix)

        # Convert the interfacial distance into fractional coordinated because they are easier to work with
        frac_int_distance_shift = np.array(
            [0, 0, self.interfacial_distance]
        ).dot(interface_lattice.inv_matrix)

        interface_inv_matrix = interface_lattice.inv_matrix

        # Convert the substrate cartesian coordinates into the interface fractional coordinates
        # and shift the bottom atom c-position to zero
        sub_interface_coords = strained_sub_coords.dot(interface_inv_matrix)
        sub_interface_coords[:, -1] -= sub_interface_coords[:, -1].min()

        # Convert the film cartesian coordinates into the interface fractional coordinates
        # and shift the bottom atom c-position to the top substrate c-position + interfacial distance
        film_interface_coords = strained_film_coords.dot(interface_inv_matrix)
        film_interface_coords[:, -1] -= film_interface_coords[:, -1].min()
        film_interface_coords[:, -1] += sub_interface_coords[:, -1].max()
        film_interface_coords += frac_int_distance_shift

        # Combine the coodinates, species, and site_properties to make the interface Structure
        interface_coords = np.r_[sub_interface_coords, film_interface_coords]
        interface_species = strained_sub.species + strained_film.species
        interface_site_properties = {
            key: strained_sub.site_properties[key]
            + strained_film.site_properties[key]
            for key in strained_sub.site_properties
            if key in strained_sub.site_properties
            and key in strained_film.site_properties
        }
        interface_site_properties["is_sub"] = np.array(
            [True] * len(strained_sub) + [False] * len(strained_film)
        )
        interface_site_properties["is_film"] = np.array(
            [False] * len(strained_sub) + [True] * len(strained_film)
        )

        # Create the non-orthogonalized interface structure
        non_ortho_interface_struc = Structure(
            lattice=interface_lattice,
            species=interface_species,
            coords=interface_coords,
            to_unit_cell=True,
            coords_are_cartesian=False,
            site_properties=interface_site_properties,
        )
        non_ortho_interface_struc.sort()

        non_ortho_interface_struc.add_site_property(
            "interface_equivalent", list(range(len(non_ortho_interface_struc)))
        )

        if self.center:
            # Get the new vacuum length, needed for shifting
            vacuum_len = interface_c_len - interface_structure_len

            # Find the fractional coordinates of shifting the structure up by half the amount of vacuum cells
            center_shift = np.ceil(vacuum_len / oriented_bulk_c) // 2
            center_shift *= oriented_bulk_c / interface_c_len

            # Center the structure in the vacuum
            non_ortho_interface_struc.translate_sites(
                indices=range(len(non_ortho_interface_struc)),
                vector=[0.0, 0.0, center_shift],
                frac_coords=True,
                to_unit_cell=True,
            )

        # Get the frac coords of the non-orthogonalized interface
        frac_coords = non_ortho_interface_struc.frac_coords

        # Find the max c-coord of the substrate
        # This is used to shift the x-y positions of the interface structure so the top atom of the substrate
        # is located at x=0, y=0. This will have no effect of the properties of the interface since all the
        # atoms are shifted, it is more of a visual thing to make the interfaces look nice.
        is_sub = np.array(non_ortho_interface_struc.site_properties["is_sub"])
        sub_frac_coords = frac_coords[is_sub]
        max_c = np.max(sub_frac_coords[:, -1])

        # Find the xy-shift in cartesian coordinates
        cart_shift = np.array([0.0, 0.0, max_c]).dot(
            non_ortho_interface_struc.lattice.matrix
        )
        cart_shift[-1] = 0.0

        # Get the projection of the non-orthogonal c-vector onto the surface normal
        proj_c = np.dot(
            self.substrate.surface_normal,
            non_ortho_interface_struc.lattice.matrix[-1],
        )

        # Get the orthogonalized c-vector of the interface (this conserves the vacuum, but breaks symmetries)
        ortho_c = self.substrate.surface_normal * proj_c

        # Create the orthogonalized lattice vectors
        new_matrix = np.vstack(
            [
                non_ortho_interface_struc.lattice.matrix[:2],
                ortho_c,
            ]
        )

        # Create the orthogonalized structure
        ortho_interface_struc = Structure(
            lattice=Lattice(new_matrix),
            species=non_ortho_interface_struc.species,
            coords=non_ortho_interface_struc.cart_coords,
            site_properties=non_ortho_interface_struc.site_properties,
            to_unit_cell=True,
            coords_are_cartesian=True,
        )

        # Shift the structure so the top substrate atom's x and y postions are zero, similar to the non-orthogonalized structure
        ortho_interface_struc.translate_sites(
            indices=range(len(ortho_interface_struc)),
            vector=-cart_shift,
            frac_coords=False,
            to_unit_cell=True,
        )

        # The next step is used extract on the film and substrate portions of the interface
        # These can be used for charge transfer calculation
        (
            ortho_film_structure,
            ortho_film_atoms,
            ortho_sub_structure,
            ortho_sub_atoms,
        ) = self._get_film_and_substrate_parts(ortho_interface_struc)
        (
            non_ortho_film_structure,
            non_ortho_film_atoms,
            non_ortho_sub_structure,
            non_ortho_sub_atoms,
        ) = self._get_film_and_substrate_parts(non_ortho_interface_struc)

        non_ortho_interface_atoms = utils.get_atoms(non_ortho_interface_struc)
        ortho_interface_atoms = utils.get_atoms(ortho_interface_struc)

        return (
            interface_M,
            non_ortho_interface_struc,
            non_ortho_sub_structure,
            non_ortho_film_structure,
            non_ortho_interface_atoms,
            non_ortho_sub_atoms,
            non_ortho_film_atoms,
            ortho_interface_struc,
            ortho_sub_structure,
            ortho_film_structure,
            ortho_interface_atoms,
            ortho_sub_atoms,
            ortho_film_atoms,
        )

    def _get_film_and_substrate_parts(
        self, interface: Structure
    ) -> Tuple[Structure, Atoms, Structure, Atoms]:
        film_inds = np.where(interface.site_properties["is_film"])[0]
        sub_inds = np.where(interface.site_properties["is_sub"])[0]

        film_structure = interface.copy()
        film_structure.remove_sites(sub_inds)
        film_atoms = utils.get_atoms(film_structure)

        sub_structure = interface.copy()
        sub_structure.remove_sites(film_inds)
        sub_atoms = utils.get_atoms(sub_structure)

        return film_structure, film_atoms, sub_structure, sub_atoms

    @property
    def _metallic_elements(self):
        elements_list = np.array(
            [
                "Li",
                "Be",
                "Na",
                "Mg",
                "Al",
                "K",
                "Ca",
                "Sc",
                "Ti",
                "V",
                "Cr",
                "Mn",
                "Fe",
                "Co",
                "Ni",
                "Cu",
                "Zn",
                "Ga",
                "Rb",
                "Sr",
                "Y",
                "Zr",
                "Nb",
                "Mo",
                "Tc",
                "Ru",
                "Rh",
                "Pd",
                "Ag",
                "Cd",
                "In",
                "Sn",
                "Cs",
                "Ba",
                "La",
                "Ce",
                "Pr",
                "Nd",
                "Pm",
                "Sm",
                "Eu",
                "Gd",
                "Tb",
                "Dy",
                "Ho",
                "Er",
                "Tm",
                "Yb",
                "Lu",
                "Hf",
                "Ta",
                "W",
                "Re",
                "Os",
                "Ir",
                "Pt",
                "Au",
                "Hg",
                "Tl",
                "Pb",
                "Bi",
                "Rn",
                "Fr",
                "Ra",
                "Ac",
                "Th",
                "Pa",
                "U",
                "Np",
                "Pu",
                "Am",
                "Cm",
                "Bk",
                "Cf",
                "Es",
                "Fm",
                "Md",
                "No",
                "Lr",
                "Rf",
                "Db",
                "Sg",
                "Bh",
                "Hs",
                "Mt",
                "Ds ",
                "Rg ",
                "Cn ",
                "Nh",
                "Fl",
                "Mc",
                "Lv",
            ]
        )
        return elements_list

    def _get_radii(self):
        sub_species = np.unique(
            np.array(self.substrate.bulk_structure.species, dtype=str)
        )
        film_species = np.unique(
            np.array(self.film.bulk_structure.species, dtype=str)
        )

        sub_elements = [Element(s) for s in sub_species]
        film_elements = [Element(f) for f in film_species]

        sub_metal = np.isin(sub_species, self._metallic_elements)
        film_metal = np.isin(film_species, self._metallic_elements)

        if sub_metal.all():
            sub_dict = {
                sub_species[i]: sub_elements[i].metallic_radius
                for i in range(len(sub_elements))
            }
        else:
            Xs = [e.X for e in sub_elements]
            X_diff = np.abs([c[0] - c[1] for c in combinations(Xs, 2)])
            if (X_diff >= 1.7).any():
                sub_dict = {
                    sub_species[i]: sub_elements[i].average_ionic_radius
                    for i in range(len(sub_elements))
                }
            else:
                sub_dict = {s: CovalentRadius.radius[s] for s in sub_species}

        if film_metal.all():
            film_dict = {
                film_species[i]: film_elements[i].metallic_radius
                for i in range(len(film_elements))
            }
        else:
            Xs = [e.X for e in film_elements]
            X_diff = np.abs([c[0] - c[1] for c in combinations(Xs, 2)])
            if (X_diff >= 1.7).any():
                film_dict = {
                    film_species[i]: film_elements[i].average_ionic_radius
                    for i in range(len(film_elements))
                }
            else:
                film_dict = {f: CovalentRadius.radius[f] for f in film_species}

        sub_dict.update(film_dict)

        return sub_dict

    def _generate_sc_for_interface_view(
        self, struc, transformation_matrix
    ) -> Tuple[Structure, np.ndarray]:
        plot_struc = Structure(
            lattice=struc.lattice,
            species=["H"],
            coords=np.zeros((1, 3)),
            to_unit_cell=True,
            coords_are_cartesian=True,
        )
        plot_struc.make_supercell(transformation_matrix)
        inv_matrix = plot_struc.lattice.inv_matrix

        return plot_struc, inv_matrix

    def _plot_interface_view(
        self,
        ax,
        zero_coord,
        supercell_shift,
        cell_vetices,
        slab_matrix,
        sc_inv_matrix,
        facecolor,
        edgecolor,
        is_film=False,
        strain=True,
    ) -> None:
        cart_coords = (
            zero_coord + supercell_shift + cell_vetices.dot(slab_matrix)
        )
        fc = np.round(cart_coords.dot(sc_inv_matrix), 3)
        if is_film:
            if strain:
                strain_matrix = self._strain_matrix
                strain_matrix[-1] = np.array([0, 0, 1])
                plot_coords = cart_coords @ strain_matrix
            else:
                plot_coords = cart_coords

            linewidth = 1.0
        else:
            plot_coords = cart_coords
            linewidth = 2.0

        center = np.round(
            np.mean(cart_coords[:-1], axis=0).dot(sc_inv_matrix),
            3,
        )
        center_in = np.logical_and(-0.0001 <= center[:2], center[:2] <= 1.0001)

        x_in = np.logical_and(fc[:, 0] > 0.0, fc[:, 0] < 1.0)
        y_in = np.logical_and(fc[:, 1] > 0.0, fc[:, 1] < 1.0)
        point_in = np.logical_and(x_in, y_in)

        if point_in.any() or center_in.all():
            poly = Polygon(
                xy=plot_coords[:, :2],
                closed=True,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
            )
            ax.add_patch(poly)

    def _get_image(
        self,
        zero_coord,
        supercell_shift,
        cell_vetices,
        slab_matrix,
        slab_inv_matrix,
        sc_inv_matrix,
    ) -> Union[None, np.ndarray]:
        cart_coords = (
            zero_coord + supercell_shift + cell_vetices.dot(slab_matrix)
        )
        fc = np.round(cart_coords.dot(sc_inv_matrix), 3)
        center = np.round(
            np.mean(cart_coords[:-1], axis=0).dot(sc_inv_matrix),
            3,
        )
        center_in = np.logical_and(-0.0001 <= center[:2], center[:2] <= 1.0001)

        x_in = np.logical_and(fc[:, 0] > 0.0, fc[:, 0] < 1.0)
        y_in = np.logical_and(fc[:, 1] > 0.0, fc[:, 1] < 1.0)
        point_in = np.logical_and(x_in, y_in)

        if point_in.any() or center_in.all():
            shifted_zero_coord = zero_coord + supercell_shift
            shifted_zero_frac = shifted_zero_coord.dot(slab_inv_matrix)

            return np.round(shifted_zero_frac).astype(int)
        else:
            return None

    def _get_oriented_cell_and_images(
        self, strain: bool = True
    ) -> List[np.ndarray]:
        sub_struc = self.substrate._orthogonal_slab_structure.copy()
        sub_a_to_i_op = SymmOp.from_rotation_and_translation(
            rotation_matrix=self._substrate_a_to_i, translation_vec=np.zeros(3)
        )
        sub_struc.apply_operation(sub_a_to_i_op)

        film_struc = self.film._orthogonal_slab_structure.copy()
        film_a_to_i_op = SymmOp.from_rotation_and_translation(
            rotation_matrix=self._film_a_to_i, translation_vec=np.zeros(3)
        )
        film_struc.apply_operation(film_a_to_i_op)

        if strain:
            film_struc = utils.apply_strain_matrix(
                structure=film_struc,
                strain_matrix=self._film_strain_matrix,
            )
            sub_struc = utils.apply_strain_matrix(
                structure=sub_struc,
                strain_matrix=self._substrate_strain_matrix,
            )

        sub_matrix = sub_struc.lattice.matrix
        film_matrix = film_struc.lattice.matrix

        prim_sub_inv_matrix = sub_struc.lattice.inv_matrix
        prim_film_inv_matrix = film_struc.lattice.inv_matrix

        sub_sc_matrix = deepcopy(self._substrate_supercell.lattice.matrix)
        film_sc_matrix = deepcopy(self._film_supercell.lattice.matrix)

        coords = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 0],
            ]
        )

        sc_shifts = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [-1, 0, 0],
                [0, -1, 0],
                [1, 1, 0],
                [-1, -1, 0],
                [1, -1, 0],
                [-1, 1, 0],
            ]
        )

        sub_sc_shifts = sc_shifts.dot(sub_sc_matrix)
        film_sc_shifts = sc_shifts.dot(film_sc_matrix)

        sub_struc, sub_inv_matrix = self._generate_sc_for_interface_view(
            struc=sub_struc,
            transformation_matrix=self.match.substrate_sl_transform,
        )

        film_struc, film_inv_matrix = self._generate_sc_for_interface_view(
            struc=film_struc,
            transformation_matrix=self.match.film_sl_transform,
        )

        sub_images = []

        for c in sub_struc.cart_coords:
            for shift in sub_sc_shifts:
                sub_image = self._get_image(
                    zero_coord=c,
                    supercell_shift=shift,
                    cell_vetices=coords,
                    slab_matrix=sub_matrix,
                    slab_inv_matrix=prim_sub_inv_matrix,
                    sc_inv_matrix=sub_inv_matrix,
                )

                if sub_image is not None:
                    sub_images.append(sub_image)

        film_images = []

        for c in film_struc.cart_coords:
            for shift in film_sc_shifts:
                film_image = self._get_image(
                    zero_coord=c,
                    supercell_shift=shift,
                    cell_vetices=coords,
                    slab_matrix=film_matrix,
                    slab_inv_matrix=prim_film_inv_matrix,
                    sc_inv_matrix=film_inv_matrix,
                )
                if film_image is not None:
                    film_images.append(film_image)

        return sub_matrix, sub_images, film_matrix, film_images

    def plot_interface(
        self,
        output: str = "interface_view.png",
        dpi: int = 400,
        film_color: str = "green",
        substrate_color: str = "orange",
        show_in_colab: bool = False,
    ) -> None:
        """
        This function will show the relative alignment of the film and substrate supercells by plotting the in-plane unit cells on top of each other

        Args:
            output: File path for the output image
            dpi: dpi (dots per inch) of the output image.
                Setting dpi=100 gives reasonably sized images when viewed in colab notebook
            film_color: Color to represent the film lattice vectors
            substrate_color: Color to represent the substrate lattice vectors
            show_in_colab: Determines if the matplotlib figure is closed or not after the plot if made.
                if show_in_colab=True the plot will show up after you run the cell in colab/jupyter notebook.
        """
        plot_match(
            match=self.match,
            padding=0.2,
            substrate_color=substrate_color,
            film_color=film_color,
            output=output,
            show_in_colab=show_in_colab,
        )
