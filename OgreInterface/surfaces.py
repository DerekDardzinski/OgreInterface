"""
This module will be used to construct the surfaces and interfaces used in this package.
"""
from OgreInterface import utils

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import Element, Species
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SymmOp
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pymatgen.analysis.local_env import CrystalNN

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from itertools import combinations, groupby
import numpy as np
import copy
from copy import deepcopy
from functools import reduce
from ase import Atoms


class Surface:
    """
    The Surface class is a container for surfaces generated with the SurfaceGenerator
    class and will be used as an input to the InterfaceGenerator class
    """

    def __init__(
        self,
        orthogonal_slab,
        non_orthogonal_slab,
        primitive_oriented_bulk,
        conventional_bulk,
        base_structure,
        transformation_matrix,
        miller_index,
        layers,
        vacuum,
        uvw_basis,
        point_group_operations,
        bottom_layer_dist,
        top_layer_dist,
    ):
        (
            self.orthogonal_slab_structure,
            self.orthogonal_slab_atoms,
        ) = self._get_atoms_and_struc(orthogonal_slab)
        (
            self.non_orthogonal_slab_structure,
            self.non_orthogonal_slab_atoms,
        ) = self._get_atoms_and_struc(non_orthogonal_slab)
        (
            self.primitive_oriented_bulk_structure,
            self.primitive_oriented_bulk_atoms,
        ) = self._get_atoms_and_struc(primitive_oriented_bulk)
        (
            self.conventional_bulk_structure,
            self.conventional_bulk_atoms,
        ) = self._get_atoms_and_struc(conventional_bulk)

        self.base_structure = base_structure
        self.transformation_matrix = transformation_matrix
        self.miller_index = miller_index
        self.layers = layers
        self.vacuum = vacuum
        self.uvw_basis = uvw_basis
        self.point_group_operations = point_group_operations
        self.bottom_layer_dist = bottom_layer_dist
        self.top_layer_dist = top_layer_dist
        self.passivated = False

    @property
    def formula(self):
        return self.conventional_bulk_structure.composition.reduced_formula

    @property
    def area(self):
        area = np.linalg.norm(
            np.cross(
                self.orthogonal_slab_structure.lattice.matrix[0],
                self.orthogonal_slab_structure.lattice.matrix[1],
            )
        )

        return area

    @property
    def inplane_vectors(self):
        matrix = deepcopy(self.orthogonal_slab_structure.lattice.matrix)
        return matrix[:2]

    @property
    def miller_index_a(self):
        return self.uvw_basis[0].astype(int)

    @property
    def miller_index_b(self):
        return self.uvw_basis[1].astype(int)

    def _get_atoms_and_struc(self, atoms_or_struc):
        if type(atoms_or_struc) == Atoms:
            init_structure = AseAtomsAdaptor.get_structure(atoms_or_struc)
            init_atoms = atoms_or_struc
        elif type(atoms_or_struc) == Structure:
            init_structure = atoms_or_struc
            init_atoms = AseAtomsAdaptor.get_atoms(atoms_or_struc)
        else:
            raise TypeError(
                f"Surface._get_atoms_and_struc() accepts 'pymatgen.core.structure.Structure' or 'ase.Atoms' not '{type(atoms_or_struc).__name__}'"
            )

        return init_structure, init_atoms

    def write_file(self, orthogonal=True, output="POSCAR_slab"):
        if orthogonal:
            slab = self.orthogonal_slab_structure
        else:
            slab = self.non_orthogonal_slab_structure

        if not self.passivated:
            Poscar(slab).write_file(output)
        else:
            comment = ":".join(
                [i for i in slab.site_properties["hydrogen_str"] if i != ""]
            )
            syms = [site.specie.symbol for site in slab]

            syms = []
            for site in slab:
                if site.specie.symbol == "H":
                    if hasattr(site.specie, "oxi_state"):
                        oxi = site.specie.oxi_state

                        if oxi < 1.0:
                            H_str = "H" + f"{oxi:.2f}"[1:]
                        elif oxi > 1.0:
                            H_str = "H" + f"{oxi:.2f}"
                        else:
                            H_str = "H"

                        syms.append(H_str)
                else:
                    syms.append(site.specie.symbol)

            comp_list = [(a[0], len(list(a[1]))) for a in groupby(syms)]
            atom_types, n_atoms = zip(*comp_list)

            poscar_str = Poscar(slab, comment=comment).get_string().split("\n")
            poscar_str[5] = " ".join(atom_types)
            poscar_str[6] = " ".join(list(map(str, n_atoms)))
            poscar_str = "\n".join(poscar_str)

            with open(output, "w") as f:
                f.write(poscar_str)

    def remove_layers(self, num_layers, top=False, atol=None):
        group_inds_conv, _ = utils.group_layers(
            structure=self.orthogonal_slab_structure, atol=atol
        )
        if top:
            group_inds_conv = group_inds_conv[::-1]

        to_delete_conv = []
        for i in range(num_layers):
            to_delete_conv.extend(group_inds_conv[i])

        self.orthogonal_slab_structure.remove_sites(to_delete_conv)

    def _get_surface_atoms(self):
        obs = self.primitive_oriented_bulk_structure.copy()
        obs.add_oxidation_state_by_guess()

        layer_struc = utils.get_layer_supercelll(structure=obs, layers=3)
        layer_struc.sort()

        layer_inds = np.array(layer_struc.site_properties["layer_index"])

        bottom_inds = np.where(layer_inds == 0)[0]
        top_inds = np.where(layer_inds == np.max(layer_inds))[0]

        cnn = CrystalNN()
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

        return layer_struc, [bottom_neighborhood, top_neighborhood]

    def _get_pseudohydrogen_charge(self, site, coordination):
        electronic_struc = site.specie.electronic_structure.split(".")[1:]
        oxi_state = site.specie.oxi_state
        valence = 0
        for orb in electronic_struc:
            if orb[1] == "d":
                if int(orb[2:]) < 10:
                    valence += int(orb[2:])
            else:
                valence += int(orb[2:])

        if oxi_state >= 0:
            charge = (8 - valence) / coordination
        else:
            charge = ((2 * coordination) - valence) / coordination

        return charge

    def _get_bond_dict(self):
        image_map = {1: "1", 0: "0", -1: "2"}
        (
            layer_struc,
            surface_neighborhoods,
        ) = self._get_surface_atoms()

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
                    layer_struc[atom_index], coordination
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
                    broken_image = broken_site.image.astype(int)
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
                            str(i),
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

    def _get_passivation_atom_index(self, struc, bulk_equivalent, top=False):
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

    def _passivate(self, struc, index, bond, bond_str, charge):
        position = struc[index].coords + bond
        position = struc[index].coords + bond
        props = {k: -1 for k in struc[index].properties}
        props["hydrogen_str"] = bond_str

        struc.append(
            Species("H", oxidation_state=charge),
            coords=position,
            coords_are_cartesian=True,
            properties=props,
        )

    def passivate(
        self, bot=True, top=True, passivated_struc=None, inplace=True
    ):
        bond_dict = self._get_bond_dict()
        ortho_slab = self.orthogonal_slab_structure.copy()
        non_ortho_slab = self.non_orthogonal_slab_structure.copy()

        ortho_slab.add_site_property("hydrogen_str", [""] * len(ortho_slab))
        non_ortho_slab.add_site_property(
            "hydrogen_str", [""] * len(non_ortho_slab)
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

        if bot:
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

        self.passivated = True

        if inplace:
            self.orthogonal_slab_structure = ortho_slab
            self.non_orthogonal_slab_structure = non_ortho_slab
        else:
            return ortho_slab, non_ortho_slab

    def get_termination(self):
        raise NotImplementedError


class Interface:
    def __init__(
        self,
        substrate,
        film,
        match,
        interfacial_distance,
        vacuum,
        center=False,
    ):
        self.center = center
        self.substrate = substrate
        self.film = film
        self.match = match
        self.vacuum = vacuum
        (
            self.substrate_supercell,
            self.substrate_supercell_uvw,
            self.substrate_supercell_scale_factors,
        ) = self._prepare_substrate()
        (
            self.film_supercell,
            self.film_supercell_uvw,
            self.film_supercell_scale_factors,
        ) = self._prepare_film()
        self.interfacial_distance = interfacial_distance
        self.interface_height = None
        self.strained_sub = self.substrate_supercell
        (
            self.strained_film,
            self.stack_transformation,
        ) = self._strain_and_orient_film()
        self.interface, self.sub_part, self.film_part = self._stack_interface()

    @property
    def area(self):
        return self.match.area

    @property
    def structure_volume(self):
        matrix = deepcopy(self.interface.lattice.matrix)
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
    def substrate_basis(self):
        return self.substrate_supercell_uvw

    @property
    def substrate_a(self):
        return self.substrate_supercell_uvw[0]

    @property
    def substrate_b(self):
        return self.substrate_supercell_uvw[1]

    @property
    def substrate_c(self):
        return self.substrate_supercell_uvw[2]

    @property
    def film_basis(self):
        return self.film_supercell_uvw

    @property
    def film_a(self):
        return self.film_supercell_uvw[0]

    @property
    def film_b(self):
        return self.film_supercell_uvw[1]

    @property
    def film_c(self):
        return self.film_supercell_uvw[2]

    def __str__(self):
        fm = self.film.miller_index
        sm = self.substrate.miller_index
        film_str = f"{self.film.formula}({fm[0]} {fm[1]} {fm[2]})"
        sub_str = f"{self.substrate.formula}({sm[0]} {sm[1]} {sm[2]})"
        s_uvw = self.substrate_supercell_uvw
        s_sf = self.substrate_supercell_scale_factors
        f_uvw = self.film_supercell_uvw
        f_sf = self.film_supercell_scale_factors
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
            "Strain Along \\vec{a} (%): "
            + f"{100*self.match.linear_strain[0]:.3f}",
            "Strain Along \\vec{b} (%): "
            + f"{100*self.match.linear_strain[1]:.3f}",
            "In-plane Angle Mismatch (%): "
            + f"{100*self.match.angle_strain:.3f}",
            "Cross Section Area (Ang^2): " + f"{self.area:.3f}",
        ]
        return_str = "\n".join(return_info)

        return return_str

    def write_file(self, output="POSCAR_interface"):
        Poscar(self.interface).write_file(output)

    def shift_film(
        self, shift, fractional=False, inplace=False, return_atoms=False
    ):
        if fractional:
            frac_shift = np.array(shift)
        else:
            shift = np.array(shift)

            if shift[-1] + self.interfacial_distance < 0.5:
                raise ValueError(
                    f"The film shift results in an interfacial distance of less than 0.5 Angstroms which is non-physical"
                )

            frac_shift = self.interface.lattice.get_fractional_coords(shift)

        film_ind = np.where(self.interface.site_properties["is_film"])[0]

        if inplace:
            self.interface.translate_sites(
                film_ind,
                frac_shift,
            )
            self.film_part.translate_sites(
                range(len(self.film_part)),
                frac_shift,
            )
            self.interface_height += frac_shift[-1] / 2
            self.interfacial_distance += shift[-1]

        else:
            shifted_interface = self.interface.copy()
            shifted_interface.translate_sites(
                film_ind,
                frac_shift,
            )

            if return_atoms:
                return AseAtomsAdaptor().get_atoms(shifted_interface)
            else:
                return shifted_interface

    def _prepare_substrate(self):
        matrix = self.match.substrate_sl_transform
        supercell_slab = self.substrate.orthogonal_slab_structure.copy()
        supercell_slab.make_supercell(scaling_matrix=matrix)

        uvw_supercell = matrix @ self.substrate.uvw_basis
        scale_factors = []
        for i, b in enumerate(uvw_supercell):
            scale = np.abs(reduce(utils._float_gcd, b))
            uvw_supercell[i] = uvw_supercell[i] / scale
            scale_factors.append(scale)

        return supercell_slab, uvw_supercell, scale_factors

    def _prepare_film(self):
        matrix = self.match.film_sl_transform
        supercell_slab = self.film.orthogonal_slab_structure.copy()
        supercell_slab.make_supercell(scaling_matrix=matrix)

        uvw_supercell = matrix @ self.film.uvw_basis
        scale_factors = []
        for i, b in enumerate(uvw_supercell):
            scale = np.abs(reduce(utils._float_gcd, b))
            uvw_supercell[i] = uvw_supercell[i] / scale
            scale_factors.append(scale)

        return supercell_slab, uvw_supercell, scale_factors

    def _strain_and_orient_film(self):
        sub_in_plane_vecs = self.substrate_supercell.lattice.matrix[:2]
        film_out_of_plane = self.film_supercell.lattice.matrix[-1]
        film_inv_matrix = self.film_supercell.lattice.inv_matrix
        new_matrix = np.vstack([sub_in_plane_vecs, film_out_of_plane])
        transform = (film_inv_matrix @ new_matrix).T
        op = SymmOp.from_rotation_and_translation(
            transform, translation_vec=np.zeros(3)
        )

        strained_film = deepcopy(self.film_supercell)
        strained_film.apply_operation(op)

        return strained_film, transform

    def _stack_interface(self):
        strained_sub = self.strained_sub
        strained_film = self.strained_film

        sub_matrix = strained_sub.lattice.matrix
        sub_c = deepcopy(sub_matrix[-1])

        strained_sub_coords = deepcopy(strained_sub.cart_coords)
        strained_film_coords = deepcopy(strained_film.cart_coords)
        strained_sub_frac_coords = deepcopy(strained_sub.frac_coords)
        strained_film_frac_coords = deepcopy(strained_film.frac_coords)

        min_sub_coords = np.min(strained_sub_frac_coords[:, -1])
        max_sub_coords = np.max(strained_sub_frac_coords[:, -1])
        min_film_coords = np.min(strained_film_frac_coords[:, -1])
        max_film_coords = np.max(strained_film_frac_coords[:, -1])

        sub_c_len = np.linalg.norm(strained_sub.lattice.matrix[-1])
        film_c_len = np.linalg.norm(strained_film.lattice.matrix[-1])
        interface_c_len = np.sum(
            [
                (max_sub_coords - min_sub_coords) * sub_c_len,
                (max_film_coords - min_film_coords) * film_c_len,
                self.vacuum,
                self.interfacial_distance,
            ]
        )
        frac_int_distance = self.interfacial_distance / interface_c_len

        interface_matrix = np.vstack(
            [sub_matrix[:2], interface_c_len * (sub_c / sub_c_len)]
        )
        interface_lattice = Lattice(matrix=interface_matrix)
        interface_inv_matrix = interface_lattice.inv_matrix

        sub_interface_coords = strained_sub_coords.dot(interface_inv_matrix)
        sub_interface_coords[:, -1] -= sub_interface_coords[:, -1].min()

        film_interface_coords = strained_film_coords.dot(interface_inv_matrix)
        film_interface_coords[:, -1] -= film_interface_coords[:, -1].min()
        film_interface_coords[:, -1] += (
            sub_interface_coords[:, -1].max() + frac_int_distance
        )

        interface_coords = np.r_[sub_interface_coords, film_interface_coords]
        interface_species = strained_sub.species + strained_film.species
        interface_site_properties = {
            key: strained_sub.site_properties[key]
            + strained_film.site_properties[key]
            for key in strained_sub.site_properties
        }
        interface_site_properties["is_sub"] = np.array(
            [True] * len(strained_sub) + [False] * len(strained_film)
        )
        interface_site_properties["is_film"] = np.array(
            [False] * len(strained_sub) + [True] * len(strained_film)
        )

        self.interface_height = sub_interface_coords[:, -1].max() + (
            0.5 * frac_int_distance
        )

        interface_struc = Structure(
            lattice=interface_lattice,
            species=interface_species,
            coords=interface_coords,
            to_unit_cell=True,
            coords_are_cartesian=False,
            site_properties=interface_site_properties,
        )
        interface_struc.sort()

        if self.center:
            interface_struc.translate_sites(
                indices=range(len(interface_struc)),
                vector=[0, 0, 0.5 - self.interface_height],
            )
            self.interface_height = 0.5

        film_inds = np.where(interface_struc.site_properties["is_film"])[0]
        sub_inds = np.where(interface_struc.site_properties["is_sub"])[0]

        film_part = interface_struc.copy()
        film_part.remove_sites(sub_inds)

        sub_part = interface_struc.copy()
        sub_part.remove_sites(film_inds)

        return interface_struc, sub_part, film_part

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

    def _generate_sc_for_interface_view(self, struc, transformation_matrix):
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
    ):
        cart_coords = (
            zero_coord + supercell_shift + cell_vetices.dot(slab_matrix)
        )
        fc = np.round(cart_coords.dot(sc_inv_matrix), 3)
        if is_film:
            plot_coords = cart_coords.dot(self.stack_transformation.T)
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

    def plot_interface(
        self,
        output="interface_view.png",
        dpi=400,
        show_in_colab=False,
    ):
        sub_matrix = self.substrate.orthogonal_slab_structure.lattice.matrix
        film_matrix = self.film.orthogonal_slab_structure.lattice.matrix
        sub_sc_matrix = deepcopy(self.substrate_supercell.lattice.matrix)
        film_sc_matrix = deepcopy(self.film_supercell.lattice.matrix)

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
        sub_sl = coords.dot(sub_sc_matrix)

        sub_struc, sub_inv_matrix = self._generate_sc_for_interface_view(
            struc=self.substrate.orthogonal_slab_structure,
            transformation_matrix=self.match.substrate_sl_transform,
        )

        film_struc, film_inv_matrix = self._generate_sc_for_interface_view(
            struc=self.film.orthogonal_slab_structure,
            transformation_matrix=self.match.film_sl_transform,
        )

        fig, ax = plt.subplots(figsize=(4, 4), dpi=dpi)

        for c in sub_struc.cart_coords:
            for shift in sub_sc_shifts:
                self._plot_interface_view(
                    ax=ax,
                    zero_coord=c,
                    supercell_shift=shift,
                    cell_vetices=coords,
                    slab_matrix=sub_matrix,
                    sc_inv_matrix=sub_inv_matrix,
                    is_film=False,
                    facecolor=(0, 0, 1, 0.2),
                    edgecolor=(0, 0, 1, 1),
                )

        for c in film_struc.cart_coords:
            for shift in film_sc_shifts:
                self._plot_interface_view(
                    ax=ax,
                    zero_coord=c,
                    supercell_shift=shift,
                    cell_vetices=coords,
                    slab_matrix=film_matrix,
                    sc_inv_matrix=film_inv_matrix,
                    is_film=True,
                    facecolor=(200 / 255, 0, 0, 0.2),
                    edgecolor=(200 / 255, 0, 0, 1),
                )

        ax.plot(
            sub_sl[:, 0],
            sub_sl[:, 1],
            color="black",
            linewidth=3,
        )

        ax.set_aspect("equal")
        ax.axis("off")

        fig.tight_layout()
        fig.savefig(output, bbox_inches="tight")

        if not show_in_colab:
            plt.close()
