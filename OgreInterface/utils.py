from __future__ import annotations
import copy
from functools import reduce
import itertools
import functools
import math
import collections
from typing import List, Tuple, Union, Optional

from pymatgen.core.structure import Structure, Molecule
from pymatgen.core.lattice import Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.periodic_table import DummySpecies
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import JmolNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from ase import Atoms
import numpy as np
import networkx as nx
import spglib


def load_bulk(
    atoms_or_structure: Union[Atoms, Structure],
    refine_structure: bool = True,
    suppress_warnings: bool = False,
) -> Structure:
    if type(atoms_or_structure) is Atoms:
        init_structure = AseAtomsAdaptor.get_structure(atoms_or_structure)
    elif type(atoms_or_structure) is Structure:
        init_structure = atoms_or_structure
    else:
        raise TypeError(
            f"load_bulk accepts 'pymatgen.core.structure.Structure' or 'ase.Atoms' not '{type(atoms_or_structure).__name__}'"
        )

    if refine_structure:
        conventional_structure = spglib_standardize(
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
            if not suppress_warnings:
                labels = ["a", "b", "c", "alpha", "beta", "gamma"]
                init_cell_str = ", ".join(
                    [
                        f"{label} = {val:.3f}"
                        for label, val in zip(labels, init_length_and_angles)
                    ]
                )
                conv_cell_str = ", ".join(
                    [
                        f"{label} = {val:.3f}"
                        for label, val in zip(labels, conv_length_and_angles)
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

        return conventional_structure
    else:
        return init_structure


def get_rounded_structure(structure: Structure, tol: int = 6):
    new_matrix = copy.deepcopy(structure.lattice.matrix)
    rounded_matrix = np.round(new_matrix, tol)
    rounded_structure = Structure(
        lattice=Lattice(matrix=rounded_matrix),
        species=structure.species,
        coords=np.mod(np.round(structure.frac_coords, tol), 1.0),
        to_unit_cell=True,
        coords_are_cartesian=False,
        site_properties=structure.site_properties,
    )

    return rounded_structure


def _float_gcd(self, a, b, rtol=1e-05, atol=1e-08):
    t = min(abs(a), abs(b))
    while abs(b) > rtol * t + atol:
        a, b = b, a % b
    return a


def hex_to_cubic_direction(uvtw) -> np.ndarray:
    u = 2 * uvtw[0] + uvtw[1]
    v = 2 * uvtw[1] + uvtw[0]
    w = uvtw[-1]

    output = np.array([u, v, w])
    output = _get_reduced_vector(output)

    return output.astype(int)


def cubic_to_hex_direction(uvw) -> np.ndarray:
    u = (1 / 3) * ((2 * uvw[0]) - uvw[1])
    v = (1 / 3) * ((2 * uvw[1]) - uvw[0])
    t = -(u + v)
    w = uvw[-1]

    output = np.array([u, v, t, w])
    output = _get_reduced_vector(output)

    return output.astype(int)


def hex_to_cubic_plane(hkil) -> np.ndarray:
    h, k, i, l = hkil

    return np.array([h, k, l]).astype(int)


def cubic_to_hex_plane(hkl) -> np.ndarray:
    h, k, l = hkl

    return np.array([h, k, -(h + k), l]).astype(int)


def get_unique_miller_indices(
    structure: Structure,
    max_index: int,
) -> np.ndarray:
    # Get the spacegroup of the input structure
    struc_sg = SpacegroupAnalyzer(structure)

    # Get the lattice object
    lattice = structure.lattice

    # Determine if it is hexagonal
    is_hexagonal = lattice.is_hexagonal

    # Get the reciprocal lattice
    recip = structure.lattice.reciprocal_lattice_crystallographic

    # Get all the point group operations of the structure
    symmops = struc_sg.get_point_group_operations(cartesian=False)

    # Get the list of all possible planes up to some max miller index
    planes = set(
        list(itertools.product(range(-max_index, max_index + 1), repeat=3))
    )
    planes.remove((0, 0, 0))

    # Get the reduced planes (i.e. (2, 2, 2) -> (1, 1, 1))
    reduced_planes = []
    for plane in planes:
        reduced_plane = _get_reduced_vector(np.array(plane).astype(float))
        reduced_plane = reduced_plane.astype(int)
        reduced_planes.append(tuple(reduced_plane))

    # Make a set to get the unique planes
    reduced_planes = set(reduced_planes)

    # If the structure is hexagonal the convert from (hkl) to (hkil) and remove
    # and surfaces with any |i| > max_index
    if is_hexagonal:
        hexagonal_planes = []
        for plane in reduced_planes:
            hexagonal_plane = cubic_to_hex_plane(hkl=plane)

            if (np.abs(hexagonal_plane) <= max_index).all():
                hexagonal_planes.append(tuple(hexagonal_plane))

        reduced_planes = set(hexagonal_planes)

    # Initialize a dictionary of all the unique planes
    planes_dict = {p: [] for p in reduced_planes}

    # Loop through all planes
    for plane in reduced_planes:
        # If the plane hasnt been found then apply all point group
        # operations to the surface normal of that plane to get all
        # symmetrically equivalent surfaces
        if plane in planes_dict:
            # If it is hexagonal go from (hkil) -> (hkl) when applying
            # point group operations
            if is_hexagonal:
                op_plane = tuple(hex_to_cubic_plane(hkil=plane))
            else:
                op_plane = copy.deepcopy(plane)

            # Get the real space plane normal in fractional coordinates
            # by multiplying by the reciprocal metric tensor
            frac_normal = np.array(op_plane).dot(recip.metric_tensor)

            # Loop through all point group operations
            for i, symmop in enumerate(symmops):
                # Apply the symmop to the surface normal in fractional coords
                frac_normal_op = symmop.apply_rotation_only(frac_normal)

                # Multiply by the realspace metric tensor to get the reciprocal
                # direction in fractional coordinates (i.e. the new (hkl))
                equiv_plane = _get_reduced_vector(
                    frac_normal_op.dot(lattice.metric_tensor)
                ).astype(int)

                # If the structure has a hexagonal lattice then convert (hkl)
                # back to (hkil)
                if is_hexagonal:
                    equiv_plane = cubic_to_hex_plane(hkl=equiv_plane)

                equiv_plane = tuple(equiv_plane)

                # Append the equivalent plane to the dictionary
                planes_dict[plane].append(equiv_plane)

                # If the equivalent plane is not equal to the initial plane
                # and the equivalent plane is still in the planes_dict keys
                # the remove it from the keys because we already know it is
                # an equivalent plane
                if equiv_plane != plane:
                    if equiv_plane in planes_dict.keys():
                        del planes_dict[equiv_plane]

    # Next we go through all equivalent plane and get the preferred surface
    # i.e. if (110), (1-10), (-1-10), (10-1) are all equivalent then we would
    # prefer to return (110) becuase it looks better
    unique_planes = []
    for k in planes_dict:
        # Get an array of all the equivalent planes
        equivalent_planes = np.array(list(set(planes_dict[k])))

        # Get the difference in signs for all planes ideally we would like
        # to have the surface with (hkl) that has (hkl) with all the same
        # signs
        diff = np.abs(np.sum(np.sign(equivalent_planes), axis=1))

        # Filter out the surfaces with the most similar signs
        like_signs = equivalent_planes[diff == np.max(diff)]

        # If there is only one surface with similar signs then select it
        if len(like_signs) == 1:
            unique_planes.append(like_signs[0])
        else:
            # If there is more than one surface with similar signs then we
            # want the surface with the first index as the largest number
            # i.e. between (012), (102), (201), & (210) we want (210) or (201)
            first_max = like_signs[
                np.abs(like_signs)[:, 0] == np.max(np.abs(like_signs)[:, 0])
            ]

            # If there is only one option after that then we select it
            if len(first_max) == 1:
                unique_planes.append(first_max[0])
            else:
                # If there is more than one surface with the first entry as
                # the max then we want the one with largest second entry
                # i.e. between (012), (102), (201), & (210) we want (210)
                second_max = first_max[
                    np.abs(first_max)[:, 1] == np.max(np.abs(first_max)[:, 1])
                ]

                # If there is only one option at that point then we select it
                if len(second_max) == 1:
                    unique_planes.append(second_max[0])
                else:
                    # If there is more than one option at this point then pick
                    # the one with the largest with the largest positive values
                    # i.e. between (-210) and (2-10) we want (2-10)
                    unique_planes.append(
                        second_max[np.argmax(np.sign(second_max).sum(axis=1))]
                    )

    # Stack all of the unique planes into an array
    unique_planes = np.vstack(unique_planes)

    # Sort the planes by the shortest norm and most positive elements
    sorted_planes = sorted(
        unique_planes, key=lambda x: (np.linalg.norm(x), -np.sign(x).sum())
    )

    return np.vstack(sorted_planes)


def get_miller_index_label(miller_index: List[int]):
    return "".join(
        [
            str(i) if i >= 0 else "$\\overline{" + str(-i) + "}$"
            for i in miller_index
        ]
    )


def get_miller_index_string(miller_index: List[int]):
    return "".join([str(i) for i in miller_index])


def add_symmetry_info(struc: Structure, return_primitive: bool = False):
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

        prim_bulk = spglib_standardize(
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


def _get_colored_molecules(struc, output):
    colored_struc = struc.copy()
    for site in colored_struc:
        if "dummy_species" in site.properties:
            ds = site.properties["dummy_species"]
        else:
            ds = site.species.Z

        site.species = DummySpecies(symbol=f"Q{chr(ds - 22 + ord('a'))}")

    colored_struc.sort()
    Poscar(colored_struc).write_file(output)


def get_latex_formula(formula: str) -> str:
    groups = itertools.groupby(formula, key=lambda x: x.isdigit())

    latex_formula = ""
    for k, group in groups:
        if k:
            part = "$_{" + "".join(list(group)) + "}$"
        else:
            part = "".join(list(group))

        latex_formula += part

    return latex_formula


def apply_strain_matrix(
    structure: Structure, strain_matrix: np.ndarray
) -> Structure:
    """
    This function applies a strain matrix to a structure to match it to another lattice
    i.e. straining a film to match with the substrate material. The strain_matrix can be calculated
    by the following equation:
        strain_matrix = np.linalg.inv(old_lattice_matrix) @ new_lattice_matrix
    """
    new_matrix = structure.lattice.matrix @ strain_matrix

    strained_structure = Structure(
        lattice=Lattice(new_matrix),
        species=structure.species,
        coords=structure.frac_coords,
        to_unit_cell=True,
        coords_are_cartesian=False,
        site_properties=structure.site_properties,
    )

    return strained_structure


def spglib_standardize(
    structure: Structure,
    to_primitive: bool = False,
    no_idealize: bool = False,
) -> Structure:
    """
    This function standardized a given structure using the spglib library

    Args:
        structure: Input pymatgen Structure
        to_primitive: Determines if the structure should be converted to it's primitive unit cell
        no_idealize: Determines if the lattice vectors should be idealized
            (i.e. rotate a cubic structure so \\vec{a} points along the [1, 0, 0] cartesian direction)

    Returns:
        The standardized structure in the form of a pymatgen Structure object
    """
    init_lattice = structure.lattice.matrix
    init_positions = structure.frac_coords
    init_numbers = np.array(structure.atomic_numbers)
    init_cell = (init_lattice, init_positions, init_numbers)

    (
        standardized_lattice,
        standardized_positions,
        standardized_numbers,
    ) = spglib.standardize_cell(
        init_cell,
        to_primitive=to_primitive,
        no_idealize=no_idealize,
    )

    standardized_structure = Structure(
        lattice=Lattice(standardized_lattice),
        species=standardized_numbers,
        coords=standardized_positions,
        to_unit_cell=True,
        coords_are_cartesian=False,
    )

    return standardized_structure


def apply_op_to_mols(struc, op):
    for site in struc:
        mol = site.properties["molecules"]
        op_mol = mol.copy()
        op_mol.translate_sites(range(len(mol)), site.coords)
        op_mol.apply_operation(op)
        centered_mol = op_mol.get_centered_molecule()
        site.properties["molecules"] = centered_mol


def replace_molecules_with_atoms(s: Structure) -> Structure:
    # Create a structure graph so we can extract the molecules
    struc_graph = StructureGraph.with_local_env_strategy(s, JmolNN())

    # Find the center of masses of all the molecules in the unit cell
    # We can do this similar to how the get_subgraphs_as_molecules()
    # function works by creating a 3x3 supercell and only keeping the
    # molecules that don't intersect the boundary of the unit cell
    struc_graph *= (3, 3, 3)
    supercell_g = nx.Graph(struc_graph.graph)

    # Extract all molecule subgraphs
    all_subgraphs = [
        supercell_g.subgraph(c) for c in nx.connected_components(supercell_g)
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
    # site_props.remove("molecule_index")
    props = {p: [] for p in site_props}
    for subgraph in molecule_subgraphs:
        cart_coords = np.vstack(
            [struc_graph.structure[n].coords for n in subgraph]
        )
        weights = np.array(
            [struc_graph.structure[n].species.weight for n in subgraph]
        )

        for p in props:
            ind = list(subgraph.nodes.keys())[0]
            props[p].append(struc_graph.structure[ind].properties[p])

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
        p: [l[i] for i in np.where(in_original_cell)[0]]
        for p, l in props.items()
    }

    # Extract the molecules who's center of mass is in the original cell
    molecules = []
    for i in np.where(in_original_cell)[0]:
        m_graph = molecule_subgraphs[i]
        coords = [struc_graph.structure[n].coords for n in m_graph.nodes()]
        species = [struc_graph.structure[n].specie for n in m_graph.nodes()]
        molecule = Molecule(species, coords)
        molecule = molecule.get_centered_molecule()
        molecules.append(molecule)

    # Create the structure with the center of mass
    # species, frac_coords, bases, mols = list(zip(*struc_data))
    if "dummy_species" not in props_in_cell:
        species = [i + 22 for i in range(len(molecules))]
        props_in_cell["dummy_species"] = species
    else:
        species = props_in_cell["dummy_species"]

    frac_coords = frac_coords_in_cell
    struc_props = {
        "molecules": molecules,
    }
    struc_props.update(props_in_cell)

    dummy_struc = Structure(
        lattice=s.lattice,
        coords=frac_coords,
        species=species,
        site_properties=struc_props,
    )
    dummy_struc.sort()

    return dummy_struc


def add_molecules(struc):
    mol_coords = []
    mol_atom_nums = []

    properties = list(struc.site_properties.keys())
    properties.remove("molecules")
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


def conv_a_to_b(struc_a: Structure, struc_b: Structure) -> np.ndarray:
    return np.round(
        struc_b.lattice.matrix @ struc_a.lattice.inv_matrix
    ).astype(int)


def get_atoms(struc):
    return AseAtomsAdaptor().get_atoms(struc)


def get_layer_supercell(
    structure: Structure, layers: int, vacuum_scale: int = 0
) -> Structure:
    base_frac_coords = structure.frac_coords
    sc_base_frac_coords = np.vstack(
        [base_frac_coords + np.array([0, 0, i]) for i in range(layers)]
    )
    sc_cart_coords = sc_base_frac_coords.dot(structure.lattice.matrix)
    sc_layer_inds = np.repeat(np.arange(layers), len(structure))

    new_site_properties = {
        k: v * layers for k, v in structure.site_properties.items()
    }
    new_site_properties["layer_index"] = sc_layer_inds.tolist()

    if "atomic_layer_index" in new_site_properties:
        atomic_layers = np.array(new_site_properties["atomic_layer_index"])
        offset = (atomic_layers.max() * sc_layer_inds) + sc_layer_inds
        new_atomic_layers = atomic_layers + offset
        new_site_properties["atomic_layer_index"] = new_atomic_layers

    layer_transform = np.eye(3)
    layer_transform[-1, -1] = layers + vacuum_scale
    layer_matrix = layer_transform @ structure.lattice.matrix

    layer_slab = Structure(
        lattice=Lattice(matrix=layer_matrix),
        species=structure.species * layers,
        coords=sc_cart_coords,
        coords_are_cartesian=True,
        to_unit_cell=True,
        site_properties=new_site_properties,
    )

    return layer_slab


def calculate_possible_shifts(
    structure: Structure,
    tol: Optional[float] = None,
):
    frac_coords = structure.frac_coords[:, -1]

    # Projection of c lattice vector in
    # direction of surface normal.
    h = structure.lattice.matrix[-1, -1]

    if tol is None:
        cart_coords = structure.cart_coords[:, -1]
        extended_cart_coords = np.round(
            np.concatenate(
                [
                    cart_coords - h,
                    cart_coords,
                    cart_coords + h,
                ]
            ),
            5,
        )
        unique_cart_coords = np.sort(np.unique(extended_cart_coords))
        diffs = np.diff(unique_cart_coords)
        max_diff = diffs.max()
        tol = 0.15 * max_diff

        print(f"{tol = }")

    n = len(frac_coords)

    if n == 1:
        # Clustering does not work when there is only one data point.
        shift = frac_coords[0] + 0.5
        return [shift - math.floor(shift)]

    # We cluster the sites according to the c coordinates. But we need to
    # take into account PBC. Let's compute a fractional c-coordinate
    # distance matrix that accounts for PBC.
    dist_matrix = np.zeros((n, n))

    for i, j in itertools.combinations(list(range(n)), 2):
        if i != j:
            cdist = frac_coords[i] - frac_coords[j]
            cdist = abs(cdist - round(cdist)) * h
            dist_matrix[i, j] = cdist
            dist_matrix[j, i] = cdist

    condensed_m = squareform(dist_matrix)
    z = linkage(condensed_m)
    clusters = fcluster(z, tol, criterion="distance")

    # Generate dict of cluster# to c val - doesn't matter what the c is.
    c_loc = {c: frac_coords[i] for i, c in enumerate(clusters)}

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


def group_layers(structure, atol=None):
    """
    This function will find the atom indices belonging to each unique atomic layer.

    Args:
        structure (pymatgen.core.structure.Structure): Slab structure
        atol (float or None): Tolarence used for grouping the layers. Useful for grouping
            layers in a structure with relaxed atomic positions.

    Returns:
        A list containing the indices of each layers.
        A list of heights of each layers in fractional coordinates.
    """
    sites = structure.sites
    zvals = np.array([site.c for site in sites])
    unique_values = np.sort(np.unique(np.round(zvals, 3)))
    diff = np.mean(np.diff(unique_values)) * 0.2

    grouped = False
    groups = []
    group_heights = []
    zvals_copy = copy.deepcopy(zvals)
    while not grouped:
        if len(zvals_copy) > 0:
            if atol is None:
                group_index = np.where(
                    np.isclose(zvals, np.min(zvals_copy), atol=diff)
                )[0]
            else:
                group_index = np.where(
                    np.isclose(zvals, np.min(zvals_copy), atol=atol)
                )[0]

            group_heights.append(np.min(zvals_copy))
            zvals_copy = np.delete(
                zvals_copy,
                np.where(np.isin(zvals_copy, zvals[group_index]))[0],
            )
            groups.append(group_index)
        else:
            grouped = True

    return groups, np.array(group_heights)


def get_reduced_basis(basis: np.ndarray) -> np.ndarray:
    """
    This function is used to find the miller indices of the slab structure
    basis vectors in their most reduced form. i.e.

    |  2  4  0 |     | 1  2  0 |
    |  0 -2  4 | ==> | 0 -1  2 |
    | 10 10 10 |     | 1  1  1 |

    Args:
        basis (np.ndarray): 3x3 matrix defining the lattice vectors

    Returns:
        Reduced integer basis in the form of miller indices
    """
    basis /= np.linalg.norm(basis, axis=1)[:, None]

    for i, b in enumerate(basis):
        basis[i] = _get_reduced_vector(b)

    return np.round(basis).astype(int)


def _get_reduced_vector(vector: np.ndarry) -> np.ndarray:
    """ """
    abs_b = np.abs(vector)
    vector /= abs_b[abs_b > 0.001].min()
    vector /= np.abs(reduce(_float_gcd, vector))

    return np.round(vector)


def _float_gcd(a, b, rtol=1e-05, atol=1e-08):
    t = min(abs(a), abs(b))
    while abs(b) > rtol * t + atol:
        a, b = b, a % b
    return a


def reduce_vectors_zur_and_mcgill(
    a: np.ndarray,
    b: np.ndarray,
    surface_normal: np.ndarray = np.array([0, 0, 1]),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vecs = np.vstack([a, b])
    mat = np.eye(3)
    reduced = False

    while not reduced:
        dot = np.round(np.dot(vecs[0], vecs[1]), 6)
        a_norm = np.round(np.linalg.norm(vecs[0]), 6)
        b_norm = np.round(np.linalg.norm(vecs[1]), 6)
        b_plus_a_norm = np.round(np.linalg.norm(vecs[1] + vecs[0]), 6)
        b_minus_a_norm = np.round(np.linalg.norm(vecs[1] - vecs[0]), 6)

        if dot < 0:
            vecs[1] *= -1
            mat[1] *= -1
            continue

        if a_norm > b_norm:
            vecs = vecs[[1, 0]]
            mat = mat[[1, 0, 2]]
            continue

        if b_norm > b_plus_a_norm:
            vecs[1] = vecs[1] + vecs[0]
            mat[1] = mat[1] + mat[0]
            continue

        if b_norm > b_minus_a_norm:
            vecs[1] = vecs[1] - vecs[0]
            mat[1] = mat[1] - mat[0]
            reduced = True
            continue

        reduced = True

    final_dot = np.dot(vecs[0], vecs[1])
    dot_0 = np.isclose(np.round(final_dot, 5), 0.0)
    a_norm = np.linalg.norm(vecs[0])
    b_norm = np.linalg.norm(vecs[1])

    basis = np.eye(3)
    basis[:2] = vecs
    basis[-1] = surface_normal
    det = np.linalg.det(basis)
    lefty = det < 0

    if dot_0 and lefty:
        vecs[1] *= -1
        mat[1] *= -1

    if not dot_0 and lefty:
        vecs = vecs[[1, 0]]
        mat = mat[[1, 0, 2]]

    return vecs[0], vecs[1], mat
