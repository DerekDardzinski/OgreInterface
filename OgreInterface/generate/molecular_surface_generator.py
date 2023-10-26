"""
This module will be used to construct the surfaces and interfaces used in this package.
"""
from typing import Union, List, TypeVar, Optional
from itertools import combinations


from pymatgen.core.structure import Structure
from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import JmolNN
from ase import Atoms
import networkx as nx
import numpy as np

from OgreInterface.generate.base_surface_generator import BaseSurfaceGenerator
from OgreInterface.surfaces.molecular_surface import MolecularSurface

SelfMolecularSurfaceGenerator = TypeVar(
    "SelfMolecularSurfaceGenerator", bound="MolecularSurfaceGenerator"
)


class MolecularSurfaceGenerator(BaseSurfaceGenerator):
    """Class for generating surfaces from a given bulk structure.

    The MolecularSurfaceGenerator classes generates surfaces with all possible terminations and contains
    information pertinent to generating interfaces with the InterfaceGenerator.

    Examples:
        Creating a MolecularSurfaceGenerator object using PyMatGen to load the structure:
        >>> from OgreInterface.generate import MolecularSurfaceGenerator
        >>> from pymatgen.core.structure import Structure
        >>> bulk = Structure.from_file("POSCAR_bulk")
        >>> surfaces = MolecularSurfaceGenerator(bulk=bulk, miller_index=[1, 1, 1], layers=5, vacuum=60)
        >>> surface = surfaces[0] # OgreInterface.Surface object

        Creating a MolecularSurfaceGenerator object using the build in from_file() method:
        >>> from OgreInterface.generate import MolecularSurfaceGenerator
        >>> surfaces = MolecularSurfaceGenerator.from_file(filename="POSCAR_bulk", miller_index=[1, 1, 1], layers=5, vacuum=60)
        >>> surface = surfaces[0] # OgreInterface.Surface object

    Args:
        bulk: Bulk crystal structure used to create the surface
        miller_index: Miller index of the surface
        layers: Number of layers to include in the surface
        minimum_thickness: Optional flag to set the minimum thickness of the slab. If this is not None, then it will override the layers value
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
        layers: Optional[int] = None,
        minimum_thickness: Optional[float] = 18.0,
        vacuum: float = 40.0,
        refine_structure: bool = True,
        make_planar: bool = True,
        generate_all: bool = True,
        lazy: bool = False,
        suppress_warnings: bool = False,
        termination_grouping_tolerance: Optional[float] = None
    ) -> SelfMolecularSurfaceGenerator:
        super().__init__(
            bulk=bulk,
            miller_index=miller_index,
            surface_type=MolecularSurface,
            layers=layers,
            minimum_thickness=minimum_thickness,
            vacuum=vacuum,
            refine_structure=refine_structure,
            make_planar=make_planar,
            generate_all=generate_all,
            lazy=lazy,
            suppress_warnings=suppress_warnings,
            termination_grouping_tolerance=termination_grouping_tolerance
        )

    @classmethod
    def from_file(
        cls,
        filename: str,
        miller_index: List[int],
        layers: Optional[int] = None,
        minimum_thickness: Optional[float] = 18.0,
        vacuum: float = 40.0,
        refine_structure: bool = True,
        make_planar: bool = True,
        generate_all: bool = True,
        lazy: bool = False,
        suppress_warnings: bool = False,
        termination_grouping_tolerance: Optional[float] = None
    ) -> SelfMolecularSurfaceGenerator:
        return super().from_file(
            filename=filename,
            miller_index=miller_index,
            layers=layers,
            minimum_thickness=minimum_thickness,
            vacuum=vacuum,
            refine_structure=refine_structure,
            make_planar=make_planar,
            generate_all=generate_all,
            lazy=lazy,
            suppress_warnings=suppress_warnings,
            termination_grouping_tolerance=termination_grouping_tolerance
        )

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

    def _replace_molecules_with_atoms(self, structure: Structure) -> Structure:
        # Create a structure graph so we can extract the molecules
        struc_graph = StructureGraph.with_local_env_strategy(
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
        molecule_tops = []
        site_props = list(structure.site_properties.keys())
        # site_props.remove("molecule_index")
        props = {p: [] for p in site_props}
        for subgraph in molecule_subgraphs:
            cart_coords = np.vstack(
                [struc_graph.structure[n].coords for n in subgraph]
            )

            projected_coords = np.dot(cart_coords, self.obs.surface_normal)
            top_ind = np.argmax(projected_coords)
            top_position = cart_coords[top_ind]
            is_top = np.zeros(len(cart_coords)).astype(bool)
            is_top[top_ind] = True

            for t, n in zip(is_top, subgraph):
                struc_graph.structure[n].properties["is_top"] = t

            for p in props:
                ind = list(subgraph.nodes.keys())[0]
                props[p].append(struc_graph.structure[ind].properties[p])

            molecule_tops.append(np.round(top_position, 6))

        molecule_tops = np.vstack(molecule_tops)

        # Now we can find which center of masses are contained in the original
        # unit cell. First we can shift the center of masses by the [1, 1, 1]
        # vector of the original unit cell so the center unit cell of the 3x3
        # supercell is positioned at (0, 0, 0)
        shift = structure.lattice.get_cartesian_coords([1, 1, 1])
        inv_matrix = structure.lattice.inv_matrix

        # Shift the center of masses
        molecule_tops -= shift

        # Convert to fractional coordinates of the original unit cell
        frac_top = molecule_tops.dot(inv_matrix)

        # The reference atoms in the unit cell should have fractional
        # coordinates between [0, 1)
        in_original_cell = np.logical_and(
            0 <= np.round(frac_top, 6),
            np.round(frac_top, 6) < 1,
        ).all(axis=1)

        # Extract the fractional coordinates in the original cell
        frac_coords_in_cell = frac_top[in_original_cell]

        # Extract the molecules that have the reference atom in the unit cell
        m_graphs_in_cell = [
            molecule_subgraphs[i] for i in np.where(in_original_cell)[0]
        ]

        # Initiate a list of pymatgen.Molecule objects
        molecules = []

        # Initial a new site property dict for the dummy atom structure
        props_in_cell = {}

        # Extract the molecules who's reference atom is in the original cell
        for i, m_graph in enumerate(m_graphs_in_cell):
            # Get the cartesian coordinates of the molecule from the graph
            coords = np.vstack(
                [struc_graph.structure[n].coords for n in m_graph.nodes()]
            )

            # Get the species of the molecule from the graph
            species = [
                struc_graph.structure[n].specie for n in m_graph.nodes()
            ]

            # Get the is_top site properties of the molecule from the graph
            # This is used to find the reference atom to shift the molecule
            is_top = [
                struc_graph.structure[n].properties["is_top"]
                for n in m_graph.nodes()
            ]

            # Get the site properties of all the atoms in the molecules
            site_props = [
                struc_graph.structure[n].properties for n in m_graph.nodes()
            ]

            # Extract the properties of the reference atom to be used as the
            # site propeties of the dummy atom in the dummy atom structure
            top_props = site_props[int(np.where(is_top)[0][0])]

            # Add these properties to the props in cell dict
            for k, v in top_props.items():
                if k in props_in_cell:
                    props_in_cell[k].append(v)
                else:
                    props_in_cell[k] = [v]

            # Get the coordinates of the reference atom
            top_coord = coords[is_top]

            # Create a Molecule with the reference atom shifted to (0, 0, 0)
            molecule = Molecule(species, coords - top_coord)

            # Add to the list of molecules
            molecules.append(molecule)

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
        connected_components = [
            list(c) for c in nx.connected_components(mol_id_graph)
        ]

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
        props_in_cell["bulk_equivalent"] = [
            bulk_equiv_mapping[i] for i in range(len(molecules))
        ]

        # Get the atomic numbers for the dummy species
        # (22 is just for nicer colors in vesta)
        species = [i + 22 for i in range(len(molecules))]
        props_in_cell["dummy_species"] = species

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
        dummy_struc.sort()

        return dummy_struc

    def _get_slab_base(self) -> Structure:
        # Replace the molecules with dummy atoms and use the dummy atom
        # structure as the slab base
        dummy_obs = self._replace_molecules_with_atoms(
            structure=self.obs.oriented_bulk_structure
        )

        # Set the oriented_bulk_structure to the dummy_obs structure
        self.obs._oriented_bulk_structure = dummy_obs

        return self.obs
