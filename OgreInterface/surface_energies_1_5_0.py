from copy import deepcopy
from typing import Union, List, TypeVar, Tuple, Dict, Optional
from itertools import combinations, product, groupby
from collections.abc import Sequence
from abc import abstractmethod
import math


from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.operations import SymmOp
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import JmolNN, CrystalNN
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from ase import Atoms
from tqdm import tqdm
import networkx as nx
import numpy as np
import spglib


from OgreInterface import utils
from OgreInterface.surfaces.oriented_bulk import OrientedBulk
from OgreInterface.surfaces.surface import Surface
from OgreInterface.surfaces.molecular_surface import MolecularSurface

SelfBaseSurfaceGenerator = TypeVar(
    "SelfBaseSurfaceGenerator", bound="BaseSurfaceGenerator"
)

class BaseSurfaceGenerator(Sequence):
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
        surface_type: Union[Surface, MolecularSurface],
        layers: Optional[int] = None,
        minimum_thickness: Optional[float] = 18.0,
        vacuum: float = 40.0,
        refine_structure: bool = True,
        make_planar: bool = True,
        generate_all: bool = True,
        lazy: bool = False,
        suppress_warnings: bool = False,
        termination_grouping_tolerance: Optional[float] = None
    ) -> None:
        super().__init__()
        self.refine_structure = refine_structure
        self.surface_type = surface_type
        self._suppress_warnings = suppress_warnings
        self._make_planar = make_planar
        #self._termination_grouping_tolerance = termination_grouping_tolerance

        self.bulk_structure = utils.load_bulk(
            atoms_or_structure=bulk,
            refine_structure=refine_structure,
            suppress_warnings=suppress_warnings
        )

        self.miller_index = miller_index

        self.vacuum = vacuum
        self.generate_all = generate_all
        self.lazy = lazy

        self.obs = OrientedBulk(
            bulk=self.bulk_structure,
            miller_index=self.miller_index,
            make_planar=self._make_planar
        )

        if layers is None and minimum_thickness is None:
            raise "Either layer or minimum_thickness must be set"
        if layers is not None:
            self.layers = layers
        if layers is None and minimum_thickness is not None:
            self.layers = int(
                (minimum_thickness // self.obs.layer_thickness) + 1
            )

        if not self.lazy:
            self._slabs = self._generate_slabs(tol=termination_grouping_tolerance)
        else:
            self._slabs = None

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
    ) -> SelfBaseSurfaceGenerator:
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
            bulk=structure,
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

    def __getitem__(self, i) -> Surface:
        if self._slabs:
            return self._slabs[i]
        else:
            print(
                "The slabs have not been generated yet, please use the generate_slabs() function to create them."
            )

    def __len__(self) -> int:
        return len(self._slabs)

    def generate_slabs(
        self,
        tol: Optional[float] = None    
    ) -> None:
        """Used to generate list of Surface objects if lazy=True"""
        if self.lazy:
            self._slabs = self._generate_slabs(tol=tol)
        else:
            print(
                "The slabs are already generated upon initialization. This function is only needed if lazy=True"
            )

    @abstractmethod
    def _get_slab_base(self) -> Structure:
        """
        Abstract method that should be replaced by the inheriting class.
        This should return the base structure that is used to generate the
        surface. For an atomic surface this should be the oriented bulk
        structure and for a molecular surface this should be the oriented
        bulk structurer replaced by dummy atoms
        """
        pass

    def _get_point_group_operations(self):
        # TODO Move this to Interface Generator
        sg = SpacegroupAnalyzer(self.bulk_structure)
        point_group_operations = sg.get_point_group_operations(cartesian=False)
        operation_array = np.round(
            np.array([p.rotation_matrix for p in point_group_operations])
        ).astype(np.int8)
        unique_operations = np.unique(operation_array, axis=0)

        return unique_operations

    def _get_slab(
        self,
        slab_base: OrientedBulk,
        shift: float = 0.0,
        tol: Optional[float] = None,
    ) -> Tuple[Structure, Structure, float, Tuple[int, ...]]:
        """
        This method takes in shift value for the c lattice direction and
        generates a slab based on the given shift. You should rarely use this
        method. Instead, it is used by other generation algorithms to obtain
        all slabs.

        Args:
            slab_base: Oriented bulk structure used to generate the slab
            shift: A shift value in fractional c-coordinates that determines
                how much the slab_base should be shifted to select a given
                termination.
            tol: Optional tolarance for grouping the atomic layers together

        Returns:
            Returns a tuple of the shifted slab base, orthogonalized slab,
            non-orthogonalized slab, actual value of the vacuum in angstroms,
            and the tuple of layer indices and bulk equivalents that is used
            to filter out duplicate surfaces.
        """
        # Shift the slab base to the termination defined by the shift input
        slab_base.translate_sites(
            vector=[0, 0, -shift],
            frac_coords=True,
        )

        # Round and mod the structure
        slab_base.round(tol=6)

        # Get the fractional c-coords
        c_coords = slab_base.oriented_bulk_structure.frac_coords[:, -1]

        # Calculate the shifts again on the shifted structure to get the upper
        # and lower bounds of where an atomic layer should be.
        shifts = self._calculate_possible_shifts(
            structure=slab_base.oriented_bulk_structure,
            tol=tol
        )
        shifts += [1.0]

        # Group the upper and lower bounds into a list of tuples
        atomic_layer_bounds = [
            (shifts[i], shifts[i + 1]) for i in range(len(shifts) - 1)
        ]

        # Define an array of -1's that will get filled in later with atomic
        # layer indices
        atomic_layers = -np.ones(len(c_coords))
        for i, (bottom_bound, top_bound) in enumerate(atomic_layer_bounds):
            # Find atoms that have c-position between the top and bottom bounds
            layer_mask = (c_coords > bottom_bound) & (c_coords < top_bound)

            # Set the atomic layer index to i
            atomic_layers[layer_mask] = i

        # Add the atomic layer site property to the slab base
        slab_base.add_site_property(
            "atomic_layer_index",
            atomic_layers.tolist(),
        )

        # Get the bulk equivalent to create the key associated with the given
        # surface so that we can extract unique surface terminations later on
        bulk_equiv = np.array(slab_base.site_properties["bulk_equivalent"])

        # The surface key is sorted by atomic layer and the bulk equivalent
        # i.e. [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), ...]
        surf_key = sorted(
            [(idx, eq) for idx, eq in zip(atomic_layers, bulk_equiv)],
            key=lambda x: (x[0], x[1]),
        )

        # Concatenate the key and turn it into one long tuple of ints
        surf_key = tuple(np.concatenate(surf_key).astype(int))

        # Get the top c-coord
        top_c = c_coords.max()

        # Get the bottom c-coord
        # bot_c = c_coords.min()

        # Get the inds of the top atoms so we can shift of so the top atom
        # has a and b positions of zero.
        max_c_inds = np.where(np.isclose(top_c, c_coords))[0]

        dists = []
        for i in max_c_inds:
            # Get distance from the origin
            dist, image = slab_base[i].distance_and_image_from_frac_coords(
                fcoords=[0.0, 0.0, 0.0]
            )
            dists.append(dist)

        # Get the atom index of the top atom that is closest to a=0, b=0
        horiz_shift_ind = max_c_inds[np.argmin(dists)]

        # Find the planar shift required to plane the top atom at a=0, b=0
        horiz_shift = -slab_base[horiz_shift_ind].frac_coords
        horiz_shift[-1] = 0

        # Shift the slab base (this is mostly just for aesthetics)
        slab_base.translate_sites(
            vector=horiz_shift,
            frac_coords=True,
        )

        # Round and mod the structure
        slab_base.round(tol=6)

        # Calculate number of empty unit cells are needed for the vacuum
        # Make sure the number is even so the surface can be nicely centered
        # in the vacuum region.
        vacuum_scale = self.vacuum // self.obs.layer_thickness

        if vacuum_scale % 2:
            vacuum_scale += 1

        if vacuum_scale == 0:
            vacuum_scale = 2

        # Get the actuall vacuum in angstroms
        vacuum = self.obs.layer_thickness * vacuum_scale

        # Create the non-orthogonalized surface
        non_orthogonal_slab = utils.get_layer_supercell(
            structure=slab_base.oriented_bulk_structure,
            layers=self.layers,
            vacuum_scale=vacuum_scale,
        )
        non_orthogonal_slab.sort()

        # Center the surfaces within the vacuum region by shifting along c
        center_shift = 0.5 * (vacuum_scale / (vacuum_scale + self.layers))

        non_orthogonal_slab.translate_sites(
            indices=range(len(non_orthogonal_slab)),
            vector=[0, 0, center_shift],
            frac_coords=True,
            to_unit_cell=True,
        )

        return (
            slab_base,
            non_orthogonal_slab,
            vacuum,
            surf_key,
        )

    def _generate_slabs(
        self,
        tol: Optional[float] = None
    ) -> List[Union[Surface, MolecularSurface]]:
        """
        This function is used to generate slab structures with all unique
        surface terminations.

        Returns:
            A list of Surface classes
        """
        # Determine if all possible terminations are generated
        slab_base = self._get_slab_base()
        possible_shifts = self._calculate_possible_shifts(
            structure=slab_base.oriented_bulk_structure,
            tol=tol
        )
        shifted_slab_bases = []
        non_orthogonal_slabs = []
        surface_keys = []

        if not self.generate_all:
            (
                shifted_slab_base,
                non_orthogonal_slab,
                actual_vacuum,
                surf_key,
            ) = self._get_slab(
                slab_base=deepcopy(slab_base),
                shift=possible_shifts[0],
                tol=tol
            )
            non_orthogonal_slab.sort_index = 0
            shifted_slab_bases.append(shifted_slab_base)
            non_orthogonal_slabs.append(non_orthogonal_slab)
            surface_keys.append((surf_key, 0))
        else:
            for i, possible_shift in enumerate(possible_shifts):
                (
                    shifted_slab_base,
                    non_orthogonal_slab,
                    actual_vacuum,
                    surf_key,
                ) = self._get_slab(
                    slab_base=deepcopy(slab_base),
                    shift=possible_shift,
                    tol=tol
                )
                non_orthogonal_slab.sort_index = i
                shifted_slab_bases.append(shifted_slab_base)
                non_orthogonal_slabs.append(non_orthogonal_slab)
                surface_keys.append((surf_key, i))

        surfaces = []

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
            surface = self.surface_type(
                slab=non_orthogonal_slabs[i],  # KEEP
                oriented_bulk=shifted_slab_bases[i],  # KEEP
                miller_index=self.miller_index,  # KEEP
                layers=self.layers,  # KEEP
                vacuum=actual_vacuum,  # KEEP
                termination_index=i,  # KEEP
            )
            surfaces.append(surface)

        return surfaces

    def _calculate_possible_shifts(
        self,
        structure: Structure,
        tol: Optional[float] = None,
    ):
        """
        This function calculates the possible shifts that need to be applied to
        the oriented bulk structure to generate different surface terminations

        Args:
            structure: Oriented bulk structure
            tol: Grouping tolarence in angstroms.
                If None, it will automatically be calculated based on the input
                structure.

        Returns:
            A list of fractional shift values along the c-vector
        """
        frac_coords = structure.frac_coords[:, -1]

        # Projection of c lattice vector in
        # direction of surface normal.
        h = self.obs.layer_thickness

        if tol is None:
            cart_coords = structure.cart_coords
            projected_coords = np.dot(cart_coords, self.obs.surface_normal)
            extended_projected_coords = np.round(
                np.concatenate(
                    [
                        projected_coords - h,
                        projected_coords,
                        projected_coords + h,
                    ]
                ),
                5,
            )
            unique_cart_coords = np.sort(np.unique(extended_projected_coords))
            diffs = np.diff(unique_cart_coords)
            max_diff = diffs.max()
            tol = 0.15 * max_diff

        n = len(frac_coords)

        if n == 1:
            # Clustering does not work when there is only one data point.
            shift = frac_coords[0] + 0.5
            return [shift - math.floor(shift)]

        # We cluster the sites according to the c coordinates. But we need to
        # take into account PBC. Let's compute a fractional c-coordinate
        # distance matrix that accounts for PBC.
        dist_matrix = np.zeros((n, n))

        for i, j in combinations(list(range(n)), 2):
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

from typing import Union, List, TypeVar, Optional
from itertools import combinations

from pymatgen.core.structure import Structure
from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import JmolNN
from ase import Atoms
import networkx as nx
import numpy as np

#from OgreInterface.generate.base_surface_generator import BaseSurfaceGenerator
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








from collections.abc import Sequence
from datetime import datetime
#from itertools import product
import os
import pickle
import shutil
import sys
from time import time
#from typing import Dict, List, Tuple, TypeVar, Union
import warnings

from ase import Atoms
from ase.io import write
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pymatgen.analysis.wulff import WulffShape
#from pymatgen.core.structure import Structure
from pymatgen.core.surface import get_symmetrically_distinct_miller_indices
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy import stats

from OgreInterface import utils
#from OgreInterface.generate.molecular_surface_generator import MolecularSurfaceGenerator
from OgreInterface.generate.surface_generator import SurfaceGenerator
from OgreInterface.surfaces.molecular_surface import MolecularSurface
from OgreInterface.surfaces.oriented_bulk import OrientedBulk
from OgreInterface.surfaces.surface import Surface

from copy import deepcopy
from itertools import combinations, groupby, product
import math
from typing import Dict, List, Optional, Tuple, TypeVar, Union

import networkx as nx
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import CrystalNN, JmolNN
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Molecule, Structure
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
import spglib
from tqdm import tqdm

from OgreInterface.lattice_match import ZurMcGill
#from OgreInterface.surfaces import Interface, Surface

import os
import sys
from time import time

SURFACE_ENERGY_ADJUSTMENT_FACTOR = -16000/2
IMPLEMENTED_SURFACE_ENERGY_METHODS = [
    "OBS",
    "Boettger",
    "slope",
    "intercept"
]
IMPLEMENTED_FILE_FORMATS = {
    "structure":
    {
        "input": ["cif"],
        "output": [
            "cif",
            "aims"
        ]
    },
    "total_energies": ["json"],
    "surface_energies": [],
    "all_data": []
}
DEFAULT_CONVERGENCE_THRESHOLD = 0.0035
#DEFAULT_VACUUM = 10.0
DEFAULT_VACUUM = 60.0
DEFAULT_FILE_TYPES = [
    "slab_structure",
    "oriented_bulk_structure",
    "surface_energies",
    "converged_surface_energies"
]

#"""
control_in_dirs = {
    "010": "331",
    "001": "431",
    "100": "431",
    "011": "331",
    "101": "431",
    "110": "331",
    "01-1": "331",
    "-101": "431",
    "1-10": "331",
    "111": "331",
    "1-11": "331",
    "-111": "331",
    "11-1": "331"
}
#"""

"""
control_in_dirs = {
    "010": "334",
    "001": "433",
    "100": "433",
    "011": "335",
    "101": "434",
    "110": "335",
    "01-1": "335",
    "-101": "433",
    "1-10": "335",
    "111": "336",
    "1-11": "336",
    "-111": "335",
    "11-1": "335"
}
"""

SelfSurfaceEnergy = TypeVar(
    "SelfSurfaceEnergy", bound="SurfaceEnergy"
)

SelfWulff = TypeVar(
    "SelfWulff", bound="Wulff"
)

def timed_method(method):
    def wrap(*args, **kwargs):
        sys.stdout.write(method.__name__ + "() started.\n")
        start_time = time()
        method_output = method(*args, **kwargs)
        duration = round(time() - start_time, 2)
        sys.stdout.write(method.__name__ + "() finished after " + str(duration) + " seconds.\n")
        return method_output
    return wrap
START_TIME = time()
def elapsed_time():
    global START_TIME
    elapsed = time() - START_TIME
    START_TIME = time()
    return elapsed
def lap_stopwatch(prefix: str = ""):
    lap_duration = str(round(elapsed_time(), 4))
    sys.stdout.write(prefix + " lap: " + lap_duration + "\n")
    return lap_duration

#TODO: see about using enumerate() for some loops

"""
def miller_name(miller_index: List[int]) -> str:
    return "".join([str(i) for i in miller_index])
"""
def miller_name(miller_index: List[int]) -> str:
    name = ""
    for hkl in miller_index:
        name += str(hkl)
    return name
#"""

def _get_unique_miller_indices(struc: Structure, max_index: int):
    struc_sg = SpacegroupAnalyzer(struc)
    lattice = struc.lattice
    recip = struc.lattice.reciprocal_lattice_crystallographic
    symmops = struc_sg.get_point_group_operations(cartesian=False)
    planes = set(list(product(range(-max_index, max_index + 1), repeat=3)))
    planes.remove((0, 0, 0))

    reduced_planes = []
    for plane in planes:
        reduced_plane = utils._get_reduced_vector(
            np.array(plane).astype(float)
        )
        reduced_plane = reduced_plane.astype(int)
        reduced_planes.append(tuple(reduced_plane))

    reduced_planes = set(reduced_planes)

    planes_dict = {p: [] for p in reduced_planes}

    for plane in reduced_planes:
        frac_vec = np.array(plane).dot(recip.metric_tensor)
        if plane in planes_dict.keys():
            for i, symmop in enumerate(symmops):
                frac_point_out = symmop.apply_rotation_only(frac_vec)
                point_out = frac_point_out.dot(lattice.metric_tensor)
                point_out = utils._get_reduced_vector(np.round(point_out))
                point_out = tuple(point_out.astype(int))
                planes_dict[plane].append(point_out)
                if point_out != plane:
                    if point_out in planes_dict.keys():
                        del planes_dict[point_out]

    unique_planes = []

    for k in planes_dict:
        equivalent_planes = np.array(list(set(planes_dict[k])))
        diff = np.abs(np.sum(np.sign(equivalent_planes), axis=1))
        like_signs = equivalent_planes[diff == np.max(diff)]
        if len(like_signs) == 1:
            unique_planes.append(like_signs[0])
        else:
            first_max = like_signs[
                np.abs(like_signs)[:, 0]
                == np.max(np.abs(like_signs)[:, 0])
            ]
            if len(first_max) == 1:
                unique_planes.append(first_max[0])
            else:
                second_max = first_max[
                    np.abs(first_max)[:, 1]
                    == np.max(np.abs(first_max)[:, 1])
                ]
                if len(second_max) == 1:
                    unique_planes.append(second_max[0])
                else:
                    unique_planes.append(
                        second_max[
                            np.argmax(np.sign(second_max).sum(axis=1))
                        ]
                    )

    unique_planes = np.vstack(unique_planes)
    sorted_planes = sorted(
        unique_planes, key=lambda x: (np.linalg.norm(x), -np.sign(x).sum())
    )

    return np.vstack(sorted_planes)

def termination_index_name(termination_index: int = None) -> str:
    if termination_index is None:
        return "0"
    return str(termination_index)

class Wulff():
    #Class for constructing, storing, and managing Wulff shapes and containing SurfaceEnergy objects

    def __init__(
        self,
        molecular: bool,
        bulk_file_name: str,
        nums_layers: List[int],
        total_energy_methods: List[str],
        species_name: str = "species",
        surface_energy_methods: List[str] = IMPLEMENTED_SURFACE_ENERGY_METHODS,
        universal_convergence_threshhold: float = None,
        forward_not_central_difference: bool = False,
        vacuum: float = DEFAULT_VACUUM,
        max_index: int = 1,
        projected_direction: List[int] = [1, 1, 1]
    ) -> None:
        # Attribute                                                                                                             Size        Type                                                            Matrix Math?    Dummy Atoms?    Dummified?
        self.molecular = molecular                                                                                              # 1         bool                                                            N/A             N/A             N
        self.species_name = species_name                                                                                        # 1         str                                                             N/A             N/A             N
        self.bulk_file_name = bulk_file_name                                                                                    # 1         str                                                             N/A             N/A             N
        self.bulk_structure = Structure.from_file(bulk_file_name)                                                               # 1         Structure                                                       N/A             N               N
        (self.nums_layers_list,
         self.nums_layers,
         self.num_nums_layers) = self._setup_nums_layers(nums_layers)
        # self.nums_layers_list                                                                                                   L         List[int]                                                       ?               N/A             N
        # self.nums_layers                                                                                                        L         np.ndarray[int]                                                 ?               N/A             N
        # self.num_nums_layers                                                                                                    == L      int                                                             N/A             N/A             N
        self.vacuum = vacuum                                                                                                    # 1         float                                                           N/A             N/A             N
        self.projected_direction = projected_direction                                                                          # 3         List[int]                                                       N               N/A             N
        #sys.stdout.write("221 self.projected_direction = " + str(self.projected_direction) + "\n")
        (self.max_index,
         self.planes,
         self.num_planes) = self._setup_planes(max_index)
        # self.max_index                                                                                                          1         int                                                             N/A             N/A             N
        # self.planes                                                                                                             P x 3     np.ndarray[int]                                                 ?               N/A             N
        # self.num_planes                                                                                                         == P      int                                                             N/A             N/A             N
        self.nums_terminations = np.zeros(self.num_planes)                                                                      # P         np.ndarray[int]                                                 ?
        self.universal_convergence_threshhold = universal_convergence_threshhold                                                    # 1         float                                                           N/A
        self.forward_not_central_difference = forward_not_central_difference                                                    # 1         bool                                                            N/A
        
        self.num_total_energy_methods = len(total_energy_methods)                                                               # == T      int                                                             N/A
        self.num_surface_energy_methods = len(surface_energy_methods)                                                           # == S      int                                                             N/A

        self.total_energy_methods = total_energy_methods                                                                        # T         List[str]                                                       N
        self.surface_energy_methods = surface_energy_methods                                                                    # S         List[str]                                                       N
        self.surface_energies = np.full((self.num_planes, self.num_total_energy_methods, self.num_surface_energy_methods), 0.0) # P x T x S np.ndarray[float]                                               ?
        self.Wulff_shapes = np.full((self.num_total_energy_methods, self.num_surface_energy_methods), None)                     # T x S     np.ndarray[WulffShape]                                          ?

        (self.surface_energy_objects,
         self.surface_generators,
         self.nums_terminations,
         self.max_num_terminations) = self._setup_surface_energy_objects()
        # self.surface_energy_objects                                                                                             P x term  np.ndarray[SurfaceEnergy]                                       ?
        # self.surface_generators                                                                                                 P x L     np.ndarray[Union[SurfaceGenerator, MolecularSurfaceGenerator]]  ?
        # self.nums_terminations                                                                                                  P         List[int]                                                       N
        # self.max_num_terminations                                                                                               1         int                                                             N/A
        #sys.stdout.write("self.surface_energy_objects.shape = " + str(self.surface_energy_objects.shape) + "\n")
        #sys.stdout.write("self.surface_generators.shape = " + str(self.surface_generators.shape) + "\n")
        #sys.stdout.write("len(self.nums_terminations) = " + str(len(self.nums_terminations)) + "\n")

    def _setup_nums_layers(
        self,
        input_nums_layers: List[int]
    ) -> Tuple[List[int],
               np.ndarray,
               int]:
        nums_layers_list = sorted(input_nums_layers)
        nums_layers = np.array(nums_layers_list)
        num_nums_layers = len(nums_layers_list)
        return (nums_layers_list,
                nums_layers,
                num_nums_layers)

    def _setup_planes(
        self,
        max_index: int
    ) -> Tuple[int,
               np.ndarray,
               int]:
        #planes = utils.get_unique_miller_indices(self.bulk_structure, max_index)
        planes = _get_unique_miller_indices(self.bulk_structure, max_index)
        num_planes = len(planes)
        return (max_index,
                planes,
                num_planes)

    def _setup_surface_energy_objects(
        self
    ) -> Tuple[np.ndarray,
               np.ndarray,
               List[int],
               int]:
        # Make SurfaceEnergy objects
        surface_generators = []
        nums_terminations = []

        if self.molecular:
            for plane in self.num_planes:
                plane_surface_generators = []
                for layer_index in range(self.num_nums_layers):
                    # This would be more efficient if there was a MolecularSurfaceGenerator.from_bulk_structure() class method
                    plane_surface_generators.append(
                        MolecularSurfaceGenerator.from_file(
                            filename=self.bulk_file_name,
                            miller_index=plane,
                            layers=self.nums_layers[layer_index],
                            vacuum=self.vacuum,
                            refine_structure=True,
                            generate_all=True,
                            lazy=False
                        )
                    )
                surface_generators.append(plane_surface_generators)
                nums_terminations.append(plane_surface_generators[0][-1].termination_index + 1)
            #sys.stdout.write("nums_terminations = " + str(nums_terminations) + "\n")
            #sys.stdout.write("self.planes = " + str(self.planes) + "\n")
            max_termination_index = max(nums_terminations)
            surface_energy_objects = np.empty((self.num_planes, max_termination_index), dtype=object)

            for plane_index in range(self.num_planes):
                for termination_index in range(nums_terminations[plane_index]):
                    surface_energy_objects[plane_index, termination_index] = SurfaceEnergy.from_molecular_surface_generators(
                        molecular_surface_generators=surface_generators[plane_index],
                        nums_layers=self.nums_layers_list,
                        total_energy_methods=self.total_energy_methods,
                        species_name=self.species_name,
                        surface_energy_methods=self.surface_energy_methods,
                        universal_convergence_threshhold=self.universal_convergence_threshhold,
                        forward_not_central_difference=self.forward_not_central_difference,
                        termination_index=termination_index
                        )
        else:
            for plane_index in range(self.num_planes):
                plane_surface_generators = []
                for layer_index in range(self.num_nums_layers):
                    # This would be more efficient if there was a SurfaceGenerator.from_bulk_structure() class method
                    plane_surface_generators.append(
                        SurfaceGenerator.from_file(
                            filename=self.bulk_file_name,
                            miller_index=self.planes[plane_index],
                            layers=self.nums_layers[layer_index],
                            vacuum=self.vacuum,
                            refine_structure=True,
                            generate_all=True,
                            lazy=False
                        )
                    )
                
                surface_generators.append(plane_surface_generators)
                nums_terminations.append(plane_surface_generators[0][-1].termination_index + 1)

            max_termination_index = max(nums_terminations)
            surface_energy_objects = np.empty((self.num_planes, max_termination_index), dtype=object)

            for plane_index in range(self.num_planes):
                for termination_index in range(nums_terminations[plane_index]):
                    surface_energy_objects[plane_index, termination_index] = SurfaceEnergy.from_surface_generators(
                        surface_generators=surface_generators[plane_index],
                        nums_layers=self.nums_layers_list,
                        total_energy_methods=self.total_energy_methods,
                        species_name=self.species_name,
                        surface_energy_methods=self.surface_energy_methods,
                        universal_convergence_threshhold=self.universal_convergence_threshhold,
                        forward_not_central_difference=self.forward_not_central_difference,
                        termination_index=termination_index
                    )

        return (surface_energy_objects,
                np.array(surface_generators),
                nums_terminations,
                max_termination_index)

    def generate_structure_files(
        self,
        base_slab_file_name: str = "",
        slab_file_names: List[str] = None,
        obs_file_name: str = None,
        file_format: str = None,
        generated_structures_directory: str = "generated_structures",
        prep_for_calc: bool = True,
        control_in: str = "control.in",
        submission_script: str = "submit.sh",
        overwrite: bool = False
    ) -> None:
        #sys.stdout.write("self.surface_energy_objects: " + str(self.surface_energy_objects) + "\n")
        for plane_index in range(self.num_planes):
            for termination_index in range(self.nums_terminations[plane_index]):
                surface_energy_object = self.surface_energy_objects[plane_index, termination_index]
                #surface_energy_object.generate_structure_files(prep_for_calc=True, control_in=os.path.join("aims_ins", control_in_dirs[surface_energy_object.miller_name], "control.in"), submission_script=os.path.join("aims_ins", "submit.sh"))
                surface_energy_object.generate_structure_files(
                    base_slab_file_name=base_slab_file_name,
                    slab_file_names=slab_file_names,
                    obs_file_name=obs_file_name,
                    file_format=file_format,
                    generated_structures_directory=generated_structures_directory,
                    prep_for_calc=prep_for_calc,
                    control_in=control_in,
                    submission_script=submission_script,
                    overwrite=overwrite)

    def read_total_energies(
        self,
        generated_structures_directory: str = "generated_structures",
        species_directory: str = None,
        miller_directory: str = None,
        termination_directory: str = None,
        obs_directory: str = None
    ) -> None:
        for plane_index in range(self.num_planes):
            for termination_index in range(self.nums_terminations[plane_index]):
                self.surface_energy_objects[plane_index, termination_index].read_aims_out_total_energies(
                    generated_structures_directory=generated_structures_directory,
                    species_directory=species_directory,
                    miller_directory=miller_directory,
                    termination_directory=termination_directory,
                    obs_directory=obs_directory)

    def calculate(
        self,
        plot: bool = False,
        projected_direction: List[int] = None,
        save: bool = False,
        Wulff_directory: str = "Wulff_shapes",
        color_set: str = "PuBu",
        grid_off: bool = True,
        axis_off: bool = True,
        show_area: bool = False,
        alpha: float = 1.0,
        off_color: str = "red",
        bar_pos: Tuple[float] = (0.75, 0.15, 0.05, 0.65),
        bar_on: bool = False,
        units_in_JPERM2: bool = True,
        legend_on: bool = True,
        aspect_ratio: Tuple[int] = (8, 8),
        custom_colors: Dict = None
    ) -> None:
        if projected_direction is None:
            projected_direction = self.projected_direction

        self._calculate_surface_energies()
        self._construct_Wulff_shapes()
        if plot:
            self.plot_Wulff_shapes(projected_direction=self.projected_direction,
                                   save=save,
                                   Wulff_directory=Wulff_directory,
                                   color_set=color_set,
                                   grid_off=grid_off,
                                   axis_off=axis_off,
                                   show_area=show_area,
                                   alpha=alpha,
                                   off_color=off_color,
                                   bar_pos=bar_pos,
                                   bar_on=bar_on,
                                   units_in_JPERM2=units_in_JPERM2,
                                   legend_on=legend_on,
                                   aspect_ratio=aspect_ratio,
                                   custom_colors=custom_colors)

    def _calculate_surface_energies(self) -> None:
        for plane_index in range(self.num_planes):
            for termination_index in range(self.nums_terminations[plane_index]):
                self.surface_energy_objects[plane_index, termination_index].calculate_surface_energies()

        for plane_index in range(self.num_planes):
            terminations_surface_energies = np.full((self.nums_terminations[plane_index], self.num_total_energy_methods, self.num_surface_energy_methods), 0.0)
            for termination_index in range(self.nums_terminations[plane_index]):
                terminations_surface_energies[termination_index, ...] = self.surface_energy_objects[plane_index, termination_index].converged_surface_energies
            self.surface_energies[plane_index] = np.min(terminations_surface_energies, axis=0)

    def _construct_Wulff_shapes(self) -> None:
        # Make, plot, and save Wulff shapes
        for total_energy_method_index in range(self.num_total_energy_methods):
            for surface_energy_method_index in range(self.num_surface_energy_methods):
                self.Wulff_shapes[total_energy_method_index, surface_energy_method_index] = WulffShape(self.bulk_structure.lattice, self.planes, self.surface_energies[:, total_energy_method_index, surface_energy_method_index].tolist())

    def plot_Wulff_shapes(
        self,
        projected_direction: List[int] = None,
        save: bool = False,
        Wulff_directory: str = "Wulff_shapes",
        color_set: str = "PuBu",
        grid_off: bool = True,
        axis_off: bool = True,
        show_area: bool = False,
        alpha: float = 1.0,
        off_color: str = "red",
        bar_pos: Tuple[float] = (0.75, 0.15, 0.05, 0.65),
        bar_on: bool = False,
        units_in_JPERM2: bool = True,
        legend_on: bool = True,
        aspect_ratio: Tuple[int] = (8, 8),
        custom_colors: Dict = None
    ) -> None:
        if projected_direction is not None:
            self.projected_direction = projected_direction
            #sys.stdout.write("409 direction = " + str(direction) + "\n")
        else:
            projected_direction = self.projected_direction
            #sys.stdout.write("412 self.projected_direction = " + str(self.projected_direction) + "\n")
        if not os.path.isdir(Wulff_directory):
            os.mkdir(Wulff_directory)
        species_directory = os.path.join(Wulff_directory, self.species_name)
        if not os.path.isdir(species_directory):
            os.mkdir(species_directory)

        for total_energy_method_index in range(self.num_total_energy_methods):
            total_energy_method_directory = os.path.join(species_directory, self.total_energy_methods[total_energy_method_index])
            if not os.path.isdir(total_energy_method_directory):
                os.mkdir(total_energy_method_directory)

            for surface_energy_method_index in range(self.num_surface_energy_methods):
                surface_energy_method_directory = os.path.join(total_energy_method_directory, self.surface_energy_methods[surface_energy_method_index])
                if not os.path.isdir(surface_energy_method_directory):
                    os.mkdir(surface_energy_method_directory)

                plt.figure()
                #sys.stdout.write("direction : " + str(direction) + " ; type: " + str(type(direction)) + "\n")
                self.Wulff_shapes[total_energy_method_index, surface_energy_method_index].get_plot(direction=projected_direction,
                                                                                                   color_set=color_set,
                                                                                                   grid_off=grid_off,
                                                                                                   axis_off=axis_off,
                                                                                                   show_area=show_area,
                                                                                                   alpha=alpha,
                                                                                                   off_color=off_color,
                                                                                                   bar_pos=bar_pos,
                                                                                                   bar_on=bar_on,
                                                                                                   units_in_JPERM2=units_in_JPERM2,
                                                                                                   legend_on=legend_on,
                                                                                                   aspect_ratio=aspect_ratio,
                                                                                                   custom_colors=custom_colors)
                
                if save:
                    plt.savefig(os.path.join(surface_energy_method_directory, "Wulff_{}_threshhold_{}.png".format(miller_name(projected_direction), str(self.universal_convergence_threshhold))),
                                dpi=400, bbox_inches="tight")

    def to_pickle(
        self,
        file_path: str = "Wulff_object_pickle"
    ) -> None:
        with open(file_path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle(
        cls,
        file_path: str
    ) -> SelfWulff:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    """
    @classmethod
    @timed_method
    def from_bulk_file(
        cls,
        bulk_file_name: str,
        nums_layers: List[int],
        total_energy_methods: List[str],
        molecular: bool = True,
        surface_energy_methods: List[str] = IMPLEMENTED_SURFACE_ENERGY_METHODS,
        universal_convergence_treshold: float = None,
        forward_not_central_difference: bool = False,
        vacuum: float = DEFAULT_VACUUM,
        max_index: int = 1,
        projected_direction: List[int] = [1, 1, 1]
    ) -> SelfWulff:
        if molecular:
            for 
            
            
            molecular_surface_generators = []
            for num_layers in nums_layers:
                start_time = time()
                sys.stdout.write("Creating MolecularSurfaceGenerator object for slab with " + str(num_layers) + " layers...")
                molecular_surface_generators.append(
                    MolecularSurfaceGenerator.from_file(
                        filename=bulk_file_name,
                        miller_index=miller_index,
                        layers=num_layers,
                        vacuum=vacuum,
                        generate_all=True, # TODO: Handle nonzero termination indices
                        lazy=False,
                    )
                )
                duration = round(time() - start_time, 2)
                sys.stdout.write("finished after " + str(duration) + " seconds.\n")

            return cls.from_molecular_surface_generators(
                molecular_surface_generators,
                nums_layers,
                total_energy_methods,
                surface_energy_methods,
                universal_convergence_treshold,
                forward_not_central_difference,
                
            )
        
        surface_generators = []
        for num_layers in nums_layers:
            start_time = time()
            sys.stdout.write("Creating SurfaceGenerator object for slab with " + str(num_layers) + " layers...")
            surface_generators.append(
                SurfaceGenerator.from_file(
                    filename=bulk_file_name,
                    miller_index=miller_index,
                    layers=num_layers,
                    vacuum=vacuum,
                    refine_structure=False, #TODO: Make sure this is right
                    generate_all=generate_all, # TODO: Handle nonzero termination indices
                    lazy=False,
                    suppress_warnings=False
                )
            )
            duration = round(time() - start_time, 2)
            sys.stdout.write("finished after " + str(duration) + " seconds.\n")

        return cls.from_surface_generators(
            surface_generators,
            nums_layers,
            total_energy_methods,
            molecular,
            surface_energy_methods,
            universal_convergence_treshold,
            forward_not_central_difference,
            termination_index
        )

    @classmethod
    @timed_method
    def from_bulk_structure_or_atoms(
        cls,
        bulk_structure_or_atoms: Union[Structure, Atoms],
        miller_index: List[int],
        nums_layers: List[int],
        total_energy_methods: List[str],
        surface_energy_methods: List[str] = IMPLEMENTED_SURFACE_ENERGY_METHODS,
        universal_convergence_treshold: float = None,
        forward_not_central_difference: bool = False,
        vacuum: float = DEFAULT_VACUUM,
        termination_index: int = 0
    ) -> SelfWulff:
        # TODO: Handle case where len(nums_layers) == 1 in a tidier way; consider messing with how nums_layers is defined and/or accessed.
        generate_all = (termination_index != 0)
        
        surface_generators = []
        for num_layers in nums_layers:
            surface_generators.append(
                SurfaceGenerator(
                    bulk=bulk_structure_or_atoms,
                    miller_index=miller_index,
                    layers=num_layers,
                    vacuum=vacuum,
                    refine_structure=False, #TODO: Make sure this is right
                    generate_all=generate_all, # TODO: Handle nonzero termination indices
                    lazy=False,
                    suppress_warnings=False
                )
            )

        return cls.from_surface_generators(
            surface_generators,
            nums_layers,
            total_energy_methods,
            surface_energy_methods,
            universal_convergence_treshold,
            forward_not_central_difference,
            termination_index
        )

    @classmethod
    def from_surface_energies(
        cls
    ) -> None:
        pass
    """
        
# Should I bother inheriting from Sequence? I'm going to try not to for now.
class SurfaceEnergy():
    """Class for computing, storing, and managing surface energy data."""

    #@timed_method
    def __init__(
        self,
        molecular: bool,
        nums_layers: List[int],
        total_energy_methods: List[str],
        species_name: str = "species",
        surface_energy_methods: Union[List[str], Dict] = IMPLEMENTED_SURFACE_ENERGY_METHODS,
        universal_convergence_threshhold: float = None,
        forward_not_central_difference: bool = False
    ) -> None:
        # Attribute                                                                                                                     Size        Type                    Matrix Math?
        
        # Flag attributes
        self.molecular = molecular                                                                                                      # 1         bool                    N/A
        """self.empty = empty                                                                                                              # 1         bool                    N/A"""

        # Size attributes
        self.num_nums_layers = len(nums_layers)                                                                                         # == L      int                     N/A
        self.num_total_energy_methods = len(total_energy_methods)                                                                       # == T      int                     N/A
        # self.num_surface_energy_methods                                                                                                 1         int                     N/A

        # Attributes for generating structures for total energy calculations
        self.bulk_structure = None                                                                                                      # 1         Structure               N/A
        self.miller_index = None                                                                                                        # 3 ("1")   List[int]               ?
        self.nums_layers_list = sorted(nums_layers)                                                                                     # L         List[int]               ?
        self.nums_layers = np.array(self.nums_layers_list)                                                                              # L         np.ndarray[int]         Y
        self.vacuum = None                                                                                                              # 1         float                   N/A
        self.surface_generators = None                                                                                                  # L         List[SurfaceGenerator]  N
        self.termination_index = None                                                                                                   # 1         int                     N/A
        self.surfaces = None                                                                                                            # L         List[Surface]           N

        # Attributes representing structures for total energy calculations
        self.slab_structures = None                                                                                                     # L         List[Structure]         N
        self.slab_atoms = None                                                                                                          # L         List[Atoms]             N
        self.oriented_bulk = None                                                                                             # 1         Structure               N/A
        #self.oriented_bulk_atoms = None                                                                                                 # 1         Atoms                   N/A

        # Attributes representing total energies (for surface energy calculations)
        self.total_energies = np.full((self.num_nums_layers, self.num_total_energy_methods), 0.0)                                       # L x T     np.ndarray[float]       Y
        self.obs_total_energies = np.full(self.num_total_energy_methods, 0.0)   #NOTE: PER LAYER                                        # T         np.ndarray[float]       Y

        # Attributes for calculating surface energies
        self.total_energy_methods = total_energy_methods                                                                                # T         List[str]               N
        (self.surface_energy_methods,
         self.convergence_threshholds,
         self.num_surface_energy_methods
        ) = self._setup_surface_energy_methods(
            surface_energy_methods,
            universal_convergence_threshhold)
        # self.surface_energy_methods                                                                                                     == S      List[str]               N
        # self.convergence_thresholds                                                                                                     N/A       Dict                    N/A
        self.forward_not_central_difference = forward_not_central_difference                                                            # 1         bool                    N/A
        self.area = None                                                                                                                # 1         float                   N/A

        # Attributes representing surface energies
        self.surface_energies = np.full((self.num_nums_layers, self.num_total_energy_methods, self.num_surface_energy_methods), 0.0)    # L x T x S np.ndarray[float]       Y
        self.converged_surface_energies = np.full((self.num_total_energy_methods, self.num_surface_energy_methods), 0.0)                # T x S     np.ndarray[float]       Y?

        # Attributes representing components of default file names
        self.species_name = species_name
        self.miller_name = ""
        self.termination_index_name = ""
        self._default_file_name_chunks = self._setup_default_file_name_chunks()                                                         # N/A       Dict                    N/A

    # private setup methods
    @timed_method
    def _setup_surface_energy_methods(
        self,
        surface_energy_methods: Union[List[str], Dict],
        universal_convergence_threshold: float = None
    ) -> Tuple[List[str], Dict, int]:
        if universal_convergence_threshold is None:
            universal_convergence_threshold = DEFAULT_CONVERGENCE_THRESHOLD

        if type(surface_energy_methods) is list:
            return (
                surface_energy_methods,
                {
                    "OBS": universal_convergence_threshold,
                    "Boettger": 
                        {
                            "d_surface_energies_d_nums_layers": universal_convergence_threshold,
                            "d_total_energies_d_nums_layers": universal_convergence_threshold
                        },
                    "slope": universal_convergence_threshold,
                    "intercept": universal_convergence_threshold
                },
                len(surface_energy_methods)
            )
        
        return (
            [x for x in surface_energy_methods.keys()],
            surface_energy_methods,
            len(surface_energy_methods.keys())
        )

    @timed_method
    def _setup_default_file_name_chunks(self) -> Dict:
        chunks = {}
        for file_type in DEFAULT_FILE_TYPES:
            chunks[file_type] = {}
            chunks[file_type]["middle"] = ""
            chunks[file_type]["suffix"] = ".cif"

            if file_type == "surface_energies" or file_type == "converged_surface_energies":
                for num_layers in self.nums_layers_list:
                    chunks[file_type]["middle"] += str(num_layers) + "_"
                chunks[file_type]["middle"] = chunks[file_type]["middle"][:-1]
                chunks[file_type]["middle"] += "-"

                for total_energy_method in self.total_energy_methods:
                    chunks[file_type]["middle"] += total_energy_method + "_"
                chunks[file_type]["middle"] = chunks[file_type]["middle"][:-1]
                chunks[file_type]["middle"] += "-"

                for surface_energy_method in self.surface_energy_methods:
                    chunks[file_type]["middle"] += surface_energy_method + "_"
                chunks[file_type]["middle"] = chunks[file_type]["middle"][:-1]
                chunks[file_type]["middle"] += "-"

                chunks[file_type]["suffix"] = ".json"

        return chunks

    # private methods
    #TODO: handle difference quotients more efficiently with numpy matrix math
    @timed_method
    def _difference_quotient(
        self,
        x: Union[np.ndarray, List[int]],
        y: Union[np.ndarray, List[float]],
        index: int,
        forward_not_central: bool,
    ) -> float:

        high_index = index + 1
        if forward_not_central:
            low_index = index
        else:
            low_index = index - 1

        return np.divide(np.subtract(y[high_index], y[low_index]), np.subtract(x[high_index], x[low_index]))

    @timed_method
    def _first_first_derivative_threshold_satisfier(
        self,
        x: np.ndarray,
        y: np.ndarray,
        forward_not_central: bool,
        threshold: float
    ) -> int:
        if forward_not_central:
            first_index = 0
            low_index_adjustment = 0
        else:
            first_index = 1
            low_index_adjustment = -1
        for index in range(first_index, len(x) - 1):
            # Skip the current index if either of the respective y-values to be checked are zero (so that unassigned values do not prematurely trigger the convergence check) or if _difference_quotient() would divide by zero
            if y[index + low_index_adjustment] == 0 or y[index + 1] == 0 or x[index + low_index_adjustment] == x[index + 1]:
                continue
            if self._difference_quotient(x, y, index, forward_not_central) <= threshold:
                return index
        return -1

    @timed_method
    def _first_second_derivative_threshold_satisfier(
        self,
        x: np.ndarray,
        y: np.ndarray,
        forward_not_central: bool,
        threshold: float
    ) -> int:
        # Assumes x's elements are in order
        if forward_not_central:
            first_index = 0
        else:
            first_index = 1
        
        previous_difference_quotient = self._difference_quotient(x, y, first_index, forward_not_central)
        for index in range(first_index + 1, len(x) - 1):
            difference_quotient = self._difference_quotient(x, y, index, forward_not_central)
            if previous_difference_quotient == 0:
                previous_difference_quotient = difference_quotient
                continue

            if np.abs((difference_quotient - previous_difference_quotient) / previous_difference_quotient) <= threshold:
                return index - 1
            
            previous_difference_quotient = difference_quotient

        return -1

    @timed_method
    def _surface_energies_from_total_energies_and_bulk_energy(
        self,
        total_energies: np.ndarray,
        bulk_energy: float
    ) -> np.ndarray:
        return 0.5 * np.divide(np.subtract(total_energies, bulk_energy * self.nums_layers), self.area)

    @timed_method
    def _OBS_surface_energies(self) -> None:
        #self.surface_energies[..., self.surface_energy_methods.index("OBS")] = 0.5 * np.divide(np.subtract(self.total_energies.T, np.outer(self.obs_total_energies, self.nums_layers)), self.areas).T
        surface_energy_method_index = self.surface_energy_methods.index("OBS")
        self.surface_energies[..., surface_energy_method_index] = 0.5 * np.divide(np.subtract(self.total_energies.T, np.outer(self.obs_total_energies, self.nums_layers)), self.area).T
        
        if self.num_nums_layers < 2 or (self.num_nums_layers < 3 and not self.forward_not_central_difference):
            return
        
        for total_energy_method_index in range(self.num_total_energy_methods):
            total_energies = self._get_total_energies_all_nums_layers(total_energy_method_index)
            # These two are equivalent. Test which is faster. Intuition tells me the latter should be faster, thanks to np.outer and only one round of calculations. But maybe the former would be faster thanks to no transposing.
            #surface_energies = self._surface_energies_from_total_energies_and_bulk_energy(total_energies, self.obs_total_energies[total_energy_method_index])
            surface_energies = self.surface_energies[:, total_energy_method_index, surface_energy_method_index]

            index_conv = self._first_first_derivative_threshold_satisfier(
                x=self.nums_layers,
                y=surface_energies,
                forward_not_central=self.forward_not_central_difference,
                threshold=self.convergence_threshholds["OBS"]
            )

            if index_conv == -1:
                self.converged_surface_energies[total_energy_method_index, surface_energy_method_index] = -1
            else:
                self.converged_surface_energies[total_energy_method_index, surface_energy_method_index] = surface_energies[index_conv]

    #TODO: further collapse with numpy functions
    @timed_method
    def _Boettger_surface_energies(self) -> None:
        if self.num_nums_layers < 2 or (self.num_nums_layers < 3 and not self.forward_not_central_difference):
            return
        
        surface_energy_method_index = self.surface_energy_methods.index("Boettger")
        
        for total_energy_method_index in range(self.num_total_energy_methods):          
            self.surface_energies[:, total_energy_method_index, surface_energy_method_index] = np.full(self.num_nums_layers, -1)
            self.converged_surface_energies[total_energy_method_index, surface_energy_method_index] = -1

            total_energies = self._get_total_energies_all_nums_layers(total_energy_method_index)

            index_prime = self._first_second_derivative_threshold_satisfier(
                x=self.nums_layers,
                y=total_energies,
                forward_not_central=self.forward_not_central_difference,
                threshold=self.convergence_threshholds["Boettger"]["d_total_energies_d_nums_layers"]
            )
            if index_prime == -1:
                continue

            for index_prime_prime in range(index_prime + 1, self.num_nums_layers):
                bulk_energy = (total_energies[index_prime_prime] - total_energies[index_prime]) / (self.nums_layers[index_prime_prime] - self.nums_layers[index_prime])
                surface_energies = self._surface_energies_from_total_energies_and_bulk_energy(total_energies, bulk_energy)

                index_conv = self._first_first_derivative_threshold_satisfier(
                    x=self.nums_layers,
                    y=surface_energies,
                    forward_not_central=self.forward_not_central_difference,
                    threshold=self.convergence_threshholds["Boettger"]["d_surface_energies_d_nums_layers"]
                )

                if index_conv != -1:
                    self.surface_energies[:, total_energy_method_index, surface_energy_method_index] = surface_energies
                    self.converged_surface_energies[total_energy_method_index, surface_energy_method_index] = surface_energies[index_conv]
                    break

    #TODO: further collapse with numpy functions
    @timed_method
    def _slope_surface_energies(self) -> None:
        if self.num_nums_layers < 2 or (self.num_nums_layers < 3 and not self.forward_not_central_difference):
            return
        
        surface_energy_method_index = self.surface_energy_methods.index("slope")
        
        for total_energy_method_index in range(self.num_total_energy_methods):
            self.surface_energies[:, total_energy_method_index, surface_energy_method_index] = np.full(self.num_nums_layers, -1)
            self.converged_surface_energies[total_energy_method_index, surface_energy_method_index] = -1

            total_energies = self._get_total_energies_all_nums_layers(total_energy_method_index)

            for index_prime in range(self.num_nums_layers - 1):             
                bulk_energy, intercept, r_value, p_value, standard_error = stats.linregress(
                    self.nums_layers[index_prime:],
                    total_energies[index_prime:]
                )
                surface_energies = self._surface_energies_from_total_energies_and_bulk_energy(total_energies, bulk_energy)

                index_conv = self._first_first_derivative_threshold_satisfier(
                    x=self.nums_layers,
                    y=surface_energies,
                    forward_not_central=self.forward_not_central_difference,
                    threshold=self.convergence_threshholds["slope"]
                )

                if index_conv != -1:
                    self.surface_energies[:, total_energy_method_index, surface_energy_method_index] = surface_energies
                    self.converged_surface_energies[total_energy_method_index, surface_energy_method_index] = surface_energies[index_conv]
                    break
    
    #TODO: further collapse with numpy functions
    @timed_method
    def _intercept_surface_energies(self) -> None:
        if self.num_nums_layers < 3:
            return
        
        surface_energy_method_index = self.surface_energy_methods.index("intercept")
        
        for total_energy_method_index in range(self.num_total_energy_methods):
            self.surface_energies[:, total_energy_method_index, surface_energy_method_index] = np.full(self.num_nums_layers, -1)
            self.converged_surface_energies[total_energy_method_index, surface_energy_method_index] = -1

            total_energies = self._get_total_energies_all_nums_layers(total_energy_method_index)

            for index_prime in range(self.num_nums_layers - 2):             
                intercepts = np.zeros(self.num_nums_layers)
                for index_prime_prime in range(index_prime + 2, self.num_nums_layers):
                    bulk_energy, intercepts[index_prime_prime], r_value, p_value, standard_error = stats.linregress(
                        self.nums_layers[index_prime:index_prime_prime],
                        total_energies[index_prime:index_prime_prime]
                    )
                surface_energies = 0.5 * np.divide(intercepts, self.area)

                index_conv = self._first_first_derivative_threshold_satisfier(
                    x=self.nums_layers,
                    y=surface_energies,
                    forward_not_central=self.forward_not_central_difference,
                    threshold=self.convergence_threshholds["intercept"]
                )

                if index_conv != -1:
                    self.surface_energies[:, total_energy_method_index, surface_energy_method_index] = surface_energies
                    self.converged_surface_energies[total_energy_method_index, surface_energy_method_index] = surface_energies[index_conv]
                    break

    @timed_method
    def _default_file_name(
        self,
        file_type: str,
        num_layers: int = None
    ) -> str:        
        if num_layers is None or num_layers not in self.nums_layers_list:
            num_layers = self.nums_layers[0]
        file_name = file_type + "-" + self.species_name + "-" + self.miller_name + "-" + self.termination_index_name + "-" + self._default_file_name_chunks[file_type]["middle"]
        if file_type == "slab_structure":
            file_name += "-" + str(num_layers) + "-"
        current_datetime = datetime.now()
        file_name += current_datetime.strftime("%d_%m_%Y_%H_%M_%S")
        file_name += self._default_file_name_chunks[file_type]["suffix"]

        return file_name

    # public methods
    """ Most important file formats to support:
            geometry.in FHI-aims        second implementation
            .cif        many   
            POSCAR      VASP            first implementation
            ...?
    """
    @timed_method
    def generate_structure_files(
        self,
        base_slab_file_name: str = "",
        slab_file_names: List[str] = None,
        obs_file_name: str = None,
        file_format: str = None,
        generated_structures_directory: str = "generated_structures",
        prep_for_calc: bool = True,
        control_in: str = "control.in",
        submission_script: str = "submit.sh",
        overwrite: bool = False
    ) -> None:
        do_obs = True
        species_directory = os.path.join(generated_structures_directory, self.species_name)
        miller_directory = os.path.join(species_directory, self.miller_name)
        termination_directory = os.path.join(miller_directory, self.termination_index_name)
        obs_directory = os.path.join(miller_directory, "OBS")
        
        if not os.path.isdir(generated_structures_directory):
            os.mkdir(generated_structures_directory)
        if not os.path.isdir(species_directory):
            os.mkdir(species_directory)
        if not os.path.isdir(miller_directory):
            os.mkdir(miller_directory)
        if not os.path.isdir(termination_directory):
            os.mkdir(termination_directory)
        if not os.path.isdir(obs_directory):
            os.mkdir(obs_directory)

        if file_format is None:
            file_format = "aims"
        if file_format == "in" or file_format == ".in" or file_format == "geometry.in" or file_format == "geometry" or file_format == "aims" or file_format == "Aims" or file_format == "AIMS" or file_format == "fhi-aims" or file_format == "FHI-aims" or file_format == "FHI-Aims" or file_format == "FHI-AIMS" or file_format == "FHI" or file_format == "fhi":
            file_format = "aims"
            if obs_file_name is None:
                obs_file_name = "geometry.in"
                if prep_for_calc:
                    shutil.copyfile(control_in, os.path.join(obs_directory, "control.in"))
                    shutil.copyfile(submission_script, os.path.join(obs_directory, "submit.sh"))
            if slab_file_names is None:
                slab_file_names = []
                for num_layers in self.nums_layers_list:
                    slab_file_names.append("geometry.in")
        else:
            if obs_file_name is None:
                obs_file_name = self._default_file_name(file_type="oriented_bulk_structure")
            if slab_file_names is None:
                slab_paths = []
                for num_layers in self.nums_layers_list:
                    slab_paths.append(os.path.join("slabs", base_slab_file_name + self._default_file_name(file_type="slab_structure", num_layers=num_layers)))

        if os.path.isfile(os.path.join(obs_directory, obs_file_name)) and not overwrite:
            do_obs = False
        
        slab_paths = []
        for layer_index in range(self.num_nums_layers):
            num_layers_directory = os.path.join(termination_directory, str(self.nums_layers[layer_index]))
            slab_paths.append(os.path.join(num_layers_directory, slab_file_names[layer_index]))
            if not os.path.isdir(num_layers_directory):
                os.mkdir(num_layers_directory)
            if prep_for_calc:
                shutil.copyfile(control_in, os.path.join(num_layers_directory, "control.in"))
                shutil.copyfile(submission_script, os.path.join(num_layers_directory, "submit.sh"))
        
        """        
        directory = os.path.join(directory_name, self.miller_name, str(self.termination_index))
        slabs_directory = os.path.join(directory, slabs_directory_name)
        obs_directory = os.path.join(directory, obs_directory_name)

        if not os.path.isdir(directory):
            os.mkdir(directory)
        if not os.path.isdir(slabs_directory):
            os.mkdir(slabs_directory)
        if not os.path.isdir(obs_directory):
            os.mkdir(obs_directory)


        obs_path = os.path.join(obs_directory, obs_file_name)

        slab_paths = []
        if file_format == "aims":
            for num_layers in self.nums_layers_list:
                num_layers_directory = os.path.join(slabs_directory, str(num_layers) + "_layers")
                if not os.path.isdir(num_layers_directory):
                    os.mkdir(num_layers_directory)
                slab_paths.append(os.path.join(num_layers_directory, "geometry.in"))
                if prep_for_calc:
                    shutil.copyfile(control_in, os.path.join(num_layers_directory, "control.in"))
                    shutil.copyfile(submission_script, os.path.join(num_layers_directory, "submit.sh"))
            obs_path = os.path.join(obs_directory, "geometry.in")
            shutil.copyfile(control_in, os.path.join(obs_directory, "control.in"))
            shutil.copyfile(submission_script, os.path.join(obs_directory, "submit.sh"))

        elif slab_file_names is None:
            for num_layers in self.nums_layers_list:
                slab_paths.append(os.path.join(slabs_directory, base_slab_file_name + self._default_file_name(file_type="slab_structure", num_layers=num_layers)))
        
        elif len(slab_file_names) < self.num_nums_layers:
            # TODO: Handle invalid number of slab file names
            for num_layers in self.nums_layers_list:
                slab_paths.append(os.path.join(slabs_directory, base_slab_file_name + self._default_file_name(file_type="slab_structure", num_layers=num_layers)))
        
        else:
            for index in range(self.num_nums_layers):
                slab_paths.append(os.path.join(slabs_directory, base_slab_file_name + slab_file_names[index]))
        # TODO: Clean up the above tree; use supplied names until either num_nums_layers is reached or you run out, in which case use default names for the rest
        """
        if self.molecular:
            # Dummy atom slab structures must be converted to true molecular structures before the respective structure files are generated
            for layer_index in range(self.num_nums_layers):
                write(slab_paths[layer_index], AseAtomsAdaptor.get_atoms(utils.add_molecules(self.slab_structures[layer_index])), file_format)
            if do_obs:
                write(os.path.join(obs_directory, obs_file_name), AseAtomsAdaptor.get_atoms(utils.add_molecules(self.oriented_bulk.oriented_bulk_structure)), file_format)
        else:
            for layer_index in range(self.num_nums_layers):
                write(slab_paths[layer_index], self.slab_atoms[layer_index], file_format)
            if do_obs:
                write(os.path.join(obs_directory, obs_file_name), self.oriented_bulk.oriented_bulk_atoms, file_format)
        
    """ Most important file formats to support:
            JSON        AimsExtractor   first implementation
            aims.out    FHI-aims
            OSZICAR     VASP            second implementation
            OUTCAR (~)  VASP  
            csv?
            ...?
    """
    #TODO: Support wider range of total energy data input file types
    # Redundant. Included for ease of use.
    #@timed_method
    def read_total_energies(
        self,
        total_energies: np.ndarray
    ) -> None:
        self.total_energies = total_energies

    def read_aims_out_total_energies(
        self,
        generated_structures_directory: str = "generated_structures",
        species_directory: str = None,
        miller_directory: str = None,
        termination_directory: str = None,
        obs_directory: str = None
    ) -> None:
        include_ts = "ts" in self.total_energy_methods
        include_pbe = "pbe" in self.total_energy_methods
        if species_directory is None:
            species_directory = os.path.join(generated_structures_directory, self.species_name)
        if miller_directory is None:
            miller_directory = os.path.join(species_directory, self.miller_name)
        if termination_directory is None:
            termination_directory = os.path.join(miller_directory, self.termination_index_name)
        if obs_directory is None:
            obs_directory = os.path.join(miller_directory, "OBS")

        if include_ts:
            ts_index = self.total_energy_methods.index("ts")
        if include_pbe:
            pbe_index = self.total_energy_methods.index("pbe")

        handled_nums_layers = []
        vdw_energy = 0.0
        total_energy = 0.0
        with open(os.path.join(obs_directory, "aims.out"), "r") as f:
            for line in f:
                """
                if "Performing Hirshfeld analysis of fragment charges and moments." in line:
                    free_volumes = []
                    hirshfeld_volumes = [] 
                elif "|   Free atom volume        :" in line:
                    free_volumes.append(float(line.split()[5]))
                elif "|   Hirshfeld volume        :" in line:
                    hirshfeld_volumes.append(float(line.split()[4]))
                """
                if "| vdW energy correction         :" in line:
                    vdw_energy = float(line.split()[-2])
                elif '| Total energy of the DFT' in line:
                    total_energy = float(line.split()[-2])
                """
                elif '| Total time                                 :' in line:
                    total_time = line.split()[4]
                elif '| Number of self-consistency cycles          :' in line:
                    scf_it = line.split()[6]
                elif '| Number of relaxation steps                 :' in line:
                    relaxation_steps = line.split()[6]
                """
        if include_ts:
            self.obs_total_energies[ts_index] = total_energy
        if include_pbe:
            self.obs_total_energies[pbe_index] = total_energy - vdw_energy

        for subdirectory_name in os.listdir(termination_directory):
            """
            split_name = subdirectory_name.split(".")

            # Read Miller indices from file name
            miller_index = []
            index_string = ""
            for char in split_name[1]:
                index_string += char
                if char != "-":
                    miller_index.append(int(index_string))
                    index_string = ""

            layers_or_obs = split_name[2]
            if layers_or_obs == "OBS":
            """
            num_layers = int(subdirectory_name)
            if num_layers in self.nums_layers_list:
                handled_nums_layers.append(num_layers)
                layers_index = self.nums_layers_list.index(num_layers)
                file_path = os.path.join(termination_directory, subdirectory_name, "aims.out")
                total_energy = 0.0
                vdw_energy = 0.0
                with open(file_path, "r") as f:
                    for line in f:
                        """
                        if "Performing Hirshfeld analysis of fragment charges and moments." in line:
                            free_volumes = []
                            hirshfeld_volumes = [] 
                        elif "|   Free atom volume        :" in line:
                            free_volumes.append(float(line.split()[5]))
                        elif "|   Hirshfeld volume        :" in line:
                            hirshfeld_volumes.append(float(line.split()[4]))
                        """
                        if "| vdW energy correction         :" in line:
                            vdw_energy = float(line.split()[-2])
                        elif '| Total energy of the DFT' in line:
                            total_energy = float(line.split()[-2])
                        """
                        elif '| Total time                                 :' in line:
                            total_time = line.split()[4]
                        elif '| Number of self-consistency cycles          :' in line:
                            scf_it = line.split()[6]
                        elif '| Number of relaxation steps                 :' in line:
                            relaxation_steps = line.split()[6]
                        """
                if include_ts:
                    self.total_energies[layers_index][ts_index] = total_energy
                if include_pbe:
                    self.total_energies[layers_index][pbe_index] = total_energy - vdw_energy

        missing_nums_layers = list(set(self.nums_layers_list) - set(handled_nums_layers))
        if len(missing_nums_layers) > 0:
            warnings.warn("ATTENTION: Total energy files for slabs with " + str(missing_nums_layers) + " layers were not found in directory " + termination_directory + ". This SurfaceEnergy object's total energies for slabs with these numbers of layers were not updated.\n")

    # TODO: Find better solution or make into a @property with getter and setter
    def _get_total_energies_all_nums_layers(
        self,
        total_energy_method_index: int 
    ) -> np.ndarray:
        if self.num_nums_layers == 1:
            return self.total_energies[0, total_energy_method_index]
        return self.total_energies[:, total_energy_method_index]

    def _set_total_energies_all_nums_layers(
        self,
        total_energies_one_method: np.ndarray,
        total_energy_method_index: int    
    ) -> None:
        pass

    @timed_method
    def calculate_surface_energies(
        self,
        silent: bool = True,
        file_name: str = None
    ) -> None:
        for surface_energy_method in self.surface_energy_methods:
            if surface_energy_method == "OBS":
                self._OBS_surface_energies()
            elif surface_energy_method == "Boettger":
                self._Boettger_surface_energies()
            elif surface_energy_method == "slope":
                self._slope_surface_energies()
            elif surface_energy_method == "intercept":
                self._intercept_surface_energies()
        
        if not silent:
            self.generate_surface_energies_output_file(file_name)

    """ Most important file formats to support:
            JSON        ?               first implementation
            npy?        ?               second implementation
            h5?         AIMNet?         
            csv??       ?
            ...?
    """
    @timed_method
    def generate_surface_energies_output_file(
        self,
        file_name: str = None
    ) -> str:
        if file_name is None:
            file_name = self._default_file_name("surface_energies")
            

        return file_name

    def to_pickle(
        self,
        file_path: str = "SurfaceEnergy_object_pickle"
    ) -> None:
        with open(file_path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    # (public) class methods
    # NOTE: as __init__ is implemented, the returned object will be "sterile" unless was instantiated with one of these class methods.
    @classmethod
    @timed_method
    def from_pickle(
        cls,
        file_path: str
    ) -> SelfSurfaceEnergy:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    @classmethod
    @timed_method
    def from_bulk_file(
        cls,
        bulk_file_name: str,
        miller_index: List[int],
        nums_layers: List[int],
        total_energy_methods: List[str],
        species_name: str = "species",
        molecular: bool = True,
        surface_energy_methods: List[str] = IMPLEMENTED_SURFACE_ENERGY_METHODS,
        universal_convergence_threshhold: float = None,
        forward_not_central_difference: bool = False,
        vacuum: float = DEFAULT_VACUUM,
        termination_index: int = 0
    ) -> SelfSurfaceEnergy:
        generate_all = (termination_index != 0)

        if molecular:

            molecular_surface_generators = []
            for num_layers in nums_layers:
                start_time = time()
                sys.stdout.write("Creating MolecularSurfaceGenerator object for slab with " + str(num_layers) + " layers...")
                molecular_surface_generators.append(
                    MolecularSurfaceGenerator.from_file(
                        filename=bulk_file_name,
                        miller_index=miller_index,
                        layers=num_layers,
                        vacuum=vacuum,
                        refine_structure=True,
                        generate_all=generate_all, # TODO: Handle nonzero termination indices
                        lazy=False,
                    )
                )
                duration = round(time() - start_time, 2)
                sys.stdout.write("finished after " + str(duration) + " seconds.\n")

            return cls.from_molecular_surface_generators(
                molecular_surface_generators,
                nums_layers,
                total_energy_methods,
                species_name,
                surface_energy_methods,
                universal_convergence_threshhold,
                forward_not_central_difference,
                termination_index
            )
        
        surface_generators = []
        for num_layers in nums_layers:
            start_time = time()
            sys.stdout.write("Creating SurfaceGenerator object for slab with " + str(num_layers) + " layers...")
            surface_generators.append(
                SurfaceGenerator.from_file(
                    filename=bulk_file_name,
                    miller_index=miller_index,
                    layers=num_layers,
                    vacuum=vacuum,
                    refine_structure=True, #TODO: Replace bulk structure file with refined structure file using utils function/spglib.standardize
                    generate_all=generate_all, # TODO: Handle nonzero termination indices
                    lazy=False,
                    suppress_warnings=False
                )
            )
            duration = round(time() - start_time, 2)
            sys.stdout.write("finished after " + str(duration) + " seconds.\n")

        return cls.from_surface_generators(
            surface_generators,
            nums_layers,
            total_energy_methods,
            species_name,
            surface_energy_methods,
            universal_convergence_threshhold,
            forward_not_central_difference,
            termination_index
        )
        
    @classmethod
    @timed_method
    def from_bulk_structure_or_atoms(
        cls,
        bulk_structure_or_atoms: Union[Structure, Atoms],
        miller_index: List[int],
        nums_layers: List[int],
        total_energy_methods: List[str],
        species_name: str = "species",
        molecular: bool = True,
        surface_energy_methods: List[str] = IMPLEMENTED_SURFACE_ENERGY_METHODS,
        universal_convergence_threshhold: float = None,
        forward_not_central_difference: bool = False,
        vacuum: float = DEFAULT_VACUUM,
        termination_index: int = 0
    ) -> SelfSurfaceEnergy:
        # TODO: Handle case where len(nums_layers) == 1 in a tidier way; consider messing with how nums_layers is defined and/or accessed.
        generate_all = (termination_index != 0)

        if molecular:
            molecular_surface_generators = []
            for num_layers in nums_layers:
                molecular_surface_generators.append(
                    MolecularSurfaceGenerator(
                        bulk=bulk_structure_or_atoms,
                        miller_index=miller_index,
                        layers=num_layers,
                        vacuum=vacuum,
                        refine_structure=True,
                        generate_all=generate_all, # TODO: Handle nonzero termination indices
                        lazy=False
                    )
                )

            return cls.from_molecular_surface_generators(
                molecular_surface_generators,
                nums_layers,
                total_energy_methods,
                species_name,
                surface_energy_methods,
                universal_convergence_threshhold,
                forward_not_central_difference,
                termination_index
            )
        
        surface_generators = []
        for num_layers in nums_layers:
            surface_generators.append(
                SurfaceGenerator(
                    bulk=bulk_structure_or_atoms,
                    miller_index=miller_index,
                    layers=num_layers,
                    vacuum=vacuum,
                    refine_structure=True, #TODO: Replace bulk structure file with refined structure file using utils function/spglib.standardize
                    generate_all=generate_all, # TODO: Handle nonzero termination indices
                    lazy=False,
                    suppress_warnings=False
                )
            )

        return cls.from_surface_generators(
            surface_generators,
            nums_layers,
            total_energy_methods,
            species_name,
            surface_energy_methods,
            universal_convergence_threshhold,
            forward_not_central_difference,
            termination_index
        )

    @classmethod
    @timed_method
    def from_molecular_surface_generators(
        cls,
        molecular_surface_generators: List[MolecularSurfaceGenerator],
        nums_layers: List[int],
        total_energy_methods: List[str],
        species_name: str = "species",
        surface_energy_methods: Union[List[str], Dict] = IMPLEMENTED_SURFACE_ENERGY_METHODS,
        universal_convergence_threshhold: float = None,
        forward_not_central_difference: bool = False,
        termination_index: int = 0
    ) -> SelfSurfaceEnergy:
        # Construct list of generated Surface objects corresponding to the passed termination index
        molecular_surfaces = []
        for index in range(len(nums_layers)):
            molecular_surfaces.append(molecular_surface_generators[index][termination_index])

        # if molecular_surface_generators[0].generate_all or termination_index > 0:
        #     surfaces = np.zeros(len(nums_layers), termination_index + 1)
        #     for layer_index in range(len(nums_layers)):
        #         generator = molecular_surface_generators[layer_index]
        #         for term_index in range(len(generator)):
        #             surface = [term_index]
        #             surfaces[index]
        #             surfaces.append(molecular_surface_generators[index][termination_index])
        # else:
        #     surfaces_list = []
        #     for layer_index in range(len(nums_layers)):
        #         surfaces_list.append(molecular_surface_generators[layer_index][0])
        #     surfaces = np.array(surfaces_list)

        # Pass list of Surface objects to the class function that returns a SurfaceEnergy object from a list of Surface objects, return SurfaceEnergy object
        return cls.from_molecular_surfaces(
            molecular_surfaces=molecular_surfaces,
            nums_layers=nums_layers,
            total_energy_methods=total_energy_methods,
            species_name=species_name,
            surface_energy_methods=surface_energy_methods,
            universal_convergence_threshhold=universal_convergence_threshhold,
            forward_not_central_difference=forward_not_central_difference,
            surface_generators=molecular_surface_generators
        )

    @classmethod
    @timed_method
    def from_surface_generators(
        cls,
        surface_generators: List[SurfaceGenerator],
        nums_layers: List[int],
        total_energy_methods: List[str],
        species_name: str = "species",
        surface_energy_methods: Union[List[str], Dict] = IMPLEMENTED_SURFACE_ENERGY_METHODS,
        universal_convergence_threshhold: float = None,
        forward_not_central_difference: bool = False,
        termination_index: int = 0
    ) -> SelfSurfaceEnergy:
        # Construct list of generated Surface objects correspnding to the passed termination index
        surfaces = []
        for index in range(len(nums_layers)):
            surfaces.append(surface_generators[index][termination_index])
        # Pass list of Surface objects to the class function that returns a SurfaceEnergy object from a list of Surface objects, return SurfaceEnergy object
        return cls.from_surfaces(
            surfaces=surfaces,
            nums_layers=nums_layers,
            total_energy_methods=total_energy_methods,
            species_name=species_name,
            molecular=False,
            surface_energy_methods=surface_energy_methods,
            universal_convergence_threshhold=universal_convergence_threshhold,
            forward_not_central_difference=forward_not_central_difference,
            surface_generators=surface_generators
        )
    
    # TODO: Have a better way to pass convergence thresholds.
    #           Don't pass surface
    @classmethod
    @timed_method
    def from_surfaces(
        cls,
        surfaces: List[Surface],
        nums_layers: List[int],
        total_energy_methods: List[str],
        species_name: str = "species",
        surface_energy_methods: Union[List[str], Dict] = IMPLEMENTED_SURFACE_ENERGY_METHODS,
        universal_convergence_threshhold: float = None,
        forward_not_central_difference: bool = False,
        surface_generators: Union[List[SurfaceGenerator], List[MolecularSurfaceGenerator]] = None
    ) -> SelfSurfaceEnergy:     
        # Initialize SurfaceEnergy object
        surface_energy_object = cls(
            molecular=False,
            nums_layers=nums_layers,
            total_energy_methods=total_energy_methods,
            species_name=species_name,
            surface_energy_methods=surface_energy_methods,
            universal_convergence_threshhold=universal_convergence_threshhold,
            forward_not_central_difference=forward_not_central_difference
        )

        # Set attributes provided as arguments but not needed for initialization
        surface_energy_object.surfaces = surfaces
        surface_energy_object.surface_generators = surface_generators

        # Derive attributes from any of the Surface objects
        surface = surfaces[0]
        surface_energy_object.bulk_structure = surface.bulk_structure
        surface_energy_object.miller_index = surface.miller_index
        surface_energy_object.miller_name = miller_name(surface.miller_index)
        surface_energy_object.termination_index_name = termination_index_name(surface.termination_index)
        surface_energy_object.vacuum = surface.vacuum
        surface_energy_object.termination_index = surface.termination_index
        surface_energy_object.oriented_bulk = surface.oriented_bulk
        #surface_energy_object.oriented_bulk_structure = surface.oriented_bulk_structure
        #surface_energy_object.oriented_bulk_atoms = surface.oriented_bulk.oriented_bulk_atoms
        surface_energy_object.area = surface.area # NOTE: Area does not vary with number of layers.

        # Derive attributes from all of the Surface objects
        slab_structures = []
        slab_atoms = []
        for index in range(surface_energy_object.num_nums_layers):
            slab_structures.append(surfaces[index]._non_orthogonal_slab_structure)
            slab_atoms.append(AseAtomsAdaptor.get_atoms(slab_structures[index]))
        surface_energy_object.slab_structures = slab_structures
        surface_energy_object.slab_atoms = slab_atoms

        # Return SurfaceEnergy object 
        return surface_energy_object
    
    # TODO: Have a better way to pass convergence thresholds.
    #           Don't pass surface
    @classmethod
    @timed_method
    def from_molecular_surfaces(
        cls,
        molecular_surfaces: List[MolecularSurface],
        nums_layers: List[int],
        total_energy_methods: List[str],
        species_name: str = "species",
        surface_energy_methods: Union[List[str], Dict] = IMPLEMENTED_SURFACE_ENERGY_METHODS,
        universal_convergence_threshhold: float = None,
        forward_not_central_difference: bool = False,
        surface_generators: Union[List[SurfaceGenerator], List[MolecularSurfaceGenerator]] = None
    ) -> SelfSurfaceEnergy:     
        # Initialize SurfaceEnergy object
        surface_energy_object = cls(
            molecular=True,
            nums_layers=nums_layers,
            total_energy_methods=total_energy_methods,
            species_name=species_name,
            surface_energy_methods=surface_energy_methods,
            universal_convergence_threshhold=universal_convergence_threshhold,
            forward_not_central_difference=forward_not_central_difference
        )

        # Set attributes provided as arguments but not needed for initialization
        surface_energy_object.surfaces = molecular_surfaces
        surface_energy_object.surface_generators = surface_generators

        # Derive attributes from any of the Surface objects
        surface = molecular_surfaces[0]
        surface_energy_object.bulk_structure = surface.bulk_structure
        surface_energy_object.miller_index = surface.miller_index
        surface_energy_object.miller_name = miller_name(surface.miller_index)
        surface_energy_object.termination_index_name = termination_index_name(surface.termination_index)
        surface_energy_object.vacuum = surface.vacuum
        surface_energy_object.termination_index = surface.termination_index
        surface_energy_object.oriented_bulk = surface.oriented_bulk
        #surface_energy_object.oriented_bulk_structure = surface.oriented_bulk_structure
        #surface_energy_object.oriented_bulk_atoms = surface.oriented_bulk_atoms
        surface_energy_object.area = surface.area # NOTE: Area does not vary with number of layers.

        # Derive attributes from all of the Surface objects
        slab_structures = []
        slab_atoms = []
        for index in range(surface_energy_object.num_nums_layers):
            slab_structures.append(molecular_surfaces[index]._non_orthogonal_slab_structure)
            slab_atoms.append(AseAtomsAdaptor.get_atoms(slab_structures[index]))
        surface_energy_object.slab_structures = slab_structures
        surface_energy_object.slab_atoms = slab_atoms

        # Return SurfaceEnergy object 
        return surface_energy_object

    """
    @classmethod
    def make_empty_like(
        cls,
        parent_object: SelfSurfaceEnergy
        ) -> SelfSurfaceEnergy:
        # Make an "empty" SurfaceEnergy object like the passed SurfaceEnergy object. An "empty" SurfaceEnergy object is intended to be used as "zeros" in numpy arrays of SurfaceEnergy objects.
        child_object = cls(
            molecular=parent_object.molecular,
            nums_layers=parent_object.nums_layers_list,
            total_energy_methods=parent_object.total_energy_methods,
            species_name=parent_object.species_name,
            surface_energy_methods=parent_object.surface_energy_methods,
            universal_convergence_treshold=None,
            forward_not_central_difference=parent_object.forward_not_central_difference,
            empty=True
        )
        child_object.convergence_thresholds = parent_object.convergence_thresholds
        return child_object
    """

def save_object(obj: SurfaceEnergy, file_path: str = "SurfaceEnergy_object_pickle") -> None:
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def test_Wulff_shapes(
    nums_layers: List[int],
    bulk_file_name: str,
    total_energy_methods: List[str],
    surface_energy_methods: List[str] = IMPLEMENTED_SURFACE_ENERGY_METHODS,
    vacuum: float = DEFAULT_VACUUM,
    termination_index: int = 0,
    max_index: int = 2,
    projected_direction: List[int] = [1, 1, 1]
) -> None:
    bulk_structure = Structure.from_file(bulk_file_name)
    all_miller_indices = get_symmetrically_distinct_miller_indices(bulk_structure, max_index)
    
    N_miller_indices = len(all_miller_indices)
    N_total_energy_methods = len(total_energy_methods)
    N_surface_energy_methods = len(surface_energy_methods)

    surface_energies = np.full((N_miller_indices, N_total_energy_methods, N_surface_energy_methods), 0.0)
    #Wulff_shapes = np.empty((N_total_energy_methods, N_surface_energy_methods))

    # Make SurfaceEnergy objects
    se_objs = []
    #miller_indices_index = 0
    for miller_indices in all_miller_indices:
        se_objs.append(
            SurfaceEnergy.from_bulk_structure_or_atoms(
            bulk_structure_or_atoms=bulk_structure,
            miller_index=miller_indices,
            nums_layers=nums_layers,
            vacuum=vacuum,
            total_energy_methods=total_energy_methods,
            surface_energy_methods=surface_energy_methods,
            termination_index=termination_index
            )
        )

        #surface_energies[miller_indices_index, ...] = se_obj.converged_surface_energies
        #miller_indices_index += 1



    # Get surface energies

    # Make, plot, and save Wulff shapes
    for total_energy_method_index in range(N_total_energy_methods):
        for surface_energy_method_index in range(N_surface_energy_methods):
            #Wulff_shapes[total_energy_method_index][surface_energy_method_index] = WulffShape(bulk_structure.lattice, all_miller_indices, surface_energies[:, total_energy_method_index, surface_energy_method_index].tolist)
            plt.figure()
            Wulff_shape = WulffShape(bulk_structure.lattice, all_miller_indices, surface_energies[:, total_energy_method_index, surface_energy_method_index].tolist)
            Wulff_shape.get_plot(direction=projected_direction)
            plt.savefig("Wulff_shapes/{}/Wulff_{}_threshhold={}.png".format(structure_name, tag, threshhold),
                        dpi=400, bbox_inches="tight")

def Wulff_plot(structure_name, structure_path, projected_direction, fitting_method, energy_results, threshhold):
    """Plot Wulff shape by pymatgen.

    Parameters:
    ----------
        structure_name: str
            Structure's name.
        structure_path: str
            The path of initial bulk structure.
        projected_direction: List[int]
            The projected direction for the Wulff shape.
        fitting_method: int
            0: linear method, 1: Boettger method.
        energy_results: dict
            Dictionary that contains the surface energy values for TS and MBD with linear and Boettger method.
        threshhold: float
            Threshhold that determines the convergence.
    """
    structure = Structure.from_file(structure_path) # Structure = bulk stucture as Pymatgen Structure via Structure.from_file()
    lattice = structure.lattice # lattice = Pymatgen lattice of bulk Structure
    print("Lattice parameters are [{}, {}, {}], please check whether they are in the same order as the input.".format(
        lattice.a, lattice.b, lattice.c))
    for tag in ["ts", "mbd"]:
        energy_result = energy_results[tag]
        data = {}
        for index, term, by, ly in energy_result:
            idx = []
            temp_idx = ""
            for char in index:
                if char == "-":
                    temp_idx += char
                else:
                    temp_idx += char
                    idx.append(temp_idx)
                    temp_idx = ""
            idx = tuple([int(x) for x in idx])
            if fitting_method == 1:
                if idx not in data or data[idx] > by:
                    data[idx] = float(by)
            else:
                if idx not in data or data[idx] > ly:
                    data[idx] = float(ly)
        with open("{}/surface_energy_{}_threshhold={}.csv".format(structure_name, tag, threshhold),'w') as f:
            f.write("Miller index\tSurface energy($mJ/m^{2}$)\n")
            for idx, energy in data.items():
                f.write("{}\t{}\n".format("".join(str(x) for x in idx), energy))
        plt.figure()
        w_r = WulffShape(lattice, data.keys(), data.values())
        w_r.get_plot(direction=projected_direction)
        plt.savefig("{}/Wulff_{}_threshhold={}.png".format(structure_name, tag, threshhold),
                    dpi=400, bbox_inches="tight")

def test_Wulff():
    """
    seobj = SurfaceEnergy.from_bulk_file(
        bulk_file_name="relaxed_bulk_files/ASPIRIN.cif",
        miller_index=[1, 0, 1],
        nums_layers=[3, 4],
        species_name="aspirin_version_test_3_one_surface_energy_object",
        total_energy_methods=["ts", "pbe"],
        surface_energy_methods=IMPLEMENTED_SURFACE_ENERGY_METHODS,
        vacuum=DEFAULT_VACUUM,
        molecular=True,
        universal_convergence_threshhold=DEFAULT_CONVERGENCE_THRESHOLD,
        forward_not_central_difference=False,
        termination_index=0
    )
    seobj.to_pickle("version_test_3_seobj_pickle")
    sys.stdout.write("pickled to 'version_test_3_seobj_pickle'\n")
    seobj.generate_structure_files(file_format="aims", prep_for_calc=True, control_in=os.path.join("aims_ins", "control.in"), submission_script=os.path.join("aims_ins", "submit.sh"))
    seobj.to_pickle("version_test_3_seobj_generated_structure_files_pickle")
    sys.stdout.write("pickled to 'version_test_3_seobj_generated_structure_files_pickle'\n")
    #species_directory=os.path.join("aims_outs_archive", "aims_outs_Q", "aspirin"), obs_directory=os.path.join("aims_outs_archive", "aims_outs_Q", "aspirin", "ASPIRIN.101.OBS.A")
    seobj.read_aims_out_total_energies(species_directory=os.path.join("generated_structures", "aspirin_60_vacuum_limited_terminations"))
    seobj.to_pickle("version_test_3_seobj_total_energies_pickle")
    sys.stdout.write("pickled to 'version_test_3_seobj_total_energies_pickle'\n")
    seobj.calculate_surface_energies()
    seobj.to_pickle("version_test_3_seobj_surface_energies_pickle")
    sys.stdout.write("pickled to 'version_test_3_seobj_surface_energies_pickle'\n")
    """

    """
    wulffs = Wulff(
        molecular=True,
        bulk_file_name=os.path.join("relaxed_bulk_files", "ASPIRIN.cif"),
        nums_layers=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        total_energy_methods=["ts", "pbe"],
        species_name="aspirin_version_test_3_60_vacuum",
        surface_energy_methods=IMPLEMENTED_SURFACE_ENERGY_METHODS,
        universal_convergence_threshhold=DEFAULT_CONVERGENCE_THRESHOLD,
        forward_not_central_difference=False,
        vacuum=DEFAULT_VACUUM,
        max_index=1,
        projected_direction=[7, -7, -1]
    )
    wulffs.to_pickle("version_test_3_wulffs_pickle")
    sys.stdout.write("pickled to 'version_test_3_wulffs_pickle'\n")
    wulffs.generate_structure_files(control_in=os.path.join("aims_ins", "control.in"), submission_script=os.path.join("aims_ins", "submit.sh"))
    wulffs.to_pickle("version_test_3_wulffs_generated_structure_files_pickle")
    sys.stdout.write("pickled to 'version_test_3_wulffs_generated_structure_files_pickle'\n")
    """
    wulffs = Wulff.from_pickle('version_test_3_wulffs_generated_structure_files_pickle')
    #wulffs.read_total_energies(species_directory=os.path.join("generated_structures", "aspirin_60_vacuum_limited_terminations"))
    wulffs.read_total_energies(species_directory=os.path.join("generated_structures_old_generate", "aspirin_60_vacuum"))
    wulffs.to_pickle("version_test_3_wulffs_total_energies_pickle")
    sys.stdout.write("pickled to 'version_test_3_wulffs_total_energies_pickle'\n")
    wulffs.calculate(plot=True, save=True, Wulff_directory="Wulff_shapes", projected_direction=[7, -7, -1])
    wulffs.to_pickle("version_test_3_wulffs_calculated_pickle")
    sys.stdout.write("pickled to 'version_test_3_wulffs_calculated_pickle'\n")

def test_terminations():
    """
    wulffs = Wulff(
        molecular=True,
        bulk_file_name=os.path.join("relaxed_bulk_files", "ASPIRIN.cif"),
        nums_layers=[1],
        total_energy_methods=["ts", "pbe"],
        species_name="aspirin_terminations_test_2_2_60_vacuum",
        surface_energy_methods=IMPLEMENTED_SURFACE_ENERGY_METHODS,
        universal_convergence_threshhold=DEFAULT_CONVERGENCE_THRESHOLD,
        forward_not_central_difference=False,
        vacuum=DEFAULT_VACUUM,
        max_index=1,
        projected_direction=[7, -7, -1]
    )
    wulffs.to_pickle("terminations_test_2_2_wulff_pickle")
    wulffs.generate_structure_files(control_in=os.path.join("aims_ins", "control.in"), submission_script=os.path.join("aims_ins", "submit.sh"))
    wulffs.to_pickle("terminations_test_2_2_wulff_generated_structure_files_pickle")
    """
    wulffs = Wulff.from_pickle("terminations_test_2_2_wulff_generated_structure_files_pickle")
    sys.stdout.write("Plane | Terminations | Default tol\n")
    for plane_index in range(wulffs.num_planes):
        obs = wulffs.surface_energy_objects[plane_index, 0].oriented_bulk
        sys.stdout.write(miller_name(wulffs.planes[plane_index]) + " | " + str(wulffs.nums_terminations[plane_index]) + "            | " + str(calc_default_tol(obs)) + "\n")
        #for termination_index in range(wulffs.nums_terminations[plane_index]):
            #wulffs.surface_energy_objects[plane_index, termination_index]

def calc_default_tol(obs):
    # a = obs._oriented_bulk_structure.lattice.matrix[0]
    # b = obs._oriented_bulk_structure.lattice.matrix[1]
    # obs.surface_normal = np.cross(a, b) / np.linalg.norm(np.cross(a, b))
    h = obs.layer_thickness # = np.dot(obs._oriented_bulk_structure.lattice.matrix[-1], obs.surface_normal)
    cart_coords = obs.oriented_bulk_structure.cart_coords
    projected_coords = np.dot(cart_coords, obs.surface_normal)
    extended_projected_coords = np.round(
        np.concatenate(
            [
                projected_coords - h,
                projected_coords,
                projected_coords + h,
            ]
        ),
        5
    )
    unique_cart_coords = np.sort(np.unique(extended_projected_coords))
    diffs = np.diff(unique_cart_coords)
    max_diff = diffs.max()
    tol = 0.15 * max_diff
    return tol

def test_tols():
    planes_to_test = [[0, 1, 0],
                      [0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 1],
                      [1, 0, 1],
                      [1, 1, 0],
                      [0, 1, -1],
                      [-1, 0, 1],
                      [1, -1, 0],
                      [1, 1, 1],
                      [1, -1, 1],
                      [-1, 1, 1],
                      [1, 1, -1]]
    tolerances_to_test = [0.335208, 0.4659165, 0.523725, 0.6356625, 0.7426605, 0.744048, 0.77319, 0.800000, 0.900000, 1.000000]
    """
    wulffs = Wulff(
        molecular=True,
        bulk_file_name=os.path.join("relaxed_bulk_files", "ASPIRIN.cif"),
        nums_layers=[1],
        total_energy_methods=["ts", "pbe"],
        species_name="aspirin_terminations_test_2_2_60_vacuum",
        surface_energy_methods=IMPLEMENTED_SURFACE_ENERGY_METHODS,
        universal_convergence_threshhold=DEFAULT_CONVERGENCE_THRESHOLD,
        forward_not_central_difference=False,
        vacuum=DEFAULT_VACUUM,
        max_index=1,
        projected_direction=[7, -7, -1]
    )
    """

    surface_generators = []
    nums_terminations = np.zeros((len(planes_to_test), len(tolerances_to_test)))
    for plane_index, plane in enumerate(planes_to_test):
        for tol_index, tol in enumerate(tolerances_to_test):
            msg = MolecularSurfaceGenerator.from_file(
                    filename=os.path.join("relaxed_bulk_files", "ASPIRIN.cif"),
                    miller_index=plane,
                    layers=1,
                    vacuum=DEFAULT_VACUUM,
                    refine_structure=True,
                    generate_all=True,
                    lazy=False
                )
            nums_terminations[plane_index, tol_index] = msg[-1].termination_index + 1
    #sys.stdout.write("nums_terminations = " + str(nums_terminations) + "\n")
    #sys.stdout.write("self.planes = " + str(self.planes) + "\n")
    #max_termination_index = max(nums_terminations)
    sys.stdout.write(" tol  : DEFAULT  | ")
    for tol in tolerances_to_test:
        sys.stdout.write(str(tol) + " | ")
    sys.stdout.write("\n")
    for plane_index, plane in enumerate(planes_to_test):
        sys.stdout.write(miller_name(plane) + ": " + "nvm      | ")
        for tol_index, tol in enumerate(tolerances_to_test):
            sys.stdout.write(str(nums_terminations[plane_index, tol_index]) + "         | ")
        sys.stdout.write("\n")

def main():
    #lap_stopwatch("main() entry")
    #, 2, 3, 4, 5, 6, 7, 8, 9, 10
    """
    seobj = SurfaceEnergy.from_bulk_file(
        bulk_file_name="relaxed_bulk_files/ASPIRIN.cif",
        miller_index=[1, 0, 1],
        nums_layers=[1, 2, 3, 4, 5],
        vacuum=DEFAULT_VACUUM,
        total_energy_methods=["ts", "pbe"],
        surface_energy_methods=IMPLEMENTED_SURFACE_ENERGY_METHODS
    )
    """
    #lap_stopwatch("seobj created")
    #seobj.to_pickle("test_101_14_seobj_pickle")
    #save_object(seobj, "test_seobj_pickle")
    #seobj = SurfaceEnergy.from_pickle("test_101_11_seobj_pickle")
    #seobj.generate_structure_files(file_format="aims", prep_for_calc=True, control_in=os.path.join("aims_ins", "control.in"), submission_script=os.path.join("aims_ins", "submit.sh"))
    #lap_stopwatch("structure files generated")


    #seobj_old = SurfaceEnergy.from_pickle("test_101_3_seobj_pickle")
    #seobj = SurfaceEnergy.from_surfaces(seobj_old.surfaces)

    #sys.stdout.write("seobj.area: " + str(seobj.area) + "\n")

    #seobj.read_aims_out_total_energies(directory_path=os.path.join("aims_outs_archive", "aims_outs_Q", "aspirin"))

    #seobj = SurfaceEnergy.from_pickle("test_101_13_totals_seobj_pickle")
    #sys.stdout.write("\nseobj.total_energies: " + str(seobj.total_energies) + "\n")
    #sys.stdout.write("\nseobj.obs_total_energies: " + str(seobj.obs_total_energies) + "\n\n")
    #seobj.to_pickle("test_101_14_totals_seobj_pickle")
    #sys.stdout.write("\nseobj.total_energies: " + str(seobj.total_energies) + "\n")
    #sys.stdout.write("\nseobj.obs_total_energies: " + str(seobj.obs_total_energies) + "\n\n")

    #seobj.calculate_surface_energies()

    #seobj = SurfaceEnergy.from_pickle("test_101_11_ses_seobj_pickle")

    #sys.stdout.write("\nseobj.surface_energies: " + str(seobj.surface_energies) + "\n\n")
    #sys.stdout.write("seobj.converged_surface_energies: " + str(seobj.converged_surface_energies) + "\n\n")

    #seobj.to_pickle("test_101_14_ses_seobj_pickle")

    """
    wulffs = Wulff(
        molecular=True,
        bulk_file_name=os.path.join("relaxed_bulk_files", "ASPIRIN.cif"),
        nums_layers=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        total_energy_methods=["ts", "pbe"],
        species_name="aspirin_60_vacuum_limited_terminations",
        surface_energy_methods=IMPLEMENTED_SURFACE_ENERGY_METHODS,
        universal_convergence_threshhold=DEFAULT_CONVERGENCE_THRESHOLD,
        forward_not_central_difference=False,
        vacuum=DEFAULT_VACUUM,
        max_index=1,
        projected_direction=[7, -7, -1]
    )
    wulffs.to_pickle("Wulff_pickle_11")
    wulffs.generate_structure_files()
    wulffs.to_pickle("Wulff_pickle_generated_11")
    """

    """
    wulffs = Wulff.from_pickle("Wulff_pickle_6")
    wulffs.universal_convergence_threshhold = DEFAULT_CONVERGENCE_THRESHOLD
    convergence_threshholds = {
                    "OBS": DEFAULT_CONVERGENCE_THRESHOLD,
                    "Boettger": 
                        {
                            "d_surface_energies_d_nums_layers": DEFAULT_CONVERGENCE_THRESHOLD,
                            "d_total_energies_d_nums_layers": DEFAULT_CONVERGENCE_THRESHOLD
                        },
                    "slope": DEFAULT_CONVERGENCE_THRESHOLD,
                    "intercept": DEFAULT_CONVERGENCE_THRESHOLD
                }
    for plane_index in range(wulffs.num_planes):
        for termination_index in range(wulffs.nums_terminations[plane_index]):
            wulffs.surface_energy_objects[plane_index, termination_index].convergence_threshholds = convergence_threshholds
    """
    #wulffs = Wulff.from_pickle("Wulff_pickle_generated_11")
    #wulffs.read_total_energies("generated_structures")
    # print missing total energies
    #wulffs.to_pickle("Wulff_pickle_total_energies_11")
    wulffs = Wulff.from_pickle("Wulff_pickle_calculated_10")
    for sem_i, sem in enumerate(wulffs.surface_energy_methods):
        sys.stdout.write("\nTS + " + sem + " surface energies:\n")
        for plane_i, plane in enumerate(wulffs.planes):
            sys.stdout.write("\t" + str(plane) + ": " + str(wulffs.surface_energies[plane_i, 0, sem_i]) + "\n")
    #custom_colors={
    #    (0, 0, 1): [0, 0, 0.9764705882352941, 1.0],
    #    (1, 0, 0): [0.3411764705882353, 0, 0, 1.0],
    #    (1, -1, 0): [0.30196078431372547, 0.18823529411764706, 0.21568627450980393, 1.0],
    #}
    #wulffs.plot_Wulff_shapes(direction=[7, -7, -1], save=True, Wulff_directory="Wulff_test_10_7-7-1", custom_colors=custom_colors)
    #wulffs.to_pickle("Wulff_pickle_plotted_10_7-7-1")
    #print(str(wulffs.Wulff_shapes[0, 3].))
    #wulffs.projected_direction = [7, -7, -1]
    #wulffs.calculate(plot=True, save=True, Wulff_directory="Wulff_shapes_10_7-7-1", )
    #wulffs.to_pickle("Wulff_pickle_calculated_11")
    #"""
    
    """
    wulffs = Wulff.from_pickle("Wulff_pickle_calculated_4")
    for plane_index in range(wulffs.num_planes):
        sys.stdout.write("plane: " + str(wulffs.planes[plane_index]) + "\n")
        for termination_index in range(wulffs.nums_terminations[plane_index]):
            sys.stdout.write("\ttermination index: " + str(termination_index) + "\n")
            seobj = wulffs.surface_energy_objects[plane_index, termination_index]

            sys.stdout.write("\t\tOBS\n")
            for total_energy_method_index in range(seobj.num_total_energy_methods):
                sys.stdout.write("\t\t\ttotal energy method: " + seobj.total_energy_methods[total_energy_method_index] + "\n")
                sys.stdout.write("\t\t\t\ttotal energy (OBS): " + str(seobj.obs_total_energies[total_energy_method_index]) + "\n")
            for num_layers_index in range(seobj.num_nums_layers):
                sys.stdout.write("\t\tlayers: " + str(seobj.nums_layers_list[num_layers_index]) + "\n")
                for total_energy_method_index in range(seobj.num_total_energy_methods):
                    sys.stdout.write("\t\t\ttotal energy method: " + seobj.total_energy_methods[total_energy_method_index] + "\n")
                    sys.stdout.write("\t\t\t\ttotal energy: " + str(seobj.total_energies[num_layers_index][total_energy_method_index]) + "\n")
    
    for plane_index in range(wulffs.num_planes):
        sys.stdout.write("plane: " + str(wulffs.planes[plane_index]) + "\n")
        for total_energy_method_index in range(seobj.num_total_energy_methods):
            sys.stdout.write("\ttotal energy method: " + seobj.total_energy_methods[total_energy_method_index] + "\n")
            for surface_energy_method_index in range(wulffs.num_surface_energy_methods):
                sys.stdout.write("\t\tsurface energy method: " + seobj.surface_energy_methods[surface_energy_method_index] + "\n")
                sys.stdout.write("\t\t\tsurface energy: " + str(wulffs.surface_energies[plane_index,
                                                                                        total_energy_method_index,
                                                                                        surface_energy_method_index]) + "\n")
    """


    """
    test_Wulff_shapes(
        nums_layers=[4,5,6,7,8,9,10],
        bulk_file_name=os.path.join("relaxed_bulk_files", "ASPIRIN.cif"),
        vacuum=60,
        total_energy_methods=["test"],
        surface_energy_methods=ALL_AVAILABLE_SURFACE_ENERGY_METHODS,
        termination_index=0,
        max_index=3,
        projected_direction=[1, 1, 1]
    )
    """

if __name__ == "__main__":
    #lap_stopwatch("namecheck entry")
    #main()
    #test_Wulff()
    #test_terminations()
    test_tols()