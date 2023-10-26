"""
This module will be used to construct the surfaces and interfaces used in this package.
"""
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
