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
from OgreInterface.surfaces.terminations import Terminator

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
        smoothest_only: bool = False,
        lazy: bool = False,
        suppress_warnings: bool = False,
        layer_grouping_tolarence: Optional[float] = None,
    ) -> None:
        super().__init__()
        self._refine_structure = refine_structure
        self._surface_type = surface_type
        self._layer_grouping_tolarence = layer_grouping_tolarence
        self._suppress_warnings = suppress_warnings
        self._make_planar = make_planar

        self.bulk_structure = utils.load_bulk(
            atoms_or_structure=bulk,
            refine_structure=self._refine_structure,
            suppress_warnings=self._suppress_warnings,
        )

        self.miller_index = miller_index

        self.vacuum = vacuum
        self.generate_all = generate_all
        self.smoothest_only = smoothest_only
        self.lazy = lazy

        self.obs = OrientedBulk(
            bulk=self.bulk_structure,
            miller_index=self.miller_index,
            make_planar=self._make_planar,
        )

        if layers is None and minimum_thickness is None:
            raise "Either layer or minimum_thickness must be set"
        if layers is not None:
            self.layers = layers
        if layers is None and minimum_thickness is not None:
            self.layers = int(
                (minimum_thickness // self.obs.layer_thickness) + 1
            )

        self.terminator = Terminator(
            bulk=self.bulk_structure,
            plane=self.miller_index,
            molecular=True if self._surface_type == MolecularSurface else False,
            generate_all=self.generate_all,
            num_layers=self.layers,
            vacuum=self.vacuum
        )
        
        if not self.lazy:
            self._slabs = self._generate_slabs()
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
        smoothest_only: bool = False,
        lazy: bool = False,
        suppress_warnings: bool = False,
        layer_grouping_tolarence: Optional[float] = None,
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
            smoothest_only=smoothest_only,
            lazy=lazy,
            suppress_warnings=suppress_warnings,
            layer_grouping_tolarence=layer_grouping_tolarence,
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

    def generate_slabs(self) -> None:
        """Used to generate list of Surface objects if lazy=True"""
        if self.lazy:
            self._slabs = self._generate_slabs()
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

    def _generate_slabs(self) -> List[Union[Surface, MolecularSurface]]:
        """
        This function is used to generate slab structures with all unique
        surface terminations.

        Returns:
            A list of Surface classes
        """

        if self.smoothest_only:
            return [self.terminator.smoothest_surface]

        return self._terminator.surfaces
