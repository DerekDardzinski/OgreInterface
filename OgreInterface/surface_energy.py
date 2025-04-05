from OgreInterface import utils
from OgreInterface.generate import SurfaceGenerator
from OgreInterface.surfaces import Surface
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import Element, Species
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.ase import AseAtomsAdaptor
from typing import Dict, Union, Iterable, List, Tuple
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import copy
from copy import deepcopy
from functools import reduce
from ase import Atoms
import warnings
from scipy import stats

SURFACE_ENERGY_ADJUSTMENT_FACTOR = -16000/2
ALL_AVAILABLE_METHODS = ["OBS", "Boettger_strict", "Boettger_lenient", "Linear_strict", "Linear_lenient"]
#                          0             1                  2                 3                4
DEFAULT_CONVERGENCE_THRESHOLD = 5

# TODO: more comments
# TODO: handle multiple terminations
# TODO: handle areas
# TODO: organize nums_layers analogously to how methods is organized; self.methods : {"method name": method_index} :: self.nums_layers : {"number of layers": num_layers_index}
#   TODO: organize the rest of the code accordingly; e.g. enforce one consistent way to loop through methods/nums_layers 
# TODO: Decide whether to stop calculations upon convergence
# TODO: self.generate_structures()
# TODO: reading in of external total energy calculation output data (OBS energies and slab energies)
# TODO: figure out how exactly OBS energies are to be used to calculate surface energies and whether there is one OBS energy per species or one OBS energy per slab per species etc.

class SurfaceEnergy2:
    """Container for surface energy data"""

    def __init__(
        self,
        bulk: Union[Structure, Atoms],
        miller_index: list[int],
        termination_index: int,
        nums_layers: list[int],
        vacuum: float,
        passivation_options: tuple,
        methods: list[str] = ALL_AVAILABLE_METHODS,
        convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD
    ) -> None:
        self.bulk = bulk
        self.miller_index = miller_index
        self.termination_index = termination_index

        self.M = len(methods)
        #methods_list = []
        #for method in methods:
        #    methods_list.append(ALL_AVAILABLE_METHODS.index(method))
        #methods_list.sort()
        #self.methods = np.array(methods_list)
        self.methods = self._setup_methods(methods)

        self.L = len(nums_layers)
        nums_layers.sort() # forces nums_layers to satisfy the strictly increasing validity condition
        self.nums_layers = np.array(nums_layers)
        self.method_validities = self._setup_method_validities()
        
        self.vacuum = vacuum
        self.passivation_options = passivation_options
        self.OBS_energy = None
        self.slab_total_energies = np.zeros(self.L)
        self.slab_bulk_energies = np.zeros((self.M, self.L))
        self.slab_surface_energies = np.zeros((self.M, self.L))
        self.surface_energies = np.zeros(self.M)
        self.convergence_threshold = convergence_threshold
        self.convergence_layers_indices = np.zeros(self.M)
        self.max_convergence_layers_index = None
        self.converged_methods = np.zeros(self.M)
    
    # This function should probably be passed a descriptive name for the error as a string rather than an error code. Maybe this function should return a specific error code rather than a nonspecific error indicator.
    def _error(self, error_code: int) -> int:
        if error_code == 2:
            # Attempted the boettger_strict method with a sequence of numbers of layers that was not a finite arithmetic progression with a common difference of +1; e.g. [2, 3, 4, 5] or [4, 5, 6] etc.
            return 1 # This 1 is not an error code. This 1 simply indicates that there was an error.
        if error_code == 3:
            return 1 # This 1 is not an error code. This 1 simply indicates that there was an error.
        return 0 # This 0 is not an error code. This 0 simply indicates that the function was not passed a valid error code.

    # nums_layers: Union[list[int], npt.NDArray[np.int_]] = None
    def _nums_layers_is_a_FAPCD1(self) -> bool:
        """
        Checks whether self.nums_layers is a Finite Arithmetic Progression with a Common Difference of +1 (FAPCD1); e.g. [2, 3, 4, 5] or [-3.2, -2.2, -1.2, -0.2, 0.8, 1.8] or ["apples" - 1, "apples", "apples" + 1, "apples" + 2] etc.
        Returns True if self.nums_layers is a FAPCD1.
        Returns False if self.nums_layers is not a FAPCD1.
        """
        #if nums_layers is None:
        #    nums_layers = self.nums_layers
        
        previous_num_layers = self.nums_layers[0]
        for num_layers in self.nums_layers[1:]:
            if num_layers - previous_num_layers != 1:
                return False
            previous_num_layers = num_layers
        return True

    def _nums_layers_strictly_increases(self) -> bool:
        """
        Checks whether self.nums_layers is a strictly increasing sequence; e.g. [2, 3, 4] or [-3.2, 0, 2, 5, "apples"] etc.
        Returns True if self.nums_layers is a strictly increasing sequence.
        Returns False if self.nums_layers is not a strictly increasing sequence.
        """
        previous_num_layers = self.nums_layers[0]
        for num_layers in self.nums_layers[1:]:
            if num_layers - previous_num_layers <= 0:
                return False
            previous_num_layers = num_layers
        return True

    def _nums_layers_multielement(self) -> bool:
        """
        Checks whether self.nums_layers contains more than one element; e.g. [-3.2, "apples"] or [0, 0, 0] etc.
        Returns True if self.nums_layers contains more than one element.
        Returns False if self.nums_layers does not contain more than one element.
        """
        if len(self.nums_layers) < 2:
            return False
        return True
    
    def _nums_layers_all_ints(self) -> bool:
        """
        Checks whether self.nums_layers exclusively contains elements of type int; e.g. [2, -700, 0, 2] or [4] etc.
        Returns True if self.nums_layers exclusively contains elements of type int.
        Returns False if self.nums_layers does not exclusively contain elements of type int.
        """
        return all(map(lambda i: isinstance(i, int), self.nums_layers))
    
    def _nums_layers_all_positive_andor_strings(self) -> bool:
        """
        Checks whether self.nums_layers exclusively contains positive elements and/or strings; e.g. [2, "apples", 3.2, 2] or [4] etc.
        Returns True if self.nums_layers exclusively contains positive elements and/or strings.
        Returns False if self.nums_layers does not exclusively contain positive elements and/or strings.
        """
        for num_layers in self.nums_layers:
            if num_layers <= 0:
                return False
        return True
    
    def _nums_layers_all_unique(self) -> bool:
        """
        Checks whether self.nums_layers exclusively contains unique elements; e.g. [2, "apples", -3.2, 0] or [4] etc.
        Returns True if self.nums_layers exclusively contains unique elements.
        Returns False if self.nums_layers does not exclusively contain unique elements.
        """
        if len(np.unique(self.nums_layers)) < self.L:
            return False
        return True    

    def _valid_nums_layers_generally(self) -> bool:
        """
        Checks whether self.nums_layers contains more than one element and exclusively contains unique positive integers; e.g. [9, 2, 700] or [4] etc.
        Returns True if self.nums_layers contains more than one element and exclusively contains unique positive integers.
        Returns False if self.nums_layers does not contain more than one element and/or does not exclusively contain unique positive integers.
        """
        if not self._nums_layers_multielement():
            return False
        if not self._nums_layers_all_ints():
            return False
        if not self._nums_layers_all_positive_andor_strings():
            return False
        #if not self._nums_layers_all_unique():
        #    return False
        return True
        
    def _valid_nums_layers(self, method: str) -> bool:
        if not self._valid_nums_layers_generally():
            return False
        elif method == "OBS" or method == "Linear_strict":
            return True
        
        if method == "Boettger_strict":
            return self._nums_layers_is_a_FAPCD1()
        if method == "Boettger_lenient" or method == "Linear_lenient":
            return self._nums_layers_strictly_increases()
        return False

    def _setup_method_validities(self) -> dict:
        # Conduct respective validity tests of self.nums_layers for each method and record the results of the validity tests to a dictionary.
        method_validities = {}
        for method in ALL_AVAILABLE_METHODS:
            if method not in self.methods.keys():
                # Mark using any method that the object was not created to handle as invalid; this may or may not be superfluous
                method_validities[method] = False
            else:
                method_validities[method] = self._valid_nums_layers(method)
        return method_validities

    def _setup_methods(methods) -> dict:
        method_indices = {}
        for method_index in range(len(methods)):
            method_indices[methods[method_index]] = method_index
        return method_indices
    
    def _slope_slab_total_energies_vs_numbers_of_layers(
        self,
        starting_index: int = 0,
        stopping_index: int = None,
        step: int = 1
    ) -> float:
        if stopping_index is None:
            stopping_index = self.L
        if stopping_index == starting_index:
            stopping_index += 1
        # TODO: Handle invalid parameter values

        slope, intercept, r_value, p_value, standard_error = stats.linregress(self.nums_layers[starting_index:stopping_index:step], self.slab_total_energies[starting_index:stopping_index:step])
        return slope

    def _calculate_slab_bulk_energies_OBS(self) -> None:
        method_name = "OBS" # Marked as a place at which further division of the class's functions into more, smaller functions may occur A
        if self.nums_layers_validities[method_name]: # Marked as a place at which further division of the class's functions into more, smaller functions may occur A
            self.slab_bulk_energies[self.methods[method_name]][:] = self.OBS_energy # Marked as a place at which further division of the class's functions into more, smaller functions may occur B
        else: # Marked as a place at which further division of the class's functions into more, smaller functions may occur A
            self._error(1) # Marked as a place at which further division of the class's functions into more, smaller functions may occur A

    def _calculate_slab_bulk_energies_Boettger_strict(self) -> None:
        method_name = "Boettger_strict" # Marked as a place at which further division of the class's functions into more, smaller functions may occur A
        if self.method_validities[method_name]: # Marked as a place at which further division of the class's functions into more, smaller functions may occur A
            method_index = self.methods[method_name] # Marked as a place at which further division of the class's functions into more, smaller functions may occur A
            for layers_index in range(1, self.L):
                # Note that this function will never assign any value to self.slab_bulk_energies[self.methods["Boettger_strict"]]; self.slab_bulk_energies[self.methods["Boettger_strict"]] will always retain its default value, unless it is ever updated by some other function.
                self.slab_bulk_energies[method_index, layers_index] = self.slab_total_energies[layers_index] - self.slab_total_energies[layers_index - 1] # Marked as a place at which further division of the class's functions into more, smaller functions may occur B
        else: # Marked as a place at which further division of the class's functions into more, smaller functions may occur A
            self._error(2) # Marked as a place at which further division of the class's functions into more, smaller functions may occur A

    def _calculate_slab_bulk_energies_Boettger_lenient(self) -> None:
        method_name = "Boettger_lenient" # Marked as a place at which further division of the class's functions into more, smaller functions may occur A
        if self.method_validities[method_name]: # Marked as a place at which further division of the class's functions into more, smaller functions may occur A
            method_index = self.methods[method_name] # Marked as a place at which further division of the class's functions into more, smaller functions may occur A
            for layers_index in range(1, self.L):
                # Note that this function will never assign any value to self.slab_bulk_energies[self.methods["Boettger_lenient"]]; self.slab_bulk_energies[self.methods["Boettger_lenient"]] will always retain its default value, unless it is ever updated by some other function.
                self.slab_bulk_energies[method_index][layers_index] = (self.slab_total_energies[layers_index] - self.slab_total_energies[layers_index - 1]) / (self.nums_layers[layers_index] - self.nums_layers[layers_index - 1]) # Marked as a place at which further division of the class's functions into more, smaller functions may occur B
        else: # Marked as a place at which further division of the class's functions into more, smaller functions may occur A
            self._error(3) # Marked as a place at which further division of the class's functions into more, smaller functions may occur A

    def _calculate_slab_bulk_energies_Linear_strict(self) -> None:
        method_name = "Linear_strict" # Marked as a place at which further division of the class's functions into more, smaller functions may occur A
        if self.method_validities[method_name]: # Marked as a place at which further division of the class's functions into more, smaller functions may occur A
            slab_bulk_energy_Linear_strict = self._slope_slab_total_energies_vs_numbers_of_layers() # Marked as a place at which further division of the class's functions into more, smaller functions may occur B
            self.slab_bulk_energies[self.methods["Linear_strict"]][:] = slab_bulk_energy_Linear_strict # Marked as a place at which further division of the class's functions into more, smaller functions may occur B
        else: # Marked as a place at which further division of the class's functions into more, smaller functions may occur A
            self._error(4) # Marked as a place at which further division of the class's functions into more, smaller functions may occur A

    def _calculate_slab_bulk_energies_Linear_lenient(self) -> None:
        method_name = "Linear_lenient" # Marked as a place at which further division of the class's functions into more, smaller functions may occur A
        if self.method_validities[method_name]: # Marked as a place at which further division of the class's functions into more, smaller functions may occur A
            method_index = self.methods[method_name] # Marked as a place at which further division of the class's functions into more, smaller functions may occur A
            # TODO: Figure out if the implementation of range-snipping in OgreOrganic is the best way to implement it.
            self.slab_bulk_energies[method_index][1] = self._slope_slab_total_energies_vs_numbers_of_layers(0, 2) # Marked as a place at which further division of the class's functions into more, smaller functions may occur B
            for layers_index in range(2, self.L): # Marked as a place at which further division of the class's functions into more, smaller functions may occur B
                # Note that this function will never assign any value to self.slab_bulk_energies[self.methods["Linear_lenient"]]; self.slab_bulk_energies[self.methods["Linear_lenient"]] will always retain its default value, unless it is ever updated by some other function.
                self.slab_bulk_energies[method_index][layers_index] = self._slope_slab_total_energies_vs_numbers_of_layers(layers_index - 2, layers_index + 1) # Marked as a place at which further division of the class's functions into more, smaller functions may occur B
        else: # Marked as a place at which further division of the class's functions into more, smaller functions may occur A
            self._error(5) # Marked as a place at which further division of the class's functions into more, smaller functions may occur A

    def _calculate_slab_bulk_energies(self) -> None:
        if "OBS" in self.methods:
            self._calculate_slab_bulk_energies_OBS()
        if "Boettger_strict" in self.methods:
            self._calculate_slab_bulk_energies_Boettger_strict()
        if "Boettger_lenient" in self.methods:
            self._calculate_slab_bulk_energies_Boettger_lenient()
        if "Linear_strict" in self.methods:
            self._calculate_slab_bulk_energies_Linear_strict()
        if "Linear_lenient" in self.methods:
            self._calculate_slab_bulk_energies_Linear_lenient()

    def _calculate_slab_surface_energies(self) -> None:
        self.slab_surface_energies = SURFACE_ENERGY_ADJUSTMENT_FACTOR * np.divide(((self.slab_bulk_energies @ np.diag(self.nums_layers)) - self.slab_total_energies), self.areas)

    def _calculate_surface_energies(self, convergence_threshold: float = None) -> None:
        # TODO: Surely there is a better way of doing this; will be easier to figure out once I figure out what excatly we want and why.
        
        if convergence_threshold is None:
            convergence_threshold = self.convergence_threshold
        
        diffs = np.zeros((self.M, self.L))
        for layers_index in range(self.L):
            diffs[:][layers_index] = 100 * np.divide(np.absolute(np.subtract(self.slab_surface_energies[:][layers_index], self.slab_surface_energies[:][layers_index - 1])), self.slab_surface_energies[:][layers_index])
        self.d_slab_surface_energy_d_layers_index = diffs # Why not dGamma/dnum_layers?

        for method_index in range(self.M):
            for layers_index in range(self.L):
                if diffs[method_index][layers_index] <= convergence_threshold:
                    self.convergence_layers_indices[method_index] = layers_index
                    self.converged_methods[method_index] = 1 # May be superfluous; we already know whether a method is converged by whether its self.convergence_num_layers is nonzero
                    break

        self.max_convergence_layers_index = np.amax(self.convergence_layers_indices)
        self.surface_energies = self.slab_surface_energies[:][self.max_convergence_layers_index] # TODO: make sure the OgerOrganic way of doing things (implemented here) is the best way of doing things
        """
        surface_energies_dict = {}
        for method in self.methods.keys():
            surface_energies_dict[method] = self.surface_energies[self.methods[method]]
        self.surface_energies_dict = surface_energies_dict
        """

    def _plot_slab_surface_energies(self) -> None:
        pass

    def _write_surface_energies(filename: str = "calculated_surface_energies.txt") -> None:
        pass

    def _write_all_energy_data(filename: str = "calculated_energy_data.txt") -> None:
        pass

    def generate_structures(
        self,
        refine_structure: bool = False
    ) -> None:
        # TODO: implement structure generation
        file_names = ["instructions.txt", "POSCAR_OBS"]
        # make instructions.txt
        # make POSCAR_OBS
        # make POSCAR_/layers/


        for num_layers in self.nums_layers:
            file_names.append("POSCAR_" + str(num_layers))

        slabs = []
        for layers_index in range(self.L):
            slabs_num_layers = SurfaceGenerator(
                self.bulk,
                self.miller_index,
                self.nums_layers[layers_index],
                self.vacuum,
                refine_structure,
                True,
                False,
                False
            )
            slabs.append

        return file_names

    def update_total_energies(
        self,
        OBS_energy: float = None,
        slab_energies: dict = None
    ) -> None:
        # TODO: implement read from files
        self.OBS_energy = OBS_energy
        pass

    def calculate(
        self,
        plot: bool = False,
        write_surface_energies: bool = False,
        surface_energies_filename: str = "calculated_surface_energies.txt",
        write_all_energy_data: bool = False,
        all_energy_data_filename: str = "calculated_energy_data.txt"
        ) -> dict:
        """
        The user-facing function by which the object is commanded to perform all calculations necessary to output number-of-layers-nonspecific surface energies from OBS/slab total energy inputs.
        Returns a dictionary with one key for each method for which the object was created to use and the corresponding number-of-layers-nonspecific surface energy as the corresponding item.
            e.g., {"OBS": 12.3456, "Boettger_strict": 78.9012, "Linear_strict": 34.5678, "Linear_lenient": 90.1234}
        """
        # Perform calculations for slab bulk energies, then slab surface energies, then slab-nonspecific surface energies
        self._calculate_slab_bulk_energies()
        self._calculate_slab_surface_energies()
        self._calculate_surface_energies()
        
        # Create the method-to-surface-energy dictionary that is to be returned from the self.surface_energies attribute that was set by the preceding calculations (secifically, self._calculate_surface_energies())
        surface_energies_dict = {}
        for method in self.methods.keys():
            surface_energies_dict[method] = self.surface_energies[self.methods[method]]
        self.surface_energies_dict = surface_energies_dict

        if plot:
            self._plot_slab_surface_energies()

        # Write output files if indicated. Perhaps move these to their own functions.
        if write_surface_energies:
            self._write_surface_energies(surface_energies_filename)
        if write_all_energy_data:
            self._write_all_energy_data(all_energy_data_filename)
        
        return surface_energies_dict
