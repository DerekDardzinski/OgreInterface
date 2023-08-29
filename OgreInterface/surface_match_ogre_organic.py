#Iterator
import os
from itertools import product
import numpy as np
import math
from pymatgen.io.cif import CifWriter
from ase.build import general_surface
from ase.spacegroup import crystal
from ase.visualize import view
from ase.lattice.surface import *
from ase.io import *
import pymatgen as mg
from pymatgen.io.vasp.inputs import Poscar
import argparse
import pymatgen as mg
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.surface import Slab, SlabGenerator, ReconstructionGenerator
from pymatgen.analysis.substrate_analyzer import SubstrateAnalyzer,ZSLGenerator
from pymatgen.symmetry.analyzer import *
from random import uniform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import *
#from mayavi.mlab import *
import scipy.optimize as optimize
import scipy.interpolate
import matplotlib.tri as tri
import scipy.spatial as ss
#import seaborn as sns
import matplotlib as mpl
import scipy.interpolate as interp
import time
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator
#from numba import jit
import math as m
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
from ase.optimize import BFGS
from ase.calculators.dftd3 import DFTD3
from ase import units
from ase.io import read, write
import torchani


working_dir = ""
sub_struc = Structure.from_file(working_dir + "POSCAR_sub")
film_struc = Structure.from_file(working_dir + "POSCAR_film")
vacuum = 40
BO_iterations = 100


sub_Cont_coords = sub_struc.cart_coords
film_Cont_coords = film_struc.cart_coords
interface_sps = sub_struc.species + film_struc.species


lattice_vecs = sub_struc.lattice.matrix
lat_vecs = sub_struc.lattice.matrix
len_vec_a, len_vec_b, len_vec_c = np.linalg.norm(lattice_vecs[0]), np.linalg.norm(lattice_vecs[1]), \
                                  np.linalg.norm(lattice_vecs[2])


def struc_generator(x_shift, y_shift, z_shift):
    sub_slab_Cont_coords = sub_Cont_coords.copy()
    film_slab_Cont_coords = film_Cont_coords.copy()

    for i in range(len(sub_slab_Cont_coords)):
        sub_slab_Cont_coords[i, 2] = sub_slab_Cont_coords[i, 2] + vacuum
    for i in range(len(film_slab_Cont_coords)):
        film_slab_Cont_coords[i, 0] = film_slab_Cont_coords[i, 0] + x_shift
        film_slab_Cont_coords[i, 1] = film_slab_Cont_coords[i, 1] + y_shift
        film_slab_Cont_coords[i, 2] = film_slab_Cont_coords[i, 2] + vacuum + z_shift

    sub_slab_Cont_coords = np.array(sub_slab_Cont_coords)
    film_slab_Cont_coords = np.array(film_slab_Cont_coords)

    interface_lat_vec = np.array([lat_vecs[0], lat_vecs[1], [0, 0, np.max(film_Cont_coords[:, 2]) + 2 * vacuum]])
    interface_coords = np.concatenate((sub_slab_Cont_coords, film_slab_Cont_coords))
    # interface_sps = sub_struc_species + film_struc_species
    interface_struc = Structure(interface_lat_vec, interface_sps, interface_coords, coords_are_cartesian=True)
    return interface_struc



def energy_calulator(x_shift, y_shift, z_shift):
    sub_slab_Cont_coords = sub_Cont_coords.copy()
    film_slab_Cont_coords = film_Cont_coords.copy()

    for i in range(len(sub_slab_Cont_coords)):
        sub_slab_Cont_coords[i, 2] = sub_slab_Cont_coords[i, 2] + vacuum
    for i in range(len(film_slab_Cont_coords)):
        film_slab_Cont_coords[i, 0] = film_slab_Cont_coords[i, 0] + x_shift
        film_slab_Cont_coords[i, 1] = film_slab_Cont_coords[i, 1] + y_shift
        film_slab_Cont_coords[i, 2] = film_slab_Cont_coords[i, 2] + vacuum + z_shift

    sub_slab_Cont_coords = np.array(sub_slab_Cont_coords)
    film_slab_Cont_coords = np.array(film_slab_Cont_coords)

    interface_lat_vec = np.array([lat_vecs[0], lat_vecs[1], [0, 0, np.max(film_Cont_coords[:, 2]) + 2 * vacuum]])
    interface_coords = np.concatenate((sub_slab_Cont_coords, film_slab_Cont_coords))
    interface_struc = Structure(interface_lat_vec, interface_sps, interface_coords, coords_are_cartesian=True)
    Poscar(interface_struc).write_file("POSCAR_BO")
    structure = read("POSCAR_BO", format='vasp')
    calculator = torchani.models.ANI2x().ase()
    structure.set_calculator(calculator)
    potential_energy = structure.get_potential_energy()
    
    structure_corr = read("POSCAR_BO", format='vasp')
    d3 = DFTD3()
    structure_corr.calc = d3
    correction_energy = structure_corr.get_potential_energy()
    tot_energy = potential_energy + correction_energy
    return tot_energy

def surface_matching():


    pbounds = {'x_shift': (0, len_vec_a), 'y_shift': (0, len_vec_b), 'z_shift': (-3, 0)}
    optimizer = BayesianOptimization(f=energy_calulator, pbounds=pbounds, verbose=2, random_state=1)
    optimizer.maximize(acquisition_function=UtilityFunction(kind='ucb', kappa=8), n_iter=BO_iterations)

    coord_diffs, score_targets = [], []
    x_diffs, y_diffs, z_diffs = [], [], []

    for bo_index in range(len(optimizer.res)):
        x_diffs.append(optimizer.res[bo_index]['params']['x_shift'])
        y_diffs.append(optimizer.res[bo_index]['params']['y_shift'])
        z_diffs.append(optimizer.res[bo_index]['params']['z_shift'])
        coord_diffs.append([optimizer.res[bo_index]['params']['x_shift'], optimizer.res[bo_index][
            'params']['y_shift'], optimizer.res[bo_index]['params']['z_shift']])
        score_targets.append(-optimizer.res[bo_index]['target'])
    # Best 3D

    targets = np.round(np.array(score_targets), 5)
    sorted_targets = sorted(targets, reverse=False)
    print(sorted_targets)
    lowest_indices = []
    for score_value in sorted_targets[0:20]:
        lowest_indices.append(np.where(targets == score_value))

    lowest_score_coords = []
    for score_index in lowest_indices:
        lowest_score_coords.append(
            [np.round(x_diffs[score_index[0][0]], 5), np.round(y_diffs[score_index[0][0]], 5),
             np.round(z_diffs[score_index[0][0]], 5)])
        print("x:", np.round(x_diffs[score_index[0][0]], 3), "  y:", np.round(y_diffs[score_index[0][0]],
                                                                              3), "  z:",
              np.round(z_diffs[score_index[0][0]], 3), " target:", np.round(score_targets[score_index[0][0]], 5))

surface_matching()
Poscar(int_struc).write_file(working_dir + "POSCAR_opt" )










