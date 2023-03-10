from OgreInterface.score_function.lj import LJ
from OgreInterface.score_function.generate_inputs import generate_dict_torch
from OgreInterface.surfaces import Interface
from OgreInterface.surface_match.base_surface_matcher import BaseSurfaceMatcher
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.local_env import CrystalNN
from ase.data import chemical_symbols, covalent_radii
from typing import List
from ase import Atoms
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from itertools import groupby, combinations_with_replacement, product


class LJSurfaceMatcher(BaseSurfaceMatcher):
    def __init__(
        self,
        interface: Interface,
        grid_density: float = 2.5,
        cutoff: float = 7.0,
    ):
        super().__init__(
            interface=interface,
            grid_density=grid_density,
        )
        self.cutoff = cutoff
        self.r0_dict = self._get_r0s(
            sub=self.interface.substrate.bulk_structure,
            film=self.interface.film.bulk_structure,
        )
        self.d_interface = self.interface.interfacial_distance
        self.film_part = self.interface._orthogonal_film_structure
        self.sub_part = self.interface._orthogonal_film_structure
        self.opt_xy_shift = np.zeros(2)

        self.z_PES_data = None

    def get_optmized_structure(self):
        opt_shift = self.opt_xy_shift

        self.interface.shift_film_inplane(
            x_shift=opt_shift[0], y_shift=opt_shift[1], fractional=True
        )

    def _get_charges(self):
        sub = self.interface.substrate.bulk_structure
        film = self.interface.film.bulk_structure
        sub_oxidation_state = sub.composition.oxi_state_guesses()[0]
        film_oxidation_state = film.composition.oxi_state_guesses()[0]

        sub_oxidation_state.update(film_oxidation_state)

        return sub_oxidation_state

    def _get_neighborhood_info(self, struc):
        Zs = np.unique(struc.atomic_numbers)
        combos = combinations_with_replacement(Zs, 2)
        neighbor_dict = {c: None for c in combos}

        neighbor_list = []

        cnn = CrystalNN(search_cutoff=7.0)
        for i, site in enumerate(struc.sites):
            info_dict = cnn.get_nn_info(struc, i)
            for neighbor in info_dict:
                dist = site.distance(neighbor["site"])
                species = tuple(
                    sorted([site.specie.Z, neighbor["site"].specie.Z])
                )
                neighbor_list.append([species, dist])

        sorted_neighbor_list = sorted(neighbor_list, key=lambda x: x[0])
        groups = groupby(sorted_neighbor_list, key=lambda x: x[0])

        for group in groups:
            nn = list(zip(*group[1]))[1]
            neighbor_dict[group[0]] = np.min(nn)

        for n, d in neighbor_dict.items():
            if d is None:
                neighbor_dict[n] = covalent_radii[n[0]] + covalent_radii[n[1]]
            else:
                if d < 4.0:
                    neighbor_dict[n] = (
                        covalent_radii[n[0]] + covalent_radii[n[1]]
                    )

        return neighbor_dict

    def _get_r0s(self, sub, film):
        sub_dict = self._get_neighborhood_info(sub)
        film_dict = self._get_neighborhood_info(film)

        interface_atomic_numbers = np.unique(
            np.concatenate([sub.atomic_numbers, film.atomic_numbers])
        )

        covalent_radius_dict = {
            n: covalent_radii[n] for n in interface_atomic_numbers
        }

        interface_combos = product(interface_atomic_numbers, repeat=2)
        interface_neighbor_dict = {}
        for c in interface_combos:
            interface_neighbor_dict[(0, 0) + c] = None
            interface_neighbor_dict[(1, 1) + c] = None
            interface_neighbor_dict[(0, 1) + c] = None
            interface_neighbor_dict[(1, 0) + c] = None

        all_keys = np.array(list(sub_dict.keys()) + list(film_dict.keys()))
        unique_keys = np.unique(all_keys, axis=0)
        unique_keys = list(map(tuple, unique_keys))

        for key in unique_keys:
            rev_key = tuple(reversed(key))
            covalent_sum_d = (
                covalent_radius_dict[key[0]] + covalent_radius_dict[key[1]]
            )
            if key in sub_dict and key in film_dict:
                sub_d = sub_dict[key]
                film_d = film_dict[key]
                interface_neighbor_dict[(0, 0) + key] = sub_d
                interface_neighbor_dict[(1, 1) + key] = film_d
                interface_neighbor_dict[(0, 1) + key] = (sub_d + film_d) / 2
                interface_neighbor_dict[(1, 0) + key] = (sub_d + film_d) / 2
                interface_neighbor_dict[(0, 0) + rev_key] = sub_d
                interface_neighbor_dict[(1, 1) + rev_key] = film_d
                interface_neighbor_dict[(0, 1) + rev_key] = (
                    sub_d + film_d
                ) / 2
                interface_neighbor_dict[(1, 0) + rev_key] = (
                    sub_d + film_d
                ) / 2

            if key in sub_dict and key not in film_dict:
                sub_d = sub_dict[key]
                interface_neighbor_dict[(0, 0) + key] = sub_d
                interface_neighbor_dict[(1, 1) + key] = covalent_sum_d
                interface_neighbor_dict[(0, 1) + key] = sub_d
                interface_neighbor_dict[(1, 0) + key] = sub_d
                interface_neighbor_dict[(0, 0) + rev_key] = sub_d
                interface_neighbor_dict[(1, 1) + rev_key] = covalent_sum_d
                interface_neighbor_dict[(0, 1) + rev_key] = sub_d
                interface_neighbor_dict[(1, 0) + rev_key] = sub_d

            if key not in sub_dict and key in film_dict:
                film_d = film_dict[key]
                interface_neighbor_dict[(1, 1) + key] = film_d
                interface_neighbor_dict[(0, 0) + key] = covalent_sum_d
                interface_neighbor_dict[(0, 1) + key] = film_d
                interface_neighbor_dict[(1, 0) + key] = film_d
                interface_neighbor_dict[(1, 1) + rev_key] = film_d
                interface_neighbor_dict[(0, 0) + rev_key] = covalent_sum_d
                interface_neighbor_dict[(0, 1) + rev_key] = film_d
                interface_neighbor_dict[(1, 0) + rev_key] = film_d

            if key not in sub_dict and key not in film_dict:
                interface_neighbor_dict[(0, 0) + key] = covalent_sum_d
                interface_neighbor_dict[(1, 1) + key] = covalent_sum_d
                interface_neighbor_dict[(0, 1) + key] = covalent_sum_d
                interface_neighbor_dict[(1, 0) + key] = covalent_sum_d
                interface_neighbor_dict[(0, 0) + rev_key] = covalent_sum_d
                interface_neighbor_dict[(1, 1) + rev_key] = covalent_sum_d
                interface_neighbor_dict[(0, 1) + rev_key] = covalent_sum_d
                interface_neighbor_dict[(1, 0) + rev_key] = covalent_sum_d

        for key, val in interface_neighbor_dict.items():
            if val is None:
                covalent_sum_d = (
                    covalent_radius_dict[key[2]] + covalent_radius_dict[key[3]]
                )
                interface_neighbor_dict[key] = covalent_sum_d

        return interface_neighbor_dict

    def _get_shifted_atoms(self, shifts: np.ndarray) -> List[Atoms]:
        atoms = []

        for shift in shifts:
            # Shift in-plane
            self.interface.shift_film_inplane(
                x_shift=shift[0], y_shift=shift[1], fractional=True
            )

            # Get inplane shifted atoms
            shifted_atoms = self.interface.get_interface(
                orthogonal=True, return_atoms=True
            )

            # Add the is_film property
            shifted_atoms.set_array(
                "is_film",
                self.interface._orthogonal_structure.site_properties[
                    "is_film"
                ],
            )

            self.interface.shift_film_inplane(
                x_shift=-shift[0], y_shift=-shift[1], fractional=True
            )

            # Add atoms to the list
            atoms.append(shifted_atoms)

        return atoms

    def _generate_inputs(self, atoms_list):
        inputs = generate_dict_torch(
            atoms=atoms_list,
            cutoff=self.cutoff,
        )

        return inputs

    def _calculate_lj(self, inputs, z_shift=False):
        lj = LJ(cutoff=self.cutoff)
        lj_energy = lj.forward(inputs, z_shift=z_shift, r0_dict=self.r0_dict)

        return lj_energy

    def _get_interpolated_data(self, Z, image):
        x_grid = np.linspace(0, 1, self.grid_density_x)
        y_grid = np.linspace(0, 1, self.grid_density_y)
        spline = RectBivariateSpline(y_grid, x_grid, Z)

        x_grid_interp = np.linspace(0, 1, 101)
        y_grid_interp = np.linspace(0, 1, 101)

        X_interp, Y_interp = np.meshgrid(x_grid_interp, y_grid_interp)
        Z_interp = spline.ev(xi=Y_interp, yi=X_interp)
        frac_shifts = (
            np.c_[
                X_interp.ravel(),
                Y_interp.ravel(),
                np.zeros(X_interp.shape).ravel(),
            ]
            + image
        )

        cart_shifts = frac_shifts.dot(self.shift_matrix)

        X_cart = cart_shifts[:, 0].reshape(X_interp.shape)
        Y_cart = cart_shifts[:, 1].reshape(Y_interp.shape)

        return X_cart, Y_cart, Z_interp

    def run_surface_matching(
        self,
        cmap: str = "jet",
        fontsize: int = 14,
        output: str = "PES.png",
        shift: bool = True,
        show_born_and_coulomb: bool = False,
        dpi: int = 400,
        show_max: bool = False,
    ) -> float:
        shifts = self.shifts
        batch_atoms_list = [self._get_shifted_atoms(shift) for shift in shifts]
        batch_inputs = [self._generate_inputs(b) for b in batch_atoms_list]

        sub_atoms = self.interface.get_substrate_supercell(return_atoms=True)
        sub_atoms.set_array("is_film", np.zeros(len(sub_atoms)).astype(bool))

        film_atoms = self.interface.get_film_supercell(return_atoms=True)
        film_atoms.set_array("is_film", np.ones(len(film_atoms)).astype(bool))

        sub_film_atoms = [sub_atoms, film_atoms]
        sub_film_inputs = self._generate_inputs(sub_film_atoms)
        sub_film_lj, sub_film_force = self._calculate_lj(sub_film_inputs)

        interface_lj, interface_force = np.dstack(
            [self._calculate_lj(b) for b in batch_inputs]
        ).transpose(0, 2, 1)

        sub_lj = sub_film_lj[0]
        film_lj = sub_film_lj[1]

        sub_force = sub_film_force[0]
        film_force = sub_film_force[1]

        x_grid = np.linspace(0, 1, self.grid_density_x)
        y_grid = np.linspace(0, 1, self.grid_density_y)
        X, Y = np.meshgrid(x_grid, y_grid)

        Z = (sub_lj + film_lj - interface_lj) / self.interface.area
        # Z = interface_lj
        Z_force = sub_force + film_force - interface_force

        a = self.matrix[0, :2]
        b = self.matrix[1, :2]

        borders = np.vstack([np.zeros(2), a, a + b, b, np.zeros(2)])

        x_size = borders[:, 0].max() - borders[:, 0].min()
        y_size = borders[:, 1].max() - borders[:, 1].min()

        ratio = y_size / x_size

        if ratio < 1:
            figx = 5 / ratio
            figy = 5
        else:
            figx = 5
            figy = 5 * ratio

        fig, ax = plt.subplots(
            figsize=(figx, figy),
            dpi=dpi,
        )

        ax.plot(
            borders[:, 0],
            borders[:, 1],
            color="black",
            linewidth=1,
            zorder=300,
        )

        max_Z = self._plot_surface_matching(
            fig=fig,
            ax=ax,
            X=X,
            Y=Y,
            Z=Z,
            Z_force=Z_force,
            dpi=dpi,
            cmap=cmap,
            fontsize=fontsize,
            show_max=show_max,
            shift=True,
        )

        ax.set_xlim(borders[:, 0].min(), borders[:, 0].max())
        ax.set_ylim(borders[:, 1].min(), borders[:, 1].max())
        ax.set_aspect("equal")

        fig.tight_layout()
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)

        return max_Z
