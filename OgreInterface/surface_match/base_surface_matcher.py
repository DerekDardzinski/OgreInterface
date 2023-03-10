from OgreInterface.surfaces import Interface
from typing import List
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Polygon
from copy import deepcopy


class BaseSurfaceMatcher:
    def __init__(
        self,
        interface: Interface,
        grid_density: float = 2.5,
    ):
        self.interface = interface
        self.matrix = deepcopy(interface._orthogonal_structure.lattice.matrix)
        self._vol = np.linalg.det(self.matrix)

        if self._vol < 0:
            self.matrix *= -1
            self._vol *= -1

        self.grid_density = grid_density

        (
            self.shift_matrix,
            self.shift_images,
            self.shifts,
        ) = self._generate_shifts()

    def _generate_shifts(self) -> List[np.ndarray]:
        (
            sub_matrix,
            sub_images,
            film_matrix,
            film_images,
        ) = self.interface._get_oriented_cell_and_images(strain=True)

        if self.interface.substrate.area < self.interface.film.area:
            shift_matrix = sub_matrix
            shift_images = sub_images
        else:
            shift_matrix = film_matrix
            shift_images = film_images

        iface_inv_matrix = (
            self.interface._orthogonal_structure.lattice.inv_matrix
        )

        grid_density_x = int(
            np.round(np.linalg.norm(shift_matrix[0]) * self.grid_density)
        )
        grid_density_y = int(
            np.round(np.linalg.norm(shift_matrix[1]) * self.grid_density)
        )

        self.grid_density_x = grid_density_x
        self.grid_density_y = grid_density_y

        grid_x = np.linspace(0, 1, grid_density_x)
        grid_y = np.linspace(0, 1, grid_density_y)

        X, Y = np.meshgrid(grid_x, grid_y)

        prim_frac_shifts = (
            np.c_[X.ravel(), Y.ravel(), np.zeros(Y.shape).ravel()]
            + shift_images[0]
        )
        prim_cart_shifts = prim_frac_shifts.dot(shift_matrix)
        iface_frac_shifts = prim_cart_shifts.dot(iface_inv_matrix).reshape(
            X.shape + (-1,)
        )

        return shift_matrix, shift_images, iface_frac_shifts

    def get_optmized_structure(self):
        opt_shift = self.opt_xy_shift

        self.interface.shift_film_inplane(
            x_shift=opt_shift[0], y_shift=opt_shift[1], fractional=True
        )

    def _plot_heatmap(
        self,
        fig,
        ax,
        X,
        Y,
        Z,
        Z_force,
        cmap,
        fontsize,
        show_max,
        add_color_bar,
    ):
        ax.set_xlabel(r"Shift in $x$ ($\AA$)", fontsize=fontsize)
        ax.set_ylabel(r"Shift in $y$ ($\AA$)", fontsize=fontsize)

        im = ax.contourf(
            X,
            Y,
            Z,
            cmap=cmap,
            levels=200,
            norm=Normalize(vmin=np.nanmin(Z), vmax=np.nanmax(Z)),
        )

        if add_color_bar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("top", size="5%", pad=0.1)
            cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
            cbar.ax.tick_params(labelsize=fontsize)
            cbar.ax.locator_params(nbins=3)

            if show_max:
                opt_ind = np.argmin(np.abs(Z_force).ravel())
                E_max = Z.ravel()[opt_ind]
                label = (
                    "$E_{adh}$ (eV/$\\AA^{2}$) : "
                    + "$E_{max}$ = "
                    + f"{E_max:.4f}"
                )
                cbar.set_label(label, fontsize=fontsize)
            else:
                label = "$E_{adh}$ (eV/$\\AA^{2}$)"
                cbar.set_label(label, fontsize=fontsize)

            cax.xaxis.set_ticks_position("top")
            cax.xaxis.set_label_position("top")
            ax.tick_params(labelsize=fontsize)

    def _plot_surface_matching(
        self,
        fig,
        ax,
        X,
        Y,
        Z,
        Z_force,
        dpi,
        cmap,
        fontsize,
        show_max,
        shift,
    ):
        for i, image in enumerate(self.shift_images):
            X_plot, Y_plot, Z_plot = self._get_interpolated_data(Z, image)
            _, _, Z_plot_forces = self._get_interpolated_data(Z_force, image)
            Z_plot_forces *= Z_plot_forces

            if i == 0:
                self._plot_heatmap(
                    fig=fig,
                    ax=ax,
                    X=X_plot,
                    Y=Y_plot,
                    Z=Z_plot,
                    Z_force=Z_force,
                    cmap=cmap,
                    fontsize=fontsize,
                    show_max=show_max,
                    add_color_bar=True,
                )

                frac_shifts = np.c_[
                    X_plot.ravel(),
                    Y_plot.ravel(),
                    np.zeros(Y_plot.shape).ravel(),
                ].dot(np.linalg.inv(self.matrix))
                # opt_shift = frac_shifts[np.argmax(Z_plot.ravel())]
                opt_ind = np.argmin(np.abs(Z_plot_forces).ravel())
                opt_shift = frac_shifts[opt_ind]
                max_Z = Z_plot.ravel()[opt_ind]
                plot_shift = opt_shift.dot(self.matrix)

                ax.scatter(
                    [plot_shift[0]],
                    [plot_shift[1]],
                    fc="white",
                    ec="black",
                    marker="X",
                    s=100,
                    zorder=10,
                )

                if shift:
                    self.opt_xy_shift = opt_shift[:2]
            else:
                self._plot_heatmap(
                    fig=fig,
                    ax=ax,
                    X=X_plot,
                    Y=Y_plot,
                    Z=Z_plot,
                    Z_force=Z_force,
                    cmap=cmap,
                    fontsize=fontsize,
                    show_max=show_max,
                    add_color_bar=False,
                )

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

            for shift in sc_shifts:
                shift_coords = (coords + shift).dot(self.matrix)
                poly = Polygon(
                    xy=shift_coords[:, :2],
                    closed=True,
                    facecolor="white",
                    edgecolor="white",
                    linewidth=1,
                    zorder=200,
                )
                ax.add_patch(poly)

        return max_Z
