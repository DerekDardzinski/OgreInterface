from __future__ import annotations
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
import numpy as np
import itertools
import functools
import math
import collections
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Literal,
    Sequence,
    SupportsIndex,
    cast,
)


def get_primitive_structure(
    struc,
    tolerance: float = 0.25,
    use_site_props: bool = False,
    constrain_latt: list | dict | None = None,
):
    """
    This finds a smaller unit cell than the input. Sometimes it doesn"t
    find the smallest possible one, so this method is recursively called
    until it is unable to find a smaller cell.

    NOTE: if the tolerance is greater than 1/2 the minimum inter-site
    distance in the primitive cell, the algorithm will reject this lattice.

    Args:
        tolerance (float), Angstroms: Tolerance for each coordinate of a
            particular site. For example, [0.1, 0, 0.1] in cartesian
            coordinates will be considered to be on the same coordinates
            as [0, 0, 0] for a tolerance of 0.25. Defaults to 0.25.
        use_site_props (bool): Whether to account for site properties in
            differentiating sites.
        constrain_latt (list/dict): List of lattice parameters we want to
            preserve, e.g. ["alpha", "c"] or dict with the lattice
            parameter names as keys and values we want the parameters to
            be e.g. {"alpha": 90, "c": 2.5}.

    Returns:
        The most primitive structure found.
    """
    if constrain_latt is None:
        constrain_latt = []

    def site_label(site):
        if not use_site_props:
            return site.species_string
        d = [site.species_string]
        for k in sorted(site.properties.keys()):
            d.append(k + "=" + str(site.properties[k]))
        return ", ".join(d)

    # group sites by species string
    sites = sorted(struc._sites, key=site_label)

    grouped_sites = [
        list(a[1]) for a in itertools.groupby(sites, key=site_label)
    ]
    grouped_fcoords = [
        np.array([s.frac_coords for s in g]) for g in grouped_sites
    ]

    # min_vecs are approximate periodicities of the cell. The exact
    # periodicities from the supercell matrices are checked against these
    # first
    min_fcoords = min(grouped_fcoords, key=lambda x: len(x))
    min_vecs = min_fcoords - min_fcoords[0]

    # fractional tolerance in the supercell
    super_ftol = np.divide(tolerance, struc.lattice.abc)
    super_ftol_2 = super_ftol * 2

    def pbc_coord_intersection(fc1, fc2, tol):
        """
        Returns the fractional coords in fc1 that have coordinates
        within tolerance to some coordinate in fc2
        """
        d = fc1[:, None, :] - fc2[None, :, :]
        d -= np.round(d)
        np.abs(d, d)
        return fc1[np.any(np.all(d < tol, axis=-1), axis=-1)]

    # here we reduce the number of min_vecs by enforcing that every
    # vector in min_vecs approximately maps each site onto a similar site.
    # The subsequent processing is O(fu^3 * min_vecs) = O(n^4) if we do no
    # reduction.
    # This reduction is O(n^3) so usually is an improvement. Using double
    # the tolerance because both vectors are approximate
    for g in sorted(grouped_fcoords, key=lambda x: len(x)):
        for f in g:
            min_vecs = pbc_coord_intersection(min_vecs, g - f, super_ftol_2)

    def get_hnf(fu):
        """
        Returns all possible distinct supercell matrices given a
        number of formula units in the supercell. Batches the matrices
        by the values in the diagonal (for less numpy overhead).
        Computational complexity is O(n^3), and difficult to improve.
        Might be able to do something smart with checking combinations of a
        and b first, though unlikely to reduce to O(n^2).
        """

        def factors(n):
            for i in range(1, n + 1):
                if n % i == 0:
                    yield i

        for det in factors(fu):
            if det == 1:
                continue
            for a in factors(det):
                for e in factors(det // a):
                    g = det // a // e
                    yield det, np.array(
                        [
                            [[a, b, c], [0, e, f], [0, 0, g]]
                            for b, c, f in itertools.product(
                                range(a), range(a), range(e)
                            )
                        ]
                    )

    # we can't let sites match to their neighbors in the supercell
    grouped_non_nbrs = []
    for gfcoords in grouped_fcoords:
        fdist = gfcoords[None, :, :] - gfcoords[:, None, :]
        fdist -= np.round(fdist)
        np.abs(fdist, fdist)
        non_nbrs = np.any(fdist > 2 * super_ftol[None, None, :], axis=-1)
        # since we want sites to match to themselves
        np.fill_diagonal(non_nbrs, True)
        grouped_non_nbrs.append(non_nbrs)

    num_fu = functools.reduce(math.gcd, map(len, grouped_sites))
    for size, ms in get_hnf(num_fu):
        inv_ms = np.linalg.inv(ms)

        # find sets of lattice vectors that are are present in min_vecs
        dist = inv_ms[:, :, None, :] - min_vecs[None, None, :, :]
        dist -= np.round(dist)
        np.abs(dist, dist)
        is_close = np.all(dist < super_ftol, axis=-1)
        any_close = np.any(is_close, axis=-1)
        inds = np.all(any_close, axis=-1)

        for inv_m, m in zip(inv_ms[inds], ms[inds]):
            new_m = np.dot(inv_m, struc.lattice.matrix)
            ftol = np.divide(tolerance, np.sqrt(np.sum(new_m**2, axis=1)))

            valid = True
            new_coords = []
            new_sp = []
            new_props = collections.defaultdict(list)
            for gsites, gfcoords, non_nbrs in zip(
                grouped_sites, grouped_fcoords, grouped_non_nbrs
            ):
                all_frac = np.dot(gfcoords, m)

                # calculate grouping of equivalent sites, represented by
                # adjacency matrix
                fdist = all_frac[None, :, :] - all_frac[:, None, :]
                fdist = np.abs(fdist - np.round(fdist))
                close_in_prim = np.all(fdist < ftol[None, None, :], axis=-1)
                groups = np.logical_and(close_in_prim, non_nbrs)

                # check that groups are correct
                if not np.all(np.sum(groups, axis=0) == size):
                    valid = False
                    break

                # check that groups are all cliques
                for g in groups:
                    if not np.all(groups[g][:, g]):
                        valid = False
                        break
                if not valid:
                    break

                # add the new sites, averaging positions
                added = np.zeros(len(gsites))
                new_fcoords = all_frac % 1
                for i, group in enumerate(groups):
                    if not added[i]:
                        added[group] = True
                        inds = np.where(group)[0]
                        coords = new_fcoords[inds[0]]
                        for n, j in enumerate(inds[1:]):
                            offset = new_fcoords[j] - coords
                            coords += (offset - np.round(offset)) / (n + 2)
                        new_sp.append(gsites[inds[0]].species)
                        for k in gsites[inds[0]].properties:
                            new_props[k].append(gsites[inds[0]].properties[k])
                        new_coords.append(coords)

            if valid:
                inv_m = np.linalg.inv(m)
                new_l = Lattice(np.dot(inv_m, struc.lattice.matrix))
                s = Structure(
                    new_l,
                    new_sp,
                    new_coords,
                    site_properties=new_props,
                    coords_are_cartesian=False,
                )

                p = s.get_primitive_structure(
                    tolerance=tolerance,
                    use_site_props=use_site_props,
                    constrain_latt=constrain_latt,
                )
                if not constrain_latt:
                    return p

                # Only return primitive structures that
                # satisfy the restriction condition
                p_latt, s_latt = p.lattice, struc.lattice
                if type(constrain_latt).__name__ == "list":
                    if all(
                        getattr(p_latt, pp) == getattr(s_latt, pp)
                        for pp in constrain_latt
                    ):
                        return p
                elif type(constrain_latt).__name__ == "dict":
                    if all(
                        getattr(p_latt, pp) == constrain_latt[pp] for pp in constrain_latt.keys()  # type: ignore
                    ):
                        return p

    return struc.copy()