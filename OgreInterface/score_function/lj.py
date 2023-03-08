from OgreInterface.score_function.scatter import scatter_add
from typing import Dict, Optional, Tuple
import torch.nn as nn
import torch


class LJ(nn.Module):
    """
    Compute Coulomb energy from a set of point charges via direct summation. Depending on the form of the
    potential function, the interaction can be damped for short distances. If a cutoff is requested, the full
    potential is shifted, so that it and its first derivative is zero starting from the cutoff.
    Args:
        energy_unit (str/float): Units used for the energy.
        position_unit (str/float): Units used for lengths and positions.
        coulomb_potential (torch.nn.Module): Distance part of the potential.
        output_key (str): Name of the energy property in the output.
        charges_key (str): Key of partial charges in the input batch.
        use_neighbors_lr (bool): Whether to use standard or long range neighbor list elements (default = True).
        cutoff (optional, float): Apply a long range cutoff (potential is shifted to 0, default=None).
    """

    def __init__(
        self,
        cutoff: Optional[float] = None,
    ):
        super(LJ, self).__init__()

        cutoff = torch.tensor(cutoff)
        self.register_buffer("cutoff", cutoff)

    def born_potential(
        self, d_ij: torch.Tensor, n_ij: torch.Tensor, B_ij: torch.Tensor
    ):
        return B_ij * ((1 / (d_ij**n_ij)) - (1 / (self.cutoff**n_ij)))

    def potential_energy(
        self, d_ij: torch.Tensor, delta_e_neg: torch.Tensor, r0: torch.Tensor
    ):
        n = 6
        sigma = r0 / (2 ** (1 / n))

        return (4 * delta_e_neg) * (
            ((sigma / d_ij) ** (2 * n)) - ((sigma / d_ij) ** n)
        )

    def potential_force(
        self, d_ij: torch.Tensor, delta_e_neg: torch.Tensor, r0: torch.Tensor
    ):
        n = 6
        sigma = r0 / (2 ** (1 / n))

        return -(4 * delta_e_neg) * (
            -((12 * sigma**12) / d_ij**13)
            + ((6 * sigma**6) / d_ij**7)
            # ((sigma / d_ij) ** (2 * n)) - ((sigma / d_ij) ** n)
        )

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        r0_dict: Dict[Tuple[int, int, int, int], float],
        z_shift: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the Born repulsion energy.
        Args:
            inputs (dict(str,torch.Tensor)): Input batch.
        Returns:
            dict(str, torch.Tensor): results with Coulomb energy.
        """
        if z_shift:
            z_str = "_z"
        else:
            z_str = ""

        e_neg = inputs["e_negs"].squeeze(-1)
        z = inputs["Z"]

        idx_m = inputs["idx_m"]

        r_ij = inputs[f"Rij{z_str}"]
        idx_i = inputs["idx_i"]
        idx_j = inputs["idx_j"]
        is_film = inputs["is_film"]

        r0_key_array = (
            torch.stack(
                [is_film[idx_i], is_film[idx_j], z[idx_i], z[idx_j]], dim=1
            )
            .numpy()
            .astype(int)
        )
        r0_keys = list(map(tuple, r0_key_array))

        e_neg_ij = 1.0 + torch.abs(e_neg[idx_i] - e_neg[idx_j])
        d_ij = torch.norm(r_ij, dim=1)
        r0_ij = torch.tensor([r0_dict[k] for k in r0_keys]).to(torch.float32)
        r0_ij = r0_ij.view(e_neg_ij.shape)

        n_atoms = z.shape[0]
        n_molecules = int(idx_m[-1]) + 1

        potential_energy = self.potential_energy(
            d_ij, delta_e_neg=e_neg_ij, r0=r0_ij
        )
        potential_force = self.potential_force(
            d_ij, delta_e_neg=e_neg_ij, r0=r0_ij
        )

        # Apply cutoff if requested (shifting to zero)
        if self.cutoff is not None:
            potential_energy = torch.where(
                d_ij <= self.cutoff,
                potential_energy,
                torch.zeros_like(potential_energy),
            )
            potential_force = torch.where(
                d_ij <= self.cutoff,
                potential_force,
                torch.zeros_like(potential_force),
            )

        y_energy = scatter_add(potential_energy, idx_i, dim_size=n_atoms)
        y_energy = scatter_add(y_energy, idx_m, dim_size=n_molecules)
        y_energy = torch.squeeze(y_energy, -1)

        y_force = scatter_add(potential_force, idx_i, dim_size=n_atoms)
        y_force = scatter_add(y_force, idx_m, dim_size=n_molecules)
        y_force = torch.squeeze(y_force, -1)

        return y_energy.numpy(), y_force.numpy()
