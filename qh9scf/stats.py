"""On-site reference energies. numpy-only.

Why per SHELL and not per AO slot: a constant on p_x, p_y, p_z that differs between the three components is NOT
rotation-invariant. A constant shared by all 2l+1 components of a shell is (D I D^T = I), so the model stays equivariant.
"""
from __future__ import annotations

import numpy as np

from .basis import NSLOT, SLOT_SHELLS, SPECIES, orbital_maps

_SP = {z: i for i, z in enumerate(SPECIES)}
SHELL_OF_SLOT = np.repeat(np.arange(len(SLOT_SHELLS)), [2 * l + 1 for l in SLOT_SHELLS])


def _species_per_ao(z, atoms):
    return np.array([_SP[int(v)] for v in z])[atoms]


def estimate_diag_ref_and_scale(flat, indices, n=500):
    """Returns (diag_ref (n_species, 14), scale).  diag_ref = mean H_mumu per (species, shell), repeated over m.
    scale = RMS of the residual matrix (diagonal minus reference), used to normalise the loss."""
    idx = [int(i) for i in list(indices)[:n]]
    n_sh = len(SLOT_SHELLS)
    ssum, scnt = np.zeros((len(SPECIES), n_sh)), np.zeros((len(SPECIES), n_sh))
    for i in idx:
        z, _, H = flat.raw(i)
        atoms, slots = orbital_maps(z)
        sp = _species_per_ao(z, atoms)
        np.add.at(ssum, (sp, SHELL_OF_SLOT[slots]), np.diag(H).astype(np.float64))
        np.add.at(scnt, (sp, SHELL_OF_SLOT[slots]), 1.0)
    ref_shell = np.divide(ssum, scnt, out=np.zeros_like(ssum), where=scnt > 0)
    ref14 = ref_shell[:, SHELL_OF_SLOT]                      # (n_species, 14)
    sq, cnt = 0.0, 0
    for i in idx:
        z, _, H = flat.raw(i)
        atoms, slots = orbital_maps(z)
        R = H.astype(np.float64).copy()
        R[np.diag_indices_from(R)] -= ref14[_species_per_ao(z, atoms), slots]
        sq += float((R ** 2).sum()); cnt += R.size
    return ref14, float(np.sqrt(sq / cnt))
