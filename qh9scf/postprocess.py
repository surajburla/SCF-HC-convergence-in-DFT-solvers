"""Hamiltonian -> density matrix, plus physics diagnostics. float64 throughout."""
from __future__ import annotations

import numpy as np
import scipy.linalg


def symmetrize(M):
    return 0.5 * (M + M.T)


def density_from_hamiltonian(H, S, n_elec: int):
    """Closed-shell P = 2 C_occ C_occ^T from  H C = S C eps.

    Guarantees by construction: P = P^T, Tr(PS) = n_elec, P S P = 2 P.
    Returns (P, eps, homo_lumo_gap).
    """
    if n_elec % 2:
        raise ValueError(f"closed-shell only; got {n_elec} electrons")
    H = symmetrize(np.asarray(H, dtype=np.float64))
    S = np.asarray(S, dtype=np.float64)
    eps, C = scipy.linalg.eigh(H, S)
    n_occ = n_elec // 2
    Co = C[:, :n_occ]
    return 2.0 * Co @ Co.T, eps, float(eps[n_occ] - eps[n_occ - 1])


def trace_error(P, S, n_elec):
    return abs(float(np.sum(P * S)) - n_elec)          # Tr(PS) with P,S symmetric


def idempotency_error(P, S):
    return float(np.linalg.norm(P @ S @ P - 2.0 * P))


def sqrt_psd(S):
    w, U = np.linalg.eigh(S)
    return (U * np.sqrt(np.clip(w, 0, None))) @ U.T


def s_norm(dP, S, Sh=None):
    """|| S^{1/2} dP S^{1/2} ||_F  -- basis-invariant, unlike the raw Frobenius norm."""
    Sh = sqrt_psd(S) if Sh is None else Sh
    return float(np.linalg.norm(Sh @ dP @ Sh))
