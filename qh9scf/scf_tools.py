"""Real PySCF measurements. No mock solver anywhere in this project."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .postprocess import density_from_hamiltonian, idempotency_error, trace_error

XC_CANDIDATES = ("b3lyp5", "b3lyp")   # QH9 used B3LYP; which VWN flavour is decided by gate 0, not by memory


def build_mol(z, pos, basis="def2-svp"):
    from pyscf import gto
    mol = gto.Mole()
    mol.atom = [(int(zi), tuple(float(x) for x in p)) for zi, p in zip(z, pos)]   # Angstrom (PySCF default)
    mol.basis, mol.spin, mol.charge, mol.verbose = basis, 0, 0, 0
    mol.build()
    return mol


def make_rks(mol, xc="b3lyp5", grid_level=3, conv_tol=1e-8, max_cycle=100):
    from pyscf import dft
    mf = dft.RKS(mol)
    mf.xc, mf.conv_tol, mf.max_cycle = xc, conv_tol, max_cycle
    mf.grids.level = grid_level
    return mf


@dataclass
class ScfResult:
    converged: bool
    cycles: int
    fock_builds: int
    energy: float
    wall: float


def run_scf(mol, xc="b3lyp5", init_guess="minao", dm0=None, **kw) -> ScfResult:
    """Fresh RKS object per call (so grids are rebuilt: no hidden cache advantage)."""
    mf = make_rks(mol, xc, **kw)
    cyc, fb = [0], [0]
    mf.callback = lambda envs: cyc.__setitem__(0, cyc[0] + 1)
    orig = mf.get_veff

    def counted(*a, **k):
        fb[0] += 1
        return orig(*a, **k)

    mf.get_veff = counted
    if dm0 is None:
        mf.init_guess = init_guess
    t0 = time.perf_counter()
    e = mf.kernel(dm0=dm0)
    return ScfResult(bool(mf.converged), cyc[0], fb[0], float(e), time.perf_counter() - t0)


def fock_from_density(mol, P, xc="b3lyp5", grid_level=3):
    mf = make_rks(mol, xc, grid_level)
    return mf.get_hcore() + mf.get_veff(mol, P)


def gate0_check(z, pos, H_true, xcs=XC_CANDIDATES, grid_level=3):
    """Dataset-convention gate: is F[P_true] == H_true?  Wrong ordering/units/functional shows up here."""
    mol = build_mol(z, pos)
    S = mol.intor("int1e_ovlp")
    if S.shape != H_true.shape:
        return dict(ok=False, reason=f"shape S {S.shape} vs H {H_true.shape}")
    P, eps, gap = density_from_hamiltonian(H_true, S, mol.nelectron)
    out = dict(ok=True, n_basis=S.shape[0], gap=gap, trace_err=trace_error(P, S, mol.nelectron),
               idem_err=idempotency_error(P, S), F_residual={})
    for xc in xcs:
        F = fock_from_density(mol, P, xc, grid_level)
        out["F_residual"][xc] = float(np.abs(F - H_true).max())
    return out
