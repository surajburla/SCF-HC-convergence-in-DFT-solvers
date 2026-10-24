"""GATE 0 - run before any training.

For n molecules: build S with PySCF, derive P_true from the stored H, then check that the B3LYP Fock
built from P_true reproduces the stored H.  If max|F[P]-H| is ~1e-5 Ha or lower, ordering, units, functional
and basis conventions all match.  If it is ~1e-1, a convention is wrong; stop and fix before training.
Also measures how many SCF cycles the oracle guess P_true needs (expect ~1-3).
"""
import argparse
import _path  # noqa: F401
import numpy as np
from qh9scf.data_prep import FlatQH9
from qh9scf.scf_tools import build_mol, gate0_check, run_scf
from qh9scf.postprocess import density_from_hamiltonian

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("-n", type=int, default=5)
ap.add_argument("--xc", nargs="+", default=["b3lyp", "b3lyp5"])
a = ap.parse_args()

flat = FlatQH9(a.root)
bad = 0
for i in range(min(a.n, len(flat))):
    z, pos, H = flat.raw(i)
    r = gate0_check(z, pos, H, tuple(a.xc))
    print(f"[{i}] {r}")
    if not r["ok"]:
        bad += 1
        continue
    best = min(r["F_residual"].values())
    if best > 1e-3:
        bad += 1
    mol = build_mol(z, pos)
    P, _, _ = density_from_hamiltonian(H, mol.intor("int1e_ovlp"), mol.nelectron)
    best_xc = min(r["F_residual"], key=r["F_residual"].get)
    o = run_scf(mol, best_xc, dm0=P)
    m = run_scf(mol, best_xc, init_guess="minao")
    print(f"     oracle dm0: {o.cycles} cycles, E={o.energy:.8f} | minao: {m.cycles} cycles, E={m.energy:.8f} | dE={abs(o.energy-m.energy):.2e}")
print("GATE 0:", "PASS" if bad == 0 else f"FAIL ({bad} problem molecules)  -> fix conventions before training")
