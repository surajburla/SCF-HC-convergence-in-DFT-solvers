"""GATE 4 - the only evidence that counts: real PySCF RKS/B3LYP/def2-SVP, held-out molecules, same grid,
same tolerance, fresh solver per run, wall-time includes model inference + eigh.

Guesses compared:  PySCF minao (default), atom, huckel, vsap, 1e | ML (H_pred -> P) | oracle P_true (bound).
Outputs CSV + summary. A speedup only counts if |E_ml - E_minao| stays below --e-tol.
"""
import argparse
import csv
import time

import _path  # noqa: F401
import numpy as np
import torch

from qh9scf.data_prep import FlatQH9
from qh9scf.model import HamiltonianNet, ModelConfig
from qh9scf.postprocess import density_from_hamiltonian, idempotency_error, s_norm, trace_error
from qh9scf.scf_tools import build_mol, run_scf

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--xc", default="b3lyp5")
ap.add_argument("-n", type=int, default=20)
ap.add_argument("--baselines", nargs="*", default=["minao", "atom", "huckel", "vsap", "1e"])
ap.add_argument("--e-tol", type=float, default=1e-6)
ap.add_argument("--csv", default="scf_benchmark.csv")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
model = HamiltonianNet(ModelConfig(**ck["cfg"]), ck["Qs"], ck["scale"]).to(dev)
model.load_state_dict(ck["state"]); model.eval()
flat = FlatQH9(a.root)
test_idx = [int(i) for i in ck["splits"]["test"][: a.n]]

rows = []
for i in test_idx:
    z, pos, H_true = flat.raw(i)
    mol = build_mol(z, pos)
    S = mol.intor("int1e_ovlp")
    row = dict(idx=i, atoms=len(z), nbasis=S.shape[0])

    ref = run_scf(mol, a.xc, "minao")
    row.update(minao_cycles=ref.cycles, minao_fock=ref.fock_builds, minao_t=ref.wall, E_ref=ref.energy, minao_conv=ref.converged)
    for g in a.baselines:
        if g == "minao":
            continue
        try:
            r = run_scf(mol, a.xc, g)
            row.update({f"{g}_cycles": r.cycles, f"{g}_t": r.wall})
        except Exception as e:  # guess not available in this PySCF build
            row[f"{g}_cycles"] = None

    t0 = time.perf_counter()
    H_pred = model.predict_hamiltonian(z, pos, dev)
    t1 = time.perf_counter()
    P_ml, _, gap = density_from_hamiltonian(H_pred, S, mol.nelectron)
    t2 = time.perf_counter()
    ml = run_scf(mol, a.xc, dm0=P_ml)
    P_true, _, _ = density_from_hamiltonian(H_true, S, mol.nelectron)
    orc = run_scf(mol, a.xc, dm0=P_true)

    row.update(
        ml_cycles=ml.cycles, ml_fock=ml.fock_builds, ml_scf_t=ml.wall, ml_infer_t=t1 - t0, ml_eigh_t=t2 - t1,
        ml_total_t=(t2 - t0) + ml.wall, dE_ml=abs(ml.energy - ref.energy), ml_conv=ml.converged,
        oracle_cycles=orc.cycles, dE_oracle=abs(orc.energy - ref.energy),
        H_mae=float(np.abs(H_pred - H_true).mean()), dP_snorm=s_norm(P_ml - P_true, S),
        trace_err=trace_error(P_ml, S, mol.nelectron), idem_err=idempotency_error(P_ml, S), gap=gap,
        RIC=ml.cycles / ref.cycles, speedup=ref.wall / ((t2 - t0) + ml.wall))
    rows.append(row)
    print(f"[{i}] atoms={row['atoms']:2d} minao={ref.cycles:2d}  ml={ml.cycles:2d}  oracle={orc.cycles:2d}  "
          f"RIC={row['RIC']:.2f} speedup={row['speedup']:.2f} dE={row['dE_ml']:.1e}")

keys = sorted({k for r in rows for k in r})
with open(a.csv, "w", newline="") as f:
    w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)

ok = [r for r in rows if r["ml_conv"] and r["dE_ml"] < a.e_tol]
cy_ref, cy_ml = sum(r["minao_cycles"] for r in ok), sum(r["ml_cycles"] for r in ok)
print("\n=== SUMMARY (valid = converged and |dE| < e-tol) ===")
print(f"valid {len(ok)}/{len(rows)} | cycle reduction vs minao: {100*(1-cy_ml/max(cy_ref,1)):.1f}% "
      f"| median RIC {np.median([r['RIC'] for r in ok]):.2f} | median wall speedup {np.median([r['speedup'] for r in ok]):.2f}x "
      f"| oracle cycles {np.mean([r['oracle_cycles'] for r in ok]):.1f}")
