"""Train the equivariant Hamiltonian predictor.

Gate 2 first:  python scripts/train.py --root R --overfit 8 --epochs 300
  -> loss must fall by orders of magnitude. If not, the model/assembly is broken.
Then a real split (held-out molecules only are ever reported).
"""
import argparse
import os
import time

import _path  # noqa: F401
import numpy as np
import torch
from torch.utils.data import DataLoader

from qh9scf.basis import from_superblocks, load_ao_basis_change, identity_ao_basis_change
from qh9scf.data_prep import FlatQH9
from qh9scf.dataset import SuperblockDataset, collate, estimate_scale, make_splits
from qh9scf.model import HamiltonianNet, ModelConfig

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--ao-cache", default=os.path.join(os.path.dirname(__file__), "..", "qh9scf", "ao_convention.npz"))
ap.add_argument("--split", default="id", choices=["id", "size"])
ap.add_argument("--subset", type=int, default=5000)
ap.add_argument("--overfit", type=int, default=0)
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="ckpt.pt")
a = ap.parse_args()

torch.manual_seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
flat = FlatQH9(a.root)
tr, va, te = make_splits(flat, a.split, a.seed, a.subset)
if a.overfit:
    tr = va = tr[: a.overfit]
scale = estimate_scale(flat, tr)
Qs = load_ao_basis_change(a.ao_cache) if os.path.exists(a.ao_cache) else identity_ao_basis_change()
if not os.path.exists(a.ao_cache):
    print("WARNING: ao_convention.npz missing -> identity Q. Hamiltonians will NOT be in PySCF AO order. "
          "Run `python -m qh9scf.basis --build-cache` before any SCF benchmark.")
cfg = ModelConfig()
model = HamiltonianNet(cfg, Qs, scale).to(dev)
print(f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M  scale={scale:.4f} Ha  device={dev}  train/val/test={len(tr)}/{len(va)}/{len(te)}")

mk = lambda idx, sh: DataLoader(SuperblockDataset(flat, idx), batch_size=a.bs, shuffle=sh, collate_fn=collate,
                                num_workers=a.workers, persistent_workers=a.workers > 0)
dl_tr, dl_va = mk(tr, True), mk(va, False)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-8)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * len(dl_tr))


def step(batch, train):
    b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
    on, pr = model(b["z"], b["pos"], b["src"], b["dst"])
    s = model.target_scale
    e_on = ((on - b["onsite"] / s).abs() * b["onsite_mask"]).sum()
    e_pr = ((pr - b["pair"] / s).abs() * b["pair_mask"]).sum()
    n_on, n_pr = b["onsite_mask"].sum().clamp(min=1), b["pair_mask"].sum().clamp(min=1)
    loss = 0.5 * (e_on / n_on + e_pr / n_pr)
    return loss, float((e_on + e_pr) / (n_on + n_pr)) * float(s)   # MAE in Hartree over all AO entries


best = 1e9
for ep in range(a.epochs):
    model.train(); t0 = time.time(); run = 0.0
    for batch in dl_tr:
        loss, mae = step(batch, True)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step(); sched.step(); run += mae
    model.eval(); vm = []
    with torch.no_grad():
        for batch in dl_va:
            vm.append(step(batch, False)[1])
    v = float(np.mean(vm))
    print(f"epoch {ep:3d}  train MAE {run/len(dl_tr):.3e} Ha  val MAE {v:.3e} Ha  ({time.time()-t0:.0f}s)")
    if v < best:
        best = v
        torch.save(dict(state=model.state_dict(), cfg=model.config_dict(), Qs=Qs, scale=scale,
                        splits=dict(train=tr, val=va, test=te)), a.out)
print("best val MAE (Ha):", best, "->", a.out)
