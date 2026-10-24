"""SE(3)/O(3)-equivariant Hamiltonian predictor (e3nn).

Pipeline:  Z,pos -> radial+angular edge features -> equivariant message passing (scalars + l=1,2)
           -> on-site and pair heads emit irreps coefficients for every (shell_a, shell_b) pair
           -> fixed CG assembly matrix turns coefficients into 14x14 AO superblocks
           -> scatter to the AO matrix (see basis.py) -> symmetrize.
Equivariance is tested numerically in tests/test_equivariance.py; do not claim it before that test passes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import e3nn
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import Gate

try:  # skip TorchScript compilation of generated graphs; a known crash source on some torch/WSL builds
    e3nn.set_optimization_defaults(jit_script_fx=False)
except Exception:
    pass

from .basis import NSLOT, SLOT_OFFS, SLOT_SHELLS, SPECIES, orbital_maps


@dataclass
class ModelConfig:
    n0: int = 16          # multiplicity of 0e
    n1: int = 8           # multiplicity of 1o
    n2: int = 8           # multiplicity of 2e
    n_layers: int = 3
    r_cut: float = 5.0    # message-passing cutoff (A)
    r_head: float = 12.0  # radial range of the pair head (A); pairs beyond it still get a block
    n_rbf: int = 10
    lmax_head: int = 4    # needed: d-d blocks contain l up to 4
    avg_neighbors: float = 12.0
    radial_hidden: int = 64


def build_block_irreps():
    counts = {}
    for la in SLOT_SHELLS:
        for lb in SLOT_SHELLS:
            p = (-1) ** (la + lb)
            for l in range(abs(la - lb), la + lb + 1):
                counts[(l, p)] = counts.get((l, p), 0) + 1
    keys = sorted(counts, key=lambda k: (k[0], -k[1]))
    irreps = o3.Irreps([(counts[k], k) for k in keys])
    assert irreps.dim == NSLOT * NSLOT, irreps
    return irreps


def build_assembly_matrix(block_irreps, Qs):
    """M (196 x dim): flat 14x14 block = M @ coefficients.  Isometric by the sqrt(2l+1) factor."""
    start, off = {}, 0
    for mul, ir in block_irreps:
        start[(ir.l, ir.p)] = off
        off += mul * ir.dim
    used = {k: 0 for k in start}
    M = np.zeros((NSLOT * NSLOT, block_irreps.dim))
    for la, oa in zip(SLOT_SHELLS, SLOT_OFFS):
        for lb, ob in zip(SLOT_SHELLS, SLOT_OFFS):
            p = (-1) ** (la + lb)
            for l in range(abs(la - lb), la + lb + 1):
                key = (l, p)
                c0 = start[key] + used[key] * (2 * l + 1)
                used[key] += 1
                w = o3.wigner_3j(la, lb, l).double().numpy() * math.sqrt(2 * l + 1)
                T = np.einsum("xi,yj,ijm->xym", Qs[la], Qs[lb], w)
                for x in range(2 * la + 1):
                    for y in range(2 * lb + 1):
                        M[(oa + x) * NSLOT + (ob + y), c0: c0 + 2 * l + 1] = T[x, y]
    assert all(used[(ir.l, ir.p)] == mul for mul, ir in block_irreps), "coefficient bookkeeping mismatch"
    return M


def make_gate(n0, n1, n2):
    return Gate(o3.Irreps(f"{n0}x0e"), [torch.nn.functional.silu],
                o3.Irreps(f"{n1 + n2}x0e"), [torch.sigmoid],
                o3.Irreps(f"{n1}x1o+{n2}x2e"))


def mlp(n_in, n_hidden, n_out):
    return nn.Sequential(nn.Linear(n_in, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden), nn.SiLU(),
                         nn.Linear(n_hidden, n_out))


class Interaction(nn.Module):
    def __init__(self, irreps_in, irreps_sh, cfg: ModelConfig):
        super().__init__()
        self.gate = make_gate(cfg.n0, cfg.n1, cfg.n2)
        self.lin_pre = o3.Linear(irreps_in, irreps_in)
        self.lin_self = o3.Linear(irreps_in, self.gate.irreps_in)
        self.tp = o3.FullyConnectedTensorProduct(irreps_in, irreps_sh, self.gate.irreps_in,
                                                 shared_weights=False, internal_weights=False)
        self.radial = mlp(cfg.n_rbf, cfg.radial_hidden, self.tp.weight_numel)
        self.inv_sqrt_deg = 1.0 / math.sqrt(cfg.avg_neighbors)

    def forward(self, h, src, dst, sh, rbf):
        msg = self.tp(self.lin_pre(h)[src], sh, self.radial(rbf))
        agg = torch.zeros(h.shape[0], msg.shape[1], dtype=msg.dtype, device=msg.device).index_add(0, dst, msg)
        return self.gate(self.lin_self(h) + agg * self.inv_sqrt_deg)


class HamiltonianNet(nn.Module):
    def __init__(self, cfg: ModelConfig = ModelConfig(), Qs=None, target_scale: float = 1.0):
        super().__init__()
        self.cfg = cfg
        Qs = Qs if Qs is not None else [np.eye(2 * l + 1) for l in range(3)]
        self.hidden = o3.Irreps(f"{cfg.n0}x0e+{cfg.n1}x1o+{cfg.n2}x2e")
        self.sh_head = o3.Irreps.spherical_harmonics(cfg.lmax_head)
        self.sh_fn = o3.SphericalHarmonics(self.sh_head, normalize=True, normalization="component")  # built ONCE
        self.sh_mp_dim = o3.Irreps.spherical_harmonics(2).dim
        self.block_irreps = build_block_irreps()

        lookup = torch.full((10,), -1, dtype=torch.long)
        lookup[list(SPECIES)] = torch.arange(len(SPECIES))
        self.register_buffer("species_lookup", lookup)
        self.register_buffer("assembly", torch.tensor(build_assembly_matrix(self.block_irreps, Qs), dtype=torch.float32))
        self.register_buffer("target_scale", torch.tensor(float(target_scale)))

        self.embed = nn.Embedding(len(SPECIES), cfg.n0)
        sh_mp = o3.Irreps.spherical_harmonics(2)
        irreps_in = [o3.Irreps(f"{cfg.n0}x0e")] + [self.hidden] * (cfg.n_layers - 1)
        self.layers = nn.ModuleList([Interaction(i, sh_mp, cfg) for i in irreps_in])

        self.lin_i = o3.Linear(self.hidden, self.hidden)
        self.lin_j = o3.Linear(self.hidden, self.hidden)
        self.pair_tp = o3.FullyConnectedTensorProduct(self.hidden, self.sh_head, self.block_irreps,
                                                      shared_weights=False, internal_weights=False)
        self.pair_radial = mlp(cfg.n_rbf + 2 * cfg.n0, 2 * cfg.radial_hidden, self.pair_tp.weight_numel)
        self.onsite_tp = o3.FullyConnectedTensorProduct(self.hidden, self.hidden, self.block_irreps)
        self.onsite_lin = o3.Linear(self.hidden, self.block_irreps)

    def _blocks(self, coeff):
        return (coeff @ self.assembly.T).view(-1, NSLOT, NSLOT)

    def forward(self, z, pos, src, dst):
        """src/dst: ALL ordered pairs i!=j inside each molecule. Returns (onsite (N,14,14), pair (E,14,14)), normalised units."""
        cfg = self.cfg
        vec = pos[dst] - pos[src]
        dist = vec.norm(dim=1)
        sh = self.sh_fn(vec)

        sel = dist < cfg.r_cut
        s_mp, d_mp, sh_mp, dist_mp = src[sel], dst[sel], sh[sel][:, : self.sh_mp_dim], dist[sel]
        rbf_mp = soft_one_hot_linspace(dist_mp, 0.0, cfg.r_cut, cfg.n_rbf, basis="gaussian", cutoff=False)
        rbf_mp = rbf_mp * math.sqrt(cfg.n_rbf) * (0.5 * (torch.cos(math.pi * dist_mp / cfg.r_cut) + 1.0))[:, None]

        h = self.embed(self.species_lookup[z])
        for layer in self.layers:
            h = layer(h, s_mp, d_mp, sh_mp, rbf_mp)

        rbf_h = soft_one_hot_linspace(dist, 0.0, cfg.r_head, cfg.n_rbf, basis="gaussian", cutoff=False) * math.sqrt(cfg.n_rbf)
        u = self.lin_i(h)[src] + self.lin_j(h)[dst]
        w = self.pair_radial(torch.cat([rbf_h, h[src][:, : cfg.n0], h[dst][:, : cfg.n0]], dim=1))
        pair = self._blocks(self.pair_tp(u, sh, w))

        onsite = self._blocks(self.onsite_tp(h, h) + self.onsite_lin(h))
        onsite = 0.5 * (onsite + onsite.transpose(1, 2))
        return onsite, pair

    @torch.no_grad()
    def predict_hamiltonian(self, z_np, pos_np, device=None):
        """Single molecule -> symmetric AO Hamiltonian in Hartree (float64 numpy)."""
        device = device or next(self.parameters()).device
        n = len(z_np)
        s, d = np.where(~np.eye(n, dtype=bool))
        z = torch.as_tensor(z_np, dtype=torch.long, device=device)
        pos = torch.as_tensor(pos_np, dtype=torch.float32, device=device)
        s, d = torch.as_tensor(s, device=device), torch.as_tensor(d, device=device)
        on, pr = self(z, pos, s, d)
        T = torch.zeros(n, n, NSLOT, NSLOT, device=device)
        ar = torch.arange(n, device=device)
        T[ar, ar] = on
        T[s, d] = pr
        atoms, slots = orbital_maps(z_np)
        atoms, slots = torch.as_tensor(atoms, device=device), torch.as_tensor(slots, device=device)
        H = T[atoms[:, None], atoms[None, :], slots[:, None], slots[None, :]]
        H = 0.5 * (H + H.T) * self.target_scale
        return H.double().cpu().numpy()

    def config_dict(self):
        return asdict(self.cfg)
