"""Bisect the segfault: runs the model stage by stage, printing before each stage.
The LAST line printed before the crash names the faulty stage. Usage:
    python scripts/debug_forward.py cpu
    python scripts/debug_forward.py cuda
"""
import faulthandler, math, sys
faulthandler.enable()
import _path  # noqa: F401
import numpy as np
import torch
import e3nn
from e3nn.math import soft_one_hot_linspace
from qh9scf.model import HamiltonianNet, ModelConfig

say = lambda m: print(m, flush=True)
dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
say(f"torch {torch.__version__} | e3nn {e3nn.__version__} | device {dev} | threads {torch.get_num_threads()}")
Z = np.array([6, 8, 7, 1, 1, 1, 1, 9]); pos = np.random.default_rng(0).normal(size=(8, 3)) * 1.3

say("1  build model"); net = HamiltonianNet(ModelConfig(), None, 1.0).to(dev).eval(); cfg = net.cfg; say("   ok")
n = len(Z); s, d = np.where(~np.eye(n, dtype=bool))
z = torch.as_tensor(Z, dtype=torch.long, device=dev); p = torch.as_tensor(pos, dtype=torch.float32, device=dev)
s, d = torch.as_tensor(s, device=dev), torch.as_tensor(d, device=dev)
vec = p[d] - p[s]; dist = vec.norm(dim=1)
with torch.no_grad():
    say("2  spherical harmonics"); sh = net.sh_fn(vec); say(f"   ok {tuple(sh.shape)}")
    sel = dist < cfg.r_cut
    s_mp, d_mp, sh_mp, dist_mp = s[sel], d[sel], sh[sel][:, : net.sh_mp_dim], dist[sel]
    say("3  radial basis")
    rbf = soft_one_hot_linspace(dist_mp, 0.0, cfg.r_cut, cfg.n_rbf, basis="gaussian", cutoff=False) * math.sqrt(cfg.n_rbf)
    say("4  embedding"); h = net.embed(net.species_lookup[z]); say("   ok")
    for k, layer in enumerate(net.layers):
        say(f"5.{k} interaction layer {k}"); h = layer(h, s_mp, d_mp, sh_mp, rbf); say(f"   ok {tuple(h.shape)}")
    say("6  pair head: radial MLP")
    rbf_h = soft_one_hot_linspace(dist, 0.0, cfg.r_head, cfg.n_rbf, basis="gaussian", cutoff=False) * math.sqrt(cfg.n_rbf)
    u = net.lin_i(h)[s] + net.lin_j(h)[d]
    w = net.pair_radial(torch.cat([rbf_h, h[s][:, : cfg.n0], h[d][:, : cfg.n0]], dim=1)); say(f"   ok weights {tuple(w.shape)}")
    say("7  pair head: tensor product"); c = net.pair_tp(u, sh, w); say(f"   ok {tuple(c.shape)}")
    say("8  on-site head"); c2 = net.onsite_tp(h, h) + net.onsite_lin(h); say(f"   ok {tuple(c2.shape)}")
    say("9  full forward (no grad)"); on, pr = net(z, p, s, d); say(f"   ok {tuple(on.shape)} {tuple(pr.shape)}")
say("10 full forward + backward"); net.train()
on, pr = net(z, p, s, d); (on.abs().mean() + pr.abs().mean()).backward(); say("   ok")
say("ALL OK")
