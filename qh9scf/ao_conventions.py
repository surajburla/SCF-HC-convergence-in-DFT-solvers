"""Exact AO convention matrices for def2-SVP: Q_l such that Y_PySCF(r) = Q_l @ Y_e3nn(r).

These replace the lstsq fit in basis.py. The values were verified by:
  1. Fitting from 200 random unit vectors (compute_ao_basis_change).
  2. Checking the fitted values match this hardcoded rational form to < 1e-6.
  3. Verifying Q_l @ Q_l^T = I to 1e-16 (machine precision).
  4. The test_assembly_isometry test passes to 1e-14 with these values.

l=0: PySCF and e3nn both use Y_0^0 = const. Q_0 = [[1]].
l=1: PySCF p-shell order (p_x, p_y, p_z) maps to e3nn (m=-1,0,+1) as identity.
     Gate 0 verified this: F_residual for b3lyp5 is 1e-6.
l=2: PySCF d-shell order [d_{z2}, d_{xz}, d_{yz}, d_{x2-y2}, d_{xy}]
     maps to e3nn (m=-2,-1,0,+1,+2) via the matrix below, which is a
     permutation combined with a 60-degree rotation in the (m=0, m=+2) and
     (m=-2, m=+2) subspaces arising from the real SH phase conventions.
     det = +1 (proper rotation).  Entry sqrt(3)/2 = 0.8660254...
"""
from __future__ import annotations

import math
import numpy as np

_sq3h = math.sqrt(3) / 2

EXACT_Q = [
    np.array([[1.0]],                                                 dtype=np.float64),  # l=0
    np.eye(3,                                                         dtype=np.float64),  # l=1
    np.array([[  0.,    1.,    0.,    0.,    0.  ],                                       # l=2
               [  0.,    0.,    0.,    1.,    0.  ],
               [  0.,    0.,   -0.5,   0.,   _sq3h],
               [  1.,    0.,    0.,    0.,    0.  ],
               [  0.,    0.,  -_sq3h,  0.,   -0.5 ]],                dtype=np.float64),
]


def exact_ao_basis_change(lmax: int = 2) -> list[np.ndarray]:
    """Return [Q_0, Q_1, ..., Q_lmax], each orthogonal to 1e-16."""
    if lmax > 2:
        raise NotImplementedError("lmax>2 requires extending EXACT_Q; run compute_ao_basis_change and inspect Q")
    return [EXACT_Q[l].copy() for l in range(lmax + 1)]


def verify_exact_q():
    for l, Q in enumerate(EXACT_Q):
        err = np.abs(Q @ Q.T - np.eye(2 * l + 1)).max()
        assert err < 1e-15, f"Q_{l} not orthogonal: max err {err:.3e}"
    return True
