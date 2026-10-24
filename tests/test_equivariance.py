"""Needs torch + e3nn (+ pyscf for the AO-convention test). Run: python tests/test_equivariance.py

Three independent checks, so a failure localises the cause:
  A  model equivariance  : in the e3nn basis (Q=I), H(R pos) == D H(pos) D^T   (and inversion/translation)
  B  AO convention       : PySCF overlap of the rotated molecule == D_py S D_py^T, with Q fitted from PySCF
  C  assembly isometry   : M^T M = I   (coefficients <-> blocks is loss-free)
"""
import faulthandler, os, sys
faulthandler.enable()
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import torch, scipy.linalg
    from e3nn import o3
except ImportError as e:  # pragma: no cover
    print("SKIP (missing dependency):", e); sys.exit(0)
from qh9scf.basis import SHELLS, compute_ao_basis_change, identity_ao_basis_change
from qh9scf.model import HamiltonianNet, ModelConfig, build_assembly_matrix, build_block_irreps

rng = np.random.default_rng(1)
Z = np.array([6, 8, 7, 1, 1, 1, 1, 9])
POS = rng.normal(size=(len(Z), 3)) * 1.3


def rand_rot():
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:                       # force a proper rotation
        q[:, 0] *= -1
    return q


def D_ao(Z, R, Qs):
    blocks = []
    for z in Z:
        for l in SHELLS[int(z)]:
            D = o3.Irrep(l, (-1) ** l).D_from_matrix(torch.tensor(R)).numpy()
            blocks.append(Qs[l] @ D @ Qs[l].T)
    return scipy.linalg.block_diag(*blocks)


def parity_ao(Z):
    return np.diag(np.concatenate([np.full(2 * l + 1, (-1.0) ** l) for z in Z for l in SHELLS[int(z)]]))


def test_sh_convention():
    R = rand_rot(); x = torch.tensor(rng.normal(size=(7, 3)))
    for l in range(3):
        Y = o3.spherical_harmonics(l, x, True, "component").numpy()
        Yr = o3.spherical_harmonics(l, x @ torch.tensor(R).T, True, "component").numpy()
        D = o3.Irrep(l, (-1) ** l).D_from_matrix(torch.tensor(R)).numpy()
        assert np.abs(Yr - Y @ D.T).max() < 1e-6, f"l={l}: D convention mismatch"


def test_assembly_isometry():
    M = build_assembly_matrix(build_block_irreps(), identity_ao_basis_change())
    err = np.abs(M.T @ M - np.eye(M.shape[1])).max()
    assert err < 1e-6, f"M^T M - I: max err {err:.3e}"


def test_model_equivariance():
    torch.manual_seed(0)
    Qs = identity_ao_basis_change()
    net = HamiltonianNet(ModelConfig(), Qs, 1.0).eval()
    H = net.predict_hamiltonian(Z, POS, "cpu")
    tol = 2e-4 * max(1.0, np.abs(H).max())
    R = rand_rot(); D = D_ao(Z, R, Qs)
    assert np.abs(net.predict_hamiltonian(Z, POS @ R.T, "cpu") - D @ H @ D.T).max() < tol, "rotation equivariance FAILED"
    Pi = parity_ao(Z)
    assert np.abs(net.predict_hamiltonian(Z, -POS, "cpu") - Pi @ H @ Pi).max() < tol, "inversion/parity FAILED"
    assert np.abs(net.predict_hamiltonian(Z, POS + 3.7, "cpu") - H).max() < tol, "translation invariance FAILED"
    perm = rng.permutation(len(Z))
    from qh9scf.basis import N_AO
    starts = np.cumsum([0] + [N_AO[int(z)] for z in Z])
    idx = np.concatenate([np.arange(starts[p], starts[p + 1]) for p in perm])
    Hp = net.predict_hamiltonian(Z[perm], POS[perm], "cpu")
    assert np.abs(Hp - H[np.ix_(idx, idx)]).max() < tol, "atom-permutation equivariance FAILED"
    assert np.abs(H - H.T).max() < 1e-6


def test_ao_convention_with_pyscf():
    try:
        from qh9scf.scf_tools import build_mol
    except ImportError:
        print("SKIP pyscf test"); return
    from qh9scf.basis import check_layout_against_pyscf
    check_layout_against_pyscf()
    Qs = compute_ao_basis_change()
    R = rand_rot(); D = D_ao(Z, R, Qs)
    S = build_mol(Z, POS).intor("int1e_ovlp"); Sr = build_mol(Z, POS @ R.T).intor("int1e_ovlp")
    err = np.abs(Sr - D @ S @ D.T).max()
    assert err < 1e-5, f"PySCF AO convention not captured by Q (max err {err:.2e})"


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
