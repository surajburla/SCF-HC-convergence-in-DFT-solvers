"""numpy-only tests (no torch/e3nn/pyscf). Run: python tests/test_pure.py"""
import os, sqlite3, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qh9scf.basis import N_AO, NSLOT, SLOTS, from_superblocks, n_basis, orbital_maps, to_superblocks
from qh9scf.data_prep import FlatQH9, from_sqlite, parse_row
from qh9scf.postprocess import density_from_hamiltonian, idempotency_error, s_norm, trace_error

rng = np.random.default_rng(0)
CH4 = np.array([6, 1, 1, 1, 1])

def rand_spd(n):
    A = rng.normal(size=(n, n)); return A @ A.T / n + np.eye(n)

def test_layout():
    assert N_AO[1] == 5 and N_AO[6] == N_AO[9] == 14 and NSLOT == 14
    assert n_basis(CH4) == 34
    assert list(SLOTS[1]) == [0, 1, 3, 4, 5]

def test_superblock_roundtrip():
    for Z in (CH4, np.array([8, 1, 1]), np.array([6, 6, 7, 8, 9, 1, 1, 1])):
        n = n_basis(Z); M = rng.normal(size=(n, n))
        T, mask = to_superblocks(M, Z)
        assert mask.sum() == n * n
        assert np.array_equal(from_superblocks(T, Z), M)
        assert T[~mask].size == 0 or np.all(T[~mask] == 0)
    atoms, slots = orbital_maps(CH4)
    assert atoms.tolist()[:14] == [0] * 14 and atoms[14] == 1 and slots[14] == 0

def test_density_construction():
    n = 20; S, H = rand_spd(n), rng.normal(size=(n, n)); H = H + H.T
    P, eps, gap = density_from_hamiltonian(H, S, 10)
    assert np.allclose(P, P.T, atol=1e-12)
    assert trace_error(P, S, 10) < 1e-10
    assert idempotency_error(P, S) < 1e-9
    U, _ = np.linalg.qr(rng.normal(size=(n, n)))
    P2, _, _ = density_from_hamiltonian(U.T @ H @ U, U.T @ S @ U, 10)
    assert np.allclose(P2, U.T @ P @ U, atol=1e-9)

def test_old_trace_layer_is_insufficient():
    n = 12; S = rand_spd(n); Pr = rng.normal(size=(n, n)); Pr = (Pr + Pr.T) / 2
    P = Pr + (10 - np.sum(Pr * S)) / np.sum(S * S) * S
    assert trace_error(P, S, 10) < 1e-10
    assert idempotency_error(P, S) > 1.0

def test_s_norm_is_basis_invariant():
    n = 10; S = rand_spd(n); D = rng.normal(size=(n, n)); D = D + D.T
    X = rng.normal(size=(n, n)) + 4 * np.eye(n)
    Xi = np.linalg.inv(X)
    S2, D2 = X.T @ S @ X, Xi @ D @ Xi.T
    assert abs(s_norm(D, S) - s_norm(D2, S2)) / s_norm(D, S) < 1e-8
    assert abs(np.linalg.norm(D) - np.linalg.norm(D2)) / np.linalg.norm(D) > 1e-2

def test_scalar_only_network_is_blind():
    F = 8; emb = {6: rng.normal(size=F), 1: rng.normal(size=F)}; W = rng.normal(size=(F, F))
    x = np.stack([emb[z] for z in CH4])
    h = x + np.stack([sum(x[j] @ W for j in range(5) if j != i) for i in range(5)])
    assert np.allclose(h[1], h[2])
    assert np.allclose(np.r_[h[1], h[1]], np.r_[h[1], h[2]])

def test_flat_pipeline_from_sqlite():
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "t.db"); c = sqlite3.connect(db)
        c.execute("CREATE TABLE data (id INTEGER, N INTEGER, Z BLOB, pos BLOB, Ham BLOB)")
        mols = []
        for k, Z in enumerate([CH4, np.array([8, 1, 1]), np.array([6, 6, 1, 1, 1, 1, 1, 8])]):
            n = n_basis(Z); H = rng.normal(size=(n, n)); H = H + H.T; pos = rng.normal(size=(len(Z), 3))
            mols.append((Z, pos, H))
            c.execute("INSERT INTO data VALUES (?,?,?,?,?)",
                      (k + 10, len(Z), Z.astype(np.int32).tobytes(), pos.tobytes(), H.tobytes()))
        c.commit(); c.close()
        out = os.path.join(d, "flat"); from_sqlite(db, out, "float64")
        fl = FlatQH9(out); assert len(fl) == 3
        for i, (Z, pos, H) in enumerate(mols):
            z2, p2, H2 = fl.raw(i)
            assert np.array_equal(z2, Z) and np.allclose(p2, pos) and np.array_equal(H2, H)
        try:
            parse_row(5, CH4.astype(np.int32).tobytes(), np.zeros((5, 3)).tobytes(), np.zeros(9 * 9).tobytes())
            raise AssertionError("old 9x9 Hamiltonian size must be rejected")
        except ValueError:
            pass

if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
