"""AO layout for def2-SVP and lossless conversion between AO matrices and per-atom-pair superblocks.

Facts used (verify with `python -m qh9scf.basis --check`, which asks PySCF):
  H          : [2s1p]   -> shells (0,0,1)        -> 5 AOs
  C, N, O, F : [3s2p1d] -> shells (0,0,0,1,1,2)  -> 14 AOs

Superblock = 14x14 slot grid with the heavy-atom layout. An H atom occupies slots
[0,1,3,4,5] of that grid; the other slots are masked out of every loss and output.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from .ao_conventions import exact_ao_basis_change

SPECIES = (1, 6, 7, 8, 9)
SHELLS = {1: (0, 0, 1), 6: (0, 0, 0, 1, 1, 2), 7: (0, 0, 0, 1, 1, 2),
          8: (0, 0, 0, 1, 1, 2), 9: (0, 0, 0, 1, 1, 2)}
SLOT_SHELLS = (0, 0, 0, 1, 1, 2)


def _shell_offsets(ls):
    offs, o = [], 0
    for l in ls:
        offs.append(o)
        o += 2 * l + 1
    return tuple(offs), o


SLOT_OFFS, NSLOT = _shell_offsets(SLOT_SHELLS)  # (0,1,2,3,6,9), 14


def _slots_for_atom(shells):
    used, slots = set(), []
    for l in shells:
        for k, lk in enumerate(SLOT_SHELLS):
            if lk == l and k not in used:
                used.add(k)
                slots.extend(range(SLOT_OFFS[k], SLOT_OFFS[k] + 2 * l + 1))
                break
        else:
            raise ValueError(f"cannot place shell l={l} into superblock layout")
    return np.array(slots, dtype=np.int64)


SLOTS = {z: _slots_for_atom(s) for z, s in SHELLS.items()}
N_AO = {z: len(v) for z, v in SLOTS.items()}


def n_basis(Z) -> int:
    return int(sum(N_AO[int(z)] for z in Z))


def orbital_maps(Z):
    """Return (atom_of_ao, slot_of_ao); both int64 arrays of length n_basis."""
    atoms = np.concatenate([np.full(N_AO[int(z)], a, dtype=np.int64) for a, z in enumerate(Z)])
    slots = np.concatenate([SLOTS[int(z)] for z in Z])
    return atoms, slots


def to_superblocks(M, Z):
    """AO matrix (n,n) -> (A,A,14,14) values and boolean mask. Exact, loss-free."""
    atoms, slots = orbital_maps(Z)
    if M.shape != (len(atoms), len(atoms)):
        raise ValueError(f"matrix {M.shape} does not match def2-SVP basis size {len(atoms)}")
    a = len(Z)
    T = np.zeros((a, a, NSLOT, NSLOT), dtype=M.dtype)
    mask = np.zeros((a, a, NSLOT, NSLOT), dtype=bool)
    idx = (atoms[:, None], atoms[None, :], slots[:, None], slots[None, :])
    T[idx] = M
    mask[idx] = True
    return T, mask


def from_superblocks(T, Z):
    """(A,A,14,14) -> AO matrix (n,n). Works for numpy arrays and torch tensors."""
    atoms, slots = orbital_maps(Z)
    if not isinstance(T, np.ndarray):  # torch path
        import torch
        atoms, slots = torch.as_tensor(atoms, device=T.device), torch.as_tensor(slots, device=T.device)
    return T[atoms[:, None], atoms[None, :], slots[:, None], slots[None, :]]


# ---------------------------------------------------------------- PySCF-dependent helpers
def derive_layout_from_pyscf(basis="def2-svp"):
    from pyscf import gto
    out = {}
    for z in SPECIES:
        mol = gto.M(atom=f"{z} 0 0 0", basis=basis, spin=z % 2, verbose=0)
        assert all(mol.bas_nctr(i) == 1 for i in range(mol.nbas)), "general contraction not supported"
        out[z] = tuple(int(mol.bas_angular(i)) for i in range(mol.nbas))
    return out


def check_layout_against_pyscf():
    got = derive_layout_from_pyscf()
    for z in SPECIES:
        assert got[z] == SHELLS[z], f"Z={z}: PySCF shells {got[z]} != hardcoded {SHELLS[z]}"
    return True


def compute_ao_basis_change(lmax=2, npts=200, seed=0):
    """Return Q_l matrices (exact for l<=2, fitted for l>2).

    For l<=2 the exact rational form is used (see ao_conventions.py), which
    is orthogonal to 1e-16.  For l>2 (lmax_head>4 blocks) a PySCF lstsq fit
    is used as a fallback.
    """
    if lmax <= 2:
        return exact_ao_basis_change(lmax)
    # fallback: lstsq for l>2 (requires pyscf + e3nn)
    import torch
    from e3nn import o3
    from pyscf import gto

    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(npts, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    Qs = []
    for l in range(lmax + 1):
        mol = gto.M(atom="H 0 0 0", basis={"H": [[l, [1.0, 1.0]]]}, spin=1, unit="Bohr", verbose=0)
        ao = mol.eval_gto("GTOval_sph", pts)          # (npts, 2l+1), radial factor constant on |r|=1
        Y = o3.spherical_harmonics(l, torch.tensor(pts), normalize=True, normalization="component").numpy()
        fit, *_ = np.linalg.lstsq(Y, ao, rcond=None)
        c = np.sqrt(np.trace(fit.T @ fit) / (2 * l + 1))
        Q = (fit / c).T
        assert np.abs(Q @ Q.T - np.eye(2 * l + 1)).max() < 1e-6, f"l={l}: Q not orthogonal"
        assert np.abs(ao - c * Y @ Q.T).max() / np.abs(ao).max() < 1e-6, f"l={l}: AO/SH fit failed"
        Qs.append(Q)
    return Qs


def save_ao_basis_change(path, Qs):
    np.savez(path, **{f"Q{l}": q for l, q in enumerate(Qs)})


def load_ao_basis_change(path):
    d = np.load(path)
    return [d[f"Q{l}"] for l in range(len(d.files))]


def identity_ao_basis_change(lmax=2):
    """Used when ao_convention.npz is absent. Returns exact Q for l<=2."""
    return exact_ao_basis_change(lmax)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare hardcoded layout with PySCF")
    ap.add_argument("--build-cache", action="store_true", help="write ao_convention.npz")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "ao_convention.npz"))
    a = ap.parse_args()
    if a.check:
        check_layout_against_pyscf()
        print("layout OK:", {z: (SHELLS[z], N_AO[z]) for z in SPECIES})
    if a.build_cache:
        Qs = compute_ao_basis_change()
        save_ao_basis_change(a.out, Qs)
        print("wrote", a.out)
        for l, q in enumerate(Qs):
            print(f"Q{l}=\n{np.round(q, 6)}")
