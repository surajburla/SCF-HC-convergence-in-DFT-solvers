"""QH9 SQLite (or legacy HDF5) -> flat memory-mappable arrays. numpy-only reader.

Layout in <out>/ :
  meta.json   {n_mol, dtype}
  atom_ptr.npy (M+1,) int64     Z.npy (sumN,) int32     pos.npy (sumN,3) float64
  ham_ptr.npy  (M+1,) int64     ham.bin  raw dtype, concatenated n*n blocks
  db_id.npy    (M,)
No per-molecule HDF5 groups, no file locks, safe for many DataLoader workers.
Copy the output to the WSL ext4 filesystem (~/), NOT /mnt/d: 9p mounts are several times slower.
"""
from __future__ import annotations

import argparse
import json
import json
import os
import sqlite3

import numpy as np

from .basis import SPECIES, n_basis


def _tqdm(it, **kw):
    try:
        from tqdm import tqdm
        return tqdm(it, **kw)
    except ImportError:      # progress bar is optional
        return it


def parse_row(n_atoms, z_blob, pos_blob, ham_blob):
    z = np.frombuffer(z_blob, dtype=np.int32)
    if len(z) != n_atoms:
        raise ValueError(f"N={n_atoms} but Z has {len(z)} entries")
    if not set(z.tolist()) <= set(SPECIES):
        raise ValueError(f"unsupported elements {sorted(set(z.tolist()) - set(SPECIES))}")
    if len(pos_blob) == n_atoms * 3 * 8:
        pos = np.frombuffer(pos_blob, dtype=np.float64).reshape(n_atoms, 3)
    elif len(pos_blob) == n_atoms * 3 * 4:
        pos = np.frombuffer(pos_blob, dtype=np.float32).reshape(n_atoms, 3).astype(np.float64)
    else:
        raise ValueError("pos blob size matches neither float32 nor float64")
    nb = n_basis(z)                                   # size decided by physics, not by guessing dtype
    if len(ham_blob) == nb * nb * 8:
        ham = np.frombuffer(ham_blob, dtype=np.float64).reshape(nb, nb)
    elif len(ham_blob) == nb * nb * 4:
        ham = np.frombuffer(ham_blob, dtype=np.float32).reshape(nb, nb)
    else:
        raise ValueError(f"Ham blob {len(ham_blob)} B != def2-SVP size {nb}^2 in float64/float32")
    return z, pos, ham


class FlatWriter:
    def __init__(self, out_dir, dtype="float32"):
        os.makedirs(out_dir, exist_ok=True)
        self.dir, self.dtype = out_dir, np.dtype(dtype)
        self.f = open(os.path.join(out_dir, "ham.bin"), "wb")
        self.Z, self.pos, self.ids = [], [], []
        self.atom_ptr, self.ham_ptr = [0], [0]

    def add(self, db_id, z, pos, ham):
        n = len(z)
        if ham.shape[0] != n_basis(z) or ham.shape[0] != ham.shape[1]:
            raise ValueError(f"id {db_id}: Ham {ham.shape} vs expected {n_basis(z)}")
        self.f.write(np.ascontiguousarray(ham, dtype=self.dtype).tobytes())
        self.Z.append(np.asarray(z, dtype=np.int32))
        self.pos.append(np.asarray(pos, dtype=np.float64))
        self.ids.append(int(db_id))
        self.atom_ptr.append(self.atom_ptr[-1] + n)
        self.ham_ptr.append(self.ham_ptr[-1] + ham.size)

    def close(self):
        self.f.close()
        np.save(os.path.join(self.dir, "Z.npy"), np.concatenate(self.Z))
        np.save(os.path.join(self.dir, "pos.npy"), np.concatenate(self.pos))
        np.save(os.path.join(self.dir, "atom_ptr.npy"), np.array(self.atom_ptr, dtype=np.int64))
        np.save(os.path.join(self.dir, "ham_ptr.npy"), np.array(self.ham_ptr, dtype=np.int64))
        np.save(os.path.join(self.dir, "db_id.npy"), np.array(self.ids, dtype=np.int64))
        with open(os.path.join(self.dir, "meta.json"), "w") as fh:
            json.dump({"n_mol": len(self.ids), "dtype": self.dtype.name}, fh)



def _find_table(cur, wanted=None):
    names = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    if wanted:
        for n in names:
            if n.lower() == wanted.lower():
                return n
    for n in names:
        cols = {c[1].lower() for c in cur.execute(f'PRAGMA table_info("{n}")')}
        if {"id", "n", "z", "pos", "ham"} <= cols:
            return n
    raise RuntimeError(f"No QH9 table found. Tables in file: {names}")


def _find_table(cur, wanted=None):
    names = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    if wanted:
        for n in names:
            if n.lower() == wanted.lower():
                return n
    for n in names:
        cols = {c[1].lower() for c in cur.execute(f'PRAGMA table_info("{n}")')}
        if {"id", "n", "z", "pos", "ham"} <= cols:
            return n
    raise RuntimeError(f"No QH9 table found. Tables in file: {names}")


def _find_table(cur, wanted=None):
    names = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    if wanted:
        for n in names:
            if n.lower() == wanted.lower():
                return n
    for n in names:
        cols = {c[1].lower() for c in cur.execute(f'PRAGMA table_info("{n}")')}
        if {"id", "n", "z", "pos", "ham"} <= cols:
            return n
    raise RuntimeError(f"No QH9 table found. Tables in file: {names}")

def from_sqlite(db_path, out_dir, dtype="float32", limit=None, table="DATA"):
    if not os.path.exists(db_path):
        raise FileNotFoundError(db_path)
    cur = sqlite3.connect(db_path).cursor()
    table = _find_table(cur, table)
    print("using table:", table)
    total = cur.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    q = f'SELECT id, N, Z, pos, Ham FROM "{table}" ORDER BY id' + (f" LIMIT {int(limit)}" if limit else "")
    w = FlatWriter(out_dir, dtype)
    for rid, n, zb, pb, hb in _tqdm(cur.execute(q), total=min(total, limit or total)):
        z, pos, ham = parse_row(n, zb, pb, hb)
        w.add(rid, z, pos, ham)
    w.close()


def from_legacy_h5(h5_path, out_dir, dtype="float32", limit=None):
    import h5py
    w = FlatWriter(out_dir, dtype)
    with h5py.File(h5_path, "r") as f:
        keys = sorted(f["Z"].keys(), key=int)[: limit or None]
        for k in _tqdm(keys):
            w.add(int(k), f["Z"][k][:], f["pos"][k][:], f["Ham"][k][:].astype(np.float64))
    w.close()


class FlatQH9:
    def __init__(self, root):
        self.root = root
        meta = json.load(open(os.path.join(root, "meta.json")))
        self.n_mol, self.dtype = meta["n_mol"], np.dtype(meta["dtype"])
        self.Z = np.load(os.path.join(root, "Z.npy"))
        self.pos = np.load(os.path.join(root, "pos.npy"))
        self.atom_ptr = np.load(os.path.join(root, "atom_ptr.npy"))
        self.ham_ptr = np.load(os.path.join(root, "ham_ptr.npy"))
        self.db_id = np.load(os.path.join(root, "db_id.npy"))
        self._ham = None                                # opened lazily, once per process

    def __len__(self):
        return self.n_mol

    def n_atoms(self, i):
        return int(self.atom_ptr[i + 1] - self.atom_ptr[i])

    def raw(self, i):
        a0, a1 = self.atom_ptr[i], self.atom_ptr[i + 1]
        if self._ham is None:
            self._ham = np.memmap(os.path.join(self.root, "ham.bin"), dtype=self.dtype, mode="r")
        h = np.array(self._ham[self.ham_ptr[i]: self.ham_ptr[i + 1]])
        n = int(round(np.sqrt(h.size)))
        return self.Z[a0:a1].astype(np.int64), self.pos[a0:a1], h.reshape(n, n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--db")
    g.add_argument("--legacy-h5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--limit", type=int, default=None, help="first N molecules (recommended: 10000-20000)")
    a = ap.parse_args()
    (from_sqlite(a.db, a.out, a.dtype, a.limit) if a.db else from_legacy_h5(a.legacy_h5, a.out, a.dtype, a.limit))
    print("done ->", a.out)
