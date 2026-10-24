# ADR-001: Predict the Hamiltonian, derive the density, validate against real PySCF

**Status:** Proposed  **Deciders:** project team + reviewer

## Context
Goal: shorten the SCF of B3LYP/def2-SVP calculations on QH9 molecules (<=9 heavy atoms; H,C,N,O,F) with a learned initial guess.
QH9 stores Hamiltonians, not density matrices (verified: QH9, NeurIPS 2023 D&B). A density matrix follows from `H C = S C eps`.
Published ML guesses for B3LYP are hard-won: reported iteration reductions are modest and can turn negative out of distribution.

## Decision
1. Target = Hamiltonian `H` in PySCF AO order. 2. Equivariant network emits irreps coefficients per atom pair; a fixed
Clebsch-Gordan matrix assembles 14x14 superblocks. 3. `P = 2 C_occ C_occ^T` from `eigh(H, S)` in float64.
4. Evaluate only with real PySCF `dm0=`, against the PySCF default guess (MINAO), including inference and eigh time.

## Options
| | A. Direct P + trace projection (old) | **B. H prediction + eigh (chosen)** | C. P prediction + McWeeny | D. Better classical guess only |
|---|---|---|---|---|
| Target exists in QH9 | No (derived) | **Yes** | No | n/a |
| Symmetry of P | by averaging | **exact** | by averaging | exact |
| Tr(PS)=N_e | exact after shift | **exact** | not preserved by purification | exact |
| Idempotency PSP=2P | **never** | **exact** | iterative, convergence not guaranteed | exact |
| Complexity | low | medium | high | low |
| Main risk | invalid density, SCF may slow | error amplified by 1/gap; AO order | tuning | no ML contribution |

Trade-off: B pays one O(n^3) eigh (n~170 -> milliseconds, estimate) and accepts gap-dependent error amplification
(first-order perturbation theory: `|dP| ~ |dH|/gap`). In exchange, every output is a physically valid closed-shell density.
D stays in the study as the comparison baseline (minao, atom, huckel, vsap, 1e), not as an alternative.

## What the old code got wrong (verified)
| # | Finding | Evidence | Severity |
|---|---|---|---|
| 1 | Orbital counts H=1, heavy=5 (STO-3G-like). def2-SVP is H=5, C/N/O/F=14. CH4: 9 vs 34. Head can write 7% of the matrix | numeric check; `model_and_dataset.py` itself uses 34 | critical |
| 2 | `FullyConnectedTensorProduct(0e x SH -> 0e)` keeps only the l=0 harmonic, a constant. Message passing ignores direction and distance | selection rule; no radial term | critical |
| 3 | Scalar-only features + fully connected graph: all same-species atoms get identical features; `pair(H_a,H_a)==pair(H_a,H_b)` | numpy test `test_scalar_only_network_is_blind` | critical |
| 4 | Output blocks come from invariants, so they cannot rotate with the molecule. "SO(3) covariant" is false | follows from #2 | critical |
| 5 | SCF "benchmark" is `ceil(log(thr/err)/log(0.82))`: a function of one norm. Zero guess is a straw man; PySCF default is MINAO | numeric check | critical |
| 6 | Reference solver `base_pySCF2.py` runs RHF; QH9 is B3LYP. Different fixed point | code | critical |
| 7 | Silent fallbacks: missing `P` -> Hamiltonian used as P; missing `S` -> identity; missing `N_e` | `__getitem__` | high |
| 8 | Trace layer fixes one scalar; `PSP=2P` error 79.5 in the test | `test_old_trace_layer_is_insufficient` | high |
| 9 | Train and evaluate on the same molecule; "memorization proves acceleration" is circular | `run_pipeline` | high |
| 10 | `generate_density_matrices.py` opens the HDF5 file `r+` and `r` in one process | h5py file locking | high |
| 11 | H cast to float32 before eigh; 400k tiny HDF5 groups; reading from `/mnt/d` (9p) per sample | code | medium |
| 12 | Audit suggestion "load all tensors into RAM" is infeasible: ~15 GB per float32 matrix type (estimate: mean n_basis ~170) | arithmetic | medium |
| 13 | Nested Python loops over atom pairs; no batching; no `.to(device)`; GPU unused | code | medium |
| 14 | Frobenius norm of dP depends on AO basis; use `||S^1/2 dP S^1/2||` | `test_s_norm_is_basis_invariant` | medium |
| 15 | `preprocess_qh9.py` v2 uses `torch_geometric`/`torch_cluster` and invented columns; QH9 has no density column | code | medium |

## What was good (keep)
Warm-start framing and QH9 choice (true Hamiltonian labels). Generalized `eigh(H,S)` for `P`, with trace check and `S`
recomputed by PySCF with a shape guard. Exact algebra of the trace shift. Symmetrization. Removal of bilinear `interpolate`
(right direction: block structure). Pure-PyTorch graph build; no PyG. WSL path translation. Resumable generation idea.
Callback iteration counter. The context-file audit of the SCF mock was correct.

## Consequences
Easier: one validated target, physical outputs, honest benchmark. Harder: needs an AO-convention fit and equivariance tests
before training; B3LYP guesses are hard to accelerate. Revisit: Delta-learning on MINAO and solver-aligned training
(backpropagation through SCF; Eberhard et al., 2026 preprint) if v1 gives RIC >= 1 out of distribution.

## Action items
1. [ ] Gate 0 passes (`validate_dataset.py`).  2. [ ] Gate 1 passes (`test_equivariance.py`).  3. [ ] Overfit gate passes.
4. [ ] Train on 5-20k subset, both `--split id` and `--split size`.  5. [ ] Report RIC, cycle reduction, wall speedup, `dE`
vs MINAO and oracle.  6. [ ] Ablations: Delta-learning from H_core; larger `r_head`; with/without l=2 hidden features.

---
# Claim-wording fixes (remove review traps)
| Old wording | Problem | Replace with |
|---|---|---|
| "bypass 30-60% of SCF iterations" | Unsupported target; B3LYP literature reports far less | "We measure the relative iteration count (RIC) against PySCF's default MINAO guess on held-out molecules." |
| "SCF takes 20-50+ cycles" | Unmeasured | "Baseline cycles: median X (measured, N molecules)." |
| "QH9 provides Hamiltonian and density matrices" | False; Hamiltonians only | "QH9 provides B3LYP/def2-SVP Hamiltonians; we derive densities by generalized diagonalization." |
| "130,831 molecules ... varying geometric strain" | Stable subset = equilibrium geometries | "130,831 stable geometries (<=9 heavy atoms). Strained geometries appear only in QH9-dynamic." |
| "strictly obeys SO(3) covariance" | False for old model | "Equivariant by construction; verified numerically (rotation, inversion, translation, permutation) to 1e-4." |
| "hard physical constraint layer (trace projection)" | Fixes one scalar | "Physics-consistent reconstruction: P from the occupied subspace gives Tr(PS)=N_e, PSP=2P, P=P^T exactly." |
| "skip the first 90% of the calculation" / "violently accelerate" | Contradicts objective; unmeasurable | Delete. |
| "ML replaces the heuristic guess" | Solver still converges | "ML supplies the initial guess; the PySCF solver performs convergence." |
| "instantly predicts" | Unmeasured | "Inference + diagonalization take X ms (measured)." |
| "sparse N x N density matrix" | Dense for <=29 atoms | "block-structured N_basis x N_basis matrix." |
| Applications: iridium catalysts, Li-ion diffusion, protein-ligand | Outside data scope (organic, <=9 heavy atoms, no metals, no periodic cells) | "Scope: small organic molecules. Larger systems and materials are future work." |
| "PyTorch Geometric DataLoader" (slide 11) | Removed from code | "Pure PyTorch dataset with e3nn." |
| "JIT builds ASTs for Clebsch-Gordan ops" / "cyclic recompilation" | Mechanism unverified | "e3nn's default compilation requires TorchScript to stay enabled; spherical harmonics are called functionally." |
| "-33.8% acceleration", "100-epoch memorization proves..." | Artifacts of the mock and train=test | Delete from the report. Describe as a discarded debugging scaffold. |
| "Graph2Mat architecture" | Not what the code does | "Block-wise matrix prediction in the spirit of PhiSNet/QHNet/Graph2Mat." |
