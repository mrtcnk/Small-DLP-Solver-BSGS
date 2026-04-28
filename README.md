# Small-Range Discrete Logarithm Solver (BSGS) for secp256k1

This repository contains a cached **Baby-Step Giant-Step (BSGS)** implementation for recovering bounded discrete logarithms on the **secp256k1** elliptic curve.
The solver targets problems of the form P = mG where

- `G` is the secp256k1 generator
- `m` is an integer in a known range
- `P` is the public point

The algorithm assumes m ∈ [0, 2^bits_total)and splits the search as m = i + j·M where

- `M = 2^l1` (baby-step size)
- `J = 2^l2` (giant-step size)
- `l2 = bits_total − l1`

---

## Cached BSGS Design

The implementation stores **baby-step tables on disk** so they only need to be constructed once.

Example cache files:
- `bsgs_baby_secp256k1_l1_18.bin`
- `bsgs_baby_secp256k1_l1_19.bin`
- `bsgs_baby_secp256k1_l1_21.bin`

After the first construction, these tables are reused across runs.
Typical load time is 0.004 – 0.18 seconds depending on the size of the baby table.
The observed cache sizes confirm the expected exponential scaling:

---

## Complexity

BSGS splits the search space into baby steps ≈ 2^l1 and giant steps ≈ 2^l2. Therefore
- Memory ≈ O(2^l1)
- Runtime ≈ O(2^l2)

Increasing `l1` increases memory but decreases runtime exponentially.

---

# Compilation


cc -O3 -Wall -Wextra -o bsgs_dlp bsgs_dlp.c -lsecp256k1


---

## Running the Solver

Run a single configuration:


./bsgs_dlp <bits_total> <l1> <trials>

### Example:

./bsgs_dlp 44 23 5

---

# Automated Experiments

The script `run_all.sh` runs a sweep of parameters automatically.


./run_all.sh


It evaluates


bits_total = 40, ..., 44, l1 = 18, ..., 23, trials = 5 and reports average solve times.

---

## Experimental Results

Representative configurations extracted from the automated experiments:

| Range | l1 | l2 | Trials | Baby Entries | Giant Steps | Cache Size | Avg Solve Time |
|---|---|---|---|---|---|---|---|
| 2^40 | 18 | 22 | 5 | 262,143 | 4.19M | **22 MB** | ~9.50 s |
| 2^40 | 20 | 20 | 5 | 1,048,575 | 1.05M | **88 MB** | ~2.65 s |
| 2^40 | 22 | 18 | 5 | 4,194,303 | 0.26M | **352 MB** | ~0.59 s |
| 2^41 | 21 | 20 | 5 | 2,097,151 | 1.05M | **176 MB** | ~3.73 s |
| 2^42 | 21 | 21 | 5 | 2,097,151 | 2.09M | **176 MB** | ~4.56 s |
| 2^43 | 22 | 21 | 5 | 4,194,303 | 2.09M | **352 MB** | ~4.24 s |
| 2^44 | 23 | 21 | 5 | 8,388,607 | 2.09M | **704 MB** | ~8.36 s |
---

# Interpretation

The experiments confirm the expected BSGS tradeoff.

Increasing `l1`:

- doubles the baby-step table size
- halves the giant-step search space

Example for the range `2^40`:

| l1 | l2 | Avg Solve Time |
|----|----|---------------|
| 18 | 22 | ~9.50 s |
| 20 | 20 | ~2.65 s |
| 22 | 18 | ~0.59 s |
| 23 | 17 | ~0.52 s |

---

# Effect of Increasing Search Range

If the baby table size is fixed while the range grows, runtime increases due to larger giant-step searches.

Example for `l1 = 21`:

| Range | l2 | Avg Solve Time |
|------|----|---------------|
| 2^40 | 19 | ~1.36 s |
| 2^41 | 20 | ~3.73 s |
| 2^42 | 21 | ~4.56 s |
| 2^43 | 22 | ~13.52 s |
| 2^44 | 23 | ~38.18 s |

Runtime therefore scales approximately with 2^l2.

---

## Notes

The current implementation performs:

- compressed point serialization
- hash-table lookup

during each giant step. These operations introduce overhead compared to specialized ECDLP solvers that operate directly in affine coordinates.

Potential improvements include:

- eliminating point serialization
- x-coordinate hashing
- multi-target BSGS
- parallel giant-step search

## Parallel implementation

| Bits | l1 | 1T (s) | 2T (s) | 4T (s) | 8T (s) |
|------|----|--------|--------|--------|--------|
| 40 | 20 | 4.216 | 1.677 | 0.801 | 0.448 |
| 40 | 21 | 1.024 | 0.784 | 0.449 | 0.271 |
| 40 | 22 | 1.029 | 0.417 | 0.253 | 0.164 |
| 40 | 23 | 0.555 | 0.122 | 0.080 | 0.037 |
| 41 | 20 | 8.901 | 4.392 | 1.986 | 0.567 |
| 41 | 21 | 2.880 | 1.487 | 0.730 | 0.550 |
| 41 | 22 | 0.630 | 0.766 | 0.434 | 0.260 |
| 41 | 23 | 0.536 | 0.588 | 0.253 | 0.102 |
| 42 | 20 | 15.919 | 5.062 | 2.642 | 2.063 |
| 42 | 21 | 7.668 | 4.321 | 2.024 | 0.826 |
| 42 | 22 | 2.895 | 0.842 | 1.011 | 0.207 |
| 42 | 23 | 1.300 | 0.432 | 0.308 | 0.270 |
| 43 | 20 | 17.567 | 10.660 | 5.927 | 5.625 |
| 43 | 21 | 11.411 | 5.567 | 3.879 | 2.044 |
| 43 | 22 | 7.694 | 4.052 | 1.482 | 0.797 |
| 43 | 23 | 4.562 | 1.776 | 0.971 | 0.564 |
| 44 | 20 | 42.802 | 24.794 | 13.063 | 10.156 |
| 44 | 21 | 20.662 | 10.726 | 7.248 | 4.794 |
| 44 | 22 | 15.049 | 10.418 | 4.577 | 1.449 |
| 44 | 23 | 9.347 | 3.146 | 1.479 | 0.972 |