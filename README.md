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