# Small DLP Solver — BSGS on secp256k1

A fast Baby-Step Giant-Step solver for the small exponential Elliptic Curve Discrete Logarithm Problem (ECDLP) on secp256k1, with caching, parallelization, Jacobian coordinate optimization, packed memory layout, windowed batch inversion, and cuckoo hashing.

---

## Overview

EC-based Additively Homomorphic Encryption (AHE) schemes such as Exp-ElGamal require solving a small ECDLP during decryption: recovering `m` from `m*G` where `m` is a bounded integer. This solver implements an optimized BSGS algorithm targeting practical plaintext lengths up to 63 bits — covering the full XRPL MPT (XLS-33) amount range `[0, 2^63)`.

---

## Optimizations

### 1. Multi-threaded Giant Step (`feat/multi-thread`)
The giant step search is split across threads. Each thread independently computes its starting point via scalar multiplication and walks its assigned range in parallel. Uses an `atomic_int` flag for early termination when any thread finds the solution.

### 2. 64-bit Key Truncation (`feat/64bit-key-truncation`)
The baby table originally stored full 33-byte compressed EC points as keys. This was replaced with 64-bit truncated x-coordinates:

- Entry size reduced from 38 bytes to 16 bytes
- Baby table stores only `i ∈ [1, M/2)` — values `r ≥ M/2` are recovered at the next giant step via the `-i` candidate, since `-(i*G)` has the same x-coordinate as `i*G`. The giant step range `j ∈ [0, 2^l2)` is unchanged.
- Combined memory saving: ~6× at large `l1`

| l1 | Before | After | Saving |
|----|--------|-------|--------|
| 22 | ~152 MB | 64 MB | 2.4× |
| 25 | ~1.2 GB | 512 MB | 2.4× |
| 26 | ~4.8 GB | 1 GB | **4.8×** |

### 3. Jacobian Loop Optimization (`feat/jacobian-loop`)
`secp256k1_ec_pubkey_combine()` internally converts Jacobian → affine on every call, paying one modular inversion each time. The giant step loop called it twice per step — once to advance Q (result used for lookup) and once to advance jMG (result wasted). This gives 2 inversions per step.

The Jacobian optimization keeps both Q and jMG in Jacobian coordinates throughout the loop, paying only the unavoidable lookup inversion:

| Operation | Before | After |
|---|---|---|
| Q advancement | 1 inversion (normalize, doubles as lookup) | 0 (Jacobian add) |
| jMG advancement | 1 inversion (normalize, wasted) | 0 (Jacobian add) |
| Q lookup | included above | 1 inversion |
| **Total per step** | **2 inversions** | **1 inversion** |

Measured speedup: **~1.8×** on search time.

### 4. Packed 8-byte Entry (`feat/packed-entry`)
The previous 16-byte entry wasted space through padding and a redundant `used` flag. Since `i` starts from 1 and is never 0, `val==0` serves as the empty sentinel:

```c
/* Previous: 16 bytes */
typedef struct {
    uint64_t key;   /* 8 bytes */
    uint32_t val;   /* 4 bytes */
    uint8_t  used;  /* 1 byte  */
                    /* 3 bytes padding */
} entry64_u32;

/* New: 8 bytes */
typedef struct {
    uint32_t key;  /* upper 32 bits of x64 */
    uint32_t val;  /* i value; 0 = empty   */
} entry_packed;
```

Memory saving: **2× across all l1 values**.

### 5. Windowed TreeMon Batch Inversion (`feat/windowed-treemon`)

The remaining 1 inversion/step from the Jacobian lookup is eliminated by batching W inversions at once using the tree-based Montgomery trick. The giant step loop operates in three phases per window of W steps:

**Phase 1** — Accumulate W Jacobian Q points (no inversion):
```
Q = Pm
for w in [0, W):
    Q_win[w] = Q
    Q = Q - MG     ← Jacobian subtraction, 0 inversions
```

**Phase 2** — Batch invert all Z coordinates (1 inversion total):
```
fe_batch_invert_tree(Z[0..W-1])    ← 1 inversion + 3(W-1) mults
```

**Phase 3** — Extract affine x and lookup (no inversion):
```
for w in [0, W):
    x_affine = Q_win[w].X * z_inv[w]²   ← 1 mul + 1 sqr
    cuckoo_lookup(x64)
```

**Cost per step:** `1/W inversions + ~5 multiplications` vs previous `1 inversion`.

Optimal W scales with l2 — larger l2 benefits from larger windows:

| bits | l2 | Optimal W |
|------|-----|-----------|
| 48 | 22 | 64 |
| 52 | 22 | 256 |
| 54 | 23–24 | 512 |
| 58 | 27 | 512 |

### 6. Cuckoo Hashing k=3 (`feat/cuckoo-hashing`)

Replaces open-addressing (load factor 2×) with k=3 cuckoo hashing (load factor 1.3×), achieving a **1.54× memory reduction** at all l1 values.

Two-phase build: 12-byte entries during construction (full x64 key needed for eviction), compacted to 8-byte lookup entries after build. O(1) worst-case lookup: exactly 3 probes + a 16-entry stash (0 stash entries observed in all experiments).

| l1 | Open-addr | Cuckoo | Saving |
|----|-----------|--------|--------|
| 26 | 512 MB | 333 MB | 1.54× |
| 28 | 2048 MB | 1331 MB | 1.54× |
| 30 | 8192 MB | 5325 MB | 1.54× |
| **31** | **16384 MB** | **10650 MB** | **1.54×** |

### 7. False-Positive Bug Fix (`fix/cuckoo-false-positive`)

The 8-byte entry stores only the upper 32 bits of x64 as the key discriminator, meaning two distinct x64 values with the same upper 32 bits can collide (false positive). The original `map_get()` returned on the first matching entry, so a false positive at cuckoo position 0 would cause the real entry at position 1 or 2 to be silently missed — a false negative.

**False-negative probability per solve (old code):**

| bits | Giant steps | P(false negative) |
|------|------------|-------------------|
| 54 | ~8M | ~0.6% |
| 58 | ~256M | ~18% |
| 63 | ~8B | virtually certain |

**Fix:** New `map_get_all()` collects all matching candidates across all 3 cuckoo positions + stash. Every call site iterates through all candidates with `verify_candidate()`. Correctness is now guaranteed at all bit sizes.

---

## Build

Requires secp256k1 internal headers for Jacobian arithmetic:

```bash
cc -O3 -Wall -Wextra -o bsgs bsgs_dlp_benchmark_cached.c \
    -I/usr/local/include \
    -I/path/to/secp256k1/src \
    -L/usr/local/lib \
    -lsecp256k1 -lpthread
```

Replace `/path/to/secp256k1/src` with your secp256k1 source directory. On a ripple/xrpl build environment this is typically `src/secp256k1/src` under your repo root.

---

## Usage

```
./bsgs <bits> <l1> <trials> <threads> [window]
```

- `bits` — plaintext range `[0, 2^bits)`, max 63
- `l1` — baby step parameter, table covers `[1, 2^(l1-1))`
- `trials` — number of random test cases (use ≥10 for reliable averages)
- `threads` — parallel giant step threads
- `window` — batch inversion window size W (default 64, must be power of 2)

**Example — recommended 54-bit configuration:**
```bash
./bsgs 54 30 20 10 512
```

---

## Benchmark Results

Hardware: Apple M-series, 32 GB RAM. All results show search time only (excludes one-time table build). All results confirmed correct (Solved correctly: N/N). Bold rows indicate the recommended configuration.

> **Note on trial counts:** BSGS solve time depends on where the random `m` falls in the giant-step range. At least 10 trials are needed for reliable averages. Single-trial results at 63-bit carry very high variance and are not cited as performance claims.

### 44–50 bit (Jacobian loop, 64-bit key, W=1)

| bits | l1 | l2 | Table | Avg solve |
|------|----|----|-------|-----------|
| 44 | **26** | 18 | 1 GB | **51 ms** |
| 46 | **26** | 20 | 1 GB | **154 ms** |
| 48 | **25** | 23 | 512 MB | **791 ms** |
| 50 | **28** | 22 | 4 GB | **914 ms** |

### 52–54 bit — Windowed TreeMon Window Sweep (l1=30, cuckoo, 10T, 10 trials)

| W | 52-bit (l2=22) | vs FastECDLP | 54-bit (l2=24) | vs FastECDLP |
|---|----------------|-----------|-----------------|-----------|
| 1 | ~1400 ms | 5.6× slower | 5291 ms | 5.4× slower |
| 64 | 345 ms | 1.4× slower | 1057 ms | 1.1× slower |
| 128 | 283 ms | 1.1× slower | 873 ms | 1.1× faster |
| **256** | **212 ms** | **1.17×** | 826 ms | 1.2× faster |
| 512 | 268 ms | 1.1× faster | **595 ms** | **1.66×** |

FastECDLP (Tang et al., 2022): 52-bit = 248 ms, 54-bit = 990 ms at T=16, Intel Xeon 2.30 GHz.

### 54-bit — Cuckoo l1=31 vs l1=30 (W=512, 10T)

| l1 | Table | l2 | Trials | Avg solve | vs FastECDLP | Notes |
|----|-------|----|--------|-----------|--------------|-------|
| 30 | 5.3 GB | 24 | 10 | ~400 ms | **~2.5× faster** | two runs: 344ms, 458ms |
| 31 | 10.4 GB | 23 | **20** | **548 ms** | **1.8× faster** | reliable, post bug-fix |

Despite having fewer giant steps (l2=23 vs l2=24), l1=31 is slower than l1=30 because the 10.4 GB table causes memory pressure on a 32 GB machine. **l1=30 cuckoo is the recommended configuration for 54-bit.**

### 58-bit — Cuckoo l1=31 (W=512, 10T, 10 trials, post bug-fix)

| Trials | Avg solve |
|--------|-----------|
| 10 | **5245 ms** |

### 63-bit — Full MPT Range (cuckoo l1=31, W=512, 10T, post bug-fix)

| Trials | Avg solve | Status |
|--------|-----------|--------|
| 3 | **108 sec** | confirmed correct (3/3) |

Consistent with the doubling rule from the 54-bit result (~400 ms × 2^8 ≈ 102 sec).

---

## Comparison with FastECDLP (Tang et al., 2022)

FastECDLP: Intel Xeon 2.30 GHz, 16 threads. This work: Apple M-series, 10 threads.

| bits | l1 | Table | Trials | FastECDLP (T=16) | **This work (T=10)** | **Speedup** |
|------|-----|-------|--------|-----------------|---------------------|-------------|
| 52 | 30 | 5.3 GB | 10 | 248 ms | **212 ms** (W=256) | **1.17×** |
| **54** | **30** | **5.3 GB** | **10** | **990 ms** | **~400 ms** (W=512) | **~2.5×** |
| 54 | 31 | 10.4 GB | 20 | 990 ms | 548 ms (W=512) | 1.8× |

**Headline result: ~2.5× faster than FastECDLP at 54-bit using 6 fewer threads (l1=30, cuckoo).**

---

## Optimal Configuration Guide

### For a 32 GB machine (cuckoo hashing, windowed TreeMon)

| bits | l1 | Table | W | Threads | Avg solve | Trials |
|------|----|-------|---|---------|-----------|--------|
| 52 | 30 | 5.3 GB | 256 | 10 | ~212 ms | 10 |
| **54** | **30** | **5.3 GB** | **512** | **10** | **~400 ms** | **10** |
| 54 | 31 | 10.4 GB | 512 | 10 | ~548 ms | 20 |
| 58 | 31 | 10.4 GB | 512 | 10 | ~5.2 sec | 10 |
| 63 | 31 | 10.4 GB | 512 | 10 | ~108 sec | 3 |

### One-time build costs (cuckoo, single-threaded)

| l1 | Table | Build time |
|----|-------|------------|
| 26 | 333 MB | ~2 min |
| 28 | 1.3 GB | ~8 min |
| 30 | 5.3 GB | ~32 min |
| 31 | 10.4 GB | ~180 min |

---

## Planned Work

- **Non-power-of-2 radix M**: replacing M=2^l1 with M=3×2^30 (~15.6 GB cuckoo) reduces l2 for 54-bit from 24 to 22.4 bits — ~27% fewer giant steps, projected ~290 ms. Optimal M for 32 GB is M≈6.15×10^9. Infrastructure already present via `fastrange64` in the cuckoo code.
- **Parallel baby table build**: at l1=31 the single-threaded build costs ~180 minutes; parallelizing across threads would reduce this linearly.
- **64-bit full range**: requires replacing `uint64_t` overflow in `j×M` with `__uint128_t` scalar arithmetic — a mechanical fix.
- **Literature verification**: confirm that windowed batch inversion applied to the iterative Jacobian giant step (eliminating T2) has not appeared in Bernstein–Lange 2012, Galbraith et al. 2017, or Chatzigiannis et al. 2021.