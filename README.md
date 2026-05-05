# Small DLP Solver — BSGS on secp256k1

A fast Baby-Step Giant-Step solver for the small exponential Elliptic Curve Discrete Logarithm Problem (ECDLP) on secp256k1, with caching, parallelization, Jacobian coordinate optimization, packed memory layout, and windowed batch inversion.

---

## Overview

EC-based Additively Homomorphic Encryption (AHE) schemes such as Exp-ElGamal require solving a small ECDLP during decryption: recovering `m` from `m*G` where `m` is a bounded integer. This solver implements an optimized BSGS algorithm targeting practical plaintext lengths up to 58+ bits.

---

## Optimizations

### 1. Multi-threaded Giant Step (`feat/multi-thread`)
The giant step search is split across threads. Each thread independently computes its starting point via scalar multiplication and walks its assigned range in parallel. Uses an `atomic_int` flag for early termination when any thread finds the solution.

### 2. 64-bit Key Truncation (`feat/64bit-key-truncation`)
The baby table originally stored full 33-byte compressed EC points as keys. This was replaced with 64-bit truncated x-coordinates:

- Entry size reduced from 38 bytes to 16 bytes
- Table entries halved by exploiting `(i*G)[x] == (-i*G)[x]` symmetry
- Combined memory saving: ~6× at large `l1`
- Fixes cache save failure at `l1=26` (4.8 GB → 1 GB)

| l1 | Before | After | Saving |
|----|--------|-------|--------|
| 22 | ~152 MB | 64 MB | 2.4× |
| 25 | ~1.2 GB | 512 MB | 2.4× |
| 26 | ~4.8 GB | 1 GB | **4.8×** |

### 3. Jacobian Loop Optimization (`feat/jacobian-loop`)
`secp256k1_ec_pubkey_combine()` internally converts Jacobian → affine on every call, paying one modular inversion each time. The giant step loop called it twice per step (to advance Q and jMG), giving 2 inversions per step.

This optimization keeps Q and jMG in Jacobian coordinates throughout the loop using internal secp256k1 types, converting to affine only once per step for the hash table lookup:

| Operation | Before | After |
|---|---|---|
| Q advancement | 1 inversion | 0 (Jacobian add) |
| jMG advancement | 1 inversion | 0 (Jacobian add) |
| jMG comparison | 0 | 2 multiplications |
| Q lookup | 1 inversion | 1 inversion (unavoidable alone) |
| **Total per step** | **2 inversions** | **1 inversion + 2 mults** |

Measured speedup: **~1.8×** on search time.

### 4. Packed 8-byte Entry (`feat/packed-entry`)
The previous 16-byte entry wasted space through padding and a redundant `used` flag. Since `i` starts from 1 and is never 0, `val==0` serves as the empty sentinel. The key is split: the full 64-bit hash positions the entry in the table (not stored), and only the upper 32 bits are stored as a discriminator:

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

Memory saving: **2× across all l1 values**, enabling l1=30 (8 GB) on a 32 GB machine.

| l1 | Before | After | Saving |
|----|--------|-------|--------|
| 28 | 4 GB | 2 GB | 2× |
| 29 | 8 GB | 4 GB | 2× |
| 30 | 16 GB | 8 GB | 2× |
| 31 | 32 GB | 16 GB | 2× |

### 5. Windowed TreeMon Batch Inversion (`feat/windowed-treemon`)

The remaining 1 inversion/step from the Jacobian lookup is eliminated by batching W inversions at once using the tree-based Montgomery trick. The giant step loop now operates in three phases per window of W steps:

**Phase 1** — Accumulate W Jacobian Q points (no inversion):
```
for w in [0, W):
    check i=0 via gej_eq_ge()      ← 0 inversions
    Q_win[w] = Qj
    gej_add_ge(&Qj, neg_MG)        ← 0 inversions
```

**Phase 2** — Batch invert all Z coordinates (1 inversion total):
```
fe_batch_invert_tree(Z[0..W-1])    ← 1 inversion + 3(W-1) mults
```

**Phase 3** — Extract affine x and lookup (no inversion):
```
for w in [0, W):
    x_affine = Q_win[w].X * z_inv[w]²   ← 1 mul + 1 sqr
    map_get(x64)
```

**Cost per step:** `1/W inversions + ~5 multiplications`
vs previous: `1 inversion + 2 multiplications`

The window size W is passed as a 5th command line argument. Optimal W scales with l2:

| bits | Optimal W | Reason |
|------|-----------|--------|
| 48 | 64 | Small l2, fits L1 cache |
| 52 | 256 | Medium l2=22 |
| 54 | 512 | Larger l2=24 |

---

## Build

Requires secp256k1 internal headers for Jacobian arithmetic (not included in the Homebrew installation):

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

- `bits` — plaintext range `[0, 2^bits)`
- `l1` — baby step parameter, table covers `[1, 2^(l1-1))`
- `trials` — number of random test cases
- `threads` — parallel giant step threads
- `window` — batch inversion window size W (default 64, must be power of 2)

**Example:**
```bash
./bsgs 54 30 10 10 512
```

---

## Benchmark Results

Hardware: Apple M-series, 10 threads, 10 trials unless noted. All results show search time only (excludes one-time table build). Bold rows indicate the recommended configuration.

### 44–50 bit (Jacobian loop, 64-bit key, W=1, 3 trials)

### 44-bit

| l1 | l2 | Table | Avg solve |
|----|----|-------|-----------|
| 19 | 25 | 8 MB | 4911 ms |
| 21 | 23 | 32 MB | 1516 ms |
| 22 | 22 | 64 MB | 896 ms |
| 23 | 21 | 128 MB | 479 ms |
| 24 | 20 | 256 MB | 142 ms |
| 25 | 19 | 512 MB | 110 ms |
| **26** | **18** | **1024 MB** | **51 ms** |
| 27 | 17 | 2048 MB | 122 ms |
| 28 | 16 | 4096 MB | 46 ms |

### 46-bit

| l1 | l2 | Table | Avg solve |
|----|----|-------|-----------|
| 21 | 25 | 32 MB | 5905 ms |
| 23 | 23 | 128 MB | 1881 ms |
| 24 | 22 | 256 MB | 817 ms |
| 25 | 21 | 512 MB | 596 ms |
| **26** | **20** | **1024 MB** | **154 ms** |
| 27 | 19 | 2048 MB | 151 ms |
| 28 | 18 | 4096 MB | 219 ms |

### 48-bit

| l1 | l2 | Table | Avg solve |
|----|----|-------|-----------|
| 23 | 25 | 128 MB | 7373 ms |
| 24 | 24 | 256 MB | 2985 ms |
| **25** | **23** | **512 MB** | **791 ms** |
| 26 | 22 | 1024 MB | 900 ms |
| 27 | 21 | 2048 MB | 338 ms |
| 28 | 20 | 4096 MB | 355 ms |

### 50-bit

| l1 | l2 | Table | Avg solve |
|----|----|-------|-----------|
| 25 | 25 | 512 MB | 3276 ms |
| 26 | 24 | 1024 MB | 2080 ms |
| **27** | **23** | **2048 MB** | **1933 ms** |
| 28 | 22 | 4096 MB | 914 ms |

---

### 52–54 bit — Windowed TreeMon vs FastECDLP (Tang et al., 2022)

Results use packed 8-byte entry + Jacobian loop + windowed TreeMon, l1=30, 8 GB table, 10 threads, **10 trials**.

#### Window sweep — 52-bit (l1=30, l2=22)

| W | Avg solve | vs FastECDLP (248 ms, T=16) |
|---|-----------|--------------------------|
| 1 | ~1400 ms | 5.6× slower |
| 64 | 345 ms | 1.4× slower |
| 128 | 283 ms | 1.1× slower |
| **256** | **212 ms** | **1.17× faster** |
| 512 | 268 ms | 1.1× faster |

#### Window sweep — 54-bit (l1=30, l2=24)

| W | Avg solve | vs FastECDLP (990 ms, T=16) |
|---|-----------|--------------------------|
| 1 | 5291 ms | 5.4× slower |
| 64 | 1057 ms | 1.1× slower |
| 128 | 873 ms | 1.1× faster |
| 256 | 826 ms | 1.2× faster |
| **512** | **595 ms** | **1.66× faster** |
| 2048 | 801 ms | 1.2× faster |

---

### 56–58 bit (packed entry, Jacobian loop, W=1)

| bits | l1 | l2 | Threads | Trials | Avg solve |
|------|----|----|---------|--------|-----------|
| 56 | 30 | 26 | 10 | 5 | 15878 ms |
| 57 | 30 | 27 | 10 | 5 | 30735 ms |
| **58** | **30** | **28** | **10** | **10** | **47485 ms** |

56–58 bit windowed TreeMon benchmarks are pending.

---

## Comparison with FastECDLP (Tang et al., 2022)

FastECDLP reports results on Intel Xeon 2.30 GHz at 16 threads. Our results use Apple M-series at 10 threads.

| bits | FastECDLP T=16 | **Ours T=10, best W** | **Speedup** |
|------|---------------|----------------------|-------------|
| 52 | 248 ms | **212 ms** (W=256) | **1.17×** |
| 54 | 990 ms | **595 ms** (W=512) | **1.66×** |

We outperform the state-of-the-art at both 52-bit and 54-bit using 6 fewer threads.

---

## Optimal Configuration Guide

### 44–50 bit (≤4 GB RAM for table)

| bits | l1 (≤1 GB) | l1 (≤2 GB) | l1 (≤4 GB) |
|------|-----------|-----------|-----------|
| 44 | 26 (~51 ms) | 27 (~122 ms) | 28 (~46 ms) |
| 46 | 26 (~154 ms) | 27 (~151 ms) | 28 (~219 ms) |
| 48 | 25 (~791 ms) | 27 (~338 ms) | 28 (~355 ms) |
| 50 | — | 27 (~1933 ms) | 28 (~914 ms) |

### 52–58 bit (packed entry + windowed TreeMon, l1=30, 8 GB table)

| bits | Best W | Avg solve | Threads |
|------|--------|-----------|---------|
| 52 | 256 | ~212 ms | 10 |
| 54 | 512 | ~595 ms | 10 |
| 56 | — | ~15.9 sec | 10 |
| 57 | — | ~30.7 sec | 10 |
| 58 | — | ~47.5 sec | 10 |

---

## Planned Work

- **Windowed TreeMon for 56–58 bit**: run window sweeps to find optimal W and update benchmarks
- **Reduce load factor** from 2.0× to ~1.3×: saves ~35% memory, matching the cuckoo hashing overhead used in FastECDLP (Tang et al., 2022)
- **Parallel baby table build**: at l1=30 the single-threaded build costs ~41 minutes; parallelizing would scale linearly with thread count