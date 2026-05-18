# Small DLP Solver — BSGS on secp256k1

A scalable Baby-Step Giant-Step (BSGS) solver for the small Elliptic Curve
Discrete Logarithm Problem (ECDLP) on secp256k1, with Jacobian coordinate
optimization, windowed batch inversion, packed memory layout, and k=3 cuckoo
hashing. Covers the full XRPL MPT (XLS-33) plaintext range `[0, 2^63)`.

---

## Overview

EC-based Additively Homomorphic Encryption (AHE) schemes such as EC-ElGamal
and Twisted ElGamal require solving a small ECDLP during decryption: recovering
`m` from `m*G` where `m` is a bounded integer. This repository contains four
implementations enabling a clean progression from the Tang et al. baseline to
our complete solver:

| File | Description |
|---|---|
| `fastecdlp_baseline.c` | Faithful re-implementation of FastECDLP (Tang et al., 2022) |
| `fastecdlp_original.c` | FastECDLP with parallelised Phase 1+2 (our extension) |
| `fastecdlp_jacobian.c` | FastECDLP + Jacobian loop (§3.1, eliminates T₂) |
| `bsgs_dlp_benchmark_cached.c` | Complete solver: Jacobian loop + windowed batch inversion (§3.2) |
| `bench_field.c` | Field operation microbenchmark (inv/mul ratio) |

---

## Algorithmic Contributions

### §3.1 Jacobian Coordinate Loop

`secp256k1_ec_pubkey_combine()` pays one field inversion per call. The
baseline giant-step loop called it twice per step — once to advance Q
(used for lookup) and once to advance jMG (result wasted).

We replace both calls with `secp256k1_gej_add_ge()` (mixed Jacobian–affine
addition, zero inversions), keeping Q and jMG in Jacobian form throughout.
The only inversion paid per step is the unavoidable lookup inversion.

| Operation | Baseline | §3.1 |
|---|---|---|
| Advance Q | 1 inversion | 0 (Jacobian add) |
| Advance jMG | 1 inversion (wasted) | 0 (Jacobian add) |
| Lookup | included above | 1 inversion |
| **Total/step** | **2 inversions** | **1 inversion** |

This contribution also **eliminates the precomputed T₂ table** required by
FastECDLP, since the giant-step point is advanced iteratively from Pm rather
than read from a precomputed affine table.

### §3.2 Windowed Batch Inversion

The remaining 1 inversion/step is amortised over a window of W steps using
Montgomery batch inversion. Each window of W Jacobian points is batch-inverted
in one shot at cost `1 inv + 3(W-1) mults`:

```
Phase 1: accumulate W Jacobian Q points (0 inversions)
Phase 2: batch-invert all W Z-coordinates (1 inversion + 3(W-1) mults)
Phase 3: extract affine x and look up in baby table (0 inversions)
```

**Per-step cost:** `1/W inversions + ~5 multiplications`

**Key property:** per-thread working memory is `W × 96 bytes = O(W)`,
independent of bit size. At W=512 this is ≈50 KB per thread, enabling
63-bit decryption where FastECDLP requires ≥175 GB for its T₂ table.

---

## Memory Optimizations (following FastECDLP)

### Packed 8-byte Entry
Each baby table entry stores the upper 32 bits of the x-coordinate as key
and the index i as a 32-bit value. Since i ≥ 1, val=0 serves as the empty
sentinel. **2× memory saving** over the 16-byte entry with padding.

### k=3 Cuckoo Hashing
Baby table uses k=3 cuckoo hashing at load factor 1/1.3×, divided into
three sections. Lookup checks exactly 3 positions in O(1) worst-case.
**1.54× memory saving** over open-addressing at all l1 values.

Since the 8-byte entry stores only the upper 32 bits of x64 as the key
discriminator, all 3 cuckoo positions must be checked on every lookup to
avoid false negatives caused by distinct x64 values sharing the same upper
32 bits. This is critical at 63-bit where ~8×10⁹ lookups are performed.

| l1 | Open-addr | Cuckoo | Saving |
|----|-----------|--------|--------|
| 30 | 8.0 GB | 5.3 GB | 1.54× |
| 31 | 16.0 GB | 10.4 GB | 1.54× |

---

## Build

Requires secp256k1 internal headers for Jacobian arithmetic:

```bash
# Our complete solver
cc -O3 -Wall -Wextra -o bsgs bsgs_dlp_benchmark_cached.c \
    -I/usr/local/include \
    -I/path/to/secp256k1/src \
    -L/usr/local/lib \
    -lsecp256k1 -lpthread

# FastECDLP baseline (Tang et al. faithful)
cc -O3 -Wall -Wextra -o fastecdlp_baseline fastecdlp_baseline.c \
    -I/usr/local/include \
    -I/path/to/secp256k1/src \
    -L/usr/local/lib \
    -lsecp256k1 -lpthread

# FastECDLP + parallel Phase 1+2
cc -O3 -Wall -Wextra -o fastecdlp_original fastecdlp_original.c \
    -I/usr/local/include \
    -I/path/to/secp256k1/src \
    -L/usr/local/lib \
    -lsecp256k1 -lpthread

# FastECDLP + Jacobian loop (no T₂)
cc -O3 -Wall -Wextra -o fastecdlp_jacobian fastecdlp_jacobian.c \
    -I/usr/local/include \
    -I/path/to/secp256k1/src \
    -L/usr/local/lib \
    -lsecp256k1 -lpthread

# Field microbenchmark
cc -O3 -Wall -Wextra -o bench_field bench_field.c \
    -I/usr/local/include \
    -I/path/to/secp256k1/src \
    -L/usr/local/lib \
    -lsecp256k1
```

Replace `/path/to/secp256k1/src` with your secp256k1 source directory.

---

## Usage

```
./bsgs <bits> <l1> <trials> <threads> [window]
```

| Argument | Description |
|---|---|
| `bits` | plaintext range `[0, 2^bits)`, max 63 |
| `l1` | baby step parameter, table covers `[1, 2^(l1-1))` |
| `trials` | number of random test cases (≥10 for reliable averages) |
| `threads` | parallel giant step threads |
| `window` | batch inversion window size W (default 64, must be power of 2) |

**Recommended 54-bit configuration:**
```bash
./bsgs 54 30 10 10 512
```

---

## Benchmark Results

Hardware: Apple M-series, 32 GB RAM, 10 threads.
All results show search time only — one-time baby table build and T₂
precomputation are excluded. All results confirmed correct (N/N trials).

### Four-Way On-Machine Comparison

| bits | l1 | Baseline (Tang et al.) | +Parallel Ph.1+2 | +Jacobian (§3.1) | This work (§3.1+§3.2) |
|---|---|---|---|---|---|
| 52 | 30 | 370 ms | **99 ms** | 207 ms | 213 ms (W=256) |
| 54 | 30 | 1474 ms | **482 ms** | 942 ms | 845 ms (W=512) |
| 54 | 31 | 768 ms | **468 ms** | 659 ms | **476 ms** (W=512) |
| 58 | 31 | 119 sec† | 125 sec† | 145 sec† | **6.75 sec** |
| 63 | 31 | infeasible | infeasible | infeasible | **164 sec** |

† severe memory pressure (>21 GB total allocation on 32 GB machine)

**Key result:** At 58-bit our solver is **21× faster** than the baseline.
At 63-bit our solver is the **only feasible approach** — FastECDLP's T₂
table would require 175 GB.

### Window Size Sweep (54-bit, l1=30, 10 threads, 10 trials)

| W | Avg solve | Speedup vs W=1 |
|---|---|---|
| 1 | 5291 ms | 1× |
| 64 | 1057 ms | 5.0× |
| 256 | 826 ms | 6.4× |
| **512** | **595 ms** | **8.9×** |
| 2048 | 801 ms | 6.6× |

Performance drop at W=2048 reflects L2 cache pressure
(W × 96 bytes exceeds L2 cache per core).

### Field Operation Costs (Apple M-series, bench_field.c)

| Operation | Best | Avg |
|---|---|---|
| fe_mul | 16 ns | 16 ns |
| fe_sqr | 14 ns | 14 ns |
| fe_inv | 2660 ns | 3208 ns |
| **inv/mul ratio** | **165×** | **198×** |

---

## Optimal Configuration (32 GB machine)

| bits | l1 | Table | W | Avg solve | Trials |
|---|---|---|---|---|---|
| 52 | 30 | 5.3 GB | 256 | ~213 ms | 10 |
| **54** | **30** | **5.3 GB** | **512** | **~845 ms** | **10** |
| 54 | 31 | 10.4 GB | 512 | ~476 ms | 10 |
| 58 | 31 | 10.4 GB | 512 | ~6.75 sec | 10 |
| 63 | 31 | 10.4 GB | 512 | ~164 sec | 10 |

**Note on 54-bit configuration:** l1=31 (476 ms) is faster than l1=30
(845 ms) when memory is not under pressure. l1=30 is recommended when
running alongside other large processes.

### One-time baby table build costs (single-threaded)

| l1 | Table | Build time |
|---|---|---|
| 30 | 5.3 GB | ~32 min |
| 31 | 10.4 GB | ~180 min |

---

## Planned Work

- **Non-power-of-2 radix M**: replacing M=2^l1 with M=3×2^30 (~15.6 GB
  cuckoo table) projects ~290 ms at 54-bit. Infrastructure already present
  via `fastrange64` in the cuckoo code.
- **Parallel baby table build**: at l1=31 the single-threaded build costs
  ~180 min; parallelising across T threads reduces to ~18 min.
- **64-bit full range**: requires `__uint128_t` scalar arithmetic for
  j×M overflow at l2=33 — a mechanical fix.
- **Literature verification**: confirm that the Jacobian iterative walk
  combined with windowed batch inversion (eliminating T₂) has not appeared
  in Bernstein–Lange 2012, Galbraith et al. 2017, or Chatzigiannis et al.
    2021.

---

## Reference

Tang et al., *Solving Small Exponential ECDLP in EC-Based Additively
Homomorphic Encryption and Applications*, ePrint 2022/1573.