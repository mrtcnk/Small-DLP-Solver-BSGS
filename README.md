# Branch Report: `feat/64bit-key-truncation`

## Overview

This branch introduces a 64-bit key truncation optimization to the baby table in the BSGS solver. It is branched from `feat/multi-thread` and builds directly on top of the parallel solve implementation.

## Problem

The original implementation stored full 33-byte compressed EC points as keys in the baby hash table. At `l1=26`, this produced a ~4.8 GB table that exceeded available disk space, causing the cache save to fail silently. As a result, every run at `l1=26` required a 227-second rebuild from scratch.

## Changes

**Key structure** (`entry33_u32` → `entry64_u32`): The 33-byte compressed point key is replaced with a `uint64_t` holding the first 8 bytes of the x-coordinate. Entry size drops from 38 bytes to 16 bytes.

**Half-range table**: Since `(i*G)[x] == (-i*G)[x]`, both `i` and `-i` share the same x-coordinate. The baby table now stores only `i` in `[1, 2^(l1-1))` instead of `[1, 2^l1)`, halving the number of entries.

**Candidate verification**: Each lookup hit now yields two candidates — `j*M + i` and `j*M - i`. A new `verify_candidate()` function checks which is correct via one scalar multiplication.

**Cache filename**: Updated to `bsgs_baby64_secp256k1_l1_*.bin` to avoid loading incompatible old-format files.

**Error diagnostics**: Added `perror()` calls to the cache save path to surface future failures clearly.

## Memory Impact

| l1 | Before | After | Saving |
|----|--------|-------|--------|
| 22 | ~152 MB | 64 MB | 2.4× |
| 25 | ~1.2 GB | 512 MB | 2.4× |
| 26 | ~4.8 GB | 1 GB | **4.8×** |

## Results

All trials solved correctly across all tested bit sizes (32, 44, 48, 49).

| bits | l1 | l2 | Threads | Avg solve |
|------|----|----|---------|-----------|
| 32   | 18 | 14 | 4       | 5.75 ms   |
| 44   | 22 | 22 | 10      | 795.89 ms |
| 44   | 25 | 19 | 10      | 177.77 ms |
| 48   | 25 | 23 | 10      | 3510.92 ms |
| 48   | 26 | 22 | 10      | 1609.70 ms |
| 49   | 26 | 23 | 10      | 3853.43 ms |

## Remaining Limitation

The baby table build loop is still single-threaded. At `l1=26` this costs ~107 seconds on first run. This is now a **one-time cost** since the cache saves correctly, but parallelizing the build loop is a natural next step.