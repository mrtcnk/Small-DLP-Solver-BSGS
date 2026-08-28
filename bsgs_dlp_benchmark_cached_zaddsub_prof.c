#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <inttypes.h>
#include <errno.h>

#include <secp256k1.h>

#include "secp256k1_gej_zaddsub_ge_var.h"


/*
 * secp256k1 internal headers for Jacobian arithmetic.
 *
 * Clone secp256k1 source alongside this file:
 *   git clone https://github.com/bitcoin-core/secp256k1 secp256k1_src
 *
 * Build:
 *   cc -O3 -Wall -Wextra -o bsgs_zaddsub bsgs_dlp_benchmark_zaddsub.c \
 *       -I/usr/local/include                                   \
 *       -I/path/to/secp256k1/src                               \
 *       -L/usr/local/lib                                       \
 *       -lsecp256k1 -lpthread
 *
 * On a ripple/xrpl environment:
 *   -I/Users/<user>/ripple2/src/secp256k1/src
 */
#include "util.h"
#include "field.h"
#include "field_impl.h"
#include "int128_impl.h"
#include "group.h"
#include "group_impl.h"

/* POSIX I/O */
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <stdatomic.h>

/* ---------------- timing ---------------- */

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ---------------- helpers ---------------- */

static void u64_to_scalar32_be(uint64_t x, unsigned char out32[32]) {
    memset(out32, 0, 32);
    for (int i = 0; i < 8; i++) {
        out32[31 - i] = (unsigned char)(x & 0xFF);
        x >>= 8;
    }
}

static int pubkey_serialize33(const secp256k1_context* ctx,
                              const secp256k1_pubkey* pk,
                              unsigned char out33[33]) {
    size_t outlen = 33;
    if (!secp256k1_ec_pubkey_serialize(ctx, out33, &outlen, pk,
                                       SECP256K1_EC_COMPRESSED))
        return 0;
    return outlen == 33;
}

static int file_exists(const char* path) {
    struct stat st;
    return stat(path, &st) == 0;
}

/* Chunked write/read to work around 2 GB single-call limit on macOS */
static int write_all(int fd, const void* buf, size_t len) {
    const unsigned char* p = (const unsigned char*)buf;
    while (len > 0) {
        size_t  chunk = len < (1ULL << 30) ? len : (1ULL << 30);
        ssize_t n     = write(fd, p, chunk);
        if (n < 0) {
            if (errno == EINTR) continue;
            perror("write_all");
            return 0;
        }
        p   += (size_t)n;
        len -= (size_t)n;
    }
    return 1;
}

static int read_all(int fd, void* buf, size_t len) {
    unsigned char* p = (unsigned char*)buf;
    while (len > 0) {
        size_t  chunk = len < (1ULL << 30) ? len : (1ULL << 30);
        ssize_t n     = read(fd, p, chunk);
        if (n < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        if (n == 0) return 0;
        p   += (size_t)n;
        len -= (size_t)n;
    }
    return 1;
}

/* ================================================================
 * Jacobian helpers using secp256k1 internal types
 * ================================================================
 *
 * Core idea:
 *   secp256k1_gej_add_ge(&r, &a, &b)  -- Jacobian + affine, NO inversion
 *   secp256k1_ge_set_gej(&r, &a)      -- Jacobian -> affine, ONE inversion
 *                                         also normalises a in-place (z->1)
 *
 * Per-step inversion reduction:
 *   Previous:  pubkey_combine(Q)   -> 1 inv
 *              pubkey_combine(jMG) -> 1 inv
 *              serialize(Q)        -> 0 (already affine)
 *              serialize(jMG)      -> 0 (already affine)
 *              Total: 2 inv/step   (+ 1 inv/step above was wrong — total was 2)
 *
 *   This version:
 *              gej_add_ge(Qj)      -> 0 inv  (Jacobian add)
 *              gej_add_ge(jMGj)    -> 0 inv  (Jacobian add)
 *              ge_set_gej(Qj)      -> 1 inv  (x-coord for lookup, unavoidable)
 *              gej_eq_ge(jMGj)     -> 0 inv  (2 mults)
 *              Total: 1 inv/step   (~2x speedup on the loop)
 */

/* Load secp256k1_pubkey into affine secp256k1_ge.
 * Mirrors libsecp256k1's internal pubkey_load() exactly. */
static void pubkey_to_ge(const secp256k1_pubkey* pk, secp256k1_ge* ge) {
    if (sizeof(secp256k1_ge_storage) == 64) {
        secp256k1_ge_storage s;
        memcpy(&s, &pk->data[0], sizeof(s));
        secp256k1_ge_from_storage(ge, &s);
    } else {
        secp256k1_fe x, y;
        secp256k1_fe_set_b32_mod(&x, pk->data);
        secp256k1_fe_set_b32_mod(&y, pk->data + 32);
        secp256k1_ge_set_xy(ge, &x, &y);
    }
}

/* Load secp256k1_pubkey as a Jacobian point with z=1. */
static void pubkey_to_gej(const secp256k1_pubkey* pk, secp256k1_gej* gej) {
    secp256k1_ge ge;
    pubkey_to_ge(pk, &ge);
    secp256k1_gej_set_ge(gej, &ge);
}

/* Negate an affine point in-place (flip y). */
static void ge_negate(secp256k1_ge* out, const secp256k1_ge* in) {
    *out = *in;
    secp256k1_fe_negate(&out->y, &out->y, 1);
    secp256k1_fe_normalize_var(&out->y);
}

/* Extract full 32-byte x-coordinate (big-endian) into buf.
 * Used as the hash input for Tang et al. window-based cuckoo hashing. */
static void ge_xb32(const secp256k1_ge* ge, unsigned char buf[32]) {
    secp256k1_fe x = ge->x;
    secp256k1_fe_normalize_var(&x);
    secp256k1_fe_get_b32(buf, &x);
}

/* ================================================================
 * Windowed TreeMon batch inversion helpers
 * ================================================================
 *
 * gej_x64_from_zinv(): extract upper-64-bit affine x from
 *   a Jacobian point given its precomputed Z-inverse.
 *   No inversion — 1 sqr + 1 mul.
 */

static void gej_xb32_from_zinv(const secp256k1_gej* pt,
                               const secp256k1_fe*  z_inv,
                               unsigned char buf[32]) {
    secp256k1_fe z2, x;
    secp256k1_fe_sqr(&z2, z_inv);
    secp256k1_fe_mul(&x, &pt->x, &z2);
    secp256k1_fe_normalize_var(&x);
    secp256k1_fe_get_b32(buf, &x);
}

/**
 * * fe_batch_invert_skewed_tree(): inverts `n` field elements using
 *   1 inversion + 3(n-1) multiplications.
 *   Works for any arbitrary n >= 1.
 *
 * @param[out] out      Output array of size `n` for the inverses.
 * @param[in]  x        Input array of size `n` containing elements to invert.
 * @param[in]  scratch  Scratchpad array of size `n` to hold prefix products.
 * @param[in]  n        Number of elements (any positive integer).
 */
static void fe_batch_invert_skewed_tree(secp256k1_fe* out,
                                   const secp256k1_fe* x,
                                   secp256k1_fe* scratch,
                                   size_t n) {
    // if (n == 0) return;

    /* 1. Forward pass: scratch[i] = x[0] * ... * x[i] */
    scratch[0] = x[0];
    for (size_t i = 1; i < n; i++) {
        secp256k1_fe_mul(&scratch[i], &scratch[i - 1], &x[i]);
    }

    /* 2. Single field inversion of the total product */
    secp256k1_fe acc;
    secp256k1_fe_inv(&acc, &scratch[n - 1]);

    /* 3. Backward pass: compute inverses and unwind accumulator */
    for (size_t i = n - 1; i > 0; i--) {
        secp256k1_fe_mul(&out[i], &scratch[i - 1], &acc);
        secp256k1_fe_mul(&acc, &acc, &x[i]);
    }

    /* Handle the first element */
    out[0] = acc;
}

/*
 * fe_xb32_from_zinv_sq(): extract upper-64-bit affine x from
 *   the x-coordinate of Jacobian point given its precomputed square of Z-inverse.
 *   No inversion — 1 mul.
 */
static void fe_xb32_from_zinv_sq(const secp256k1_fe* x_jac,
                               const secp256k1_fe*  z_inv_sq,
                               unsigned char buf[32]) {
    secp256k1_fe x;
    secp256k1_fe_mul(&x, x_jac, z_inv_sq);
    secp256k1_fe_normalize_var(&x);
    secp256k1_fe_get_b32(buf, &x);
}

/* ================================================================
 * Cuckoo Hashing — k=3 sections, load factor ~1.3×
 * ================================================================
 *
 * Design:
 *   - 3 hash functions map x64 into 3 disjoint sections of size
 *     section_size ≈ 1.3n/3 each.  Total bins ≈ 1.3n.
 *   - Lookup: O(1) worst-case — exactly 3 probes (+ tiny stash).
 *   - Build: two-pass, 12-byte build_entry (3 positions + val),
 *     then compact to 8-byte entry_packed, then second pass fills keys.
 *
 * Memory vs open-addressing (load 0.5):
 *   l1=30: 1.3×8×2^29 ≈  5.6 GB  (was  8 GB)
 *   l1=31: 1.3×8×2^30 ≈ 11.2 GB  (was 16 GB) ← matches paper!
 *
 * Build-phase memory (12-byte entries, freed after compaction):
 *   l1=30: ≈  8.4 GB  (one-time)
 *   l1=31: ≈ 16.8 GB  (one-time, then compacts to 11.2 GB)
 */

/* ---- constants ---- */
#define CUCKOO_K          3
#define CUCKOO_MAX_RELOC  512
#define CUCKOO_STASH_SZ   16

/* ---- types ---- */

/* 8-byte lookup entry — kept in memory after build */
typedef struct {
    uint32_t key;   /* per-section 4-byte window key — discriminator */
    uint32_t val;   /* i value; 0 = empty                            */
} entry_packed;

/* 12-byte build entry — used during cuckoo construction only.
 * Stores only precomputed positions (for eviction rehashing) and val.
 * Keys are NOT stored here — they are filled in a second pass after
 * compaction by re-walking the i*G points. This keeps build memory
 * at 12 bytes/entry matching the original implementation. */
typedef struct {
    uint32_t pos[3];  /* precomputed position within each section (offset from section base) */
    uint32_t val;     /* i value; 0 = empty */
} build_entry;

/* cuckoo map (lookup phase, after compaction) */
typedef struct {
    entry_packed* tab;          /* 3 × section_size entries        */
    size_t        section_size; /* bins per section (not pow2)     */
    size_t        total_bins;   /* 3 × section_size                */
    size_t        size;         /* entries successfully inserted    */
    /* overflow stash (rare: ~0 entries with load 1.3×, k=3) */
    unsigned char stash_xb[CUCKOO_STASH_SZ][32];
    uint32_t stash_val[CUCKOO_STASH_SZ];
    int      stash_count;
} cuckoo_map;

/* ---- helpers ---- */

/* Tang et al. window-based cuckoo hash (Section IV.C.2):
 * The full 32-byte x-coordinate X is split into three 8-byte windows.
 * Each window contributes 4 bytes for position and 4 bytes for key:
 *   Window 0: X[0:4]  → position, X[4:8]   → key
 *   Window 1: X[8:12] → position, X[12:16] → key
 *   Window 2: X[16:20]→ position, X[20:24] → key
 *   X[24:31] unused
 * No mixing function — raw byte extraction. */

static inline uint32_t u32be(const unsigned char* b) {
    return ((uint32_t)b[0]<<24)|((uint32_t)b[1]<<16)|
    ((uint32_t)b[2]<<8)|(uint32_t)b[3];
}

/* Position within section for (section, xb) */
static inline size_t cpos(int sec, const unsigned char xb[32], size_t s) {
    uint32_t h = u32be(xb + sec * 8);
    return (size_t)((__uint128_t)h * (__uint128_t)s >> 32) + (size_t)sec * s;
}

/* Per-section fingerprint key */
static inline uint32_t ckey(int sec, const unsigned char xb[32]) {
    return u32be(xb + sec * 8 + 4);
}

/* ---- lookup (O(1), 3 probes + stash) ---- */

/*
 * map_get_all(): collect ALL candidates matching xb window keys
 * across all 3 cuckoo sections and the stash.
 *
 * Why this is necessary:
 *   Each section uses an independent 4-byte window key. A different xb'
 *   sharing the same window key (a "false positive") may occupy one of
 *   the 3 positions before the real entry.  If map_get() returned on the
 *   first match, verify_candidate() would reject the false positive and
 *   the real entry would be silently skipped.
 *
 *   False-positive probability per lookup: 1/2^32 ≈ 2.3e-10.
 *   Expected false positives per solve at 58-bit (~256M steps): ~18%.
 *   At 63-bit (~8B steps): virtually certain.
 *
 *   By collecting all matching candidates (at most 3 + stash), the caller
 *   can verify each one and correctly handle false positives.
 *
 * out[]  : caller-supplied array of at least (3 + CUCKOO_STASH_SZ) uint32_t
 * returns: number of candidates found (0 = definite miss)
 */
static int map_get_all(const cuckoo_map* m, const unsigned char xb[32],
                       uint32_t* out) {
    size_t s = m->section_size;
    int    n = 0;

    const entry_packed* e;
    e = &m->tab[cpos(0, xb, s)];
    if (e->val && e->key == ckey(0, xb)) out[n++] = e->val;
    e = &m->tab[cpos(1, xb, s)];
    if (e->val && e->key == ckey(1, xb)) out[n++] = e->val;
    e = &m->tab[cpos(2, xb, s)];
    if (e->val && e->key == ckey(2, xb)) out[n++] = e->val;

    for (int i = 0; i < m->stash_count; i++)
        if (memcmp(m->stash_xb[i], xb, 32) == 0) out[n++] = m->stash_val[i];

    return n;
}



/* ---- cuckoo insert into BUILD table ---- */

static int cuckoo_insert_build(build_entry* btab, size_t s,
                               cuckoo_map* m,
                               const unsigned char xb[32], uint32_t val) {
    build_entry cur;
    for (int k = 0; k < CUCKOO_K; k++)
        cur.pos[k] = (uint32_t)(cpos(k, xb, s) - (size_t)k * s);
    cur.val = val;
    int sec = 0;

    for (int iter = 0; iter < CUCKOO_MAX_RELOC; iter++) {
        size_t pos = (size_t)cur.pos[sec] + (size_t)sec * s;

        if (btab[pos].val == 0) {
            btab[pos] = cur;
            m->size++;
            return 1;
        }

        /* occupied — evict and continue */
        build_entry ev = btab[pos];
        btab[pos] = cur;
        cur = ev;
        sec = (sec + 1) % CUCKOO_K;
    }

    /* eviction chain too long — fall back to stash */
    if (m->stash_count < CUCKOO_STASH_SZ) {
        /* stash stores val only; keys filled in second pass */
        memset(m->stash_xb[m->stash_count], 0, 32);
        m->stash_val[m->stash_count] = cur.val;
        m->stash_count++;
        m->size++;
        return 1;
    }
    return 0;
}

/* ---- build then compact: 12-byte → 8-byte ---- */

/*
 * Builds the cuckoo table from n (x64, val) pairs fed one at a time
 * via repeated calls to cuckoo_insert_build(), then compacts the
 * 12-byte build table into the 8-byte lookup table.
 *
 * Caller fills the build_entry* before calling compact:
 *   btab allocated by cuckoo_alloc_build()
 *   entries inserted by cuckoo_insert_build()
 *   cuckoo_compact() finalises m->tab and frees btab
 */
static build_entry* cuckoo_alloc_build(cuckoo_map* m, size_t n) {
    memset(m, 0, sizeof(*m));
    /* section_size = ceil(1.3 × n / 3) — exact, no power-of-2 rounding */
    size_t s = (n * 13 + 29) / 30 + 2;
    size_t total = CUCKOO_K * s;

    build_entry* btab = (build_entry*)calloc(total, sizeof(build_entry));
    if (!btab) return NULL;

    m->section_size = s;
    m->total_bins   = total;
    return btab;
}

static int cuckoo_compact(cuckoo_map* m, build_entry* btab) {
    size_t total = m->total_bins;
    m->tab = (entry_packed*)calloc(total, sizeof(entry_packed));
    if (!m->tab) { free(btab); return 0; }

    /* Copy vals only — keys are filled in a second pass by re-walking i*G */
    for (size_t i = 0; i < total; i++)
        m->tab[i].val = btab[i].val;

    free(btab);
    return 1;
}

static void map_free(cuckoo_map* m) {
    free(m->tab);
    memset(m, 0, sizeof(*m));
}

/* ---------------- cached baby table ---------------- */

/* New magic to distinguish from previous open-addressing format */
#define BABY_MAGIC 0x4B43554B4F4F4355ULL  /* "UCOOKUCK" — cuckoo format */

typedef struct {
    uint64_t magic;
    uint32_t version;       /* 4 = cuckoo format */
    uint32_t l1;
    uint64_t section_size;  /* bins per section (not cap!) */
    uint64_t used_count;
    int32_t  stash_count;
    uint32_t _pad;
} baby_hdr;

static void baby_cache_path(char* out, size_t outlen, int l1) {
    snprintf(out, outlen, "bsgs_baby_cuckoo_secp256k1_l1_%d_window.bin", l1);
}

static int baby_save(const char* path, int l1, const cuckoo_map* baby) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror("baby_save open"); return 0; }

    baby_hdr hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic        = BABY_MAGIC;
    hdr.version      = 4;
    hdr.l1           = (uint32_t)l1;
    hdr.section_size = (uint64_t)baby->section_size;
    hdr.used_count   = (uint64_t)baby->size;
    hdr.stash_count  = (int32_t)baby->stash_count;

    int ok = write_all(fd, &hdr, sizeof(hdr));
    /* stash */
    ok &= write_all(fd, baby->stash_xb, sizeof(baby->stash_xb));
    ok &= write_all(fd, baby->stash_val, sizeof(baby->stash_val));
    /* main table */
    ok &= write_all(fd, baby->tab, baby->total_bins * sizeof(entry_packed));
    if (!ok) perror("baby_save write");
    close(fd); return ok;
}

static int baby_load(const char* path, int expected_l1, cuckoo_map* baby_out) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;

    baby_hdr hdr;
    if (!read_all(fd, &hdr, sizeof(hdr))) { close(fd); return 0; }

    if (hdr.magic   != BABY_MAGIC ||
        hdr.version != 4          ||
        (int)hdr.l1 != expected_l1 ||
        hdr.section_size == 0) { close(fd); return 0; }

    memset(baby_out, 0, sizeof(*baby_out));
    baby_out->section_size = (size_t)hdr.section_size;
    baby_out->total_bins   = CUCKOO_K * (size_t)hdr.section_size;
    baby_out->size         = (size_t)hdr.used_count;
    baby_out->stash_count  = (int)hdr.stash_count;

    if (!read_all(fd, baby_out->stash_xb, sizeof(baby_out->stash_xb)) ||
        !read_all(fd, baby_out->stash_val, sizeof(baby_out->stash_val))) {
        close(fd); return 0;
    }

    baby_out->tab = (entry_packed*)calloc(baby_out->total_bins, sizeof(entry_packed));
    if (!baby_out->tab) { close(fd); return 0; }

    if (!read_all(fd, baby_out->tab,
                  baby_out->total_bins * sizeof(entry_packed))) {
        free(baby_out->tab); baby_out->tab = NULL;
        close(fd); return 0;
    }

    close(fd); return 1;
}

/* ================================================================
 * Reusable BSGS context
 * ================================================================ */

typedef struct {
    secp256k1_context* ctx;
    int bits_total, l1;
    uint64_t M, Mhalf, J;

    secp256k1_pubkey G, MG;
    secp256k1_ge MG_ge;
    secp256k1_ge neg_MG_ge;

    uint64_t offset;
    secp256k1_pubkey neg_offsetG_pk;

    cuckoo_map baby;   /* ← cuckoo hash table */
} bsgs_ctx;

static int bsgs_ctx_init(bsgs_ctx* b, secp256k1_context* ctx,
                                int bits_total, int l1) {
    memset(b, 0, sizeof(*b));
    b->ctx = ctx; b->bits_total = bits_total; b->l1 = l1;
    if (bits_total <= 0 || bits_total > 63) return 0;
    if (l1 <= 0 || l1 >= bits_total)        return 0;

    b->M     = 1ULL << l1;
    b->Mhalf = 1ULL << (l1 - 1);
    b->J     = 1ULL << (bits_total - l1 - 1);

    unsigned char one[32] = {0}; one[31] = 1;
    if (!secp256k1_ec_pubkey_create(ctx, &b->G, one)) return 0;

    unsigned char Msc[32];
    u64_to_scalar32_be(b->M, Msc);
    if (!secp256k1_ec_pubkey_create(ctx, &b->MG, Msc)) return 0;

    pubkey_to_ge(&b->MG, &b->MG_ge);
    ge_negate(&b->neg_MG_ge, &b->MG_ge);
    
    b->offset = 1ULL << (bits_total - 1);
    unsigned char offset_c[32]; u64_to_scalar32_be(b->offset, offset_c);
    if (!secp256k1_ec_pubkey_create(ctx, &b->neg_offsetG_pk, offset_c)) return 0;
    secp256k1_ec_pubkey_negate(ctx, &b->neg_offsetG_pk);
    

    char path[128];
    baby_cache_path(path, sizeof(path), l1);
    if (file_exists(path)) {
        if (baby_load(path, l1, &b->baby)) {
            printf("Loaded cuckoo table: %s "
                   "(section=%zu, total=%zu, used=%zu, stash=%d)\n",
                   path, b->baby.section_size, b->baby.total_bins,
                   b->baby.size, b->baby.stash_count);
            printf("Table memory: %.2f MB\n",
                   (double)(b->baby.total_bins * sizeof(entry_packed)) / (1<<20));
            return 1;
        }
        printf("Cache exists but failed to load (rebuilding): %s\n", path);
    }

    /*
     * Build cuckoo table — two-pass approach:
     *
     * Pass 1: walk i*G, insert (positions, val) into 12-byte build table.
     *         No keys stored — keeps build memory at 12 bytes/entry.
     * Compact: allocate 8-byte packed table, copy vals only.
     * Pass 2: re-walk i*G, fill per-section keys into packed table.
     *         Also fill stash xb[] for exact stash matching.
     */
    size_t n = (size_t)(b->Mhalf);
    build_entry* btab = cuckoo_alloc_build(&b->baby, n);
    if (!btab) { fprintf(stderr, "cuckoo_alloc_build failed\n"); return 0; }

    printf("Building cuckoo table: n=%zu, section_size=%zu, total_bins=%zu\n"
           "  Build memory: %.2f MB (12-byte entries)\n",
           n, b->baby.section_size, b->baby.total_bins,
           (double)(b->baby.total_bins * sizeof(build_entry)) / (1<<20));

    secp256k1_pubkey cur = b->G;
    unsigned char ser[33];
    size_t s = b->baby.section_size;

    /* Pass 1: insert positions */
    for (uint64_t i = 1; i <= b->Mhalf; i++) {
        if (!pubkey_serialize33(ctx, &cur, ser)) {
            free(btab); map_free(&b->baby); return 0;
        }
        unsigned char xb[32];
        memcpy(xb, ser + 1, 32);

        if (!cuckoo_insert_build(btab, s, &b->baby, xb, (uint32_t)i)) {
            fprintf(stderr, "cuckoo insert failed at i=%llu\n",
                    (unsigned long long)i);
            free(btab); map_free(&b->baby); return 0;
        }

        if (i + 1 <= b->Mhalf) {
            const secp256k1_pubkey* pts[2] = { &cur, &b->G };
            secp256k1_pubkey nxt;
            if (!secp256k1_ec_pubkey_combine(ctx, &nxt, pts, 2)) {
                free(btab); map_free(&b->baby); return 0;
            }
            cur = nxt;
        }
    }

    printf("  Stash used: %d entries\n", b->baby.stash_count);

    /* Compact: vals only, keys to be filled in Pass 2 */
    if (!cuckoo_compact(&b->baby, btab)) {
        fprintf(stderr, "cuckoo_compact failed\n");
        map_free(&b->baby); return 0;
    }

    /* Pass 2: re-walk i*G, fill per-section keys into packed table */
    cur = b->G;
    for (uint64_t i = 1; i <= b->Mhalf; i++) {
        if (!pubkey_serialize33(ctx, &cur, ser)) {
            map_free(&b->baby); return 0;
        }
        unsigned char xb[32];
        memcpy(xb, ser + 1, 32);

        /* find which slot(s) hold this i and fill the key */
        for (int sec = 0; sec < CUCKOO_K; sec++) {
            size_t pos = cpos(sec, xb, s);
            if (b->baby.tab[pos].val == (uint32_t)i)
            b->baby.tab[pos].key = ckey(sec, xb);
        }
        /* fill stash xb if this i is in the stash */
        for (int si = 0; si < b->baby.stash_count; si++) {
            if (b->baby.stash_val[si] == (uint32_t)i)
            memcpy(b->baby.stash_xb[si], xb, 32);
        }

        if (i + 1 <= b->Mhalf) {
            const secp256k1_pubkey* pts[2] = { &cur, &b->G };
            secp256k1_pubkey nxt;
            if (!secp256k1_ec_pubkey_combine(ctx, &nxt, pts, 2)) {
                map_free(&b->baby); return 0;
            }
            cur = nxt;
        }
    }

    printf("  Compacted to %.2f MB (8-byte entries)\n",
           (double)(b->baby.total_bins * sizeof(entry_packed)) / (1<<20));

    if (!baby_save(path, l1, &b->baby))
        printf("Warning: failed to save cache: %s\n", path);
    else
        printf("Saved cuckoo cache: %s\n", path);

    return 1;
}

static void bsgs_ctx_free(bsgs_ctx* b) {
    map_free(&b->baby); memset(b, 0, sizeof(*b));
}

/* Adapted from the original verify_candidate function for signed m_cand */
static int verify_candidate(const secp256k1_context* ctx,
                            int64_t m_cand,
                            const unsigned char target33[33]) {
    if(m_cand == 0) return 0;
    if(m_cand < 0) {
        // Handle negative candidate by computing -m_cand * G
        unsigned char sc[32];
        u64_to_scalar32_be((uint64_t)(-m_cand), sc);
        secp256k1_pubkey pk;
        if (!secp256k1_ec_pubkey_create(ctx, &pk, sc)) return 0;
        secp256k1_ec_pubkey_negate(ctx, &pk); // Negate the public key
        unsigned char got[33]; size_t glen = 33;
        secp256k1_ec_pubkey_serialize(ctx, got, &glen, &pk, SECP256K1_EC_COMPRESSED);
        return memcmp(got, target33, 33) == 0;
    } else {
        unsigned char sc[32]; u64_to_scalar32_be(m_cand, sc);
        secp256k1_pubkey pk;
        if (!secp256k1_ec_pubkey_create(ctx, &pk, sc)) return 0;
        unsigned char got[33]; size_t glen = 33;
        secp256k1_ec_pubkey_serialize(ctx, got, &glen, &pk, SECP256K1_EC_COMPRESSED);
        return memcmp(got, target33, 33) == 0;
    }
}

/* ================================================================
 * Profiling globals — measure time in each phase
 * ================================================================ */
static double g_time_arithmetic = 0.0;  /* zaddsub point advancement */
static double g_time_inversion   = 0.0;  /* batch inversion           */
static double g_time_lookup      = 0.0;  /* cuckoo baby table lookup  */

static inline double now_prof(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ================================================================
 * Single-threaded solve — Windowed TreeMon + Cuckoo lookup
 * ================================================================
 *
 * 3-phase windowed loop:
 *   Phase 1: accumulate W Jacobian Q points, check i=0 — 0 inv
 *   Phase 2: batch invert W Z-coords via TreeMon  — 1 inv total
 *   Phase 3: extract x64 from z_inv, cuckoo lookup — 0 inv
 *
 *   Inversions per step: 1/W  (vs 1 before TreeMon)
 */
static inline void process_window_single_thread( size_t w_count,
                                   uint64_t j_start,
                                   secp256k1_gej * restrict Qj_add, 
                                   secp256k1_gej * restrict Qj_sub,
                                   const bsgs_ctx * restrict b,
                                   const unsigned char * restrict target33,
                                   int * restrict found,
                                   int64_t * restrict out_m,
                                   secp256k1_fe * restrict x_add,
                                   secp256k1_fe * restrict x_sub,
                                   uint64_t * restrict j_win,
                                   secp256k1_fe * restrict z_c,
                                   secp256k1_fe * restrict z_inv,
                                   secp256k1_fe * restrict scratch)
{

    for (size_t w = 0; w < w_count; w++) {
        if (EXPECT(secp256k1_gej_is_infinity(Qj_sub), 0)) {
            *out_m = (j_start + (uint64_t)w) * b->M;
            *found = 1;
            return;                                   /* skip invert/lookup entirely */
        }
        if (EXPECT(secp256k1_gej_is_infinity(Qj_add), 0)) {
            *out_m = -(int64_t)((j_start + (uint64_t)w) * b->M);
            *found = 1;
            return;                                   /* skip invert/lookup entirely */
        }

        x_sub[w] = Qj_sub->x; x_add[w] = Qj_add->x; z_c[w] = Qj_add->z;
        j_win[w] = j_start + (uint64_t)w;

        /* Phase 1: Co-Z arithmetic */
        double _t0 = now_prof();
        secp256k1_gej_zaddsub_ge_var(Qj_add, Qj_sub, Qj_add, Qj_sub, &b->MG_ge);
        g_time_arithmetic += now_prof() - _t0;
    }

    /* Phase 2: batch inversion */
    double _t1 = now_prof();
    fe_batch_invert_skewed_tree(z_inv, z_c, scratch, w_count);
    g_time_inversion += now_prof() - _t1;

    /* Phase 3: lookup */
    double _t2 = now_prof();
    for (size_t w = 0; w < w_count; w++) {
        secp256k1_fe z_inv_sq;
        secp256k1_fe_sqr(&z_inv_sq, &z_inv[w]);

        unsigned char qxb[32];
        fe_xb32_from_zinv_sq(&x_sub[w], &z_inv_sq, qxb);
        uint32_t cands[3 + CUCKOO_STASH_SZ];
        int nc = map_get_all(&b->baby, qxb, cands);
        for (int ci = 0; ci < nc; ci++) {
            int64_t m1 = (int64_t)((int64_t)(j_win[w] * b->M) + (uint64_t)cands[ci]);
            int64_t m2 = (int64_t)((int64_t)(j_win[w] * b->M) - (uint64_t)cands[ci]);
            if (EXPECT(verify_candidate(b->ctx, m1, target33), 0)) { *out_m = m1; *found = 1; return; }
            else if (EXPECT(verify_candidate(b->ctx, m2, target33), 0)) { *out_m = m2; *found = 1; return; }
        }

        fe_xb32_from_zinv_sq(&x_add[w], &z_inv_sq, qxb);
        nc = map_get_all(&b->baby, qxb, cands);
        for (int ci = 0; ci < nc; ci++) {
            int64_t m1 = (int64_t)((-(int64_t)(j_win[w] * b->M)) + (uint64_t)cands[ci]);
            int64_t m2 = (int64_t)((-(int64_t)(j_win[w] * b->M)) - (uint64_t)cands[ci]);
            if (EXPECT(verify_candidate(b->ctx, m1, target33), 0)) { *out_m = m1; *found = 1; g_time_lookup += now_prof() - _t2; return; }
            else if (EXPECT(verify_candidate(b->ctx, m2, target33), 0)) { *out_m = m2; *found = 1; g_time_lookup += now_prof() - _t2; return; }
        }
    }
    g_time_lookup += now_prof() - _t2;
}

static int bsgs_solve(const bsgs_ctx* b,
                      const secp256k1_pubkey* targetPm,
                      int window,
                      int64_t* out_m) {

    unsigned char t33[33];
    if (!pubkey_serialize33(b->ctx, targetPm, t33)) return 0;

    /* j=0: direct baby lookup for Pm = i.G */
    unsigned char txb[32];
    memcpy(txb, t33 + 1, 32); /* x-coordinate from compressed pubkey */
    uint32_t cands[3 + CUCKOO_STASH_SZ];
    int nc = map_get_all(&b->baby, txb, cands);
    for (int ci = 0; ci < nc; ci++) {
        if (verify_candidate(b->ctx, (int64_t)((uint64_t)cands[ci]), t33)) {
            *out_m = (int64_t)(uint64_t)cands[ci]; return 1;
        } else if (verify_candidate(b->ctx, -(int64_t)((uint64_t)cands[ci]), t33)) {
            *out_m = -(int64_t)(uint64_t)cands[ci]; return 1;
        }
    }

    size_t W = (size_t)window;
    secp256k1_fe*  x_add   = (secp256k1_fe*) malloc(W * sizeof(secp256k1_fe));
    secp256k1_fe*  x_sub   = (secp256k1_fe*) malloc(W * sizeof(secp256k1_fe));
    secp256k1_fe*  z_c     = (secp256k1_fe*) malloc(W * sizeof(secp256k1_fe));
    uint64_t*      j_win   = (uint64_t*)     malloc(W * sizeof(uint64_t));
    secp256k1_fe*  z_inv   = (secp256k1_fe*) malloc(W * sizeof(secp256k1_fe));
    secp256k1_fe*  scratch = (secp256k1_fe*) malloc(W * sizeof(secp256k1_fe));

    if (!x_add || !x_sub || !z_c || !j_win || !z_inv || !scratch) {
        free(x_add); free(x_sub); free(z_c); free(j_win); free(z_inv); free(scratch);
        return 0;
    }

    uint64_t j = 1;
    uint64_t total_steps = b->J;
    uint64_t full_loops  = total_steps / (uint64_t)W;
    size_t   rem         = (size_t)(total_steps % (uint64_t)W);

    secp256k1_gej Qj_add, Qj_sub;
    pubkey_to_gej(targetPm, &Qj_add);
    Qj_sub = Qj_add;
        
    secp256k1_gej_zaddsub_ge_var(&Qj_add, &Qj_sub, &Qj_add, &Qj_sub, &b->MG_ge);

    int found = 0;

    /* ---- windowed main loop ---- */
    for (uint64_t l = 0; l < full_loops; l++) {
        if (found) break;
        process_window_single_thread(W, j, &Qj_add, &Qj_sub, b, t33, &found, out_m, x_add, x_sub, j_win, z_c, z_inv, scratch);
        j += (uint64_t)W;
    }
    /* ---- remaining steps (< W) ---- */
    if (rem > 0 && !found) {
        process_window_single_thread(rem, j, &Qj_add, &Qj_sub, b, t33, &found, out_m, x_add, x_sub, j_win, z_c, z_inv, scratch);
        j += (uint64_t)rem;
    }

    free(x_add); free(x_sub); free(z_c); free(j_win); free(z_inv); free(scratch);
    return found;
}

/* ================================================================
 * Parallel solve — Windowed TreeMon
 * ================================================================ */

typedef struct {
    const bsgs_ctx*  b;
    secp256k1_pubkey targetPm;
    unsigned char    target33[33];
    secp256k1_ge     target_ge;
    uint64_t         j_start, j_end;
    int              window;
    atomic_int*      found;
    int64_t*        found_m;
    pthread_mutex_t* found_mu;
} bsgs_worker_args;


static inline void process_window(secp256k1_context *ctx, size_t w_count,
                                   uint64_t j_start, 
                                   secp256k1_gej * restrict Qj_add, 
                                   secp256k1_gej * restrict Qj_sub,
                                   const bsgs_ctx * restrict b,
                                   bsgs_worker_args * restrict a,
                                   secp256k1_fe * restrict x_add,
                                   secp256k1_fe * restrict x_sub,
                                   uint64_t * restrict j_win,
                                   secp256k1_fe * restrict z_c,
                                   secp256k1_fe * restrict z_inv,
                                   secp256k1_fe * restrict scratch)
{

    for (size_t w = 0; w < w_count; w++) {
        if (EXPECT(secp256k1_gej_is_infinity(Qj_sub), 0)) {
            int64_t m = (j_start + (uint64_t)w) * b->M;
            pthread_mutex_lock(a->found_mu);
            if (!atomic_load_explicit(a->found, memory_order_relaxed)) {
                *a->found_m = m;
                atomic_store_explicit(a->found, 1, memory_order_relaxed);
            }
            pthread_mutex_unlock(a->found_mu);
            return;                                   /* skip invert/lookup entirely */
        }
        if (EXPECT(secp256k1_gej_is_infinity(Qj_add), 0)) {
            int64_t m = -(int64_t)((j_start + (uint64_t)w) * b->M);
            pthread_mutex_lock(a->found_mu);
            if (!atomic_load_explicit(a->found, memory_order_relaxed)) {
                *a->found_m = m;
                atomic_store_explicit(a->found, 1, memory_order_relaxed);
            }
            pthread_mutex_unlock(a->found_mu);
            return;                                   /* skip invert/lookup entirely */
        }

        x_sub[w] = Qj_sub->x; x_add[w] = Qj_add->x; z_c[w] = Qj_add->z;
        j_win[w] = j_start + (uint64_t)w;
            
        secp256k1_gej_zaddsub_ge_var(Qj_add, Qj_sub, Qj_add, Qj_sub, &b->MG_ge);
    }

    fe_batch_invert_skewed_tree(z_inv, z_c, scratch, w_count);

    for (size_t w = 0; w < w_count; w++) {
        if (EXPECT(atomic_load_explicit(a->found, memory_order_relaxed), 0)) return;

        secp256k1_fe z_inv_sq;
        secp256k1_fe_sqr(&z_inv_sq, &z_inv[w]);
        
        unsigned char qxb[32];
        fe_xb32_from_zinv_sq(&x_sub[w], &z_inv_sq, qxb);
        uint32_t cands[3 + CUCKOO_STASH_SZ];
        int nc = map_get_all(&b->baby, qxb, cands);
        for (int ci = 0; ci < nc; ci++) {
            int64_t m1 = (int64_t)((int64_t)(j_win[w] * b->M) + (uint64_t)cands[ci]);
            int64_t m2 = (int64_t)((int64_t)(j_win[w] * b->M) - (uint64_t)cands[ci]);
            int64_t m_ok; int got = 0;
            if (verify_candidate(ctx, m1, a->target33)) { m_ok = m1; got = 1; }
            else if (verify_candidate(ctx, m2, a->target33)) { m_ok = m2; got = 1; }
            if (EXPECT(got, 0)) {
                pthread_mutex_lock(a->found_mu);
                if (!atomic_load_explicit(a->found, memory_order_relaxed)) {
                    *a->found_m = m_ok;
                    atomic_store_explicit(a->found, 1, memory_order_relaxed);
                }
                pthread_mutex_unlock(a->found_mu);
                return;
            }
        }

        fe_xb32_from_zinv_sq(&x_add[w], &z_inv_sq, qxb);
        nc = map_get_all(&b->baby, qxb, cands);
        for (int ci = 0; ci < nc; ci++) {
            int64_t m1 = (int64_t)((-(int64_t)(j_win[w] * b->M)) + (uint64_t)cands[ci]);
            int64_t m2 = (int64_t)((-(int64_t)(j_win[w] * b->M)) - (uint64_t)cands[ci]);
            int64_t m_ok; int got = 0;
            if (verify_candidate(ctx, m1, a->target33)) { m_ok = m1; got = 1; }
            else if (verify_candidate(ctx, m2, a->target33)) { m_ok = m2; got = 1; }
            if (EXPECT(got, 0)) {
                pthread_mutex_lock(a->found_mu);
                if (!atomic_load_explicit(a->found, memory_order_relaxed)) {
                    *a->found_m = m_ok;
                    atomic_store_explicit(a->found, 1, memory_order_relaxed);
                }
                pthread_mutex_unlock(a->found_mu);
                return;
            }
        }
    }
}

static void* bsgs_worker_thread(void* argp) {
    bsgs_worker_args* a = (bsgs_worker_args*)argp;
    const bsgs_ctx*   b = a->b;
    size_t            W = (size_t)a->window;

    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_VERIFY);
    if (!ctx) return NULL;


    secp256k1_fe*  x_add   = (secp256k1_fe*) malloc(W * sizeof(secp256k1_fe));
    secp256k1_fe*  x_sub   = (secp256k1_fe*) malloc(W * sizeof(secp256k1_fe));
    secp256k1_fe*  z_c     = (secp256k1_fe*) malloc(W * sizeof(secp256k1_fe));
    uint64_t*      j_win   = (uint64_t*)     malloc(W * sizeof(uint64_t));
    secp256k1_fe*  z_inv   = (secp256k1_fe*) malloc(W * sizeof(secp256k1_fe));
    secp256k1_fe*  scratch = (secp256k1_fe*) malloc(W * sizeof(secp256k1_fe));

    if (!x_add || !x_sub || !z_c || !j_win || !z_inv || !scratch) {
        free(x_add); free(x_sub); free(z_c); free(j_win); free(z_inv); free(scratch);
        secp256k1_context_destroy(ctx); return NULL;
    }

    uint64_t j = a->j_start;
    uint64_t total_steps = (a->j_end > j) ? (a->j_end - j) : 0;
    uint64_t full_loops  = total_steps / (uint64_t)W;
    size_t   rem         = (size_t)(total_steps % (uint64_t)W);

    unsigned char jm_sc[32];
    u64_to_scalar32_be(j * b->M, jm_sc);
    secp256k1_pubkey jMG_pub;
    if (!secp256k1_ec_pubkey_create(ctx, &jMG_pub, jm_sc)) {
        free(x_add); free(x_sub); free(z_c); free(j_win); free(z_inv); free(scratch);
        secp256k1_context_destroy(ctx); return NULL;
    }

    secp256k1_ge jMG_ge;
    pubkey_to_ge(&jMG_pub, &jMG_ge);

    secp256k1_gej Qj_add, Qj_sub;
    pubkey_to_gej(&a->targetPm, &Qj_add);
    Qj_sub = Qj_add;
        
    secp256k1_gej_zaddsub_ge_var(&Qj_add, &Qj_sub, &Qj_add, &Qj_sub, &jMG_ge);

    /* ---- windowed main loop ---- */
    for (uint64_t l = 0; l < full_loops; l++) {
        if (atomic_load_explicit(a->found, memory_order_relaxed)) break;
        process_window(ctx, W, j, &Qj_add, &Qj_sub, b, a, x_add, x_sub, j_win, z_c, z_inv, scratch);
        j += (uint64_t)W;
    }
    /* ---- remaining steps (< W) ---- */
    if (rem > 0 && !atomic_load_explicit(a->found, memory_order_relaxed)) {
        process_window(ctx, rem, j, &Qj_add, &Qj_sub, b, a, x_add, x_sub, j_win, z_c, z_inv, scratch);
        j += (uint64_t)rem;
    }

    free(x_add); free(x_sub); free(z_c); free(j_win); free(z_inv); free(scratch);
    secp256k1_context_destroy(ctx);
    return NULL;
}

static int bsgs_solve_zaddsub_parallel(const bsgs_ctx* b,
                               const secp256k1_pubkey* targetPm,
                               int threads, int window,
                               int64_t* out_m) {
    if (threads < 1) threads = 1;
    if (threads == 1) return bsgs_solve(b, targetPm, window, out_m);

    unsigned char t33[33];
    if (!pubkey_serialize33(b->ctx, targetPm, t33)) return 0;

    /* j=0: direct baby lookup for Pm = i.G */
    unsigned char txb[32];
    memcpy(txb, t33 + 1, 32); /* x-coordinate from compressed pubkey */
    uint32_t cands[3 + CUCKOO_STASH_SZ];
    int nc = map_get_all(&b->baby, txb, cands);
    for (int ci = 0; ci < nc; ci++) {
        if (verify_candidate(b->ctx, (int64_t)((uint64_t)cands[ci]), t33)) {
            *out_m = (int64_t)(uint64_t)cands[ci]; return 1;
        } else if (verify_candidate(b->ctx, -(int64_t)((uint64_t)cands[ci]), t33)) {
            *out_m = -(int64_t)(uint64_t)cands[ci]; return 1;
        }
    }

    uint64_t J = b->J;
    uint64_t chunk = (J + (uint64_t)threads - 1)/(uint64_t)threads;
    pthread_t*        tids = (pthread_t*)       calloc((size_t)threads, sizeof(pthread_t));
    bsgs_worker_args* args = (bsgs_worker_args*)calloc((size_t)threads, sizeof(bsgs_worker_args));
    if (!tids || !args) { free(tids); free(args); return 0; }

    atomic_int     found   = 0;
    int64_t        found_m = 0;
    pthread_mutex_t found_mu;
    pthread_mutex_init(&found_mu, NULL);

    secp256k1_ge target_ge;
    pubkey_to_ge(targetPm, &target_ge);

    uint64_t jcur = 1;
    for (int t = 0; t < threads; t++) {
        args[t].b         = b;
        args[t].targetPm  = *targetPm;
        memcpy(args[t].target33, t33, 33);
        args[t].target_ge = target_ge;
        args[t].j_start   = jcur;
        args[t].j_end     = jcur + chunk < J + 1 ?
                            jcur + chunk : J + 1 ; /* j_end is not included */
        args[t].window    = window;
        args[t].found     = &found;
        args[t].found_m   = &found_m;
        args[t].found_mu  = &found_mu;
        pthread_create(&tids[t], NULL, bsgs_worker_thread, &args[t]);
        jcur += chunk;
    }

    for (int t = 0; t < threads; t++) pthread_join(tids[t], NULL);

    int ok = atomic_load_explicit(&found, memory_order_relaxed);
    if (ok) *out_m = found_m;

    pthread_mutex_destroy(&found_mu);
    free(tids); free(args);
    return ok;
}

/* ─────────────────────── Wrapper ─────────────────────────────── */
/*
 * Wrapper allowing the recovery of an unsigned message m in [0, 2^l) 
 * using `bsgs_solve_zaddsub_parallel`, which operates in [-2^(l-1), 2^(l-1)). 
 */
static int bsgs_solve_zaddsub_parallel_wrapper(const bsgs_ctx* restrict b,
                           const secp256k1_pubkey* restrict targetPm,
                           int threads, int window,
                           uint64_t* out_m) {

    /* Compute Pm' = Pm - 2^(bits_total-1).G = Pm + (-offset.G) */
    secp256k1_pubkey Pm_adj_pk;  
    const secp256k1_pubkey *points[2] = { targetPm, &b->neg_offsetG_pk };
    secp256k1_ec_pubkey_combine(b->ctx, &Pm_adj_pk, points, 2);
    
    int found;
    int64_t out_m_signed = 0;
    found = bsgs_solve_zaddsub_parallel(b,&Pm_adj_pk,threads,window,&out_m_signed);
    *out_m = out_m_signed + b->offset;
    return found;
}




/* ================================================================
 * Benchmark
 * ================================================================ */

static void benchmark_bsgs(int bits_total, int l1, int trials, int threads,
                           int window, unsigned int seed) {
    
    if (bits_total == 0 || bits_total > 64) {
        fprintf(stderr,"Error: invalid bits_total value: %i\n", bits_total);
        return;
    }

    if (trials <= 0) {
        fprintf(stderr, "Invalid trials: %d\n", trials);
        return;
    }

    printf("=== BSGS (secp256k1, cuckoo k=3, Jacobian, windowed TreeMon) ===\n");
    printf("Range  : m in [0, 2^%d)\n", bits_total);
    printf("Split  : l1=%d, l2=%d\n", l1, bits_total - l1);
    printf("Window : W=%d\n", window);
    printf("Entry  : %zu bytes lookup | Trials: %d | Threads: %d\n\n",
           sizeof(entry_packed), trials, threads);

    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_VERIFY);
    if (!ctx) { printf("Failed to create context\n"); return; }

    bsgs_ctx solver;
    double t0 = now_seconds();
    if (!bsgs_ctx_init(&solver, ctx, bits_total, l1)) {
        printf("Failed to init solver\n");
        secp256k1_context_destroy(ctx); return;
    }
    double t1 = now_seconds();
    printf("Table : %.2f MB | Init: %.6f sec\n\n",
           (double)(solver.baby.total_bins * sizeof(entry_packed)) / (1 << 20), t1 - t0);

    /* Run trials */
    uint64_t mask = (bits_total == 64) ? ~0ULL : ((1ULL << bits_total) - 1ULL);
    uint64_t m_Max = mask;
    int ok = 0;
    double ts = 0;
    double recovery_time = 0;
    
    for (int t = 0; t < trials; t++) {
        uint64_t m;

        /* seed = 0: benchmark the worst case (m = 2^l - 1) */
        if (seed){
            m = ((uint64_t)rand() << 32) ^ (uint64_t)rand();
            m &= mask; if (m == 0) m = 1;
        }else{
            m = m_Max;
        }
        unsigned char sc[32]; u64_to_scalar32_be(m, sc);
        secp256k1_pubkey Pm_pk;
        if (!secp256k1_ec_pubkey_create(ctx, &Pm_pk, sc)) { printf("create failed\n"); break; }

        uint64_t recovered = 0;

        ts = now_seconds();
        int found = bsgs_solve_zaddsub_parallel_wrapper(&solver, &Pm_pk, threads, window, &recovered);
        recovery_time += now_seconds() - ts;
        
        if ( found && recovered == m)
            ok++;
        else
            printf("Trial %d FAILED: m=%"PRIu64" recovered=%"PRIu64"\n", t, m, recovered);        
    }

    printf("Random Seed      : %u\n", seed);
    printf("Solved correctly : %d/%d\n", ok, trials);
    printf("Total search time: %.3f sec\n", recovery_time);
    printf("Average per solve: %.3f sec (%.2f ms)\n\n",
           recovery_time/trials,(recovery_time/trials)*1e3);

    /* Profiling breakdown */
    double total_prof = g_time_arithmetic + g_time_inversion + g_time_lookup;
    printf("=== Phase Timing Breakdown ===\n");
    printf("Co-Z arithmetic  : %.3f sec (%5.1f%%)\n",
           g_time_arithmetic, 100.0 * g_time_arithmetic / total_prof);
    printf("Batch inversion  : %.3f sec (%5.1f%%)\n",
           g_time_inversion,  100.0 * g_time_inversion  / total_prof);
    printf("Baby table lookup: %.3f sec (%5.1f%%)\n",
           g_time_lookup,     100.0 * g_time_lookup     / total_prof);
    printf("Total measured   : %.3f sec\n\n", total_prof);
    bsgs_ctx_free(&solver);
    secp256k1_context_destroy(ctx);
}

int main(int argc, char** argv) {
    int      bits_total = 40, l1 = 18, trials = 1, threads = 1, window = 64;
    unsigned int seed = (unsigned int)time(NULL);

    if (argc >= 2) bits_total = atoi(argv[1]);
    if (argc >= 3) l1         = atoi(argv[2]);
    if (argc >= 4) trials     = atoi(argv[3]);
    if (argc >= 5) threads    = atoi(argv[4]);
    if (argc >= 6) window     = atoi(argv[5]);
    if (argc >= 7) {
        /* seed = 0: benchmark the worst case (m = 2^l - 1) */
        seed = (unsigned int)strtoull(argv[6], NULL, 10);
    }

    /* Window must be a power of 2 >= 1 */
    if (window < 1) window = 1;
    { int w = 1; while (w < window) w <<= 1; window = w; }

    srand(seed);
    benchmark_bsgs(bits_total, l1, trials, threads, window, seed);
    return 0;
}