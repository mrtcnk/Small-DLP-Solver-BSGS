#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <inttypes.h>
#include <errno.h>

#include <secp256k1.h>

/*
 * secp256k1 internal headers for Jacobian arithmetic.
 *
 * Clone secp256k1 source alongside this file:
 *   git clone https://github.com/bitcoin-core/secp256k1 secp256k1_src
 *
 * Build:
 *   cc -O3 -Wall -Wextra -o bsgs bsgs_dlp_benchmark_cached.c \
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

#define secp256k1_fe_equal_var(a,b) (secp256k1_fe_cmp_var((a),(b)) == 0)
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

/* Extract the first 8 bytes of the x-coordinate as a uint64 (big-endian).
 * Used as the 64-bit truncated key for the hash table. */
static uint64_t ge_x64(const secp256k1_ge* ge) {
    secp256k1_fe x = ge->x;
    secp256k1_fe_normalize_var(&x);
    unsigned char buf[32];
    secp256k1_fe_get_b32(buf, &x);
    uint64_t h = 0;
    for (int i = 0; i < 8; i++) h = (h << 8) | (uint64_t)buf[i];
    return h;
}

/*
 * Check Jacobian point a == affine point b WITHOUT any inversion.
 *
 * a = (X:Y:Z) in Jacobian, b = (x,y) in affine.
 * Equal iff: X == x * Z^2  AND  Y == y * Z^3  (mod p)
 *
 * Cost: 1 sqr + 2 mul + 2 normalise + 2 compare.
 */
static int gej_eq_ge(const secp256k1_gej* a, const secp256k1_ge* b) {
    secp256k1_fe z2, z3, u, s, ax, ay;
    if (a->infinity) return b->infinity;
    if (b->infinity) return 0;

    secp256k1_fe_sqr(&z2, &a->z);
    secp256k1_fe_mul(&z3, &z2, &a->z);

    secp256k1_fe_mul(&u, &b->x, &z2);
    ax = a->x;
    secp256k1_fe_normalize_var(&ax);
    secp256k1_fe_normalize_var(&u);
    if (!secp256k1_fe_equal_var(&ax, &u)) return 0;

    secp256k1_fe_mul(&s, &b->y, &z3);
    ay = a->y;
    secp256k1_fe_normalize_var(&ay);
    secp256k1_fe_normalize_var(&s);
    return secp256k1_fe_equal_var(&ay, &s);
}

/* ================================================================
 * Windowed TreeMon batch inversion helpers
 * ================================================================
 *
 * fe_batch_invert_tree(): inverts n field elements using
 *   1 inversion + 3(n-1) multiplications.
 *   n must be a power of 2. bt1/bt2 are level-order binary trees
 *   of size 2n (bt1[n..2n-1] = input; bt2[n..2n-1] = 1/input).
 *
 * gej_x64_from_zinv(): extract upper-64-bit affine x from
 *   a Jacobian point given its precomputed Z-inverse.
 *   No inversion — 1 sqr + 1 mul.
 */

static void fe_batch_invert_tree(secp256k1_fe* bt1,
                                 secp256k1_fe* bt2,
                                 size_t n) {
    for (size_t i = n - 1; i >= 1; i--)
        secp256k1_fe_mul(&bt1[i], &bt1[2*i], &bt1[2*i+1]);
    secp256k1_fe_inv(&bt2[1], &bt1[1]);
    for (size_t i = 1; i < n; i++) {
        secp256k1_fe_mul(&bt2[2*i],   &bt1[2*i+1], &bt2[i]);
        secp256k1_fe_mul(&bt2[2*i+1], &bt1[2*i],   &bt2[i]);
    }
}

static uint64_t gej_x64_from_zinv(const secp256k1_gej* pt,
                                  const secp256k1_fe*  z_inv) {
    secp256k1_fe z2, x;
    secp256k1_fe_sqr(&z2, z_inv);
    secp256k1_fe_mul(&x, &pt->x, &z2);
    secp256k1_fe_normalize_var(&x);
    unsigned char buf[32];
    secp256k1_fe_get_b32(buf, &x);
    uint64_t h = 0;
    for (int k = 0; k < 8; k++) h = (h << 8) | (uint64_t)buf[k];
    return h;
}

/* ================================================================
 * Cuckoo Hashing — k=3 sections, load factor ~1.3×
 * ================================================================
 *
 * Design:
 *   - 3 hash functions map x64 into 3 disjoint sections of size
 *     section_size ≈ 1.3n/3 each.  Total bins ≈ 1.3n.
 *   - Lookup: O(1) worst-case — exactly 3 probes (+ tiny stash).
 *   - Build: uses 12-byte build_entry (full x64 + val) for eviction,
 *     then compacts to 8-byte entry_packed (upper32(x64) + val).
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
#define CUCKOO_SEED1      0x9e3779b97f4a7c15ULL
#define CUCKOO_SEED2      0xd1b54a32d192ed03ULL

/* ---- types ---- */

/* 8-byte lookup entry — kept in memory after build */
typedef struct {
    uint32_t key;   /* upper 32 bits of x64 — discriminator */
    uint32_t val;   /* i value; 0 = empty                   */
} entry_packed;

/* 12-byte build entry — used during cuckoo construction only */
typedef struct __attribute__((packed)) {
    uint64_t x64;   /* full key (needed for eviction rehashing) */
    uint32_t val;   /* i value; 0 = empty                       */
} build_entry;

/* cuckoo map (lookup phase, after compaction) */
typedef struct {
    entry_packed* tab;          /* 3 × section_size entries        */
    size_t        section_size; /* bins per section (not pow2)     */
    size_t        total_bins;   /* 3 × section_size                */
    size_t        size;         /* entries successfully inserted    */
    /* overflow stash (rare: ~0 entries with load 1.3×, k=3) */
    uint64_t stash_x64[CUCKOO_STASH_SZ];
    uint32_t stash_val[CUCKOO_STASH_SZ];
    int      stash_count;
} cuckoo_map;

/* ---- helpers ---- */

static uint64_t mix64(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31; return x;
}

/* Fast uniform mapping to [0, n) — no power-of-2 required */
static inline size_t fastrange64(uint64_t h, size_t n) {
return (size_t)((__uint128_t)h * (__uint128_t)n >> 64);
}

/* Position within the flat array for (section, x64) */
static inline size_t cpos(int sec, uint64_t x64, size_t s) {
    uint64_t seed = (sec == 0) ? 0ULL :
                    (sec == 1) ? CUCKOO_SEED1 : CUCKOO_SEED2;
    return fastrange64(mix64(x64 ^ seed), s) + (size_t)sec * s;
}

/* ---- lookup (O(1), 3 probes + stash) ---- */

/*
 * map_get_all(): collect ALL candidates matching x64's upper-32-bit key
 * across all 3 cuckoo sections and the stash.
 *
 * Why this is necessary:
 *   The stored key is only the upper 32 bits of x64.  A different x64'
 *   with the same upper 32 bits (a "false positive") may occupy one of
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
static int map_get_all(const cuckoo_map* m, uint64_t x64,
                       uint32_t* out) {
    uint32_t k = (uint32_t)(x64 >> 32);
    size_t   s = m->section_size;
    int      n = 0;

    const entry_packed* e;
    e = &m->tab[cpos(0, x64, s)];
    if (e->val && e->key == k) out[n++] = e->val;
    e = &m->tab[cpos(1, x64, s)];
    if (e->val && e->key == k) out[n++] = e->val;
    e = &m->tab[cpos(2, x64, s)];
    if (e->val && e->key == k) out[n++] = e->val;

    for (int i = 0; i < m->stash_count; i++)
        if (m->stash_x64[i] == x64) out[n++] = m->stash_val[i];

    return n;
}

/* Convenience wrapper: returns first verified candidate via ±i check.
 * Use map_get_all() directly in hot loops for efficiency. */
static int map_get(const cuckoo_map* m, uint64_t x64, uint32_t* out) {
    uint32_t cands[3 + CUCKOO_STASH_SZ];
    int nc = map_get_all(m, x64, cands);
    if (!nc) return 0;
    *out = cands[0];   /* caller must verify all candidates if nc > 1 */
    return 1;
}

/* ---- cuckoo insert into BUILD table ---- */

static int cuckoo_insert_build(build_entry* btab, size_t s,
                               cuckoo_map* m,
                               uint64_t x64, uint32_t val) {
    uint64_t cur_x64 = x64;
    uint32_t cur_val = val;
    int sec = 0;

    for (int iter = 0; iter < CUCKOO_MAX_RELOC; iter++) {
        size_t pos = cpos(sec, cur_x64, s);

        if (btab[pos].val == 0) {
            /* empty slot — insert */
            btab[pos].x64 = cur_x64;
            btab[pos].val = cur_val;
            m->size++;
            return 1;
        }

        /* occupied — evict and continue */
        uint64_t ev_x64 = btab[pos].x64;
        uint32_t ev_val = btab[pos].val;
        btab[pos].x64   = cur_x64;
        btab[pos].val   = cur_val;
        cur_x64 = ev_x64;
        cur_val = ev_val;
        sec = (sec + 1) % CUCKOO_K;
    }

    /* eviction chain too long — fall back to stash */
    if (m->stash_count < CUCKOO_STASH_SZ) {
        m->stash_x64[m->stash_count] = cur_x64;
        m->stash_val[m->stash_count] = cur_val;
        m->stash_count++;
        m->size++;
        return 1;
    }
    return 0; /* stash full — should not happen with 1.3× load, k=3 */
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

    for (size_t i = 0; i < total; i++) {
        m->tab[i].key = (uint32_t)(btab[i].x64 >> 32);
        m->tab[i].val = btab[i].val;
    }
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
    snprintf(out, outlen, "bsgs_baby_cuckoo_secp256k1_l1_%d.bin", l1);
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
    ok &= write_all(fd, baby->stash_x64, sizeof(baby->stash_x64));
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

    if (!read_all(fd, baby_out->stash_x64, sizeof(baby_out->stash_x64)) ||
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

    cuckoo_map baby;   /* ← cuckoo hash table */
} bsgs_ctx;

static int bsgs_ctx_init_cached(bsgs_ctx* b, secp256k1_context* ctx,
                                int bits_total, int l1) {
    memset(b, 0, sizeof(*b));
    b->ctx = ctx; b->bits_total = bits_total; b->l1 = l1;
    if (bits_total <= 0 || bits_total > 63) return 0;
    if (l1 <= 0 || l1 >= bits_total)        return 0;

    b->M     = 1ULL << l1;
    b->Mhalf = 1ULL << (l1 - 1);
    b->J     = 1ULL << (bits_total - l1);

    unsigned char one[32] = {0}; one[31] = 1;
    if (!secp256k1_ec_pubkey_create(ctx, &b->G, one)) return 0;

    unsigned char Msc[32];
    u64_to_scalar32_be(b->M, Msc);
    if (!secp256k1_ec_pubkey_create(ctx, &b->MG, Msc)) return 0;

    pubkey_to_ge(&b->MG, &b->MG_ge);
    ge_negate(&b->neg_MG_ge, &b->MG_ge);

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
     * Build cuckoo table.
     *
     * Phase 1: allocate 12-byte build_entry table (section_size computed
     *          inside cuckoo_alloc_build from n = Mhalf-1).
     * Phase 2: walk i*G for i in [1, Mhalf), insert each (x64, i).
     * Phase 3: compact 12-byte → 8-byte, free build table.
     */
    size_t n = (size_t)(b->Mhalf - 1);
    build_entry* btab = cuckoo_alloc_build(&b->baby, n);
    if (!btab) { fprintf(stderr, "cuckoo_alloc_build failed\n"); return 0; }

    printf("Building cuckoo table: n=%zu, section_size=%zu, total_bins=%zu\n"
           "  Build memory: %.2f MB (12-byte entries)\n",
           n, b->baby.section_size, b->baby.total_bins,
           (double)(b->baby.total_bins * sizeof(build_entry)) / (1<<20));

    secp256k1_pubkey cur = b->G;
    unsigned char ser[33];
    size_t s = b->baby.section_size;

    for (uint64_t i = 1; i < b->Mhalf; i++) {
        if (!pubkey_serialize33(ctx, &cur, ser)) {
            free(btab); map_free(&b->baby); return 0;
        }
        uint64_t x64 = 0;
        for (int k = 1; k <= 8; k++) x64 = (x64 << 8) | ser[k];

        if (!cuckoo_insert_build(btab, s, &b->baby, x64, (uint32_t)i)) {
            fprintf(stderr, "cuckoo insert failed at i=%llu\n",
                    (unsigned long long)i);
            free(btab); map_free(&b->baby); return 0;
        }

        if (i + 1 < b->Mhalf) {
            const secp256k1_pubkey* pts[2] = { &cur, &b->G };
            secp256k1_pubkey nxt;
            if (!secp256k1_ec_pubkey_combine(ctx, &nxt, pts, 2)) {
                free(btab); map_free(&b->baby); return 0;
            }
            cur = nxt;
        }
    }

    printf("  Stash used: %d entries\n", b->baby.stash_count);

    /* Compact: 12-byte build entries → 8-byte lookup entries */
    if (!cuckoo_compact(&b->baby, btab)) {
        fprintf(stderr, "cuckoo_compact failed\n");
        map_free(&b->baby); return 0;
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

/* Verify candidate m: check m*G == target. Called at most once per solve. */
static int verify_candidate(const secp256k1_context* ctx, uint64_t m,
                            const unsigned char target33[33]) {
    if (m == 0) return 0;
    unsigned char sc[32]; u64_to_scalar32_be(m, sc);
    secp256k1_pubkey chk;
    if (!secp256k1_ec_pubkey_create(ctx, &chk, sc)) return 0;
    unsigned char c33[33];
    if (!pubkey_serialize33(ctx, &chk, c33)) return 0;
    return memcmp(c33, target33, 33) == 0;
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
static int bsgs_solve(const bsgs_ctx* b,
                      const secp256k1_pubkey* targetPm,
                      int window,
                      uint64_t* out_m) {
    const secp256k1_context* ctx = b->ctx;
    unsigned char t33[33];
    if (!pubkey_serialize33(ctx, targetPm, t33)) return 0;

    uint64_t tx64 = 0;
    for (int i = 1; i <= 8; i++) tx64 = (tx64 << 8) | t33[i];

    {
        uint32_t cands[3 + CUCKOO_STASH_SZ];
        int nc = map_get_all(&b->baby, tx64, cands);
        for (int ci = 0; ci < nc; ci++)
            if (verify_candidate(ctx, (uint64_t)cands[ci], t33)) {
            *out_m = (uint64_t)cands[ci]; return 1;
        }
    }

    secp256k1_ge target_ge;
    pubkey_to_ge(targetPm, &target_ge);

    secp256k1_gej Qj;
    pubkey_to_gej(targetPm, &Qj);
    secp256k1_gej_add_ge(&Qj, &Qj, &b->neg_MG_ge);

    secp256k1_gej jMGj;
    secp256k1_gej_set_ge(&jMGj, &b->MG_ge);

    size_t W = (size_t)window;
    secp256k1_gej* Q_win = (secp256k1_gej*)malloc(W * sizeof(secp256k1_gej));
    uint64_t*      j_win = (uint64_t*)     malloc(W * sizeof(uint64_t));
    secp256k1_fe*  bt1   = (secp256k1_fe*) malloc(2 * W * sizeof(secp256k1_fe));
    secp256k1_fe*  bt2   = (secp256k1_fe*) malloc(2 * W * sizeof(secp256k1_fe));
    if (!Q_win || !j_win || !bt1 || !bt2) {
        free(Q_win); free(j_win); free(bt1); free(bt2); return 0;
    }

    int result = 0;
    uint64_t j = 1;

    while (j + (uint64_t)W <= b->J && !result) {
        int early = 0;
        for (size_t w = 0; w < W; w++) {
            if (gej_eq_ge(&jMGj, &target_ge)) {
                *out_m = (j + (uint64_t)w) * b->M; result = 1; early = 1; break;
            }
            Q_win[w] = Qj; j_win[w] = j + (uint64_t)w;
            secp256k1_gej_add_ge(&Qj,   &Qj,   &b->neg_MG_ge);
            secp256k1_gej_add_ge(&jMGj, &jMGj, &b->MG_ge);
        }
        if (early) break;
        j += (uint64_t)W;

        for (size_t w = 0; w < W; w++) {
            bt1[W + w] = Q_win[w].z;
            secp256k1_fe_normalize_var(&bt1[W + w]);
        }
        fe_batch_invert_tree(bt1, bt2, W);

        for (size_t w = 0; w < W && !result; w++) {
            uint64_t qx64 = gej_x64_from_zinv(&Q_win[w], &bt2[W + w]);
            uint32_t cands[3 + CUCKOO_STASH_SZ];
            int nc = map_get_all(&b->baby, qx64, cands);
            for (int ci = 0; ci < nc && !result; ci++) {
                uint64_t m1 = j_win[w] * b->M + (uint64_t)cands[ci];
                uint64_t m2 = j_win[w] * b->M - (uint64_t)cands[ci];
                if (verify_candidate(ctx, m1, t33)) { *out_m = m1; result = 1; }
                else if (verify_candidate(ctx, m2, t33)) { *out_m = m2; result = 1; }
            }
        }
    }

    /* Remaining steps (< W) — single inversion fallback */
    for (; j < b->J && !result; j++) {
        if (gej_eq_ge(&jMGj, &target_ge)) { *out_m = j * b->M; result = 1; break; }
        secp256k1_ge Q_ge;
        secp256k1_ge_set_gej(&Q_ge, &Qj);
        uint64_t qx64 = ge_x64(&Q_ge);
        uint32_t cands[3 + CUCKOO_STASH_SZ];
        int nc = map_get_all(&b->baby, qx64, cands);
        for (int ci = 0; ci < nc && !result; ci++) {
            uint64_t m1 = j * b->M + (uint64_t)cands[ci];
            uint64_t m2 = j * b->M - (uint64_t)cands[ci];
            if (verify_candidate(ctx, m1, t33)) { *out_m = m1; result = 1; }
            else if (verify_candidate(ctx, m2, t33)) { *out_m = m2; result = 1; }
        }
        if (!result) {
            secp256k1_gej_add_ge(&Qj,   &Qj,   &b->neg_MG_ge);
            secp256k1_gej_add_ge(&jMGj, &jMGj, &b->MG_ge);
        }
    }

    free(Q_win); free(j_win); free(bt1); free(bt2);
    return result;
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
    uint64_t*        found_m;
    pthread_mutex_t* found_mu;
} bsgs_worker_args;

static void* bsgs_worker_thread(void* argp) {
    bsgs_worker_args* a = (bsgs_worker_args*)argp;
    const bsgs_ctx*   b = a->b;
    size_t            W = (size_t)a->window;

    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    if (!ctx) return NULL;

    secp256k1_gej* Q_win = (secp256k1_gej*)malloc(W * sizeof(secp256k1_gej));
    uint64_t*      j_win = (uint64_t*)     malloc(W * sizeof(uint64_t));
    secp256k1_fe*  bt1   = (secp256k1_fe*) malloc(2 * W * sizeof(secp256k1_fe));
    secp256k1_fe*  bt2   = (secp256k1_fe*) malloc(2 * W * sizeof(secp256k1_fe));
    if (!Q_win || !j_win || !bt1 || !bt2) {
        free(Q_win); free(j_win); free(bt1); free(bt2);
        secp256k1_context_destroy(ctx); return NULL;
    }

    uint64_t j = a->j_start;
    unsigned char jm_sc[32];
    u64_to_scalar32_be(j * b->M, jm_sc);
    secp256k1_pubkey jMG_pub;
    if (!secp256k1_ec_pubkey_create(ctx, &jMG_pub, jm_sc)) {
        free(Q_win); free(j_win); free(bt1); free(bt2);
        secp256k1_context_destroy(ctx); return NULL;
    }

    secp256k1_gej jMGj;
    pubkey_to_gej(&jMG_pub, &jMGj);

    secp256k1_ge jMG_ge, neg_jMG_ge;
    pubkey_to_ge(&jMG_pub, &jMG_ge);
    ge_negate(&neg_jMG_ge, &jMG_ge);

    secp256k1_gej Qj;
    pubkey_to_gej(&a->targetPm, &Qj);
    secp256k1_gej_add_ge(&Qj, &Qj, &neg_jMG_ge);

    /* ---- windowed main loop ---- */
    while (j + (uint64_t)W <= a->j_end) {
        if (atomic_load_explicit(a->found, memory_order_relaxed)) break;

        int early = 0;
        for (size_t w = 0; w < W; w++) {
            if (gej_eq_ge(&jMGj, &a->target_ge)) {
                uint64_t m = (j + (uint64_t)w) * b->M;
                pthread_mutex_lock(a->found_mu);
                if (!atomic_load_explicit(a->found, memory_order_relaxed)) {
                    *a->found_m = m;
                    atomic_store_explicit(a->found, 1, memory_order_relaxed);
                }
                pthread_mutex_unlock(a->found_mu);
                early = 1; break;
            }
            Q_win[w] = Qj; j_win[w] = j + (uint64_t)w;
            secp256k1_gej_add_ge(&Qj,   &Qj,   &b->neg_MG_ge);
            secp256k1_gej_add_ge(&jMGj, &jMGj, &b->MG_ge);
        }
        if (early) break;
        j += (uint64_t)W;

        for (size_t w = 0; w < W; w++) {
            bt1[W + w] = Q_win[w].z;
            secp256k1_fe_normalize_var(&bt1[W + w]);
        }
        fe_batch_invert_tree(bt1, bt2, W);

        int found_in_window = 0;
        for (size_t w = 0; w < W; w++) {
            if (atomic_load_explicit(a->found, memory_order_relaxed)) {
                found_in_window = 1; break;
            }
            uint64_t qx64 = gej_x64_from_zinv(&Q_win[w], &bt2[W + w]);
            uint32_t cands[3 + CUCKOO_STASH_SZ];
            int nc = map_get_all(&b->baby, qx64, cands);
            for (int ci = 0; ci < nc && !found_in_window; ci++) {
                uint64_t m1 = j_win[w] * b->M + (uint64_t)cands[ci];
                uint64_t m2 = j_win[w] * b->M - (uint64_t)cands[ci];
                uint64_t m_ok = 0; int got = 0;
                if (verify_candidate(ctx, m1, a->target33)) { m_ok = m1; got = 1; }
                else if (verify_candidate(ctx, m2, a->target33)) { m_ok = m2; got = 1; }
                if (got) {
                    pthread_mutex_lock(a->found_mu);
                    if (!atomic_load_explicit(a->found, memory_order_relaxed)) {
                        *a->found_m = m_ok;
                        atomic_store_explicit(a->found, 1, memory_order_relaxed);
                    }
                    pthread_mutex_unlock(a->found_mu);
                    found_in_window = 1;
                }
            }
        }
        if (found_in_window) break;
    }

    /* ---- remaining steps (< W) ---- */
    while (j < a->j_end &&
           !atomic_load_explicit(a->found, memory_order_relaxed)) {
        if (gej_eq_ge(&jMGj, &a->target_ge)) {
            uint64_t m = j * b->M;
            pthread_mutex_lock(a->found_mu);
            if (!atomic_load_explicit(a->found, memory_order_relaxed)) {
                *a->found_m = m;
                atomic_store_explicit(a->found, 1, memory_order_relaxed);
            }
            pthread_mutex_unlock(a->found_mu);
            break;
        }
        secp256k1_ge Q_ge;
        secp256k1_ge_set_gej(&Q_ge, &Qj);
        uint64_t qx64 = ge_x64(&Q_ge);
        uint32_t cands[3 + CUCKOO_STASH_SZ];
        int nc = map_get_all(&b->baby, qx64, cands);
        int got_it = 0;
        for (int ci = 0; ci < nc && !got_it; ci++) {
            uint64_t m1 = j * b->M + (uint64_t)cands[ci];
            uint64_t m2 = j * b->M - (uint64_t)cands[ci];
            uint64_t m_ok = 0; int got = 0;
            if (verify_candidate(ctx, m1, a->target33)) { m_ok = m1; got = 1; }
            else if (verify_candidate(ctx, m2, a->target33)) { m_ok = m2; got = 1; }
            if (got) {
                pthread_mutex_lock(a->found_mu);
                if (!atomic_load_explicit(a->found, memory_order_relaxed)) {
                    *a->found_m = m_ok;
                    atomic_store_explicit(a->found, 1, memory_order_relaxed);
                }
                pthread_mutex_unlock(a->found_mu);
                got_it = 1;
            }
        }
        if (got_it) break;
        secp256k1_gej_add_ge(&Qj,   &Qj,   &b->neg_MG_ge);
        secp256k1_gej_add_ge(&jMGj, &jMGj, &b->MG_ge);
        j++;
    }

    free(Q_win); free(j_win); free(bt1); free(bt2);
    secp256k1_context_destroy(ctx);
    return NULL;
}

static int bsgs_solve_parallel(const bsgs_ctx* b,
                               const secp256k1_pubkey* targetPm,
                               int nthreads, int window,
                               uint64_t* out_m) {
    if (nthreads < 1) nthreads = 1;
    if (nthreads == 1) return bsgs_solve(b, targetPm, window, out_m);

    const secp256k1_context* ctx0 = b->ctx;
    unsigned char t33[33];
    if (!pubkey_serialize33(ctx0, targetPm, t33)) return 0;

    /* j=0: direct baby lookup */
    uint64_t tx64 = 0;
    for (int i = 1; i <= 8; i++) tx64 = (tx64 << 8) | t33[i];
    {
        uint32_t cands[3 + CUCKOO_STASH_SZ];
        int nc = map_get_all(&b->baby, tx64, cands);
        for (int ci = 0; ci < nc; ci++)
            if (verify_candidate(ctx0, (uint64_t)cands[ci], t33)) {
            *out_m = (uint64_t)cands[ci]; return 1;
        }
    }

    uint64_t J = b->J;
    if (J <= 1) return 0;
    if ((uint64_t)nthreads > (J - 1)) nthreads = (int)(J - 1);

    pthread_t*        tids = (pthread_t*)       calloc((size_t)nthreads, sizeof(pthread_t));
    bsgs_worker_args* args = (bsgs_worker_args*)calloc((size_t)nthreads, sizeof(bsgs_worker_args));
    if (!tids || !args) { free(tids); free(args); return 0; }

    atomic_int      found   = 0;
    uint64_t        found_m = 0;
    pthread_mutex_t found_mu;
    pthread_mutex_init(&found_mu, NULL);

    secp256k1_ge target_ge;
    pubkey_to_ge(targetPm, &target_ge);

    uint64_t total = J - 1;
    uint64_t chunk = total / (uint64_t)nthreads;
    uint64_t rem   = total % (uint64_t)nthreads;
    uint64_t jcur  = 1;

    for (int t = 0; t < nthreads; t++) {
        uint64_t take = chunk + (t < (int)rem ? 1 : 0);
        args[t].b         = b;
        args[t].targetPm  = *targetPm;
        memcpy(args[t].target33, t33, 33);
        args[t].target_ge = target_ge;
        args[t].j_start   = jcur;
        args[t].j_end     = jcur + take;
        args[t].window    = window;
        args[t].found     = &found;
        args[t].found_m   = &found_m;
        args[t].found_mu  = &found_mu;
        jcur += take;
        pthread_create(&tids[t], NULL, bsgs_worker_thread, &args[t]);
    }

    for (int t = 0; t < nthreads; t++) pthread_join(tids[t], NULL);

    int ok = atomic_load_explicit(&found, memory_order_relaxed);
    if (ok) *out_m = found_m;

    pthread_mutex_destroy(&found_mu);
    free(tids); free(args);
    return ok;
}

/* ================================================================
 * Benchmark
 * ================================================================ */

static void benchmark_bsgs(int bits_total, int l1, int trials, int threads, int window) {
    printf("=== BSGS (secp256k1, cuckoo k=3, Jacobian, windowed TreeMon) ===\n");
    printf("Range  : m in [0, 2^%d)\n", bits_total);
    printf("Split  : l1=%d, l2=%d\n", l1, bits_total - l1);
    printf("Window : W=%d\n", window);
    printf("Entry  : %zu bytes lookup | Trials: %d | Threads: %d\n\n",
           sizeof(entry_packed), trials, threads);

    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    if (!ctx) { printf("Failed to create context\n"); return; }

    bsgs_ctx solver;
    double t0 = now_seconds();
    if (!bsgs_ctx_init_cached(&solver, ctx, bits_total, l1)) {
        printf("Failed to init solver\n");
        secp256k1_context_destroy(ctx); return;
    }
    double t1 = now_seconds();
    printf("Table : %.2f MB | Init: %.6f sec\n\n",
           (double)(solver.baby.total_bins * sizeof(entry_packed)) / (1 << 20), t1 - t0);

    uint64_t mask = (bits_total == 64) ? ~0ULL : ((1ULL << bits_total) - 1ULL);
    int ok = 0;
    double ts = now_seconds();
    for (int t = 0; t < trials; t++) {
        uint64_t m = ((uint64_t)rand() << 32) ^ (uint64_t)rand();
        m &= mask; if (m == 0) m = 1;
        unsigned char sc[32]; u64_to_scalar32_be(m, sc);
        secp256k1_pubkey Pm;
        if (!secp256k1_ec_pubkey_create(ctx, &Pm, sc)) { printf("create failed\n"); break; }
        uint64_t recovered = 0;
        if (bsgs_solve_parallel(&solver, &Pm, threads, window, &recovered) && recovered == m)
            ok++;
        else
            printf("Trial %d FAILED: m=%"PRIu64" recovered=%"PRIu64"\n", t, m, recovered);
    }
    double te = now_seconds();
    double total = te - ts;
    printf("Solved correctly: %d/%d\n", ok, trials);
    printf("Total search time: %.6f sec\n", total);
    printf("Average per solve: %.6f sec (%.2f ms)\n\n",
           total / trials, (total / trials) * 1e3);

    bsgs_ctx_free(&solver);
    secp256k1_context_destroy(ctx);
}

int main(int argc, char** argv) {
    int bits_total = 40, l1 = 18, trials = 1, threads = 1, window = 64;
    if (argc >= 2) bits_total = atoi(argv[1]);
    if (argc >= 3) l1         = atoi(argv[2]);
    if (argc >= 4) trials     = atoi(argv[3]);
    if (argc >= 5) threads    = atoi(argv[4]);
    if (argc >= 6) window     = atoi(argv[5]);

    /* Window must be a power of 2 >= 1 */
    if (window < 1) window = 1;
    { int w = 1; while (w < window) w <<= 1; window = w; }

    srand((unsigned)time(NULL));
    benchmark_bsgs(bits_total, l1, trials, threads, window);
    return 0;
}