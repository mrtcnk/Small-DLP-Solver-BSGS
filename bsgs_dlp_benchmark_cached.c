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
 * Windowed Tree-based Montgomery Batch Inversion (TreeMon)
 * ================================================================
 *
 * fe_batch_invert_tree():
 *   Inverts n field elements using:
 *     - 1 field inversion  (at the tree root)
 *     - 3*(n-1) field multiplications
 *
 *   n must be a power of 2.
 *
 *   Binary tree stored level-order (1-indexed, size 2n):
 *     BT[1]       = root
 *     BT[n..2n-1] = leaves
 *     BT[i] children: BT[2i], BT[2i+1]
 *
 *   BT1 (product tree, bottom-up):
 *     BT1[leaf i] = Z[i]
 *     BT1[parent] = BT1[left] * BT1[right]
 *
 *   BT2 (inverse tree, top-down):
 *     BT2[root]  = 1 / BT1[root]           <- ONE inversion
 *     BT2[left]  = BT1[right] * BT2[parent]
 *     BT2[right] = BT1[left]  * BT2[parent]
 *
 *   Output: BT2[n+i] = 1/Z[i]
 *
 * Caller provides bt1[2n] and bt2[2n] as pre-allocated workspaces.
 * bt1[n..2n-1] must be filled with input values before calling.
 * bt2[n..2n-1] holds the results after return.
 */
static void fe_batch_invert_tree(secp256k1_fe* bt1,
                                 secp256k1_fe* bt2,
                                 size_t        n) {
    /* Build BT1 bottom-up: parent = left * right */
    for (size_t i = n - 1; i >= 1; i--)
        secp256k1_fe_mul(&bt1[i], &bt1[2*i], &bt1[2*i + 1]);

    /* BT2 root = 1 / BT1 root  (single inversion) */
    secp256k1_fe_inv(&bt2[1], &bt1[1]);

    /* Build BT2 top-down: children swap siblings from BT1 */
    for (size_t i = 1; i < n; i++) {
        secp256k1_fe_mul(&bt2[2*i],     &bt1[2*i + 1], &bt2[i]);
        secp256k1_fe_mul(&bt2[2*i + 1], &bt1[2*i],     &bt2[i]);
    }
    /* bt2[n+i] = 1/bt1[n+i] for i in [0, n) */
}

/*
 * Extract the 64-bit truncated x-coordinate from a Jacobian point
 * given its precomputed Z-inverse.
 *
 * x_affine = X / Z^2 = X * z_inv^2
 * Returns upper 64 bits of x_affine as a uint64 (big-endian).
 * Does NOT modify the point.
 */
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
 * map: uint64 key -> uint32 value  (packed 8-byte entry)
 * ================================================================
 *
 * CHANGE from 16-byte entry:
 *   Previous: key(8) + val(4) + used(1) + pad(3) = 16 bytes
 *   Now:      key(4) + val(4)                    =  8 bytes
 *
 * Design:
 *   x64  = 64-bit truncated x-coordinate
 *   key  = upper 32 bits of x64 (stored as discriminator)
 *   idx  = hash(x64) & mask     (position in table, not stored)
 *   val  = i value; val==0 means empty (i starts from 1, never 0)
 *
 * False positives: two distinct x64 values with the same upper 32 bits
 * will collide. verify_candidate() already handles this correctly.
 *
 * Memory saving: 2× vs previous entry size.
 */
typedef struct {
    uint32_t key;  /* upper 32 bits of x64 */
    uint32_t val;  /* i value; 0 = empty   */
} entry_packed;    /* 8 bytes, no padding  */

typedef struct {
    entry_packed* tab;
    size_t cap;
    size_t mask;
    size_t size;
} map64_u32;

static size_t next_pow2(size_t x) {
    size_t p = 1; while (p < x) p <<= 1; return p;
}

static int map_init_cap(map64_u32* m, size_t cap_pow2) {
    m->tab = (entry_packed*)calloc(cap_pow2, sizeof(entry_packed));
    if (!m->tab) return 0;
    m->cap = cap_pow2; m->mask = cap_pow2 - 1; m->size = 0;
    return 1;
}
static int map_init(map64_u32* m, size_t want) {
    return map_init_cap(m, next_pow2(want * 2 + 1));
}
static void map_free(map64_u32* m) {
    free(m->tab); memset(m, 0, sizeof(*m));
}

static uint64_t mix64(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31; return x;
}

static int map_put(map64_u32* m, uint64_t x64, uint32_t val) {
    uint32_t stored_key = (uint32_t)(x64 >> 32); /* upper 32 bits */
    size_t   idx        = (size_t)mix64(x64) & m->mask;
    for (;;) {
        entry_packed* e = &m->tab[idx];
        if (e->val == 0) {                        /* empty slot   */
            e->key = stored_key; e->val = val;
            m->size++; return 1;
        }
        if (e->key == stored_key) return 1;       /* duplicate    */
        idx = (idx + 1) & m->mask;
    }
}
static int map_get(const map64_u32* m, uint64_t x64, uint32_t* out) {
    uint32_t stored_key = (uint32_t)(x64 >> 32); /* upper 32 bits */
    size_t   idx        = (size_t)mix64(x64) & m->mask;
    for (;;) {
        const entry_packed* e = &m->tab[idx];
        if (e->val == 0) return 0;                /* empty = miss  */
        if (e->key == stored_key) { *out = e->val; return 1; }
        idx = (idx + 1) & m->mask;
    }
}

/* ---------------- cached baby table ---------------- */

#define BABY_MAGIC 0x44454B434150ULL  /* "PACKED" — new format */

typedef struct {
    uint64_t magic; uint32_t version; uint32_t l1;
    uint64_t cap;   uint64_t used_count;
} baby_hdr;

static void baby_cache_path(char* out, size_t outlen, int l1) {
    snprintf(out, outlen, "bsgs_baby_packed_secp256k1_l1_%d.bin", l1);
}

static int baby_save(const char* path, int l1, const map64_u32* baby) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror("baby_save open"); return 0; }
    baby_hdr hdr = { BABY_MAGIC, 3, (uint32_t)l1,
                     (uint64_t)baby->cap, (uint64_t)baby->size };
    int ok = write_all(fd, &hdr, sizeof(hdr));
    ok    &= write_all(fd, baby->tab, baby->cap * sizeof(entry_packed));
    if (!ok) perror("baby_save write");
    close(fd); return ok;
}

static int baby_load(const char* path, int expected_l1, map64_u32* out) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;
    baby_hdr hdr;
    if (!read_all(fd, &hdr, sizeof(hdr))) { close(fd); return 0; }
    if (hdr.magic != BABY_MAGIC || hdr.version != 3 || (int)hdr.l1 != expected_l1
        || hdr.cap == 0 || (hdr.cap & (hdr.cap-1)) != 0) { close(fd); return 0; }
    if (!map_init_cap(out, (size_t)hdr.cap)) { close(fd); return 0; }
    if (!read_all(fd, out->tab, (size_t)hdr.cap * sizeof(entry_packed))) {
        map_free(out); close(fd); return 0;
    }
    out->size = (size_t)hdr.used_count;
    close(fd); return 1;
}

/* ================================================================
 * Reusable BSGS context
 * ================================================================ */

typedef struct {
    secp256k1_context* ctx;
    int bits_total, l1;
    uint64_t M, Mhalf, J;

    secp256k1_pubkey G, MG;  /* kept for scalar-mult init */

    /* NEW: affine secp256k1_ge versions used with gej_add_ge in hot loop */
    secp256k1_ge MG_ge;
    secp256k1_ge neg_MG_ge;

    map64_u32 baby;
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

    /* Precompute affine MG_ge and neg_MG_ge for the Jacobian loop */
    pubkey_to_ge(&b->MG, &b->MG_ge);
    ge_negate(&b->neg_MG_ge, &b->MG_ge);

    char path[128];
    baby_cache_path(path, sizeof(path), l1);
    if (file_exists(path)) {
        if (baby_load(path, l1, &b->baby)) {
            printf("Loaded baby table cache: %s (cap=%zu, used=%zu)\n",
                   path, b->baby.cap, b->baby.size);
            return 1;
        }
        printf("Cache exists but failed to load (rebuilding): %s\n", path);
    }

    if (!map_init(&b->baby, (size_t)(b->Mhalf - 1))) return 0;

    secp256k1_pubkey cur = b->G;
    unsigned char ser[33];
    for (uint64_t i = 1; i < b->Mhalf; i++) {
        if (!pubkey_serialize33(ctx, &cur, ser)) { map_free(&b->baby); return 0; }
        uint64_t xkey = 0;
        for (int k = 1; k <= 8; k++) xkey = (xkey << 8) | ser[k];
        if (!map_put(&b->baby, xkey, (uint32_t)i)) { map_free(&b->baby); return 0; }
        if (i + 1 < b->Mhalf) {
            const secp256k1_pubkey* pts[2] = { &cur, &b->G };
            secp256k1_pubkey nxt;
            if (!secp256k1_ec_pubkey_combine(ctx, &nxt, pts, 2)) {
                map_free(&b->baby); return 0;
            }
            cur = nxt;
        }
    }
    if (!baby_save(path, l1, &b->baby))
        printf("Warning: failed to save cache: %s\n", path);
    else
        printf("Built and saved: %s (cap=%zu, used=%zu)\n",
               path, b->baby.cap, b->baby.size);
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
 * Single-threaded solve — Windowed TreeMon
 * ================================================================
 *
 * Giant step loop processes W steps at a time:
 *
 *   Phase 1 (W steps, no inversion):
 *     - Check i=0 via gej_eq_ge()        — 0 inversions
 *     - Store Qj in Q_win[], advance Qj  — 0 inversions
 *
 *   Phase 2 (once per window):
 *     - fe_batch_invert_tree()           — 1 inversion + ~3W mults
 *
 *   Phase 3 (W lookups, no inversion):
 *     - x_affine = Q_win[w].X * z_inv^2 — 1 sqr + 1 mul each
 *     - map_get()
 *
 *   Cost per step: 1/W inversions + ~5 multiplications
 *   vs previous:  1   inversion  + ~2 multiplications
 *
 *   Remaining steps after last full window fall back to
 *   single-element inversion (at most W-1 extra inversions).
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

    /* j=0: target itself may be in baby table */
    uint32_t i_found;
    if (map_get(&b->baby, tx64, &i_found))
        if (verify_candidate(ctx, (uint64_t)i_found, t33)) {
            *out_m = (uint64_t)i_found; return 1;
        }

    secp256k1_ge target_ge;
    pubkey_to_ge(targetPm, &target_ge);

    secp256k1_gej Qj;
    pubkey_to_gej(targetPm, &Qj);
    secp256k1_gej_add_ge(&Qj, &Qj, &b->neg_MG_ge);

    secp256k1_gej jMGj;
    secp256k1_gej_set_ge(&jMGj, &b->MG_ge);

    size_t W = (size_t)window;

    /* Pre-allocate window workspace */
    secp256k1_gej* Q_win = (secp256k1_gej*)malloc(W * sizeof(secp256k1_gej));
    uint64_t*      j_win = (uint64_t*)     malloc(W * sizeof(uint64_t));
    secp256k1_fe*  bt1   = (secp256k1_fe*) malloc(2 * W * sizeof(secp256k1_fe));
    secp256k1_fe*  bt2   = (secp256k1_fe*) malloc(2 * W * sizeof(secp256k1_fe));
    if (!Q_win || !j_win || !bt1 || !bt2) {
        free(Q_win); free(j_win); free(bt1); free(bt2);
        return 0;
    }

    int result = 0;
    uint64_t j = 1;

    /* ---- Main loop: full windows of W steps ---- */
    while (j + (uint64_t)W <= b->J) {

        /* Phase 1: accumulate W Jacobian Q points, check i=0 each step */
        int early = 0;
        for (size_t w = 0; w < W; w++) {
            if (gej_eq_ge(&jMGj, &target_ge)) {
                *out_m = (j + (uint64_t)w) * b->M;
                result = 1; early = 1; break;
            }
            Q_win[w] = Qj;
            j_win[w] = j + (uint64_t)w;
            secp256k1_gej_add_ge(&Qj,   &Qj,   &b->neg_MG_ge);
            secp256k1_gej_add_ge(&jMGj, &jMGj, &b->MG_ge);
        }
        if (early) break;
        j += (uint64_t)W;

        /* Phase 2: fill BT1 leaves with Z coords, batch invert */
        for (size_t w = 0; w < W; w++) {
            bt1[W + w] = Q_win[w].z;
            secp256k1_fe_normalize_var(&bt1[W + w]);
        }
        fe_batch_invert_tree(bt1, bt2, W);

        /* Phase 3: x_affine = X * z_inv^2, lookup */
        for (size_t w = 0; w < W; w++) {
            uint64_t qx64 = gej_x64_from_zinv(&Q_win[w], &bt2[W + w]);
            if (map_get(&b->baby, qx64, &i_found)) {
                uint64_t m1 = j_win[w] * b->M + (uint64_t)i_found;
                uint64_t m2 = j_win[w] * b->M - (uint64_t)i_found;
                if (verify_candidate(ctx, m1, t33)) { *out_m = m1; result = 1; break; }
                if (verify_candidate(ctx, m2, t33)) { *out_m = m2; result = 1; break; }
            }
        }
        if (result) break;
    }

    /* ---- Cleanup: remaining steps (< W) with single inversion ---- */
    if (!result) {
        for (; j < b->J; j++) {
            if (gej_eq_ge(&jMGj, &target_ge)) {
                *out_m = j * b->M; result = 1; break;
            }
            secp256k1_ge Q_ge;
            secp256k1_ge_set_gej(&Q_ge, &Qj);
            uint64_t qx64 = ge_x64(&Q_ge);
            if (map_get(&b->baby, qx64, &i_found)) {
                uint64_t m1 = j * b->M + (uint64_t)i_found;
                uint64_t m2 = j * b->M - (uint64_t)i_found;
                if (verify_candidate(ctx, m1, t33)) { *out_m = m1; result = 1; break; }
                if (verify_candidate(ctx, m2, t33)) { *out_m = m2; result = 1; break; }
            }
            secp256k1_gej_add_ge(&Qj,   &Qj,   &b->neg_MG_ge);
            secp256k1_gej_add_ge(&jMGj, &jMGj, &b->MG_ge);
        }
    }

    free(Q_win); free(j_win); free(bt1); free(bt2);
    return result;
}

/* ================================================================
 * Parallel solve
 * ================================================================ */

typedef struct {
    const bsgs_ctx*  b;
    secp256k1_pubkey targetPm;
    unsigned char    target33[33];
    secp256k1_ge     target_ge;
    uint64_t         j_start, j_end;
    int              window;        /* NEW: batch inversion window size */
    atomic_int*      found;
    uint64_t*        found_m;
    pthread_mutex_t* found_mu;
} bsgs_worker_args;

static void* bsgs_worker_thread(void* argp) {
    bsgs_worker_args* a = (bsgs_worker_args*)argp;
    const bsgs_ctx*   b = a->b;
    size_t            W = (size_t)a->window;

    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_VERIFY);
    if (!ctx) return NULL;

    /* Per-thread workspace — allocated once, reused across all windows */
    secp256k1_gej* Q_win = (secp256k1_gej*)malloc(W * sizeof(secp256k1_gej));
    uint64_t*      j_win = (uint64_t*)     malloc(W * sizeof(uint64_t));
    secp256k1_fe*  bt1   = (secp256k1_fe*) malloc(2 * W * sizeof(secp256k1_fe));
    secp256k1_fe*  bt2   = (secp256k1_fe*) malloc(2 * W * sizeof(secp256k1_fe));
    if (!Q_win || !j_win || !bt1 || !bt2) {
        free(Q_win); free(j_win); free(bt1); free(bt2);
        secp256k1_context_destroy(ctx);
        return NULL;
    }

    /* Compute j_start * MG via scalar mult (one-time per thread) */
    uint64_t j = a->j_start;
    unsigned char jm_sc[32];
    u64_to_scalar32_be(j * b->M, jm_sc);
    secp256k1_pubkey jMG_pub;
    if (!secp256k1_ec_pubkey_create(ctx, &jMG_pub, jm_sc)) {
        free(Q_win); free(j_win); free(bt1); free(bt2);
        secp256k1_context_destroy(ctx);
        return NULL;
    }

    secp256k1_gej jMGj;
    pubkey_to_gej(&jMG_pub, &jMGj);

    secp256k1_ge jMG_ge, neg_jMG_ge;
    pubkey_to_ge(&jMG_pub, &jMG_ge);
    ge_negate(&neg_jMG_ge, &jMG_ge);

    secp256k1_gej Qj;
    pubkey_to_gej(&a->targetPm, &Qj);
    secp256k1_gej_add_ge(&Qj, &Qj, &neg_jMG_ge);

    /* ---- Main loop: full windows of W steps ---- */
    while (j + (uint64_t)W <= a->j_end) {
        if (atomic_load_explicit(a->found, memory_order_relaxed)) break;

        /* Phase 1: accumulate W points, check i=0 each step */
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
            Q_win[w] = Qj;
            j_win[w] = j + (uint64_t)w;
            secp256k1_gej_add_ge(&Qj,   &Qj,   &b->neg_MG_ge);
            secp256k1_gej_add_ge(&jMGj, &jMGj, &b->MG_ge);
        }
        if (early) break;
        j += (uint64_t)W;

        /* Phase 2: batch invert Z coordinates */
        for (size_t w = 0; w < W; w++) {
            bt1[W + w] = Q_win[w].z;
            secp256k1_fe_normalize_var(&bt1[W + w]);
        }
        fe_batch_invert_tree(bt1, bt2, W);

        /* Phase 3: compute x64 from z_inv, lookup */
        int found_in_window = 0;
        for (size_t w = 0; w < W; w++) {
            if (atomic_load_explicit(a->found, memory_order_relaxed)) {
                found_in_window = 1; break;
            }
            uint64_t qx64 = gej_x64_from_zinv(&Q_win[w], &bt2[W + w]);
            uint32_t i_found;
            if (map_get(&b->baby, qx64, &i_found)) {
                uint64_t m1 = j_win[w] * b->M + (uint64_t)i_found;
                uint64_t m2 = j_win[w] * b->M - (uint64_t)i_found;
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
                    found_in_window = 1; break;
                }
            }
        }
        if (found_in_window) break;
    }

    /* ---- Cleanup: remaining steps (< W) with single inversion ---- */
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

        uint32_t i_found;
        if (map_get(&b->baby, qx64, &i_found)) {
            uint64_t m1 = j * b->M + (uint64_t)i_found;
            uint64_t m2 = j * b->M - (uint64_t)i_found;
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
                break;
            }
        }

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
                               int nthreads,
                               int window,
                               uint64_t* out_m) {
    if (nthreads < 1) nthreads = 1;
    if (nthreads == 1) return bsgs_solve(b, targetPm, window, out_m);

    const secp256k1_context* ctx0 = b->ctx;
    unsigned char t33[33];
    if (!pubkey_serialize33(ctx0, targetPm, t33)) return 0;

    /* j=0: direct baby lookup */
    uint64_t tx64 = 0;
    for (int i = 1; i <= 8; i++) tx64 = (tx64 << 8) | t33[i];
    uint32_t i_found;
    if (map_get(&b->baby, tx64, &i_found))
        if (verify_candidate(ctx0, (uint64_t)i_found, t33)) {
            *out_m = (uint64_t)i_found; return 1;
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
    printf("=== BSGS (secp256k1, packed 8-byte, Jacobian, windowed TreeMon) ===\n");
    printf("Range  : m in [0, 2^%d)\n", bits_total);
    printf("Split  : l1=%d, l2=%d\n", l1, bits_total - l1);
    printf("Window : W=%d  (1 inversion per %d steps, ~%.1fx vs W=1)\n",
           window, window, (double)window * 46.0 / (window * 5.0 + 10346.0 / 46.0));
    printf("Entry  : %zu bytes | Trials: %d | Threads: %d\n\n",
           sizeof(entry_packed), trials, threads);

    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_VERIFY);
    if (!ctx) { printf("Failed to create context\n"); return; }

    bsgs_ctx solver;
    double t0 = now_seconds();
    if (!bsgs_ctx_init_cached(&solver, ctx, bits_total, l1)) {
        printf("Failed to init solver\n");
        secp256k1_context_destroy(ctx); return;
    }
    double t1 = now_seconds();
    printf("Table : %.2f MB | Init: %.6f sec\n\n",
           (double)(solver.baby.cap * sizeof(entry_packed)) / (1 << 20), t1 - t0);
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