/*
 * fastecdlp_parallel.c
 *
 * FastECDLP (Tang et al., ePrint 2022/1573) with parallelised Phase 1+2 on secp256k1.
 *
 * This implements Tang et al.'s approach faithfully:
 *   - Precompute T2 = {j*M*G in affine} for j in [0, 2^l2)  [one-time, saved to disk]
 *   - Per solve: compute denom[j] = Pm.x - T2[j].x for all j
 *   - Apply Montgomery batch inversion (TreeMon) over all 2^l2 denominators
 *   - Compute x(Pm - T2[j]) = lambda^2 - Pm.x - T2[j].x using inverted denominators
 *   - Look up x64 in k=3 cuckoo baby table
 *
 * Key difference from our contributions:
 *   FastECDLP (this file):  T2 precomputed in affine, per-solve = 0 Jacobian additions
 *   + Jacobian (§3.1):      No T2, per-solve = J Jacobian additions + 1 full batch inv
 *   + Windowed  (§3.2):     No T2, per-solve = J Jacobian additions + J/W batch inv
 *
 * T2 memory: 2^(l2-1) * sizeof(secp256k1_ge) bytes
 *   l2=22 (52-bit): ~1.1 GB     l2=24 (54-bit): ~4.3 GB
 *
 * T2 cache file: fastecdlp_t2_secp256k1_l1_<l1>.bin
 *
 * Build:
 *   cc -O3 -Wall -Wextra -o fastecdlp_parallel fastecdlp_parallel.c \
 *       -I/usr/local/include                                          \
 *       -I/path/to/secp256k1/src                                      \
 *       -L/usr/local/lib                                              \
 *       -lsecp256k1 -lpthread
 *
 * Usage: ./fastecdlp_parallel <bits_total> <l1> <trials> <threads>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>

#include <secp256k1.h>
#include "util.h"
#include "field.h"
#include "field_impl.h"
#include "int128_impl.h"
#include "group.h"
#include "group_impl.h"

#include <errno.h>

#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define secp256k1_fe_equal_var(a,b) (secp256k1_fe_cmp_var((a),(b)) == 0)

/* ─────────────────────── timing ─────────────────────────────────── */
static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ─────────────────────── helpers ─────────────────────────────────── */
static void u64_to_scalar32_be(uint64_t x, unsigned char out32[32]) {
    memset(out32, 0, 32);
    for (int i = 0; i < 8; i++) { out32[31-i] = (unsigned char)(x & 0xFF); x >>= 8; }
}

int file_exists(const char* path) {
    struct stat st;
    return stat(path, &st) == 0;
}

int pubkey_serialize33(const secp256k1_context* ctx,
                              const secp256k1_pubkey* pk,
                              unsigned char out33[33]) {
    size_t outlen = 33;
    if (!secp256k1_ec_pubkey_serialize(ctx, out33, &outlen, pk,
                                       SECP256K1_EC_COMPRESSED))
        return 0;
    return outlen == 33;
}  

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

/* Negate an affine point in-place (flip y). */
void ge_negate(secp256k1_ge* out, const secp256k1_ge* in) {
    *out = *in;
    secp256k1_fe_negate(&out->y, &out->y, 1);
    secp256k1_fe_normalize_var(&out->y);
}
/* Adapted from the original verify_candidate function for signed m_cand */
int verify_candidate(const secp256k1_context* ctx,
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
        unsigned char c33[33];
        if (!pubkey_serialize33(ctx, &pk, c33)) return 0;
        return memcmp(c33, target33, 33) == 0;
    }
}

/* ─────────────────────── I/O ─────────────────────────────────────── */
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

int read_all(int fd, void* buf, size_t len) {
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

/* ─────────────────────── cuckoo table ───────────────────────────── */
#define CUCKOO_K        3
#define CUCKOO_STASH_SZ 16
#define BABY_MAGIC      0x4B43554B4F4F4355ULL

typedef struct { uint32_t key; uint32_t val; } entry_packed;
typedef struct {
    entry_packed* tab;
    size_t section_size, total_bins, size;
    unsigned char stash_xb[CUCKOO_STASH_SZ][32];
    uint32_t stash_val[CUCKOO_STASH_SZ];
    int stash_count;
} cuckoo_map;

typedef struct {
    uint64_t magic; uint32_t version, l1;
    uint64_t section_size, used_count;
    int32_t stash_count; uint32_t _pad;
} baby_hdr;

typedef struct {
    secp256k1_context* ctx;
    int bits_total, l1;
    uint64_t M, J;

    secp256k1_ge MG_ge;
    secp256k1_ge neg_MG_ge;

    uint64_t offset;
    secp256k1_pubkey neg_offsetG_pk;

    cuckoo_map baby;   /* ← cuckoo hash table */
} bsgs_ctx;

static inline uint32_t u32be(const unsigned char* b) {
    return ((uint32_t)b[0]<<24)|((uint32_t)b[1]<<16)|
    ((uint32_t)b[2]<<8)|(uint32_t)b[3];
}
static inline size_t cpos(int sec, const unsigned char xb[32], size_t s) {
    uint32_t h = u32be(xb + sec * 8);
    return (size_t)((__uint128_t)h * (__uint128_t)s >> 32) + (size_t)sec * s;
}
static inline uint32_t ckey(int sec, const unsigned char xb[32]) {
    return u32be(xb + sec * 8 + 4);
}
static int map_get_all(const cuckoo_map* m, const unsigned char xb[32], uint32_t* out) {
    size_t s=m->section_size; int n=0;
    const entry_packed* e;
    e=&m->tab[cpos(0,xb,s)]; if(e->val&&e->key==ckey(0,xb)) out[n++]=e->val;
    e=&m->tab[cpos(1,xb,s)]; if(e->val&&e->key==ckey(1,xb)) out[n++]=e->val;
    e=&m->tab[cpos(2,xb,s)]; if(e->val&&e->key==ckey(2,xb)) out[n++]=e->val;
    for(int i=0;i<m->stash_count;i++)
        if(memcmp(m->stash_xb[i],xb,32)==0) out[n++]=m->stash_val[i];
    return n;
}

static void baby_cache_path(char* out, size_t outlen, int l1) {
    snprintf(out, outlen, "bsgs_baby_cuckoo_secp256k1_l1_%d_window.bin", l1);
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

static int bsgs_ctx_init(bsgs_ctx* b, secp256k1_context* ctx,
                                int bits_total, int l1) {
    memset(b, 0, sizeof(*b));
    b->ctx = ctx; b->bits_total = bits_total; b->l1 = l1;
    if (bits_total <= 0 || bits_total > 63) return 0;
    if (l1 <= 0 || l1 >= bits_total)        return 0;

    b->M     = 1ULL << l1;
    b->J     = 1ULL << (bits_total - l1 - 1);

    unsigned char Msc[32];
    u64_to_scalar32_be(b->M, Msc);
    secp256k1_pubkey MG;
    if (!secp256k1_ec_pubkey_create(ctx, &MG, Msc)) return 0;

    pubkey_to_ge(&MG, &b->MG_ge);
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
            printf("Table memory: %.2f MB\n\n",
                   (double)(b->baby.total_bins * sizeof(entry_packed)) / (1<<20));
            return 1;
        }
        printf("Cache exists but failed to load (rebuilding): %s\n", path);
    } else {
        printf("Generate baby table first by running baby_table!\n");
        return 0;
    }

    return 1;
}

static void map_free(cuckoo_map* m) {
    free(m->tab);
    memset(m, 0, sizeof(*m));
}

static void bsgs_ctx_free(bsgs_ctx* b) {
    map_free(&b->baby); memset(b, 0, sizeof(*b));
}

/* ------- T2 (Giant table) helpers -------*/

/* ─────────────────── T2 table: precompute and cache ─────────────── */
/*
 * T2[j] = j * M * G  in affine coordinates, for j = 0..2^(l2-1).
 *
 * Built once, saved to disk as fastecdlp_t2_secp256k1_l1_<l1>.bin.
 * Reused across all solves — this is FastECDLP's key precomputation.
 *
 * Memory: 2^(l2-1) * sizeof(secp256k1_ge) bytes.
 */
static void t2_cache_path(char* out, size_t outlen, int l1) {
    snprintf(out, outlen, "fastecdlp_t2_secp256k1_l1_%d.bin", l1);
}

static secp256k1_ge* build_t2(const secp256k1_context* ctx,
                              uint64_t M, int l2, double* build_time_out) {
    uint64_t J = 1ULL << (l2-1);
    printf("Building T2: %"PRIu64" affine points...\n", J);
    /* Step 1: compute M*G */
    unsigned char sc_M[32]; u64_to_scalar32_be(M, sc_M);
    secp256k1_pubkey pk_MG;
    secp256k1_ec_pubkey_create(ctx, &pk_MG, sc_M);
    secp256k1_ge MG_ge; pubkey_to_ge(&pk_MG, &MG_ge);

    /* Step 2: walk j*M*G in Jacobian */
    secp256k1_gej* jac = (secp256k1_gej*)malloc(J * sizeof(secp256k1_gej));
    if (!jac) { fprintf(stderr, "OOM: T2 Jacobian\n"); return NULL; }

    double t0 = now_seconds();
    secp256k1_gej acc; secp256k1_gej_set_infinity(&acc);
    for (uint64_t j = 0; j < J; j++) {
        secp256k1_gej_add_ge(&acc, &acc, &MG_ge);
        jac[j] = acc;
    }

    /* Step 3: batch normalize all J Jacobian points to affine */
    secp256k1_ge* T2 = (secp256k1_ge*)malloc(J * sizeof(secp256k1_ge));
    if (!T2) { free(jac); fprintf(stderr, "OOM: T2 affine\n"); return NULL; }
    secp256k1_ge_set_all_gej_var(T2, jac, (size_t)J);
    free(jac);

    *build_time_out = now_seconds() - t0;
    printf("T2 built: %.1f sec | Memory: %.2f GB\n",
           *build_time_out,
           (double)(J * sizeof(secp256k1_ge)) / (1ULL<<30));
    return T2;
}

static int save_t2(const char* path, const secp256k1_ge* T2, uint64_t J) {
    int fd = open(path, O_WRONLY|O_CREAT|O_TRUNC, 0644);
    if (fd < 0) { perror(path); return 0; }
    int ok = write_all(fd, T2, J * sizeof(secp256k1_ge));
    close(fd);
    return ok;
}

static secp256k1_ge* load_t2(const char* path, uint64_t J) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    secp256k1_ge* T2 = (secp256k1_ge*)malloc(J * sizeof(secp256k1_ge));
    if (!T2) { close(fd); return NULL; }
    if (!read_all(fd, T2, J * sizeof(secp256k1_ge))) {
        free(T2); close(fd); return NULL;
    }
    close(fd);
    return T2;
}

static secp256k1_ge* get_t2(const secp256k1_context* ctx,
                            uint64_t M, int l1, int l2,
                            double* build_time_out) {
    char path[128]; t2_cache_path(path, sizeof(path), l1);
    uint64_t J = 1ULL << (l2-1);
    *build_time_out = 0.0;

    struct stat st;
    if (stat(path, &st) == 0 &&
        (uint64_t)st.st_size == J * sizeof(secp256k1_ge)) {
        printf("Loading T2 from cache: %s (%.2f GB)...\n",
               path, (double)st.st_size / (1ULL<<30));
        double t0 = now_seconds();
        secp256k1_ge* T2 = load_t2(path, J);
        if (T2) {
            printf("T2 loaded: %.2f sec\n\n", now_seconds() - t0);
            return T2;
        }
        printf("Load failed, rebuilding.\n");
    }

    secp256k1_ge* T2 = build_t2(ctx, M, l2, build_time_out);
    if (T2) {
        printf("Saving T2 to %s...\n", path);
        if (!save_t2(path, T2, J))
            printf("Warning: save failed (continuing without cache)\n");
        else
            printf("T2 saved.\n");
    }
    return T2;
}

/* ─────────────────── per-thread solve ───────────────────────────── */
/*
 * FastECDLP per-solve per-thread:
 *   Given T2 (precomputed), Pm, and a range [j_start, j_end):
 *
 *   1. Compute denom[k] = Pm.x - T2[j].x  (no point additions!)
 *   2. Montgomery batch inversion over chunk denominators
 *   3. Compute lambda[k] = (Pm.y + T2[j].y) * inv_denom[k]
 *      (note: Pm - T2[j] = Pm + (-T2[j]), so lambda num = Pm.y + T2[j].y)
 *   4. x(Pm - T2[j]) = lambda^2 - Pm.x - T2[j].x
 *   5. Look up in baby table
 */
typedef struct {
    const bsgs_ctx*  b;
    const secp256k1_ge*    T2;
    secp256k1_fe           Pm_x;   /* normalized */
    secp256k1_fe           Pm_y;   /* normalized */
    uint64_t               j_start;
    uint64_t               j_end;
    const unsigned char*   target33;
    atomic_int*            found_flag;
    int64_t                result_m;
    int                    found;
} thread_args;

static void* thread_fn(void* varg) {
    thread_args* a = (thread_args*)varg;
    const bsgs_ctx *b = a->b;
    uint64_t chunk = a->j_end - a->j_start;

    /* Allocate per-thread arrays */
    secp256k1_fe* denom   = (secp256k1_fe*)malloc(chunk * sizeof(secp256k1_fe));
    secp256k1_fe* inv_den = (secp256k1_fe*)malloc(chunk * sizeof(secp256k1_fe));
    secp256k1_fe* prefix  = (secp256k1_fe*)malloc(chunk * sizeof(secp256k1_fe));
    if (!denom || !inv_den || !prefix) {
        fprintf(stderr, "thread OOM\n");
        free(denom); free(inv_den); free(prefix); return NULL;
    }

    /* ── Phase 1: compute denominators denom[k] = Pm.x - T2[j].x ── */
    for (uint64_t k = 0; k < chunk; k++) {
        uint64_t j = a->j_start + k;
        secp256k1_fe tx=a->T2[j-1].x;
        secp256k1_fe_normalize_var(&tx);
        secp256k1_fe_negate(&denom[k],&tx,1);
        secp256k1_fe_add(&denom[k],&a->Pm_x);
        secp256k1_fe_normalize_var(&denom[k]);
        if(secp256k1_fe_is_zero(&denom[k])){
            int64_t m1 = j*b->M;
            int64_t m2 = -j*b->M;
            if(verify_candidate(b->ctx,m1,a->target33)){
                a->result_m=m1; a->found=1;
                atomic_store(a->found_flag,1);
            } else if(verify_candidate(b->ctx,m2,a->target33)){
                a->result_m=m2; a->found=1;
                atomic_store(a->found_flag,1);
            }
            /* skip Phase 2+3 — denom contains zero, cannot invert */
            free(denom); free(inv_den); free(prefix);
            return NULL;
        }
    }

    /* ── Phase 2: Montgomery batch inversion over chunk denominators ── */

    prefix[0] = denom[0];
    for (uint64_t k = 1; k < chunk; k++)
        secp256k1_fe_mul(&prefix[k], &prefix[k-1], &denom[k]);

    secp256k1_fe acc;
    secp256k1_fe_inv(&acc, &prefix[chunk-1]);   /* single inversion */

    for (uint64_t k = chunk-1; k >= 1; k--) {
        secp256k1_fe_mul(&inv_den[k], &prefix[k-1], &acc);
        secp256k1_fe_mul(&acc, &acc, &denom[k]);
    }
    inv_den[0] = acc;

    free(prefix); free(denom);

    /* ── Phase 3: compute x(Pm - T2[j]) and look up ── */
    for (uint64_t k = 0; k < chunk && !atomic_load(a->found_flag); k++) {
        uint64_t j = a->j_start + k;

        secp256k1_fe tx = a->T2[j-1].x, ty = a->T2[j-1].y;
        secp256k1_fe_normalize_var(&tx);
        secp256k1_fe_normalize_var(&ty);

        /* lambda = (Pm.y + T2[j].y) * inv_denom
         * Because Pm - T2[j] = Pm + (-T2[j]), and -T2[j] has y = -ty,
         * so lambda = (Pm.y - (-ty)) / (Pm.x - tx) = (Pm.y + ty) / (Pm.x - tx) */
        secp256k1_fe num = a->Pm_y;
        secp256k1_fe_add(&num, &ty);
        secp256k1_fe lam;
        secp256k1_fe_mul(&lam, &num, &inv_den[k]);

        /* x3 = lambda^2 - Pm.x - T2[j].x */
        secp256k1_fe lam2, x3, neg_sum_x1_x2;
        secp256k1_fe_sqr(&lam2, &lam);
        x3 = lam2;

        neg_sum_x1_x2 = a->Pm_x;
        secp256k1_fe_add(&neg_sum_x1_x2, &tx);
        secp256k1_fe_negate(&neg_sum_x1_x2, &neg_sum_x1_x2, 1);
        secp256k1_fe_add(&x3, &neg_sum_x1_x2);
        secp256k1_fe_normalize_var(&x3);

        /* Lookup */
        unsigned char xb[32]; secp256k1_fe_get_b32(xb, &x3);
        uint32_t cands[CUCKOO_K + CUCKOO_STASH_SZ];
        int nc = map_get_all(&b->baby, xb, cands);
        for (int ci = 0; ci < nc; ci++) {
            uint64_t m1 = j * b->M + (uint64_t)cands[ci];
            uint64_t m2 = j * b->M - (uint64_t)cands[ci];
            if (verify_candidate(b->ctx, m1, a->target33)) {
                a->result_m = m1; a->found = 1;
                atomic_store(a->found_flag, 1); break;
            }
            if (verify_candidate(b->ctx, m2, a->target33)) {
                a->result_m = m2; a->found = 1;
                atomic_store(a->found_flag, 1); break;
            }
        }

        if(atomic_load(a->found_flag)) break;
        /* lambda = (Pm.y - T2[j].y) * inv_den[j] */
        num = ty; secp256k1_fe_negate(&num,&num,1);
        secp256k1_fe_add(&num, &a->Pm_y);
        secp256k1_fe_mul(&lam, &num, &inv_den[k]);

        secp256k1_fe_sqr(&lam2, &lam);
        x3 = lam2;
        secp256k1_fe_add(&x3, &neg_sum_x1_x2);
        secp256k1_fe_normalize_var(&x3);

        secp256k1_fe_get_b32(xb,&x3);
        nc=map_get_all(&b->baby,xb,cands);
        for(int ci=0;ci<nc;ci++){
            int64_t m1=(int64_t)(-(int64_t)(j*b->M))+(uint64_t)cands[ci];
            int64_t m2=(int64_t)(-(int64_t)(j*b->M))-(uint64_t)cands[ci];
            if(verify_candidate(b->ctx,m1,a->target33)){
                a->result_m=m1;a->found=1;
                atomic_store(a->found_flag,1);break;}
            if(verify_candidate(b->ctx,m2,a->target33)){
                a->result_m=m2;a->found=1;
                atomic_store(a->found_flag,1);break;}
        }
    }

    free(inv_den);
    return NULL;
}

/* ─────────────────── parallel solve ─────────────────────────────── */
static int fastecdlp_solve_parallel(const bsgs_ctx *b,
                           const secp256k1_ge* T2,
                           const secp256k1_pubkey* targetPm,
                           int threads,
                           int64_t* out_m) {
    uint64_t J = b->J;

    unsigned char target33[33];
    if (!pubkey_serialize33(b->ctx, targetPm, target33)) return 0;

    secp256k1_ge Pm_ge;
    pubkey_to_ge(targetPm, &Pm_ge);

    secp256k1_fe Pm_x = Pm_ge.x; secp256k1_fe_normalize_var(&Pm_x);
    secp256k1_fe Pm_y = Pm_ge.y; secp256k1_fe_normalize_var(&Pm_y);
    
    /* j=0: direct baby lookup for Pm = i.G */
    unsigned char txb[32];
    memcpy(txb, target33 + 1, 32); /* x-coordinate from compressed pubkey */
    uint32_t cands[3 + CUCKOO_STASH_SZ];
    int nc = map_get_all(&b->baby, txb, cands);
    for (int ci = 0; ci < nc; ci++) {
        if (verify_candidate(b->ctx, (int64_t)((uint64_t)cands[ci]), target33)) {
            *out_m = (int64_t)(uint64_t)cands[ci]; return 1;
        } else if (verify_candidate(b->ctx, -(int64_t)((uint64_t)cands[ci]), target33)) {
            *out_m = -(int64_t)(uint64_t)cands[ci]; return 1;
        }
    }

    uint64_t chunk = (J + (uint64_t)threads - 1)/(uint64_t)threads;
    pthread_t*   tids = (pthread_t*)  malloc((size_t)threads * sizeof(pthread_t));
    thread_args* args = (thread_args*)malloc((size_t)threads * sizeof(thread_args));
    atomic_int found_flag; atomic_init(&found_flag, 0);

    uint64_t jcur = 1;
    for (int t = 0; t < threads; t++) {
        args[t].b          = b;
        args[t].T2         = T2;
        args[t].Pm_x       = Pm_x;
        args[t].Pm_y       = Pm_y;
        args[t].j_start    = jcur;
        args[t].j_end      = jcur + chunk < J + 1 ?
                             jcur + chunk : J + 1 ; /* j_end is not included */
        args[t].target33   = target33;
        args[t].found_flag = &found_flag;
        args[t].result_m   = 0;
        args[t].found      = 0;
        pthread_create(&tids[t], NULL, thread_fn, &args[t]);
        jcur += chunk;
    }

    for (int t = 0; t < threads; t++) pthread_join(tids[t], NULL);

    int found = 0;
    for (int t = 0; t < threads; t++)
        if (args[t].found) { *out_m = args[t].result_m; found = 1; break; }

    free(tids); free(args);
    return found;
}
/* ─────────────────────── Wrapper ─────────────────────────────── */
/*
 * Wrapper allowing the recovery of an unsigned message m in [0, 2^l) 
 * using `fastecdlp_solve_parallel`, which operates in [-2^(l-1), 2^(l-1)). 
 */
static int fastecdlp_solve_parallel_wrapper(const bsgs_ctx* restrict b,
                           const secp256k1_ge* restrict T2,
                           const secp256k1_pubkey* restrict targetPm,
                           int threads,
                           uint64_t* out_m) {

    /* Compute Pm' = Pm - 2^(bits_total-1).G = Pm + (-offset.G) */
    secp256k1_pubkey Pm_adj_pk;  
    const secp256k1_pubkey *points[2] = { targetPm, &b->neg_offsetG_pk };
    secp256k1_ec_pubkey_combine(b->ctx, &Pm_adj_pk, points, 2);
    
    int found;
    int64_t out_m_signed = 0;
    found = fastecdlp_solve_parallel(b,T2,&Pm_adj_pk,threads,&out_m_signed);
    *out_m = out_m_signed + b->offset;
    return found;
}



/* ─────────────────────── benchmark ─────────────────────────────── */
static void benchmark_bsgs(int bits_total, int l1, int trials, int threads, unsigned int seed) {
    if (bits_total == 0 || bits_total > 64) {
        fprintf(stderr,"Error: invalid bits_total value: %i\n",bits_total);
        return;
    }

    if (trials <= 0) {
        fprintf(stderr, "Invalid trials: %d\n", trials);
        return;
    }

    int l2=bits_total-l1;
    uint64_t J = 1ULL << (l2 - 1);

    uint64_t chunk = (J + (uint64_t)threads - 1) / (uint64_t)threads;
    double mem_t2_gb  = (double)(J * sizeof(secp256k1_ge)) / (1ULL<<30);
    double mem_thr_gb = (double)(chunk * 3 * sizeof(secp256k1_fe)) / (1ULL<<30);

    printf("=== FastECDLP Parallel (Tang et al. + our parallel Ph.1+2) "
           "— secp256k1, k=3 cuckoo, T2 table ===\n");
    printf("Range   : m in [0, 2^%d)\n", bits_total);
    printf("Split   : l1=%d, l2=%d  (J=%"PRIu64")\n", l1, l2, J);
    printf("Threads : %d  (chunk=%"PRIu64" steps/thread)\n", threads, chunk);
    printf("T2 mem  : %.2f GB (precomputed, one-time)\n", mem_t2_gb);
    printf("Per-thr : %.2f GB (denom + prefix + inv arrays)\n", mem_thr_gb);
    printf("Trials  : %d\n\n", trials);

    secp256k1_context* ctx =
            secp256k1_context_create(SECP256K1_CONTEXT_SIGN|SECP256K1_CONTEXT_VERIFY);

    if (ctx == NULL) {fprintf(stderr,"Error: could not create secp256k1 context\n");
        return;
    }

    bsgs_ctx solver;
    if (!bsgs_ctx_init(&solver, ctx, bits_total, l1)) {
        printf("Failed to init solver\n");
        secp256k1_context_destroy(ctx); return;
    }   


    /* Get T2 (load from cache or build) */
    double t2_time;
    secp256k1_ge* T2 = get_t2(ctx, solver.M, l1, l2, &t2_time);
    if (!T2) {
        fprintf(stderr, "Failed to get T2\n");
        bsgs_ctx_free(&solver); secp256k1_context_destroy(ctx); return;
    }
    if (t2_time > 0)
        printf("T2 build time: %.1f sec (one-time, cached for future runs)\n\n",
               t2_time);
    else
        printf("\n");

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
        
        ts = now_seconds();
        uint64_t recovered=0;
        int found = fastecdlp_solve_parallel_wrapper(&solver, T2, &Pm_pk, threads, &recovered);
        recovery_time += now_seconds() - ts;
        
        if(found && (recovered == m))
            ok++;
        else
            printf("Trial %d FAILED: m=%"PRIu64" recovered=%"PRIu64"\n",
                t, m, recovered);
    }

    printf("Random Seed      : %u\n", seed);
    printf("Solved correctly : %d/%d\n", ok, trials);
    printf("Total search time: %.3f sec\n", recovery_time);
    printf("Average per solve: %.3f sec (%.2f ms)\n\n",
           recovery_time/trials,(recovery_time/trials)*1e3);
    printf("Note: T2 build (%.1f sec) excluded above.\n",t2_time);

    free(T2); bsgs_ctx_free(&solver);
    secp256k1_context_destroy(ctx);
}

int main(int argc, char** argv) {
    int      bits_total = 40, l1 = 18, trials = 1, threads = 1;
    unsigned int seed = (unsigned int)time(NULL);

    if (argc >= 2) bits_total = atoi(argv[1]);
    if (argc >= 3) l1         = atoi(argv[2]);
    if (argc >= 4) trials     = atoi(argv[3]);
    if (argc >= 5) threads    = atoi(argv[4]);
    if (argc >= 6) {
        /* seed = 0: benchmark the worst case (m = 2^l - 1) */
        seed = (unsigned int)strtoull(argv[5], NULL, 10);
    }

    srand(seed);
    benchmark_bsgs(bits_total, l1, trials, threads, seed);
    return 0;
}