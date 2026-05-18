/*
 * fastecdlp_original.c
 *
 * Original FastECDLP algorithm (Tang et al., ePrint 2022/1573) on secp256k1.
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
 * T2 memory: 2^l2 * sizeof(secp256k1_ge) bytes
 *   l2=22 (52-bit): ~1.1 GB     l2=24 (54-bit): ~4.3 GB
 *
 * T2 cache file: fastecdlp_t2_secp256k1_l1_<l1>.bin
 *
 * Build:
 *   cc -O3 -Wall -Wextra -o fastecdlp_original fastecdlp_original.c \
 *       -I/usr/local/include                                          \
 *       -I/path/to/secp256k1/src                                      \
 *       -L/usr/local/lib                                              \
 *       -lsecp256k1 -lpthread
 *
 * Usage: ./fastecdlp_original <bits> <l1> <trials> <threads>
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

static int verify_candidate(const secp256k1_context* ctx,
                            uint64_t m_cand,
                            const unsigned char target33[33]) {
    if (m_cand == 0) return 0;
    unsigned char sc[32]; u64_to_scalar32_be(m_cand, sc);
    secp256k1_pubkey pk;
    if (!secp256k1_ec_pubkey_create(ctx, &pk, sc)) return 0;
    unsigned char got[33]; size_t glen = 33;
    secp256k1_ec_pubkey_serialize(ctx, got, &glen, &pk, SECP256K1_EC_COMPRESSED);
    return memcmp(got, target33, 33) == 0;
}

/* ─────────────────────── I/O ─────────────────────────────────────── */
static int read_all(int fd, void* buf, size_t len) {
    unsigned char* p = (unsigned char*)buf;
    while (len > 0) {
        size_t chunk = len < (1ULL<<30) ? len : (1ULL<<30);
        ssize_t n = read(fd, p, chunk);
        if (n <= 0) return 0;
        p += n; len -= (size_t)n;
    }
    return 1;
}
static int write_all(int fd, const void* buf, size_t len) {
    const unsigned char* p = (const unsigned char*)buf;
    while (len > 0) {
        size_t chunk = len < (1ULL<<30) ? len : (1ULL<<30);
        ssize_t n = write(fd, p, chunk);
        if (n <= 0) return 0;
        p += n; len -= (size_t)n;
    }
    return 1;
}

/* ─────────────────────── cuckoo table ───────────────────────────── */
#define CUCKOO_K        3
#define CUCKOO_STASH_SZ 16
#define CUCKOO_SEED1    0x9e3779b97f4a7c15ULL
#define CUCKOO_SEED2    0xd1b54a32d192ed03ULL
#define BABY_MAGIC      0x4B43554B4F4F4355ULL

typedef struct { uint32_t key; uint32_t val; } entry_packed;
typedef struct {
    entry_packed* tab;
    size_t section_size, total_bins, size;
    uint64_t stash_x64[CUCKOO_STASH_SZ];
    uint32_t stash_val[CUCKOO_STASH_SZ];
    int stash_count;
} cuckoo_map;
typedef struct {
    uint64_t magic; uint32_t version, l1;
    uint64_t section_size, used_count;
    int32_t stash_count; uint32_t _pad;
} baby_hdr;

static uint64_t mix64(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31; return x;
}
static size_t fastrange64(uint64_t h, size_t n) {
    return (size_t)((__uint128_t)h * (__uint128_t)n >> 64);
}
static size_t cpos(int sec, uint64_t x64, size_t s) {
    uint64_t seed = (sec==0)?0ULL:(sec==1)?CUCKOO_SEED1:CUCKOO_SEED2;
    return fastrange64(mix64(x64^seed), s) + (size_t)sec * s;
}
static int map_get_all(const cuckoo_map* m, uint64_t x64, uint32_t* out) {
    uint32_t k=(uint32_t)(x64>>32); size_t s=m->section_size; int n=0;
    const entry_packed* e;
    e=&m->tab[cpos(0,x64,s)]; if(e->val&&e->key==k) out[n++]=e->val;
    e=&m->tab[cpos(1,x64,s)]; if(e->val&&e->key==k) out[n++]=e->val;
    e=&m->tab[cpos(2,x64,s)]; if(e->val&&e->key==k) out[n++]=e->val;
    for(int i=0;i<m->stash_count;i++) if(m->stash_x64[i]==x64) out[n++]=m->stash_val[i];
    return n;
}
static int load_baby_table(cuckoo_map* baby, int l1) {
    char fname[128];
    snprintf(fname,sizeof(fname),"bsgs_baby_cuckoo_secp256k1_l1_%d.bin",l1);
    int fd=open(fname,O_RDONLY); if(fd<0){perror(fname);return 0;}
    baby_hdr hdr;
    if(!read_all(fd,&hdr,sizeof(hdr))||hdr.magic!=BABY_MAGIC||
       hdr.version!=4||(int)hdr.l1!=l1){fprintf(stderr,"Bad header %s\n",fname);close(fd);return 0;}
    memset(baby,0,sizeof(*baby));
    baby->section_size=(size_t)hdr.section_size;
    baby->total_bins=CUCKOO_K*(size_t)hdr.section_size;
    baby->size=(size_t)hdr.used_count; baby->stash_count=(int)hdr.stash_count;
    if(!read_all(fd,baby->stash_x64,sizeof(baby->stash_x64))||
       !read_all(fd,baby->stash_val,sizeof(baby->stash_val))){close(fd);return 0;}
    baby->tab=(entry_packed*)calloc(baby->total_bins,sizeof(entry_packed));
    if(!baby->tab||!read_all(fd,baby->tab,baby->total_bins*sizeof(entry_packed))){
        free(baby->tab);close(fd);return 0;}
    close(fd);
    printf("Baby table: %s (section=%zu, total=%zu, stash=%d, %.2f MB)\n",
           fname,baby->section_size,baby->total_bins,baby->stash_count,
           (double)(baby->total_bins*sizeof(entry_packed))/(1<<20));
    return 1;
}

/* ─────────────────── T2 table: precompute and cache ─────────────── */
/*
 * T2[j] = j * M * G  in affine coordinates, for j = 0..2^l2-1.
 *
 * Built once, saved to disk as fastecdlp_t2_secp256k1_l1_<l1>.bin.
 * Reused across all solves — this is FastECDLP's key precomputation.
 *
 * Memory: 2^l2 * sizeof(secp256k1_ge) bytes.
 */
static void t2_cache_path(char* out, size_t outlen, int l1) {
    snprintf(out, outlen, "fastecdlp_t2_secp256k1_l1_%d.bin", l1);
}

static secp256k1_ge* build_t2(const secp256k1_context* ctx,
                              uint64_t M, int l2,
                              double* build_time_out) {
    uint64_t J = 1ULL << l2;
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
        jac[j] = acc;
        secp256k1_gej_add_ge(&acc, &acc, &MG_ge);
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
    uint64_t J = 1ULL << l2;
    *build_time_out = 0.0;

    struct stat st;
    if (stat(path, &st) == 0 &&
        (uint64_t)st.st_size == J * sizeof(secp256k1_ge)) {
        printf("Loading T2 from cache: %s (%.2f GB)...\n",
               path, (double)st.st_size / (1ULL<<30));
        double t0 = now_seconds();
        secp256k1_ge* T2 = load_t2(path, J);
        if (T2) {
            printf("T2 loaded: %.2f sec\n", now_seconds() - t0);
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
    const cuckoo_map*      baby;
    const secp256k1_context* ctx;
    const secp256k1_ge*    T2;
    secp256k1_fe           Pm_x;   /* normalized */
    secp256k1_fe           Pm_y;   /* normalized */
    uint64_t               M;
    uint64_t               j_start;
    uint64_t               j_end;
    const unsigned char*   target33;
    atomic_int*            found_flag;
    uint64_t               result_m;
    int                    found;
} thread_args;

static void* thread_fn(void* varg) {
    thread_args* a = (thread_args*)varg;
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
        if (a->T2[j].infinity) {
            secp256k1_fe_set_int(&denom[k], 1); /* placeholder */
        } else {
            secp256k1_fe tx = a->T2[j].x;
            secp256k1_fe_normalize_var(&tx);
            secp256k1_fe_negate(&denom[k], &tx, 1);
            secp256k1_fe_add(&denom[k], &a->Pm_x);
            secp256k1_fe_normalize_var(&denom[k]);
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

        if (a->T2[j].infinity) {
            /* T2[j] = 0 => Pm - 0 = Pm => m = j*M */
            if (verify_candidate(a->ctx, j * a->M, a->target33)) {
                a->result_m = j * a->M; a->found = 1;
                atomic_store(a->found_flag, 1);
            }
            continue;
        }

        secp256k1_fe tx = a->T2[j].x, ty = a->T2[j].y;
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
        secp256k1_fe lam2, x3;
        secp256k1_fe_sqr(&lam2, &lam);
        x3 = lam2;
        secp256k1_fe neg_px = a->Pm_x;
        secp256k1_fe_negate(&neg_px, &neg_px, 1);
        secp256k1_fe_add(&x3, &neg_px);
        secp256k1_fe neg_tx = tx;
        secp256k1_fe_negate(&neg_tx, &neg_tx, 1);
        secp256k1_fe_add(&x3, &neg_tx);
        secp256k1_fe_normalize_var(&x3);

        /* Extract x64 */
        unsigned char buf[32];
        secp256k1_fe_get_b32(buf, &x3);
        uint64_t x64 = 0;
        for (int i = 0; i < 8; i++) x64 = (x64 << 8) | buf[i];

        /* Lookup */
        uint32_t cands[CUCKOO_K + CUCKOO_STASH_SZ];
        int nc = map_get_all(a->baby, x64, cands);
        for (int ci = 0; ci < nc; ci++) {
            uint64_t m1 = j * a->M + (uint64_t)cands[ci];
            uint64_t m2 = j * a->M - (uint64_t)cands[ci];
            if (verify_candidate(a->ctx, m1, a->target33)) {
                a->result_m = m1; a->found = 1;
                atomic_store(a->found_flag, 1); break;
            }
            if (verify_candidate(a->ctx, m2, a->target33)) {
                a->result_m = m2; a->found = 1;
                atomic_store(a->found_flag, 1); break;
            }
        }
    }

    free(inv_den);
    return NULL;
}

/* ─────────────────── parallel solve ─────────────────────────────── */
static int fastecdlp_solve(const cuckoo_map* baby,
                           const secp256k1_context* ctx,
                           const secp256k1_ge* T2,
                           const secp256k1_ge* Pm_ge,
                           uint64_t M, int l2, int threads,
                           const unsigned char target33[33],
                           uint64_t* out_m) {
    uint64_t J     = 1ULL << l2;
    uint64_t chunk = (J + (uint64_t)threads - 1) / (uint64_t)threads;

    secp256k1_fe Pm_x = Pm_ge->x; secp256k1_fe_normalize_var(&Pm_x);
    secp256k1_fe Pm_y = Pm_ge->y; secp256k1_fe_normalize_var(&Pm_y);

    pthread_t*   tids = (pthread_t*)  malloc((size_t)threads * sizeof(pthread_t));
    thread_args* args = (thread_args*)malloc((size_t)threads * sizeof(thread_args));
    atomic_int found_flag; atomic_init(&found_flag, 0);

    for (int t = 0; t < threads; t++) {
        args[t].baby       = baby;
        args[t].ctx        = ctx;
        args[t].T2         = T2;
        args[t].Pm_x       = Pm_x;
        args[t].Pm_y       = Pm_y;
        args[t].M          = M;
        args[t].j_start    = (uint64_t)t * chunk;
        args[t].j_end      = (uint64_t)t * chunk + chunk < J
                             ? (uint64_t)t * chunk + chunk : J;
        args[t].target33   = target33;
        args[t].found_flag = &found_flag;
        args[t].result_m   = 0;
        args[t].found      = 0;
        pthread_create(&tids[t], NULL, thread_fn, &args[t]);
    }

    for (int t = 0; t < threads; t++) pthread_join(tids[t], NULL);

    int found = 0;
    for (int t = 0; t < threads; t++)
        if (args[t].found) { *out_m = args[t].result_m; found = 1; break; }

    free(tids); free(args);
    return found;
}

/* ─────────────────────── benchmark ─────────────────────────────── */
static void benchmark(int bits, int l1, int trials, int threads) {
    int l2 = bits - l1;
    uint64_t M = 1ULL << l1;
    uint64_t J = 1ULL << l2;
    uint64_t chunk = (J + (uint64_t)threads - 1) / (uint64_t)threads;
    double mem_t2_gb  = (double)(J * sizeof(secp256k1_ge)) / (1ULL<<30);
    double mem_thr_gb = (double)(chunk * 2 * sizeof(secp256k1_fe)) / (1ULL<<30);

    printf("=== FastECDLP Original (Tang et al.) "
           "— secp256k1, k=3 cuckoo, T2 table ===\n");
    printf("Range   : m in [0, 2^%d)\n", bits);
    printf("Split   : l1=%d, l2=%d  (J=%"PRIu64")\n", l1, l2, J);
    printf("Threads : %d  (chunk=%"PRIu64" steps/thread)\n", threads, chunk);
    printf("T2 mem  : %.2f GB (precomputed, one-time)\n", mem_t2_gb);
    printf("Per-thr : %.2f GB (denom + inv arrays)\n", mem_thr_gb);
    printf("Trials  : %d\n\n", trials);

    secp256k1_context* ctx =
            secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);

    /* Load baby table */
    cuckoo_map baby; memset(&baby, 0, sizeof(baby));
    if (!load_baby_table(&baby, l1)) {
        secp256k1_context_destroy(ctx); return;
    }

    /* Get T2 (load from cache or build) */
    double t2_build_time;
    secp256k1_ge* T2 = get_t2(ctx, M, l1, l2, &t2_build_time);
    if (!T2) {
        fprintf(stderr, "Failed to get T2\n");
        free(baby.tab); secp256k1_context_destroy(ctx); return;
    }
    if (t2_build_time > 0)
        printf("T2 build time: %.1f sec (one-time, cached for future runs)\n\n",
               t2_build_time);
    else
        printf("\n");

    /* Run trials */
    uint64_t mask = (bits == 64) ? ~0ULL : ((1ULL << bits) - 1ULL);
    int ok = 0;
    double ts = now_seconds();

    for (int t = 0; t < trials; t++) {
        uint64_t m = ((uint64_t)rand() << 32) ^ (uint64_t)rand();
        m &= mask; if (m == 0) m = 1;

        unsigned char sc[32]; u64_to_scalar32_be(m, sc);
        secp256k1_pubkey Pm_pk;
        secp256k1_ec_pubkey_create(ctx, &Pm_pk, sc);
        secp256k1_ge Pm_ge; pubkey_to_ge(&Pm_pk, &Pm_ge);
        unsigned char t33[33]; size_t tlen = 33;
        secp256k1_ec_pubkey_serialize(ctx, t33, &tlen, &Pm_pk, SECP256K1_EC_COMPRESSED);

        uint64_t recovered = 0;
        if (fastecdlp_solve(&baby, ctx, T2, &Pm_ge, M, l2, threads, t33, &recovered)
            && recovered == m)
            ok++;
        else
            printf("Trial %d FAILED: m=%"PRIu64" recovered=%"PRIu64"\n",
                t, m, recovered);
    }

    double te = now_seconds();
    double total = te - ts;
    printf("Solved correctly : %d/%d\n", ok, trials);
    printf("Total time       : %.3f sec\n", total);
    printf("Average per solve: %.3f sec (%.2f ms)\n\n",
           total / trials, (total / trials) * 1e3);
    printf("Note: T2 build time (%.1f sec) is one-time and excluded above.\n",
           t2_build_time);

    free(T2); free(baby.tab);
    secp256k1_context_destroy(ctx);
}

int main(int argc, char** argv) {
    int bits = 52, l1 = 30, trials = 5, threads = 10;
    if (argc >= 2) bits    = atoi(argv[1]);
    if (argc >= 3) l1      = atoi(argv[2]);
    if (argc >= 4) trials  = atoi(argv[3]);
    if (argc >= 5) threads = atoi(argv[4]);
    srand((unsigned)time(NULL));
    benchmark(bits, l1, trials, threads);
    return 0;
}