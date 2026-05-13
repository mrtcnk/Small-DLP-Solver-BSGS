/*
 * fastecdlp_baseline.c  —  parallel FastECDLP implementation
 *
 * Faithful multi-threaded implementation of FastECDLP
 * (Tang et al., ePrint 2022/1573) on secp256k1.
 *
 * Parallelism: split giant-step range [0, J) across T threads.
 *   Thread t covers j in [t*chunk, (t+1)*chunk).
 *   Each thread computes its starting point independently,
 *   walks its chunk in Jacobian, batch-inverts its Z-coords,
 *   and looks up in the shared baby table.
 *
 * Per-thread working memory: chunk * (sizeof(gej) + 2*sizeof(fe))
 *   52-bit T=10: 4M/10 * ~200 bytes = ~80 MB/thread  (cache-friendly)
 *   54-bit T=10: 16M/10 * ~200 bytes = ~320 MB/thread
 *
 * Build:
 *   cc -O3 -Wall -Wextra -o fastecdlp_baseline fastecdlp_baseline.c \
 *       -I/usr/local/include                                          \
 *       -I/path/to/secp256k1/src                                      \
 *       -L/usr/local/lib                                              \
 *       -lsecp256k1 -lpthread
 *
 * Usage: ./fastecdlp_baseline <bits> <l1> <trials> <threads>
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

static void ge_negate(secp256k1_ge* out, const secp256k1_ge* in) {
    *out = *in;
    secp256k1_fe_negate(&out->y, &out->y, 1);
    secp256k1_fe_normalize_var(&out->y);
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
    uint64_t seed = (sec == 0) ? 0ULL :
                    (sec == 1) ? CUCKOO_SEED1 : CUCKOO_SEED2;
    return fastrange64(mix64(x64 ^ seed), s) + (size_t)sec * s;
}
static int map_get_all(const cuckoo_map* m, uint64_t x64, uint32_t* out) {
    uint32_t k = (uint32_t)(x64 >> 32); size_t s = m->section_size; int n = 0;
    const entry_packed* e;
    e=&m->tab[cpos(0,x64,s)]; if(e->val&&e->key==k) out[n++]=e->val;
    e=&m->tab[cpos(1,x64,s)]; if(e->val&&e->key==k) out[n++]=e->val;
    e=&m->tab[cpos(2,x64,s)]; if(e->val&&e->key==k) out[n++]=e->val;
    for(int i=0;i<m->stash_count;i++) if(m->stash_x64[i]==x64) out[n++]=m->stash_val[i];
    return n;
}

static int load_baby_table(cuckoo_map* baby, int l1) {
    char fname[128];
    snprintf(fname, sizeof(fname), "bsgs_baby_cuckoo_secp256k1_l1_%d.bin", l1);
    int fd = open(fname, O_RDONLY);
    if (fd < 0) { perror(fname); return 0; }
    baby_hdr hdr;
    if (!read_all(fd, &hdr, sizeof(hdr)) || hdr.magic != BABY_MAGIC ||
        hdr.version != 4 || (int)hdr.l1 != l1) {
        fprintf(stderr, "Bad header in %s\n", fname); close(fd); return 0;
    }
    memset(baby, 0, sizeof(*baby));
    baby->section_size = (size_t)hdr.section_size;
    baby->total_bins   = CUCKOO_K * (size_t)hdr.section_size;
    baby->size         = (size_t)hdr.used_count;
    baby->stash_count  = (int)hdr.stash_count;
    if (!read_all(fd, baby->stash_x64, sizeof(baby->stash_x64)) ||
        !read_all(fd, baby->stash_val, sizeof(baby->stash_val))) {
        close(fd); return 0;
    }
    baby->tab = (entry_packed*)calloc(baby->total_bins, sizeof(entry_packed));
    if (!baby->tab || !read_all(fd, baby->tab,
                                baby->total_bins * sizeof(entry_packed))) {
        free(baby->tab); close(fd); return 0;
    }
    close(fd);
    printf("Loaded baby table: %s (section=%zu, total=%zu, stash=%d)\n",
           fname, baby->section_size, baby->total_bins, baby->stash_count);
    printf("Baby table memory: %.2f MB\n",
           (double)(baby->total_bins * sizeof(entry_packed)) / (1<<20));
    return 1;
}

/* ─────────────────────── thread work ────────────────────────────── */

typedef struct {
    /* inputs (shared, read-only) */
    const cuckoo_map*     baby;
    const secp256k1_context* ctx;
    secp256k1_ge          Q_start;   /* Pm - j_start*M*G  (affine) */
    const secp256k1_ge*   neg_MG_ge;
    uint64_t              M;
    uint64_t              j_start;
    uint64_t              j_end;     /* exclusive */
    const unsigned char*  target33;
    atomic_int*           found_flag;

    /* output */
    uint64_t              result_m;
    int                   found;
} thread_args;

static void* thread_fn(void* arg) {
    thread_args* a = (thread_args*)arg;
    uint64_t chunk = a->j_end - a->j_start;

    /* ── Phase 1: walk chunk points in Jacobian ── */
    secp256k1_gej* Qjac = (secp256k1_gej*)malloc(chunk * sizeof(secp256k1_gej));
    if (!Qjac) { fprintf(stderr, "thread OOM: Qjac\n"); return NULL; }

    secp256k1_gej Q;
    secp256k1_gej_set_ge(&Q, &a->Q_start);
    for (uint64_t k = 0; k < chunk; k++) {
        Qjac[k] = Q;
        secp256k1_gej_add_ge(&Q, &Q, a->neg_MG_ge);
    }

    /* ── Phase 2: batch inversion over chunk ── */
    secp256k1_fe* prefix = (secp256k1_fe*)malloc(chunk * sizeof(secp256k1_fe));
    secp256k1_fe* z_inv  = (secp256k1_fe*)malloc(chunk * sizeof(secp256k1_fe));
    if (!prefix || !z_inv) {
        fprintf(stderr, "thread OOM: inversion\n");
        free(Qjac); free(prefix); free(z_inv); return NULL;
    }

    /* Forward pass */
    if (Qjac[0].infinity) secp256k1_fe_set_int(&prefix[0], 1);
    else prefix[0] = Qjac[0].z;
    for (uint64_t k = 1; k < chunk; k++) {
        if (Qjac[k].infinity) prefix[k] = prefix[k-1];
        else secp256k1_fe_mul(&prefix[k], &prefix[k-1], &Qjac[k].z);
    }

    /* Single inversion */
    secp256k1_fe acc;
    secp256k1_fe_inv(&acc, &prefix[chunk-1]);

    /* Backward pass */
    for (uint64_t k = chunk-1; k >= 1; k--) {
        if (!Qjac[k].infinity) {
            secp256k1_fe_mul(&z_inv[k], &prefix[k-1], &acc);
            secp256k1_fe_mul(&acc, &acc, &Qjac[k].z);
        }
    }
    z_inv[0] = acc;
    free(prefix);

    /* ── Phase 3: lookup ── */
    for (uint64_t k = 0; k < chunk && !atomic_load(a->found_flag); k++) {
        uint64_t j = a->j_start + k;
        if (Qjac[k].infinity) {
            if (verify_candidate(a->ctx, j * a->M, a->target33)) {
                a->result_m = j * a->M; a->found = 1;
                atomic_store(a->found_flag, 1);
            }
            continue;
        }
        uint64_t x64 = gej_x64_from_zinv(&Qjac[k], &z_inv[k]);
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

    free(Qjac); free(z_inv);
    return NULL;
}

/* ─────────────────────── parallel solve ─────────────────────────── */

static int fastecdlp_solve_parallel(
        const cuckoo_map* baby,
        const secp256k1_context* ctx,
        const secp256k1_ge* Pm_ge,
        const secp256k1_ge* neg_MG_ge,
        const secp256k1_ge* MG_ge,
        uint64_t M, int l2, int threads,
        const unsigned char target33[33],
        uint64_t* out_m) {

    uint64_t J     = 1ULL << l2;
    uint64_t chunk = (J + (uint64_t)threads - 1) / (uint64_t)threads;

    pthread_t*    tids = (pthread_t*)   malloc((size_t)threads * sizeof(pthread_t));
    thread_args*  args = (thread_args*) malloc((size_t)threads * sizeof(thread_args));
    atomic_int found_flag;
    atomic_init(&found_flag, 0);

    /*
     * Precompute each thread's starting point:
     *   Q_start[t] = Pm - t*chunk*M*G
     *
     * We walk from Pm, subtracting chunk*M*G for each thread's start.
     * chunk*M*G is computed via scalar multiplication (one-time).
     */
    unsigned char sc_chunkM[32];
    u64_to_scalar32_be(chunk * M, sc_chunkM);
    secp256k1_pubkey pk_chunkMG;
    secp256k1_ec_pubkey_create(ctx, &pk_chunkMG, sc_chunkM);
    secp256k1_ge chunkMG_ge; pubkey_to_ge(&pk_chunkMG, &chunkMG_ge);
    secp256k1_ge neg_chunkMG_ge; ge_negate(&neg_chunkMG_ge, &chunkMG_ge);

    /* Build start points: Q_start[t] = Pm - t*chunk*M*G */
    secp256k1_gej Q_cur;
    secp256k1_gej_set_ge(&Q_cur, Pm_ge);

    for (int t = 0; t < threads; t++) {
        uint64_t j_start = (uint64_t)t * chunk;
        uint64_t j_end   = j_start + chunk < J ? j_start + chunk : J;

        /* Convert Q_cur to affine for thread start */
        secp256k1_ge Q_start_ge;
        secp256k1_gej Q_cur_copy = Q_cur;
        secp256k1_ge_set_gej(&Q_start_ge, &Q_cur_copy);

        args[t].baby        = baby;
        args[t].ctx         = ctx;
        args[t].Q_start     = Q_start_ge;
        args[t].neg_MG_ge   = neg_MG_ge;
        args[t].M           = M;
        args[t].j_start     = j_start;
        args[t].j_end       = j_end;
        args[t].target33    = target33;
        args[t].found_flag  = &found_flag;
        args[t].result_m    = 0;
        args[t].found       = 0;

        /* Advance Q_cur by -chunk*M*G for next thread's start */
        if (t + 1 < threads)
            secp256k1_gej_add_ge(&Q_cur, &Q_cur, &neg_chunkMG_ge);

        pthread_create(&tids[t], NULL, thread_fn, &args[t]);
    }

    /* Wait for all threads */
    for (int t = 0; t < threads; t++)
        pthread_join(tids[t], NULL);

    /* Collect result */
    int found = 0;
    for (int t = 0; t < threads; t++) {
        if (args[t].found) { *out_m = args[t].result_m; found = 1; break; }
    }

    free(tids); free(args);
    return found;
}

/* ─────────────────────── benchmark ────────────────────────────────── */

static void benchmark(int bits, int l1, int trials, int threads) {
    int l2 = bits - l1;
    uint64_t M = 1ULL << l1;
    uint64_t J = 1ULL << l2;
    uint64_t chunk = (J + (uint64_t)threads - 1) / (uint64_t)threads;
    double mem_per_thread_gb =
            (double)(chunk * (sizeof(secp256k1_gej) + 2*sizeof(secp256k1_fe)))
            / (1ULL<<30);

    printf("=== FastECDLP Baseline "
           "(secp256k1, k=3 cuckoo, parallel full-batch) ===\n");
    printf("Range   : m in [0, 2^%d)\n", bits);
    printf("Split   : l1=%d, l2=%d  (J=%"PRIu64")\n", l1, l2, J);
    printf("Threads : %d  (chunk=%"PRIu64" steps/thread)\n", threads, chunk);
    printf("Memory  : %.2f GB/thread  (%.2f GB total working)\n",
           mem_per_thread_gb, mem_per_thread_gb * threads);
    printf("Entry   : %zu bytes lookup | Trials: %d\n\n",
           sizeof(entry_packed), trials);

    secp256k1_context* ctx =
            secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);

    /* Load baby table */
    cuckoo_map baby; memset(&baby, 0, sizeof(baby));
    double t0 = now_seconds();
    if (!load_baby_table(&baby, l1)) {
        secp256k1_context_destroy(ctx); return;
    }
    printf("Baby table init: %.6f sec\n\n", now_seconds() - t0);

    /* Precompute MG and neg_MG */
    unsigned char Msc[32]; u64_to_scalar32_be(M, Msc);
    secp256k1_pubkey pk_MG;
    secp256k1_ec_pubkey_create(ctx, &pk_MG, Msc);
    secp256k1_ge MG_ge;   pubkey_to_ge(&pk_MG, &MG_ge);
    secp256k1_ge neg_MG_ge; ge_negate(&neg_MG_ge, &MG_ge);

    /* Run trials */
    uint64_t mask = (bits == 64) ? ~0ULL : ((1ULL << bits) - 1ULL);
    int ok = 0;
    double ts = now_seconds();

    for (int t = 0; t < trials; t++) {
        uint64_t m = ((uint64_t)rand() << 32) ^ (uint64_t)rand();
        m &= mask; if (m == 0) m = 1;

        unsigned char sc[32]; u64_to_scalar32_be(m, sc);
        secp256k1_pubkey Pm;
        secp256k1_ec_pubkey_create(ctx, &Pm, sc);
        secp256k1_ge Pm_ge; pubkey_to_ge(&Pm, &Pm_ge);
        unsigned char t33[33]; size_t tlen = 33;
        secp256k1_ec_pubkey_serialize(ctx, t33, &tlen, &Pm, SECP256K1_EC_COMPRESSED);

        uint64_t recovered = 0;
        if (fastecdlp_solve_parallel(&baby, ctx, &Pm_ge, &neg_MG_ge, &MG_ge,
                                     M, l2, threads, t33, &recovered)
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

    free(baby.tab);
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