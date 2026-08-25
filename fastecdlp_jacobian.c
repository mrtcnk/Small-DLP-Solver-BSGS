/*
 * fastecdlp_jacobian.c  —  parallel FastECDLP implementation
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
 *   cc -O3 -Wall -Wextra -o fastecdlp_jacobian fastecdlp_jacobian.c \
 *       -I/usr/local/include                                          \
 *       -I/path/to/secp256k1/src                                      \
 *       -L/usr/local/lib                                              \
 *       -lsecp256k1 -lpthread
 *
 * Usage: ./fastecdlp_jacobian <bits_total> <l1> <trials> <threads>
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


/* ─────────────────────── timing ─────────────────────────────────── */

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ─────────────────────── helpers ─────────────────────────────────── */

static void u64_to_scalar32_be(uint64_t x, unsigned char out32[32]) {
    memset(out32, 0, 32);
    for (int i = 0; i < 8; i++) {
        out32[31 - i] = (unsigned char)(x & 0xFF);
        x >>= 8;
    }
}

static int file_exists(const char* path) {
    struct stat st;
    return stat(path, &st) == 0;
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

static void gej_xb32_from_zinv(const secp256k1_gej* pt,
                               const secp256k1_fe*  z_inv,
                               unsigned char buf[32]) {
    secp256k1_fe z2, x;
    secp256k1_fe_sqr(&z2, z_inv);
    secp256k1_fe_mul(&x, &pt->x, &z2);
    secp256k1_fe_normalize_var(&x);
    secp256k1_fe_get_b32(buf, &x);
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
    size_t s = m->section_size; int n = 0;
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

typedef struct {
    secp256k1_context* ctx;
    int bits_total, l1;
    uint64_t M, Mhalf, J;

    secp256k1_pubkey G, MG;
    secp256k1_ge MG_ge;
    secp256k1_ge neg_MG_ge;

    cuckoo_map baby;   /* ← cuckoo hash table */
} bsgs_ctx;

static int bsgs_ctx_init(bsgs_ctx* b, secp256k1_context* ctx,
                                int bits_total, int l1) {
    memset(b, 0, sizeof(*b));
    b->ctx = ctx; b->bits_total = bits_total; b->l1 = l1;
    if (bits_total <= 0 || bits_total > 63) return 0;
    if (l1 <= 0 || l1 >= bits_total)        return 0;

    b->M     = 1ULL << l1;
    b->J     = 1ULL << (bits_total - l1);

    unsigned char Msc[32];
    u64_to_scalar32_be(b->M, Msc);
    secp256k1_pubkey MG;
    if (!secp256k1_ec_pubkey_create(ctx, &MG, Msc)) return 0;

    pubkey_to_ge(&MG, &b->MG_ge);
    ge_negate(&b->neg_MG_ge, &b->MG_ge);

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
        printf("Generate baby table first by running bsgs!\n");
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


/* ─────────────────────── thread work ────────────────────────────── */

typedef struct {
    const bsgs_ctx*  b;
    secp256k1_ge          Q_start;   /* Pm - j_start*M*G  (affine) */
    uint64_t              j_start;
    uint64_t              j_end;     /* exclusive */
    atomic_int*           found_flag;
    const unsigned char*     target33;

    /* output */
    uint64_t              result_m;
    int                   found;
} thread_args_jac;

static void* thread_fn_jac(void* arg) {
    thread_args_jac* a = (thread_args_jac*)arg;
    const bsgs_ctx *b = a->b;
    uint64_t chunk = a->j_end - a->j_start;

    /* ── Phase 1: walk chunk points in Jacobian ── */
    secp256k1_gej* Qjac = (secp256k1_gej*)malloc(chunk * sizeof(secp256k1_gej));
    if (!Qjac) { fprintf(stderr, "thread OOM: Qjac\n"); return NULL; }

    secp256k1_gej Q;
    secp256k1_gej_set_ge(&Q, &a->Q_start);
    for (uint64_t k = 0; k < chunk; k++) {
        Qjac[k] = Q;
        secp256k1_gej_add_ge(&Q, &Q, &b->neg_MG_ge);
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
            if (verify_candidate(b->ctx, j * b->M, a->target33)) {
                a->result_m = j * b->M; a->found = 1;
                atomic_store(a->found_flag, 1);
            }
            continue;
        }
        unsigned char xb[32];
        gej_xb32_from_zinv(&Qjac[k], &z_inv[k], xb);
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
    }

    free(Qjac); free(z_inv);
    return NULL;
}


/* ─────────────────────── parallel solve ─────────────────────────── */

static int fastecdlp_solve_jacobian(const bsgs_ctx *b,
                           const secp256k1_pubkey* targetPm,
                           int threads,
                           uint64_t* out_m) {
    uint64_t J = b->J + 1;
    uint64_t chunk = (J + (uint64_t)threads - 1) / (uint64_t)threads;

    unsigned char target33[33];
    if (!pubkey_serialize33(b->ctx, targetPm, target33)) return 0;

    secp256k1_ge Pm_ge;
    pubkey_to_ge(targetPm, &Pm_ge);

    pthread_t*    tids = (pthread_t*)   malloc((size_t)threads * sizeof(pthread_t));
    thread_args_jac*  args = (thread_args_jac*) malloc((size_t)threads * sizeof(thread_args_jac));
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
    u64_to_scalar32_be(chunk * b->M, sc_chunkM);
    secp256k1_pubkey pk_chunkMG;
    secp256k1_ec_pubkey_create(b->ctx, &pk_chunkMG, sc_chunkM);
    secp256k1_ge chunkMG_ge; pubkey_to_ge(&pk_chunkMG, &chunkMG_ge);
    secp256k1_ge neg_chunkMG_ge; ge_negate(&neg_chunkMG_ge, &chunkMG_ge);

    /* Build start points: Q_start[t] = Pm - t*chunk*M*G */
    secp256k1_gej Q_cur;
    secp256k1_gej_set_ge(&Q_cur, &Pm_ge);

    for (int t = 0; t < threads; t++) {
        uint64_t j_start = (uint64_t)t * chunk;
        uint64_t j_end   = j_start + chunk < J ? j_start + chunk : J;

        /* Convert Q_cur to affine for thread start */
        secp256k1_ge Q_start_ge;
        secp256k1_gej Q_cur_copy = Q_cur;
        secp256k1_ge_set_gej(&Q_start_ge, &Q_cur_copy);

        args[t].b           = b;
        args[t].Q_start     = Q_start_ge;
        args[t].j_start     = j_start;
        args[t].j_end       = j_end;
        args[t].target33    = target33;
        args[t].found_flag  = &found_flag;
        args[t].result_m    = 0;
        args[t].found       = 0;

        /* Advance Q_cur by -chunk*M*G for next thread's start */
        if (t + 1 < threads)
            secp256k1_gej_add_ge(&Q_cur, &Q_cur, &neg_chunkMG_ge);

        pthread_create(&tids[t], NULL, thread_fn_jac, &args[t]);
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

static void benchmark_bsgs(int bits_total, int l1, int trials, int threads, unsigned int seed) {
    if (bits_total == 0 || bits_total > 64) {
        fprintf(stderr,"Error: invalid bits_total value: %i\n",bits_total);
        return;
    }

    if (trials <= 0) {
        fprintf(stderr, "Invalid trials: %d\n", trials);
        return;
    }
    
    int l2 = bits_total - l1;
    uint64_t J = 1ULL << l2;
    uint64_t chunk = (J + (uint64_t)threads - 1) / (uint64_t)threads;
    double mem_per_thread_gb =
            (double)(chunk * (sizeof(secp256k1_gej) + 2*sizeof(secp256k1_fe)))
            / (1ULL<<30);

    printf("=== FastECDLP Jacobian "
           "(secp256k1, k=3 cuckoo, parallel full-batch) ===\n");
    printf("Range   : m in [0, 2^%d)\n", bits_total);
    printf("Split   : l1=%d, l2=%d  (J=%"PRIu64")\n", l1, l2, J);
    printf("Threads : %d  (chunk=%"PRIu64" steps/thread)\n", threads, chunk);
    printf("Memory  : %.2f GB/thread  (%.2f GB total working)\n",
           mem_per_thread_gb, mem_per_thread_gb * threads);
    printf("Entry   : %zu bytes lookup | Trials: %d\n\n",
           sizeof(entry_packed), trials);

    secp256k1_context* ctx =
            secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);

    if (ctx == NULL) {fprintf(stderr,"Error: could not create secp256k1 context\n");
        return;
    }

    bsgs_ctx solver;
    if (!bsgs_ctx_init(&solver, ctx, bits_total, l1)) {
        printf("Failed to init solver\n");
        secp256k1_context_destroy(ctx); return;
    }

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
        int found = fastecdlp_solve_jacobian(&solver, &Pm_pk, threads, &recovered);
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

    bsgs_ctx_free(&solver);
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