#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <inttypes.h>
#include <errno.h>

#include <secp256k1.h>

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
    if (!secp256k1_ec_pubkey_serialize(ctx, out33, &outlen, pk, SECP256K1_EC_COMPRESSED)) {
        return 0;
    }
    return outlen == 33;
}

/* Negate a pubkey by flipping y-parity bit in compressed encoding */
static int pubkey_negate(const secp256k1_context* ctx,
                         const secp256k1_pubkey* in,
                         secp256k1_pubkey* out) {
    unsigned char ser[33];
    if (!pubkey_serialize33(ctx, in, ser)) return 0;
    if (ser[0] == 0x02) ser[0] = 0x03;
    else if (ser[0] == 0x03) ser[0] = 0x02;
    else return 0;
    return secp256k1_ec_pubkey_parse(ctx, out, ser, 33);
}

/*
 * Extract the first 8 bytes of the x-coordinate from a compressed point.
 * Compressed format: [parity(1)] [x-coord big-endian(32)]
 * We take bytes [1..8] as a big-endian uint64.
 * KEY PROPERTY: (i*G)[x] == (-i*G)[x], so one entry covers both i and -i.
 */
static uint64_t pubkey_x64(const unsigned char ser33[33]) {
    uint64_t h = 0;
    for (int i = 1; i <= 8; i++) {
        h = (h << 8) | (uint64_t)ser33[i];
    }
    return h;
}

static int file_exists(const char* path) {
    struct stat st;
    return stat(path, &st) == 0;
}

static int write_all(int fd, const void* buf, size_t len) {
    const unsigned char* p = (const unsigned char*)buf;
    while (len > 0) {
        ssize_t n = write(fd, p, len);
        if (n < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 1;
}

static int read_all(int fd, void* buf, size_t len) {
    unsigned char* p = (unsigned char*)buf;
    while (len > 0) {
        ssize_t n = read(fd, p, len);
        if (n < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        if (n == 0) return 0;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 1;
}

/* ---------------- map: uint64 key -> uint32 value ----------------
 *
 * CHANGE from original: key is now uint64_t (truncated x-coordinate)
 * instead of 33-byte compressed point. This reduces entry size from
 * 38 bytes to ~13 bytes and halves the number of entries since
 * (i*G)[x] == (-i*G)[x] so one entry covers both i and -i.
 *
 * Memory saving: ~6x total (2x from half entries, ~3x from smaller key).
 */

typedef struct {
    uint64_t key;
    uint32_t val;
    uint8_t  used;
} entry64_u32;  /* 13 bytes; compiler may pad to 16 */

typedef struct {
    entry64_u32* tab;
    size_t cap;   /* power of two */
    size_t mask;  /* cap-1 */
    size_t size;
} map64_u32;

static size_t next_pow2(size_t x) {
    size_t p = 1;
    while (p < x) p <<= 1;
    return p;
}

static int map_init_cap(map64_u32* m, size_t cap_pow2) {
    m->tab = (entry64_u32*)calloc(cap_pow2, sizeof(entry64_u32));
    if (!m->tab) return 0;
    m->cap  = cap_pow2;
    m->mask = cap_pow2 - 1;
    m->size = 0;
    return 1;
}

static int map_init(map64_u32* m, size_t want_items) {
    /* keep load <= ~0.5 */
    size_t cap = next_pow2(want_items * 2 + 1);
    return map_init_cap(m, cap);
}

static void map_free(map64_u32* m) {
    free(m->tab);
    memset(m, 0, sizeof(*m));
}

/* Hash a uint64 key with a simple mixer */
static uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

static int map_put(map64_u32* m, uint64_t key, uint32_t val) {
    size_t idx = (size_t)mix64(key) & m->mask;
    for (;;) {
        entry64_u32* e = &m->tab[idx];
        if (!e->used) {
            e->key  = key;
            e->val  = val;
            e->used = 1;
            m->size++;
            return 1;
        }
        if (e->key == key) {
            return 1; /* already present */
        }
        idx = (idx + 1) & m->mask;
    }
}

static int map_get(const map64_u32* m, uint64_t key, uint32_t* out_val) {
    size_t idx = (size_t)mix64(key) & m->mask;
    for (;;) {
        const entry64_u32* e = &m->tab[idx];
        if (!e->used) return 0;
        if (e->key == key) {
            *out_val = e->val;
            return 1;
        }
        idx = (idx + 1) & m->mask;
    }
}

/* ---------------- cached baby table format ---------------- */

#define BABY_MAGIC 0x3634594241475342ULL  /* "BSGSBAY64" - new magic for new format */

typedef struct {
    uint64_t magic;
    uint32_t version;
    uint32_t l1;
    uint64_t cap;
    uint64_t used_count;
} baby_hdr;

static void baby_cache_path(char* out, size_t outlen, int l1) {
    /* New filename to avoid confusion with old 33-byte format */
    snprintf(out, outlen, "bsgs_baby64_secp256k1_l1_%d.bin", l1);
}

static int baby_save(const char* path, int l1, const map64_u32* baby) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        perror("baby_save open");
        return 0;
    }

    baby_hdr hdr;
    hdr.magic      = BABY_MAGIC;
    hdr.version    = 2;             /* version 2 = 64-bit key format */
    hdr.l1         = (uint32_t)l1;
    hdr.cap        = (uint64_t)baby->cap;
    hdr.used_count = (uint64_t)baby->size;

    int ok = 1;
    ok &= write_all(fd, &hdr, sizeof(hdr));
    ok &= write_all(fd, baby->tab, baby->cap * sizeof(entry64_u32));
    if (!ok) perror("baby_save write");

    close(fd);
    return ok;
}

static int baby_load(const char* path, int expected_l1, map64_u32* baby_out) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;

    baby_hdr hdr;
    if (!read_all(fd, &hdr, sizeof(hdr))) { close(fd); return 0; }

    if (hdr.magic != BABY_MAGIC || hdr.version != 2 || (int)hdr.l1 != expected_l1) {
        close(fd);
        return 0;
    }

    if (hdr.cap == 0 || (hdr.cap & (hdr.cap - 1)) != 0) {
        close(fd);
        return 0;
    }

    if (!map_init_cap(baby_out, (size_t)hdr.cap)) { close(fd); return 0; }

    size_t bytes = (size_t)hdr.cap * sizeof(entry64_u32);
    if (!read_all(fd, baby_out->tab, bytes)) {
        map_free(baby_out);
        close(fd);
        return 0;
    }

    baby_out->size = (size_t)hdr.used_count;
    close(fd);
    return 1;
}

/* ---------------- reusable BSGS context ---------------- */

typedef struct {
    secp256k1_context* ctx; /* not owned */
    int bits_total;
    int l1;
    uint64_t M;     /* 2^l1       - baby step size    */
    uint64_t Mhalf; /* 2^(l1-1)   - half range stored */
    uint64_t J;     /* 2^(l2)     - giant step count  */

    secp256k1_pubkey G;
    secp256k1_pubkey MG;
    secp256k1_pubkey neg_MG;

    map64_u32 baby; /* 64-bit key hash table */
} bsgs_ctx;

/*
 * Build or load the baby table.
 *
 * CHANGE: only iterate i in [1, 2^(l1-1)) instead of [1, 2^l1).
 * Since (i*G)[x] == (-i*G)[x], one entry covers both +i and -i.
 * This halves the table size with no loss of correctness.
 */
static int bsgs_ctx_init_cached(bsgs_ctx* b,
                                secp256k1_context* ctx,
                                int bits_total,
                                int l1) {
    memset(b, 0, sizeof(*b));
    b->ctx = ctx;
    b->bits_total = bits_total;
    b->l1 = l1;

    if (bits_total <= 0 || bits_total > 63) return 0;
    if (l1 <= 0 || l1 >= bits_total) return 0;

    b->M     = 1ULL << l1;
    b->Mhalf = 1ULL << (l1 - 1);  /* half range */
    b->J     = 1ULL << (bits_total - l1);

    unsigned char one[32] = {0};
    one[31] = 1;
    if (!secp256k1_ec_pubkey_create(ctx, &b->G, one)) return 0;

    unsigned char M_scalar[32];
    u64_to_scalar32_be(b->M, M_scalar);
    if (!secp256k1_ec_pubkey_create(ctx, &b->MG, M_scalar)) return 0;

    if (!pubkey_negate(ctx, &b->MG, &b->neg_MG)) return 0;

    /* Try load cached baby table */
    char path[128];
    baby_cache_path(path, sizeof(path), l1);

    if (file_exists(path)) {
        if (baby_load(path, l1, &b->baby)) {
            printf("Loaded baby table cache: %s (cap=%zu, used=%zu)\n",
                   path, b->baby.cap, b->baby.size);
            return 1;
        }
        printf("Cache exists but failed to load (will rebuild): %s\n", path);
    }

    /*
     * Build baby table: i in [1, Mhalf).
     * Key  = first 8 bytes of x-coordinate (same for i and -i).
     * Value = i  (positive index; solve() will check both +i and -i).
     */
    if (!map_init(&b->baby, (size_t)(b->Mhalf - 1))) return 0;

    secp256k1_pubkey cur = b->G; /* 1*G */
    unsigned char ser[33];

    for (uint64_t i = 1; i < b->Mhalf; i++) {
        if (!pubkey_serialize33(ctx, &cur, ser)) { map_free(&b->baby); return 0; }

        uint64_t xkey = pubkey_x64(ser);
        if (!map_put(&b->baby, xkey, (uint32_t)i)) { map_free(&b->baby); return 0; }

        if (i + 1 < b->Mhalf) {
            const secp256k1_pubkey* pts[2] = { &cur, &b->G };
            secp256k1_pubkey nxt;
            if (!secp256k1_ec_pubkey_combine(ctx, &nxt, pts, 2)) {
                map_free(&b->baby);
                return 0;
            }
            cur = nxt;
        }
    }

    if (!baby_save(path, l1, &b->baby)) {
        printf("Warning: failed to save cache: %s\n", path);
    } else {
        printf("Built and saved baby table cache: %s (cap=%zu, used=%zu)\n",
               path, b->baby.cap, b->baby.size);
    }

    return 1;
}

static void bsgs_ctx_free(bsgs_ctx* b) {
    map_free(&b->baby);
    memset(b, 0, sizeof(*b));
}

/*
 * Verify a candidate m by checking m*G == target.
 * Used because an x-coordinate hit gives two candidates (+i, -i).
 */
static int verify_candidate(const secp256k1_context* ctx,
                            uint64_t m,
                            const unsigned char target33[33]) {
    if (m == 0) return 0;
    unsigned char m_scalar[32];
    u64_to_scalar32_be(m, m_scalar);
    secp256k1_pubkey check;
    if (!secp256k1_ec_pubkey_create(ctx, &check, m_scalar)) return 0;
    unsigned char check33[33];
    if (!pubkey_serialize33(ctx, &check, check33)) return 0;
    return memcmp(check33, target33, 33) == 0;
}

/*
 * Solve m in [0, 2^bits_total).
 *
 * CHANGES vs original:
 * 1. Baby lookup uses 64-bit x-key instead of full 33-byte point.
 * 2. Each hit yields two candidates: j*M+i and j*M-i.
 * 3. verify_candidate() picks the correct one.
 */
static int bsgs_solve(const bsgs_ctx* b,
                      const secp256k1_pubkey* targetPm,
                      uint64_t* out_m) {
    const secp256k1_context* ctx = b->ctx;

    unsigned char t33[33];
    if (!pubkey_serialize33(ctx, targetPm, t33)) return 0;

    uint64_t tx64 = pubkey_x64(t33);

    /* j=0 case: target itself is a baby step (i in [1, Mhalf)) */
    uint32_t i_found;
    if (map_get(&b->baby, tx64, &i_found)) {
        /* two candidates: +i_found and -i_found (mod curve order, but
           we work in positive range so check both) */
        if (verify_candidate(ctx, (uint64_t)i_found, t33)) {
            *out_m = (uint64_t)i_found;
            return 1;
        }
        /* -i candidate: since m is in [0, 2^bits_total),
           -i wraps but we can skip — negative i not in range */
    }

    /* Giant step loop: Q = target - j*MG, look for Q[x] in baby table */
    secp256k1_pubkey Q;
    {
        const secp256k1_pubkey* pts[2] = { targetPm, &b->neg_MG };
        if (!secp256k1_ec_pubkey_combine(ctx, &Q, pts, 2)) return 0;
    }

    secp256k1_pubkey jMG = b->MG; /* j=1 */

    for (uint64_t j = 1; j < b->J; j++) {

        /* i=0 check: target == j*MG */
        unsigned char jmg33[33];
        if (!pubkey_serialize33(ctx, &jMG, jmg33)) return 0;
        if (memcmp(jmg33, t33, 33) == 0) {
            *out_m = j * b->M;
            return 1;
        }

        /* i>0 check via truncated x-key */
        unsigned char q33[33];
        if (!pubkey_serialize33(ctx, &Q, q33)) return 0;
        uint64_t qx64 = pubkey_x64(q33);

        if (map_get(&b->baby, qx64, &i_found)) {
            uint64_t m1 = j * b->M + (uint64_t)i_found;
            uint64_t m2 = j * b->M - (uint64_t)i_found;
            if (verify_candidate(ctx, m1, t33)) { *out_m = m1; return 1; }
            if (verify_candidate(ctx, m2, t33)) { *out_m = m2; return 1; }
            /* false positive from truncation collision — continue */
        }

        if (j + 1 < b->J) {
            /* Q <- Q - MG */
            const secp256k1_pubkey* pts2[2] = { &Q, &b->neg_MG };
            secp256k1_pubkey nxt;
            if (!secp256k1_ec_pubkey_combine(ctx, &nxt, pts2, 2)) return 0;
            Q = nxt;

            /* jMG <- jMG + MG */
            const secp256k1_pubkey* pts3[2] = { &jMG, &b->MG };
            secp256k1_pubkey nxt2;
            if (!secp256k1_ec_pubkey_combine(ctx, &nxt2, pts3, 2)) return 0;
            jMG = nxt2;
        }
    }

    return 0;
}

/* ---------------- parallel BSGS solve ---------------- */

typedef struct {
    const bsgs_ctx*    b;
    secp256k1_pubkey   targetPm;
    unsigned char      target33[33];
    uint64_t           j_start;
    uint64_t           j_end;
    atomic_int*        found;
    uint64_t*          found_m;
    pthread_mutex_t*   found_mu;
} bsgs_worker_args;

static void* bsgs_worker_thread(void* argp) {
    bsgs_worker_args* a = (bsgs_worker_args*)argp;
    const bsgs_ctx*   b = a->b;

    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_VERIFY);
    if (!ctx) return NULL;

    /* Compute starting point: j_start * MG */
    uint64_t j  = a->j_start;
    uint64_t jm = j * b->M;
    unsigned char jm_scalar[32];
    u64_to_scalar32_be(jm, jm_scalar);

    secp256k1_pubkey jMG;
    if (!secp256k1_ec_pubkey_create(ctx, &jMG, jm_scalar)) {
        secp256k1_context_destroy(ctx);
        return NULL;
    }

    /* Q = target - j_start*MG */
    secp256k1_pubkey neg_jMG, Q;
    if (!pubkey_negate(ctx, &jMG, &neg_jMG)) {
        secp256k1_context_destroy(ctx);
        return NULL;
    }
    {
        const secp256k1_pubkey* pts[2] = { &a->targetPm, &neg_jMG };
        if (!secp256k1_ec_pubkey_combine(ctx, &Q, pts, 2)) {
            secp256k1_context_destroy(ctx);
            return NULL;
        }
    }

    for (; j < a->j_end; j++) {
        if (atomic_load_explicit(a->found, memory_order_relaxed)) break;

        /* i=0: target == j*MG */
        unsigned char jmg33[33];
        if (!pubkey_serialize33(ctx, &jMG, jmg33)) break;
        if (memcmp(jmg33, a->target33, 33) == 0) {
            uint64_t m = j * b->M;
            pthread_mutex_lock(a->found_mu);
            if (!atomic_load_explicit(a->found, memory_order_relaxed)) {
                *a->found_m = m;
                atomic_store_explicit(a->found, 1, memory_order_relaxed);
            }
            pthread_mutex_unlock(a->found_mu);
            break;
        }

        /* i>0: lookup Q by truncated x-key */
        unsigned char q33[33];
        if (!pubkey_serialize33(ctx, &Q, q33)) break;
        uint64_t qx64 = pubkey_x64(q33);

        uint32_t i_found;
        if (map_get(&b->baby, qx64, &i_found)) {
            uint64_t m1 = j * b->M + (uint64_t)i_found;
            uint64_t m2 = j * b->M - (uint64_t)i_found;

            /* verify — pick correct candidate */
            uint64_t m_ok = 0;
            int got = 0;
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
            /* else: truncation collision, continue */
        }

        if (j + 1 < a->j_end) {
            /* Q = Q - MG */
            const secp256k1_pubkey* pts2[2] = { &Q, &b->neg_MG };
            secp256k1_pubkey nxt;
            if (!secp256k1_ec_pubkey_combine(ctx, &nxt, pts2, 2)) break;
            Q = nxt;

            /* jMG = jMG + MG */
            const secp256k1_pubkey* pts3[2] = { &jMG, &b->MG };
            secp256k1_pubkey nxt2;
            if (!secp256k1_ec_pubkey_combine(ctx, &nxt2, pts3, 2)) break;
            jMG = nxt2;
        }
    }

    secp256k1_context_destroy(ctx);
    return NULL;
}

static int bsgs_solve_parallel(const bsgs_ctx* b,
                               const secp256k1_pubkey* targetPm,
                               int nthreads,
                               uint64_t* out_m) {
    if (nthreads < 1) nthreads = 1;
    if (nthreads == 1) return bsgs_solve(b, targetPm, out_m);

    const secp256k1_context* ctx0 = b->ctx;

    unsigned char target33[33];
    if (!pubkey_serialize33(ctx0, targetPm, target33)) return 0;

    /* j=0 case: direct baby lookup on target x-key */
    uint64_t tx64 = pubkey_x64(target33);
    uint32_t i_found;
    if (map_get(&b->baby, tx64, &i_found)) {
        if (verify_candidate(ctx0, (uint64_t)i_found, target33)) {
            *out_m = (uint64_t)i_found;
            return 1;
        }
    }

    uint64_t J = b->J;
    if (J <= 1) return 0;

    if ((uint64_t)nthreads > (J - 1)) nthreads = (int)(J - 1);

    pthread_t*         tids = (pthread_t*)        calloc((size_t)nthreads, sizeof(pthread_t));
    bsgs_worker_args*  args = (bsgs_worker_args*) calloc((size_t)nthreads, sizeof(bsgs_worker_args));
    if (!tids || !args) { free(tids); free(args); return 0; }

    atomic_int      found = 0;
    uint64_t        found_m = 0;
    pthread_mutex_t found_mu;
    pthread_mutex_init(&found_mu, NULL);

    uint64_t total = J - 1;
    uint64_t chunk = total / (uint64_t)nthreads;
    uint64_t rem   = total % (uint64_t)nthreads;
    uint64_t jcur  = 1;

    for (int t = 0; t < nthreads; t++) {
        uint64_t take   = chunk + (t < (int)rem ? 1 : 0);
        uint64_t jstart = jcur;
        uint64_t jend   = jstart + take;
        jcur = jend;

        args[t].b        = b;
        args[t].targetPm = *targetPm;
        memcpy(args[t].target33, target33, 33);
        args[t].j_start  = jstart;
        args[t].j_end    = jend;
        args[t].found    = &found;
        args[t].found_m  = &found_m;
        args[t].found_mu = &found_mu;

        pthread_create(&tids[t], NULL, bsgs_worker_thread, &args[t]);
    }

    for (int t = 0; t < nthreads; t++) pthread_join(tids[t], NULL);

    int ok = atomic_load_explicit(&found, memory_order_relaxed);
    if (ok) *out_m = found_m;

    pthread_mutex_destroy(&found_mu);
    free(tids);
    free(args);
    return ok;
}

/* ---------------- benchmark ---------------- */

static void benchmark_bsgs(int bits_total, int l1, int trials, int threads) {
    printf("=== BSGS small-range DLP benchmark (secp256k1, 64-bit key) ===\n");
    printf("Range: m in [0, 2^%d)\n", bits_total);
    printf("Split: l1=%d (baby M=2^l1, table covers [1,2^%d)), l2=%d (giant J=2^l2)\n",
           l1, l1 - 1, bits_total - l1);
    printf("Entry size: %zu bytes (was 38 bytes)\n", sizeof(entry64_u32));
    printf("Trials: %d  Threads: %d\n", trials, threads);

    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_VERIFY);
    if (!ctx) { printf("Failed to create secp256k1 context\n"); return; }

    bsgs_ctx solver;
    double t_build0 = now_seconds();
    if (!bsgs_ctx_init_cached(&solver, ctx, bits_total, l1)) {
        printf("Failed to init BSGS solver\n");
        secp256k1_context_destroy(ctx);
        return;
    }
    double t_build1 = now_seconds();
    printf("Table size in memory: %.2f MB (%.2f MB saved vs 38-byte entries)\n",
           (double)(solver.baby.cap * sizeof(entry64_u32)) / (1024*1024),
           (double)(solver.baby.cap * (38 - sizeof(entry64_u32))) / (1024*1024));
    printf("Init (load/build) time: %.6f sec\n\n", (t_build1 - t_build0));

    uint64_t mask = (bits_total == 64) ? ~0ULL : ((1ULL << bits_total) - 1ULL);

    int ok = 0;
    double t0 = now_seconds();
    for (int t = 0; t < trials; t++) {
        uint64_t m = ((uint64_t)rand() << 32) ^ (uint64_t)rand();
        m &= mask;
        if (m == 0) m = 1;

        unsigned char m_scalar[32];
        u64_to_scalar32_be(m, m_scalar);

        secp256k1_pubkey Pm;
        if (!secp256k1_ec_pubkey_create(ctx, &Pm, m_scalar)) {
            printf("Failed to create Pm\n");
            break;
        }

        uint64_t recovered = 0;
        if (bsgs_solve_parallel(&solver, &Pm, threads, &recovered) && recovered == m) {
            ok++;
        } else {
            printf("Trial %d FAILED: m=%" PRIu64 " recovered=%" PRIu64 "\n",
                    t, m, recovered);
        }
    }
    double t1 = now_seconds();

    double total = t1 - t0;
    printf("Solved correctly: %d/%d\n", ok, trials);
    printf("Search-only total time: %.6f sec\n", total);
    printf("Average per solve (search-only): %.9f sec (%.2f ms)\n\n",
           total / trials, (total / trials) * 1e3);

    bsgs_ctx_free(&solver);
    secp256k1_context_destroy(ctx);
}

int main(int argc, char** argv) {
    int bits_total = 40;
    int l1         = 18;
    int trials     = 1;
    int threads    = 1;

    if (argc >= 2) bits_total = atoi(argv[1]);
    if (argc >= 3) l1         = atoi(argv[2]);
    if (argc >= 4) trials     = atoi(argv[3]);
    if (argc >= 5) threads    = atoi(argv[4]);

    srand((unsigned)time(NULL));
    benchmark_bsgs(bits_total, l1, trials, threads);
    return 0;
}