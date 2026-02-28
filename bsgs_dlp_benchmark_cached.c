#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <inttypes.h>
#include <errno.h>

#include <secp256k1.h>

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

static int default_threads(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1) return 1;
    if (n > 64) n = 64;
    return (int)n;
}

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
    if (!secp256k1_ec_pubkey_serialize(ctx, out33, &outlen, pk, SECP256K1_EC_COMPRESSED))
        return 0;
    return outlen == 33;
}

static int pubkey_negate(const secp256k1_context* ctx,
                         const secp256k1_pubkey* in,
                         secp256k1_pubkey* out) {
    unsigned char ser[33];
    if (!pubkey_serialize33(ctx, in, ser)) return 0;
    ser[0] ^= 1;
    return secp256k1_ec_pubkey_parse(ctx, out, ser, 33);
}

static int pubkey_equal33(const unsigned char a[33], const unsigned char b[33]) {
    return memcmp(a, b, 33) == 0;
}

static uint64_t fnv1a64(const unsigned char* data, size_t len) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < len; i++) {
        h ^= (uint64_t)data[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static size_t next_pow2(size_t x) {
    size_t p = 1;
    while (p < x) p <<= 1;
    return p;
}

static int file_exists(const char* path) {
    struct stat st;
    return stat(path, &st) == 0;
}

/* ---------------- map ---------------- */

typedef struct {
    unsigned char key[33];
    uint32_t val;
    uint8_t used;
} entry33_u32;

typedef struct {
    entry33_u32* tab;
    size_t cap;
    size_t mask;
    size_t size;
} map33_u32;

static int map_init(map33_u32* m, size_t want_items) {
    size_t cap = next_pow2(want_items * 2 + 1);
    m->tab = calloc(cap, sizeof(entry33_u32));
    if (!m->tab) return 0;
    m->cap = cap;
    m->mask = cap - 1;
    m->size = 0;
    return 1;
}

static void map_free(map33_u32* m) {
    free(m->tab);
}

static int map_put(map33_u32* m, const unsigned char key[33], uint32_t val) {
    uint64_t h = fnv1a64(key, 33);
    size_t idx = h & m->mask;

    for (;;) {
        entry33_u32* e = &m->tab[idx];

        if (!e->used) {
            memcpy(e->key, key, 33);
            e->val = val;
            e->used = 1;
            m->size++;
            return 1;
        }

        if (pubkey_equal33(e->key, key))
            return 1;

        idx = (idx + 1) & m->mask;
    }
}

static int map_get(const map33_u32* m, const unsigned char key[33], uint32_t* out_val) {
    uint64_t h = fnv1a64(key, 33);
    size_t idx = h & m->mask;

    for (;;) {
        const entry33_u32* e = &m->tab[idx];

        if (!e->used)
            return 0;

        if (pubkey_equal33(e->key, key)) {
            *out_val = e->val;
            return 1;
        }

        idx = (idx + 1) & m->mask;
    }
}

/* ---------------- BSGS context ---------------- */

typedef struct {
    secp256k1_context* ctx;
    int bits_total;
    int l1;

    uint64_t M;
    uint64_t J;

    secp256k1_pubkey G;
    secp256k1_pubkey MG;
    secp256k1_pubkey neg_MG;

    map33_u32 baby;

} bsgs_ctx;

/* ---------------- worker ---------------- */

typedef struct {

    const bsgs_ctx* b;
    const secp256k1_pubkey* targetPm;

    const unsigned char* t33;

    uint64_t j_start;
    uint64_t j_end;

    atomic_int* found;
    atomic_ullong* found_m;

} worker_args;

static void* bsgs_worker(void* argp) {

    worker_args* a = argp;
    const bsgs_ctx* b = a->b;
    const secp256k1_context* ctx = b->ctx;

    secp256k1_pubkey jMG;

    unsigned char scalar[32];
    u64_to_scalar32_be(a->j_start * b->M, scalar);

    if (!secp256k1_ec_pubkey_create(ctx, &jMG, scalar))
        return NULL;

    secp256k1_pubkey neg_jMG;
    pubkey_negate(ctx, &jMG, &neg_jMG);

    secp256k1_pubkey Q;

    const secp256k1_pubkey* pts[2] = {a->targetPm, &neg_jMG};

    if (!secp256k1_ec_pubkey_combine(ctx, &Q, pts, 2))
        return NULL;

    for (uint64_t j = a->j_start; j < a->j_end; j++) {

        if (atomic_load(a->found))
            return NULL;

        unsigned char jmg33[33];
        pubkey_serialize33(ctx, &jMG, jmg33);

        if (pubkey_equal33(jmg33, a->t33)) {

            uint64_t m = j * b->M;

            if (!atomic_exchange(a->found, 1))
                atomic_store(a->found_m, m);

            return NULL;
        }

        unsigned char q33[33];
        pubkey_serialize33(ctx, &Q, q33);

        uint32_t i_found;

        if (map_get(&b->baby, q33, &i_found)) {

            uint64_t m = j * b->M + i_found;

            if (!atomic_exchange(a->found, 1))
                atomic_store(a->found_m, m);

            return NULL;
        }

        if (j + 1 < a->j_end) {

            const secp256k1_pubkey* pts2[2] = {&Q, &b->neg_MG};
            if (!secp256k1_ec_pubkey_combine(ctx, &Q, pts2, 2))
                return NULL;

            const secp256k1_pubkey* pts3[2] = {&jMG, &b->MG};
            if(!secp256k1_ec_pubkey_combine(ctx, &jMG, pts3, 2))
                return NULL;
        }
    }

    return NULL;
}

/* ---------------- solve ---------------- */

static int bsgs_solve(const bsgs_ctx* b,
                      const secp256k1_pubkey* targetPm,
                      uint64_t* out_m,
                      int threads) {

    unsigned char t33[33];
    pubkey_serialize33(b->ctx, targetPm, t33);

    uint32_t i_found;

    if (map_get(&b->baby, t33, &i_found)) {
        *out_m = i_found;
        return 1;
    }

    if (threads < 1)
        threads = 1;

    pthread_t* th = calloc(threads, sizeof(pthread_t));
    worker_args* args = calloc(threads, sizeof(worker_args));

    atomic_int found = 0;
    atomic_ullong found_m = 0;

    uint64_t total = b->J - 1;
    uint64_t chunk = (total + threads - 1) / threads;

    for (int t = 0; t < threads; t++) {

        uint64_t j0 = 1 + t * chunk;
        uint64_t j1 = j0 + chunk;

        if (j0 >= b->J)
            j0 = b->J;

        if (j1 > b->J)
            j1 = b->J;

        args[t].b = b;
        args[t].targetPm = targetPm;
        args[t].t33 = t33;
        args[t].j_start = j0;
        args[t].j_end = j1;
        args[t].found = &found;
        args[t].found_m = &found_m;

        pthread_create(&th[t], NULL, bsgs_worker, &args[t]);
    }

    for (int t = 0; t < threads; t++)
        pthread_join(th[t], NULL);

    free(th);
    free(args);

    if (atomic_load(&found)) {
        *out_m = atomic_load(&found_m);
        return 1;
    }

    return 0;
}

/* ---------------- cached baby table format ---------------- */

#define BABY_MAGIC 0x4253475342414259ULL /* "BSGSBABY" */

typedef struct {
    uint64_t magic;
    uint32_t version;
    uint32_t l1;
    uint64_t cap;
    uint64_t used_count;
} baby_hdr;

/* write entire buffer */
static int write_all(int fd, const void* buf, size_t len) {

    const unsigned char* p = buf;

    while (len > 0) {

        ssize_t n = write(fd, p, len);

        if (n < 0) {
            if (errno == EINTR) continue;
            return 0;
        }

        p += n;
        len -= n;
    }

    return 1;
}

/* read entire buffer */
static int read_all(int fd, void* buf, size_t len) {

    unsigned char* p = buf;

    while (len > 0) {

        ssize_t n = read(fd, p, len);

        if (n < 0) {
            if (errno == EINTR) continue;
            return 0;
        }

        if (n == 0)
            return 0;

        p += n;
        len -= n;
    }

    return 1;
}

/* save table to disk */
static int baby_save(const char* path, int l1, const map33_u32* baby) {

    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);

    if (fd < 0)
        return 0;

    baby_hdr hdr;

    hdr.magic = BABY_MAGIC;
    hdr.version = 1;
    hdr.l1 = (uint32_t)l1;
    hdr.cap = baby->cap;
    hdr.used_count = baby->size;

    int ok = 1;

    ok &= write_all(fd, &hdr, sizeof(hdr));
    ok &= write_all(fd, baby->tab, baby->cap * sizeof(entry33_u32));

    close(fd);

    return ok;
}

/* load table from disk */
static int baby_load(const char* path, int expected_l1, map33_u32* baby_out) {

    int fd = open(path, O_RDONLY);

    if (fd < 0)
        return 0;

    baby_hdr hdr;

    if (!read_all(fd, &hdr, sizeof(hdr))) {
        close(fd);
        return 0;
    }

    if (hdr.magic != BABY_MAGIC ||
        hdr.version != 1 ||
        hdr.l1 != (uint32_t)expected_l1) {

        close(fd);
        return 0;
    }

    if (!map_init(baby_out, hdr.cap)) {
        close(fd);
        return 0;
    }

    size_t bytes = hdr.cap * sizeof(entry33_u32);

    if (!read_all(fd, baby_out->tab, bytes)) {

        map_free(baby_out);
        close(fd);

        return 0;
    }

    baby_out->size = hdr.used_count;

    close(fd);

    return 1;
}

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

    b->M = 1ULL << l1;
    b->J = 1ULL << (bits_total - l1);

    unsigned char one[32] = {0};
    one[31] = 1;

    if (!secp256k1_ec_pubkey_create(ctx, &b->G, one))
        return 0;

    unsigned char M_scalar[32];
    u64_to_scalar32_be(b->M, M_scalar);

    if (!secp256k1_ec_pubkey_create(ctx, &b->MG, M_scalar))
        return 0;

    if (!pubkey_negate(ctx, &b->MG, &b->neg_MG))
        return 0;

    /* load cached table if exists */

    char path[128];
    snprintf(path, sizeof(path),
             "bsgs_baby_secp256k1_l1_%d.bin", l1);

    if (file_exists(path)) {

        if (baby_load(path, l1, &b->baby)) {

            printf("Loaded baby table cache: %s (cap=%zu, used=%zu)\n",
                   path, b->baby.cap, b->baby.size);

            return 1;
        }

        printf("Cache exists but failed to load, rebuilding\n");
    }

    /* build baby table */

    if (!map_init(&b->baby, (size_t)(b->M - 1)))
        return 0;

    secp256k1_pubkey cur = b->G;

    for (uint64_t i = 1; i < b->M; i++) {

        unsigned char key33[33];

        if (!pubkey_serialize33(ctx, &cur, key33))
            return 0;

        map_put(&b->baby, key33, (uint32_t)i);

        if (i + 1 < b->M) {

            const secp256k1_pubkey* pts[2] = { &cur, &b->G };

            secp256k1_pubkey nxt;

            if (!secp256k1_ec_pubkey_combine(ctx, &nxt, pts, 2))
                return 0;

            cur = nxt;
        }
    }

    if (!baby_save(path, l1, &b->baby)) {

        printf("Warning: failed to save cache\n");

    } else {

        printf("Built and saved baby table cache: %s (cap=%zu, used=%zu)\n",
               path, b->baby.cap, b->baby.size);
    }

    return 1;
}


static void benchmark_bsgs(int bits_total, int l1, int trials, int threads) {

    printf("=== BSGS threaded benchmark ===\n");
    printf("Range: m in [0,2^%d)\n", bits_total);
    printf("Split: l1=%d\n", l1);
    printf("Trials: %d\n", trials);
    printf("Threads: %d\n", threads);

    secp256k1_context* ctx =
            secp256k1_context_create(SECP256K1_CONTEXT_VERIFY);

    bsgs_ctx solver;

    /* init (you already have this function) */
    if (!bsgs_ctx_init_cached(&solver, ctx, bits_total, l1)) {
        printf("Failed to init solver\n");
        return;
    }

    uint64_t mask =
            (bits_total == 64) ? ~0ULL : ((1ULL << bits_total) - 1ULL);

    int ok = 0;

    double t0 = now_seconds();

    for (int t = 0; t < trials; t++) {

        uint64_t m =
                ((uint64_t)rand() << 32) ^ rand();

        m &= mask;

        if (m == 0)
            m = 1;

        unsigned char scalar[32];

        u64_to_scalar32_be(m, scalar);

        secp256k1_pubkey Pm;

        if(!secp256k1_ec_pubkey_create(ctx, &Pm, scalar)) {
            printf("Failed to create pubkey for m=%" PRIu64 "\n", m);
            continue;
        }gi

        uint64_t recovered = 0;

        if (bsgs_solve(&solver, &Pm, &recovered, threads) &&
            recovered == m)
            ok++;
    }

    double t1 = now_seconds();

    double total = t1 - t0;

    printf("Solved correctly: %d/%d\n", ok, trials);
    printf("Search-only total time: %.6f sec\n", total);
    printf("Average per solve: %.6f sec\n\n", total / trials);

    secp256k1_context_destroy(ctx);
}
int main(int argc, char** argv) {

    int bits_total = 40;
    int l1 = 18;
    int trials = 1;
    int threads = default_threads();

    if (argc >= 2)
        bits_total = atoi(argv[1]);

    if (argc >= 3)
        l1 = atoi(argv[2]);

    if (argc >= 4)
        trials = atoi(argv[3]);

    if (argc >= 5)
        threads = atoi(argv[4]);

    srand((unsigned)time(NULL));

    benchmark_bsgs(bits_total, l1, trials, threads);

    return 0;
}