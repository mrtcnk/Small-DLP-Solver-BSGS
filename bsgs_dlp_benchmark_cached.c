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

/* ---------------- map: 33-byte key -> uint32 ---------------- */

typedef struct {
    unsigned char key[33];
    uint32_t val;
    uint8_t used;
    /* padding may exist */
} entry33_u32;

typedef struct {
    entry33_u32* tab;
    size_t cap;   /* power of two */
    size_t mask;  /* cap-1 */
    size_t size;
} map33_u32;

static int map_init_cap(map33_u32* m, size_t cap_pow2) {
    m->tab = (entry33_u32*)calloc(cap_pow2, sizeof(entry33_u32));
    if (!m->tab) return 0;
    m->cap = cap_pow2;
    m->mask = cap_pow2 - 1;
    m->size = 0;
    return 1;
}

static int map_init(map33_u32* m, size_t want_items) {
    /* keep load <= ~0.5 */
    size_t cap = next_pow2(want_items * 2 + 1);
    return map_init_cap(m, cap);
}

static void map_free(map33_u32* m) {
    free(m->tab);
    memset(m, 0, sizeof(*m));
}

static int map_put(map33_u32* m, const unsigned char key[33], uint32_t val) {
    uint64_t h = fnv1a64(key, 33);
    size_t idx = (size_t)h & m->mask;
    for (;;) {
        entry33_u32* e = &m->tab[idx];
        if (!e->used) {
            memcpy(e->key, key, 33);
            e->val = val;
            e->used = 1;
            m->size++;
            return 1;
        }
        if (pubkey_equal33(e->key, key)) {
            return 1;
        }
        idx = (idx + 1) & m->mask;
    }
}

static int map_get(const map33_u32* m, const unsigned char key[33], uint32_t* out_val) {
    uint64_t h = fnv1a64(key, 33);
    size_t idx = (size_t)h & m->mask;
    for (;;) {
        const entry33_u32* e = &m->tab[idx];
        if (!e->used) return 0;
        if (pubkey_equal33(e->key, key)) {
            *out_val = e->val;
            return 1;
        }
        idx = (idx + 1) & m->mask;
    }
}

/* ---------------- cached baby table format ----------------
   Header + raw entries.
   We store the *hash table layout* (cap + entry array), so loading is O(1).
*/

#define BABY_MAGIC 0x4253475342414259ULL /* "BSGSBABY" */
typedef struct {
    uint64_t magic;
    uint32_t version;
    uint32_t l1;
    uint64_t cap;
    uint64_t used_count;
} baby_hdr;

static void baby_cache_path(char* out, size_t outlen, int l1) {
    snprintf(out, outlen, "bsgs_baby_secp256k1_l1_%d.bin", l1);
}

static int baby_save(const char* path, int l1, const map33_u32* baby) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return 0;

    baby_hdr hdr;
    hdr.magic = BABY_MAGIC;
    hdr.version = 1;
    hdr.l1 = (uint32_t)l1;
    hdr.cap = (uint64_t)baby->cap;
    hdr.used_count = (uint64_t)baby->size;

    int ok = 1;
    ok &= write_all(fd, &hdr, sizeof(hdr));
    ok &= write_all(fd, baby->tab, baby->cap * sizeof(entry33_u32));

    close(fd);
    return ok;
}

static int baby_load(const char* path, int expected_l1, map33_u32* baby_out) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;

    baby_hdr hdr;
    if (!read_all(fd, &hdr, sizeof(hdr))) { close(fd); return 0; }

    if (hdr.magic != BABY_MAGIC || hdr.version != 1 || (int)hdr.l1 != expected_l1) {
        close(fd);
        return 0;
    }

    if (hdr.cap == 0 || (hdr.cap & (hdr.cap - 1)) != 0) { /* must be pow2 */
        close(fd);
        return 0;
    }

    if (!map_init_cap(baby_out, (size_t)hdr.cap)) { close(fd); return 0; }

    size_t bytes = (size_t)hdr.cap * sizeof(entry33_u32);
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
    uint64_t M;
    uint64_t J;

    secp256k1_pubkey G;
    secp256k1_pubkey MG;
    secp256k1_pubkey neg_MG;

    map33_u32 baby; /* cached hash table */
} bsgs_ctx;

/* Build or load the baby table once; precompute MG and -MG once */
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
            printf("Loaded baby table cache: %s (cap=%zu, used=%zu)\n", path, b->baby.cap, b->baby.size);
            return 1;
        }
        printf("Cache exists but failed to load (will rebuild): %s\n", path);
    }

    /* Build baby table if cache not present/invalid */
    if (!map_init(&b->baby, (size_t)(b->M - 1))) return 0;

    secp256k1_pubkey cur = b->G; /* 1*G */
    for (uint64_t i = 1; i < b->M; i++) {
        unsigned char key33[33];
        if (!pubkey_serialize33(ctx, &cur, key33)) { map_free(&b->baby); return 0; }
        if (!map_put(&b->baby, key33, (uint32_t)i)) { map_free(&b->baby); return 0; }

        if (i + 1 < b->M) {
            const secp256k1_pubkey* pts[2] = { &cur, &b->G };
            secp256k1_pubkey nxt;
            if (!secp256k1_ec_pubkey_combine(ctx, &nxt, pts, 2)) { map_free(&b->baby); return 0; }
            cur = nxt;
        }
    }

    if (!baby_save(path, l1, &b->baby)) {
        printf("Warning: failed to save cache: %s\n", path);
    } else {
        printf("Built and saved baby table cache: %s (cap=%zu, used=%zu)\n", path, b->baby.cap, b->baby.size);
    }

    return 1;
}

static void bsgs_ctx_free(bsgs_ctx* b) {
    map_free(&b->baby);
    memset(b, 0, sizeof(*b));
}

/* Solve for m in [0,2^bits_total), assumes m != 0 (infinity not representable). */
static int bsgs_solve(const bsgs_ctx* b,
                      const secp256k1_pubkey* targetPm,
                      uint64_t* out_m) {
    const secp256k1_context* ctx = b->ctx;

    unsigned char t33[33];
    if (!pubkey_serialize33(ctx, targetPm, t33)) return 0;

    uint32_t i_found;
    if (map_get(&b->baby, t33, &i_found)) {
        *out_m = (uint64_t)i_found;
        return 1;
    }

    /* Q = target - MG */
    secp256k1_pubkey Q;
    {
        const secp256k1_pubkey* pts[2] = { targetPm, &b->neg_MG };
        if (!secp256k1_ec_pubkey_combine(ctx, &Q, pts, 2)) return 0;
    }

    secp256k1_pubkey jMG = b->MG; /* j=1 */

    for (uint64_t j = 1; j < b->J; j++) {
        /* i=0 check: target == jMG */
        unsigned char jmg33[33];
        if (!pubkey_serialize33(ctx, &jMG, jmg33)) return 0;
        if (pubkey_equal33(jmg33, t33)) {
            *out_m = j * b->M;
            return 1;
        }

        unsigned char q33[33];
        if (!pubkey_serialize33(ctx, &Q, q33)) return 0;

        if (map_get(&b->baby, q33, &i_found)) {
            *out_m = j * b->M + (uint64_t)i_found;
            return 1;
        }

        if (j + 1 < b->J) {
            /* Q <- Q - MG */
            {
                const secp256k1_pubkey* pts2[2] = { &Q, &b->neg_MG };
                secp256k1_pubkey nxt;
                if (!secp256k1_ec_pubkey_combine(ctx, &nxt, pts2, 2)) return 0;
                Q = nxt;
            }
            /* jMG <- jMG + MG */
            {
                const secp256k1_pubkey* pts3[2] = { &jMG, &b->MG };
                secp256k1_pubkey nxt2;
                if (!secp256k1_ec_pubkey_combine(ctx, &nxt2, pts3, 2)) return 0;
                jMG = nxt2;
            }
        }
    }

    return 0;
}

/* ---------------- benchmark ---------------- */

static void benchmark_bsgs(int bits_total, int l1, int trials) {
    printf("=== BSGS small-range DLP benchmark (secp256k1) ===\n");
    printf("Range: m in [0, 2^%d)\n", bits_total);
    printf("Split: l1=%d (baby M=2^l1), l2=%d (giant J=2^l2)\n", l1, bits_total - l1);
    printf("Trials: %d\n", trials);

    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_VERIFY);
    if (!ctx) {
        printf("Failed to create secp256k1 context\n");
        return;
    }

    bsgs_ctx solver;
    double t_build0 = now_seconds();
    if (!bsgs_ctx_init_cached(&solver, ctx, bits_total, l1)) {
        printf("Failed to init BSGS solver\n");
        secp256k1_context_destroy(ctx);
        return;
    }
    double t_build1 = now_seconds();
    printf("Init (load/build) time: %.6f sec\n", (t_build1 - t_build0));

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
        if (bsgs_solve(&solver, &Pm, &recovered) && recovered == m) {
            ok++;
        }
    }
    double t1 = now_seconds();

    double total = (t1 - t0);
    printf("Solved correctly: %d/%d\n", ok, trials);
    printf("Search-only total time: %.6f sec\n", total);
    printf("Average per solve (search-only): %.9f sec (%.2f ms)\n\n",
           total / trials, (total / trials) * 1e3);

    bsgs_ctx_free(&solver);
    secp256k1_context_destroy(ctx);
}

int main(int argc, char** argv) {
    int bits_total = 40;
    int l1 = 18;
    int trials = 1;

    if (argc >= 2) bits_total = atoi(argv[1]);
    if (argc >= 3) l1 = atoi(argv[2]);
    if (argc >= 4) trials = atoi(argv[3]);

    srand((unsigned)time(NULL));
    benchmark_bsgs(bits_total, l1, trials);
    return 0;
}
