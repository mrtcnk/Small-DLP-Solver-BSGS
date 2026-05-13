/*
 * bench_field.c
 *
 * Microbenchmark: secp256k1 field inversion vs field multiplication ratio.
 *
 * Measures the cost of:
 *   - secp256k1_fe_inv   (field inversion via Fermat: a^{p-2} mod p)
 *   - secp256k1_fe_mul   (field multiplication)
 *   - secp256k1_fe_sqr   (field squaring)
 *
 * Runs each operation N times with chained inputs (to prevent dead-code
 * elimination) and reports mean time and the inv/mul ratio.
 *
 * Build:
 *   cc -O3 -Wall -Wextra -o bench_field bench_field.c \
 *       -I/usr/local/include                           \
 *       -I/path/to/secp256k1/src                       \
 *       -L/usr/local/lib                               \
 *       -lsecp256k1
 *
 * Usage: ./bench_field
 */

#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <string.h>

#include <secp256k1.h>
#include "util.h"
#include "field.h"
#include "field_impl.h"
#include "int128_impl.h"
#include "group.h"
#include "group_impl.h"

/* ─────────────────────── timing ─────────────────────────────────── */

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

/* ─────────────────────── benchmark helpers ───────────────────────── */

#define N_WARMUP   1000
#define N_TRIALS   10000

/* Prevent compiler from optimising away the result */
static volatile uint32_t sink;

static void randomish_fe(secp256k1_fe* out, uint64_t seed) {
    unsigned char buf[32];
    for (int i = 0; i < 32; i++) {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        buf[i] = (unsigned char)(seed & 0xFF);
    }
    buf[0] &= 0x0F; /* keep below p */
    secp256k1_fe_set_b32_mod(out, buf);
    secp256k1_fe_normalize_var(out);
}

/* ─────────────────────── mul benchmark ──────────────────────────── */

static double bench_mul(void) {
    secp256k1_fe a, b, r;
    randomish_fe(&a, 0xdeadbeefULL);
    randomish_fe(&b, 0xcafebabeULL);

    /* warmup */
    for (int i = 0; i < N_WARMUP; i++) {
        secp256k1_fe_mul(&r, &a, &b);
        secp256k1_fe_mul(&a, &r, &b);
    }

    double t0 = now_ns();
    for (int i = 0; i < N_TRIALS; i++) {
        secp256k1_fe_mul(&r, &a, &b);
        secp256k1_fe_mul(&a, &r, &b);   /* chain: a depends on previous r */
    }
    double t1 = now_ns();

    /* use result to prevent dead-code elimination */
    unsigned char buf[32]; secp256k1_fe_get_b32(buf, &a);
    sink = buf[0];

    /* each iteration = 2 muls, return ns per mul */
    return (t1 - t0) / (2.0 * N_TRIALS);
}

/* ─────────────────────── sqr benchmark ──────────────────────────── */

static double bench_sqr(void) {
    secp256k1_fe a, r;
    randomish_fe(&a, 0x12345678ULL);

    for (int i = 0; i < N_WARMUP; i++) {
        secp256k1_fe_sqr(&r, &a);
        a = r;
    }

    double t0 = now_ns();
    for (int i = 0; i < N_TRIALS; i++) {
        secp256k1_fe_sqr(&r, &a);
        a = r;
    }
    double t1 = now_ns();

    unsigned char buf[32]; secp256k1_fe_get_b32(buf, &a);
    sink = buf[0];

    return (t1 - t0) / (double)N_TRIALS;
}

/* ─────────────────────── inv benchmark ──────────────────────────── */

static double bench_inv(void) {
    secp256k1_fe a, r;
    randomish_fe(&a, 0xabcdef01ULL);

    /* warmup */
    for (int i = 0; i < 20; i++) {
        secp256k1_fe_inv(&r, &a);
        a = r;
    }

    /* fewer trials — inversion is ~200× slower */
    int n = N_TRIALS / 10;
    double t0 = now_ns();
    for (int i = 0; i < n; i++) {
        secp256k1_fe_inv(&r, &a);
        a = r;   /* chain: each inv depends on previous result */
    }
    double t1 = now_ns();

    unsigned char buf[32]; secp256k1_fe_get_b32(buf, &a);
    sink = buf[0];

    return (t1 - t0) / (double)n;
}

/* ─────────────────────── main ───────────────────────────────────── */

int main(void) {
    printf("=== secp256k1 Field Operation Microbenchmark ===\n");
    printf("Warmup: %d ops | Trials: %d (inv: %d)\n\n",
           N_WARMUP, N_TRIALS, N_TRIALS / 10);

    /* Run 5 rounds and take the median */
    double mul_ns[5], sqr_ns[5], inv_ns[5];

    printf("Running mul benchmarks...\n");
    for (int r = 0; r < 5; r++) mul_ns[r] = bench_mul();

    printf("Running sqr benchmarks...\n");
    for (int r = 0; r < 5; r++) sqr_ns[r] = bench_sqr();

    printf("Running inv benchmarks...\n");
    for (int r = 0; r < 5; r++) inv_ns[r] = bench_inv();

    /* simple min as best-case estimate (avoids OS scheduling noise) */
    double mul_best = mul_ns[0], sqr_best = sqr_ns[0], inv_best = inv_ns[0];
    double mul_sum = 0, sqr_sum = 0, inv_sum = 0;
    for (int r = 0; r < 5; r++) {
        if (mul_ns[r] < mul_best) mul_best = mul_ns[r];
        if (sqr_ns[r] < sqr_best) sqr_best = sqr_ns[r];
        if (inv_ns[r] < inv_best) inv_best = inv_ns[r];
        mul_sum += mul_ns[r]; sqr_sum += sqr_ns[r]; inv_sum += inv_ns[r];
    }
    double mul_avg = mul_sum / 5, sqr_avg = sqr_sum / 5, inv_avg = inv_sum / 5;

    printf("\n=== Results (5 rounds) ===\n\n");
    printf("%-12s  %8s  %8s  %8s\n", "Operation", "Best(ns)", "Avg(ns)", "Ratio vs mul");
    printf("%-12s  %8.2f  %8.2f  %8s\n", "fe_mul",
           mul_best, mul_avg, "1.00x");
    printf("%-12s  %8.2f  %8.2f  %8.2fx\n", "fe_sqr",
           sqr_best, sqr_avg, sqr_best / mul_best);
    printf("%-12s  %8.2f  %8.2f  %8.2fx\n", "fe_inv",
           inv_best, inv_avg, inv_best / mul_best);

    printf("\n=== Key ratio for paper ===\n");
    printf("  1 field inversion = %.1fx a field multiplication (best)\n",
           inv_best / mul_best);
    printf("  1 field inversion = %.1fx a field multiplication (avg)\n",
           inv_avg / mul_avg);
    printf("\nNote: batch inversion of W elements costs\n");
    printf("  1 inv + 3*(W-1) muls = %.2f muls equivalent at W=512\n",
           inv_best/mul_best + 3.0*511);
    printf("  amortised per element: %.2f muls\n",
           (inv_best/mul_best + 3.0*511) / 512.0);

    return (int)sink & 0;   /* use sink to prevent dead-code elimination */
}