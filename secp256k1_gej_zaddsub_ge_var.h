#include "secp256k1.h"
#include "group.h"
#include "field.h"
#include "util.h"

/**
 * Optimized Co-Z combined ADD and SUB (15M + 4S) with Edge-Case Fallback
 *
 * Input:
 *   Two Jacobian EC points P_a, P_s with a shared Z-coordinate
 *   (P_a.Z == P_s.Z), and an affine EC point S = (x, y).
 *   Precondition: none of P_a, P_s, S may be the point at infinity.
 *
 * Computes:
 *   res_a = P_a + S
 *   res_s = P_s - S
 *
 * Postcondition:
 *   res_a and res_s are returned fully normalized (X, Y, Z).
 *
 * Guaranteed Invariant:
 *   res_a and res_s share Z (res_a->z == res_s->z) whenever neither
 *   result is the point at infinity. Callers MUST check the infinity
 *   flag on both outputs before relying on shared Z.
 */
static void secp256k1_gej_zaddsub_ge_var(
    secp256k1_gej *res_a,
    secp256k1_gej *res_s,
    const secp256k1_gej *pa,
    const secp256k1_gej *ps,
    const secp256k1_ge *s
) {
    secp256k1_fe z_sq, z_cu, u, v;
    secp256k1_fe ha, hs, da, ds;
    secp256k1_fe hc, hc2, hc3;
    secp256k1_fe wu, w1, w2, k1, k2;
    secp256k1_fe la, ls, la2, ls2;
    secp256k1_fe tmp, sum;

    /* Aliases to point components for readability */
    const secp256k1_fe *x  = &s->x;
    const secp256k1_fe *y  = &s->y;
    const secp256k1_fe *z  = &pa->z; /* pa and ps must share the same Z */
    const secp256k1_fe *x1 = &pa->x;
    const secp256k1_fe *y1 = &pa->y;
    const secp256k1_fe *x2 = &ps->x;
    const secp256k1_fe *y2 = &ps->y;

    /* ==========================================================
     * 1. SETUP PHASE 
     * ========================================================== */
    secp256k1_fe_sqr(&z_sq, z);              /* Z_sq = Z^2 */
    secp256k1_fe_mul(&z_cu, &z_sq, z);       /* Z_cu = Z^3 */

    secp256k1_fe_mul(&u, x, &z_sq);          /* U = x * Z^2 */
    secp256k1_fe_mul(&v, y, &z_cu);          /* V = y * Z^3 */

    /* H_a = U - X_1 */
    secp256k1_fe_negate(&ha, x1, 1);
    secp256k1_fe_add(&ha, &u);

    /* H_s = U - X_2 */
    secp256k1_fe_negate(&hs, x2, 1);
    secp256k1_fe_add(&hs, &u);

    /* D_a = V - Y_1 */
    secp256k1_fe_negate(&da, y1, 1);
    secp256k1_fe_add(&da, &v);

    /* D_s = -(V + Y_2) */
    sum = v; secp256k1_fe_add(&sum, y2);
    secp256k1_fe_negate(&ds, &sum, 2);

    /* ==========================================================
     * 2. HIGH-PERFORMANCE EDGE CASE FALLBACK
     * Triggered if P_a or P_s has the same X-coordinate as S.
     * ========================================================== */
    if (EXPECT(secp256k1_fe_normalizes_to_zero_var(&ha) ||
           secp256k1_fe_normalizes_to_zero_var(&hs), 0)) {
        
        /* 
         * STATE SPACE OF ALL 8 CASES:
         * 1) Pa = Ps = S      => res_a: Double, res_s: Infinity
         * 2) Pa = Ps = -S     => res_a: Infinity, res_s: Double
         * 3) Pa = Ps <> S     => Normal Shared Z (Caught by Fast Path)
         * 4) Pa <> Ps, Pa = S => res_a: Double, res_s: Normal Sub
         * 5) Pa <> Ps, Pa = -S=> res_a: Infinity, res_s: Normal Sub
         * 6) Pa <> Ps, Ps = S => res_a: Normal Add, res_s: Infinity
         * 7) Pa <> Ps, Ps = -S=> res_a: Normal Add, res_s: Double
         * 8) Pa <> Ps <> S    => Normal Shared Z (Caught by Fast Path)
         */

        /* Handle res_a (P_a + S) */
        if (secp256k1_fe_normalizes_to_zero_var(&ha)) {
            if (secp256k1_fe_normalizes_to_zero_var(&da)) {
                secp256k1_gej_double_var(res_a, pa, NULL);    /* Cases 1, 4 */
            } else {
                secp256k1_gej_set_infinity(res_a);            /* Cases 2, 5 */
            }
        } else {
            secp256k1_gej_add_ge_var(res_a, pa, s, NULL);     /* Cases 6, 7 */
        }

        /* Handle res_s (P_s - S) */
        if (secp256k1_fe_normalizes_to_zero_var(&hs)) {
            if (secp256k1_fe_normalizes_to_zero_var(&ds)) {
                secp256k1_gej_double_var(res_s, ps, NULL);    /* Cases 2, 7 */
            } else {
                secp256k1_gej_set_infinity(res_s);            /* Cases 1, 6 */
            }
        } else {
            secp256k1_ge neg_s = *s;
            secp256k1_fe_negate(&neg_s.y, &neg_s.y, 1);
            secp256k1_fe_normalize_var(&neg_s.y);
            secp256k1_gej_add_ge_var(res_s, ps, &neg_s, NULL); /* Cases 4, 5 */
        }
        
        /* 
         * FIX FOR CASES 4 & 7 (and any un-synced valid combinations):
         * If both resulting points are valid (neither is infinity), they 
         * currently have mismatched Z coordinates from the fallback functions.
         * We apply Jacobian cross-scaling to force them to share Z = Z_a * Z_s.
         */
        if (!res_a->infinity && !res_s->infinity) {
            secp256k1_fe za2, za3, zs2, zs3, z_shared;

            secp256k1_fe_sqr(&za2, &res_a->z);                 /* Z_a^2 */
            secp256k1_fe_mul(&za3, &za2, &res_a->z);           /* Z_a^3 */

            secp256k1_fe_sqr(&zs2, &res_s->z);                 /* Z_s^2 */
            secp256k1_fe_mul(&zs3, &zs2, &res_s->z);           /* Z_s^3 */

            /* Scale res_a by Z_s */
            secp256k1_fe_mul(&res_a->x, &res_a->x, &zs2);
            secp256k1_fe_mul(&res_a->y, &res_a->y, &zs3);

            /* Scale res_s by Z_a */
            secp256k1_fe_mul(&res_s->x, &res_s->x, &za2);
            secp256k1_fe_mul(&res_s->y, &res_s->y, &za3);

            /* Set the unified shared Z coordinate */
            secp256k1_fe_mul(&z_shared, &res_a->z, &res_s->z);
            res_a->z = z_shared;
            res_s->z = z_shared;
        }

        return; 
    }
    /* ========================================================== */

    /* ==========================================================
     * 3. FAST PATH: THE SHARED H_c CORE (15M + 4S)
     * Resumes here for Cases 3 and 8 (General Case).
     * ========================================================== */
    secp256k1_fe_mul(&hc, &ha, &hs);         /* H_c = H_a * H_s */
    secp256k1_fe_mul(&res_a->z, z, &hc);     /* Z_c = Z * H_c */
    secp256k1_fe_normalize_var(&res_a->z);
    res_s->z = res_a->z;                     /* Force sync shared Z */

    secp256k1_fe_sqr(&hc2, &hc);             /* H_c2 = H_c^2 */
    secp256k1_fe_mul(&hc3, &hc2, &hc);       /* H_c3 = H_c^3 */

    /* Branch Variables */
    secp256k1_fe_mul(&wu, &u, &hc2);         /* W_U = U * H_c2 */
    secp256k1_fe_mul(&w1, x1, &hc2);         /* W_1 = X_1 * H_c2 */
    secp256k1_fe_mul(&w2, x2, &hc2);         /* W_2 = X_2 * H_c2 */

    secp256k1_fe_mul(&k1, y1, &hc3);         /* K_1 = Y_1 * H_c3 */
    secp256k1_fe_mul(&k2, y2, &hc3);         /* K_2 = Y_2 * H_c3 */

    secp256k1_fe_mul(&la, &da, &hs);         /* L_a = D_a * H_s */
    secp256k1_fe_mul(&ls, &ds, &ha);         /* L_s = D_s * H_a */

    /* ==========================================================
     * 4. FINAL COORDINATES
     * ========================================================== */
    secp256k1_fe_sqr(&la2, &la);             /* L_a2 = L_a^2 */
    secp256k1_fe_sqr(&ls2, &ls);             /* L_s2 = L_s^2 */

    /* X_a = L_a2 - (W_U + W_1) */
    sum = wu; secp256k1_fe_add(&sum, &w1);
    secp256k1_fe_negate(&tmp, &sum, 2);           
    res_a->x = la2;
    secp256k1_fe_add(&res_a->x, &tmp);
    secp256k1_fe_normalize_var(&res_a->x);

    /* X_s = L_s2 - (W_U + W_2) */
    sum = wu; secp256k1_fe_add(&sum, &w2);
    secp256k1_fe_negate(&tmp, &sum, 2);
    res_s->x = ls2;
    secp256k1_fe_add(&res_s->x, &tmp);
    secp256k1_fe_normalize_var(&res_s->x);

    /* Y_a = L_a * (W_1 - X_a) - K_1 */
    secp256k1_fe_negate(&tmp, &res_a->x, 1);
    secp256k1_fe_add(&tmp, &w1);             /* tmp = W_1 - X_a */
    secp256k1_fe_mul(&res_a->y, &la, &tmp);
    secp256k1_fe_negate(&tmp, &k1, 1);
    secp256k1_fe_add(&res_a->y, &tmp);

    /* Y_s = L_s * (W_2 - X_s) - K_2 */
    secp256k1_fe_negate(&tmp, &res_s->x, 1);
    secp256k1_fe_add(&tmp, &w2);             /* tmp = W_2 - X_s */
    secp256k1_fe_mul(&res_s->y, &ls, &tmp);
    secp256k1_fe_negate(&tmp, &k2, 1);
    secp256k1_fe_add(&res_s->y, &tmp);

    /* Final canonical normalization */
    secp256k1_fe_normalize_var(&res_a->y);
    secp256k1_fe_normalize_var(&res_s->y);

    /* Mark as valid affine-mapped points */
    res_a->infinity = 0;
    res_s->infinity = 0;
}