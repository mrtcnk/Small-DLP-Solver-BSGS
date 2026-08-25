#include "secp256k1.h"
#include "group.h"
#include "field.h"
#include "util.h"

/**
 * Co-Z Conjugate Addition: ZADDC (6M + 3S) with Edge-Case Fallback
 *
 * Input:
 *   Two Jacobian EC points P_1, P_2 with a shared Z-coordinate
 *   (P_1.Z == P_2.Z).
 *   Precondition: neither P_1 nor P_2 may be the point at infinity.
 *
 * Computes:
 *   res_a = P_1 + P_2
 *   res_s = P_1 - P_2
 *
 * Postcondition:
 *   res_a and res_s are returned fully normalized (X, Y, Z).
 *
 * Guaranteed Invariant:
 *   In the generic case (X_1 != X_2), res_a and res_s share Z
 *   (res_a->z == res_s->z). If X_1 == X_2, P_1 and P_2 coincide up to
 *   sign (P_1 == P_2 or P_1 == -P_2); exactly one of res_a, res_s
 *   becomes the point at infinity and the other becomes 2*P_1. In
 *   that case there is NO shared-Z guarantee between res_a and res_s
 *   -- callers must check the infinity flag before relying on Z.
 */
void secp256k1_gej_zaddc_var(
    secp256k1_gej *res_a,
    secp256k1_gej *res_s,
    const secp256k1_gej *p1,
    const secp256k1_gej *p2
) {
    secp256k1_fe c, w1, w2, da, ds, a1;
    secp256k1_fe dx, dmy, dpy, sum_w, neg_sum_w, diff_w, tmp;

    const secp256k1_fe *z  = &p1->z; /* p1 and p2 must share Z */
    const secp256k1_fe *x1 = &p1->x;
    const secp256k1_fe *y1 = &p1->y;
    const secp256k1_fe *x2 = &p2->x;
    const secp256k1_fe *y2 = &p2->y;

    /* dx = X1 - X2 */
    dx = *x1;
    secp256k1_fe_negate(&tmp, x2, 1);
    secp256k1_fe_add(&dx, &tmp);                     /* dx mag 3 */

    /* ==========================================================
     * EDGE CASE: X1 == X2  =>  P1 == P2  or  P1 == -P2
     * ========================================================== */
    if (EXPECT(secp256k1_fe_normalizes_to_zero_var(&dx), 0)) {
        /* dmy = Y1 - Y2 distinguishes the two sub-cases */
        dmy = *y1;
        secp256k1_fe_negate(&tmp, y2, 1);
        secp256k1_fe_add(&dmy, &tmp);

        if (secp256k1_fe_normalizes_to_zero_var(&dmy)) {
            /* P1 == P2 */
            secp256k1_gej_double_var(res_a, p1, NULL);   /* P_a = 2*P1 */
            secp256k1_gej_set_infinity(res_s);           /* P_s = O    */
        } else {
            /* P1 == -P2 */
            secp256k1_gej_double_var(res_s, p1, NULL);    /* P_s = 2*P1 */
            secp256k1_gej_set_infinity(res_a);           /* P_a = O    */
        }
        /* No shared-Z requirement in this branch -- done. */
        return;
    }

    /* ==========================================================
     * FAST PATH (6M + 3S)
     * ========================================================== */
    dmy = *y1;
    secp256k1_fe_negate(&tmp, y2, 1);
    secp256k1_fe_add(&dmy, &tmp);                     /* dmy = Y1-Y2, mag 3 */

    dpy = *y1;
    secp256k1_fe_add(&dpy, y2);                        /* dpy = Y1+Y2, mag 2 */

    secp256k1_fe_sqr(&c, &dx);                         /* C  = dx^2      [1S] */
    secp256k1_fe_mul(&res_a->z, z, &dx);               /* Zc = Z*dx      [1M] */

    secp256k1_fe_mul(&w1, x1, &c);                     /* W1 = X1*C      [1M] */
    secp256k1_fe_mul(&w2, x2, &c);                     /* W2 = X2*C      [1M] */
    secp256k1_fe_sqr(&da, &dmy);                       /* Da = (Y1-Y2)^2 [1S] */
    secp256k1_fe_sqr(&ds, &dpy);                        /* Ds = (Y1+Y2)^2 [1S] */

    /* X_a, X_s both subtract (W1+W2); compute that sum once. */
    sum_w = w1;
    secp256k1_fe_add(&sum_w, &w2);                     /* sum_w = W1+W2, mag 2 */
    secp256k1_fe_negate(&neg_sum_w, &sum_w, 2);        /* mag 3 */

    res_a->x = da;
    secp256k1_fe_add(&res_a->x, &neg_sum_w);           /* X_a = Da-W1-W2, mag 4 */

    res_s->x = ds;
    secp256k1_fe_add(&res_s->x, &neg_sum_w);           /* X_s = Ds-W1-W2, mag 4 */

    /* A1 = Y1*(W1-W2) */
    diff_w = w1;
    secp256k1_fe_negate(&tmp, &w2, 1);
    secp256k1_fe_add(&diff_w, &tmp);                   /* diff_w mag 3 */
    secp256k1_fe_mul(&a1, y1, &diff_w);                /* [1M] */

    /* Y_a = (Y1-Y2)*(W1 - X_a) - A1 */
    secp256k1_fe_negate(&tmp, &res_a->x, 4);           /* mag 5 */
    secp256k1_fe_add(&tmp, &w1);                       /* mag 6 */
    secp256k1_fe_mul(&res_a->y, &dmy, &tmp);           /* [1M] */
    secp256k1_fe_negate(&tmp, &a1, 1);
    secp256k1_fe_add(&res_a->y, &tmp);                 /* Y_a mag 3 */

    /* Y_s = (Y1+Y2)*(W1 - X_s) - A1 */
    secp256k1_fe_negate(&tmp, &res_s->x, 4);           /* mag 5 */
    secp256k1_fe_add(&tmp, &w1);                       /* mag 6 */
    secp256k1_fe_mul(&res_s->y, &dpy, &tmp);           /* [1M] */
    secp256k1_fe_negate(&tmp, &a1, 1);
    secp256k1_fe_add(&res_s->y, &tmp);                 /* Y_s mag 3 */

    /* Zc computed once above; dedupe the copy+normalize (Z is shared). */
    secp256k1_fe_normalize_var(&res_a->z);
    res_s->z = res_a->z;

    secp256k1_fe_normalize_var(&res_a->x);
    secp256k1_fe_normalize_var(&res_s->x);
    secp256k1_fe_normalize_var(&res_a->y);
    secp256k1_fe_normalize_var(&res_s->y);

    res_a->infinity = 0;
    res_s->infinity = 0;
}
