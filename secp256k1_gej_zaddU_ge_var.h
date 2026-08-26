#include "secp256k1.h"
#include "group.h"
#include "field.h"
#include "util.h"

/**
 * ZADDU: mixed addition (a + b) with Co-Z update of a companion point p.
 *
 * Built directly from secp256k1_gej_add_ge_var's own implementation.
 * Computes r = a + b exactly as secp256k1_gej_add_ge_var does, and
 * additionally rescales a second Jacobian point p -- which must already
 * share a's Z-coordinate -- onto r's new Z, using only the internal
 * scaling factor lambda = H_a that add_ge_var already computes as a
 * byproduct of computing r.
 *
 * Input:
 *   a, b: as in secp256k1_gej_add_ge_var.
 *   p: a Jacobian point with p->z == a->z.
 *   Precondition: a and b are not the point at infinity.
 *
 * Output:
 *   r = a + b (identical to secp256k1_gej_add_ge_var's r).
 *   p_out = p rescaled onto r's Z:
 *           p_out = (Ha^2 * p.x : Ha^3 * p.y : r.z)
 *           i.e. the same affine point as p, now sharing Z with r.
 *
 * Aliasing: r is permitted to alias a (i.e. r == a is safe, matching
 * secp256k1_gej_add_ge_var's own supported usage), and p_out is
 * permitted to alias p (p_out == p is safe). This covers the common
 * in-place ladder pattern
 *     secp256k1_gej_zaddU_ge_var(&r, &p, &r, &b, &p);
 * No other aliasing between {r, p_out, a, b, p} is checked or supported.
 *
 * Postcondition:
 *   r and p_out are returned fully normalized (X, Y, Z).
 *
 * Special cases (X-coordinate collision between a and b):
 *   a == b:  r = 2*a (doubling). p_out IS properly synced here too --
 *            the doubling formula's own Z-scaling factor (lambda =
 *            a->y, since Z3 = Y1*Z1) updates p onto r's new Z exactly
 *            like the generic path does. res_a->z == p_out->z holds.
 *   a == -b: r = infinity (Z = 0 by convention). There is no nonzero
 *            scaling factor to sync with here -- p_out is returned as
 *            an UNSCALED COPY of p and does NOT share Z with r.
 * Callers must check r->infinity and, if infinite, must not assume
 * r->z == p_out->z.
 */
static void secp256k1_gej_zaddU_ge_var(secp256k1_gej *r, secp256k1_gej *p_out,
                                        const secp256k1_gej *a, const secp256k1_ge *b,
                                        const secp256k1_gej *p) {
    secp256k1_fe z12, u1, u2, s1, s2, h, i, h2, h3, t;
    secp256k1_fe neg_ha3, ha2_pos, ha3_pos;

    SECP256K1_GEJ_VERIFY(a);
    SECP256K1_GE_VERIFY(b);
    SECP256K1_GEJ_VERIFY(p);
    VERIFY_CHECK(!a->infinity);
    VERIFY_CHECK(!b->infinity);

    secp256k1_fe_sqr(&z12, &a->z);
    u1 = a->x;
    secp256k1_fe_mul(&u2, &b->x, &z12);
    s1 = a->y;
    secp256k1_fe_mul(&s2, &b->y, &z12); secp256k1_fe_mul(&s2, &s2, &a->z);
    secp256k1_fe_negate(&h, &u1, SECP256K1_GEJ_X_MAGNITUDE_MAX); secp256k1_fe_add(&h, &u2);
    secp256k1_fe_negate(&i, &s2, 1); secp256k1_fe_add(&i, &s1);
    if (EXPECT(secp256k1_fe_normalizes_to_zero_var(&h), 0)) {
        if (secp256k1_fe_normalizes_to_zero_var(&i)) {
            /* P1 == P2 (a == b): r = 2*a. secp256k1_gej_double's own
             * code computes Z3 = Y1*Z1 (and secp256k1_gej_double_var's
             * rzr, when requested, is exactly a->y) -- so lambda = a->y
             * is r's real Co-Z scaling factor here, not a fallback.
             * Since a->z == p->z, that same lambda updates p onto r's
             * new Z exactly like the generic path does. */
            secp256k1_fe ay, ay2, ay3;
            ay = a->y;               /* capture BEFORE double_var -- r may alias a, which
                                        would otherwise overwrite a->y before we read it below */
            secp256k1_gej_double_var(r, a, NULL);

            secp256k1_fe_sqr(&ay2, &ay);
            secp256k1_fe_mul(&ay3, &ay2, &ay);
            secp256k1_fe_mul(&p_out->x, &p->x, &ay2);
            secp256k1_fe_mul(&p_out->y, &p->y, &ay3);
            p_out->z = r->z;
            p_out->infinity = 0;
        } else {
            /* P1 == -P2 (a == -b): r = infinity, Z = 0 by convention
             * (secp256k1_gej_set_infinity). There is no nonzero scaling
             * factor to sync with -- p_out is an UNSYNCED copy of p and
             * does NOT share Z with r. */
            secp256k1_gej_set_infinity(r);
            *p_out = *p;
        }
        secp256k1_fe_normalize_var(&r->x);
        secp256k1_fe_normalize_var(&r->y);
        secp256k1_fe_normalize_var(&r->z);
        secp256k1_fe_normalize_var(&p_out->x);
        secp256k1_fe_normalize_var(&p_out->y);
        secp256k1_fe_normalize_var(&p_out->z);
        SECP256K1_GEJ_VERIFY(r);
        SECP256K1_GEJ_VERIFY(p_out);
        return;
    }

    r->infinity = 0;
    secp256k1_fe_mul(&r->z, &a->z, &h);

    secp256k1_fe_sqr(&h2, &h);                       /* h2 = +Ha^2, mag 1 */
    ha2_pos = h2;                                    /* save +Ha^2 before negating h2 in place below */
    secp256k1_fe_negate(&h2, &h2, 1);                /* h2 = -Ha^2, mag 2 (for the rest of this formula) */
    secp256k1_fe_mul(&h3, &h2, &h);                  /* h3 = -Ha^3, mag 1 */
    neg_ha3 = h3;                                    /* save before h3 gets reused below */
    secp256k1_fe_mul(&t, &u1, &h2);

    secp256k1_fe_sqr(&r->x, &i);
    secp256k1_fe_add(&r->x, &h3);
    secp256k1_fe_add(&r->x, &t);
    secp256k1_fe_add(&r->x, &t);

    secp256k1_fe_add(&t, &r->x);
    secp256k1_fe_mul(&r->y, &t, &i);
    secp256k1_fe_mul(&h3, &h3, &s1);
    secp256k1_fe_add(&r->y, &h3);

    SECP256K1_GEJ_VERIFY(r);

    /* ---- Co-Z update of p onto r's new Z ----
     * ha2_pos was captured for free above (a copy, before h2 got
     * negated in place) -- only ha3_pos needs an explicit sign-flip,
     * since h3 was only ever computed in its negative form. */
    secp256k1_fe_negate(&ha3_pos, &neg_ha3, 1);      /* +Ha^3, mag 2 */
    secp256k1_fe_mul(&p_out->x, &p->x, &ha2_pos);
    secp256k1_fe_mul(&p_out->y, &p->y, &ha3_pos);
    p_out->z = r->z;
    p_out->infinity = 0;

    secp256k1_fe_normalize_var(&r->x);
    secp256k1_fe_normalize_var(&r->y);
    secp256k1_fe_normalize_var(&r->z);
    secp256k1_fe_normalize_var(&p_out->x);
    secp256k1_fe_normalize_var(&p_out->y);
    secp256k1_fe_normalize_var(&p_out->z);

    SECP256K1_GEJ_VERIFY(p_out);
}