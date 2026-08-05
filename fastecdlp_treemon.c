/*
 * fastecdlp_treemon.c
 *
 * Faithful C re-implementation of FastECDLP (Tang et al., ePrint 2022/1573)
 * using the full binary product tree (TreeMon) for batch inversion,
 * matching the original Go/C++ architecture as closely as possible.
 *
 * Architecture (matching original FastECDLP):
 *   Phase 1: compute global denom[j] = Pm.x - T2[j].x  (sequential, single thread)
 *   Phase 2: full binary TreeMon over all J denominators (parallelised per level)
 *            - Forward pass: build product tree bottom-up
 *            - Single inversion at root
 *            - Backward pass: propagate inverses top-down
 *   Phase 3: T threads search their J/T chunk using shared inv_den
 *
 * Build:
 *   cc -O3 -Wall -Wextra -o fastecdlp_treemon fastecdlp_treemon.c \
 *       -I/usr/local/include                                        \
 *       -I/path/to/secp256k1/src                                    \
 *       -L/usr/local/lib                                            \
 *       -lsecp256k1 -lpthread
 *
 * Usage: ./fastecdlp_treemon <bits> <l1> <trials> <threads>
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
    for (int i = 0; i < 8; i++) { out32[31-i]=(unsigned char)(x&0xFF); x>>=8; }
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
    size_t s=m->section_size; int n=0;
    const entry_packed* e;
    e=&m->tab[cpos(0,xb,s)]; if(e->val&&e->key==ckey(0,xb)) out[n++]=e->val;
    e=&m->tab[cpos(1,xb,s)]; if(e->val&&e->key==ckey(1,xb)) out[n++]=e->val;
    e=&m->tab[cpos(2,xb,s)]; if(e->val&&e->key==ckey(2,xb)) out[n++]=e->val;
    for(int i=0;i<m->stash_count;i++)
        if(memcmp(m->stash_xb[i],xb,32)==0) out[n++]=m->stash_val[i];
    return n;
}
static int load_baby_table(cuckoo_map* baby, int l1) {
    char fname[128];
    snprintf(fname,sizeof(fname),"bsgs_baby_cuckoo_secp256k1_l1_%d_window.bin",l1);
    int fd=open(fname,O_RDONLY); if(fd<0){perror(fname);return 0;}
    baby_hdr hdr;
    if(!read_all(fd,&hdr,sizeof(hdr))||hdr.magic!=BABY_MAGIC||
       hdr.version!=4||(int)hdr.l1!=l1){
        fprintf(stderr,"Bad header %s\n",fname);close(fd);return 0;}
    memset(baby,0,sizeof(*baby));
    baby->section_size=(size_t)hdr.section_size;
    baby->total_bins=CUCKOO_K*(size_t)hdr.section_size;
    baby->size=(size_t)hdr.used_count; baby->stash_count=(int)hdr.stash_count;
    if(!read_all(fd,baby->stash_xb,sizeof(baby->stash_xb))||
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

/* ─────────────────── T2 table ───────────────────────────────────── */
static secp256k1_ge* build_t2(const secp256k1_context* ctx,
                              uint64_t M, int l2, double* t_out) {
    uint64_t J = 1ULL << l2;
    printf("Building T2: %"PRIu64" affine points...\n", J);
    unsigned char sc_M[32]; u64_to_scalar32_be(M, sc_M);
    secp256k1_pubkey pk_MG;
    secp256k1_ec_pubkey_create(ctx, &pk_MG, sc_M);
    secp256k1_ge MG_ge; pubkey_to_ge(&pk_MG, &MG_ge);
    secp256k1_gej* jac=(secp256k1_gej*)malloc(J*sizeof(secp256k1_gej));
    if(!jac){fprintf(stderr,"OOM: T2 Jacobian\n");return NULL;}
    double t0=now_seconds();
    secp256k1_gej acc; secp256k1_gej_set_infinity(&acc);
    for(uint64_t j=0;j<J;j++){
        jac[j]=acc;
        secp256k1_gej_add_ge(&acc,&acc,&MG_ge);
    }
    secp256k1_ge* T2=(secp256k1_ge*)malloc(J*sizeof(secp256k1_ge));
    if(!T2){free(jac);fprintf(stderr,"OOM: T2 affine\n");return NULL;}
    secp256k1_ge_set_all_gej_var(T2,jac,(size_t)J);
    free(jac);
    *t_out=now_seconds()-t0;
    printf("T2 built: %.1f sec | Memory: %.2f GB\n",
           *t_out,(double)(J*sizeof(secp256k1_ge))/(1ULL<<30));
    return T2;
}

static secp256k1_ge* get_t2(const secp256k1_context* ctx,
                            uint64_t M, int l1, int l2, double* t_out) {
    char path[128];
    snprintf(path, sizeof(path), "fastecdlp_t2_secp256k1_l1_%d.bin", l1);
    uint64_t J=1ULL<<l2; *t_out=0.0;
    struct stat st;
    if(stat(path,&st)==0&&(uint64_t)st.st_size==J*sizeof(secp256k1_ge)){
        printf("Loading T2 from cache: %s (%.2f GB)...\n",
               path,(double)st.st_size/(1ULL<<30));
        double t0=now_seconds();
        int fd=open(path,O_RDONLY);
        if(fd>=0){
            secp256k1_ge* T2=(secp256k1_ge*)malloc(J*sizeof(secp256k1_ge));
            if(T2&&read_all(fd,T2,J*sizeof(secp256k1_ge))){
                close(fd);
                printf("T2 loaded: %.2f sec\n",now_seconds()-t0);
                return T2;
            }
            free(T2); close(fd);
        }
    }
    secp256k1_ge* T2=build_t2(ctx,M,l2,t_out);
    if(T2){
        int fd=open(path,O_WRONLY|O_CREAT|O_TRUNC,0644);
        if(fd>=0){
            printf("Saving T2 to %s...\n",path);
            write_all(fd,T2,J*sizeof(secp256k1_ge));
            close(fd); printf("T2 saved.\n");
        }
    }
    return T2;
}

/* ─────────────── Parallel worker for Phase 1 and TreeMon ────────── */

/* Phase 1 worker: compute denom[j] = Pm.x - T2[j].x for a chunk */
typedef struct {
    const secp256k1_ge* T2;
    secp256k1_fe*       denom;
    secp256k1_fe        Pm_x;
    uint64_t            j_start;
    uint64_t            j_end;
} phase1_args;

static void* phase1_worker(void* varg) {
    phase1_args* a = (phase1_args*)varg;
    for (uint64_t j = a->j_start; j < a->j_end; j++) {
        if (a->T2[j].infinity) {
            secp256k1_fe_set_int(&a->denom[j], 1);
        } else {
            secp256k1_fe tx = a->T2[j].x;
            secp256k1_fe_normalize_var(&tx);
            secp256k1_fe_negate(&a->denom[j], &tx, 1);
            secp256k1_fe_add(&a->denom[j], &a->Pm_x);
            secp256k1_fe_normalize_var(&a->denom[j]);
        }
    }
    return NULL;
}

/* TreeMon worker: multiply pairs at one level of the binary tree */
typedef struct {
    secp256k1_fe* tree;   /* full tree array */
    uint64_t      level_start;  /* first node index at this level */
    uint64_t      node_start;   /* first node this thread handles */
    uint64_t      node_end;     /* exclusive */
} treemon_worker_args;

static void* treemon_forward_worker(void* varg) {
    treemon_worker_args* a = (treemon_worker_args*)varg;
    /* Each node at level_start+k = product of two children */
    for (uint64_t k = a->node_start; k < a->node_end; k++) {
        uint64_t node  = a->level_start + k;
        uint64_t left  = 2*node + 1;
        uint64_t right = 2*node + 2;
        secp256k1_fe_mul(&a->tree[node], &a->tree[left], &a->tree[right]);
    }
    return NULL;
}

static void* treemon_backward_worker(void* varg) {
    treemon_worker_args* a = (treemon_worker_args*)varg;
    for (uint64_t k = a->node_start; k < a->node_end; k++) {
        uint64_t node  = a->level_start + k;
        uint64_t left  = 2*node + 1;
        uint64_t right = 2*node + 2;
        /* left_inv  = parent_inv * right_child */
        /* right_inv = parent_inv * left_child  */
        secp256k1_fe left_inv, right_inv;
        secp256k1_fe_mul(&left_inv,  &a->tree[node], &a->tree[right]);
        secp256k1_fe_mul(&right_inv, &a->tree[node], &a->tree[left]);
        a->tree[left]  = left_inv;
        a->tree[right] = right_inv;
    }
    return NULL;
}

/*
 * TreeMon: full binary product tree batch inversion.
 *
 * For n = 2^l2 leaves (denom[0..n-1]):
 *   Tree size = 2n - 1 nodes stored in array of size 2n.
 *   Leaves are stored at indices [n-1, 2n-2] (0-indexed).
 *   Internal nodes at [0, n-2], root at [0].
 *
 *   Forward pass: build product tree bottom-up (parallel per level)
 *   Single inversion at root
 *   Backward pass: propagate inverses top-down (parallel per level)
 *   Result: tree[n-1+j] = denom[j]^{-1}
 */
static secp256k1_fe* treemon(secp256k1_fe* denom, uint64_t n, int threads) {
    /* n must be a power of 2 */
    uint64_t tree_size = 2 * n;
    secp256k1_fe* tree = (secp256k1_fe*)malloc(tree_size * sizeof(secp256k1_fe));
    if (!tree) { fprintf(stderr,"OOM: treemon\n"); return NULL; }

    /* Copy leaves into tree */
    for (uint64_t j = 0; j < n; j++)
        tree[n - 1 + j] = denom[j];

    pthread_t* tids = (pthread_t*)malloc((size_t)threads * sizeof(pthread_t));
    treemon_worker_args* wargs = (treemon_worker_args*)malloc(
    (size_t)threads * sizeof(treemon_worker_args));

    /* ── Forward pass: bottom-up, level by level ── */
    uint64_t level_size = n / 2;      /* number of nodes at this level */
    uint64_t level_start = n/2 - 1;  /* index of first node at this level */

    while (level_size >= 1) {
        uint64_t chunk = (level_size + (uint64_t)threads - 1) / (uint64_t)threads;
        int t_count = (level_size < (uint64_t)threads) ? (int)level_size : threads;

        for (int t = 0; t < t_count; t++) {
            wargs[t].tree        = tree;
            wargs[t].level_start = level_start;
            wargs[t].node_start  = (uint64_t)t * chunk;
            wargs[t].node_end    = (uint64_t)t*chunk+chunk < level_size
                                   ? (uint64_t)t*chunk+chunk : level_size;
            pthread_create(&tids[t], NULL, treemon_forward_worker, &wargs[t]);
        }
        for (int t = 0; t < t_count; t++) pthread_join(tids[t], NULL);

        if (level_size == 1) break;
        level_start = (level_start - 1) / 2;
        level_size /= 2;
    }

    /* ── Single inversion at root ── */
    secp256k1_fe_inv(&tree[0], &tree[0]);

    /* ── Backward pass: top-down, level by level ── */
    level_size  = 1;
    level_start = 0;

    while (level_size <= n/2) {
        uint64_t chunk = (level_size + (uint64_t)threads - 1) / (uint64_t)threads;
        int t_count = (level_size < (uint64_t)threads) ? (int)level_size : threads;

        for (int t = 0; t < t_count; t++) {
            wargs[t].tree        = tree;
            wargs[t].level_start = level_start;
            wargs[t].node_start  = (uint64_t)t * chunk;
            wargs[t].node_end    = (uint64_t)t*chunk+chunk < level_size
                                   ? (uint64_t)t*chunk+chunk : level_size;
            pthread_create(&tids[t], NULL, treemon_backward_worker, &wargs[t]);
        }
        for (int t = 0; t < t_count; t++) pthread_join(tids[t], NULL);

        level_size  *= 2;
        level_start  = 2*level_start + 1;
    }

    free(tids); free(wargs);
    /* Leaves now contain the inverses */
    return tree;  /* caller reads tree[n-1+j] for denom[j]^{-1} */
}

/* ─────────────────── Phase 3: parallel search thread ────────────── */
typedef struct {
    const cuckoo_map*        baby;
    const secp256k1_context* ctx;
    const secp256k1_ge*      T2;
    const secp256k1_fe*      inv_den; /* tree leaves: inv_den[n-1+j] = z_j^{-1} */
    uint64_t                 n;       /* = J = 2^l2 (leaf offset) */
    secp256k1_fe             Pm_x;
    secp256k1_fe             Pm_y;
    uint64_t                 M;
    uint64_t                 j_start;
    uint64_t                 j_end;
    const unsigned char*     target33;
    atomic_int*              found_flag;
    uint64_t                 result_m;
    int                      found;
} search_args;

static void* search_worker(void* varg) {
    search_args* a = (search_args*)varg;

    for (uint64_t j = a->j_start;
         j < a->j_end && !atomic_load(a->found_flag); j++) {

        secp256k1_fe tx = a->T2[j].x, ty = a->T2[j].y;
        secp256k1_fe_normalize_var(&tx);
        secp256k1_fe_normalize_var(&ty);

        secp256k1_fe num = a->Pm_y;
        secp256k1_fe_add(&num, &ty);
        secp256k1_fe lam;
        /* leaf index in tree = n-1+j */
        secp256k1_fe_mul(&lam, &num, &a->inv_den[a->n - 1 + j]);

        secp256k1_fe lam2, x3, neg;
        secp256k1_fe_sqr(&lam2, &lam);
        x3 = lam2;
        neg = a->Pm_x; secp256k1_fe_negate(&neg,&neg,1);
        secp256k1_fe_add(&x3, &neg);
        neg = tx; secp256k1_fe_negate(&neg,&neg,1);
        secp256k1_fe_add(&x3, &neg);
        secp256k1_fe_normalize_var(&x3);

        unsigned char xb[32]; secp256k1_fe_get_b32(xb,&x3);
        uint32_t cands[CUCKOO_K+CUCKOO_STASH_SZ];
        int nc=map_get_all(a->baby,xb,cands);
        for(int ci=0;ci<nc;ci++){
            uint64_t m1=j*a->M+(uint64_t)cands[ci];
            uint64_t m2=j*a->M-(uint64_t)cands[ci];
            if(verify_candidate(a->ctx,m1,a->target33)){
                a->result_m=m1;a->found=1;
                atomic_store(a->found_flag,1);break;}
            if(verify_candidate(a->ctx,m2,a->target33)){
                a->result_m=m2;a->found=1;
                atomic_store(a->found_flag,1);break;}
        }
    }
    return NULL;
}

/* ─────────────────── solve ──────────────────────────────────────── */
static int fastecdlp_solve(const cuckoo_map* baby,
                           const secp256k1_context* ctx,
                           const secp256k1_ge* T2,
                           const secp256k1_ge* Pm_ge,
                           uint64_t M, int l2, int threads,
                           const unsigned char target33[33],
                           uint64_t* out_m) {
    uint64_t J = 1ULL << l2;

    secp256k1_fe Pm_x=Pm_ge->x; secp256k1_fe_normalize_var(&Pm_x);
    secp256k1_fe Pm_y=Pm_ge->y; secp256k1_fe_normalize_var(&Pm_y);

    /* j=0: direct baby lookup for Pm = i.G */
    unsigned char txb[32];
    memcpy(txb, target33 + 1, 32); /* x-coordinate from compressed pubkey */
    uint32_t cands[3 + CUCKOO_STASH_SZ];
    int nc = map_get_all(baby, txb, cands);
    for (int ci = 0; ci < nc; ci++) {
        if (verify_candidate(ctx, (uint64_t)cands[ci], target33)) {
            *out_m = (uint64_t)cands[ci]; return 1;
        }
    }

    /* ── Phase 1: sequential denom computation (matches Tang et al. C++ source)
     * BuildZAndTryZeroDiff in ahesm2.cc is a plain sequential loop.
     * Only Phase 2 (TreeMon) and Phase 3 (search) are parallelised.
     * Bug fix: detect zero denominator early —
     * when denom[j]=0 it means Pm.x == T2[j].x, i.e. Pm = ±j*M*G (m = j*M),
     * which would cause division by zero in batch inversion. ── */
    secp256k1_fe* denom = (secp256k1_fe*)malloc(J * sizeof(secp256k1_fe));
    if (!denom) { fprintf(stderr,"OOM: denom\n"); return 0; }

    int found = 0;
    secp256k1_fe_set_int(&denom[0], 1);
    for (uint64_t j = 1; j < J; j++) {
        secp256k1_fe tx = T2[j].x;
        secp256k1_fe_normalize_var(&tx);
        secp256k1_fe_negate(&denom[j], &tx, 1);
        secp256k1_fe_add(&denom[j], &Pm_x);
        secp256k1_fe_normalize_var(&denom[j]);
        if (secp256k1_fe_is_zero(&denom[j])) {
            /* Pm = ±j*M*G — handle without batch inversion */
            if (verify_candidate(ctx, j*M, target33)) {
                *out_m = j*M; found = 1; break;
            }
            if (verify_candidate(ctx, (uint64_t)(-(int64_t)(j*M)), target33)) {
                *out_m = (uint64_t)(-(int64_t)(j*M)); found = 1; break;
            }
        }
    }
    if (found) { free(denom); return 1; }

    /* ── Phase 2: parallel binary TreeMon ── */
    secp256k1_fe* tree = treemon(denom, J, threads);
    free(denom);
    if (!tree) return 0;
    /* tree[J-1+j] = denom[j]^{-1} */

    /* ── Phase 3: parallel search ── */
    uint64_t chunk = (J+(uint64_t)threads-1)/(uint64_t)threads;
    pthread_t*   tids  = (pthread_t*)  malloc((size_t)threads*sizeof(pthread_t));
    search_args* sargs = (search_args*)malloc((size_t)threads*sizeof(search_args));
    atomic_int found_flag; atomic_init(&found_flag,0);

    for (int t=0;t<threads;t++) {
        sargs[t].baby       = baby;
        sargs[t].ctx        = ctx;
        sargs[t].T2         = T2;
        sargs[t].inv_den    = tree;
        sargs[t].n          = J;
        sargs[t].Pm_x       = Pm_x;
        sargs[t].Pm_y       = Pm_y;
        sargs[t].M          = M;
        sargs[t].j_start    = (uint64_t)t*chunk + 1;
        sargs[t].j_end      = (uint64_t)t*chunk + 1 + chunk < J
                              ? (uint64_t)t*chunk + 1 + chunk : J;
        sargs[t].target33   = target33;
        sargs[t].found_flag = &found_flag;
        sargs[t].result_m   = 0;
        sargs[t].found      = 0;
        pthread_create(&tids[t],NULL,search_worker,&sargs[t]);
    }
    for (int t=0;t<threads;t++) pthread_join(tids[t],NULL);

    int search_found=0;
    for (int t=0;t<threads;t++)
        if(sargs[t].found){*out_m=sargs[t].result_m;search_found=1;break;}

    free(tree); free(tids); free(sargs);
    return search_found;
}

/* ─────────────────────── benchmark ─────────────────────────────── */
static void benchmark(int bits, int l1, int trials, int threads) {
    int l2=bits-l1;
    uint64_t M=1ULL<<l1, J=1ULL<<l2;
    double t2_gb  = (double)(J*sizeof(secp256k1_ge))/(1ULL<<30);
    double inv_gb = (double)(J*2*sizeof(secp256k1_fe))/(1ULL<<30);

    printf("=== FastECDLP TreeMon (Tang et al., sequential Ph.1 + parallel binary tree Ph.2) ===\n");
    printf("Range   : m in [0, 2^%d)\n", bits);
    printf("Split   : l1=%d, l2=%d  (J=%"PRIu64")\n", l1, l2, J);
    printf("Threads : %d  (chunk=%"PRIu64")\n",
            threads,(J+(uint64_t)threads-1)/(uint64_t)threads);
    printf("T2 mem  : %.2f GB (one-time)\n", t2_gb);
    printf("Inv mem : %.2f GB (tree = 2J fe elements, per-solve)\n", inv_gb*1.5);
    printf("Trials  : %d\n\n", trials);

    secp256k1_context* ctx =
            secp256k1_context_create(SECP256K1_CONTEXT_SIGN|SECP256K1_CONTEXT_VERIFY);

    cuckoo_map baby; memset(&baby,0,sizeof(baby));
    if(!load_baby_table(&baby,l1)){secp256k1_context_destroy(ctx);return;}

    double t2_time;
    secp256k1_ge* T2=get_t2(ctx,M,l1,l2,&t2_time);
    if(!T2){free(baby.tab);secp256k1_context_destroy(ctx);return;}
    if(t2_time>0) printf("T2 build: %.1f sec (one-time, cached)\n\n",t2_time);
    else printf("\n");

    uint64_t mask=(bits==64)?~0ULL:((1ULL<<bits)-1ULL);
    int ok=0;
    double ts=now_seconds();

    for(int t=0;t<trials;t++){
        uint64_t m=((uint64_t)rand()<<32)^(uint64_t)rand();
        m&=mask; if(m==0)m=1;
        unsigned char sc[32]; u64_to_scalar32_be(m,sc);
        secp256k1_pubkey Pm_pk;
        secp256k1_ec_pubkey_create(ctx,&Pm_pk,sc);
        secp256k1_ge Pm_ge; pubkey_to_ge(&Pm_pk,&Pm_ge);
        unsigned char t33[33]; size_t tlen=33;
        secp256k1_ec_pubkey_serialize(ctx,t33,&tlen,&Pm_pk,SECP256K1_EC_COMPRESSED);

        uint64_t recovered=0;
        if(fastecdlp_solve(&baby,ctx,T2,&Pm_ge,M,l2,threads,t33,&recovered)
           &&recovered==m)
            ok++;
        else
            printf("Trial %d FAILED: m=%"PRIu64" recovered=%"PRIu64"\n",
                t,m,recovered);
    }

    double te=now_seconds();
    double total=te-ts;
    printf("Solved correctly : %d/%d\n",ok,trials);
    printf("Total time       : %.3f sec\n",total);
    printf("Average per solve: %.3f sec (%.2f ms)\n\n",
           total/trials,(total/trials)*1e3);
    printf("Note: T2 build (%.1f sec) excluded above.\n",t2_time);

    free(T2); free(baby.tab);
    secp256k1_context_destroy(ctx);
}

int main(int argc, char** argv){
    int bits=52,l1=30,trials=5,threads=10;
    if(argc>=2) bits   =atoi(argv[1]);
    if(argc>=3) l1     =atoi(argv[2]);
    if(argc>=4) trials =atoi(argv[3]);
    if(argc>=5) threads=atoi(argv[4]);
    srand((unsigned)time(NULL));
    benchmark(bits,l1,trials,threads);
    return 0;
}