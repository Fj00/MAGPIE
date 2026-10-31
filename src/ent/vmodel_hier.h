#ifndef VMODEL_HIER_H
#define VMODEL_HIER_H

// Hierarchical win% model (one per bag), the replacement for the bucketed
// V-model (vmodel.c). Exported from the B repo's hier_fit.py by
// hier_export_vmh.py; this file must reproduce hier_model.py's prediction
// exactly (checked by tools/vmh_check.c against Python on real moves).
//
//   z = alpha_s(d) + sum_f x_f * coef_f(row)
//   coef_f(row) = W[f] . Sel(row)
//   Sel = [ B(d) | 1[len] | 1[kind] | 1[type] | 1[blank] | 1[len] if exchange ]
//   alpha_s(d) = Bi(d) . (a0 + a_s[stratum])
//
// B / Bi are clamped cubic B-spline bases over the move's diff. x is the union
// feature vector (static leave value, kept-tile counts, pair counts and
// presence, unseen counts / bag_eq indicators or the pass's unseen block,
// exchanged-tile counts). Self-contained: libc + libm only.
//
// Tile encoding: 27-tile alphabet, index 0 = '?', 1 = 'A' .. 26 = 'Z'.

#include <stdbool.h>
#include <stdint.h>

#define VMH_NT 27
#define VMH_MAX_KNOTS 64
#define VMH_MAX_STRATA 128
#define VMH_N_LEN 8
#define VMH_N_KIND 3
#define VMH_N_TYPE 7      // all cons mixed vowel cons? mixed? vowel?
#define VMH_N_BLANK 2
#define VMH_N_XLEN 8
#define VMH_N_TAIL 2          // int_tails: intercept slopes below / above the knot range
#define VMH_TAIL_SCALE 100.0  // hier_model.TAIL_SCALE
#define VMH_SPREAD_SCALE 100.0  // hier_model.SPREAD_SCALE (spread models' target unit)

typedef struct VModelHier {
  int bag;
  int turn;                         // bag-93 turn model (1..6); 0 = a bag model
  int feat_k, int_k, J;
  bool g_len, g_kind, g_type, g_blank, g_xlen;
  bool l2_counts, blank_types, blank_mixed;
  bool count_steps;                 // kept-tile counts as 1/2/3+ steps (cstep_)
  bool count_cum;                   // cumulative >=1..>=4 steps (cge_), no XX pairs
  bool play_unseen_pairs;           // plays / exchanges get unseen pair counts too
  bool pass_sixpass;                // pass rows were fit on sixpass_diff
  bool int_tails;                   // a0 / a_s carry 2 linear tail slopes after the spline
  bool pass_shared_unseen;          // the pass uses the plays' unseen block, not its own
  bool target_spread;               // a spread model: predict returns points, not win%
  int unseen_mode;                  // 0 raw (ucnt/beq + pass block), 1 exp, 2 split
  int bexp_row[VMH_NT];             // expected draws per tile (draws x u_t / U)
  int brest_row[VMH_NT];            // split: not-to-you copies per tile (u_t - bexp_t)
  bool next_rack_vowels;            // nrv_: leave vowels + expected vowels drawn, curve
  int nrv_row[5];                   // quadratic B-spline bases 0..4 (2 = reference, -1)
  int useen_row[3];                 // shared: blanks, S, vowels unseen
  int pass_unseen;
  // stratum index by (kind, keep length, type); -1 = no curve (shared one only)
  int16_t sidx[VMH_N_KIND][VMH_N_LEN][VMH_N_TYPE];
  bool type_len[VMH_N_LEN];
  bool type_kind[VMH_N_KIND];
  int n_kf, n_ki;
  double knots_f[VMH_MAX_KNOTS], knots_i[VMH_MAX_KNOTS];
  double *a0;                       // int_k
  int n_strata;
  char strata_key[VMH_MAX_STRATA][24];
  double *a_s;                      // n_strata x int_k
  int n_rows;                       // feature rows stored
  double *w;                        // n_rows x J
  // feature -> row index (-1 = absent / zero weight)
  int static_row;
  int cnt_row[VMH_NT];
  int cstep_row[VMH_NT][4];         // [t][1..3]
  int cge_row[VMH_NT][5];           // [t][1..4]
  int pair_row[VMH_NT][VMH_NT];     // [i][j], i <= j
  int pairp_row[VMH_NT][VMH_NT];
  int ucnt_row[VMH_NT];
  int beq_row[VMH_NT][2];
  int exch_row[VMH_NT];
  int pcnt_row[VMH_NT];
  int upair_row[VMH_NT][VMH_NT];
} VModelHier;

// Load a .vmh file; NULL (and a message on stderr) on any parse error.
VModelHier *vmodel_hier_create(const char *path);
void vmodel_hier_destroy(VModelHier *m);

// Win probability for one move.
//   rack[27], leave[27]: tile counts (leave = rack for a pass, rack - played
//     tiles for a play, the kept tiles for an exchange)
//   kind: 0 pass, 1 exchange, 2 play
//   diff: the model's diff axis -- lead after the move from the mover's view
//     for plays / exchanges (bag 93: the opening score), the six-pass tally
//     for the pass (as the training strata; the caller converts)
//   unseen[27]: the mover's unseen pool (bag + opponent rack); at bag 93 the
//     full bag minus the rack
//   static_leave: the static leave value of the kept tiles (caller's table;
//     0 for an empty leave), as hier_model's static_leave feature
double vmodel_hier_predict(const VModelHier *m, const uint8_t rack[VMH_NT],
                           const uint8_t leave[VMH_NT], int kind, int diff,
                           const int unseen[VMH_NT], float static_leave);

// The type index (0..6) hier_model.leave_type would give this leave.
int vmodel_hier_leave_type(const VModelHier *m, const uint8_t leave[VMH_NT],
                           int kind);

// sixpass_diff (v_model_features.sixpass_diff): lead - (mover's rack points) +
// (points of the RACK_SIZE cheapest unseen tiles). The pass's diff axis when the
// model was fit on it (m->pass_sixpass); otherwise the pass uses the plain lead.
int vmodel_hier_sixpass_diff(const uint8_t rack[VMH_NT], const int unseen[VMH_NT],
                             int lead);

// ---------------------------------------------------------------- turn cache
// Same answer as vmodel_hier_predict, much less work in a picker: the unseen
// block (and the pass's block) is summed once per turn, each distinct leave's
// features once per turn (a rack has at most 127 leaves), and every move then
// costs one dot product of <= 9 selector entries.
typedef float (*VmhStaticFn)(const uint8_t *leave_idx, int leave_len, void *ctx);

typedef struct VmhTurnCache VmhTurnCache;
VmhTurnCache *vmh_turn_cache_create(void);
void vmh_turn_cache_destroy(VmhTurnCache *c);
// rack_idx: the mover's rack as sorted tile indices; unseen: the mover's pool.
// static_fn(ctx) gives a leave's static value (called once per distinct leave).
void vmh_turn_begin(VmhTurnCache *c, const VModelHier *m, const uint8_t *rack_idx,
                    int rack_len, const int unseen[VMH_NT], VmhStaticFn static_fn,
                    void *static_ctx);
// leave_idx: sorted tile indices of the kept tiles; diff as vmodel_hier_predict
// (the caller converts a pass to sixpass_diff when m->pass_sixpass).
double vmh_turn_predict(VmhTurnCache *c, const uint8_t *leave_idx, int leave_len,
                        int kind, int diff);

#endif
