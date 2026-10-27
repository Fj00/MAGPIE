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

typedef struct VModelHier {
  int bag;
  int feat_k, int_k, J;
  bool g_len, g_kind, g_type, g_blank, g_xlen;
  bool l2_counts, blank_types, blank_mixed;
  int pass_unseen;
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

#endif
