#ifndef VMODEL_FEATURES_H
#define VMODEL_FEATURES_H

#include <stdbool.h>
#include <stdint.h>

#include "static_leaves.h"
#include "vmodel.h"

// V-model feature extraction. Mirrors
// `bots/winpct/scripts/v_model_features.py:extract_features` exactly so
// the trained coefficients line up.
//
// Input rack/leave are python-canonical tile indices (0='?'..26='Z').
// `feature_buf` must be at least `bucket->n_coefs` floats.

// Compute the feature vector for one record. Returns the number of
// features written (== bucket->n_coefs on success), or -1 on schema
// mismatch (a hard bug — log the stratum/bucket and abort the caller).
// `unseen_vec`: optional 27-vector of unseen-tile counts (physical bag +
// opponent rack) from the on-turn player's perspective. Required at bag<93
// (endgame): the opener derivation `TILE_BAG - own_rack` is only correct when
// the board is empty. Pass NULL at bag==93 to keep the opener derivation.
int vmodel_extract_features(float *feature_buf, int buf_capacity,
                             const VStratum *s, const VBucket *b,
                             const uint8_t *rack_indices, int rack_len,
                             const uint8_t *leave_indices, int leave_len,
                             int diff, int bag, int turn,
                             const int *unseen_vec,
                             int play_unseen, int pass_unseen,
                             const StaticLeaves *sl);

// Convenience: extract features, dot-product with coefs + intercept,
// return sigmoid. Returns -1.0f on schema mismatch. `unseen_vec` as in
// vmodel_extract_features (NULL at bag==93; required for endgame bag<93).
float vmodel_predict(const VModel *m,
                      const uint8_t *rack_indices, int rack_len,
                      const uint8_t *leave_indices, int leave_len,
                      int kind, int diff, int turn,
                      const int *unseen_vec,
                      const StaticLeaves *sl);

// Per-turn prediction cache: same answer as vmodel_predict, far less work.
//
// Within one turn (one rack, one unseen pool), a play's prediction depends on
// only two things: its LEAVE and its SCORE DIFF. Everything else in the feature
// vector -- the static leave value, leave counts and pairs, the unseen/bag
// block, exchange counts -- is a function of the leave or of the turn alone.
// And diff enters the model linearly, as feature 0 of a has_diff bucket, ahead
// of the sigmoid. So for each (kind, leave, bucket) the dot product over every
// feature but diff is computed ONCE and reused:
//
//     z = [intercept + sum_{i>=1} f_i c_i]   (cached)   +   diff * c_0
//
// A rack yields at most 127 leaves but can yield 20,000 moves; vmodel_predict
// rebuilt the full ~285-feature vector (incl. a binary search into the static
// leave table and an snprintf+strcmp stratum lookup) for every one of them.
// Profiling bag-40 LABEL put 86.5% of all CPU in the playout picker on 2026-10-11.
//
// Not bit-identical: the cached sum adds the diff term last instead of first,
// so double-precision rounding can differ at the ~1e-16 level. That moves a
// 4-decimal-rounded win% only when it sits on a rounding knife-edge.
// MAGPIE_VMODEL_CACHE_VERIFY=1 computes both and counts every disagreement.
//
// Falls back to vmodel_predict for passes (six-pass diff transform), terminal
// zero-score models (turn 6), and strata with more buckets than the cache holds.
typedef struct VModelTurnCache VModelTurnCache;

// Thread-local cache, allocated on first use.
VModelTurnCache *vmodel_turn_cache_get(void);

// Start a turn: invalidates every cached entry in O(1).
void vmodel_turn_cache_begin(VModelTurnCache *c, const VModel *m,
                             const uint8_t *rack_indices, int rack_len,
                             const int *unseen_vec, const StaticLeaves *sl);

float vmodel_turn_cache_predict(VModelTurnCache *c,
                                const uint8_t *leave_indices, int leave_len,
                                int kind, int diff);

// Under MAGPIE_VMODEL_CACHE_VERIFY=1: cache misses scored by both the sparse
// and the dense path, and how many of those z0 values were not bit-identical.
extern _Atomic uint64_t g_vmtc_sparse_checks;
extern _Atomic uint64_t g_vmtc_sparse_mismatch;

#endif
