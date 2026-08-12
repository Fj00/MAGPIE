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
                             int play_unseen,
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

#endif
