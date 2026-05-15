#ifndef OUTCOME_PRIORS_H
#define OUTCOME_PRIORS_H

#include <stdbool.h>
#include <stdint.h>

#include "play_index.h"

// Outcome priors keyed by the option-F tuple
//   (leave_len, n_blanks, n_S, n_J, n_Q, n_X, n_Z, n_vowels,
//    leave_pts_residual, score)
// Loaded from a packed binary produced by build_t6_baseline.py
// (priors_by_action_class.bin). Each entry stores the natural-bag-
// weighted P(W), P(L), P(T) for plays whose post-play leave + score
// match the key. Used by Phase 3 scheduler to weight rack/action
// selection by likely W vs L outcome alignment with per-cell deficit.

typedef struct {
  uint8_t  leave_len;
  uint8_t  n_blanks;
  uint8_t  n_S;
  uint8_t  n_J;
  uint8_t  n_Q;
  uint8_t  n_X;
  uint8_t  n_Z;
  uint8_t  n_vowels;
  int16_t  leave_pts_residual;
  int16_t  score;
} OptionFKey;

typedef struct OutcomePriors OutcomePriors;

// Load packed binary at `path`. Returns NULL on missing file or
// magic/version mismatch.
OutcomePriors *outcome_priors_load(const char *path);

void outcome_priors_destroy(OutcomePriors *op);

int outcome_priors_num_buckets(const OutcomePriors *op);

// Compute the option-F key for a play_index record. `leave_str` in
// the PlayRecord is ASCII (e.g. "RY?"). Returns true on success.
bool outcome_priors_compute_key(const PlayRecord *pr, OptionFKey *out);

// Look up the prior for a key. Writes p_W, p_L, p_T into out_p* and
// returns true on hit. Returns false on miss; out_p* untouched.
bool outcome_priors_lookup(const OutcomePriors *op, const OptionFKey *key,
                            float *out_pw, float *out_pl, float *out_pt);

#endif
