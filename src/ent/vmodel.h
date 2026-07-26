#ifndef VMODEL_H
#define VMODEL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "../def/letter_distribution_defs.h"
#include "letter_distribution.h"

// V-model inference engine — load a trained V-model from the flat-text
// `.vmt` format (export_v_model_text.py) and predict win% per candidate
// move. CSR-style coefficient storage: 20 strata × ~94 buckets total,
// ~800 KB on-heap. Lookup is O(log n_buckets) per move plus a small
// dot-product.

#define VMODEL_MAX_TILES 27   // '?' + A-Z
#define VMODEL_KEY_MAX   32   // stratum_key, e.g. "K2_L3_mixed"

typedef enum {
    VMODEL_TYPE_ALL   = 0,
    VMODEL_TYPE_CONS  = 1,
    VMODEL_TYPE_MIXED = 2,
    VMODEL_TYPE_VOWEL = 3,
} VModelLeaveType;

// One per (stratum, bucket). All optional indicator arrays are NULL+0
// when absent. coefs is always present and indexes match
// vmodel_features.c's extract_features() ordering.
typedef struct VBucket {
    int     diff_lo;
    int     diff_hi;
    bool    has_diff;
    float   intercept;
    int     n_coefs;
    float  *coefs;          // [n_coefs]
    int     n_pts;
    int    *pts;            // [n_pts]  optional pts_indicators
    int     n_drawn;
    double *drawn;          // [n_drawn]  optional drawn_indicators (double:
                            // Python rounds to 4dp and compares exact-equal;
                            // float32 would round-trip-lose precision)
    int     n_bag;
    uint8_t *bag_tiles;     // [n_bag]  python-canonical tile index (0='?'..26='Z')
    int8_t  *bag_counts;    // [n_bag]
} VBucket;

typedef struct VStratum {
    char    key[VMODEL_KEY_MAX];   // e.g. "K2_L3_mixed"
    int     kind;                  // 0=pass, 1=exchange, 2=play
    int     leave_length;
    int     type;                  // VModelLeaveType
    int     n_tiles;
    uint8_t tiles[VMODEL_MAX_TILES];  // python-canonical indices (0='?'..26='Z')
    int     n_pairs;
    uint8_t (*pairs)[2];           // [n_pairs] each is two python-canonical indices
    int     n_buckets;
    VBucket *buckets;              // [n_buckets]
} VStratum;

typedef struct VModel {
    int       version;
    int       turn;
    int       bag;
    int       n_strata;
    // PASS (K0_L7) bucketed on the six-pass endgame tally: when true,
    // vmodel_predict converts the incoming raw lead into sixpass_diff before the
    // stratum/bucket lookup and the move_score feature. Set from the .vmt PASS6
    // header flag (export_v_model_text.py); false for older lead-bucketed models.
    bool      pass_sixpass;
    VStratum *strata;              // [n_strata]
} VModel;

// Load a V-model from a `.vmt` text file. Returns NULL on parse error
// (already logged via log_warn). Caller owns the returned VModel and
// must free it via vmodel_destroy().
VModel *vmodel_create(const char *vmt_path);

void vmodel_destroy(VModel *m);

// Classify a leave as one of the four type buckets. Y is treated as a
// vowel (matches winpct/scripts convention). Blanks are ignored.
// `leave_chars` is the python-canonical tile indices (0='?'..26='Z').
VModelLeaveType vmodel_classify_leave(const uint8_t *leave_indices,
                                       int leave_len);

// True iff (turn, kind) is a terminal-zero-score state (T6 + kind∈{0,1}).
// Drives the K1_L3..6 → "all" stratum collapse.
bool vmodel_terminal_zero_score(int turn, int kind);

// Lookup the stratum index for (kind, leave_length, leave_type) at this
// turn. Applies the terminal-zero collapse rule. Returns -1 if no such
// stratum is present in the model.
int vmodel_stratum_index(const VModel *m, int kind, int leave_length,
                          VModelLeaveType type, int turn);

// Lookup the bucket index whose [diff_lo, diff_hi] contains `diff`.
// Returns -1 if no bucket matches.
int vmodel_bucket_index(const VStratum *s, int diff);

// Convert one ASCII tile char ('?' or 'A'..'Z') to the python-canonical
// tile index (0='?', 1='A', ..., 26='Z'). Returns -1 for invalid input.
int vmodel_tile_index_from_char(char c);

#endif
