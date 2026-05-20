#ifndef STATIC_LEAVES_H
#define STATIC_LEAVES_H

#include <stdbool.h>
#include <stdint.h>

// Static-leaves table for the V-model `static_leave` feature. Loaded
// from one or two CSV files matching v_model_features.py:load_static_leaves:
//
//   CSW24_gen_6.csv          `<leave>,<value>`              (L1..L6 leaves)
//   CSW24_7tile_gen_6.csv    `<rack>,<count>,<value>,...`   (L7 racks)
//
// Internal storage is a sorted array of (canonical_leave, value) pairs
// keyed by the canonical-form leave string (blanks first then letters
// in alpha order, e.g. "??AE"). Lookup is binary search on the packed
// key.

// Key packing: 7 bytes for sorted canonical-leave tile indices
// (python-canonical, 0='?'..26='Z'), zero-padded. Lookup callers must
// pre-canonicalize.
#define STATIC_LEAVES_MAX_LEN 7

typedef struct StaticLeavesEntry {
    uint8_t key[STATIC_LEAVES_MAX_LEN]; // canonical, zero-padded
    uint8_t key_len;
    float   value;
} StaticLeavesEntry;

typedef struct StaticLeaves {
    int                n_entries;
    StaticLeavesEntry *entries;        // sorted by (key_len, key bytes)
} StaticLeaves;

// Load static-leaves from one or two CSVs. Either may be NULL; both NULL
// returns a table containing only the empty leave (value 0). Returns
// NULL on hard error (alloc fail).
StaticLeaves *static_leaves_create(const char *short_csv_path,
                                    const char *full_csv_path);

void static_leaves_destroy(StaticLeaves *sl);

// Look up the value for the given canonical-form indices (sorted,
// blanks first). Returns 0.0f if not found.
float static_leaves_lookup(const StaticLeaves *sl,
                            const uint8_t *canonical_indices, int len);

// Canonicalize a leave (indices 0..26 in python-canonical encoding).
// Sorts blanks first then alpha. Writes up to `len` bytes to `out_buf`
// which must be at least STATIC_LEAVES_MAX_LEN long.
void static_leaves_canonicalize(const uint8_t *indices, int len,
                                 uint8_t *out_buf);

#endif
