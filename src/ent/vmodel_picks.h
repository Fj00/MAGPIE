#ifndef VMODEL_PICKS_H
#define VMODEL_PICKS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "move.h"

// Precomputed V-model picks: rack -> chosen Move lookup table loaded
// from a .picks binary file (written by
// bots/winpct/scripts/precompute_vmodel_picks.py).
//
// At every bag=93 board-empty moment in EB-cycle self-play, the
// V-model's chosen move is fully determined by the player's rack.
// Rather than re-running movegen + inference every call, magpie loads
// a one-time precomputed table at startup and does a binary-search
// lookup (sorted-array form) per call.
//
// File format (little-endian, matching the .picks writer):
//   Header (32 bytes):
//     uint32 magic = 0x504B5631 ("PKV1")
//     uint32 version
//     uint32 model_turn        (1..6, dispatch slot)
//     uint32 n_entries
//     uint32 entry_size        (8 + 2 + 1 + max_repr_len = 35 by default)
//     uint32 max_repr_len      (24 by default)
//     uint64 reserved
//   Entries (n_entries * entry_size, sorted by rack_canonical):
//     uint8[8]   rack_canonical  (blanks-FIRST then sorted alpha,
//                                 NUL-padded; matches magpie's
//                                 BLANK_MACHINE_LETTER=0 ordering)
//     int16      score
//     uint8      repr_len
//     uint8[max_repr_len] action_repr  ("pass", "8H FARM", "(exch ABC)")

#define VMODEL_PICKS_MAGIC   0x504B5631U
#define VMODEL_PICKS_VERSION 1
#define VMODEL_PICKS_RACK_KEY_LEN 8

typedef struct VModelPicks {
  int  model_turn;        // 1..6, dispatched into vmodel_picks[turn]
  int  n_entries;
  // Parallel arrays — keys[i] keys move at moves[i]. Both sorted
  // ascending by rack_canonical key.
  uint8_t (*keys)[VMODEL_PICKS_RACK_KEY_LEN];  // [n_entries]
  Move    *moves;                              // [n_entries]
} VModelPicks;

// Load a .picks file. Parses every entry's action_repr into a Move
// struct (one-time at startup; ~few seconds for 3.2M entries).
// Returns NULL on file/format error (logged via log_warn).
VModelPicks *vmodel_picks_create(const char *picks_path);

void vmodel_picks_destroy(VModelPicks *picks);

// Build the 8-byte canonical-rack key for a given Rack. The key is
// blanks-FIRST then letters in machine-letter order (ML 1 = A,
// ML 2 = B, ..., ML 26 = Z), NUL-padded if total tiles < 8.
//
// Convention matches both the .picks file writer (Python
// canonicalize_leave) and magpie's internal BLANK_MACHINE_LETTER=0.
// Returns the number of bytes filled (= total tiles on the rack).
// `out_buf` must have at least VMODEL_PICKS_RACK_KEY_LEN bytes.
int vmodel_picks_key_for_rack(const struct Rack *rack, uint8_t *out_buf);

// Lookup the chosen Move for a given rack key. Returns NULL on miss.
// `rack_key` should be built via `vmodel_picks_key_for_rack`.
const Move *vmodel_picks_lookup(const VModelPicks *picks,
                                 const uint8_t *rack_key);

#endif
