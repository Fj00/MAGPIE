#ifndef OPENER_LOOKUP_H
#define OPENER_LOOKUP_H

#include <stdbool.h>
#include <stdint.h>

#include "../def/board_defs.h"
#include "letter_distribution.h"
#include "rack.h"

// Per-play record stored in opener_lookup.bin. Layout matches the binary
// format produced by /Users/eric/B/bots/winpct/scripts/compute_lookup_table.py
// (struct.pack `<BBhB8sIB15sBBBB`, 37 bytes).
//
// IMPORTANT: structure is packed; field order matches the on-disk layout so
// the mmap'd region can be cast directly to (const OpenerPlay *).
typedef struct __attribute__((packed)) {
  uint8_t action_type;     // 0=play, 1=exchange
  uint8_t _flags;
  int16_t score;
  uint8_t leave_len;
  uint8_t leave_mls[8];    // sorted ascending; 0 = blank, 1..26 = A..Z
  uint32_t equity_q_bits;  // reinterpret as int32 (microequity)
  uint8_t word_len;
  uint8_t word_mls[15];    // high bit = blank-as-letter
  uint8_t row_start;
  uint8_t col_start;
  uint8_t dir;             // 0=horizontal, 1=vertical
  uint8_t _pad;
} OpenerPlay;

typedef struct OpenerLookup OpenerLookup;

// Create from a binary file produced by compute_lookup_table.py. Validates
// magic + lex_hash; returns NULL on mismatch (callers should fall back to
// regular movegen). On Linux, mmap'd with MAP_POPULATE to pre-fault pages.
OpenerLookup *opener_lookup_create(const char *path,
                                   const LetterDistribution *ld);

void opener_lookup_destroy(OpenerLookup *ol);

// Look up a 7-tile rack by its 28-byte canonical key (sorted ASCII letters,
// then blanks, NUL-padded). Returns base + count of plays in equity-DESC
// order, or NULL/0 if the rack is not in the table.
const OpenerPlay *opener_lookup_query(const OpenerLookup *ol,
                                      const char *canon_rack,
                                      int *out_count);

// Build a canonical 28-byte rack key from a Rack. The buffer must be 28
// bytes; the caller is responsible for ensuring that.
void opener_lookup_canonical_rack(const Rack *rack,
                                  const LetterDistribution *ld,
                                  char canon_out[28]);

#endif
