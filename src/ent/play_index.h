#ifndef PLAY_INDEX_H
#define PLAY_INDEX_H

#include <stdbool.h>
#include <stdint.h>

#include "force_table.h"
#include "letter_distribution.h"

// Cell-driven (rack, play) index for T6 force-table sampling. Built
// offline by build_t6_play_index.py from opener_all_moves.shards
// against a force_targets.csv. Loaded as mmap'd binary files at
// magpie startup.
//
// The index supports two queries used at the T6 recording turn:
//
//   1) play_index_sample_rack_deficit_aware(): pick a rack whose
//      best-covering play maximizes the sum of remaining deficit
//      across all cells the play credits. Replaces the 5-pool
//      uniform-random rack sampler.
//
//   2) play_index_pick_targeted_plays(): given an injected rack,
//      return the top-N play_ids whose covered cells maximize
//      remaining deficit. Used to augment the T6 fanout slot list
//      with deficit-targeted plays beyond the bot's natural choice.
//
// Cells are resolved to ForceTarget* once at load time via
// force_table_lookup_target_by_key. Per-target deficit is read
// lock-free at sample time via atomic_load on the existing
// ForceTarget._Atomic deficit field.

typedef struct PlayIndex PlayIndex;

PlayIndex *play_index_create(const char *dir_path, ForceTable *force_table,
                             const LetterDistribution *ld);

void play_index_destroy(PlayIndex *idx);

int play_index_num_racks(const PlayIndex *idx);
int play_index_num_plays(const PlayIndex *idx);
int play_index_num_cells(const PlayIndex *idx);

// Pick a rack via top-K cell-driven scoring. Returns NULL if no cell
// has any remaining deficit or no covered play exists. When non-NULL,
// also writes the picked rack's rack_id to *out_rack_id so the caller
// can pass it to play_index_pick_targeted_plays in the same call site.
const char *play_index_sample_rack_deficit_aware(const PlayIndex *idx,
                                                  uint64_t seed,
                                                  uint32_t *out_rack_id);

// Pick up to `max_n` distinct play_ids for the given rack_id, ranked
// by deficit-coverage rarity (Σ deficit[c] / supply[c]). Only plays
// scoring above `threshold` are included. Writes play_ids into `out`.
// Returns the number written (<= max_n).
int play_index_pick_targeted_plays(const PlayIndex *idx,
                                   uint32_t rack_id, double threshold,
                                   int max_n, uint32_t *out_play_ids);

// Resolve a rack string to its rack_id (linear scan; for the rare case
// of needing the rack_id from outside the sample path).
int play_index_lookup_rack_id(const PlayIndex *idx, const char *rack_str);

// Read a play record by play_id. Out parameters are pointers into the
// mmap'd plays.bin (do not free). action_repr is NUL-terminated in the
// arena; the caller may strdup if needed across atomic writes.
typedef struct {
  uint32_t rack_id;
  uint8_t  action_kind;        // 0=pass 1=exch 2=play
  int16_t  score;
  uint8_t  ar_len;
  uint16_t n_cells;
  uint8_t  leave_len;
  const char *action_repr;
  const char *leave_str;       // ≤ 7 chars, not NUL-terminated
  const uint32_t *cell_ids;
} PlayRecord;

bool play_index_get_play(const PlayIndex *idx, uint32_t play_id,
                          PlayRecord *out);

#endif
