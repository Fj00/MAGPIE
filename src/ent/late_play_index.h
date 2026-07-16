#ifndef LATE_PLAY_INDEX_H
#define LATE_PLAY_INDEX_H

#include <stdbool.h>
#include <stdint.h>

#include "force_table.h"

// Late-game (bags 8-14) play index: for each force-table cell, the set of
// (position, rack, bag-emptying play) tuples that produce it. Built offline by
// build_late_play_index.py from magpie's MAGPIE_PP_INDEX_BUILD movegen-only
// shards. The board-dependent analog of the opener's play_index (src/ent/
// play_index.c), which is 1-D (empty board -> rack only).
//
// Cells are resolved to the force table BY ORDINAL: cell_id == the play index's
// row == force_targets.csv row order == force_table_target_index. So no by-key
// resolution is needed (unlike the opener), which also sidesteps the 2-ML
// cells.bin limit for the L3/L4 4-tile "leave" cells.
//
// It drives the fill's supply straight through: sample a still-deficient cell
// (deficit-weighted) -> a play covering it -> return that play's (position,
// rack). The caller loads the board + injects the rack and runs the existing
// fanout+solve+gate, which fills the deficient cell(s) that (position, rack)
// produces. As the force table drains, satisfied cells stop being sampled;
// when all deficit is gone the sampler returns false (clean terminus).

typedef struct LatePlayIndex LatePlayIndex;

// Load the index at dir_path and resolve its cells to `ft` by ordinal. Requires
// num_cells == force_table_num_targets(ft). Returns NULL on failure (logs why).
LatePlayIndex *late_play_index_create(const char *dir_path, ForceTable *ft);

void late_play_index_destroy(LatePlayIndex *idx);

// Deficit-weighted pick of a still-deficient cell, then a uniform-random play
// covering it. On success writes:
//   *out_pool_idx : the play's position (position pool array index; feed to
//                   position_pool_get_cgp / game_load_cgp)
//   *out_rack     : the play's injected rack, a NUL-terminated pointer into the
//                   mmap (valid for the index lifetime; do not free)
// Returns false when no cell has deficit>0 (all drained) -> caller terminates.
bool late_play_index_sample(const LatePlayIndex *idx, uint64_t seed,
                            uint32_t *out_pool_idx, const char **out_rack);

#endif
