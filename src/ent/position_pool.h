#ifndef POSITION_POOL_H
#define POSITION_POOL_H

// Position pool for the 91-8 rack-injection self-play (MAGPIE_POSITION_POOL).
//
// Instead of starting fresh games, the position-pool mode loads mid-game
// board states sampled from the natural trajectory recorder and injects a
// (deficit-aware) rack + forced move at each, recording the counterfactual
// move's outcome via the existing trajectory recorder format. See
// project_winpct_diff_bucketing memory + the noble-popping-kitten plan.
//
// Manifest format: one complete on-turn-first CGP per line, e.g. the
// board_cgp column of trajectory positions/bag_<NN>.csv. Blank lines and
// lines beginning with '#' are skipped. A leading "<id>\t" is optional;
// if present it is parsed as the position id, else the line index is used.
//
//   <cgp>
//   <id>\t<cgp>
//
// Env var: MAGPIE_POSITION_POOL=<manifest_path>

#include <stdint.h>

typedef struct PositionPool PositionPool;

PositionPool *position_pool_create(const char *path);
void position_pool_destroy(PositionPool *pp);

int position_pool_count(const PositionPool *pp);

// Returns the CGP string for entry i (0 <= i < count), or NULL if out of
// range. The returned pointer is owned by the pool.
const char *position_pool_get_cgp(const PositionPool *pp, int i);

// Returns the position id for entry i (the parsed leading id, or i itself
// when no id column was present).
uint64_t position_pool_get_id(const PositionPool *pp, int i);

#endif
