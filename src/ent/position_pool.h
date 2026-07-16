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
// An optional trailing "\t<opp_leave>" carries the opponent's strategic leave
// (the tiles it kept last turn); the fill preserves those tiles instead of
// fully re-drawing the opponent. Absent -> empty leave -> caller full-swaps.
//
//   <cgp>
//   <id>\t<cgp>
//   <cgp>\t<opp_leave>
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

// Returns the opponent's strategic leave (tiles it kept last turn) for entry i,
// or "" when the pool line carried no leave field. Pointer owned by the pool.
const char *position_pool_get_opp_leave(const PositionPool *pp, int i);

// Opener-conditioned pool (MAGPIE_OPENER_POOL): for the opening bags, replay a
// specific opener move to synthesize a (bag, diff) board the natural pool never
// produces (blank-burn openers -> diff 0/2, rare scores -> 22). magpie sets the
// starting player's rack, movegen-matches (score, leave), plays that opener;
// sign '+' then passes the opponent so the opener is on turn (+diff), '-'
// leaves the responder on turn (-diff). The resulting bag = 93 - tiles_played
// falls out of the move and is recorded by the trajectory recorder's per-bag
// sharding, so one manifest can mix opener sizes (bags 91..86).
//
// Manifest line (tab-separated): bag \t rack \t score \t leave \t sign
// Env var: MAGPIE_OPENER_POOL=<manifest_path>

typedef struct OpenerPool OpenerPool;

OpenerPool *opener_pool_create(const char *path);
void opener_pool_destroy(OpenerPool *op);
int opener_pool_count(const OpenerPool *op);
const char *opener_pool_get_rack(const OpenerPool *op, int i);
const char *opener_pool_get_leave(const OpenerPool *op, int i);
int opener_pool_get_score(const OpenerPool *op, int i);
char opener_pool_get_sign(const OpenerPool *op, int i);

#endif
