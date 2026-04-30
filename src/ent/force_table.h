#ifndef FORCE_TABLE_H
#define FORCE_TABLE_H

#include <stdbool.h>
#include <stdint.h>

#include "../def/board_defs.h"
#include "../def/letter_distribution_defs.h"
#include "letter_distribution.h"
#include "rack.h"

typedef enum {
  FORCE_TARGET_STRATUM = 0,
  FORCE_TARGET_TILE = 1,
  FORCE_TARGET_PAIR = 2,
} ForceTargetKind;

typedef enum {
  LEAVE_TYPE_ALL = 0,
  LEAVE_TYPE_CONS = 1,
  LEAVE_TYPE_MIXED = 2,
  LEAVE_TYPE_VOWEL = 3,
} LeaveType;

typedef struct ForceTarget {
  ForceTargetKind kind;
  int bag;
  int leave_length;
  LeaveType leave_type;
  int exchange;
  MachineLetter subleave_mls[2];
  int subleave_count;
  int64_t deficit;
  // Diff range for this target. Used to constrain a target to a specific
  // adaptive diff-bucket within a stratum (per the 10×10×10 spec — each
  // bucket fits one of the 10 averaged models, and each model needs ≥EPV
  // observations of each feature). Inclusive bounds.
  // INT_MIN/INT_MAX = no constraint (backward-compatible with 10-column CSVs).
  int diff_min;
  int diff_max;
} ForceTarget;

typedef struct ForceTable ForceTable;

ForceTable *force_table_create(const char *csv_path,
                               const LetterDistribution *ld);

void force_table_destroy(ForceTable *table);

// Returns the array of targets at the given bag count, and sets *count.
// Returned pointer is owned by the table and valid until force_table_destroy.
// NULL is returned and *count set to 0 if no targets exist for the bag.
ForceTarget **force_table_lookup(ForceTable *table, int bag, int *count);

// A2: lookup targets at a specific (bag, leave_length, exchange) shape. Lets
// per-move matching skip iteration over targets whose shape can't match the
// move's leave. Same target pointers as force_table_lookup, just bucketed.
ForceTarget **force_table_lookup_by_shape(ForceTable *table, int bag,
                                          int leave_length, int exchange,
                                          int *count);

// Check whether a candidate move's leave + score + current score-diff matches
// the given target. `leave` is the post-move kept-tile rack. `score` is the
// move's score. `diff` is the current score difference (player_on_turn minus
// opp) at the time of the candidate move; checked against target's
// [diff_min, diff_max]. Blanks are treated per the aggregation spec (ignored
// for type classification, but counted as a distinct tile '?' for subleave
// matching).
bool force_target_matches(const ForceTarget *target, const Rack *leave,
                          int score, int diff);

// Decrement a target's deficit. If the deficit transitions from 1 to 0,
// the table's active-target counter is decremented. Returns true if the
// target is now satisfied.
bool force_table_decrement_target(ForceTable *table, ForceTarget *target);

// Credit a finished game's outcome against a target hit during the game.
// For stratum-kind targets: looks up the per-(target, diff) tally and
// decrements the target's deficit only when the outcome would increase
// min(wins, losses) for that diff bucket. The tally is always updated.
// Race-free across threads via atomic fetch-add of a packed (wins:32,
// losses:32) bucket — each game's pre-increment value uniquely determines
// whether IT was the one that bumped min_wl.
//
// For tile/pair-kind targets: decrements deficit by 1 (their semantics
// are count-based, not min(w,l)-based — original behavior preserved).
//
// `diff` is the score difference at the force turn from the forcing
// player's perspective (same convention as the FJ recorder's score_diff).
// `is_win`/`is_tie` describe the game's final outcome for that player;
// `is_tie` takes precedence over `is_win`.
void force_table_credit_game(ForceTable *table, ForceTarget *target,
                             int diff, bool is_win, bool is_tie);

// True once every target has been satisfied (all deficits are zero). Safe
// to call from multiple threads; the underlying counter is atomic.
bool force_table_is_exhausted(const ForceTable *table);

// Total remaining deficit across all targets (for progress reporting).
int64_t force_table_total_remaining(const ForceTable *table);

// Dump the remaining deficits to a CSV with the same schema as the input,
// skipping targets whose deficit is 0. Used for progress reporting and to
// produce an updated force_targets.csv for subsequent runs.
void force_table_dump_remaining(const ForceTable *table, const char *csv_path,
                                const LetterDistribution *ld);

// Number of targets currently loaded (including satisfied ones).
int force_table_num_targets(const ForceTable *table);

// Classify a leave as all/cons/mixed/vowel per the aggregation rule.
// Blanks (machine letter 0) ignored for the v/c count; leaves of length <= 2
// always return LEAVE_TYPE_ALL.
LeaveType force_classify_leave(const Rack *leave,
                               const LetterDistribution *ld);

#endif
