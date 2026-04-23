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
} ForceTarget;

typedef struct ForceTable ForceTable;

ForceTable *force_table_create(const char *csv_path,
                               const LetterDistribution *ld);

void force_table_destroy(ForceTable *table);

// Returns the array of targets at the given bag count, and sets *count.
// Returned pointer is owned by the table and valid until force_table_destroy.
// NULL is returned and *count set to 0 if no targets exist for the bag.
ForceTarget **force_table_lookup(ForceTable *table, int bag, int *count);

// Check whether a candidate move's leave + score matches the given target.
// `leave` is the post-move kept-tile rack. `score` is the move's score.
// Blanks are treated per the aggregation spec (ignored for type
// classification, but counted as a distinct tile '?' for subleave matching).
bool force_target_matches(const ForceTarget *target, const Rack *leave,
                          int score);

// Decrement a target's deficit. Returns true if the target is now satisfied.
bool force_target_decrement(ForceTarget *target);

// Total remaining deficit across all targets (for progress reporting).
int64_t force_table_total_remaining(const ForceTable *table);

// Number of targets currently loaded (including satisfied ones).
int force_table_num_targets(const ForceTable *table);

// Classify a leave as all/cons/mixed/vowel per the aggregation rule.
// Blanks (machine letter 0) ignored for the v/c count; leaves of length <= 2
// always return LEAVE_TYPE_ALL.
LeaveType force_classify_leave(const Rack *leave,
                               const LetterDistribution *ld);

#endif
