#ifndef OPENING_PASS_H
#define OPENING_PASS_H

#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>

#include "letter_distribution.h"

// Opening-pass counterfactual training: for each candidate opening rack,
// run paired games (same seed, same starting player, same forced opening
// rack) where one branch forces a pass on the opening turn and the other
// plays normally. Records aggregated outcomes per (rack, branch) for a
// downstream lookup table: `rack -> (wp_pass, wp_play, paired_delta_se)`.
//
// Branch encoding: 0 = pass, 1 = play.
//
// Loaded via env var MAGPIE_OPENING_PASS_TABLE=path/to/candidates.txt
// (one rack per line, sorted, e.g., "AEINRST" or "AEINRS?").
// Output written to MAGPIE_OPENING_PASS_OUT=path/results.csv at end.

typedef struct OpeningPassTable OpeningPassTable;

// Loads the candidate rack list from a plain-text file (one rack per line).
// Returns NULL if path is unreadable or empty.
OpeningPassTable *opening_pass_table_create(const char *path,
                                            const LetterDistribution *ld);

// Loads previously-saved counters from a results CSV (in the format
// produced by opening_pass_table_dump). For each row whose rack matches
// a rack in the candidate table, the counters (wins/losses/ties for both
// branches, n_pairs, sum_diff, sum_sq_diff) are added to that rack's
// state. Rows for racks not in the current candidate list are silently
// ignored. Allows split / resume runs: the new run continues from the
// existing counts and only races toward the target N for racks still
// short. Returns the number of rows successfully merged.
int opening_pass_table_resume(OpeningPassTable *table, const char *csv_path);

void opening_pass_table_destroy(OpeningPassTable *table);

int opening_pass_table_num_racks(const OpeningPassTable *table);

int opening_pass_table_target_n(const OpeningPassTable *table);

// Given a paired game's pair_id, returns the rack string (NUL-terminated)
// to use as the opening rack for both games in the pair, and writes the
// corresponding rack index to *rack_idx. pair_id is typically the
// iter_count (each iter_count produces one pair of games).
const char *opening_pass_table_get_rack(const OpeningPassTable *table,
                                        uint64_t pair_id, int *rack_idx);

// Records a finished game's outcome into the per-rack aggregator.
// branch: 0 = pass branch, 1 = play branch.
// outcome: 0 = loss, 1 = tie, 2 = win (for the player with the forced rack).
//
// For paired-delta tracking: when both games of a pair have completed,
// the caller invokes opening_pass_record_pair() with both outcomes so
// the paired sum_diff and sum_sq_diff can be updated atomically.
void opening_pass_record(OpeningPassTable *table, int rack_idx, int branch,
                         int outcome);

// Records a completed pair: pass_outcome and play_outcome are both in
// {0, 1, 2} (loss/tie/win). Updates n_pairs, sum_diff, sum_sq_diff
// atomically. Caller is responsible for ensuring this is invoked exactly
// once per pair after both games finish.
//
// Encoding: outcome_int = 2 * outcome (so 0/2/4 represent L/T/W and the
// paired diff is in {-4, -3, ..., +4}). Diff = pass_outcome_int -
// play_outcome_int.
void opening_pass_record_pair(OpeningPassTable *table, int rack_idx,
                              int pass_outcome, int play_outcome);

// True when every (rack, branch) cell has reached the target N. Used by
// MAGPIE_OPENING_PASS_STOP_ON_COMPLETE to halt autoplay early.
bool opening_pass_table_is_complete(const OpeningPassTable *table);

// Total games still needed across all (rack, branch) cells to reach
// target N. For progress reporting.
int64_t opening_pass_table_remaining(const OpeningPassTable *table);

// Writes the aggregated results to a CSV at csv_path. One row per rack:
//   rack,n_pairs,
//   pass_wins,pass_losses,pass_ties,
//   play_wins,play_losses,play_ties,
//   sum_diff,sum_sq_diff
//
// From this, downstream Python computes:
//   wp_pass = pass_wins / n_pairs
//   wp_play = play_wins / n_pairs
//   paired_delta = sum_diff / (4 * n_pairs)   (factor 4 from outcome_int = 2x)
//   paired_var   = sum_sq_diff / (16 * n_pairs) - paired_delta^2
//   paired_se    = sqrt(paired_var / n_pairs)
void opening_pass_table_dump(const OpeningPassTable *table, const char *csv_path);

#endif
