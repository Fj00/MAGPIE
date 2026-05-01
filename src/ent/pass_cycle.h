#ifndef PASS_CYCLE_H
#define PASS_CYCLE_H

#include <stdbool.h>
#include <stdint.h>

// Pass-cycle counterfactual training. Both players draw from a combined pool
// of pass-favorable and exchange-prone racks. P1 is sampled by P(rack) weight;
// P2 is sampled from the subset drawable after P1's tiles are removed from the
// bag. Branch 0 = P1 passes every turn; branch 1 = P1 plays normally.
//
// Env vars:
//   MAGPIE_PASS_CYCLE_RACKS=path  (CSV: rack,weight — combined pool)
//   MAGPIE_PASS_CYCLE_OUT=path    (output CSV)

typedef struct PassCycleTable PassCycleTable;

// Returns NULL on any failure.
PassCycleTable *pass_cycle_table_create(const char *pool_path,
                                        const char *out_path);

void pass_cycle_table_destroy(PassCycleTable *table);

// Sample P1 and P2 racks for pair_id. P1 is drawn from the full pool;
// P2 is drawn from the subset still drawable after P1's tiles are removed.
// Returns false only if no drawable P2 rack exists (should not happen in
// practice with a standard Scrabble bag).
bool pass_cycle_sample_racks(const PassCycleTable *table, uint64_t pair_id,
                             const char **p1_rack_out,
                             const char **p2_rack_out);

// Sample only P1's rack from the weighted pool (used by mix-random mode
// where P2 is drawn naturally from the bag rather than the pool). Returns
// the rack string by pointer (do not free).
void pass_cycle_sample_p1(const PassCycleTable *table, uint64_t pair_id,
                          const char **p1_rack_out);

// Look up a canonical rack string (sorted A-Z, blanks LAST, e.g. "ENOTUV?")
// in the pool. Returns 1 if pass-favorable, 0 if exchange-only, -1 if not in
// pool. Used during play to decide whether the current player's rack
// triggers force-pass.
int pass_cycle_lookup_is_pass(const PassCycleTable *table,
                              const char *canonical_rack);

// Write one game result row. Thread-safe.
// branch:     0 = P1 passes turn 0 (while board empty), 1 = P1 plays normally.
// outcome:    0 = loss, 1 = tie, 2 = win (from P1's perspective).
// p{1,2}_final_rack: rack composition at game-end (after final draw),
//                    used for end-game rack-value scoring.
// history:    pipe-joined per-turn "<rack>:<move>" trace, or NULL.
void pass_cycle_record(PassCycleTable *table, uint64_t pair_id,
                       const char *p1_rack, const char *p2_rack,
                       const char *p1_final_rack, const char *p2_final_rack,
                       int branch, int outcome, int num_turns,
                       const char *history);

#endif
