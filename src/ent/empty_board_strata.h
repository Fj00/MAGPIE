#ifndef EMPTY_BOARD_STRATA_H
#define EMPTY_BOARD_STRATA_H

#include <stdint.h>

// Strata-folder output for the empty-board cycle recorder. Routes each
// row to a per-(turn, kind, leave_len, type) CSV file:
//
//   <base>/T<turn>/K<kind>/L<leave_len>/type=<TYPE>.csv
//
// turn      1..6 (cycle position)
// kind      0=pass, 1=exch, 2=play
// leave_len 0..7 (post-action rack size; pass=7, exch/play = 7-action_size)
// TYPE      "all" for leave_len < 3 or > 6, else cons/vowel/mixed by
//           leave letter composition (Y treated as consonant; blank is
//           ignored in the count).
//
// Per-row schema (turn / kind / leave_len / TYPE implicit from path):
//
//   pair_id, branch_id, rack, opp_action_history, action_repr,
//   action_size, leave, eventual_outcome, p2_rack_source, natural_slot,
//   move_score, divergence_turn, p1_rack_source, p1_force_kind
//
// move_score is the score the player earned on this turn: 0 for pass and
// exchange, the actual play score for K=2. With both players at 0 score
// across the cycle, move_score equals the score differential after the
// move.
//
// divergence_turn is the "anchor turn" of this leaf:
//   -1: pure natural — no fork in this branch's ancestry chose a
//       non-natural slot; what real HastyBot/force-pass play produces.
//   k (1..6): the chain followed natural through T1..T(k-1) and then
//       chose a non-natural slot at Tk; from Tk+1 onward the chain again
//       followed natural. Useful for "natural-from-Tk-anchor" subsets.
// Multi-divergence leaves (>=2 non-natural choices) are NOT emitted —
// they're counterfactual-of-counterfactual data with no clean
// interpretation. Lookup cells are populated by all (-1, 1..6) leaves.
//
// p1_rack_source: where P1's starting rack came from.
//   0: pool (sampled weighted from pass_cycle_racks.csv)
//   1: non-pool (sampled uniformly from bag — racks that wouldn't
//      naturally cycle, force-cycled here for diagnostic data)
//
// p1_force_kind: how T1 was forced for P1.
//   0: T1 force-pass
//   1: T1 force-exchange
// For pool racks, force_kind is determined by the rack's is_pass bit
// (is_pass=1 → pass, is_pass=0 → exch). For non-pool racks, force_kind
// is set per-pair from an iter_count bit, giving 4 distinct groups
// across (rack_source, force_kind) for cross-comparison.
//
// Files lazy-opened on first write. Mutex per file. Compatible with
// running ALONGSIDE the flat MAGPIE_EMPTY_BOARD_OUT recorder (both can
// be set; both write).
//
// Env var: MAGPIE_EMPTY_BOARD_STRATA=base_dir

typedef struct EmptyBoardStrataRecorder EmptyBoardStrataRecorder;

EmptyBoardStrataRecorder *empty_board_strata_create(const char *base_dir);
void empty_board_strata_destroy(EmptyBoardStrataRecorder *r);

void empty_board_strata_write(
    EmptyBoardStrataRecorder *r, uint64_t pair_id, uint64_t branch_id,
    int turn_on_empty_board, const char *rack,
    const char *opp_action_history, int action_kind, const char *action_repr,
    int action_size, const char *leave, int eventual_outcome,
    int p2_rack_source, int natural_slot, int move_score,
    int divergence_turn, int p1_rack_source, int p1_force_kind);

#endif
