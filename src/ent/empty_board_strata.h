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
// TYPE      "all" unless leave_len is 5 or 6, in which case cons/vowel/mixed
//           by leave letter composition (Y treated as consonant; blank is
//           ignored in the count). L0-L4 and L7 are "all".
//
// Per-row schema (turn / kind / leave_len / TYPE implicit from path):
//
//   pair_id, branch_id, rack, opp_action_history, action_repr,
//   action_size, leave, eventual_outcome, p2_rack_source, natural_slot,
//   move_score, divergence_turn, p1_rack_source, p1_force_kind,
//   p2_force_kind
//
// move_score is the score the player earned on this turn: 0 for pass and
// exchange, the actual play score for K=2.
//
// divergence_turn is the "anchor turn" of this leaf (see autoplay.c).
//
// p1_rack_source: pinned to 0 in the new mode (P1 always pool-sampled);
//                 retained in the schema for older-data compatibility.
// p1_force_kind:  derived from P1 rack's is_pass classification.
//                 0 = T1 force-pass (is_pass=1 rack), 1 = T1 force-exch.
// p2_rack_source: 0 = pool (rejection-sampled by force_kind),
//                 1 = bag-random (play-prone P2, exposes the
//                                 "best play +4 vs exchange" tradeoff).
// p2_force_kind:  0 = T2 force-pass, 1 = T2 force-exch.
// (P2 force is set per-pair from an iter_count bit, giving balanced
//  (rack_source × force_kind) groups for T2 analysis.)
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
    int divergence_turn, int p1_rack_source, int p1_force_kind,
    int p2_force_kind);

#endif
