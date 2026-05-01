#ifndef EMPTY_BOARD_H
#define EMPTY_BOARD_H

#include <stdint.h>

// Per-decision recorder for the empty-board / pass-cycle value sub-model
// dataset (see bots/winpct/docs/empty_board_cycle_spec.md).
//
// One row per cycle-alive empty-board decision (turns 1..6) for the player
// on turn. Cycle entry + per-turn force-pass logic is reused from the
// existing PassCycleTable; this module only provides the recording channel.
//
// Output is independent from the FJ recorder strata layout and from
// MAGPIE_PASS_CYCLE_OUT.
//
// Env var:
//   MAGPIE_EMPTY_BOARD_OUT=path   CSV records file

typedef struct EmptyBoardRecorder EmptyBoardRecorder;

EmptyBoardRecorder *empty_board_recorder_create(const char *out_path);
void empty_board_recorder_destroy(EmptyBoardRecorder *r);

// Write one decision row. Thread-safe.
//
// pair_id, branch_id              identify the (pair, fork) the row belongs to.
//                                 Slice 1: branch_id mirrors the binary
//                                 pass_cycle_branch (0 = P1 force-pass, 1 =
//                                 P1 plays normally). Slice 2 will encode
//                                 K-way fork paths.
// turn_on_empty_board ∈ {1..6}    cycle position of this decision.
// rack                             canonical rack of player on turn (e.g.
//                                 "AEINRST"; blanks last as "?").
// bag_counts[27]                   unseen tile counts from this player's view
//                                 (ML index 0=blank, 1..26=A..Z). Equals
//                                 total_distribution - my_rack.
// opp_action_history               pipe-joined opp action codes so far in
//                                 cycle ("" if turn 1; tokens "P", "X<tiles>",
//                                 "T<tiles>"). Slice 1 captures whatever the
//                                 binary-cycle opp actually did.
// action_kind                      0=pass, 1=exchange, 2=play.
// action_repr                      "" for pass, exchanged tiles for exch,
//                                 played tiles for play.
// action_size                      0 for pass, N for exch, tiles_played for
//                                 play.
// eventual_outcome                 0=loss, 1=tie, 2=win, from this player's
//                                 perspective.
// eventual_margin                  signed final score diff (this player - opp).
void empty_board_recorder_write(
    EmptyBoardRecorder *r, uint64_t pair_id, int branch_id,
    int turn_on_empty_board, const char *rack, const uint8_t bag_counts[27],
    const char *opp_action_history, int action_kind, const char *action_repr,
    int action_size, int eventual_outcome, int eventual_margin);

#endif
