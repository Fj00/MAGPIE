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
//                                 pass_cycle_branch (0 / 1).
//                                 Slice 2: branch_id is a packed fork-path
//                                 encoding (uint64). Each fork left-shifts by
//                                 8 and OR's in (slot_index + 1); 0 bytes
//                                 mark fork-depth not reached. Slot 0 is
//                                 always pass; slots 1..N depend on mode
//                                 (K=3 mode → 1=best-exch, 2=best-play;
//                                 subset-fan-out mode → 1..N=exchange subsets
//                                 in equity order, last slot=best-play if
//                                 turn 6).
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
//                                 perspective. (No margin field — for the
//                                 value sub-model only the W/L/T outcome
//                                 matters; per-play score is already in the
//                                 move and not needed at this aggregation.)
// p2_rack_source                   0=pool (P2 drawn from is_pass pool, current
//                                 default), 1=random (P2 drawn from regular
//                                 bag; mix-random mode only). Identifies the
//                                 distribution shift between cycle-favorable
//                                 and realistic-opp halves of a mixed run.
void empty_board_recorder_write(
    EmptyBoardRecorder *r, uint64_t pair_id, uint64_t branch_id,
    int turn_on_empty_board, const char *rack, const uint8_t bag_counts[27],
    const char *opp_action_history, int action_kind, const char *action_repr,
    int action_size, int eventual_outcome, int p2_rack_source);

#endif
