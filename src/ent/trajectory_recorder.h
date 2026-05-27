#ifndef TRAJECTORY_RECORDER_H
#define TRAJECTORY_RECORDER_H

#include <stdint.h>

// Per-turn position snapshot recorder for the 91-8 V-model training
// pipeline (see bots/winpct/docs/per_turn_self_play_spec.md and
// project_nonopener_indexing memory).
//
// At each turn of HastyBot self-play, before the move plays, write one
// row capturing the full game state (board + both racks + both scores)
// plus the action about to be played. Output is stratified per-bag
// (= unseen total = bag_letters + RACK_SIZE) so downstream tools can
// process one bag's worth of positions independently.
//
// Env var:
//   MAGPIE_TRAJECTORY_RECORDER=<dir>   recorder root directory
//
// Layout: <dir>/positions/bag_<NN>.csv for NN in {8..91, 93}.
// (bag=92 never occurs naturally; 93 is opener.)

typedef struct TrajectoryRecorder TrajectoryRecorder;

TrajectoryRecorder *trajectory_recorder_create(const char *base_dir);
void trajectory_recorder_destroy(TrajectoryRecorder *r);

// Write one position snapshot. Thread-safe.
//
// game_id            unique identifier for this game (game_runner->game_number).
// turn               1-indexed game turn (1 = opener, 2 = opp's first move, ...).
// bag                project-convention unseen total = bag_letters + RACK_SIZE.
//                    Routes the write to the bag_<bag>.csv file. Out-of-range
//                    bag values are silently dropped.
// on_turn            0 or 1, identifying the player about to play.
// p1_rack, p2_rack   canonical rack strings (blanks as '?', sorted alpha).
// p1_score, p2_score current scores (post all previous moves, pre this one).
// board_cgp          full CGP FEN string capturing board+racks+scores+bag.
//                    Reproducible by feeding back into magpie's `cgp` command.
// action_kind        0=pass, 1=exchange, 2=play.
// action_repr        "" for pass, exchanged tiles for exch, played-tile string
//                    for play (per game_string.c convention).
// action_size        0 for pass, N tiles thrown for exch, N tiles played for
//                    play.
// leave              post-action rack (canonical, sorted). Equals current rack
//                    for pass; rack minus exchanged tiles for exch; rack minus
//                    played tiles for play.
void trajectory_recorder_write(
    TrajectoryRecorder *r, uint64_t game_id, int turn, int bag,
    int on_turn, const char *p1_rack, const char *p2_rack,
    int p1_score, int p2_score, const char *board_cgp,
    int action_kind, const char *action_repr, int action_size,
    const char *leave);

#endif
