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
// Two-phase commit: rows are first staged in a per-worker buffer (one
// game at a time per worker), then flushed to disk only if the game
// ends normally (GAME_END_REASON_STANDARD). Games ending via
// GAME_END_REASON_CONSECUTIVE_ZEROS (6-pass terminus) are discarded —
// those games produce pathological positions where one or both players
// got stuck, distorting the training data.
//
// Env var:
//   MAGPIE_TRAJECTORY_RECORDER=<dir>   recorder root directory
//
// Layout: <dir>/positions/bag_<NN>.csv for NN in {1..91, 93}.
// (bag=92 never occurs naturally; 93 is opener.) bag=1..7 is the
// post-empty-bag endgame regime.

typedef struct TrajectoryRecorder TrajectoryRecorder;
typedef struct TrajectoryGameBuffer TrajectoryGameBuffer;

TrajectoryRecorder *trajectory_recorder_create(const char *base_dir);
void trajectory_recorder_destroy(TrajectoryRecorder *r);

// Per-worker buffer that stages rows for the current game in flight.
// One buffer per AutoplayWorker; reused across games (reset on game
// start, commit or discard on game end).
TrajectoryGameBuffer *trajectory_game_buffer_create(void);
void trajectory_game_buffer_destroy(TrajectoryGameBuffer *b);

// Clear buffered rows. Call at the start of each new game.
void trajectory_game_buffer_reset(TrajectoryGameBuffer *b);

// Stage one position row in the buffer. Same field semantics as the
// pre-staging API. Returns immediately on a NULL buffer (recorder
// disabled).
void trajectory_game_buffer_add(
    TrajectoryGameBuffer *b, uint64_t game_id, int turn, int bag,
    int on_turn, const char *p1_rack, const char *p2_rack,
    int p1_score, int p2_score, const char *board_cgp,
    int action_kind, const char *action_repr, int action_size,
    int move_score, int score_diff_pre, int score_diff_post,
    const char *leave);

// Flush all staged rows to disk under the recorder. Call when the
// just-played game ended normally. Each row's per-game eventual_outcome
// (0=loss, 1=win, 2=tie) is computed at commit time from the FINAL
// p1_score / p2_score (after end-of-game tile-bag penalty adjustment)
// against the row's on_turn player. Resets the buffer.
void trajectory_game_buffer_commit(
    TrajectoryRecorder *r, TrajectoryGameBuffer *b,
    int final_p1_score, int final_p2_score);

// Drop all staged rows without writing. Call when the just-played game
// ended via consecutive-zeros / pass-out (pathological positions).
// Resets the buffer.
void trajectory_game_buffer_discard(TrajectoryGameBuffer *b);

#endif
