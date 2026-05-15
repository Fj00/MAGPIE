#ifndef T6_BASELINE_H
#define T6_BASELINE_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "../compat/cpthread.h"

// Phase 2 offline baseline tool. Activated via:
//   MAGPIE_T6_BASELINE_TASK_FILE=path  (input task CSV)
//   MAGPIE_T6_BASELINE_OUT_FILE=path   (output results CSV)
//   MAGPIE_T6_BASELINE_POOL_FILE=path  (pre_t6_pool.csv for opp sampling)
//
// When all three are set, autoplay's worker loop replaces the normal
// per-game flow with per-task batches:
//   - Read next task row: target_rack, action_repr, n_games
//   - For each of n_games: sample opp_rack from the pre_t6_pool (weighted),
//     run a T6-from-scratch game (target=offline player 0, opp=player 1),
//     force target's first move to action_repr, play out post-T1 with
//     HastyBot
//   - Aggregate (n_W, n_L, n_T, target_score_sum, opp_score_sum)
//   - Write one result row
//
// Renumbering convention: in the offline frame, T1 = live T6. So the
// "target" rack we're characterizing is offline player 0 (= live P2 at
// T6), and "opp" from the pool is offline player 1 (= live P1 at T6).

typedef struct T6BaselineState {
  FILE *task_file;
  cpthread_mutex_t task_mutex;
  FILE *out_file;
  cpthread_mutex_t out_mutex;
  char line_buf[1024];
  bool exhausted;
  uint64_t n_tasks_dispatched;

  // Opp-rack pool: cumulative-weight-sampled at runtime.
  char (*pool_racks)[16];        // [pool_size][16] — packed rack strings
  double *pool_cum_weight;       // [pool_size] cumulative sum of weights
  int pool_size;
  double pool_total_weight;
} T6BaselineState;

typedef struct T6BaselineTask {
  char target_rack[16];
  char action_repr[64];
  int n_games;
} T6BaselineTask;

T6BaselineState *t6_baseline_state_create(const char *task_path,
                                          const char *out_path,
                                          const char *pool_path);

void t6_baseline_state_destroy(T6BaselineState *state);

bool t6_baseline_get_next_task(T6BaselineState *state, T6BaselineTask *out);

void t6_baseline_record_result(T6BaselineState *state,
                               const T6BaselineTask *task,
                               int n_games_run, int n_W, int n_L, int n_T,
                               int64_t target_score_sum,
                               int64_t opp_score_sum);

// Sample an opp rack from the pool (weighted). Writes into out_buf
// (must be ≥ 16 bytes). seed perturbs the pick.
const char *t6_baseline_sample_opp(const T6BaselineState *state,
                                    uint64_t seed);

#endif
