#ifndef T6_BASELINE_H
#define T6_BASELINE_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "../compat/cpthread.h"

// Phase 2 offline baseline tool. Activated via MAGPIE_T6_BASELINE_TASK_FILE
// env var. The autoplay worker loop, when this is set, replaces the
// normal per-game flow with a per-task batch:
//   - Read next task row: target_rack, opp_rack, action_repr, n_games
//   - Run n_games games starting at "T6" with both racks set, target's
//     T6 action forced; play out post-T6 with HastyBot
//   - Aggregate (n_W, n_L, n_T, target_score_sum, opp_score_sum) over
//     the batch
//   - Write one row to results CSV
//
// Concurrency: workers compete on a shared mutex for the next task and
// for the results-file writer. The task file is read sequentially.

typedef struct T6BaselineState {
  FILE *task_file;
  cpthread_mutex_t task_mutex;
  FILE *out_file;
  cpthread_mutex_t out_mutex;
  // Reusable line buffer for task reads (held under task_mutex).
  char line_buf[1024];
  bool exhausted;  // task file fully consumed
  uint64_t n_tasks_dispatched;
} T6BaselineState;

typedef struct T6BaselineTask {
  char target_rack[16];
  char opp_rack[16];
  char action_repr[64];
  int n_games;
} T6BaselineTask;

// Open task and out files. Writes CSV header to out_file. Returns NULL
// on failure (file open error).
T6BaselineState *t6_baseline_state_create(const char *task_path,
                                          const char *out_path);

void t6_baseline_state_destroy(T6BaselineState *state);

// Pull the next task. Returns false if no more tasks (sets state's
// exhausted flag). Thread-safe.
bool t6_baseline_get_next_task(T6BaselineState *state, T6BaselineTask *out);

// Record the aggregated outcome of running a task's n_games games.
// Thread-safe.
void t6_baseline_record_result(T6BaselineState *state,
                               const T6BaselineTask *task,
                               int n_W, int n_L, int n_T,
                               int64_t target_score_sum,
                               int64_t opp_score_sum);

#endif
