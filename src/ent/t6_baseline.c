#include "t6_baseline.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

T6BaselineState *t6_baseline_state_create(const char *task_path,
                                          const char *out_path) {
  T6BaselineState *s = (T6BaselineState *)calloc(1, sizeof(T6BaselineState));
  if (!s) return NULL;
  s->task_file = fopen(task_path, "r");
  if (!s->task_file) {
    fprintf(stderr, "t6_baseline: cannot open task file %s\n", task_path);
    free(s);
    return NULL;
  }
  // Skip header line if present (peek at first byte; if non-digit/letter
  // assume header).
  if (fgets(s->line_buf, sizeof(s->line_buf), s->task_file) != NULL) {
    if (s->line_buf[0] == 't' || s->line_buf[0] == 'T') {
      // header detected; line consumed
    } else {
      // not a header — rewind so we don't drop the first task.
      fseek(s->task_file, 0, SEEK_SET);
    }
  }
  s->out_file = fopen(out_path, "w");
  if (!s->out_file) {
    fprintf(stderr, "t6_baseline: cannot open out file %s\n", out_path);
    fclose(s->task_file);
    free(s);
    return NULL;
  }
  fprintf(s->out_file,
          "target_rack,opp_rack,action_repr,n_W,n_L,n_T,"
          "target_score_sum,opp_score_sum\n");
  fflush(s->out_file);
  cpthread_mutex_init(&s->task_mutex);
  cpthread_mutex_init(&s->out_mutex);
  s->exhausted = false;
  s->n_tasks_dispatched = 0;
  return s;
}

void t6_baseline_state_destroy(T6BaselineState *state) {
  if (!state) return;
  if (state->task_file) fclose(state->task_file);
  if (state->out_file) fclose(state->out_file);
  free(state);
}

bool t6_baseline_get_next_task(T6BaselineState *state, T6BaselineTask *out) {
  cpthread_mutex_lock(&state->task_mutex);
  if (state->exhausted) {
    cpthread_mutex_unlock(&state->task_mutex);
    return false;
  }
  char *line = fgets(state->line_buf, sizeof(state->line_buf), state->task_file);
  if (!line) {
    state->exhausted = true;
    cpthread_mutex_unlock(&state->task_mutex);
    return false;
  }
  // Parse: target_rack,opp_rack,action_repr,n_games
  // Use a private buffer to avoid clobbering line_buf for the next reader.
  char buf[1024];
  size_t llen = strlen(line);
  if (llen >= sizeof(buf)) llen = sizeof(buf) - 1;
  memcpy(buf, line, llen);
  buf[llen] = '\0';
  state->n_tasks_dispatched++;
  cpthread_mutex_unlock(&state->task_mutex);

  // Strip trailing newline
  size_t b = strlen(buf);
  while (b > 0 && (buf[b - 1] == '\n' || buf[b - 1] == '\r')) buf[--b] = '\0';

  // Tokenize on commas. action_repr may contain spaces (e.g. "8H WaXY")
  // but no commas, so simple comma split works.
  char *p = buf;
  char *fields[4] = {NULL, NULL, NULL, NULL};
  int n_fields = 0;
  fields[n_fields++] = p;
  while (*p && n_fields < 4) {
    if (*p == ',') {
      *p = '\0';
      fields[n_fields++] = p + 1;
    }
    p++;
  }
  if (n_fields < 4) {
    fprintf(stderr, "t6_baseline: malformed task line (got %d fields): %s\n",
            n_fields, buf);
    return false;
  }
  snprintf(out->target_rack, sizeof(out->target_rack), "%s", fields[0]);
  snprintf(out->opp_rack, sizeof(out->opp_rack), "%s", fields[1]);
  snprintf(out->action_repr, sizeof(out->action_repr), "%s", fields[2]);
  out->n_games = atoi(fields[3]);
  if (out->n_games <= 0) out->n_games = 1;
  return true;
}

void t6_baseline_record_result(T6BaselineState *state,
                               const T6BaselineTask *task,
                               int n_W, int n_L, int n_T,
                               int64_t target_score_sum,
                               int64_t opp_score_sum) {
  cpthread_mutex_lock(&state->out_mutex);
  fprintf(state->out_file,
          "%s,%s,%s,%d,%d,%d,%lld,%lld\n",
          task->target_rack, task->opp_rack, task->action_repr,
          n_W, n_L, n_T,
          (long long)target_score_sum, (long long)opp_score_sum);
  cpthread_mutex_unlock(&state->out_mutex);
}
