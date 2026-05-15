#include "t6_baseline.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool load_pool(T6BaselineState *s, const char *path) {
  FILE *fh = fopen(path, "r");
  if (!fh) {
    fprintf(stderr, "t6_baseline: cannot open pool file %s\n", path);
    return false;
  }
  // Header line: "p1_rack,weight"
  char line[256];
  if (fgets(line, sizeof(line), fh) == NULL) {
    fclose(fh);
    return false;
  }
  // Two-pass: count rows, then allocate and fill.
  int cap = 0;
  while (fgets(line, sizeof(line), fh)) cap++;
  if (cap <= 0) {
    fprintf(stderr, "t6_baseline: empty pool file %s\n", path);
    fclose(fh);
    return false;
  }
  fseek(fh, 0, SEEK_SET);
  fgets(line, sizeof(line), fh);  // skip header

  s->pool_racks = (char (*)[16])calloc(cap, 16);
  s->pool_cum_weight = (double *)calloc(cap, sizeof(double));
  if (!s->pool_racks || !s->pool_cum_weight) {
    fprintf(stderr, "t6_baseline: oom loading pool\n");
    fclose(fh);
    return false;
  }
  int n = 0;
  double cum = 0.0;
  while (fgets(line, sizeof(line), fh) && n < cap) {
    char *comma = strchr(line, ',');
    if (!comma) continue;
    *comma = '\0';
    double w = atof(comma + 1);
    if (w <= 0) continue;
    size_t rl = strlen(line);
    if (rl >= 16) rl = 15;
    memcpy(s->pool_racks[n], line, rl);
    s->pool_racks[n][rl] = '\0';
    cum += w;
    s->pool_cum_weight[n] = cum;
    n++;
  }
  s->pool_size = n;
  s->pool_total_weight = cum;
  fclose(fh);
  fprintf(stderr,
          "t6_baseline: loaded %d opp racks from pool (total_weight=%.0f)\n",
          n, cum);
  return n > 0;
}

T6BaselineState *t6_baseline_state_create(const char *task_path,
                                          const char *out_path,
                                          const char *pool_path) {
  T6BaselineState *s = (T6BaselineState *)calloc(1, sizeof(T6BaselineState));
  if (!s) return NULL;
  s->task_file = fopen(task_path, "r");
  if (!s->task_file) {
    fprintf(stderr, "t6_baseline: cannot open task file %s\n", task_path);
    free(s);
    return NULL;
  }
  // Skip header
  if (fgets(s->line_buf, sizeof(s->line_buf), s->task_file) != NULL) {
    if (s->line_buf[0] == 't' || s->line_buf[0] == 'T') {
      // header
    } else {
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
          "target_rack,action_repr,leave,blanks_used,score,"
          "n_games_run,n_W,n_L,n_T,"
          "target_score_sum,opp_score_sum\n");
  fflush(s->out_file);
  cpthread_mutex_init(&s->task_mutex);
  cpthread_mutex_init(&s->out_mutex);
  s->exhausted = false;
  s->n_tasks_dispatched = 0;
  if (!load_pool(s, pool_path)) {
    fclose(s->task_file);
    fclose(s->out_file);
    free(s->pool_racks);
    free(s->pool_cum_weight);
    free(s);
    return NULL;
  }
  return s;
}

void t6_baseline_state_destroy(T6BaselineState *state) {
  if (!state) return;
  if (state->task_file) fclose(state->task_file);
  if (state->out_file) fclose(state->out_file);
  free(state->pool_racks);
  free(state->pool_cum_weight);
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
  char buf[1024];
  size_t llen = strlen(line);
  if (llen >= sizeof(buf)) llen = sizeof(buf) - 1;
  memcpy(buf, line, llen);
  buf[llen] = '\0';
  state->n_tasks_dispatched++;
  cpthread_mutex_unlock(&state->task_mutex);

  size_t b = strlen(buf);
  while (b > 0 && (buf[b - 1] == '\n' || buf[b - 1] == '\r')) buf[--b] = '\0';

  // Parse: target_rack,action_repr,n_games
  char *p = buf;
  char *fields[3] = {NULL, NULL, NULL};
  int n_fields = 0;
  fields[n_fields++] = p;
  while (*p && n_fields < 3) {
    if (*p == ',') {
      *p = '\0';
      fields[n_fields++] = p + 1;
    }
    p++;
  }
  if (n_fields < 3) {
    fprintf(stderr, "t6_baseline: malformed task (got %d fields): %s\n",
            n_fields, buf);
    return false;
  }
  snprintf(out->target_rack, sizeof(out->target_rack), "%s", fields[0]);
  snprintf(out->action_repr, sizeof(out->action_repr), "%s", fields[1]);
  out->n_games = atoi(fields[2]);
  if (out->n_games <= 0) out->n_games = 1;
  return true;
}

const char *t6_baseline_sample_opp(const T6BaselineState *state,
                                    uint64_t seed) {
  if (state->pool_size <= 0) return NULL;
  // Splitmix-style hash to a uniform double in [0, total_weight)
  uint64_t z = seed;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  z = z ^ (z >> 31);
  double r = (double)(z & ((1ULL << 53) - 1)) / (double)(1ULL << 53);
  double target = r * state->pool_total_weight;
  // Binary search on cum_weight
  int lo = 0, hi = state->pool_size - 1;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (state->pool_cum_weight[mid] < target) lo = mid + 1;
    else hi = mid;
  }
  return state->pool_racks[lo];
}

void t6_baseline_record_result(T6BaselineState *state,
                               const T6BaselineTask *task,
                               const char *leave_str, int blanks_used,
                               int score,
                               int n_games_run, int n_W, int n_L, int n_T,
                               int64_t target_score_sum,
                               int64_t opp_score_sum) {
  cpthread_mutex_lock(&state->out_mutex);
  fprintf(state->out_file,
          "%s,%s,%s,%d,%d,%d,%d,%d,%d,%lld,%lld\n",
          task->target_rack, task->action_repr,
          leave_str ? leave_str : "", blanks_used, score,
          n_games_run, n_W, n_L, n_T,
          (long long)target_score_sum, (long long)opp_score_sum);
  cpthread_mutex_unlock(&state->out_mutex);
}
