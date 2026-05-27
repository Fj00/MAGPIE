#include "trajectory_recorder.h"

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>

#include "../util/io_util.h"

// Valid bag range: 1..91 (endgame through mid-game) plus 93 (opener).
// Index 0..90 for bag=1..91; index 91 for bag=93. bag=92 never occurs
// naturally. bag=0 = game over, no decision to record.
#define TR_BAG_MIN 1
#define TR_BAG_MAX 91
#define TR_OPENER_BAG 93
#define TR_NUM_BAGS ((TR_BAG_MAX - TR_BAG_MIN) + 1 + 1)  // 91 + 1 = 92; +1 opener

static int tr_bag_to_index(int bag) {
  if (bag == TR_OPENER_BAG) return TR_BAG_MAX - TR_BAG_MIN + 1;
  if (bag < TR_BAG_MIN || bag > TR_BAG_MAX) return -1;
  return bag - TR_BAG_MIN;
}

struct TrajectoryRecorder {
  char *base_dir;
  pthread_mutex_t open_mutex;
  pthread_mutex_t mutexes[TR_NUM_BAGS];
  FILE *fhs[TR_NUM_BAGS];
};

// Pre-formatted CSV line + which per-bag file it routes to. Avoids any
// re-parsing at commit time.
typedef struct StagedRow {
  int bag_idx;
  char *line;   // heap-allocated, owned by the buffer
} StagedRow;

struct TrajectoryGameBuffer {
  StagedRow *rows;
  int count;
  int capacity;
};

static void tr_mkdir_or_die(const char *path) {
  if (mkdir(path, 0775) != 0 && errno != EEXIST) {
    log_fatal("trajectory_recorder: mkdir %s: %s", path, strerror(errno));
  }
}

static void tr_raise_nofile_limit(void) {
  struct rlimit rl;
  if (getrlimit(RLIMIT_NOFILE, &rl) != 0) return;
  rlim_t want = 8192;
  if (rl.rlim_max != RLIM_INFINITY && want > rl.rlim_max) want = rl.rlim_max;
  if (rl.rlim_cur < want) {
    rl.rlim_cur = want;
    (void)setrlimit(RLIMIT_NOFILE, &rl);
  }
}

TrajectoryRecorder *trajectory_recorder_create(const char *base_dir) {
  if (!base_dir || !base_dir[0]) return NULL;
  tr_raise_nofile_limit();
  TrajectoryRecorder *r = malloc_or_die(sizeof(TrajectoryRecorder));
  r->base_dir = malloc_or_die(strlen(base_dir) + 1);
  strcpy(r->base_dir, base_dir);
  pthread_mutex_init(&r->open_mutex, NULL);
  for (int i = 0; i < TR_NUM_BAGS; i++) {
    pthread_mutex_init(&r->mutexes[i], NULL);
    r->fhs[i] = NULL;
  }
  tr_mkdir_or_die(r->base_dir);
  char positions_dir[1024];
  snprintf(positions_dir, sizeof(positions_dir), "%s/positions", r->base_dir);
  tr_mkdir_or_die(positions_dir);
  fprintf(stderr,
          "trajectory_recorder: base_dir=%s; lazy per-bag files in "
          "%s/bag_<NN>.csv; per-game commit on STANDARD end only "
          "(consecutive-zeros games discarded)\n",
          r->base_dir, positions_dir);
  return r;
}

void trajectory_recorder_destroy(TrajectoryRecorder *r) {
  if (!r) return;
  for (int i = 0; i < TR_NUM_BAGS; i++) {
    if (r->fhs[i]) {
      fflush(r->fhs[i]);
      fclose(r->fhs[i]);
    }
    pthread_mutex_destroy(&r->mutexes[i]);
  }
  pthread_mutex_destroy(&r->open_mutex);
  free(r->base_dir);
  free(r);
}

// Lazy-open the per-bag file. Double-check pattern: outer check is
// lock-free; if NULL, take the open_mutex and re-check under it before
// opening.
static FILE *tr_get_file(TrajectoryRecorder *r, int bag_idx, int bag) {
  if (r->fhs[bag_idx]) return r->fhs[bag_idx];
  pthread_mutex_lock(&r->open_mutex);
  if (!r->fhs[bag_idx]) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/positions/bag_%02d.csv",
             r->base_dir, bag);
    FILE *fh = fopen(path, "w");
    if (!fh) {
      log_fatal("trajectory_recorder: cannot open %s: %s", path,
                strerror(errno));
    }
    fprintf(fh,
            "game_id,turn,bag,on_turn,p1_rack,p2_rack,p1_score,p2_score,"
            "board_cgp,action_kind,action_repr,action_size,"
            "move_score,score_diff_pre,score_diff_post,leave\n");
    fflush(fh);
    r->fhs[bag_idx] = fh;
  }
  pthread_mutex_unlock(&r->open_mutex);
  return r->fhs[bag_idx];
}

// ---- Per-game buffer ----

TrajectoryGameBuffer *trajectory_game_buffer_create(void) {
  TrajectoryGameBuffer *b = malloc_or_die(sizeof(TrajectoryGameBuffer));
  b->capacity = 64;  // games usually 20-30 turns; resize if needed
  b->count = 0;
  b->rows = malloc_or_die(sizeof(StagedRow) * b->capacity);
  return b;
}

static void tr_buf_free_rows(TrajectoryGameBuffer *b) {
  for (int i = 0; i < b->count; i++) free(b->rows[i].line);
  b->count = 0;
}

void trajectory_game_buffer_destroy(TrajectoryGameBuffer *b) {
  if (!b) return;
  tr_buf_free_rows(b);
  free(b->rows);
  free(b);
}

void trajectory_game_buffer_reset(TrajectoryGameBuffer *b) {
  if (!b) return;
  tr_buf_free_rows(b);
}

void trajectory_game_buffer_add(
    TrajectoryGameBuffer *b, uint64_t game_id, int turn, int bag,
    int on_turn, const char *p1_rack, const char *p2_rack,
    int p1_score, int p2_score, const char *board_cgp,
    int action_kind, const char *action_repr, int action_size,
    int move_score, int score_diff_pre, int score_diff_post,
    const char *leave) {
  if (!b) return;
  const int bag_idx = tr_bag_to_index(bag);
  if (bag_idx < 0) return;
  // Format the row line. Use a big stack scratch; CGP can be ~200 chars
  // plus overhead. 1 KB is plenty.
  char line[1024];
  int n = snprintf(
      line, sizeof(line),
      "%llu,%d,%d,%d,%s,%s,%d,%d,\"%s\",%d,%s,%d,%d,%d,%d,%s\n",
      (unsigned long long)game_id, turn, bag, on_turn,
      p1_rack ? p1_rack : "", p2_rack ? p2_rack : "",
      p1_score, p2_score, board_cgp ? board_cgp : "",
      action_kind, action_repr ? action_repr : "", action_size,
      move_score, score_diff_pre, score_diff_post,
      leave ? leave : "");
  if (n <= 0 || n >= (int)sizeof(line)) return;  // overflow, skip
  if (b->count >= b->capacity) {
    b->capacity *= 2;
    b->rows = realloc(b->rows, sizeof(StagedRow) * b->capacity);
    if (!b->rows) log_fatal("trajectory_recorder: buffer realloc failed");
  }
  StagedRow *row = &b->rows[b->count++];
  row->bag_idx = bag_idx;
  row->line = malloc_or_die((size_t)n + 1);
  memcpy(row->line, line, (size_t)n + 1);
}

void trajectory_game_buffer_commit(
    TrajectoryRecorder *r, TrajectoryGameBuffer *b) {
  if (!r || !b) return;
  // Reconstruct bag from bag_idx for the file-open path; tr_get_file
  // needs it for filename formatting.
  for (int i = 0; i < b->count; i++) {
    const int bag_idx = b->rows[i].bag_idx;
    const int bag = (bag_idx == TR_BAG_MAX - TR_BAG_MIN + 1)
                       ? TR_OPENER_BAG
                       : bag_idx + TR_BAG_MIN;
    FILE *fh = tr_get_file(r, bag_idx, bag);
    if (!fh) continue;
    pthread_mutex_lock(&r->mutexes[bag_idx]);
    fputs(b->rows[i].line, fh);
    pthread_mutex_unlock(&r->mutexes[bag_idx]);
  }
  tr_buf_free_rows(b);
}

void trajectory_game_buffer_discard(TrajectoryGameBuffer *b) {
  if (!b) return;
  tr_buf_free_rows(b);
}
