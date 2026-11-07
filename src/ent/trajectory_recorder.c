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
  int shard;  // MAGPIE_TRAJECTORY_SHARD=1 -> per-worker shard files
  pthread_mutex_t open_mutex;
  pthread_mutex_t mutexes[TR_NUM_BAGS];
  FILE *fhs[TR_NUM_BAGS];
  // POOL MODE (2026-11-06; B repo plan noble-popping-kitten.md): MAGPIE_TRAJECTORY_POOL=1
  // writes fill-pool lines "on-turn-first CGP<TAB>opponent leave" to
  // positions/pp_bag<NN>[.w<W>].txt instead of the CSV rows; =2 writes both (test).
  // MAGPIE_POOL_SPEC: bag,lead_lo,lead_hi,p_mid -- keep every position with lead <=
  // lead_lo or >= lead_hi, the rest with probability p_mid (hash of the CGP, so a
  // repeated position is kept or dropped consistently). Bags without a spec row: none.
  int pool;
  uint8_t pool_has[TR_NUM_BAGS];
  int pool_lo[TR_NUM_BAGS], pool_hi[TR_NUM_BAGS];
  double pool_p[TR_NUM_BAGS];
  FILE *pool_fhs[TR_NUM_BAGS];  // non-shard mode: shared per-bag pool files
};

static const char *TR_CSV_HEADER =
    "game_id,turn,bag,on_turn,p1_rack,p2_rack,p1_score,p2_score,"
    "board_cgp,action_kind,action_repr,action_size,"
    "move_score,score_diff_pre,score_diff_post,leave,eventual_outcome,"
    "final_spread,pick\n";
// pick (2026-11-07, LAST column so positional readers are unaffected): why a
// position-pool fan-out recorded this move -- bit 1 = cell pick (credits the
// force table), bit 2 = ranked pick (top equity / best exchange / best bingo,
// MAGPIE_PP_RANK_TOPK; credits nothing). 0 = an ordinary game move.

// Pre-formatted CSV line (without trailing newline / eventual_outcome)
// + which per-bag file it routes to + which player is on turn. The
// outcome column is appended at commit time once the game winner is
// known. Avoids re-parsing the full line at commit; only the trailing
// ",<outcome>\n" suffix is written there.
typedef struct StagedRow {
  int bag_idx;
  uint8_t on_turn;  // 0 = p1, 1 = p2
  char *line;       // heap-allocated, owned by the buffer; no trailing \n
  // pool mode only:
  int turn;
  int lead;          // score_diff_pre (mover's view)
  uint8_t mover7;    // mover holds 7 tiles
  char *pool_cgp;    // on-turn-first CGP (heap) or NULL
  char leave[12];    // this move's leave = the NEXT row's opponent leave
  uint8_t pick;      // the trailing `pick` column
} StagedRow;

struct TrajectoryGameBuffer {
  StagedRow *rows;
  int count;
  int capacity;
  int worker_index;
  int next_pick;                    // pick of the next staged row (then reset)
  int pool;                         // MAGPIE_TRAJECTORY_POOL at creation
  FILE *own_fhs[TR_NUM_BAGS];       // shard mode: this worker's per-bag files
  FILE *own_pool_fhs[TR_NUM_BAGS];  // shard mode: this worker's per-bag pool files
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

static void tr_load_pool_spec(TrajectoryRecorder *r) {
  const char *pe = getenv("MAGPIE_TRAJECTORY_POOL");
  r->pool = (pe && pe[0]) ? atoi(pe) : 0;
  for (int i = 0; i < TR_NUM_BAGS; i++) {
    r->pool_has[i] = 0;
    r->pool_fhs[i] = NULL;
  }
  if (r->pool <= 0) return;
  const char *sp = getenv("MAGPIE_POOL_SPEC");
  if (!sp || !sp[0]) log_fatal("trajectory_recorder: MAGPIE_TRAJECTORY_POOL needs MAGPIE_POOL_SPEC");
  FILE *f = fopen(sp, "r");
  if (!f) log_fatal("trajectory_recorder: cannot open %s: %s", sp, strerror(errno));
  char line[256];
  int n = 0;
  while (fgets(line, sizeof(line), f)) {
    int bag, lo, hi;
    double p;
    if (line[0] == '#' || sscanf(line, "%d,%d,%d,%lf", &bag, &lo, &hi, &p) != 4) continue;
    const int bi = tr_bag_to_index(bag);
    if (bi < 0) continue;
    r->pool_has[bi] = 1;
    r->pool_lo[bi] = lo;
    r->pool_hi[bi] = hi;
    r->pool_p[bi] = p;
    n++;
  }
  fclose(f);
  if (n == 0) log_fatal("trajectory_recorder: no bag rows in %s", sp);
  fprintf(stderr, "trajectory_recorder: POOL mode %d -- %d bag(s) from %s; writes "
          "pp_bag<NN>%s.txt%s\n", r->pool, n, sp, r->shard ? ".w<WORKER>" : "",
          r->pool == 2 ? " AND the CSV rows" : " (no CSV rows)");
}

TrajectoryRecorder *trajectory_recorder_create(const char *base_dir) {
  if (!base_dir || !base_dir[0]) return NULL;
  tr_raise_nofile_limit();
  TrajectoryRecorder *r = malloc_or_die(sizeof(TrajectoryRecorder));
  r->base_dir = malloc_or_die(strlen(base_dir) + 1);
  strcpy(r->base_dir, base_dir);
  const char *shard_env = getenv("MAGPIE_TRAJECTORY_SHARD");
  r->shard = (shard_env && shard_env[0] == '1') ? 1 : 0;
  pthread_mutex_init(&r->open_mutex, NULL);
  for (int i = 0; i < TR_NUM_BAGS; i++) {
    pthread_mutex_init(&r->mutexes[i], NULL);
    r->fhs[i] = NULL;
  }
  tr_load_pool_spec(r);
  tr_mkdir_or_die(r->base_dir);
  char positions_dir[1024];
  snprintf(positions_dir, sizeof(positions_dir), "%s/positions", r->base_dir);
  tr_mkdir_or_die(positions_dir);
  fprintf(stderr,
          "trajectory_recorder: base_dir=%s; %s in %s/; per-game commit on "
          "STANDARD end only (consecutive-zeros games discarded)\n",
          r->base_dir,
          r->shard ? "per-worker shard files bag_<NN>.w<WORKER>.csv (lock-free)"
                   : "lazy per-bag files bag_<NN>.csv",
          positions_dir);
  return r;
}

void trajectory_recorder_destroy(TrajectoryRecorder *r) {
  if (!r) return;
  for (int i = 0; i < TR_NUM_BAGS; i++) {
    if (r->fhs[i]) {
      fflush(r->fhs[i]);
      fclose(r->fhs[i]);
    }
    if (r->pool_fhs[i]) {
      fflush(r->pool_fhs[i]);
      fclose(r->pool_fhs[i]);
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
    fputs(TR_CSV_HEADER, fh);
    fflush(fh);
    r->fhs[bag_idx] = fh;
  }
  pthread_mutex_unlock(&r->open_mutex);
  return r->fhs[bag_idx];
}

// ---- Per-game buffer ----

TrajectoryGameBuffer *trajectory_game_buffer_create(int worker_index) {
  TrajectoryGameBuffer *b = malloc_or_die(sizeof(TrajectoryGameBuffer));
  b->capacity = 64;  // games usually 20-30 turns; resize if needed
  b->count = 0;
  b->rows = malloc_or_die(sizeof(StagedRow) * b->capacity);
  b->worker_index = worker_index;
  b->next_pick = 0;
  const char *pe = getenv("MAGPIE_TRAJECTORY_POOL");
  b->pool = (pe && pe[0]) ? atoi(pe) : 0;
  for (int i = 0; i < TR_NUM_BAGS; i++) {
    b->own_fhs[i] = NULL;
    b->own_pool_fhs[i] = NULL;
  }
  return b;
}

// Shard mode: lazy-open this worker's own per-bag file (no lock — the worker
// owns it). positions/bag_<NN>.w<WORKER>.csv.
static FILE *tr_get_own_file(TrajectoryRecorder *r, TrajectoryGameBuffer *b,
                             int bag_idx, int bag) {
  if (b->own_fhs[bag_idx]) return b->own_fhs[bag_idx];
  char path[1024];
  snprintf(path, sizeof(path), "%s/positions/bag_%02d.w%d.csv", r->base_dir,
           bag, b->worker_index);
  FILE *fh = fopen(path, "w");
  if (!fh) {
    log_fatal("trajectory_recorder: cannot open %s: %s", path, strerror(errno));
  }
  fputs(TR_CSV_HEADER, fh);
  b->own_fhs[bag_idx] = fh;
  return fh;
}

static void tr_buf_free_rows(TrajectoryGameBuffer *b) {
  for (int i = 0; i < b->count; i++) {
    free(b->rows[i].line);
    free(b->rows[i].pool_cgp);
  }
  b->count = 0;
}

void trajectory_game_buffer_destroy(TrajectoryGameBuffer *b) {
  if (!b) return;
  tr_buf_free_rows(b);
  for (int i = 0; i < TR_NUM_BAGS; i++) {
    if (b->own_fhs[i]) {
      fflush(b->own_fhs[i]);
      fclose(b->own_fhs[i]);
    }
    if (b->own_pool_fhs[i]) {
      fflush(b->own_pool_fhs[i]);
      fclose(b->own_pool_fhs[i]);
    }
  }
  free(b->rows);
  free(b);
}

void trajectory_game_buffer_reset(TrajectoryGameBuffer *b) {
  if (!b) return;
  tr_buf_free_rows(b);
}

void trajectory_game_buffer_set_next_pick(TrajectoryGameBuffer *b, int pick) {
  if (b) b->next_pick = pick;
}

// The recorded CGP is P1-FIRST ("board r1/r2 s1/s2 ..."); magpie's game_load_cgp
// treats rack1 as the player on turn. For on_turn == 1 swap racks and scores --
// exactly B repo build_bag_balanced_pool.to_on_turn_first. Returns a heap string.
static char *tr_on_turn_first(const char *cgp, int on_turn) {
  const size_t len = strlen(cgp);
  char *out = malloc_or_die(len + 1);
  memcpy(out, cgp, len + 1);
  if (on_turn != 1) return out;
  const char *sp1 = strchr(cgp, ' ');
  const char *sp2 = sp1 ? strchr(sp1 + 1, ' ') : NULL;
  if (!sp1 || !sp2) return out;
  const char *sp3 = strchr(sp2 + 1, ' ');
  const char *racks = sp1 + 1, *racks_end = sp2;
  const char *scores = sp2 + 1, *scores_end = sp3 ? sp3 : cgp + len;
  const char *rs = memchr(racks, '/', (size_t)(racks_end - racks));
  const char *ss = memchr(scores, '/', (size_t)(scores_end - scores));
  if (!rs || !ss) return out;
  char *w = out + (sp1 - cgp) + 1;   // rewrite from the racks field on
  const size_t r2 = (size_t)(racks_end - rs - 1), r1 = (size_t)(rs - racks);
  memcpy(w, rs + 1, r2); w += r2; *w++ = '/';
  memcpy(w, racks, r1); w += r1; *w++ = ' ';
  const size_t s2 = (size_t)(scores_end - ss - 1), s1 = (size_t)(ss - scores);
  memcpy(w, ss + 1, s2); w += s2; *w++ = '/';
  memcpy(w, scores, s1); w += s1;
  memcpy(w, scores_end, len - (size_t)(scores_end - cgp) + 1);   // rest incl. '\0'
  return out;
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
  // Format the row line WITHOUT trailing newline / eventual_outcome —
  // those are appended at commit time when the game winner is known.
  // 1 KB scratch (CGP ~200 chars + overhead is comfortable).
  char line[1024];
  int n = snprintf(
      line, sizeof(line),
      "%llu,%d,%d,%d,%s,%s,%d,%d,\"%s\",%d,%s,%d,%d,%d,%d,%s",
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
  row->on_turn = (uint8_t)(on_turn ? 1 : 0);
  row->line = malloc_or_die((size_t)n + 1);
  memcpy(row->line, line, (size_t)n + 1);
  // pool fields: every row (the previous row supplies the next one's opponent leave);
  // the pool CGP only where a pool line can be written (bags 8..91)
  row->turn = turn;
  row->lead = score_diff_pre;
  const char *mr = on_turn ? p2_rack : p1_rack;
  row->mover7 = (uint8_t)(mr && strlen(mr) == 7);
  snprintf(row->leave, sizeof(row->leave), "%s", leave ? leave : "");
  row->pick = (uint8_t)b->next_pick;
  b->next_pick = 0;
  row->pool_cgp = (b->pool > 0 && bag >= 8 && bag <= TR_BAG_MAX && board_cgp)
                      ? tr_on_turn_first(board_cgp, on_turn ? 1 : 0)
                      : NULL;
}

// ---- pool mode writer ----
static uint64_t tr_fnv1a(const char *s) {
  uint64_t h = 1469598103934665603ULL;
  for (; *s; s++) { h ^= (unsigned char)*s; h *= 1099511628211ULL; }
  return h;
}

static FILE *tr_get_pool_file(TrajectoryRecorder *r, TrajectoryGameBuffer *b,
                              int bag_idx, int bag) {
  FILE **slot = r->shard ? &b->own_pool_fhs[bag_idx] : &r->pool_fhs[bag_idx];
  if (*slot) return *slot;
  if (!r->shard) pthread_mutex_lock(&r->open_mutex);
  if (!*slot) {
    char path[1024];
    if (r->shard)
      snprintf(path, sizeof(path), "%s/positions/pp_bag%02d.w%d.txt", r->base_dir, bag,
               b->worker_index);
    else
      snprintf(path, sizeof(path), "%s/positions/pp_bag%02d.txt", r->base_dir, bag);
    FILE *fh = fopen(path, "w");
    if (!fh) log_fatal("trajectory_recorder: cannot open %s: %s", path, strerror(errno));
    *slot = fh;
  }
  if (!r->shard) pthread_mutex_unlock(&r->open_mutex);
  return *slot;
}

// One pool line per kept row: on-turn-first CGP <TAB> the opponent's leave (the
// previous row's leave when that row is the opponent's move on the previous turn --
// same as build_pool_full's (game, turn-1) join; "" otherwise).
static void tr_write_pool(TrajectoryRecorder *r, TrajectoryGameBuffer *b) {
  if (r->pool <= 0) return;
  for (int i = 0; i < b->count; i++) {
    const StagedRow *row = &b->rows[i];
    const int bi = row->bag_idx;
    if (!row->pool_cgp || !row->mover7 || !r->pool_has[bi]) continue;
    const int tail = row->lead <= r->pool_lo[bi] || row->lead >= r->pool_hi[bi];
    if (!tail) {
      const double u = (double)(tr_fnv1a(row->pool_cgp) >> 11) * (1.0 / 9007199254740992.0);
      if (u >= r->pool_p[bi]) continue;
    }
    const char *opp = "";
    if (i > 0 && b->rows[i - 1].on_turn != row->on_turn &&
        b->rows[i - 1].turn == row->turn - 1)
      opp = b->rows[i - 1].leave;
    const int bag = bi + TR_BAG_MIN;
    FILE *fh = tr_get_pool_file(r, b, bi, bag);
    if (!r->shard) pthread_mutex_lock(&r->mutexes[bi]);
    fputs(row->pool_cgp, fh);
    fputc('\t', fh);
    fputs(opp, fh);
    fputc('\n', fh);
    if (!r->shard) pthread_mutex_unlock(&r->mutexes[bi]);
  }
}

void trajectory_game_buffer_commit(
    TrajectoryRecorder *r, TrajectoryGameBuffer *b,
    int final_p1_score, int final_p2_score) {
  if (!r || !b) return;
  tr_write_pool(r, b);
  if (r->pool == 1) {   // pool lines only, no CSV rows
    tr_buf_free_rows(b);
    return;
  }
  // Per-row eventual_outcome (from row->on_turn player's perspective):
  //   1 = on_turn player won, 0 = lost, 2 = tied.
  // Derive once per game; per-row routing only needs row->on_turn.
  const int p1_outcome = (final_p1_score > final_p2_score) ? 1
                       : (final_p1_score < final_p2_score) ? 0
                                                            : 2;
  const int p2_outcome = (p1_outcome == 2) ? 2 : (1 - p1_outcome);
  for (int i = 0; i < b->count; i++) {
    const int bag_idx = b->rows[i].bag_idx;
    const int bag = (bag_idx == TR_BAG_MAX - TR_BAG_MIN + 1)
                       ? TR_OPENER_BAG
                       : bag_idx + TR_BAG_MIN;
    const int outcome = b->rows[i].on_turn ? p2_outcome : p1_outcome;
    // Final game-end spread from the row's on_turn player's perspective
    // (positive = that player finished ahead). Same frame as `outcome`.
    const int spread = b->rows[i].on_turn ? (final_p2_score - final_p1_score)
                                          : (final_p1_score - final_p2_score);
    if (r->shard) {
      // Per-worker shard file — this worker owns it, no lock needed.
      FILE *fh = tr_get_own_file(r, b, bag_idx, bag);
      fputs(b->rows[i].line, fh);
      fprintf(fh, ",%d,%d,%d\n", outcome, spread, b->rows[i].pick);
    } else {
      FILE *fh = tr_get_file(r, bag_idx, bag);
      if (!fh) continue;
      pthread_mutex_lock(&r->mutexes[bag_idx]);
      fputs(b->rows[i].line, fh);
      fprintf(fh, ",%d,%d,%d\n", outcome, spread, b->rows[i].pick);
      pthread_mutex_unlock(&r->mutexes[bag_idx]);
    }
  }
  tr_buf_free_rows(b);
}

// A game stopped early (MAGPIE_TRAJECTORY_MIN_BAG): same rows and routing as
// trajectory_game_buffer_commit, but eventual_outcome = -1 and final_spread = 0 --
// there is no game end. Such rows feed position pools only; anything that reads
// outcomes (aggregate_trajectory_wl) must skip outcome -1.
void trajectory_game_buffer_commit_unfinished(TrajectoryRecorder *r,
                                              TrajectoryGameBuffer *b) {
  if (!r || !b) return;
  tr_write_pool(r, b);
  if (r->pool == 1) {   // pool lines only, no CSV rows
    tr_buf_free_rows(b);
    return;
  }
  for (int i = 0; i < b->count; i++) {
    const int bag_idx = b->rows[i].bag_idx;
    const int bag = (bag_idx == TR_BAG_MAX - TR_BAG_MIN + 1)
                       ? TR_OPENER_BAG
                       : bag_idx + TR_BAG_MIN;
    if (r->shard) {
      FILE *fh = tr_get_own_file(r, b, bag_idx, bag);
      fputs(b->rows[i].line, fh);
      fprintf(fh, ",-1,0,%d\n", b->rows[i].pick);
    } else {
      FILE *fh = tr_get_file(r, bag_idx, bag);
      if (!fh) continue;
      pthread_mutex_lock(&r->mutexes[bag_idx]);
      fputs(b->rows[i].line, fh);
      fprintf(fh, ",-1,0,%d\n", b->rows[i].pick);
      pthread_mutex_unlock(&r->mutexes[bag_idx]);
    }
  }
  tr_buf_free_rows(b);
}

void trajectory_game_buffer_discard(TrajectoryGameBuffer *b) {
  if (!b) return;
  tr_buf_free_rows(b);
}
