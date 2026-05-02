#include "empty_board_strata.h"

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/resource.h>

#include "../util/io_util.h"

// Dimensions: turn 0..6 (slot 0 unused), kind 0..2, leave_len 0..7,
// type 0..3. Total file slots = 7*3*8*4 = 672. Most never opened.
#define EBS_NUM_TURNS 7  // index 1..6
#define EBS_NUM_KINDS 3  // 0=pass, 1=exch, 2=play
#define EBS_NUM_LL    8  // 0..7
#define EBS_NUM_TYPES 4  // 0=all, 1=cons, 2=vowel, 3=mixed

static const char *EBS_TYPE_NAMES[EBS_NUM_TYPES] = {"all", "cons", "vowel",
                                                    "mixed"};

struct EmptyBoardStrataRecorder {
  char *base_dir;
  pthread_mutex_t open_mutex;  // protects lazy-open path
  pthread_mutex_t mutexes[EBS_NUM_TURNS][EBS_NUM_KINDS][EBS_NUM_LL]
                         [EBS_NUM_TYPES];
  FILE *fhs[EBS_NUM_TURNS][EBS_NUM_KINDS][EBS_NUM_LL][EBS_NUM_TYPES];
};

static void ebs_mkdir_or_die(const char *path) {
  if (mkdir(path, 0775) != 0 && errno != EEXIST) {
    log_fatal("error creating directory: %s (%s)", path, strerror(errno));
  }
}

// Bump RLIMIT_NOFILE in case the per-(turn,kind,ll,type) file count plus
// the FJ recorder etc. crosses the default soft limit (1024).
static void ebs_raise_nofile_limit(void) {
  struct rlimit rl;
  if (getrlimit(RLIMIT_NOFILE, &rl) != 0) return;
  rlim_t want = 8192;
  if (rl.rlim_max != RLIM_INFINITY && want > rl.rlim_max) want = rl.rlim_max;
  if (rl.rlim_cur < want) {
    rl.rlim_cur = want;
    if (setrlimit(RLIMIT_NOFILE, &rl) != 0) {
      log_fatal("error raising RLIMIT_NOFILE soft to %llu: %s",
                (unsigned long long)want, strerror(errno));
    }
  }
}

EmptyBoardStrataRecorder *empty_board_strata_create(const char *base_dir) {
  if (!base_dir || !base_dir[0]) return NULL;
  ebs_raise_nofile_limit();
  EmptyBoardStrataRecorder *r = malloc_or_die(sizeof(EmptyBoardStrataRecorder));
  r->base_dir = malloc_or_die(strlen(base_dir) + 1);
  strcpy(r->base_dir, base_dir);
  pthread_mutex_init(&r->open_mutex, NULL);
  for (int T = 0; T < EBS_NUM_TURNS; T++) {
    for (int K = 0; K < EBS_NUM_KINDS; K++) {
      for (int L = 0; L < EBS_NUM_LL; L++) {
        for (int ty = 0; ty < EBS_NUM_TYPES; ty++) {
          pthread_mutex_init(&r->mutexes[T][K][L][ty], NULL);
          r->fhs[T][K][L][ty] = NULL;
        }
      }
    }
  }
  ebs_mkdir_or_die(r->base_dir);
  fprintf(stderr,
          "empty_board_strata: base_dir=%s; lazy per-(turn,kind,ll,type) "
          "files\n",
          r->base_dir);
  return r;
}

void empty_board_strata_destroy(EmptyBoardStrataRecorder *r) {
  if (!r) return;
  for (int T = 0; T < EBS_NUM_TURNS; T++) {
    for (int K = 0; K < EBS_NUM_KINDS; K++) {
      for (int L = 0; L < EBS_NUM_LL; L++) {
        for (int ty = 0; ty < EBS_NUM_TYPES; ty++) {
          if (r->fhs[T][K][L][ty]) {
            fflush(r->fhs[T][K][L][ty]);
            fclose(r->fhs[T][K][L][ty]);
          }
          pthread_mutex_destroy(&r->mutexes[T][K][L][ty]);
        }
      }
    }
  }
  pthread_mutex_destroy(&r->open_mutex);
  free(r->base_dir);
  free(r);
}

// cons/vowel/mixed for 3-6 tile leaves, else "all". Y as consonant; blank
// (?) ignored in the count.
static int ebs_classify_type(const char *leave, int leave_len) {
  if (leave_len < 3 || leave_len > 6) return 0;  // "all"
  int vc = 0, cc = 0;
  for (int i = 0; i < leave_len; i++) {
    char c = leave[i];
    if (c == '?') continue;
    if (c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U') vc++;
    else cc++;
  }
  if (vc == 0) return 1;   // cons
  if (cc == 0) return 2;   // vowel
  return 3;                 // mixed
}

// Lazy-open the per-cell file. Double-check pattern under r->open_mutex
// so two threads racing to open the same file don't end up with two FHs.
static FILE *ebs_get_file(EmptyBoardStrataRecorder *r, int T, int K, int L,
                          int ty) {
  if (r->fhs[T][K][L][ty]) return r->fhs[T][K][L][ty];
  pthread_mutex_lock(&r->open_mutex);
  if (!r->fhs[T][K][L][ty]) {
    char dir1[256], dir2[256], dir3[256], path[512];
    snprintf(dir1, sizeof(dir1), "%s/T%d", r->base_dir, T);
    ebs_mkdir_or_die(dir1);
    snprintf(dir2, sizeof(dir2), "%s/K%d", dir1, K);
    ebs_mkdir_or_die(dir2);
    snprintf(dir3, sizeof(dir3), "%s/L%d", dir2, L);
    ebs_mkdir_or_die(dir3);
    snprintf(path, sizeof(path), "%s/%s.csv", dir3, EBS_TYPE_NAMES[ty]);
    FILE *fh = fopen(path, "w");
    if (!fh) {
      log_fatal("empty_board_strata: cannot open %s (%s)", path,
                strerror(errno));
    }
    fprintf(fh,
            "pair_id,branch_id,rack,opp_action_history,action_repr,"
            "action_size,leave,eventual_outcome,p2_rack_source,"
            "natural_slot,move_score\n");
    fflush(fh);
    r->fhs[T][K][L][ty] = fh;
  }
  pthread_mutex_unlock(&r->open_mutex);
  return r->fhs[T][K][L][ty];
}

void empty_board_strata_write(
    EmptyBoardStrataRecorder *r, uint64_t pair_id, uint64_t branch_id,
    int turn_on_empty_board, const char *rack,
    const char *opp_action_history, int action_kind, const char *action_repr,
    int action_size, const char *leave, int eventual_outcome,
    int p2_rack_source, int natural_slot, int move_score) {
  if (!r) return;
  if (turn_on_empty_board < 1 || turn_on_empty_board >= EBS_NUM_TURNS) return;
  if (action_kind < 0 || action_kind >= EBS_NUM_KINDS) return;
  const int leave_len = leave ? (int)strlen(leave) : 0;
  if (leave_len < 0 || leave_len >= EBS_NUM_LL) return;
  const int type_idx = ebs_classify_type(leave ? leave : "", leave_len);
  FILE *fh = ebs_get_file(r, turn_on_empty_board, action_kind, leave_len,
                          type_idx);
  if (!fh) return;
  pthread_mutex_lock(&r->mutexes[turn_on_empty_board][action_kind][leave_len]
                                [type_idx]);
  fprintf(fh, "%llu,%llu,%s,%s,%s,%d,%s,%d,%d,%d,%d\n",
          (unsigned long long)pair_id, (unsigned long long)branch_id,
          rack ? rack : "",
          opp_action_history ? opp_action_history : "",
          action_repr ? action_repr : "", action_size,
          leave ? leave : "", eventual_outcome, p2_rack_source, natural_slot,
          move_score);
  pthread_mutex_unlock(&r->mutexes[turn_on_empty_board][action_kind]
                                  [leave_len][type_idx]);
}
