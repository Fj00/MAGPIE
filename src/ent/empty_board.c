#include "empty_board.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#include "../util/io_util.h"

struct EmptyBoardRecorder {
  FILE *out_file;
  pthread_mutex_t mutex;
};

EmptyBoardRecorder *empty_board_recorder_create(const char *out_path) {
  if (!out_path || !out_path[0]) return NULL;
  FILE *out = fopen(out_path, "w");
  if (!out) {
    fprintf(stderr, "empty_board: cannot open output %s\n", out_path);
    return NULL;
  }
  fprintf(out,
          "pair_id,branch_id,turn_on_empty_board,rack,"
          "opp_action_history,action_kind,action_repr,action_size,leave,"
          "eventual_outcome,p2_rack_source,natural_slot\n");
  fflush(out);

  EmptyBoardRecorder *r = malloc_or_die(sizeof(EmptyBoardRecorder));
  r->out_file = out;
  pthread_mutex_init(&r->mutex, NULL);
  fprintf(stderr, "empty_board: recorder output -> %s\n", out_path);
  return r;
}

void empty_board_recorder_destroy(EmptyBoardRecorder *r) {
  if (!r) return;
  if (r->out_file) {
    fflush(r->out_file);
    fclose(r->out_file);
  }
  pthread_mutex_destroy(&r->mutex);
  free(r);
}

void empty_board_recorder_write(
    EmptyBoardRecorder *r, uint64_t pair_id, uint64_t branch_id,
    int turn_on_empty_board, const char *rack,
    const char *opp_action_history, int action_kind, const char *action_repr,
    int action_size, const char *leave, int eventual_outcome,
    int p2_rack_source, int natural_slot) {
  if (!r) return;
  pthread_mutex_lock(&r->mutex);
  fprintf(r->out_file, "%llu,%llu,%d,%s,%s,%d,%s,%d,%s,%d,%d,%d\n",
          (unsigned long long)pair_id, (unsigned long long)branch_id,
          turn_on_empty_board, rack ? rack : "",
          opp_action_history ? opp_action_history : "", action_kind,
          action_repr ? action_repr : "", action_size, leave ? leave : "",
          eventual_outcome, p2_rack_source, natural_slot);
  pthread_mutex_unlock(&r->mutex);
}
