
#include "autoplay.h"

#include "../compat/cpthread.h"
#include "../compat/ctime.h"
#include "../def/autoplay_defs.h"
#include "../def/cpthread_defs.h"
#include "../def/game_defs.h"
#include "../def/game_history_defs.h"
#include "../def/letter_distribution_defs.h"
#include "../def/players_data_defs.h"
#include "../def/rack_defs.h"
#include "../def/thread_control_defs.h"
#include "../ent/autoplay_results.h"
#include "../ent/bag.h"
#include "../ent/board.h"
#include "../ent/checkpoint.h"
#include "../ent/data_filepaths.h"
#include "../ent/equity.h"
#include "../ent/force_table.h"
#include "../ent/game.h"
#include "../ent/empty_board.h"
#include "../ent/empty_board_strata.h"
#include "../ent/opening_pass.h"
#include "../ent/pass_cycle.h"
#include "../ent/rare_pool.h"
#include "../ent/inference_args.h"
#include "../ent/inference_results.h"
#include "../ent/klv.h"
#include "../ent/klv_csv.h"
#include "../ent/letter_distribution.h"
#include "../ent/move.h"
#include "../ent/player.h"
#include "../ent/players_data.h"
#include "../ent/rack.h"
#include "../ent/sim_results.h"
#include "../ent/thread_control.h"
#include "../ent/xoshiro.h"
#include "../str/game_string.h"
#include "../str/inference_string.h"
#include "../str/move_string.h"
#include "../str/sim_string.h"
#include "../util/io_util.h"
#include "../util/string_util.h"
#include "gameplay.h"
#include "move_gen.h"
#include "rack_list.h"
#include "simmer.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct LeavegenSharedData {
  int num_gens;
  int gens_completed;
  uint64_t gen_start_games;
  int *min_rack_targets;
  AutoplayResults *gen_autoplay_results;
  const LetterDistribution *ld;
  const char *data_paths;
  KLV *klv;
  RackList *rack_list;
  Checkpoint *postgen_checkpoint;
  AutoplayResults *primary_autoplay_results;
  AutoplayResults **autoplay_results_list;
} LeavegenSharedData;

typedef struct AutoplaySharedData {
  int num_threads;
  int print_interval;
  Timer timer;
  uint64_t max_iter_count;
  uint64_t seed;
  XoshiroPRNG *prng;
  uint64_t iter_count;
  cpthread_mutex_t iter_mutex;
  uint64_t iter_count_completed;
  cpthread_mutex_t iter_completed_mutex;
  ThreadControl *thread_control;
  LeavegenSharedData *leavegen_shared_data;
  ForceTable *force_table;
  bool stop_on_force_exhaust;
  OpeningPassTable *opening_pass_table;
  bool stop_on_opening_pass_complete;
  PassCycleTable *pass_cycle_table;
  // T6 rack-source pools — loaded for the 5-way 20% mix at the recording
  // turn (see per_turn_self_play_spec.md). Each game's T6 picks one of
  // {pass, exch, bingo, rare, play} uniformly; the on-turn player's rack
  // is returned to the bag and a rack is drawn from the chosen pool so
  // the recording fan-out (pass + exchanges + best-play + force-target
  // plays) covers that source's decision space.
  //
  // Env vars (each independent; missing pool → no inject for its 1/5):
  //   MAGPIE_EB_PASS_POOL=path.csv  — pass-favorable racks
  //                                   (format: rack,weight,is_pass)
  //   MAGPIE_EB_EXCH_POOL=path.csv  — exchange-favorable racks (same fmt)
  //   MAGPIE_EB_BINGO_POOL=path.csv — bingo-favorable racks (same fmt)
  //   MAGPIE_EB_PLAY_POOL=path.csv  — non-bingo play-favorable racks
  //                                   (same fmt) — replaces the natural
  //                                   "probability" fall-through with
  //                                   uniform play-class coverage.
  //   MAGPIE_EB_RARE_POOL=path.csv  — rare-cell-supporting racks
  //                                   (format: rack,length,type,kind,
  //                                   subleave,diff — multi-row per rack;
  //                                   built by build_rare_pool_with_cells.py;
  //                                   deficit-aware sampler at runtime)
  PassCycleTable *pass_pool;
  PassCycleTable *exch_pool;
  PassCycleTable *bingo_pool;
  PassCycleTable *play_pool;
  RarePool *rare_rack_cells;
  // 0 = P1 receives the T6 inject (recording), 1 = P2. Set by
  // MAGPIE_EB_RARE_TARGET=p1|p2. Default p2 (T6 is P2's turn in the
  // canonical EB cycle).
  int rare_target_player;
  EmptyBoardRecorder *empty_board_recorder;
  EmptyBoardStrataRecorder *empty_board_strata_recorder;
  // Slice 2: K-way fork branching at empty-board cycle-alive turns. Activated
  // by MAGPIE_EMPTY_BOARD_BRANCH=1. When set, play_autoplay_game_or_game_pair
  // dispatches a single-runner DFS that explores forks at branchable turns.
  bool eb_branch_active;
  // Bitmask of empty-board cycle turns at which to consult the force_table
  // and add matching forced moves as additional DFS slots (scope-B forcing).
  // Bit N set = active at turn N. Set by MAGPIE_EB_FORCE_TURNS env var
  // (comma-separated turn list, e.g. "6" or "5,6"). Requires force_table.
  int eb_force_turns_mask;
  // Single-target-turn recording: which turn (1..6) is the recording
  // target. All other turns are pass-cycle plumbing or post-target natural
  // play. Set by MAGPIE_EB_TARGET_TURN; default 6.
  int eb_target_turn;
  const LetterDistribution *ld;
} AutoplaySharedData;

typedef struct AutoplayIterOutput {
  uint64_t seed;
  uint64_t iter_count;
} AutoplayIterOutput;

typedef struct AutoplayIterCompletedOutput {
  uint64_t iter_count_completed;
  double time_elapsed;
  bool print_info;
} AutoplayIterCompletedOutput;

// Returns true if the iter_count is already greater than or equal to
// stop_iter_count and does nothing else.
// Returns false if the iter_count is less than the stop_iter_count
// and increments the iter_count and sets the next seed.
bool autoplay_get_next_iter_output(AutoplaySharedData *shared_data,
                                   AutoplayIterOutput *iter_output) {
  bool at_stop_count = false;
  cpthread_mutex_lock(&shared_data->iter_mutex);
  if (shared_data->iter_count >= shared_data->max_iter_count) {
    at_stop_count = true;
  } else if (shared_data->stop_on_force_exhaust &&
             force_table_is_exhausted(shared_data->force_table)) {
    at_stop_count = true;
  } else if (shared_data->stop_on_opening_pass_complete &&
             opening_pass_table_is_complete(shared_data->opening_pass_table)) {
    at_stop_count = true;
  } else {
    iter_output->seed = prng_next(shared_data->prng);
    iter_output->iter_count = shared_data->iter_count++;
  }
  cpthread_mutex_unlock(&shared_data->iter_mutex);
  return at_stop_count;
}

// This function should be called when a thread has completed computation
// for an iteration given by autoplay_get_next_iter_output.
// It increments the count completed and records the elapsed time.
void autoplay_complete_iter(
    AutoplaySharedData *shared_data,
    AutoplayIterCompletedOutput *iter_completed_output) {
  cpthread_mutex_lock(&shared_data->iter_completed_mutex);
  // Update internal fields
  shared_data->iter_count_completed++;
  // Set output
  iter_completed_output->iter_count_completed =
      shared_data->iter_count_completed;
  iter_completed_output->time_elapsed =
      ctimer_elapsed_seconds(&shared_data->timer);
  iter_completed_output->print_info =
      shared_data->print_interval > 0 &&
      shared_data->iter_count_completed % shared_data->print_interval == 0;
  cpthread_mutex_unlock(&shared_data->iter_completed_mutex);
}

// Copies the thread control PRNG to the other PRNG and performs a PRNG
// jump on the thread control PRNG.
void autoplay_shared_data_copy_to_dst_and_jump(AutoplaySharedData *shared_data,
                                               XoshiroPRNG *dst) {
  prng_copy(dst, shared_data->prng);
  prng_jump(shared_data->prng);
}

void postgen_prebroadcast_func(void *data) {
  AutoplaySharedData *shared_data = (AutoplaySharedData *)data;
  LeavegenSharedData *lg_shared_data = shared_data->leavegen_shared_data;
  rack_list_write_to_klv(lg_shared_data->rack_list, lg_shared_data->ld,
                         lg_shared_data->klv);
  lg_shared_data->gens_completed++;

  // Write the KLV for the current generation.
  char *label = get_formatted_string("_gen_%d", lg_shared_data->gens_completed);
  char *gen_labeled_klv_name =
      insert_before_dot(lg_shared_data->klv->name, label);

  ErrorStack *error_stack = error_stack_create();

  char *gen_labeled_klv_filename = data_filepaths_get_writable_filename(
      lg_shared_data->data_paths, gen_labeled_klv_name, DATA_FILEPATH_TYPE_KLV,
      error_stack);
  char *leaves_filename = data_filepaths_get_writable_filename(
      lg_shared_data->data_paths, gen_labeled_klv_name,
      DATA_FILEPATH_TYPE_LEAVES, error_stack);

  if (!error_stack_is_empty(error_stack)) {
    error_stack_print_and_reset(error_stack);
    log_fatal("leavegen failed to write results to file");
  }

  klv_write(lg_shared_data->klv, lg_shared_data->data_paths,
            gen_labeled_klv_name, error_stack);
  if (!error_stack_is_empty(error_stack)) {
    error_stack_print_and_reset(error_stack);
    log_fatal("leavegen failed to write klv to file: %s",
              gen_labeled_klv_filename);
  }

  klv_write_to_csv(lg_shared_data->klv, lg_shared_data->ld,
                   lg_shared_data->data_paths, gen_labeled_klv_name, NULL,
                   error_stack);
  if (!error_stack_is_empty(error_stack)) {
    error_stack_print_and_reset(error_stack);
    log_fatal("leavegen failed to write klv to CSV");
  }

  // Get total game data.
  autoplay_results_consolidate(lg_shared_data->autoplay_results_list,
                               shared_data->num_threads,
                               lg_shared_data->primary_autoplay_results);

  // Get generational game data
  autoplay_results_reset(lg_shared_data->gen_autoplay_results);
  autoplay_results_consolidate(lg_shared_data->autoplay_results_list,
                               shared_data->num_threads,
                               lg_shared_data->gen_autoplay_results);

  for (int i = 0; i < shared_data->num_threads; i++) {
    autoplay_results_reset(lg_shared_data->autoplay_results_list[i]);
  }

  // Print info about the current state.
  StringBuilder *leave_gen_sb = string_builder_create();

  string_builder_add_string(
      leave_gen_sb, "************************\n"
                    "Cumulative Autoplay Data\n************************\n\n");

  string_builder_add_formatted_string(
      leave_gen_sb, "Seconds: %f\n",
      ctimer_elapsed_seconds(&shared_data->timer));
  char *cumul_game_data_str = autoplay_results_to_string(
      lg_shared_data->primary_autoplay_results, true, false);
  string_builder_add_string(leave_gen_sb, cumul_game_data_str);
  free(cumul_game_data_str);

  string_builder_add_string(
      leave_gen_sb,
      "\n**************************\n"
      "Generational Autoplay Data\n**************************\n\n");

  char *gen_game_data_str = autoplay_results_to_string(
      lg_shared_data->gen_autoplay_results, true, false);
  string_builder_add_string(leave_gen_sb, gen_game_data_str);
  free(gen_game_data_str);

  string_builder_add_formatted_string(
      leave_gen_sb,
      "\nTarget Minimum "
      "Leave "
      "Count: %d\nLeaves Under "
      "Target Minimum Leave Count: %d\n\n",
      rack_list_get_target_rack_count(lg_shared_data->rack_list),
      rack_list_get_racks_below_target_count(lg_shared_data->rack_list));

  char *report_name_prefix =
      cut_off_after_last_char(gen_labeled_klv_filename, '.');
  char *report_name = get_formatted_string("%s_report.txt", report_name_prefix);

  write_string_to_file(report_name, "w", string_builder_peek(leave_gen_sb),
                       error_stack);
  if (!error_stack_is_empty(error_stack)) {
    error_stack_print_and_reset(error_stack);
    log_fatal("leavegen failed to write result summary to file");
  }

  string_builder_destroy(leave_gen_sb);
  error_stack_destroy(error_stack);

  free(report_name);
  free(report_name_prefix);
  free(gen_labeled_klv_filename);
  free(gen_labeled_klv_name);
  free(label);
  free(leaves_filename);

  // Reset data for the next generation.
  if (lg_shared_data->gens_completed < lg_shared_data->num_gens) {
    rack_list_reset(
        lg_shared_data->rack_list,
        lg_shared_data->min_rack_targets[lg_shared_data->gens_completed]);
    lg_shared_data->gen_start_games = shared_data->iter_count;
  }
}

typedef struct AutoplayWorker {
  int worker_index;
  AutoplayArgs args;
  AutoplayResults *autoplay_results;
  AutoplaySharedData *shared_data;
  XoshiroPRNG *prng;
  int *min_rack_targets;
  SimCtx *sim_ctx;
  SimResults *sim_results;
  InferenceResults *inference_results;
  ErrorStack *error_stack;
  Rack target_played_tiles;
  Rack nontarget_known_rack;
  Rack target_known_rack;
  MoveList *move_lists[2];
  // Used only when a force table is active. Holds the full move list with
  // MOVE_RECORD_ALL so we can filter by leave profile.
  MoveList *force_move_list;
  // Large-capacity move list for empty-board cycle action enumeration. Per-
  // player move_lists are sized to num_plays (often 1), so MOVE_RECORD_ALL
  // there would prune to top-1. This list is sized to fit all moves at any
  // empty-board position (~127 exchanges + ~500 plays).
  MoveList *eb_move_list;
  // Pre-computed leaves per move in force_move_list, populated once per turn
  // before the target-matching loop. Avoids 46M get_leave_for_move calls/turn
  // in the worst case (B1 optimization).
  Rack *force_leaves;
  uint32_t *force_leave_bitmaps;
  int force_leaves_capacity;
} AutoplayWorker;

AutoplayWorker *autoplay_worker_create(const AutoplayArgs *args,
                                       const AutoplayResults *target,
                                       int worker_index,
                                       AutoplaySharedData *shared_data) {
  AutoplayWorker *autoplay_worker = malloc_or_die(sizeof(AutoplayWorker));
  autoplay_worker->args = *args;
  // 0 plies indicate that the player is using static equity, so the move list
  // only needs a capacity of 1
  if (autoplay_worker->args.p1_sim_args.num_plays == 0) {
    autoplay_worker->args.p1_sim_args.num_plays = 1;
  }
  if (autoplay_worker->args.p2_sim_args.num_plays == 0) {
    autoplay_worker->args.p2_sim_args.num_plays = 1;
  }
  autoplay_worker->worker_index = worker_index;
  autoplay_worker->autoplay_results =
      autoplay_results_create_empty_copy(target);
  autoplay_worker->prng = NULL;
  if (shared_data->leavegen_shared_data) {
    autoplay_worker->prng = prng_create(0);
    autoplay_shared_data_copy_to_dst_and_jump(shared_data,
                                              autoplay_worker->prng);
  }
  autoplay_worker->shared_data = shared_data;
  autoplay_worker->sim_ctx = NULL;
  const AutoplayArgs *ap_args = &autoplay_worker->args;

  autoplay_worker->move_lists[0] =
      move_list_create(ap_args->p1_sim_args.num_plays);
  autoplay_worker->move_lists[1] =
      move_list_create(ap_args->p2_sim_args.num_plays);
  autoplay_worker->force_move_list = NULL;
  autoplay_worker->force_leaves = NULL;
  autoplay_worker->force_leave_bitmaps = NULL;
  autoplay_worker->force_leaves_capacity = 0;
  if (shared_data->force_table) {
    int cap = 32768;
    const char *cap_env = getenv("MAGPIE_FORCE_MOVE_SCAN_CAP");
    if (cap_env != NULL) {
      int v = atoi(cap_env);
      if (v >= 100 && v <= 1000000) cap = v;
    }
    autoplay_worker->force_move_list = move_list_create(cap);
    autoplay_worker->force_leaves_capacity = cap;
    autoplay_worker->force_leaves =
        malloc_or_die(sizeof(Rack) * (size_t)cap);
    autoplay_worker->force_leave_bitmaps =
        malloc_or_die(sizeof(uint32_t) * (size_t)cap);
  }
  autoplay_worker->eb_move_list = NULL;
  if (shared_data->eb_branch_active) {
    // Sized to match force_leaves_capacity when force-table is active so the
    // T6 force-target append can consider plays past the top-2000 by equity
    // (rare-leave plays often sit much deeper in the equity-sorted list).
    int eb_cap =
        autoplay_worker->force_leaves_capacity > 0 ? autoplay_worker->force_leaves_capacity : 2000;
    autoplay_worker->eb_move_list = move_list_create(eb_cap);
  }

  autoplay_worker->sim_results = NULL;
  autoplay_worker->inference_results = NULL;
  autoplay_worker->error_stack = NULL;

  // Only allocate sim structs if at least one of the players running a sim.
  if (ap_args->p1_sim_args.num_plies > 0 ||
      ap_args->p2_sim_args.num_plies > 0) {
    autoplay_worker->sim_results = sim_results_create(ap_args->cutoff);
    autoplay_worker->inference_results = inference_results_create(NULL);
    autoplay_worker->error_stack = error_stack_create();
    rack_set_dist_size_and_reset(&autoplay_worker->target_played_tiles,
                                 ld_get_size(ap_args->game_args->ld));
    rack_set_dist_size_and_reset(&autoplay_worker->nontarget_known_rack,
                                 ld_get_size(ap_args->game_args->ld));
    rack_set_dist_size_and_reset(&autoplay_worker->target_known_rack,
                                 ld_get_size(ap_args->game_args->ld));
  }

  return autoplay_worker;
}

void autoplay_worker_destroy(AutoplayWorker *autoplay_worker) {
  if (!autoplay_worker) {
    return;
  }
  autoplay_results_destroy(autoplay_worker->autoplay_results);
  prng_destroy(autoplay_worker->prng);
  sim_ctx_destroy(autoplay_worker->sim_ctx);
  sim_results_destroy(autoplay_worker->sim_results);
  inference_results_destroy(autoplay_worker->inference_results);
  error_stack_destroy(autoplay_worker->error_stack);
  move_list_destroy(autoplay_worker->move_lists[0]);
  move_list_destroy(autoplay_worker->move_lists[1]);
  move_list_destroy(autoplay_worker->force_move_list);
  move_list_destroy(autoplay_worker->eb_move_list);
  free(autoplay_worker->force_leaves);
  free(autoplay_worker->force_leave_bitmaps);
  free(autoplay_worker);
}

LeavegenSharedData *leavegen_shared_data_create(
    AutoplayResults *primary_autoplay_results,
    AutoplayResults **autoplay_results_list, const LetterDistribution *ld,
    const char *data_paths, KLV *klv, int number_of_threads, int num_gens,
    int *min_rack_targets) {
  LeavegenSharedData *shared_data = malloc_or_die(sizeof(LeavegenSharedData));

  shared_data->num_gens = num_gens;
  shared_data->gens_completed = 0;
  shared_data->gen_start_games = 0;
  shared_data->klv = klv;
  shared_data->gen_autoplay_results =
      autoplay_results_create_empty_copy(primary_autoplay_results);
  shared_data->primary_autoplay_results = primary_autoplay_results;
  shared_data->autoplay_results_list = autoplay_results_list;
  shared_data->ld = ld;
  shared_data->data_paths = data_paths;
  shared_data->min_rack_targets = min_rack_targets;
  shared_data->rack_list = rack_list_create(ld, min_rack_targets[0]);
  shared_data->postgen_checkpoint =
      checkpoint_create(number_of_threads, postgen_prebroadcast_func);
  return shared_data;
}

// Use NULL for the KLV when not running in leave gen mode.
AutoplaySharedData *
autoplay_shared_data_create(const AutoplayArgs *args, int num_autoplay_threads,
                            const uint64_t first_gen_num_games,
                            AutoplayResults *primary_autoplay_results,
                            AutoplayResults **autoplay_results_list, KLV *klv,
                            int num_gens, int *min_rack_targets) {
  AutoplaySharedData *shared_data = malloc_or_die(sizeof(AutoplaySharedData));
  shared_data->num_threads = num_autoplay_threads;
  shared_data->print_interval = args->print_interval;
  ctimer_start(&shared_data->timer);
  shared_data->max_iter_count = first_gen_num_games;
  shared_data->seed = args->seed;
  shared_data->prng = prng_create(args->seed);
  shared_data->iter_count = 0;
  cpthread_mutex_init(&shared_data->iter_mutex);
  shared_data->iter_count_completed = 0;
  cpthread_mutex_init(&shared_data->iter_completed_mutex);
  shared_data->thread_control = args->thread_control;
  shared_data->leavegen_shared_data = NULL;
  if (klv) {
    shared_data->leavegen_shared_data = leavegen_shared_data_create(
        primary_autoplay_results, autoplay_results_list, args->game_args->ld,
        args->data_paths, klv, num_autoplay_threads, num_gens,
        min_rack_targets);
  }
  shared_data->force_table = NULL;
  shared_data->ld = args->game_args->ld;
  const char *ft_path = args->force_table_path;
  if (!ft_path || ft_path[0] == '\0') {
    // Fallback for testing before CLI flag is wired: env var.
    ft_path = getenv("MAGPIE_FORCE_TABLE");
  }
  if (ft_path && ft_path[0] != '\0') {
    shared_data->force_table =
        force_table_create(ft_path, args->game_args->ld);
  }
  shared_data->stop_on_force_exhaust =
      shared_data->force_table != NULL &&
      getenv("MAGPIE_FORCE_STOP_ON_EXHAUST") != NULL;

  shared_data->opening_pass_table = NULL;
  const char *op_path = getenv("MAGPIE_OPENING_PASS_TABLE");
  if (op_path && op_path[0] != '\0') {
    shared_data->opening_pass_table =
        opening_pass_table_create(op_path, args->game_args->ld);
    const char *resume_path = getenv("MAGPIE_OPENING_PASS_RESUME");
    if (shared_data->opening_pass_table && resume_path &&
        resume_path[0] != '\0') {
      opening_pass_table_resume(shared_data->opening_pass_table, resume_path);
    }
  }
  shared_data->stop_on_opening_pass_complete =
      shared_data->opening_pass_table != NULL &&
      getenv("MAGPIE_OPENING_PASS_STOP_ON_COMPLETE") != NULL;

  shared_data->pass_cycle_table = NULL;
  const char *pc_racks = getenv("MAGPIE_PASS_CYCLE_RACKS");
  const char *pc_out = getenv("MAGPIE_PASS_CYCLE_OUT");
  if (pc_racks && pc_racks[0] != '\0' && pc_out && pc_out[0] != '\0') {
    shared_data->pass_cycle_table =
        pass_cycle_table_create(pc_racks, pc_out);
  }

  shared_data->pass_pool = NULL;
  shared_data->exch_pool = NULL;
  shared_data->bingo_pool = NULL;
  shared_data->play_pool = NULL;
  shared_data->rare_rack_cells = NULL;
  // Parse target turn first; rare_target_player is derived from its parity.
  shared_data->eb_target_turn = 6;
  const char *eb_target = getenv("MAGPIE_EB_TARGET_TURN");
  if (eb_target && eb_target[0] != '\0') {
    const int t = atoi(eb_target);
    if (t >= 1 && t <= 6) {
      shared_data->eb_target_turn = t;
    } else {
      fprintf(stderr,
              "empty_board: invalid MAGPIE_EB_TARGET_TURN=%s (must be 1..6); "
              "using default 6\n", eb_target);
    }
  }
  // Recording player parity: T_odd = P1 (index 0), T_even = P2 (index 1).
  shared_data->rare_target_player =
      (shared_data->eb_target_turn % 2 == 1) ? 0 : 1;
  fprintf(stderr, "empty_board: target turn = T%d (recording player = P%d)\n",
          shared_data->eb_target_turn,
          shared_data->rare_target_player + 1);
  const char *pp = getenv("MAGPIE_EB_PASS_POOL");
  if (pp && pp[0] != '\0') {
    shared_data->pass_pool = pass_cycle_table_create(pp, "/dev/null");
  }
  const char *ep = getenv("MAGPIE_EB_EXCH_POOL");
  if (ep && ep[0] != '\0') {
    shared_data->exch_pool = pass_cycle_table_create(ep, "/dev/null");
  }
  const char *bp = getenv("MAGPIE_EB_BINGO_POOL");
  if (bp && bp[0] != '\0') {
    shared_data->bingo_pool = pass_cycle_table_create(bp, "/dev/null");
  }
  const char *plp = getenv("MAGPIE_EB_PLAY_POOL");
  if (plp && plp[0] != '\0') {
    shared_data->play_pool = pass_cycle_table_create(plp, "/dev/null");
  }
  const char *rare_pool = getenv("MAGPIE_EB_RARE_POOL");
  if (rare_pool && rare_pool[0] != '\0' && shared_data->force_table) {
    shared_data->rare_rack_cells = rare_pool_create(
        rare_pool, shared_data->force_table, shared_data->ld);
  }
  if (shared_data->pass_pool || shared_data->exch_pool ||
      shared_data->bingo_pool || shared_data->play_pool ||
      shared_data->rare_rack_cells) {
    fprintf(stderr,
            "empty_board: T%d source mix ENABLED (recording=P%d) — pools: "
            "pass=%s exch=%s bingo=%s play=%s rare=%s\n",
            shared_data->eb_target_turn,
            shared_data->rare_target_player + 1,
            shared_data->pass_pool ? "yes" : "no",
            shared_data->exch_pool ? "yes" : "no",
            shared_data->bingo_pool ? "yes" : "no",
            shared_data->play_pool ? "yes" : "no",
            shared_data->rare_rack_cells ? "yes" : "no");
  }

  shared_data->empty_board_recorder = NULL;
  const char *eb_out = getenv("MAGPIE_EMPTY_BOARD_OUT");
  if (eb_out && eb_out[0] != '\0') {
    shared_data->empty_board_recorder = empty_board_recorder_create(eb_out);
  }
  shared_data->empty_board_strata_recorder = NULL;
  const char *eb_strata = getenv("MAGPIE_EMPTY_BOARD_STRATA");
  if (eb_strata && eb_strata[0] != '\0') {
    shared_data->empty_board_strata_recorder =
        empty_board_strata_create(eb_strata);
  }
  shared_data->eb_branch_active = false;
  const char *eb_branch = getenv("MAGPIE_EMPTY_BOARD_BRANCH");
  if (eb_branch && eb_branch[0] != '\0' && eb_branch[0] != '0') {
    shared_data->eb_branch_active = true;
    fprintf(stderr,
            "empty_board: K-way fork branching ENABLED (turns 3-5 if "
            "is_pass rack)\n");
  }
  shared_data->eb_force_turns_mask = 0;
  const char *eb_force = getenv("MAGPIE_EB_FORCE_TURNS");
  if (eb_force && eb_force[0] != '\0') {
    char buf[64];
    snprintf(buf, sizeof(buf), "%s", eb_force);
    char *tok = strtok(buf, ",");
    while (tok != NULL) {
      const int t = atoi(tok);
      if (t >= 2 && t <= 6) shared_data->eb_force_turns_mask |= (1 << t);
      tok = strtok(NULL, ",");
    }
    fprintf(stderr, "empty_board: force-table ACTIVE at turns mask 0x%x "
            "(input: %s)\n", shared_data->eb_force_turns_mask, eb_force);
  }
  return shared_data;
}

void leavegen_shared_data_destroy(LeavegenSharedData *lg_shared_data) {
  if (!lg_shared_data) {
    return;
  }
  rack_list_destroy(lg_shared_data->rack_list);
  checkpoint_destroy(lg_shared_data->postgen_checkpoint);
  autoplay_results_destroy(lg_shared_data->gen_autoplay_results);
  free(lg_shared_data);
}

void autoplay_shared_data_destroy(AutoplaySharedData *shared_data) {
  if (!shared_data) {
    return;
  }
  prng_destroy(shared_data->prng);
  leavegen_shared_data_destroy(shared_data->leavegen_shared_data);
  if (shared_data->force_table) {
    fprintf(stderr, "force_table: remaining deficit = %lld across %d targets\n",
            (long long)force_table_total_remaining(shared_data->force_table),
            force_table_num_targets(shared_data->force_table));
    // Dump updated deficits (can be passed back as the next run's input).
    const char *dump_path = getenv("MAGPIE_FORCE_TABLE_DUMP");
    if (dump_path && dump_path[0] != '\0') {
      force_table_dump_remaining(shared_data->force_table, dump_path,
                                 shared_data->ld);
    }
  }
  force_table_destroy(shared_data->force_table);
  if (shared_data->opening_pass_table) {
    fprintf(stderr,
            "opening_pass: %d racks; remaining target games = %lld\n",
            opening_pass_table_num_racks(shared_data->opening_pass_table),
            (long long)opening_pass_table_remaining(
                shared_data->opening_pass_table));
    const char *out_path = getenv("MAGPIE_OPENING_PASS_OUT");
    if (out_path && out_path[0] != '\0') {
      opening_pass_table_dump(shared_data->opening_pass_table, out_path);
    }
  }
  opening_pass_table_destroy(shared_data->opening_pass_table);
  pass_cycle_table_destroy(shared_data->pass_cycle_table);
  pass_cycle_table_destroy(shared_data->pass_pool);
  pass_cycle_table_destroy(shared_data->exch_pool);
  pass_cycle_table_destroy(shared_data->bingo_pool);
  pass_cycle_table_destroy(shared_data->play_pool);
  rare_pool_destroy(shared_data->rare_rack_cells);
  empty_board_recorder_destroy(shared_data->empty_board_recorder);
  empty_board_strata_destroy(shared_data->empty_board_strata_recorder);
  free(shared_data);
}

typedef struct GameRunner {
  bool force_draw;
  // True once any forced move has been picked in this game. While false and
  // force_table is active, wordstats recording for this game's turns is
  // suppressed (pre-force turns are "tainted" per the gen 2 spec).
  bool force_triggered;
  int turn_number;
  int pair_game_number; // 0 for non-paired games, 1 or 2 for game pairs
  uint64_t game_number;
  uint64_t seed;
  Game *game;
  // Used for inference args in autoplay with
  // inference
  Game *game_one_move_behind;
  Move previous_move;
  AutoplaySharedData *shared_data;
  // Set by try_forced_move when a force fires. The deficit decrement is
  // deferred to game-end (autoplay_add_game) so that stratum-kind targets
  // can be credited only when the game's outcome bumps min(wins, losses).
  ForceTarget *pending_force_target;
  int pending_force_diff; // forcing player's score - opponent's at force turn
  int pending_force_player_index; // 0 or 1
  // Opening-pass mode state. opening_pass_rack_idx >= 0 indicates this
  // game has a forced opening rack from the opening_pass_table.
  // opening_pass_branch: 0 = pass branch (force pass on opening turn),
  //                     1 = play branch (normal play, used as counterfactual).
  // opening_pass_player_index: which player has the forced rack (always
  // the starter, so the first move belongs to them).
  int opening_pass_rack_idx;
  int opening_pass_branch;
  int opening_pass_player_index;
  // Pass-cycle mode: both players' racks are forced; branch 0 = bot always
  // passes every turn, branch 1 = bot plays normally.
  bool pass_cycle_active;
  int pass_cycle_branch;
  int pass_cycle_bot_player;  // = starting_player_index (P1's seat for outcome)
  const char *pass_cycle_bot_rack_str;
  const char *pass_cycle_opp_rack_str;
  // Per-turn (rack, move) trace for downstream counterfactual reconstruction.
  // Each entry encodes "<rack>:<move>" where move is "P" or "X<tiles>".
  char pass_cycle_history[6][24];
  int pass_cycle_n_moves;
  // Set true once we know this game cannot be a recorded 6-pass-cycle game
  // (e.g. a tile was placed). Causes game_runner_is_game_over to short-circuit.
  bool pass_cycle_abandoned;

  // Empty-board / pass-cycle value sub-model recorder state (slice 1).
  // One snapshot per cycle-alive empty-board turn the player on turn faced;
  // emitted at game-end with eventual_outcome filled in.
  struct {
    int player_on_turn;       // 0 or 1
    int turn_on_empty_board;  // 1..6
    char rack[RACK_SIZE + 2];
    char opp_history[96];
    int action_kind;          // 0=pass, 1=exch, 2=play
    char action_repr[16];     // exch: tiles given back; play: tiles placed
    int action_size;
    char leave[RACK_SIZE + 2];  // post-action rack (canonical sorted, blanks
                                // last). Pass: leave == rack. Exch/play: rack
                                // minus action tiles, recovered via
                                // get_leave_for_move so the play case (where
                                // action_repr can't easily encode boardplay
                                // tiles like blank-as-letter) is reliable.
    int natural_slot;         // slot HastyBot/force-pass would have chosen
                              // at this fork (-1 if not a fork point)
    int move_score;           // points scored on this turn (0 for pass/exch)
  } eb_snaps[6];
  int eb_n_snaps;
  // Per-player action history accumulated this cycle (pipe-joined). Used as
  // opp_action_history when capturing the OTHER player's snapshot.
  char eb_actions_p0[96];
  int eb_actions_p0_off;
  char eb_actions_p1[96];
  int eb_actions_p1_off;
  // Slice 1: piggybacks on pass_cycle binary mode. True iff recorder is set
  // AND this game has pass_cycle_active.
  bool eb_active;

  // Slice 2 K-way fork DFS state.
  // eb_forced_move: when non-NULL, game_runner_play_move uses this exact move
  //                 instead of the normal selection logic. The DFS sets it
  //                 right before each branch's play_move call.
  // eb_action_buf:  pre-allocated Move slots for enumerate_eb_actions.
  // (Per-recursion-level game checkpoints are allocated dynamically via
  // game_duplicate inside play_eb_dfs — a shared buffer would be overwritten
  // by inner forks.)
  Move *eb_forced_move;
  // P1 rack source / forcing tag.
  //   eb_p1_rack_source: pinned to 0 in the new mode (P1 is always sampled
  //                      from the pool). Kept in the schema for downstream
  //                      analyses that group by rack source.
  //   eb_p1_force_kind:  0 = T1 force-pass (P1 rack is is_pass=1),
  //                      1 = T1 force-exchange (P1 rack is is_pass=0).
  //                      Drives pool rejection sampling AND the T1 force.
  // P2 rack source / forcing tag (new in the T2-balanced mode).
  //   eb_p2_rack_source: 0 = pool (rejection-sampled to match force_kind),
  //                      1 = bag (uniform draw — play-prone P2 racks for
  //                              the "best play is +4, exchange better?"
  //                              counterfactual).
  //   eb_p2_force_kind:  0 = T2 force-pass,
  //                      1 = T2 force-exchange.
  int eb_p1_rack_source;
  int eb_p1_force_kind;
  int eb_p2_rack_source;
  int eb_p2_force_kind;
  // Natural slot at the most recent enumerate_actions call: which slot
  // index HastyBot or force-pass would have chosen without DFS forcing.
  // -1 = no fork at this turn (or pass-cycle / opp-pass / etc. — non-fork).
  int eb_natural_slot;
  // Tracks "anchor" semantics — the leaf's first divergence from natural,
  // and whether every subsequent fork chose natural.
  //   eb_divergence_turn: -1 = chain has never diverged (pure natural);
  //     1..6 = the turn at which the chain FIRST chose a non-natural slot.
  //   eb_n_divergences: count of non-natural sibling choices in this
  //     branch's ancestry. A leaf is useful for analysis iff this is <=1
  //     (one or zero divergences = pure natural OR anchored-at-T_k); a
  //     leaf with >=2 divergences is in counterfactual-of-counterfactual
  //     territory and should be skipped.
  // Both are saved/restored across sibling iterations by eb_meta_save.
  int eb_divergence_turn;
  int eb_last_divergence_turn;  // turn of the LAST non-natural choice (-1 if none)
  int eb_n_divergences;
  // Slot 0=pass; slots 1..N=enumerated actions in stable order.
  // Capacity covers worst case: pass + 127 distinct exch subsets of a 7-tile
  // rack + 1 best-play + slack. eb_action_present[i] is true iff slot i is
  // populated this enumeration; the DFS skips false slots.
#define EB_MAX_ACTIONS 320
  Move *eb_action_buf[EB_MAX_ACTIONS];
  bool eb_action_present[EB_MAX_ACTIONS];
  int eb_n_action_buf;
  // Force-table target attached to each enumerated slot (NULL if natural).
  // Set by eb_append_force_target_slots; consumed in play_eb_dfs which
  // copies the chosen slot's target into eb_snap_force_target[turn].
  ForceTarget *eb_force_target_for_slot[EB_MAX_ACTIONS];
  // Per-snap force-target (indexed by snap turn 1..6). Set when DFS
  // descends into a forced slot; eb_emit_leaf reads to credit the deficit
  // per emitted snap. Saved/restored across sibling forks via EbMetaSave.
  ForceTarget *eb_snap_force_target[7];
} GameRunner;

GameRunner *game_runner_create(AutoplayWorker *autoplay_worker) {
  const AutoplayArgs *args = &autoplay_worker->args;
  GameRunner *game_runner = malloc_or_die(sizeof(GameRunner));
  game_runner->shared_data = autoplay_worker->shared_data;
  game_runner->game = game_create(args->game_args);
  game_runner->game_one_move_behind = NULL;
  if (args->p1_sim_args.use_inference || args->p2_sim_args.use_inference) {
    game_runner->game_one_move_behind = game_create(args->game_args);
  }
  game_runner->pair_game_number =
      0; // Will be set in game_runner_start if using pairs
  game_runner->eb_forced_move = NULL;
  game_runner->eb_p1_rack_source = 0;
  game_runner->eb_p1_force_kind = 0;
  game_runner->eb_p2_rack_source = 0;
  game_runner->eb_p2_force_kind = 0;
  game_runner->eb_natural_slot = -1;
  game_runner->eb_divergence_turn = -1;
  game_runner->eb_last_divergence_turn = -1;
  game_runner->eb_n_divergences = 0;
  game_runner->eb_n_action_buf = 0;
  for (int i = 0; i < EB_MAX_ACTIONS; i++) {
    game_runner->eb_action_buf[i] = NULL;
    game_runner->eb_force_target_for_slot[i] = NULL;
  }
  for (int i = 0; i < 7; i++) game_runner->eb_snap_force_target[i] = NULL;
  if (autoplay_worker->shared_data->eb_branch_active) {
    game_runner->eb_n_action_buf = EB_MAX_ACTIONS;
    for (int i = 0; i < EB_MAX_ACTIONS; i++) {
      game_runner->eb_action_buf[i] = move_create();
    }
  }
  return game_runner;
}

void game_runner_destroy(GameRunner *game_runner) {
  if (!game_runner) {
    return;
  }
  game_destroy(game_runner->game);
  game_destroy(game_runner->game_one_move_behind);
  for (int i = 0; i < game_runner->eb_n_action_buf; i++) {
    free(game_runner->eb_action_buf[i]);
  }
  free(game_runner);
}

void game_runner_start(AutoplayWorker *autoplay_worker, GameRunner *game_runner,
                       const AutoplayIterOutput *iter_output,
                       int starting_player_index, int pair_game_number) {
  Game *game = game_runner->game;
  game_reset(game);
  game_runner->seed = iter_output->seed;
  game_runner->game_number = iter_output->iter_count;
  game_runner->pair_game_number = pair_game_number;
  game_seed(game, iter_output->seed);
  autoplay_worker->args.p1_sim_args.seed = iter_output->seed;
  autoplay_worker->args.p2_sim_args.seed = iter_output->seed;
  game_set_starting_player_index(game, starting_player_index);

  // Opening-pass mode: force the starter's rack to a candidate from the
  // table; opponent draws normally from the remaining bag. Branch derived
  // from pair_game_number (1=pass, 2=play; 0 falls back to pass for
  // unpaired runs, though paired runs are the supported configuration).
  game_runner->opening_pass_rack_idx = -1;
  game_runner->opening_pass_branch = 0;
  game_runner->opening_pass_player_index = starting_player_index;
  bool used_forced_rack = false;
  if (game_runner->shared_data->opening_pass_table) {
    int rack_idx = -1;
    const char *rack_str = opening_pass_table_get_rack(
        game_runner->shared_data->opening_pass_table,
        iter_output->iter_count, &rack_idx);
    if (rack_str) {
      const int n = draw_rack_string_from_bag(game, starting_player_index,
                                              rack_str);
      if (n > 0) {
        draw_to_full_rack(game, 1 - starting_player_index);
        game_runner->opening_pass_rack_idx = rack_idx;
        game_runner->opening_pass_branch =
            (pair_game_number == 2) ? 1 : 0;
        used_forced_rack = true;
      }
    }
  }
  // Pass-cycle mode: force both racks. Bot gets a pass-favorable rack
  // (round-robin by iter_count); opp gets a weighted exchange-prone rack.
  // Both games in a pair use the same racks but different branches.
  game_runner->pass_cycle_active = false;
  game_runner->pass_cycle_branch = 0;
  game_runner->pass_cycle_bot_player = starting_player_index;
  game_runner->pass_cycle_bot_rack_str = NULL;
  game_runner->pass_cycle_opp_rack_str = NULL;
  game_runner->pass_cycle_n_moves = 0;
  game_runner->pass_cycle_abandoned = false;
  if (!used_forced_rack && game_runner->shared_data->pass_cycle_table) {
    PassCycleTable *pct = game_runner->shared_data->pass_cycle_table;
    const int p1 = starting_player_index;
    const int p2 = 1 - starting_player_index;
    // P1: always pool-sampled. Rejection-sample to match eb_p1_force_kind
    // so each (is_pass=1, is_pass=0) class is equally represented despite
    // the pool's natural skew.
    const int p1_target_is_pass =
        (game_runner->eb_p1_force_kind == 0) ? 1 : 0;
    const char *p1_rack = NULL;
    pass_cycle_sample_p1_target_is_pass(
        pct, iter_output->iter_count, p1_target_is_pass, &p1_rack);
    if (p1_rack != NULL) {
      const int n1 = draw_rack_string_from_bag(game, p1, p1_rack);
      if (n1 > 0) {
        // Override force_kind from the actual sampled rack's is_pass —
        // rejection sampling can fall back to an unfiltered sample.
        const int is_pass = pass_cycle_lookup_is_pass(pct, p1_rack);
        game_runner->eb_p1_force_kind = (is_pass == 1) ? 0 : 1;

        // P2 rack: pool (rejection-sampled by eb_p2_force_kind) or
        // bag-random per eb_p2_rack_source. Rare-rack injection happens
        // at T6 (the recording turn), not here — see
        // inject_rare_rack_at_recording_turn.
        bool p2_drawn = false;
        if (game_runner->eb_p2_rack_source == 0) {
          const int p2_target_is_pass =
              (game_runner->eb_p2_force_kind == 0) ? 1 : 0;
          const char *p2_rack = NULL;
          // Perturb the seed so P1 and P2 sample independently.
          pass_cycle_sample_p1_target_is_pass(
              pct, iter_output->iter_count ^ 0xa5a5a5a5a5a5a5a5ULL,
              p2_target_is_pass, &p2_rack);
          if (p2_rack != NULL) {
            const int n2 = draw_rack_string_from_bag(game, p2, p2_rack);
            if (n2 > 0) {
              game_runner->pass_cycle_opp_rack_str = p2_rack;
              p2_drawn = true;
            }
          }
        }
        if (!p2_drawn) {
          // Bag-random P2 (or pool sample failed → fall back to bag).
          draw_to_full_rack(game, p2);
        }
        game_runner->pass_cycle_active = true;
        game_runner->pass_cycle_branch = (pair_game_number == 2) ? 1 : 0;
        game_runner->pass_cycle_bot_player = p1;
        game_runner->pass_cycle_bot_rack_str = p1_rack;
        used_forced_rack = true;
      }
    }
  }
  if (!used_forced_rack) {
    draw_starting_racks(game);
  }

  // Empty-board recorder state (slice 1): active iff recorder configured AND
  // cycle entry happened via pass_cycle.
  game_runner->eb_n_snaps = 0;
  game_runner->eb_actions_p0[0] = '\0';
  game_runner->eb_actions_p0_off = 0;
  game_runner->eb_actions_p1[0] = '\0';
  game_runner->eb_actions_p1_off = 0;
  game_runner->eb_active =
      (game_runner->shared_data->empty_board_recorder != NULL ||
       game_runner->shared_data->empty_board_strata_recorder != NULL) &&
      game_runner->pass_cycle_active;
  game_runner->eb_forced_move = NULL;

  if (game_runner->game_one_move_behind) {
    Game *game_one_move_behind = game_runner->game_one_move_behind;
    game_reset(game_one_move_behind);
    game_seed(game_one_move_behind, iter_output->seed);
    game_set_starting_player_index(game_one_move_behind, starting_player_index);
    draw_starting_racks(game_one_move_behind);
  }

  game_runner->turn_number = 0;
  game_runner->force_draw = false;
  game_runner->pending_force_target = NULL;
  game_runner->pending_force_diff = 0;
  game_runner->pending_force_player_index = 0;
  // If every target is satisfied, treat the game as already-forced so all
  // its turns get recorded normally (unbiased hasty self-play).
  game_runner->force_triggered =
      game_runner->shared_data->force_table != NULL &&
      force_table_is_exhausted(game_runner->shared_data->force_table);
  if (game_runner->shared_data->leavegen_shared_data &&
      // We only force draws if we've played enough games for this
      // generation.
      (iter_output->iter_count -
       game_runner->shared_data->leavegen_shared_data->gen_start_games) >=
          (uint64_t)autoplay_worker->args.games_before_force_draw_start) {
    game_runner->force_draw = true;
  }
}

bool game_runner_is_game_over(GameRunner *game_runner) {
  // pass_cycle_abandoned is a fast-fail for games that can't satisfy the
  // 6-pass-cycle output recorder. But the EB strata recorder needs the
  // full game to play out so K=2 (play) branches get a meaningful
  // eventual_outcome — otherwise plays terminate the moment a tile lands
  // and trivially win 100% of the time.
  const bool eb_strata_active =
      game_runner->shared_data->empty_board_strata_recorder != NULL;
  return game_over(game_runner->game) ||
         (game_runner->shared_data->leavegen_shared_data &&
          bag_get_letters(game_get_bag(game_runner->game)) < (RACK_SIZE)) ||
         (game_runner->pass_cycle_abandoned && !eb_strata_active);
}

const Move *game_runner_get_top_simming_move(AutoplayWorker *autoplay_worker,
                                             GameRunner *game_runner) {
  Game *game = game_runner->game;
  const int player_on_turn_index = game_get_player_on_turn_index(game);
  MoveList *move_list = autoplay_worker->move_lists[player_on_turn_index];
  SimArgs *sim_args = (player_on_turn_index == 0)
                          ? &autoplay_worker->args.p1_sim_args
                          : &autoplay_worker->args.p2_sim_args;
  sim_args->move_list = move_list;
  sim_args->game = game_runner->game;

  const bool player_uses_inference = sim_args->use_inference;
  sim_args->use_inference =
      player_uses_inference && game_runner->turn_number > 0 &&
      move_get_type(&game_runner->previous_move) != GAME_EVENT_PASS;
  if (sim_args->use_inference) {
    InferenceArgs *infer_args = &sim_args->inference_args;
    // Set target played tiles
    rack_reset(&autoplay_worker->target_played_tiles);
    const int move_tiles_length =
        move_get_tiles_length(&game_runner->previous_move);
    if (move_get_type(&game_runner->previous_move) ==
        GAME_EVENT_TILE_PLACEMENT_MOVE) {
      for (int i = 0; i < move_tiles_length; i++) {
        if (move_get_tile(&game_runner->previous_move, i) !=
            PLAYED_THROUGH_MARKER) {
          if (get_is_blanked(move_get_tile(&game_runner->previous_move, i))) {
            rack_add_letter(&autoplay_worker->target_played_tiles,
                            BLANK_MACHINE_LETTER);
          } else {
            rack_add_letter(&autoplay_worker->target_played_tiles,
                            move_get_tile(&game_runner->previous_move, i));
          }
        }
      }
    }
    // Set nontarget known rack
    rack_copy(&autoplay_worker->nontarget_known_rack,
              player_get_rack(game_get_player(game, player_on_turn_index)));
    // The target known rack was set to empty when the autoplay worker was
    // created. It does not need to be modified after initial creation as it
    // will always be empty because autoplay does not support challenged phonies
    // (yet).
    infer_args_fill(
        infer_args, infer_args->leave_list_capacity, infer_args->equity_margin,
        infer_args->game_history, game_runner->game_one_move_behind,
        infer_args->num_threads, infer_args->parent_worker_thread_index,
        infer_args->print_interval, infer_args->thread_control,
        infer_args->use_game_history,
        infer_args->use_inference_cutoff_optimization,
        // We can use 1 - player_on_turn_index for the target index because
        // autoplay does not support challenged phonies (yet).
        1 - player_on_turn_index, move_get_score(&game_runner->previous_move),
        move_get_type(&game_runner->previous_move) == GAME_EVENT_EXCHANGE
            ? move_get_tiles_played(&game_runner->previous_move)
            : 0,
        &autoplay_worker->target_played_tiles,
        &autoplay_worker->target_known_rack,
        &autoplay_worker->nontarget_known_rack);
  }

  ErrorStack *error_stack = autoplay_worker->error_stack;
  const Move *move = get_top_simming_move(
      game, autoplay_worker->worker_index, move_list, sim_args,
      &autoplay_worker->sim_ctx, autoplay_worker->sim_results, error_stack);
  if (!error_stack_is_empty(error_stack)) {
    error_stack_print_and_reset(error_stack);
    log_fatal("autoplay worker %d failed to get top simming move for player %d "
              "on turn %d of game number %llu with seed %llu",
              autoplay_worker->worker_index, player_on_turn_index,
              game_runner->turn_number + 1,
              (unsigned long long)game_runner->game_number + 1,
              (unsigned long long)game_runner->seed);
  }
  sim_args->use_inference = player_uses_inference;
  return move;
}

// If a force target at the current bag count matches any candidate move,
// pick the highest-equity match and return it. Marks the runner's
// force_triggered flag on success. Returns NULL if no forcing applied.
static const Move *try_forced_move(AutoplayWorker *autoplay_worker,
                                   GameRunner *game_runner) {
  ForceTable *ft = game_runner->shared_data->force_table;
  if (!ft || game_runner->force_triggered || !autoplay_worker->force_move_list) {
    return NULL;
  }
  Game *game = game_runner->game;
  // force_targets.csv uses the "unseen" bag count convention:
  // physical bag + opponent's rack = bag_get_letters + RACK_SIZE.
  // So bag=93 = opening turn (86 physical + 7 opp rack).
  const int bag_count = bag_get_letters(game_get_bag(game)) + (RACK_SIZE);
  int target_count = 0;
  ForceTarget **targets = force_table_lookup(ft, bag_count, &target_count);
  if (target_count == 0) {
    return NULL;
  }
  bool any_active = false;
  for (int i = 0; i < target_count; i++) {
    if (targets[i]->deficit > 0) {
      any_active = true;
      break;
    }
  }
  if (!any_active) {
    return NULL;
  }

  MoveList *ml = autoplay_worker->force_move_list;
  const LetterDistribution *ld = game_get_ld(game);
  const uint16_t ld_size = ld_get_size(ld);

  // Diff once per turn — the player_on_turn's score minus opp's.
  const int p_idx = game_get_player_on_turn_index(game);
  const int my_score =
      equity_to_int(player_get_score(game_get_player(game, p_idx)));
  const int opp_score =
      equity_to_int(player_get_score(game_get_player(game, 1 - p_idx)));
  const int cur_diff = my_score - opp_score;

  // B4 fast path: try the natural best-equity move against active targets first.
  // Avoids MOVE_RECORD_ALL enumeration on most turns since natural-best
  // commonly matches at least one PAIR/TILE/STRATUM target. Falls back to full
  // enum below if no match. (Tried MOVE_RECORD_WITHIN_X_EQUITY_OF_BEST with a
  // wider margin — no measurable speedup over single-best, simpler code wins.)
  {
    const MoveGenArgs best_args = {
        .game = game,
        .move_list = ml,
        .move_record_type = MOVE_RECORD_BEST,
        .move_sort_type = MOVE_SORT_EQUITY,
        .override_kwg = NULL,
        .thread_index = autoplay_worker->worker_index,
        .eq_margin_movegen = 0,
        .target_equity = EQUITY_MAX_VALUE,
        .target_leave_size_for_exchange_cutoff = UNSET_LEAVE_SIZE,
        .tiles_played_bv = NULL,
        .initial_tiles_bv = 0};
    generate_moves(&best_args);
    if (move_list_get_count(ml) > 0) {
      Move *best_move = move_list_get_move(ml, 0);
      Rack best_leave;
      rack_set_dist_size(&best_leave, ld_size);
      get_leave_for_move(best_move, game, &best_leave);
      const int best_score = equity_to_int(move_get_score(best_move));
      const int best_leave_len = (int)best_leave.number_of_letters;
      const bool best_is_exch = (best_score == 0);
      // Use the shape bucket for natural-best. Slot iteration plus parallel
      // required-tile-bitmap pre-filter for fast rejection.
      int best_bucket_count = 0;
      ForceTargetSlot *best_slots = NULL;
      uint32_t *best_bitmaps = NULL;
      uint32_t best_leave_bitmap = 0;
      if (best_leave_len >= 0 && best_leave_len < 8) {
        best_slots = force_table_lookup_slots_by_shape(
            ft, bag_count, best_leave_len, best_is_exch ? 1 : 0,
            &best_bucket_count);
        best_bitmaps = force_table_lookup_bitmaps_by_shape(
            ft, bag_count, best_leave_len, best_is_exch ? 1 : 0);
        // Compute leave's tile bitmap once for the bucket scan's pre-filter.
        for (uint16_t i = 0; i < best_leave.dist_size && i < 32; i++) {
          if (best_leave.array[i] > 0) {
            best_leave_bitmap |= ((uint32_t)1) << i;
          }
        }
      }
      if (best_bucket_count > 0 && best_bitmaps != NULL) {
        bool best_type_known = false;
        LeaveType best_type = LEAVE_TYPE_ALL;
        for (int priority = FORCE_TARGET_PAIR;
             priority >= FORCE_TARGET_STRATUM; priority--) {
          for (int t = 0; t < best_bucket_count; t++) {
            // Pre-filter: target's required tiles must all be present in leave.
            // Compact 4-byte AND-equal check rejects most slots without ever
            // touching the slot struct or its cache lines.
            const uint32_t req = best_bitmaps[t];
            if ((best_leave_bitmap & req) != req) continue;
            ForceTargetSlot *slot = &best_slots[t];
            if (slot->deficit <= 0 || (int)slot->kind != priority) {
              continue;
            }
            if (cur_diff < slot->diff_min || cur_diff > slot->diff_max) {
              continue;
            }
            // Same-tile pair (e.g. "??") still needs the count >= 2 check.
            if (slot->subleave_count == 2 &&
                slot->subleave_mls[0] == slot->subleave_mls[1] &&
                best_leave.array[slot->subleave_mls[0]] < 2) {
              continue;
            }
            if (slot->exchange == 0 && slot->leave_length >= 3) {
              if (!best_type_known) {
                best_type = force_classify_leave(&best_leave, ld);
                best_type_known = true;
              }
              if (best_type != (LeaveType)slot->leave_type) {
                continue;
              }
            }
            ForceTarget *target = slot->cold;
            game_runner->force_triggered = true;
            game_runner->pending_force_target = target;
            game_runner->pending_force_player_index = p_idx;
            game_runner->pending_force_diff = cur_diff;
            return best_move;
          }
        }
      }
    }
  }

  // Fallback: full MOVE_RECORD_ALL enumeration when natural-best didn't match.
  const MoveGenArgs all_args = {
      .game = game,
      .move_list = ml,
      .move_record_type = MOVE_RECORD_ALL,
      .move_sort_type = MOVE_SORT_EQUITY,
      .override_kwg = NULL,
      .thread_index = autoplay_worker->worker_index,
      .eq_margin_movegen = 0,
      .target_equity = EQUITY_MAX_VALUE,
      .target_leave_size_for_exchange_cutoff = UNSET_LEAVE_SIZE,
      .tiles_played_bv = NULL,
      .initial_tiles_bv = 0};
  generate_moves(&all_args);
  const int n_moves = move_list_get_count(ml);
  if (n_moves == 0) {
    return NULL;
  }

  // B1: pre-compute leaves for every move once. Also compute each leave's
  // tile bitmap for the per-target bitmap pre-filter.
  Rack *leaves = autoplay_worker->force_leaves;
  uint32_t *leave_bitmaps = autoplay_worker->force_leave_bitmaps;
  for (int m = 0; m < n_moves; m++) {
    rack_set_dist_size(&leaves[m], ld_size);
    Move *move = move_list_get_move(ml, m);
    get_leave_for_move(move, game, &leaves[m]);
    uint32_t bm = 0;
    for (uint16_t i = 0; i < leaves[m].dist_size && i < 32; i++) {
      if (leaves[m].array[i] > 0) {
        bm |= ((uint32_t)1) << i;
      }
    }
    leave_bitmaps[m] = bm;
  }

  // A2 + slot-based fallback + bitmap pre-filter. For each move, scan the
  // shape bucket using the parallel required-tile-bitmap array first to
  // reject targets whose required tiles aren't in the leave (4-byte AND-equal,
  // few cycles per slot).
  (void)targets; // bag-level array no longer used in inner loop
  for (int priority = FORCE_TARGET_PAIR; priority >= FORCE_TARGET_STRATUM;
       priority--) {
    for (int m = 0; m < n_moves; m++) {
      Move *move = move_list_get_move(ml, m);
      const int score = equity_to_int(move_get_score(move));
      const bool is_exch = (score == 0);
      const int leave_len = (int)leaves[m].number_of_letters;
      if (leave_len < 0 || leave_len >= 8) {
        continue;
      }
      int bucket_count = 0;
      ForceTargetSlot *slots = force_table_lookup_slots_by_shape(
          ft, bag_count, leave_len, is_exch ? 1 : 0, &bucket_count);
      uint32_t *bitmaps = force_table_lookup_bitmaps_by_shape(
          ft, bag_count, leave_len, is_exch ? 1 : 0);
      if (bucket_count == 0 || bitmaps == NULL) {
        continue;
      }
      const uint32_t leave_bm = leave_bitmaps[m];
      LeaveType move_type = LEAVE_TYPE_ALL;
      bool move_type_known = false;
      for (int t = 0; t < bucket_count; t++) {
        const uint32_t req = bitmaps[t];
        if ((leave_bm & req) != req) continue;
        ForceTargetSlot *slot = &slots[t];
        if (slot->deficit <= 0 || (int)slot->kind != priority) {
          continue;
        }
        // Aggregator stores diff = pre_action_diff + move_score (post-action,
        // from the perspective of the player on turn). Match against same.
        // For exchanges/passes score==0 so eff_diff == cur_diff.
        const int eff_diff = cur_diff + score;
        if (eff_diff < slot->diff_min || eff_diff > slot->diff_max) {
          continue;
        }
        if (slot->subleave_count == 2 &&
            slot->subleave_mls[0] == slot->subleave_mls[1] &&
            leaves[m].array[slot->subleave_mls[0]] < 2) {
          continue;
        }
        if (slot->exchange == 0 && slot->leave_length >= 3) {
          if (!move_type_known) {
            move_type = force_classify_leave(&leaves[m], ld);
            move_type_known = true;
          }
          if (move_type != (LeaveType)slot->leave_type) {
            continue;
          }
        }
        ForceTarget *target = slot->cold;
        game_runner->force_triggered = true;
        game_runner->pending_force_target = target;
        game_runner->pending_force_player_index = p_idx;
        game_runner->pending_force_diff = cur_diff;
        return move;
      }
    }
  }
  return NULL;
}

const Move *game_runner_get_best_move(AutoplayWorker *autoplay_worker,
                                      GameRunner *game_runner) {
  const int player_on_turn_index =
      game_get_player_on_turn_index(game_runner->game);
  const SimArgs *sim_args = (player_on_turn_index == 0)
                                ? &autoplay_worker->args.p1_sim_args
                                : &autoplay_worker->args.p2_sim_args;
  if (sim_args->num_plies == 0) {
    return get_top_equity_move(
        game_runner->game, autoplay_worker->worker_index,
        autoplay_worker->move_lists[player_on_turn_index]);
  }
  return game_runner_get_top_simming_move(autoplay_worker, game_runner);
}

// Returns the played move
const Move *game_runner_play_move(AutoplayWorker *autoplay_worker,
                                  GameRunner *game_runner) {
  if (game_runner_is_game_over(game_runner)) {
    log_fatal("game runner attempted to play a move when the game is over");
  }
  Game *game = game_runner->game;
  const int player_on_turn_index = game_get_player_on_turn_index(game);
  LeavegenSharedData *lg_shared_data =
      game_runner->shared_data->leavegen_shared_data;
  // If we are forcing a draw, we need to draw a rare leave. The drawn
  // leave does not necessarily fit in the bag. If we've reached the
  // target minimum leave count for all leaves, no rare leave can be
  // drawn.
  Rack *player_rack =
      player_get_rack(game_get_player(game, player_on_turn_index));
  const int ld_size = ld_get_size(game_get_ld(game));
  Rack rare_rack_or_move_leave;
  rack_set_dist_size(&rare_rack_or_move_leave, ld_size);

  if (game_runner->force_draw &&
      rack_list_get_rare_rack(lg_shared_data->rack_list, autoplay_worker->prng,
                              &rare_rack_or_move_leave)) {
    // Backup the original rack
    Rack original_rack;
    rack_copy(&original_rack, player_rack);

    // Set the rack to the rare leave
    rack_copy(player_rack, &rare_rack_or_move_leave);

    const Move *forced_move =
        game_runner_get_best_move(autoplay_worker, game_runner);
    rack_list_add_rack(lg_shared_data->rack_list, &rare_rack_or_move_leave,
                       equity_to_double(move_get_equity(forced_move)));

    rack_copy(player_rack, &original_rack);
  }

  const Move *move = NULL;
  // Empty-board K-way DFS (slice 2): if caller pre-decided the action for
  // this fork branch, use it verbatim. Highest priority — overrides every
  // other selection path.
  if (game_runner->eb_forced_move) {
    Move *spare = move_list_get_spare_move(
        autoplay_worker->move_lists[player_on_turn_index]);
    move_copy(spare, game_runner->eb_forced_move);
    move = spare;
  }
  // Opening-pass mode: on the opening turn of the starter (the player with
  // the forced rack) in the pass branch, override the move with a pass.
  if (!move && game_runner->opening_pass_rack_idx >= 0 &&
      game_runner->opening_pass_branch == 0 &&
      game_runner->turn_number == 0 &&
      player_on_turn_index == game_runner->opening_pass_player_index) {
    Move *spare = move_list_get_spare_move(
        autoplay_worker->move_lists[player_on_turn_index]);
    move_set_as_pass(spare);
    move = spare;
  }
  // Single-target-turn EB role-driven forcing. Replaces the legacy T1/T2
  // explicit force block. Runs only when EB DFS is active and we're still
  // on an empty board pre-target.
  if (!move && game_runner->eb_active &&
      game_runner->pass_cycle_active &&
      game_runner->pass_cycle_branch == 0 &&
      board_get_tiles_played(game_get_board(game)) == 0) {
    const int target = game_runner->shared_data->eb_target_turn;
    const int turn = game_runner->pass_cycle_n_moves + 1;
    const EbTurnRole role = eb_classify_turn(target, turn);
    if (role == EB_ROLE_REC_PRE) {
      // Recording player pre-target turn — rack will be discarded at TARGET
      // inject, just force pass to keep board empty.
      Move *spare = move_list_get_spare_move(
          autoplay_worker->move_lists[player_on_turn_index]);
      move_set_as_pass(spare);
      move = spare;
    } else if (role == EB_ROLE_OPP_MID) {
      // OPP_MID returned 0 from eb_enumerate_actions when rack is_pass=0
      // (single-branch case). Force best-equity exchange so the cycle
      // continues without playing tiles. is_pass=1 falls through to the
      // existing pass_cycle force-pass block below (which handles it).
      char canonical[RACK_SIZE + 2] = {0};
      {
        const LetterDistribution *ld = game_get_ld(game);
        const uint16_t dist_size = rack_get_dist_size(player_rack);
        int n = 0;
        for (uint16_t i = 1; i < dist_size && n < RACK_SIZE; i++) {
          const int c = rack_get_letter(player_rack, i);
          for (int k = 0; k < c && n < RACK_SIZE; k++) {
            canonical[n++] = ld->ld_ml_to_hl[i][0];
          }
        }
        const int nblanks = rack_get_letter(player_rack, 0);
        for (int k = 0; k < nblanks && n < RACK_SIZE; k++) {
          canonical[n++] = '?';
        }
        canonical[n] = '\0';
      }
      PassCycleTable *pct = game_runner->shared_data->pass_cycle_table;
      const bool is_pass =
          pass_cycle_lookup_is_pass(pct, canonical) == 1;
      if (!is_pass) {
        const MoveGenArgs gen_args = {
            .game = game,
            .move_list = autoplay_worker->move_lists[player_on_turn_index],
            .move_record_type = MOVE_RECORD_ALL,
            .move_sort_type = MOVE_SORT_EQUITY,
            .override_kwg = NULL,
            .thread_index = autoplay_worker->worker_index,
            .eq_margin_movegen = 0,
            .target_equity = EQUITY_MAX_VALUE,
            .target_leave_size_for_exchange_cutoff = UNSET_LEAVE_SIZE,
            .tiles_played_bv = NULL,
            .initial_tiles_bv = 0};
        generate_moves(&gen_args);
        MoveList *ml = autoplay_worker->move_lists[player_on_turn_index];
        const int n_moves = move_list_get_count(ml);
        move_list_sort_moves(ml);
        Move *best_exch = NULL;
        for (int m = 0; m < n_moves; m++) {
          Move *cand = move_list_get_move(ml, m);
          if (move_get_type(cand) == GAME_EVENT_EXCHANGE) {
            best_exch = cand;
            break;
          }
        }
        if (best_exch) {
          move = best_exch;
        } else {
          Move *spare = move_list_get_spare_move(
              autoplay_worker->move_lists[player_on_turn_index]);
          move_set_as_pass(spare);
          move = spare;
        }
      }
    }
    // OPP_FIRST, OPP_CLOSEST, TARGET, POST: handled by DFS via
    // eb_forced_move (set at the top of this function), or natural HastyBot
    // for POST. Nothing to do here.
  }
  // Pass-cycle mode (branch 0, empty board): each turn, check the player
  // on turn's CURRENT rack against the pool. is_pass=1 → force pass.
  // Otherwise leave the move to natural HastyBot play (which will exchange
  // for is_pass=0 racks — those were classified as exchange — and play
  // tiles for racks not in the pool). Applied symmetrically to both
  // players: after an exchange, the new rack is re-checked, so a player
  // who exchanges into a pass-favorable rack starts passing.
  if (!move && game_runner->pass_cycle_active &&
      game_runner->pass_cycle_branch == 0 &&
      board_get_tiles_played(game_get_board(game)) == 0) {
    char canonical[RACK_SIZE + 2] = {0};
    {
      const LetterDistribution *ld = game_get_ld(game);
      const uint16_t dist_size = rack_get_dist_size(player_rack);
      int n = 0;
      // Letters first (skip ML 0 = blank).
      for (uint16_t i = 1; i < dist_size && n < RACK_SIZE; i++) {
        const int c = rack_get_letter(player_rack, i);
        for (int k = 0; k < c && n < RACK_SIZE; k++) {
          canonical[n++] = ld->ld_ml_to_hl[i][0];
        }
      }
      // Blanks last.
      const int nblanks = rack_get_letter(player_rack, 0);
      for (int k = 0; k < nblanks && n < RACK_SIZE; k++) {
        canonical[n++] = '?';
      }
      canonical[n] = '\0';
    }
    PassCycleTable *pct = game_runner->shared_data->pass_cycle_table;
    if (pass_cycle_lookup_is_pass(pct, canonical) == 1) {
      Move *spare = move_list_get_spare_move(
          autoplay_worker->move_lists[player_on_turn_index]);
      move_set_as_pass(spare);
      move = spare;
    }
  }
  // (Legacy T1/T2 explicit force-kind block removed — replaced by the
  // role-driven force block above which uses eb_classify_turn instead of
  // iter_count-derived force_kind bits.)
  if (!move) {
    move = try_forced_move(autoplay_worker, game_runner);
  }
  if (!move) {
    move = game_runner_get_best_move(autoplay_worker, game_runner);
  }

  if (lg_shared_data) {
    rack_list_add_rack(lg_shared_data->rack_list, player_rack,
                       equity_to_double(move_get_equity(move)));
  }
  get_leave_for_move(move, game, &rare_rack_or_move_leave);
  // Skip recording pre-force turns when a force table is active: those turns
  // are "tainted" (the game was always going to include a force), and the
  // gen 2 spec excludes them from training data.
  const bool skip_recording =
      game_runner->shared_data->force_table != NULL &&
      !game_runner->force_triggered;
  if (!skip_recording) {
    autoplay_results_add_move(autoplay_worker->autoplay_results,
                              game_runner->game, move,
                              &rare_rack_or_move_leave);
  }

  // Print board with move about to be played if requested
  if (autoplay_worker->args.print_boards) {
    StringBuilder *output = string_builder_create();
    if (game_runner->pair_game_number == 0) {
      string_builder_add_formatted_string(
          output, "\n=== Game %llu, Turn %d ===\n",
          (unsigned long long)game_runner->game_number + 1,
          game_runner->turn_number + 1);
    } else {
      string_builder_add_formatted_string(
          output, "\n=== Game Pair %llu, Game %d, Turn %d ===\n",
          (unsigned long long)game_runner->game_number + 1,
          game_runner->pair_game_number, game_runner->turn_number + 1);
    }
    string_builder_add_game(
        game, NULL, autoplay_worker->args.game_string_options, NULL, output);
    string_builder_add_move(output, game_get_board(game), move,
                            game_get_ld(game), true);
    string_builder_add_string(output, "\n");
    const SimArgs *sim_args = (player_on_turn_index == 0)
                                  ? &autoplay_worker->args.p1_sim_args
                                  : &autoplay_worker->args.p2_sim_args;
    if (sim_args->num_plies > 0) {
      char *sim_str = sim_results_get_string(
          game, autoplay_worker->sim_results, sim_args->max_num_display_plays,
          sim_args->max_num_display_plies, -1, -1, NULL, 0, false, false, NULL);
      string_builder_add_string(output, sim_str);
      free(sim_str);
      if (sim_args->use_inference && game_runner->turn_number > 0 &&
          move_get_type(&game_runner->previous_move) != GAME_EVENT_PASS) {
        string_builder_add_inference(
            output, autoplay_worker->inference_results, game_get_ld(game),
            sim_args->inference_args.leave_list_capacity, false);
      }
    }
    thread_control_print(autoplay_worker->args.thread_control,
                         string_builder_peek(output));
    string_builder_destroy(output);
  }

  // Pass-cycle history: capture (rack, move) for the first 6 turns. Only
  // 6-pass-end games get recorded, so anything past turn 5 is irrelevant.
  if (game_runner->pass_cycle_active && game_runner->pass_cycle_n_moves < 6) {
    char *slot =
        game_runner->pass_cycle_history[game_runner->pass_cycle_n_moves];
    char rack_buf[RACK_SIZE + 2] = {0};
    int rb = 0;
    const LetterDistribution *ld = game_get_ld(game);
    const uint16_t dist_size = rack_get_dist_size(player_rack);
    for (uint16_t i = 0; i < dist_size && rb < RACK_SIZE; i++) {
      const int n = rack_get_letter(player_rack, i);
      for (int k = 0; k < n && rb < RACK_SIZE; k++) {
        rack_buf[rb++] = ld->ld_ml_to_hl[i][0];
      }
    }
    rack_buf[rb] = '\0';
    const game_event_t mt = move_get_type(move);
    if (mt == GAME_EVENT_EXCHANGE) {
      char tiles[RACK_SIZE + 2] = {0};
      const int nt = move_get_tiles_played(move);
      for (int i = 0; i < nt && i < RACK_SIZE; i++) {
        const MachineLetter ml = move_get_tile(move, i);
        tiles[i] = ld->ld_ml_to_hl[ml][0];
      }
      snprintf(slot, 24, "%s:X%s", rack_buf, tiles);
    } else if (mt == GAME_EVENT_PASS) {
      snprintf(slot, 24, "%s:P", rack_buf);
    } else {
      // Tile placement on what should still be an empty board: shouldn't
      // happen in 6-pass paths, but encode defensively.
      snprintf(slot, 24, "%s:T", rack_buf);
    }
    game_runner->pass_cycle_n_moves++;
  }

  // Empty-board recorder snapshot (slice 1): one row per cycle-alive
  // empty-board decision faced by the player on turn. eventual_outcome and
  // eventual_margin are filled in at game-end and rows emitted there.
  // Cycle-alive = no tile placed yet (board still empty before this move).
  if (game_runner->eb_active && game_runner->eb_n_snaps < 6 &&
      board_get_tiles_played(game_get_board(game)) == 0) {
    const int slot = game_runner->eb_n_snaps;
    game_runner->eb_snaps[slot].turn_on_empty_board = slot + 1;
    game_runner->eb_snaps[slot].player_on_turn = player_on_turn_index;

    // Canonical rack: A..Z then blanks.
    char *rb = game_runner->eb_snaps[slot].rack;
    int rn = 0;
    const LetterDistribution *ld_eb = game_get_ld(game);
    const uint16_t ds_eb = rack_get_dist_size(player_rack);
    for (uint16_t i = 1; i < ds_eb && rn < RACK_SIZE; i++) {
      const int c = rack_get_letter(player_rack, i);
      for (int k = 0; k < c && rn < RACK_SIZE; k++) {
        rb[rn++] = ld_eb->ld_ml_to_hl[i][0];
      }
    }
    const int nblanks_eb = rack_get_letter(player_rack, 0);
    for (int k = 0; k < nblanks_eb && rn < RACK_SIZE; k++) {
      rb[rn++] = '?';
    }
    rb[rn] = '\0';

    // Leave (post-action rack) — derived per kind:
    //   pass: leave = rack (no change)
    //   exch: leave = rack minus exchanged tiles
    //   play: leave = rack minus played tiles (computed via
    //         get_leave_for_move so blank-as-letter / play-through tiles
    //         are handled correctly)
    Rack tmp_leave;
    rack_set_dist_size(&tmp_leave, ds_eb);
    rack_reset(&tmp_leave);
    if (move_get_type(move) == GAME_EVENT_PASS) {
      rack_copy(&tmp_leave, player_rack);
    } else {
      get_leave_for_move(move, game, &tmp_leave);
    }
    char *lb = game_runner->eb_snaps[slot].leave;
    int ln = 0;
    for (uint16_t i = 1; i < ds_eb && ln < RACK_SIZE; i++) {
      const int c = rack_get_letter(&tmp_leave, i);
      for (int k = 0; k < c && ln < RACK_SIZE; k++) {
        lb[ln++] = ld_eb->ld_ml_to_hl[i][0];
      }
    }
    const int leave_blanks = rack_get_letter(&tmp_leave, 0);
    for (int k = 0; k < leave_blanks && ln < RACK_SIZE; k++) {
      lb[ln++] = '?';
    }
    lb[ln] = '\0';

    // opp_history = other player's actions accumulated so far this cycle.
    const char *oh = (player_on_turn_index == 0)
                         ? game_runner->eb_actions_p1
                         : game_runner->eb_actions_p0;
    snprintf(game_runner->eb_snaps[slot].opp_history,
             sizeof(game_runner->eb_snaps[slot].opp_history), "%s", oh);

    // Action kind/repr/size from the chosen move.
    const game_event_t mt_eb = move_get_type(move);
    if (mt_eb == GAME_EVENT_PASS) {
      game_runner->eb_snaps[slot].action_kind = 0;
      game_runner->eb_snaps[slot].action_repr[0] = '\0';
      game_runner->eb_snaps[slot].action_size = 0;
    } else if (mt_eb == GAME_EVENT_EXCHANGE) {
      game_runner->eb_snaps[slot].action_kind = 1;
      char *ar = game_runner->eb_snaps[slot].action_repr;
      const int nt = move_get_tiles_played(move);
      int an = 0;
      const int cap = (int)sizeof(game_runner->eb_snaps[slot].action_repr) - 1;
      for (int i = 0; i < nt && an < cap; i++) {
        const MachineLetter ml = move_get_tile(move, i);
        ar[an++] = ld_eb->ld_ml_to_hl[ml][0];
      }
      ar[an] = '\0';
      game_runner->eb_snaps[slot].action_size = nt;
    } else {
      // Tile placement: cycle-break decision. action_repr left empty for
      // slice 1 (action_kind+size+post-state suffice for the sub-model).
      game_runner->eb_snaps[slot].action_kind = 2;
      game_runner->eb_snaps[slot].action_repr[0] = '\0';
      game_runner->eb_snaps[slot].action_size = move_get_tiles_played(move);
    }
    game_runner->eb_snaps[slot].natural_slot = game_runner->eb_natural_slot;
    game_runner->eb_snaps[slot].move_score =
        (mt_eb == GAME_EVENT_TILE_PLACEMENT_MOVE)
            ? equity_to_int(move_get_score(move))
            : 0;
    game_runner->eb_n_snaps++;

    // Append this action to player_on_turn's per-cycle action history.
    char *hist = (player_on_turn_index == 0) ? game_runner->eb_actions_p0
                                             : game_runner->eb_actions_p1;
    int *off = (player_on_turn_index == 0) ? &game_runner->eb_actions_p0_off
                                           : &game_runner->eb_actions_p1_off;
    const char *prefix = (*off == 0) ? "" : "|";
    const int rem = 96 - *off;
    if (rem > 0) {
      int wrote = 0;
      if (mt_eb == GAME_EVENT_PASS) {
        wrote = snprintf(hist + *off, (size_t)rem, "%sP", prefix);
      } else if (mt_eb == GAME_EVENT_EXCHANGE) {
        wrote = snprintf(hist + *off, (size_t)rem, "%sX%s", prefix,
                         game_runner->eb_snaps[slot].action_repr);
      } else {
        wrote = snprintf(hist + *off, (size_t)rem, "%sT%d", prefix,
                         game_runner->eb_snaps[slot].action_size);
      }
      if (wrote > 0 && wrote < rem) *off += wrote;
    }
  }

  // Pass-cycle early-exit: once a tile is placed, the game can no longer
  // qualify for 6-pass-cycle recording (which requires tiles_played==0).
  // Same once we've seen 6 zero-score turns without the engine ending the
  // game (rare but means the cycle won't trigger here).
  if (game_runner->pass_cycle_active &&
      move_get_type(move) == GAME_EVENT_TILE_PLACEMENT_MOVE) {
    game_runner->pass_cycle_abandoned = true;
  }

  play_move(move, game, NULL);

  // After 6 turns: if the game didn't end via consecutive-zeros on this move,
  // it never will be a recorded 6-pass-cycle game.
  if (game_runner->pass_cycle_active && game_runner->pass_cycle_n_moves >= 6 &&
      !game_over(game)) {
    game_runner->pass_cycle_abandoned = true;
  }
  if (game_runner->game_one_move_behind && game_runner->turn_number > 0) {
    play_move(&game_runner->previous_move, game_runner->game_one_move_behind,
              NULL);
  }
  move_copy(&game_runner->previous_move, move);
  game_runner->turn_number++;
  return move;
}

void print_current_status(AutoplayWorker *autoplay_worker,
                          AutoplayIterCompletedOutput *iter_completed_output) {
  StringBuilder *status_sb = string_builder_create();
  AutoplaySharedData *shared_data = autoplay_worker->shared_data;
  string_builder_add_formatted_string(
      status_sb, "Played %ld games in %.3f seconds.",
      iter_completed_output->iter_count_completed,
      iter_completed_output->time_elapsed);
  const LeavegenSharedData *lg_shared_data = shared_data->leavegen_shared_data;
  if (lg_shared_data) {
    string_builder_add_formatted_string(
        status_sb,
        " Played %ld games in generation %d with %ld rack under target "
        "count.\n",
        iter_completed_output->iter_count_completed -
            lg_shared_data->gen_start_games,
        lg_shared_data->gens_completed + 1,
        rack_list_get_racks_below_target_count(lg_shared_data->rack_list));
  } else {
    string_builder_add_string(status_sb, "\n");
  }
  thread_control_print(autoplay_worker->args.thread_control,
                       string_builder_peek(status_sb));
  string_builder_destroy(status_sb);
}

void autoplay_add_game(AutoplayWorker *autoplay_worker,
                       const GameRunner *game_runner, bool divergent) {
  autoplay_results_add_game(autoplay_worker->autoplay_results,
                            game_runner->game, game_runner->turn_number,
                            divergent, game_runner->seed);
  // Credit the force table for this game's final outcome. For stratum-kind
  // targets this is the moment the deficit actually decrements (and only
  // when the outcome bumps min(wins, losses) at the force-turn diff).
  // Skip if EB-cycle force-table mode is active: per-emit credit in
  // eb_emit_leaf already handled deficits for forced T2-T6 slots.
  ForceTable *ft = autoplay_worker->shared_data->force_table;
  if (ft != NULL && game_runner->pending_force_target != NULL &&
      autoplay_worker->shared_data->eb_force_turns_mask == 0) {
    const Game *game = game_runner->game;
    const int p_idx = game_runner->pending_force_player_index;
    const int my_final =
        equity_to_int(player_get_score(game_get_player(game, p_idx)));
    const int opp_final =
        equity_to_int(player_get_score(game_get_player(game, 1 - p_idx)));
    const bool is_tie = (my_final == opp_final);
    const bool is_win = (my_final > opp_final);
    force_table_credit_game(ft, game_runner->pending_force_target,
                            game_runner->pending_force_diff, is_win, is_tie);
  }
  AutoplayIterCompletedOutput iter_completed_output;
  autoplay_complete_iter(autoplay_worker->shared_data, &iter_completed_output);
  if (iter_completed_output.print_info) {
    print_current_status(autoplay_worker, &iter_completed_output);
  }
}

// Encode a player's current rack as ASCII chars (e.g. "AEINRST", "?ABDIJ").
// out must have room for RACK_SIZE+1 bytes; the LD's first ld_ml_to_hl byte
// is used per machine letter (English Scrabble = single ASCII char per tile).
static void pass_cycle_stringify_rack(const Game *game, int player_index,
                                      char *out) {
  const LetterDistribution *ld = game_get_ld(game);
  const Rack *rack = player_get_rack(game_get_player(game, player_index));
  const uint16_t dist_size = rack_get_dist_size(rack);
  int n = 0;
  for (uint16_t i = 0; i < dist_size && n < RACK_SIZE; i++) {
    const int c = rack_get_letter(rack, i);
    for (int k = 0; k < c && n < RACK_SIZE; k++) {
      out[n++] = ld->ld_ml_to_hl[i][0];
    }
  }
  out[n] = '\0';
}

// ---- Slice 2: K-way fork DFS over the empty-board cycle subtree ----
//
// At a fork point (cycle-alive empty-board turn, bot's current rack is
// is_pass=1, turn ∈ {3,4,5}), the DFS enumerates K=3 actions {pass,
// best-exchange, best-play}, plays each with full state save/restore, and
// recurses to game-end. Each leaf branch emits its captured eb_snaps with
// the leaf's eventual_outcome.
//
// Subset fan-out at turn 5 (no plays) and turn 6 is deferred to slice 2b.

// Returns canonical rack string for player on turn into out (>= RACK_SIZE+2).
static void eb_canonical_rack(const Game *game, int player_index, char *out) {
  const LetterDistribution *ld = game_get_ld(game);
  const Rack *rack = player_get_rack(game_get_player(game, player_index));
  const uint16_t dist_size = rack_get_dist_size(rack);
  int n = 0;
  for (uint16_t i = 1; i < dist_size && n < RACK_SIZE; i++) {
    const int c = rack_get_letter(rack, i);
    for (int k = 0; k < c && n < RACK_SIZE; k++) {
      out[n++] = ld->ld_ml_to_hl[i][0];
    }
  }
  const int nblanks = rack_get_letter(rack, 0);
  for (int k = 0; k < nblanks && n < RACK_SIZE; k++) {
    out[n++] = '?';
  }
  out[n] = '\0';
}

// Per-turn role for single-target-turn EB recording. Each turn (1..6) gets
// classified relative to the target turn; the role drives both the fan-out
// branch set in eb_enumerate_actions and the natural-play forcing in
// game_runner_get_move.
typedef enum {
  EB_ROLE_TARGET,       // The recording turn — full subset fan-out + 5-way inject.
  EB_ROLE_OPP_FIRST,    // Earliest opp turn before target — 50/50 pool-sampled
                        // rack, 2 branches always (pass + best-exch).
  EB_ROLE_OPP_CLOSEST,  // Opp turn at target-1 (and not also OPP_FIRST) —
                        // inherited rack, 2 branches always.
  EB_ROLE_OPP_MID,      // Opp turn between OPP_FIRST and OPP_CLOSEST — inherited
                        // rack, 2 branches if is_pass, 1 branch (force-exch) else.
  EB_ROLE_REC_PRE,      // Recording-player turn before target — bag-random rack,
                        // force pass unconditionally (rack discarded at target).
  EB_ROLE_POST,         // Turn after target — natural HastyBot play, no fork,
                        // no recording.
} EbTurnRole;

// Classify a turn relative to the target. Player parity: T_odd = P1, T_even = P2.
// Recording player at Tn = P1 if n is odd else P2. First opp turn (the earliest
// turn whose player is the opponent) is T2 for odd-target (P1 records) and T1
// for even-target (P2 records).
static EbTurnRole eb_classify_turn(int target_turn, int turn) {
  if (turn == target_turn) return EB_ROLE_TARGET;
  if (turn > target_turn) return EB_ROLE_POST;
  const bool same_player = ((turn ^ target_turn) & 1) == 0;
  if (same_player) return EB_ROLE_REC_PRE;
  const int first_opp = (target_turn % 2 == 1) ? 2 : 1;
  if (turn == first_opp) return EB_ROLE_OPP_FIRST;
  if (turn == target_turn - 1) return EB_ROLE_OPP_CLOSEST;
  return EB_ROLE_OPP_MID;
}

// Render a Rack's tile multiset to a canonical sorted-letters-then-blanks
// string (matches eb_canonical_rack format). Used for fan-out dedup on
// (action_type, score, leave).
static void eb_render_leave(const Rack *leave, const LetterDistribution *ld,
                            char *out, size_t cap) {
  const uint16_t dist_size = rack_get_dist_size(leave);
  size_t n = 0;
  for (uint16_t i = 1; i < dist_size && n + 1 < cap; i++) {
    const int c = rack_get_letter(leave, i);
    for (int k = 0; k < c && n + 1 < cap; k++) {
      out[n++] = ld->ld_ml_to_hl[i][0];
    }
  }
  const int nblanks = rack_get_letter(leave, 0);
  for (int k = 0; k < nblanks && n + 1 < cap; k++) {
    out[n++] = '?';
  }
  out[n] = '\0';
}

// Save/restore wrapper for eb_active per-runner state that lives outside
// game state (snap stack offsets and per-player action history buffers).
typedef struct EbMetaSave {
  int n_snaps;
  int actions_p0_off;
  int actions_p1_off;
  char actions_p0[96];
  char actions_p1[96];
  bool pass_cycle_abandoned;
  int pass_cycle_n_moves;
  int turn_number;
  int divergence_turn;
  int last_divergence_turn;
  int n_divergences;
} EbMetaSave;

static void eb_meta_save(const GameRunner *gr, EbMetaSave *s) {
  s->n_snaps = gr->eb_n_snaps;
  s->actions_p0_off = gr->eb_actions_p0_off;
  s->actions_p1_off = gr->eb_actions_p1_off;
  memcpy(s->actions_p0, gr->eb_actions_p0, sizeof(s->actions_p0));
  memcpy(s->actions_p1, gr->eb_actions_p1, sizeof(s->actions_p1));
  s->pass_cycle_abandoned = gr->pass_cycle_abandoned;
  s->pass_cycle_n_moves = gr->pass_cycle_n_moves;
  s->turn_number = gr->turn_number;
  s->divergence_turn = gr->eb_divergence_turn;
  s->last_divergence_turn = gr->eb_last_divergence_turn;
  s->n_divergences = gr->eb_n_divergences;
}

static void eb_meta_restore(GameRunner *gr, const EbMetaSave *s) {
  gr->eb_n_snaps = s->n_snaps;
  gr->eb_actions_p0_off = s->actions_p0_off;
  gr->eb_actions_p1_off = s->actions_p1_off;
  memcpy(gr->eb_actions_p0, s->actions_p0, sizeof(s->actions_p0));
  memcpy(gr->eb_actions_p1, s->actions_p1, sizeof(s->actions_p1));
  gr->pass_cycle_abandoned = s->pass_cycle_abandoned;
  gr->pass_cycle_n_moves = s->pass_cycle_n_moves;
  gr->turn_number = s->turn_number;
  gr->eb_divergence_turn = s->divergence_turn;
  gr->eb_last_divergence_turn = s->last_divergence_turn;
  gr->eb_n_divergences = s->n_divergences;
}

// After natural slots are populated, query the force_table for active
// targets and append matching moves as ADDITIONAL DFS slots. Each appended
// slot has its eb_force_target_for_slot[] set so the per-emit decrement
// in eb_emit_leaf can credit the target's deficit.
//
// Per move we attach AT MOST ONE target (first match by priority); each
// game's DFS thus contributes one credit per chosen forced slot. Multiple
// active targets may match the same move — only one is credited (and the
// other targets must wait for a different game). This avoids over-counting.
//
// The per-emit credit only fires for snaps EMITTED by the natural-post-Tk
// filter; forced slots at turn k always emit their k-row (last_div=k <= k),
// so each forced slot in a leaf's path produces exactly one credit.
static int eb_append_force_target_slots(AutoplayWorker *w, GameRunner *gr,
                                         MoveList *ml, int n_moves,
                                         int max_slot) {
  ForceTable *ft = w->shared_data->force_table;
  if (!ft || !w->force_leaves || n_moves == 0) return max_slot;
  Game *game = gr->game;
  // Force-table cell diff ranges are computed from the bag=93 opener
  // aggregate (cur_diff_pre = 0; recorded diff = score). At T6 of K-way
  // branched paths where T2-T5 included non-pass plays, the board has
  // tiles and cur_diff != 0, so eff_diff = cur_diff + score never falls
  // in the cell's narrow range. Only fire force-table at the all-pass
  // (board-empty) leaf so the cell ranges align with the opener convention.
  if (board_get_tiles_played(game_get_board(game)) > 0) {
    return max_slot;
  }
  const int bag_count = bag_get_letters(game_get_bag(game)) + (RACK_SIZE);
  int target_count = 0;
  ForceTarget **targets = force_table_lookup(ft, bag_count, &target_count);
  if (target_count == 0) return max_slot;
  bool any_active = false;
  for (int i = 0; i < target_count; i++) {
    if (targets[i]->deficit > 0) { any_active = true; break; }
  }
  if (!any_active) return max_slot;
  (void)targets;

  // Rack-level early reject: compute the player's rack bitmap once and
  // check whether ANY active target's required-leave bitmap is a subset
  // of the rack. If no target is reachable from this rack, the per-move
  // get_leave_for_move scan is pure waste.
  const int p_idx_pre =
      game_get_player_on_turn_index(game);
  const Rack *p_rack =
      player_get_rack(game_get_player(game, p_idx_pre));
  uint32_t rack_bm = 0;
  for (uint16_t i = 0; i < p_rack->dist_size && i < 32; i++) {
    if (p_rack->array[i] > 0) rack_bm |= ((uint32_t)1) << i;
  }
  bool any_reachable = false;
  for (int b_len = 0; b_len < 8 && !any_reachable; b_len++) {
    for (int b_ex = 0; b_ex < 2 && !any_reachable; b_ex++) {
      int bc = 0;
      ForceTargetSlot *bsl = force_table_lookup_slots_by_shape(
          ft, bag_count, b_len, b_ex, &bc);
      uint32_t *bbm = force_table_lookup_bitmaps_by_shape(
          ft, bag_count, b_len, b_ex);
      if (!bsl || !bbm) continue;
      for (int t = 0; t < bc; t++) {
        if (bsl[t].deficit <= 0) continue;
        if ((rack_bm & bbm[t]) == bbm[t]) { any_reachable = true; break; }
      }
    }
  }
  if (!any_reachable) return max_slot;

  // Compute leaves + bitmaps for every move (mirrors try_forced_move).
  const LetterDistribution *ld = game_get_ld(game);
  const uint16_t ld_size = ld_get_size(ld);
  Rack *leaves = w->force_leaves;
  if (n_moves > w->force_leaves_capacity) n_moves = w->force_leaves_capacity;
  uint32_t *leave_bitmaps = w->force_leave_bitmaps;
  for (int m = 0; m < n_moves; m++) {
    rack_set_dist_size(&leaves[m], ld_size);
    Move *move = move_list_get_move(ml, m);
    get_leave_for_move(move, game, &leaves[m]);
    uint32_t bm = 0;
    for (uint16_t i = 0; i < leaves[m].dist_size && i < 32; i++) {
      if (leaves[m].array[i] > 0) bm |= ((uint32_t)1) << i;
    }
    leave_bitmaps[m] = bm;
  }

  const int p_idx = game_get_player_on_turn_index(game);
  const int my_score =
      equity_to_int(player_get_score(game_get_player(game, p_idx)));
  const int opp_score =
      equity_to_int(player_get_score(game_get_player(game, 1 - p_idx)));
  const int cur_diff = my_score - opp_score;

  // Match in PAIR-then-TILE-then-STRATUM priority order ACROSS all moves
  // (mirrors try_forced_move's outer-loop priority). This guarantees rare-
  // pair targets get first dibs on the limited slot budget; common stratum
  // matches don't crowd them out by virtue of appearing earlier in the
  // equity-sorted move list.
  int slot = max_slot + 1;
  // Per-move scratch buffers for the priority-first-then-move loop. Sized
  // to the same cap as force_leaves; capped here just defensively so the
  // stack doesn't blow up if the caller ever exceeds the cap.
  enum { PRIORITY_LOOP_CAP = 32768 };
  if (n_moves > PRIORITY_LOOP_CAP) n_moves = PRIORITY_LOOP_CAP;
  static _Thread_local bool used_move[PRIORITY_LOOP_CAP];
  static _Thread_local int8_t move_type_cache[PRIORITY_LOOP_CAP];
  for (int i = 0; i < n_moves; i++) {
    used_move[i] = false;
    move_type_cache[i] = -1;
  }

  for (int priority = FORCE_TARGET_PAIR;
       priority >= FORCE_TARGET_STRATUM && slot < EB_MAX_ACTIONS; priority--) {
    for (int m = 0; m < n_moves && slot < EB_MAX_ACTIONS; m++) {
      if (used_move[m]) continue;
      Move *move = move_list_get_move(ml, m);
      const int score = equity_to_int(move_get_score(move));
      const bool is_exch = (score == 0);
      const int leave_len = (int)leaves[m].number_of_letters;
      if (leave_len < 0 || leave_len >= 8) continue;
      int bucket_count = 0;
      ForceTargetSlot *bucket_slots = force_table_lookup_slots_by_shape(
          ft, bag_count, leave_len, is_exch ? 1 : 0, &bucket_count);
      uint32_t *bitmaps = force_table_lookup_bitmaps_by_shape(
          ft, bag_count, leave_len, is_exch ? 1 : 0);
      if (bucket_count == 0 || bitmaps == NULL) continue;
      const uint32_t leave_bm = leave_bitmaps[m];
      ForceTarget *matched = NULL;
      for (int t = 0; t < bucket_count; t++) {
        const uint32_t req = bitmaps[t];
        if ((leave_bm & req) != req) continue;
        ForceTargetSlot *fs = &bucket_slots[t];
        if (fs->deficit <= 0 || (int)fs->kind != priority) continue;
        // Aggregator stores diff = pre_action_diff + move_score (post-action,
        // from the perspective of the player on turn). Match against same.
        // For exchanges/passes score==0 so eff_diff == cur_diff.
        const int eff_diff = cur_diff + score;
        if (eff_diff < fs->diff_min || eff_diff > fs->diff_max) continue;
        if (fs->subleave_count == 2 &&
            fs->subleave_mls[0] == fs->subleave_mls[1] &&
            leaves[m].array[fs->subleave_mls[0]] < 2) continue;
        if (fs->exchange == 0 && fs->leave_length >= 3) {
          if (move_type_cache[m] < 0) {
            move_type_cache[m] = (int8_t)force_classify_leave(&leaves[m], ld);
          }
          if ((LeaveType)move_type_cache[m] != (LeaveType)fs->leave_type)
            continue;
        }
        matched = fs->cold;
        break;
      }
      if (matched != NULL) {
        move_copy(gr->eb_action_buf[slot], move);
        gr->eb_action_present[slot] = true;
        gr->eb_force_target_for_slot[slot] = matched;
        used_move[m] = true;
        slot++;
      }
    }
  }
  return slot - 1;
}

// Enumerate the action set for the current decision into gr->eb_action_buf.
// Populated slots are flagged in gr->eb_action_present; returns the highest
// populated slot index + 1 (so callers iterate 0..N-1 and check the flag),
// or 0 if not branchable.
//
// Annotate already-populated subset_mode slots with their matching
// force-target so eb_emit_leaf can credit the deficit. Mirrors the
// PAIR-then-TILE-then-STRATUM priority order of eb_append_force_target_slots
// but operates on slots (not on the full move list), since subset_mode
// already enumerated every distinct (action_type, score, leave) action.
// Each slot is matched at most once; later slots aren't displaced.
static void eb_annotate_force_targets_to_slots(AutoplayWorker *w,
                                               GameRunner *gr, int max_slot) {
  ForceTable *ft = w->shared_data->force_table;
  if (!ft) return;
  Game *game = gr->game;
  // Same all-pass-leaf restriction as eb_append_force_target_slots: cell
  // diff ranges are bag=93 opener-derived, only valid when board is empty.
  if (board_get_tiles_played(game_get_board(game)) > 0) return;
  const int bag_count = bag_get_letters(game_get_bag(game)) + (RACK_SIZE);
  int target_count = 0;
  ForceTarget **targets = force_table_lookup(ft, bag_count, &target_count);
  if (target_count == 0) return;
  bool any_active = false;
  for (int i = 0; i < target_count; i++) {
    if (targets[i]->deficit > 0) { any_active = true; break; }
  }
  if (!any_active) return;
  (void)targets;

  const LetterDistribution *ld = game_get_ld(game);
  Rack leave;
  rack_set_dist_size(&leave, ld_get_size(ld));

  const int p_idx = game_get_player_on_turn_index(game);
  const int my_score =
      equity_to_int(player_get_score(game_get_player(game, p_idx)));
  const int opp_score =
      equity_to_int(player_get_score(game_get_player(game, 1 - p_idx)));
  const int cur_diff = my_score - opp_score;

  bool used_slot[EB_MAX_ACTIONS] = {false};
  for (int priority = FORCE_TARGET_PAIR;
       priority >= FORCE_TARGET_STRATUM; priority--) {
    for (int s = 1; s <= max_slot; s++) {
      if (used_slot[s] || !gr->eb_action_present[s]) continue;
      Move *mv = gr->eb_action_buf[s];
      const game_event_t mt = move_get_type(mv);
      if (mt != GAME_EVENT_TILE_PLACEMENT_MOVE && mt != GAME_EVENT_EXCHANGE) {
        continue;  // pass slot — never a force-target match
      }
      rack_reset(&leave);
      get_leave_for_move(mv, game, &leave);
      const int leave_len = (int)leave.number_of_letters;
      if (leave_len < 0 || leave_len >= 8) continue;
      const int score = equity_to_int(move_get_score(mv));
      const bool is_exch = (mt == GAME_EVENT_EXCHANGE);
      uint32_t leave_bm = 0;
      for (uint16_t i = 0; i < leave.dist_size && i < 32; i++) {
        if (leave.array[i] > 0) leave_bm |= ((uint32_t)1) << i;
      }
      int bucket_count = 0;
      ForceTargetSlot *bucket_slots = force_table_lookup_slots_by_shape(
          ft, bag_count, leave_len, is_exch ? 1 : 0, &bucket_count);
      uint32_t *bitmaps = force_table_lookup_bitmaps_by_shape(
          ft, bag_count, leave_len, is_exch ? 1 : 0);
      if (bucket_count == 0 || bitmaps == NULL) continue;
      ForceTarget *matched = NULL;
      for (int t = 0; t < bucket_count; t++) {
        const uint32_t req = bitmaps[t];
        if ((leave_bm & req) != req) continue;
        ForceTargetSlot *fs = &bucket_slots[t];
        if (fs->deficit <= 0 || (int)fs->kind != priority) continue;
        const int eff_diff = cur_diff + score;
        if (eff_diff < fs->diff_min || eff_diff > fs->diff_max) continue;
        if (fs->subleave_count == 2 &&
            fs->subleave_mls[0] == fs->subleave_mls[1] &&
            leave.array[fs->subleave_mls[0]] < 2) continue;
        if (fs->exchange == 0 && fs->leave_length >= 3) {
          const LeaveType lt = force_classify_leave(&leave, ld);
          if ((LeaveType)lt != (LeaveType)fs->leave_type) continue;
        }
        matched = fs->cold;
        break;
      }
      if (matched != NULL) {
        gr->eb_force_target_for_slot[s] = matched;
        used_slot[s] = true;
      }
    }
  }
}

// Modes:
//  - turn 1: never branched
//  - turn 2: branched only on bag-random P2 half (mix_random mode)
//  - turn 3-5: always branched
//  - turn 5 (no legal plays): subset fan-out — pass (if is_pass) +
//            every distinct exch subset
//  - turn 6: always branched, full subset fan-out — pass (if is_pass) +
//            every distinct (action_type, score, leave) action (pass +
//            exch subsets + plays)
//
// Pass is included only when the current player's rack is is_pass=1; for
// non-pass racks pass is never a meaningful action and gets omitted to
// halve compute.
//
// Slot layout is STABLE across runs and across is_pass/non-is_pass racks:
//   K=3 mode:    slot 0 = pass, slot 1 = best-exch, slot 2 = best-play
//   subset mode: slot 0 = pass, slots 1..M = exch subsets,
//                slots M+1..N = distinct play (score, leave) signatures
// Empty slots (pass omitted, or no legal play, etc.) keep their index and
// the iterator skips them via the present-flag — branch_id encoding stays
// reproducible regardless of which actions a particular position offers.
static int eb_enumerate_actions(AutoplayWorker *w, GameRunner *gr) {
  // Clear presence flags first so any early return leaves a sane state.
  for (int i = 0; i < EB_MAX_ACTIONS; i++) {
    gr->eb_action_present[i] = false;
    gr->eb_force_target_for_slot[i] = NULL;
  }
  gr->eb_natural_slot = -1;

  if (!gr->eb_active || gr->eb_n_snaps >= 6) return 0;
  if (board_get_tiles_played(game_get_board(gr->game)) > 0) return 0;
  const int turn = gr->eb_n_snaps + 1;
  const int target_turn = w->shared_data->eb_target_turn;
  const EbTurnRole role = eb_classify_turn(target_turn, turn);
  // REC_PRE and POST never fork — natural-play path drives the move (force
  // pass for REC_PRE via the role-driven force block; HastyBot for POST).
  if (role == EB_ROLE_REC_PRE || role == EB_ROLE_POST) return 0;

  const int p_idx = game_get_player_on_turn_index(gr->game);
  char canon[RACK_SIZE + 2] = {0};
  eb_canonical_rack(gr->game, p_idx, canon);
  const bool is_pass_rack =
      pass_cycle_lookup_is_pass(w->shared_data->pass_cycle_table, canon) == 1;

  MoveList *ml = w->eb_move_list;
  const MoveGenArgs gen_args = {
      .game = gr->game,
      .move_list = ml,
      .move_record_type = MOVE_RECORD_ALL,
      .move_sort_type = MOVE_SORT_EQUITY,
      .override_kwg = NULL,
      .thread_index = w->worker_index,
      .eq_margin_movegen = 0,
      .target_equity = EQUITY_MAX_VALUE,
      .target_leave_size_for_exchange_cutoff = UNSET_LEAVE_SIZE,
      .tiles_played_bv = NULL,
      .initial_tiles_bv = 0};
  generate_moves(&gen_args);
  const int n_moves = move_list_get_count(ml);
  move_list_sort_moves(ml);

  const LetterDistribution *ld = game_get_ld(gr->game);
  const uint16_t ld_size = ld_get_size(ld);
  int max_slot = -1;
  int play_slot = -1;  // populated slot index of the best-play action

  if (role == EB_ROLE_TARGET) {
    // Full fan-out: pass + every distinct (action_type, score, leave) action.
    // Blank-letter equivalents (MANE/SANE/CANE — same score and leave,
    // different blank assignment) collapse to one slot. Empirical max across
    // all 3.2M valid racks is 294 slots (1 pass + 127 exch + 166 play).
    bool has_play = false;
    for (int m = 0; m < n_moves; m++) {
      if (move_get_type(move_list_get_move(ml, m)) ==
          GAME_EVENT_TILE_PLACEMENT_MOVE) {
        has_play = true;
        break;
      }
    }
    move_set_as_pass(gr->eb_action_buf[0]);
    gr->eb_action_present[0] = true;
    max_slot = 0;
    Rack tmp_leave;
    rack_set_dist_size(&tmp_leave, ld_size);
    enum { SIG_LEN = 16 };
    static _Thread_local char slot_sig[EB_MAX_ACTIONS][SIG_LEN];
    int n_sigs = 0;
    int slot = 1;
    for (int phase = 0; phase < 2 && slot < EB_MAX_ACTIONS; phase++) {
      const game_event_t kind = (phase == 0)
          ? GAME_EVENT_EXCHANGE
          : GAME_EVENT_TILE_PLACEMENT_MOVE;
      if (kind == GAME_EVENT_TILE_PLACEMENT_MOVE && !has_play) continue;
      for (int m = 0; m < n_moves && slot < EB_MAX_ACTIONS; m++) {
        Move *cand = move_list_get_move(ml, m);
        if (move_get_type(cand) != kind) continue;
        rack_reset(&tmp_leave);
        get_leave_for_move(cand, gr->game, &tmp_leave);
        char leave_str[RACK_SIZE + 2];
        eb_render_leave(&tmp_leave, ld, leave_str, sizeof(leave_str));
        const int score = equity_to_int(move_get_score(cand));
        char sig[SIG_LEN];
        const char tc = (kind == GAME_EVENT_EXCHANGE) ? 'X' : 'P';
        snprintf(sig, sizeof(sig), "%c%d|%s", tc, score, leave_str);
        bool dup = false;
        for (int k = 0; k < n_sigs; k++) {
          if (strcmp(slot_sig[k], sig) == 0) { dup = true; break; }
        }
        if (dup) continue;
        memcpy(slot_sig[n_sigs++], sig, SIG_LEN);
        move_copy(gr->eb_action_buf[slot], cand);
        gr->eb_action_present[slot] = true;
        if (kind == GAME_EVENT_TILE_PLACEMENT_MOVE && play_slot < 0) {
          play_slot = slot;
        }
        if (slot > max_slot) max_slot = slot;
        slot++;
      }
    }
  } else {
    // OPP_FIRST / OPP_CLOSEST → 2 slots always: pass + best-equity-exch.
    // OPP_MID → if is_pass: same 2 slots; else 0 slots (natural play forces
    //          best-equity-exch via the role-driven force block).
    if (role == EB_ROLE_OPP_MID && !is_pass_rack) return 0;
    Move *best_exch = NULL;
    for (int m = 0; m < n_moves; m++) {
      Move *cand = move_list_get_move(ml, m);
      if (move_get_type(cand) == GAME_EVENT_EXCHANGE) {
        best_exch = cand;
        break;
      }
    }
    if (best_exch == NULL) return 0;  // Can't fork with no exchange option.
    move_set_as_pass(gr->eb_action_buf[0]);
    gr->eb_action_present[0] = true;
    move_copy(gr->eb_action_buf[1], best_exch);
    gr->eb_action_present[1] = true;
    max_slot = 1;
  }

  // Natural-slot scan — only meaningful at the TARGET turn (where the
  // divergence filter and snap recording need it). For OPP_* turns we don't
  // record, so natural_slot stays -1.
  if (role == EB_ROLE_TARGET && max_slot >= 0) {
    if (is_pass_rack) {
      gr->eb_natural_slot = 0;
    } else {
      for (int m = 0; m < n_moves && gr->eb_natural_slot < 0; m++) {
        Move *cand = move_list_get_move(ml, m);
        const game_event_t mt = move_get_type(cand);
        if (mt == GAME_EVENT_TILE_PLACEMENT_MOVE) {
          if (play_slot >= 0) gr->eb_natural_slot = play_slot;
          break;
        }
        if (mt == GAME_EVENT_EXCHANGE) {
          const int nt_cand = move_get_tiles_played(cand);
          for (int s = 1; s <= max_slot; s++) {
            if (!gr->eb_action_present[s]) continue;
            Move *slot_mv = gr->eb_action_buf[s];
            if (move_get_type(slot_mv) != GAME_EVENT_EXCHANGE) continue;
            if (move_get_tiles_played(slot_mv) != nt_cand) continue;
            uint8_t cnt_a[MAX_ALPHABET_SIZE] = {0};
            uint8_t cnt_b[MAX_ALPHABET_SIZE] = {0};
            for (int i = 0; i < nt_cand; i++) {
              cnt_a[move_get_tile(cand, i)]++;
              cnt_b[move_get_tile(slot_mv, i)]++;
            }
            if (memcmp(cnt_a, cnt_b, sizeof(cnt_a)) == 0) {
              gr->eb_natural_slot = s;
              break;
            }
          }
        }
      }
    }
  }

  // Force-table annotation: at TARGET, match populated slots to active
  // force-targets so eb_emit_leaf can credit deficits. Force-table fires
  // only at the target turn (validated at startup).
  if (role == EB_ROLE_TARGET &&
      w->shared_data->force_table != NULL &&
      (w->shared_data->eb_force_turns_mask & (1 << turn))) {
    eb_annotate_force_targets_to_slots(w, gr, max_slot);
  }

  // Need at least 2 populated slots for the fork to be meaningful.
  int populated = 0;
  for (int i = 0; i <= max_slot; i++) {
    if (gr->eb_action_present[i]) populated++;
  }
  return populated >= 2 ? max_slot + 1 : 0;
}

// Emit captured eb_snaps for this leaf branch with its eventual_outcome.
static void eb_emit_leaf(AutoplayWorker *w, GameRunner *gr, uint64_t branch_id) {
  EmptyBoardRecorder *ebr = w->shared_data->empty_board_recorder;
  EmptyBoardStrataRecorder *ebs = w->shared_data->empty_board_strata_recorder;
  if ((!ebr && !ebs) || gr->eb_n_snaps == 0) return;
  // Single-target-turn recording: emit one row at the target snap only.
  // Pre-target snaps are pass-cycle plumbing; post-target turns play
  // naturally with no fork (so the natural-post-Tk filter is trivially
  // satisfied for the target snap).
  const int target_turn = w->shared_data->eb_target_turn;
  const int idx = target_turn - 1;
  if (idx < 0 || idx >= gr->eb_n_snaps) return;
  const int s0 =
      equity_to_int(player_get_score(game_get_player(gr->game, 0)));
  const int s1 =
      equity_to_int(player_get_score(game_get_player(gr->game, 1)));
  const int p = gr->eb_snaps[idx].player_on_turn;
  const int my = (p == 0) ? s0 : s1;
  const int opp = (p == 0) ? s1 : s0;
  const int outcome = my > opp ? 2 : (my == opp ? 1 : 0);
  empty_board_recorder_write(
      ebr, gr->game_number, branch_id,
      gr->eb_snaps[idx].turn_on_empty_board, gr->eb_snaps[idx].rack,
      gr->eb_snaps[idx].opp_history, gr->eb_snaps[idx].action_kind,
      gr->eb_snaps[idx].action_repr, gr->eb_snaps[idx].action_size,
      gr->eb_snaps[idx].leave, outcome, gr->eb_p2_rack_source,
      gr->eb_snaps[idx].natural_slot);
  empty_board_strata_write(
      ebs, gr->game_number,
      branch_id, gr->eb_snaps[idx].turn_on_empty_board, gr->eb_snaps[idx].rack,
      gr->eb_snaps[idx].opp_history, gr->eb_snaps[idx].action_kind,
      gr->eb_snaps[idx].action_repr, gr->eb_snaps[idx].action_size,
      gr->eb_snaps[idx].leave, outcome, gr->eb_p2_rack_source,
      gr->eb_snaps[idx].natural_slot, gr->eb_snaps[idx].move_score,
      gr->eb_divergence_turn, gr->eb_p1_rack_source,
      gr->eb_p1_force_kind, gr->eb_p2_force_kind);

  // Force-target credit at the target snap.
  ForceTarget *ft_target = gr->eb_snap_force_target[target_turn];
  if (ft_target != NULL && w->shared_data->force_table != NULL) {
    const int p_idx = gr->eb_snaps[idx].player_on_turn;
    const int my_final =
        equity_to_int(player_get_score(game_get_player(gr->game, p_idx)));
    const int opp_final =
        equity_to_int(
            player_get_score(game_get_player(gr->game, 1 - p_idx)));
    const bool is_tie = (my_final == opp_final);
    const bool is_win = (my_final > opp_final);
    force_table_credit_game(w->shared_data->force_table, ft_target,
                            gr->eb_snaps[idx].move_score, is_win, is_tie);
  }
}

// Snapshot the on-turn player's rack as a canonical string (sorted A-Z,
// blanks last). Used to restore the rack on inject failure.
static void snapshot_player_rack(const Game *game, int p, char *out) {
  Rack *cur = player_get_rack(game_get_player((Game *)game, p));
  const LetterDistribution *ld = game_get_ld((Game *)game);
  const uint16_t dist_size = rack_get_dist_size(cur);
  int n = 0;
  for (uint16_t i = 1; i < dist_size && n < RACK_SIZE; i++) {
    const int c = rack_get_letter(cur, i);
    for (int k = 0; k < c && n < RACK_SIZE; k++) {
      out[n++] = ld->ld_ml_to_hl[i][0];
    }
  }
  const int nblanks = rack_get_letter(cur, 0);
  for (int k = 0; k < nblanks && n < RACK_SIZE; k++) {
    out[n++] = '?';
  }
  out[n] = '\0';
}

// Swap the on-turn player's rack to `target_rack` by returning their
// current 7 tiles to the bag and drawing the target. On failure (target
// not drawable from current bag state) restores the snapshot. Returns
// true iff target was successfully drawn.
static bool swap_player_rack(Game *game, int p, const char *target_rack) {
  char snapshot[RACK_SIZE + 2] = {0};
  snapshot_player_rack(game, p, snapshot);
  return_rack_to_bag(game, p);
  if (target_rack != NULL &&
      draw_rack_string_from_bag(game, p, target_rack) > 0) {
    return true;
  }
  // Restore — bag still holds the snapshot tiles (we just returned them).
  if (draw_rack_string_from_bag(game, p, snapshot) <= 0) {
    draw_to_full_rack(game, p);
  }
  return false;
}

// At the start of the target turn, pick a rack source uniformly from
// {pass, exch, bingo, rare, play} (5 cells × 20%) and inject a rack from
// the chosen source. Each pool is independent — missing pools leave the
// natural rack for that 1/5. Only fires when the on-turn player matches
// rare_target_player (auto-derived from target turn parity).
static void inject_target_turn_rack_by_category(
    AutoplayWorker *w, GameRunner *gr, uint64_t seed) {
  const int p = game_get_player_on_turn_index(gr->game);
  if (p != w->shared_data->rare_target_player) return;

  // Pick category 0..4 uniformly. 0=pass, 1=exch, 2=bingo, 3=rare, 4=play.
  const uint64_t h = seed * 0x9e3779b97f4a7c15ULL + 0x123456789abcdef0ULL;
  const int cat = (int)((h >> 33) % 5ULL);

  const char *target = NULL;
  if (cat == 0 && w->shared_data->pass_pool != NULL) {
    pass_cycle_sample_p1(w->shared_data->pass_pool, seed, &target);
  } else if (cat == 1 && w->shared_data->exch_pool != NULL) {
    pass_cycle_sample_p1(w->shared_data->exch_pool, seed, &target);
  } else if (cat == 2 && w->shared_data->bingo_pool != NULL) {
    pass_cycle_sample_p1(w->shared_data->bingo_pool, seed, &target);
  } else if (cat == 3 && w->shared_data->rare_rack_cells != NULL) {
    const int idx = rare_pool_sample_deficit_aware(
        w->shared_data->rare_rack_cells, seed);
    if (idx >= 0) {
      target = rare_pool_get_rack(w->shared_data->rare_rack_cells, idx);
    }
  } else if (cat == 4 && w->shared_data->play_pool != NULL) {
    pass_cycle_sample_p1(w->shared_data->play_pool, seed, &target);
  }
  if (target == NULL) return;  // pool unset → leave natural rack
  swap_player_rack(gr->game, p, target);
}

// Recursive DFS. Plays moves until a fork point or game-end. At a fork
// point, enumerates K actions, recursively explores each, then returns.
// At game-end, emits the leaf branch's records.
//
// branch_id encoding: each fork left-shifts by 8 bits and OR's in
// (slot_index + 1). Up to 8 forks fit in uint64; in practice the cycle
// caps at 6 turns so depth is at most 4-6.
static void play_eb_dfs(AutoplayWorker *w, GameRunner *gr,
                        uint64_t branch_id) {
  while (!game_runner_is_game_over(gr)) {
    // Target-turn source-mix injection: at the start of the target turn
    // (eb_n_snaps == target_turn - 1, board still empty) pick a rack source
    // uniformly from {pass, exch, bingo, rare, play} and inject a rack from
    // the chosen source. Skips if no pools loaded or on-turn player isn't the
    // rare_target_player.
    if (gr->eb_active &&
        gr->eb_n_snaps == w->shared_data->eb_target_turn - 1 &&
        board_get_tiles_played(game_get_board(gr->game)) == 0 &&
        (w->shared_data->pass_pool != NULL ||
         w->shared_data->exch_pool != NULL ||
         w->shared_data->bingo_pool != NULL ||
         w->shared_data->play_pool != NULL ||
         w->shared_data->rare_rack_cells != NULL)) {
      inject_target_turn_rack_by_category(w, gr, gr->seed ^ branch_id);
    }
    const int n_actions = eb_enumerate_actions(w, gr);
    if (n_actions == 0) {
      // Not a fork point — play one move using normal selection.
      game_runner_play_move(w, gr);
      continue;
    }

    // Snapshot enumerated actions to stack BEFORE recursing — eb_action_buf
    // is per-runner and inner forks during recursion will overwrite it.
    // n_actions is max_slot+1; iterate stable slot indices and skip absent.
    Move local_actions[EB_MAX_ACTIONS];
    bool local_present[EB_MAX_ACTIONS];
    ForceTarget *local_force_target[EB_MAX_ACTIONS];
    for (int s = 0; s < n_actions; s++) {
      local_present[s] = gr->eb_action_present[s];
      local_force_target[s] = gr->eb_force_target_for_slot[s];
      if (local_present[s]) {
        move_copy(&local_actions[s], gr->eb_action_buf[s]);
      }
    }
    // Snapshot natural_slot too — inner recursive enumerate calls overwrite
    // gr->eb_natural_slot, and the capture in each sibling iteration's
    // play_move needs THIS fork's natural slot (not whatever the inner
    // recursion left behind).
    const int fork_natural_slot = gr->eb_natural_slot;
    EbMetaSave saved_meta;
    eb_meta_save(gr, &saved_meta);
    // Per-recursion-level game checkpoint. A single shared per-runner buffer
    // would be overwritten by inner forks' game_copy and corrupt the outer
    // fork's restore, leaving the game in a foreign player's mid-iteration
    // state.
    Game *saved_game = game_duplicate(gr->game);

    const int fork_turn = saved_meta.n_snaps + 1;  // turn about to be played
    for (int s = 0; s < n_actions; s++) {
      if (!local_present[s]) continue;
      gr->eb_forced_move = &local_actions[s];
      gr->eb_natural_slot = fork_natural_slot;
      // eb_divergence_turn: FIRST non-natural turn (-1 if pure natural).
      // eb_last_divergence_turn: LAST non-natural turn (-1 if pure natural).
      // eb_emit_leaf uses last_divergence_turn to filter: emit snap at Tk
      // only if last_divergence_turn <= Tk (no divergence after Tk), i.e.
      // the post-Tk playout was fully natural.
      gr->eb_divergence_turn = saved_meta.divergence_turn;
      gr->eb_last_divergence_turn = saved_meta.last_divergence_turn;
      gr->eb_n_divergences = saved_meta.n_divergences;
      if (s != fork_natural_slot) {
        if (gr->eb_divergence_turn == -1) {
          gr->eb_divergence_turn = fork_turn;
        }
        gr->eb_last_divergence_turn = fork_turn;
        gr->eb_n_divergences++;
      }
      // Track force-target attached to this slot at this turn (NULL if
      // natural). eb_emit_leaf consults this per-snap to credit deficits.
      if (fork_turn >= 1 && fork_turn <= 6) {
        gr->eb_snap_force_target[fork_turn] = local_force_target[s];
      }
      game_runner_play_move(w, gr);
      gr->eb_forced_move = NULL;
      // Encode the SLOT INDEX (not iteration index) so the action at each
      // fork-depth is recoverable regardless of which slots happened to be
      // populated for that position.
      play_eb_dfs(w, gr, (branch_id << 8) | (uint64_t)(s + 1));
      game_copy(gr->game, saved_game);
      eb_meta_restore(gr, &saved_meta);
    }
    game_destroy(saved_game);
    return;
  }
  eb_emit_leaf(w, gr, branch_id);
}

void play_autoplay_game_or_game_pair(AutoplayWorker *autoplay_worker,
                                     GameRunner *game_runner1,
                                     GameRunner *game_runner2,
                                     const AutoplayIterOutput *iter_output) {
  const int starting_player_index = (int)(iter_output->iter_count % 2);

  // EB single-target-turn dispatch: single-runner mode, each pair explores
  // a small fork tree (forks at OPP_FIRST/OPP_CLOSEST/OPP_MID(is_pass) and
  // full fan-out at TARGET) and emits one record per leaf at the target.
  if (autoplay_worker->shared_data->eb_branch_active &&
      autoplay_worker->shared_data->pass_cycle_table != NULL) {
    // Pin starter to P1 (index 0) so the on-turn player at the target turn
    // always matches the auto-derived rare_target_player. With alternating
    // starter, half of pairs would have wrong-parity on-turn at target and
    // miss the inject. eb_p1_force_kind drives the STARTER's rack class —
    // with starter pinned, this is global P1.
    //
    // Opponent at OPP_FIRST gets a 50/50 pool sample (is_pass=1 vs is_pass=0)
    // alternating per pair via iter_count bit 0. Recording player's
    // pre-target rack is replaced at TARGET inject, so its force_kind is
    // immaterial — pin to 0.
    const int target = autoplay_worker->shared_data->eb_target_turn;
    const int opp_force_kind = (int)(iter_output->iter_count & 1ULL);
    game_runner1->eb_p1_rack_source = 0;
    game_runner1->eb_p2_rack_source = 0;
    if (target % 2 == 0) {
      // Even target → P2 records, P1 is opponent. Alternate P1's force_kind.
      game_runner1->eb_p1_force_kind = opp_force_kind;
      game_runner1->eb_p2_force_kind = 0;
    } else {
      // Odd target → P1 records, P2 is opponent. Alternate P2's force_kind.
      // (target=1 has no opp pre-target turns; opp_force_kind is unused but
      // the assignment is harmless.)
      game_runner1->eb_p1_force_kind = 0;
      game_runner1->eb_p2_force_kind = opp_force_kind;
    }
    game_runner_start(autoplay_worker, game_runner1, iter_output,
                      /*starter=*/0, 0);
    play_eb_dfs(autoplay_worker, game_runner1, 0);
    autoplay_add_game(autoplay_worker, game_runner1, false);
    return;
  }

  // Opening-pass paired mode: both runners use the SAME starting player so
  // the forced rack and force-pass logic apply to the same player slot in
  // both games. The branch (pass vs play) differentiates them via
  // pair_game_number.
  const bool opening_pass_paired =
      autoplay_worker->shared_data->opening_pass_table != NULL &&
      game_runner2 != NULL;
  const bool pass_cycle_paired =
      autoplay_worker->shared_data->pass_cycle_table != NULL &&
      game_runner2 != NULL;
  const int runner2_starter = (opening_pass_paired || pass_cycle_paired)
                                  ? starting_player_index
                                  : 1 - starting_player_index;
  game_runner_start(autoplay_worker, game_runner1, iter_output,
                    starting_player_index, game_runner2 ? 1 : 0);
  if (game_runner2) {
    game_runner_start(autoplay_worker, game_runner2, iter_output,
                      runner2_starter, 2);
  }
  bool games_are_divergent = false;
  while (true) {
    const Move *move1 = NULL;
    bool game1_is_over = game_runner_is_game_over(game_runner1);
    if (!game1_is_over) {
      move1 = game_runner_play_move(autoplay_worker, game_runner1);
    }

    const Move *move2 = NULL;
    bool game2_is_over = true;
    if (game_runner2) {
      game2_is_over = game_runner_is_game_over(game_runner2);
      if (!game2_is_over) {
        move2 = game_runner_play_move(autoplay_worker, game_runner2);
      }
    }

    if (game1_is_over && game2_is_over) {
      break;
    }

    // It is guaranteed that at least one move is not null
    // at this point.
    if (!games_are_divergent &&
        (!move1 || !move2 ||
         compare_moves_without_equity(move1, move2, true) != -1)) {
      games_are_divergent = true;
    }
  }
  if (autoplay_worker->args.print_boards) {
    StringBuilder *output = string_builder_create();
    if (game_runner1->pair_game_number == 0) {
      string_builder_add_formatted_string(
          output, "\n=== Game %llu (Final) ===\n",
          (unsigned long long)game_runner1->game_number + 1);
    } else {
      string_builder_add_formatted_string(
          output, "\n=== Game Pair %llu, Game %d (Final) ===\n",
          (unsigned long long)game_runner1->game_number + 1,
          game_runner1->pair_game_number);
    }
    string_builder_add_game(game_runner1->game, NULL,
                            autoplay_worker->args.game_string_options, NULL,
                            output);
    if (game_runner2) {
      string_builder_add_formatted_string(
          output, "\n=== Game Pair %llu, Game %d (Final) ===\n",
          (unsigned long long)game_runner2->game_number + 1,
          game_runner2->pair_game_number);
      string_builder_add_game(game_runner2->game, NULL,
                              autoplay_worker->args.game_string_options, NULL,
                              output);
    }
    thread_control_print(autoplay_worker->args.thread_control,
                         string_builder_peek(output));
    string_builder_destroy(output);
  }
  autoplay_add_game(autoplay_worker, game_runner1, games_are_divergent);
  if (game_runner2) {
    // We do not check for min leave counts here because leave gen
    // does not use game pairs and therefore does not have a second
    // game runner.
    autoplay_add_game(autoplay_worker, game_runner2, games_are_divergent);
  }

  // Opening-pass mode: record per-game outcomes and (if paired) the
  // paired-difference statistics for the forced-rack player's perspective.
  OpeningPassTable *op_table =
      autoplay_worker->shared_data->opening_pass_table;
  if (op_table && game_runner1->opening_pass_rack_idx >= 0) {
    const int rack_idx = game_runner1->opening_pass_rack_idx;
    const int p1 = game_runner1->opening_pass_player_index;
    const int s1_my = equity_to_int(
        player_get_score(game_get_player(game_runner1->game, p1)));
    const int s1_opp = equity_to_int(
        player_get_score(game_get_player(game_runner1->game, 1 - p1)));
    const int outcome1 =
        s1_my > s1_opp ? 2 : (s1_my == s1_opp ? 1 : 0);
    opening_pass_record(op_table, rack_idx,
                        game_runner1->opening_pass_branch, outcome1);

    if (game_runner2 && game_runner2->opening_pass_rack_idx == rack_idx) {
      const int p2 = game_runner2->opening_pass_player_index;
      const int s2_my = equity_to_int(
          player_get_score(game_get_player(game_runner2->game, p2)));
      const int s2_opp = equity_to_int(
          player_get_score(game_get_player(game_runner2->game, 1 - p2)));
      const int outcome2 =
          s2_my > s2_opp ? 2 : (s2_my == s2_opp ? 1 : 0);
      opening_pass_record(op_table, rack_idx,
                          game_runner2->opening_pass_branch, outcome2);
      // Pair outcome (pass=branch 0, play=branch 1)
      const int pass_outcome = (game_runner1->opening_pass_branch == 0)
                                   ? outcome1
                                   : outcome2;
      const int play_outcome = (game_runner1->opening_pass_branch == 0)
                                   ? outcome2
                                   : outcome1;
      opening_pass_record_pair(op_table, rack_idx, pass_outcome,
                               play_outcome);
    }
  }

  // Pass-cycle mode: only record games where the 6-pass cycle fired from
  // the OPENING (turns 1-6, board still empty). Late-game lockups where
  // the cycle fires after tiles were placed don't belong to the
  // pass-strategy dataset (those are different positions).
  PassCycleTable *pct = autoplay_worker->shared_data->pass_cycle_table;
  if (pct && game_runner1->pass_cycle_active) {
    const bool g1_six_pass =
        game_get_game_end_reason(game_runner1->game) ==
            GAME_END_REASON_CONSECUTIVE_ZEROS &&
        board_get_tiles_played(game_get_board(game_runner1->game)) == 0;
    if (g1_six_pass) {
      const int bp1 = game_runner1->pass_cycle_bot_player;
      const int ms1 = equity_to_int(
          player_get_score(game_get_player(game_runner1->game, bp1)));
      const int os1 = equity_to_int(
          player_get_score(game_get_player(game_runner1->game, 1 - bp1)));
      const int outcome1 = ms1 > os1 ? 2 : (ms1 == os1 ? 1 : 0);
      char hist1[160] = {0};
      int off1 = 0;
      for (int i = 0; i < game_runner1->pass_cycle_n_moves; i++) {
        off1 += snprintf(hist1 + off1, sizeof(hist1) - (size_t)off1,
                         i == 0 ? "%s" : "|%s",
                         game_runner1->pass_cycle_history[i]);
      }
      char p1_final[16] = {0}, p2_final[16] = {0};
      pass_cycle_stringify_rack(game_runner1->game, bp1, p1_final);
      pass_cycle_stringify_rack(game_runner1->game, 1 - bp1, p2_final);
      pass_cycle_record(pct, game_runner1->game_number,
                        game_runner1->pass_cycle_bot_rack_str,
                        game_runner1->pass_cycle_opp_rack_str,
                        p1_final, p2_final,
                        game_runner1->pass_cycle_branch, outcome1,
                        game_runner1->turn_number, hist1);
    }
    if (game_runner2 && game_runner2->pass_cycle_active) {
      const bool g2_six_pass =
          game_get_game_end_reason(game_runner2->game) ==
              GAME_END_REASON_CONSECUTIVE_ZEROS &&
          board_get_tiles_played(game_get_board(game_runner2->game)) == 0;
      if (g2_six_pass) {
        const int bp2 = game_runner2->pass_cycle_bot_player;
        const int ms2 = equity_to_int(
            player_get_score(game_get_player(game_runner2->game, bp2)));
        const int os2 = equity_to_int(
            player_get_score(game_get_player(game_runner2->game, 1 - bp2)));
        const int outcome2 = ms2 > os2 ? 2 : (ms2 == os2 ? 1 : 0);
        char hist2[160] = {0};
        int off2 = 0;
        for (int i = 0; i < game_runner2->pass_cycle_n_moves; i++) {
          off2 += snprintf(hist2 + off2, sizeof(hist2) - (size_t)off2,
                           i == 0 ? "%s" : "|%s",
                           game_runner2->pass_cycle_history[i]);
        }
        char p1_final2[16] = {0}, p2_final2[16] = {0};
        pass_cycle_stringify_rack(game_runner2->game, bp2, p1_final2);
        pass_cycle_stringify_rack(game_runner2->game, 1 - bp2, p2_final2);
        pass_cycle_record(pct, game_runner2->game_number,
                          game_runner2->pass_cycle_bot_rack_str,
                          game_runner2->pass_cycle_opp_rack_str,
                          p1_final2, p2_final2,
                          game_runner2->pass_cycle_branch, outcome2,
                          game_runner2->turn_number, hist2);
      }
    }
  }

  // Empty-board recorder (slice 1): emit one row per captured snapshot with
  // eventual_outcome from that snap's player_on_turn perspective. Records are
  // emitted regardless of how the game ended — the value-function dataset
  // wants every (state, action, outcome) triple, not just cycle-end games.
  EmptyBoardRecorder *ebr = autoplay_worker->shared_data->empty_board_recorder;
  if (ebr) {
    GameRunner *runners[2] = {game_runner1, game_runner2};
    for (int gi = 0; gi < 2; gi++) {
      GameRunner *gr = runners[gi];
      if (!gr || !gr->eb_active || gr->eb_n_snaps == 0) continue;
      const int s0 = equity_to_int(
          player_get_score(game_get_player(gr->game, 0)));
      const int s1 = equity_to_int(
          player_get_score(game_get_player(gr->game, 1)));
      for (int i = 0; i < gr->eb_n_snaps; i++) {
        const int p = gr->eb_snaps[i].player_on_turn;
        const int my = (p == 0) ? s0 : s1;
        const int opp = (p == 0) ? s1 : s0;
        const int outcome = my > opp ? 2 : (my == opp ? 1 : 0);
        empty_board_recorder_write(
            ebr, gr->game_number, (uint64_t)gr->pass_cycle_branch,
            gr->eb_snaps[i].turn_on_empty_board, gr->eb_snaps[i].rack,
            gr->eb_snaps[i].opp_history, gr->eb_snaps[i].action_kind,
            gr->eb_snaps[i].action_repr, gr->eb_snaps[i].action_size,
            gr->eb_snaps[i].leave, outcome, 0, -1);
        empty_board_strata_write(
            autoplay_worker->shared_data->empty_board_strata_recorder,
            gr->game_number, (uint64_t)gr->pass_cycle_branch,
            gr->eb_snaps[i].turn_on_empty_board, gr->eb_snaps[i].rack,
            gr->eb_snaps[i].opp_history, gr->eb_snaps[i].action_kind,
            gr->eb_snaps[i].action_repr, gr->eb_snaps[i].action_size,
            gr->eb_snaps[i].leave, outcome, 0, -1,
            gr->eb_snaps[i].move_score, -1, 0, 0, 0);
      }
    }
  }
}

bool target_min_leave_count_reached(AutoplayWorker *autoplay_worker) {
  const LeavegenSharedData *leavegen_shared_data =
      autoplay_worker->shared_data->leavegen_shared_data;
  return leavegen_shared_data && rack_list_get_racks_below_target_count(
                                     leavegen_shared_data->rack_list) == 0;
}

void autoplay_single_generation(AutoplayWorker *autoplay_worker,
                                GameRunner *game_runner1,
                                GameRunner *game_runner2) {
  ThreadControl *thread_control = autoplay_worker->args.thread_control;
  AutoplayIterOutput iter_output;
  while (
      // Check if autoplay was exited by the user.
      thread_control_get_status(thread_control) !=
          THREAD_CONTROL_STATUS_USER_INTERRUPT &&
      // Check if the maximum iteration has been reached.
      !autoplay_get_next_iter_output(autoplay_worker->shared_data,
                                     &iter_output) &&
      // Check if the target minimum leave count has been reached.
      // This will never be true for the default autoplay mode.
      !target_min_leave_count_reached(autoplay_worker)) {
    play_autoplay_game_or_game_pair(autoplay_worker, game_runner1, game_runner2,
                                    &iter_output);
  }
}

void autoplay_leave_gen(AutoplayWorker *autoplay_worker,
                        GameRunner *game_runner) {
  AutoplaySharedData *shared_data = autoplay_worker->shared_data;
  LeavegenSharedData *lg_shared_data = shared_data->leavegen_shared_data;
  for (int i = 0; i < lg_shared_data->num_gens; i++) {
    autoplay_single_generation(autoplay_worker, game_runner, NULL);
    checkpoint_wait(lg_shared_data->postgen_checkpoint, shared_data);
    if (thread_control_get_status(shared_data->thread_control) ==
        THREAD_CONTROL_STATUS_USER_INTERRUPT) {
      break;
    }
  }
}

// - The sim args for autoplay share the same inference results, since only one
//   inference will be running at a time per autoplay worker.
// - The game of the sim args needs to be set explicitly before each move, since
//   there is only one pair of p1 and p2 sim args but potentially 2 games if
//   using game pairs.
void init_sim_args_for_player(AutoplayWorker *autoplay_worker,
                              int player_index) {
  SimArgs *sim_args = (player_index == 0) ? &autoplay_worker->args.p1_sim_args
                                          : &autoplay_worker->args.p2_sim_args;
  sim_args->bai_options.parent_worker_thread_index =
      autoplay_worker->worker_index;
  sim_args->inference_args.parent_worker_thread_index =
      autoplay_worker->worker_index;
  sim_args->inference_results = autoplay_worker->inference_results;
}

void *autoplay_worker(void *uncasted_autoplay_worker) {
  AutoplayWorker *autoplay_worker = (AutoplayWorker *)uncasted_autoplay_worker;
  const AutoplayArgs *args = &autoplay_worker->args;
  GameRunner *game_runner1 = game_runner_create(autoplay_worker);
  init_sim_args_for_player(autoplay_worker, 0);
  init_sim_args_for_player(autoplay_worker, 1);
  GameRunner *game_runner2 = NULL;
  switch (args->type) {
  case AUTOPLAY_TYPE_DEFAULT:
    if (args->use_game_pairs) {
      game_runner2 = game_runner_create(autoplay_worker);
    }
    autoplay_single_generation(autoplay_worker, game_runner1, game_runner2);
    game_runner_destroy(game_runner2);
    break;
  case AUTOPLAY_TYPE_LEAVE_GEN:
    autoplay_leave_gen(autoplay_worker, game_runner1);
    break;
  }

  game_runner_destroy(game_runner1);
  return NULL;
}

void parse_min_rack_targets(const AutoplayArgs *args,
                            const StringSplitter *split_min_rack_targets,
                            int *min_rack_targets, ErrorStack *error_stack) {
  int num_gens = string_splitter_get_number_of_items(split_min_rack_targets);
  for (int i = 0; i < num_gens; i++) {
    const char *item = string_splitter_get_item(split_min_rack_targets, i);
    if (is_string_empty_or_whitespace(item)) {
      error_stack_push(
          error_stack, ERROR_STATUS_AUTOPLAY_MALFORMED_MINIMUM_LEAVE_TARGETS,
          get_formatted_string("found an empty value for one or more of the "
                               "minimum rack targets: %s",
                               args->num_games_or_min_rack_targets));
      return;
    }
    min_rack_targets[i] = string_to_int(item, error_stack);
    if (!error_stack_is_empty(error_stack) || min_rack_targets[i] < 0) {
      error_stack_push(
          error_stack, ERROR_STATUS_AUTOPLAY_MALFORMED_MINIMUM_LEAVE_TARGETS,
          get_formatted_string("failed to parse minimum rack targets: %s",
                               args->num_games_or_min_rack_targets));
      return;
    }
  }
}

void valid_autoplay_results_options(const AutoplayResults *autoplay_results,
                                    const AutoplayArgs *args,
                                    ErrorStack *error_stack) {
  const uint64_t options = autoplay_results_get_options(autoplay_results);
  if (options == 0) {
    return;
  }
  if (options != autoplay_results_build_option(AUTOPLAY_RECORDER_TYPE_GAME) &&
      args->use_game_pairs) {
    error_stack_push(
        error_stack, ERROR_STATUS_AUTOPLAY_INVALID_OPTIONS,
        string_duplicate(
            "the game pairs setting can only be used with the games recorder"));
    return;
  }
}

void autoplay(const AutoplayArgs *args, AutoplayResults *autoplay_results,
              ErrorStack *error_stack) {
  const bool is_leavegen_mode = args->type == AUTOPLAY_TYPE_LEAVE_GEN;
  int num_gens = 1;
  int *min_rack_targets = NULL;
  uint64_t first_gen_num_games;
  if (is_leavegen_mode) {
    StringSplitter *split_min_rack_targets =
        split_string(args->num_games_or_min_rack_targets, ',', false);
    num_gens = string_splitter_get_number_of_items(split_min_rack_targets);
    min_rack_targets = malloc_or_die((sizeof(int)) * (num_gens));
    parse_min_rack_targets(args, split_min_rack_targets, min_rack_targets,
                           error_stack);
    string_splitter_destroy(split_min_rack_targets);
    if (!error_stack_is_empty(error_stack)) {
      free(min_rack_targets);
      error_stack_push(
          error_stack, ERROR_STATUS_AUTOPLAY_MALFORMED_MINIMUM_LEAVE_TARGETS,
          get_formatted_string("failed to parse minimum rack targets: %s",
                               args->num_games_or_min_rack_targets));
      return;
    }
    first_gen_num_games = UINT64_MAX;
  } else {
    first_gen_num_games =
        string_to_uint64(args->num_games_or_min_rack_targets, error_stack);
    if (!error_stack_is_empty(error_stack)) {
      error_stack_push(error_stack, ERROR_STATUS_AUTOPLAY_MALFORMED_NUM_GAMES,
                       get_formatted_string(
                           "failed to parse the specified number of games: %s",
                           args->num_games_or_min_rack_targets));
      return;
    }
  }

  valid_autoplay_results_options(autoplay_results, args, error_stack);
  if (!error_stack_is_empty(error_stack)) {
    return;
  }

  ThreadControl *thread_control = args->thread_control;

  autoplay_results_reset(autoplay_results);

  KLV *klv = NULL;
  bool show_divergent_results = args->use_game_pairs;
  if (is_leavegen_mode) {
    // We can use player index 0 here since it is guaranteed that
    // players share the the KLV.
    klv = players_data_get_klv(args->game_args->players_data, 0);
    show_divergent_results = false;
  }

  const int autoplay_num_threads = args->num_threads;

  AutoplayResults **autoplay_results_list =
      malloc_or_die((sizeof(AutoplayResults *)) * (autoplay_num_threads));

  AutoplaySharedData *shared_data = autoplay_shared_data_create(
      args, autoplay_num_threads, first_gen_num_games, autoplay_results,
      autoplay_results_list, klv, num_gens, min_rack_targets);

  AutoplayWorker **autoplay_workers =
      malloc_or_die((sizeof(AutoplayWorker *)) * (autoplay_num_threads));
  cpthread_t *worker_ids =
      malloc_or_die((sizeof(cpthread_t)) * (autoplay_num_threads));

  for (int thread_index = 0; thread_index < autoplay_num_threads;
       thread_index++) {
    autoplay_workers[thread_index] = autoplay_worker_create(
        args, autoplay_results, thread_index, shared_data);
    autoplay_results_list[thread_index] =
        autoplay_workers[thread_index]->autoplay_results;
    cpthread_create(&worker_ids[thread_index], autoplay_worker,
                    autoplay_workers[thread_index]);
  }

  autoplay_results_set_status_data(
      autoplay_results, autoplay_results_list, autoplay_num_threads, false,
      args->human_readable, show_divergent_results);

  for (int thread_index = 0; thread_index < autoplay_num_threads;
       thread_index++) {
    cpthread_join(worker_ids[thread_index]);
  }

  // The stats have already been combined in leavegen mode
  if (!is_leavegen_mode) {
    autoplay_results_consolidate(autoplay_results_list, autoplay_num_threads,
                                 autoplay_results);
  }

  autoplay_results_set_status_data(autoplay_results, NULL, 0, true,
                                   args->human_readable,
                                   show_divergent_results);

  free(autoplay_results_list);

  for (int thread_index = 0; thread_index < autoplay_num_threads;
       thread_index++) {
    autoplay_worker_destroy(autoplay_workers[thread_index]);
  }

  free(autoplay_workers);
  free(worker_ids);
  autoplay_shared_data_destroy(shared_data);
  free(min_rack_targets);

  // Only reload KLV if it was modified during leavegen
  if (is_leavegen_mode) {
    players_data_reload(args->game_args->players_data, PLAYERS_DATA_TYPE_KLV,
                        args->data_paths, error_stack);
  }

  char *autoplay_results_string = autoplay_results_to_string(
      autoplay_results, args->human_readable, show_divergent_results);
  thread_control_print(thread_control, autoplay_results_string);
  free(autoplay_results_string);
}
