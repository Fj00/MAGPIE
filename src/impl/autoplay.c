
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
  EmptyBoardRecorder *empty_board_recorder;
  EmptyBoardStrataRecorder *empty_board_strata_recorder;
  // Slice 2: K-way fork branching at empty-board cycle-alive turns. Activated
  // by MAGPIE_EMPTY_BOARD_BRANCH=1. When set, play_autoplay_game_or_game_pair
  // dispatches a single-runner DFS that explores forks at branchable turns.
  bool eb_branch_active;
  // Slice 2c: 50/50 mix between pool-sampled P2 (current default; cycle
  // favorable, deep-cycle data) and bag-random P2 (realistic-opp half;
  // turn 2 branched explicitly to capture pass/exch/play V at turn 2).
  // Activated by MAGPIE_EMPTY_BOARD_MIX_RANDOM=1; alternates per pair_id.
  bool eb_mix_random;
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
  autoplay_worker->force_leaves_capacity = 0;
  if (shared_data->force_table) {
    autoplay_worker->force_move_list = move_list_create(2000);
    autoplay_worker->force_leaves_capacity = 2000;
    autoplay_worker->force_leaves =
        malloc_or_die(sizeof(Rack) * (size_t)autoplay_worker->force_leaves_capacity);
  }
  autoplay_worker->eb_move_list = NULL;
  if (shared_data->eb_branch_active) {
    autoplay_worker->eb_move_list = move_list_create(2000);
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
  shared_data->eb_mix_random = false;
  const char *eb_mix = getenv("MAGPIE_EMPTY_BOARD_MIX_RANDOM");
  if (eb_mix && eb_mix[0] != '\0' && eb_mix[0] != '0') {
    shared_data->eb_mix_random = true;
    fprintf(stderr,
            "empty_board: mix-random ENABLED — alternating pool / "
            "bag-random P2 per pair, turn 2 branched on bag-random half\n");
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
  // Slice 2c mix-random: 0 = pool-sampled P2 (default); 1 = bag-random P2.
  // Set by play_autoplay_game_or_game_pair before game_runner_start; consumed
  // by the rack-draw branch in game_runner_start, by the turn-2 branchability
  // check in eb_enumerate_actions, and emitted in every record.
  int eb_p2_random;
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
  int eb_n_divergences;
  // Slot 0=pass; slots 1..N=enumerated actions in stable order.
  // Capacity covers worst case: pass + 127 distinct exch subsets of a 7-tile
  // rack + 1 best-play + slack. eb_action_present[i] is true iff slot i is
  // populated this enumeration; the DFS skips false slots.
#define EB_MAX_ACTIONS 132
  Move *eb_action_buf[EB_MAX_ACTIONS];
  bool eb_action_present[EB_MAX_ACTIONS];
  int eb_n_action_buf;
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
  game_runner->eb_p2_random = 0;
  game_runner->eb_natural_slot = -1;
  game_runner->eb_divergence_turn = -1;
  game_runner->eb_n_divergences = 0;
  game_runner->eb_n_action_buf = 0;
  for (int i = 0; i < EB_MAX_ACTIONS; i++) {
    game_runner->eb_action_buf[i] = NULL;
  }
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
    const char *p1_rack = NULL;
    const char *p2_rack = NULL;
    bool sampled = false;
    if (game_runner->eb_p2_random) {
      // Mix-random half: P1 from pool (force-pass via is_pass lookup at
      // turn 1), P2 drawn naturally from the bag after P1's tiles removed.
      pass_cycle_sample_p1(pct, iter_output->iter_count, &p1_rack);
      sampled = (p1_rack != NULL);
    } else if (pass_cycle_sample_racks(pct, iter_output->iter_count, &p1_rack,
                                       &p2_rack)) {
      sampled = true;
    }
    if (sampled) {
      // P1 is the starting player (the one who passes in branch 0).
      const int p1 = starting_player_index;
      const int p2 = 1 - starting_player_index;
      const int n1 = draw_rack_string_from_bag(game, p1, p1_rack);
      if (n1 > 0) {
        int n2 = 1;
        if (game_runner->eb_p2_random) {
          // P2 random: just refill from remaining bag.
          draw_to_full_rack(game, p2);
        } else {
          n2 = draw_rack_string_from_bag(game, p2, p2_rack);
        }
        if (n2 <= 0) {
          // Should not happen given pass_cycle_sample_racks filtering,
          // but fall back to random draw just in case.
          draw_to_full_rack(game, p2);
        }
        game_runner->pass_cycle_active = true;
        game_runner->pass_cycle_branch = (pair_game_number == 2) ? 1 : 0;
        game_runner->pass_cycle_bot_player = p1;
        game_runner->pass_cycle_bot_rack_str = p1_rack;
        game_runner->pass_cycle_opp_rack_str = p2_rack;
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
  uint32_t leave_bitmaps[2048] = {0};
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
        if (cur_diff < slot->diff_min || cur_diff > slot->diff_max) {
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
  ForceTable *ft = autoplay_worker->shared_data->force_table;
  if (ft != NULL && game_runner->pending_force_target != NULL) {
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
  gr->eb_n_divergences = s->n_divergences;
}

// Enumerate the action set for the current decision into gr->eb_action_buf.
// Populated slots are flagged in gr->eb_action_present; returns the highest
// populated slot index + 1 (so callers iterate 0..N-1 and check the flag),
// or 0 if not branchable.
//
// Modes:
//  - turn 1: never branched
//  - turn 2: branched only on bag-random P2 half (mix_random mode)
//  - turn 3-5: always branched
//  - turn 5 (no legal plays): subset fan-out — pass (if is_pass) +
//            every distinct exch subset
//  - turn 6: always branched, full subset fan-out — pass (if is_pass) +
//            every distinct exch subset + best-play
//
// Pass is included only when the current player's rack is is_pass=1; for
// non-pass racks pass is never a meaningful action and gets omitted to
// halve compute. Best-exch and best-play (K=3 mode) or all exch subsets
// (subset mode) are always included.
//
// Slot layout is STABLE across runs and across is_pass/non-is_pass racks:
//   K=3 mode:    slot 0 = pass, slot 1 = best-exch, slot 2 = best-play
//   subset mode: slot 0 = pass, slots 1..N = exch subsets in equity order,
//                slot N+1 = best-play (turn 6 only)
// Empty slots (pass omitted, or no legal play, etc.) keep their index and
// the iterator skips them via the present-flag — branch_id encoding stays
// reproducible regardless of which actions a particular position offers.
static int eb_enumerate_actions(AutoplayWorker *w, GameRunner *gr) {
  // Clear presence flags first so any early return leaves a sane state.
  for (int i = 0; i < EB_MAX_ACTIONS; i++) gr->eb_action_present[i] = false;
  gr->eb_natural_slot = -1;

  if (!gr->eb_active || gr->eb_n_snaps >= 6) return 0;
  if (board_get_tiles_played(game_get_board(gr->game)) > 0) return 0;
  const int turn = gr->eb_n_snaps + 1;
  // T1 never branched (the cycle anchor — starter's force-pass on is_pass
  // rack is the experimental design that opens the cycle subtree).
  // T2-T6 always branched for full V-function coverage of action
  // counterfactuals at every cycle-alive empty-board decision.
  if (turn == 1) return 0;

  const int p_idx = game_get_player_on_turn_index(gr->game);
  // Pass slot inclusion (per spec):
  //   T5, T6: pass always included — opp_history accumulated by then makes
  //           pass a meaningful counterfactual even for natural racks.
  //   T3, T4: pass only if rack is is_pass=1 — otherwise pass is never
  //           strategically correct at these early-cycle decisions.
  //   T2 (random mode): same as T3/T4 — only for is_pass racks.
  // is_pass status is also used downstream to decide the natural slot:
  // is_pass racks force-pass via pass_cycle, so natural=slot 0 regardless
  // of equity; non-is_pass racks fall through to HastyBot equity.
  char canon[RACK_SIZE + 2] = {0};
  eb_canonical_rack(gr->game, p_idx, canon);
  const bool is_pass_rack =
      pass_cycle_lookup_is_pass(w->shared_data->pass_cycle_table, canon) == 1;
  const bool include_pass = is_pass_rack || turn >= 5;

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

  bool has_play = false;
  for (int m = 0; m < n_moves; m++) {
    if (move_get_type(move_list_get_move(ml, m)) ==
        GAME_EVENT_TILE_PLACEMENT_MOVE) {
      has_play = true;
      break;
    }
  }

  // T5 and T6 both use subset-mode exchange fan-out (skipping the blank
  // at T5, plus 1-pointers at T6 — see filter below). T2-T4 stay K=3
  // (one best-equity exchange counterfactual).
  const bool subset_mode = (turn >= 5);
  const bool include_play_in_subset = subset_mode && has_play;
  int max_slot = -1;
  int play_slot = -1;  // populated slot index of the best-play action

  // Slot 0 always reserved for pass; populate only if rack is_pass.
  if (include_pass) {
    move_set_as_pass(gr->eb_action_buf[0]);
    gr->eb_action_present[0] = true;
    max_slot = 0;
  }

  if (subset_mode) {
    // Filter subset-mode exchanges to drop strictly-dominated subsets:
    //   T6 (cycle terminus): drop any exchange containing blank or
    //     1-point tiles — exchanging cheap tiles when the game ends
    //     immediately can never be right.
    //   T5 (no legal plays): drop any exchange containing the blank only.
    //     Exchanging 1-pointers can still be correct here if the opponent
    //     plays after, so we keep those subsets.
    const LetterDistribution *ld_filt = game_get_ld(gr->game);
    const int min_score = (turn == 6) ? 2 : 1;
    int slot = 1;
    for (int m = 0; m < n_moves && slot < EB_MAX_ACTIONS; m++) {
      Move *cand = move_list_get_move(ml, m);
      if (move_get_type(cand) != GAME_EVENT_EXCHANGE) continue;
      const int nt = move_get_tiles_played(cand);
      bool ok = true;
      for (int i = 0; i < nt; i++) {
        const MachineLetter tml = move_get_tile(cand, i);
        if (equity_to_int(ld_get_score(ld_filt, tml)) < min_score) {
          ok = false;
          break;
        }
      }
      if (!ok) continue;
      move_copy(gr->eb_action_buf[slot], cand);
      gr->eb_action_present[slot] = true;
      if (slot > max_slot) max_slot = slot;
      slot++;
    }
    if (include_play_in_subset && slot < EB_MAX_ACTIONS) {
      for (int m = 0; m < n_moves; m++) {
        Move *cand = move_list_get_move(ml, m);
        if (move_get_type(cand) == GAME_EVENT_TILE_PLACEMENT_MOVE) {
          move_copy(gr->eb_action_buf[slot], cand);
          gr->eb_action_present[slot] = true;
          play_slot = slot;
          if (slot > max_slot) max_slot = slot;
          break;
        }
      }
    }
  } else {
    // K=3 mode: slot 1 = best-exch, slot 2 = best-play.
    Move *best_exch = NULL;
    Move *best_play = NULL;
    for (int m = 0; m < n_moves && (!best_exch || !best_play); m++) {
      Move *cand = move_list_get_move(ml, m);
      const game_event_t mt = move_get_type(cand);
      if (!best_exch && mt == GAME_EVENT_EXCHANGE) best_exch = cand;
      else if (!best_play && mt == GAME_EVENT_TILE_PLACEMENT_MOVE)
        best_play = cand;
    }
    if (best_exch) {
      move_copy(gr->eb_action_buf[1], best_exch);
      gr->eb_action_present[1] = true;
      if (1 > max_slot) max_slot = 1;
    }
    if (best_play) {
      move_copy(gr->eb_action_buf[2], best_play);
      gr->eb_action_present[2] = true;
      play_slot = 2;
      if (2 > max_slot) max_slot = 2;
    }
  }

  // Determine the natural slot — what HastyBot/force-pass would have
  // chosen from OUR ENUMERATED SET. Scan movegen output in equity order;
  // the first move whose action matches an enumerated slot is natural.
  // For exchanges that's a multiset comparison of exchanged tiles; for
  // tile placements any play matches our single best-play slot (since
  // movegen emits in equity order and our play_slot holds the best).
  // is_pass racks bypass equity entirely — pass_cycle force-pass would
  // dominate regardless.
  if (max_slot >= 0) {
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
            // Multiset equality on machine-letter tile lists.
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
  // Skip multi-divergence leaves (counterfactual-of-counterfactual data).
  // Pure natural (n_div=0) and single-anchor (n_div=1) leaves are kept.
  if (gr->eb_n_divergences > 1) return;
  const int s0 =
      equity_to_int(player_get_score(game_get_player(gr->game, 0)));
  const int s1 =
      equity_to_int(player_get_score(game_get_player(gr->game, 1)));
  for (int i = 0; i < gr->eb_n_snaps; i++) {
    // Skip pre-divergence snaps — they're identical to the natural-chain
    // snaps emitted by the pure-natural leaf, so re-emitting them just
    // duplicates rows. Pure natural (divergence_turn == -1) emits all
    // snaps; an anchor=k leaf only emits its Tk..T6 rows.
    if (gr->eb_divergence_turn != -1 &&
        gr->eb_snaps[i].turn_on_empty_board < gr->eb_divergence_turn) {
      continue;
    }
    const int p = gr->eb_snaps[i].player_on_turn;
    const int my = (p == 0) ? s0 : s1;
    const int opp = (p == 0) ? s1 : s0;
    const int outcome = my > opp ? 2 : (my == opp ? 1 : 0);
    empty_board_recorder_write(
        ebr, gr->game_number, branch_id,
        gr->eb_snaps[i].turn_on_empty_board, gr->eb_snaps[i].rack,
        gr->eb_snaps[i].opp_history, gr->eb_snaps[i].action_kind,
        gr->eb_snaps[i].action_repr, gr->eb_snaps[i].action_size,
        gr->eb_snaps[i].leave, outcome, gr->eb_p2_random,
        gr->eb_snaps[i].natural_slot);
    empty_board_strata_write(
        w->shared_data->empty_board_strata_recorder, gr->game_number,
        branch_id, gr->eb_snaps[i].turn_on_empty_board, gr->eb_snaps[i].rack,
        gr->eb_snaps[i].opp_history, gr->eb_snaps[i].action_kind,
        gr->eb_snaps[i].action_repr, gr->eb_snaps[i].action_size,
        gr->eb_snaps[i].leave, outcome, gr->eb_p2_random,
        gr->eb_snaps[i].natural_slot, gr->eb_snaps[i].move_score,
        gr->eb_divergence_turn);
  }
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
    for (int s = 0; s < n_actions; s++) {
      local_present[s] = gr->eb_action_present[s];
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
      // Anchor tracking: a non-natural sibling at any T2-T6 fork counts
      // as a divergence. First divergence stamps eb_divergence_turn;
      // subsequent divergences bump eb_n_divergences (used to skip
      // multi-divergence leaves).
      gr->eb_divergence_turn = saved_meta.divergence_turn;
      gr->eb_n_divergences = saved_meta.n_divergences;
      if (s != fork_natural_slot) {
        if (gr->eb_divergence_turn == -1) {
          gr->eb_divergence_turn = fork_turn;
        }
        gr->eb_n_divergences++;
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

  // Slice 2 dispatch: K-way fork DFS over the empty-board cycle subtree.
  // Single-runner mode — game_runner2 is unused. Each pair_id explores its
  // own fork tree, emitting per-leaf records.
  if (autoplay_worker->shared_data->eb_branch_active &&
      autoplay_worker->shared_data->pass_cycle_table != NULL) {
    int starter = starting_player_index;
    int p2_random = 0;
    if (autoplay_worker->shared_data->eb_mix_random) {
      // Decorrelate p2_rack_source from starting_player_index: bit 0 picks
      // pool vs random, bit 1 picks starter.
      p2_random = (int)(iter_output->iter_count & 1ULL);
      starter = (int)((iter_output->iter_count >> 1) & 1ULL);
    }
    game_runner1->eb_p2_random = p2_random;
    game_runner_start(autoplay_worker, game_runner1, iter_output, starter, 0);
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
            gr->eb_snaps[i].move_score, -1);
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
