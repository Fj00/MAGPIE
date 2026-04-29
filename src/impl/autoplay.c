
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
  if (shared_data->force_table) {
    autoplay_worker->force_move_list = move_list_create(2000);
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
  int pass_cycle_bot_player;
  const char *pass_cycle_bot_rack_str;
  const char *pass_cycle_opp_rack_str;
  // Per-turn (rack, move) trace for downstream counterfactual reconstruction.
  // Each entry encodes "<rack>:<move>" where move is "P" or "X<tiles>".
  char pass_cycle_history[6][24];
  int pass_cycle_n_moves;
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
  return game_runner;
}

void game_runner_destroy(GameRunner *game_runner) {
  if (!game_runner) {
    return;
  }
  game_destroy(game_runner->game);
  game_destroy(game_runner->game_one_move_behind);
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
  if (!used_forced_rack && game_runner->shared_data->pass_cycle_table) {
    PassCycleTable *pct = game_runner->shared_data->pass_cycle_table;
    const char *p1_rack = NULL;
    const char *p2_rack = NULL;
    if (pass_cycle_sample_racks(pct, iter_output->iter_count, &p1_rack,
                                &p2_rack)) {
      // P1 is the starting player (the one who passes in branch 0).
      const int p1 = starting_player_index;
      const int p2 = 1 - starting_player_index;
      const int n1 = draw_rack_string_from_bag(game, p1, p1_rack);
      if (n1 > 0) {
        const int n2 = draw_rack_string_from_bag(game, p2, p2_rack);
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
  return game_over(game_runner->game) ||
         (game_runner->shared_data->leavegen_shared_data &&
          bag_get_letters(game_get_bag(game_runner->game)) < (RACK_SIZE));
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
  const MoveGenArgs mg_args = {
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
  generate_moves(&mg_args);
  const int n_moves = move_list_get_count(ml);
  if (n_moves == 0) {
    return NULL;
  }

  const LetterDistribution *ld = game_get_ld(game);
  Rack leave;
  rack_set_dist_size(&leave, ld_get_size(ld));
  for (int priority = FORCE_TARGET_PAIR; priority >= FORCE_TARGET_STRATUM;
       priority--) {
    for (int t = 0; t < target_count; t++) {
      ForceTarget *target = targets[t];
      if (target->deficit <= 0 || (int)target->kind != priority) {
        continue;
      }
      // Compute current score-diff once per (target × move-list pass) so the
      // force_target_matches diff-range check has it available. Diff is the
      // player_on_turn's score minus opp's at the moment of decision.
      const int p_idx = game_get_player_on_turn_index(game);
      const int my_score =
          equity_to_int(player_get_score(game_get_player(game, p_idx)));
      const int opp_score =
          equity_to_int(player_get_score(game_get_player(game, 1 - p_idx)));
      const int cur_diff = my_score - opp_score;
      for (int m = 0; m < n_moves; m++) {
        Move *move = move_list_get_move(ml, m);
        get_leave_for_move(move, game, &leave);
        const int score = equity_to_int(move_get_score(move));
        if (!force_target_matches(target, &leave, score, cur_diff)) {
          continue;
        }
        // force_target_matches does not check the cons/mixed/vowel type
        // because it does not carry the LetterDistribution. Do it here.
        if (target->exchange == 0 && target->leave_length >= 3) {
          if (force_classify_leave(&leave, ld) != target->leave_type) {
            continue;
          }
        }
        game_runner->force_triggered = true;
        // Capture the force-turn state. The deficit is credited at game-end
        // (autoplay_add_game) once the diff/outcome are final, so stratum
        // targets only debit when min(wins, losses) actually moves.
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
  // Opening-pass mode: on the opening turn of the starter (the player with
  // the forced rack) in the pass branch, override the move with a pass.
  if (game_runner->opening_pass_rack_idx >= 0 &&
      game_runner->opening_pass_branch == 0 &&
      game_runner->turn_number == 0 &&
      player_on_turn_index == game_runner->opening_pass_player_index) {
    Move *spare = move_list_get_spare_move(
        autoplay_worker->move_lists[player_on_turn_index]);
    move_set_as_pass(spare);
    move = spare;
  }
  // Pass-cycle mode: P1 force-passes while the board is still empty (the
  // empty-board rack classification only applies until tiles are placed).
  // The board can stay empty across multiple turns if the opponent keeps
  // exchanging. Once any tile is placed, P1 reverts to normal play.
  if (!move && game_runner->pass_cycle_active &&
      game_runner->pass_cycle_branch == 0 &&
      player_on_turn_index == game_runner->pass_cycle_bot_player &&
      board_get_tiles_played(game_get_board(game)) == 0) {
    Move *spare = move_list_get_spare_move(
        autoplay_worker->move_lists[player_on_turn_index]);
    move_set_as_pass(spare);
    move = spare;
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

  play_move(move, game, NULL);
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

void play_autoplay_game_or_game_pair(AutoplayWorker *autoplay_worker,
                                     GameRunner *game_runner1,
                                     GameRunner *game_runner2,
                                     const AutoplayIterOutput *iter_output) {
  const int starting_player_index = (int)(iter_output->iter_count % 2);
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
