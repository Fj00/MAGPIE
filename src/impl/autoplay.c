
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
#include "../ent/endgame_results.h"
#include "../ent/equity.h"
#include "../ent/force_table.h"
#include "../ent/static_leaves.h"
#include "endgame.h"
#include "../ent/vmodel.h"
#include "../ent/vmodel_features.h"
#include "../ent/vmodel_picks.h"
#include "../ent/game.h"
#include "../ent/empty_board.h"
#include "../ent/empty_board_strata.h"
#include "../ent/opening_pass.h"
#include "../ent/pass_cycle.h"
#include "../ent/outcome_priors.h"
#include "../ent/play_index.h"
#include "../ent/rare_pool.h"
#include "../ent/t6_baseline.h"
#include "../ent/trajectory_recorder.h"
#include "../ent/position_pool.h"
#include "../ent/leave_deficit.h"
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
#include "cgp.h"
#include "gameplay.h"
#include "move_gen.h"
#include "rack_list.h"
#include "simmer.h"
#include <math.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Diagnostic counter — incremented once per eb_emit_leaf call that gets
// past the early-return guards. Paired with force-table credit counters in
// the progress line to investigate the multi-credit regression (cf.
// noble-popping-kitten plan).
static _Atomic uint64_t g_eb_leaves_emitted = 0;
// Per-leaf cost decomposition counters (noble-popping-kitten plan, T5
// vs T6 ms/leaf gap). All reset per progress tick alongside leaves.
//   post_turns: HastyBot turns played past the TARGET turn, summed across
//               leaves. Divide by leaves → avg POST length.
//   movegen_calls: eb_enumerate_actions calls that reached generate_moves
//                  (i.e. role != REC_PRE/POST and board empty).
//   game_copies: game_duplicate + game_copy invocations inside play_eb_dfs.
//   target_fanout_total / _count: sum and count of TARGET-role populated
//                                  slot counts → avg fanout per TARGET
//                                  enumeration.
//   force_decrements: force_table_decrement_target calls from eb_emit_leaf.
static _Atomic uint64_t g_eb_post_turns_total = 0;
static _Atomic uint64_t g_eb_dfs_movegen_calls = 0;
static _Atomic uint64_t g_eb_dfs_game_copies = 0;
static _Atomic uint64_t g_eb_target_fanout_total = 0;
static _Atomic uint64_t g_eb_target_fanout_count = 0;
static _Atomic uint64_t g_eb_force_decrement_calls = 0;
// Blank-exchange tracing (investigating why K1 strata never record a
// thrown blank). Three stages along the fanout path at the TARGET turn:
//   exch_gen:   exchange moves emitted by generate_moves
//   exch_genbk: of those, ones whose thrown tiles include a blank
//   exch_slot:  exchange slots that survived dedup into the fanout
//   exch_slotbk: of those, ones with a blank in the thrown tiles
// If genbk>0 but slotbk==0 the loss is in eb_enumerate_actions dedup;
// if genbk==0 the loss is in generate_moves for this code path.
static _Atomic uint64_t g_eb_exch_gen = 0;
static _Atomic uint64_t g_eb_exch_genbk = 0;
static _Atomic uint64_t g_eb_exch_slot = 0;
static _Atomic uint64_t g_eb_exch_slotbk = 0;
// Multi-credit pipeline cell-count diagnostics. Each is incremented at a
// distinct stage so we can see whether cells are lost between annotate
// (slot population), dfs (per-fork-iter propagation into snap arrays),
// and emit (credit-loop call site). If all three match per progress
// tick, the multi-credit path is internally consistent.
static _Atomic uint64_t g_eb_annotate_cells_added = 0;
static _Atomic uint64_t g_eb_dfs_cells_propagated = 0;
static _Atomic uint64_t g_eb_emit_cells_credited = 0;
// Per-kind STRATUM-added counter inside eb_annotate's collect scan.
// Index = leave_length [0..5]; tracks how many length-N STRATUM cells
// were added to slots by eb_annotate.
static _Atomic uint64_t g_eb_annot_stratum_by_len[6] = {0};
// Per-leave-length count of STRATUM slots SEEN in collect scan, regardless
// of whether they pass predicates. If this is 0 for L>=3, the STRATUM cells
// aren't even being iterated (bucket lookup issue or cap-eviction).
static _Atomic uint64_t g_eb_stratum_seen_by_len[6] = {0};
// Diagnose length-3+ filtering: count cells reaching the type check
// at length>=3 (in collect scan), vs those that pass it.
static _Atomic uint64_t g_eb_l3plus_type_check_hit = 0;
static _Atomic uint64_t g_eb_l3plus_type_check_passed = 0;
// Per-leaf-length-3+ enumeration count: how many length-3+ leaves get
// to eb_annotate's outer loop at all?
static _Atomic uint64_t g_eb_l3plus_outer_loop = 0;
// Per-priority gate matched count at length>=3
static _Atomic uint64_t g_eb_l3p_gate_match_by_prio[4] = {0};
static _Atomic uint64_t g_eb_l3p_gate_iter_by_prio[4] = {0};

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
  // T6 cell-driven (rack, play) index. When set via MAGPIE_PLAY_INDEX_DIR
  // the deficit-aware rack sampler in play_index.c replaces the 5-pool
  // category sampler at the recording turn. The 5 pool fields above are
  // ignored when play_index is loaded.
  PlayIndex *play_index;
  // Phase 3: outcome-aware sampler. When MAGPIE_OUTCOME_PRIORS_PATH is
  // set, play_index_sample_rack_outcome_aware blends per-bucket P(W/L)
  // priors with deficit coverage. lambda controls the blend weight
  // (0 = pure deficit, default 1.0).
  OutcomePriors *outcome_priors;
  double outcome_priors_lambda;

  // Phase 0 capture: when MAGPIE_PRE_T6_CAPTURE=path is set, write one
  // CSV row per game at T6 entry: P1's rack + bag tiles. Used to build
  // the pre_t6_pool.csv distribution for skip-pre-T6 game setup.
  FILE *pre_t6_capture_file;
  cpthread_mutex_t pre_t6_capture_mutex;

  // Phase 2: when MAGPIE_T6_BASELINE_TASK_FILE and
  // MAGPIE_T6_BASELINE_OUT_FILE are both set, the worker loop
  // switches to baseline-mode: pulls (target_rack, opp_rack,
  // action_repr, n_games) tasks from the task file, runs each as a
  // T6-from-scratch batch, writes aggregated outcomes to out file.
  T6BaselineState *t6_baseline;
  // 0 = P1 receives the T6 inject (recording), 1 = P2. Set by
  // MAGPIE_EB_RARE_TARGET=p1|p2. Default p2 (T6 is P2's turn in the
  // canonical EB cycle).
  int rare_target_player;
  // T6 source-mix probability for the rare-pool slot. Adaptive ramp:
  //   rare_frac = MIN + (MAX - MIN) * completion^2
  // where completion = 1 - current_deficit / initial_deficit.
  // Quadratic ramp keeps rare_frac near MIN early (let natural pools
  // crush common cells) and bumps to MAX as the slow tail dominates.
  //
  // Defaults MIN=0.20, MAX=0.60: starts at 20/20/20/20/20 (uniform 5-way),
  // ramps to 60/10/10/10/10 at full completion.
  // Override via MAGPIE_EB_RARE_FRAC_MIN / MAGPIE_EB_RARE_FRAC_MAX, or
  // MAGPIE_EB_RARE_FRAC (legacy: sets both MIN and MAX to the same value
  // for static fraction).
  double rare_frac_min;
  double rare_frac_max;
  // Cached at startup from force_table_total_remaining; the denominator
  // for the completion ratio. Stays constant once set.
  int64_t initial_total_deficit;
  // Phase 4: late-stage targeted scheduler. When true, EB target-turn
  // injection skips category sampling and calls
  // play_index_pick_starved_rack instead. Set via MAGPIE_EB_LATE_STAGE=1
  // (force-on at startup) or flipped by the auto-switch trigger when
  // broad efficiency drops below eb_auto_late_threshold. Atomic so
  // worker reads stay safe across the one-shot transition.
  _Atomic bool eb_late_stage;
  // Auto-switch threshold: when (credits_landed_per_window / window_size)
  // < threshold, flip eb_late_stage on. 0.0 disables auto-switch (C
  // default). Set via MAGPIE_EB_AUTO_LATE_THRESHOLD — the launcher
  // script `launch_eb_run.sh` owns the policy (default 1.0 = trip when
  // broad falls to about late-stage's own per-game credit rate).
  double eb_auto_late_threshold;
  EmptyBoardRecorder *empty_board_recorder;
  EmptyBoardStrataRecorder *empty_board_strata_recorder;
  // Slice 2: K-way fork branching at empty-board cycle-alive turns. Activated
  // by MAGPIE_EMPTY_BOARD_BRANCH=1. When set, play_autoplay_game_or_game_pair
  // dispatches a single-runner DFS that explores forks at branchable turns.
  bool eb_branch_active;
  // Single-target-turn recording: which turn (1..6) is the recording
  // target. All other turns are pass-cycle plumbing or post-target natural
  // play. Set by MAGPIE_EB_TARGET_TURN; default 6.
  int eb_target_turn;
  // V-model inference hooks — one slot per game turn (1..6). Each .vmt
  // file declares its own training turn (m->turn); the loader places it
  // in the matching slot. game_runner_play_move dispatches based on the
  // current game turn: if `vmodels[turn]` is non-NULL AND board is
  // empty, the V model replaces HastyBot's pick. Other turns / non-
  // empty boards use the existing decision path unchanged. Env vars:
  //   MAGPIE_VMODEL_PATHS       — colon-separated .vmt file list; each
  //                               loads into its training-turn slot
  //   MAGPIE_VMODEL_PATH        — backward-compat single file (placed
  //                               in slot matching its training turn)
  //   MAGPIE_VMODEL_LEAVES_6    — CSW24_gen_6.csv  (static_leave L1..L6)
  //   MAGPIE_VMODEL_LEAVES_7    — CSW24_7tile_gen_6.csv  (L7 racks)
  //   MAGPIE_VMODEL_TURN        — (deprecated) which game-turn to apply
  //                               the model at; ignored when more than
  //                               one .vmt is loaded
  VModel *vmodels[7];   // indexed by game turn (1..6); slot 0 unused
  StaticLeaves *vmodel_static_leaves;
  int vmodel_any_loaded;
  // Precomputed rack -> chosen move lookup tables, one slot per game
  // turn. Each .picks file declares its training turn; the loader
  // places it in the matching slot. When set, `try_vmodel_picks_pick`
  // short-circuits ahead of the inference-based `vmodel_pick_top_move`,
  // collapsing the per-call cost to ~200 ns (binary search) instead of
  // tens of µs of movegen + scoring. Env var:
  //   MAGPIE_VMODEL_PICKS_PATHS — colon-separated .picks files
  VModelPicks *vmodel_picks[7];
  int vmodel_picks_any_loaded;
  const LetterDistribution *ld;
  // Trajectory recorder: when MAGPIE_TRAJECTORY_RECORDER=<dir> is set,
  // every game turn (before the move plays) writes a row to
  // <dir>/positions/bag_<NN>.csv. Used by the 91-8 V-model pipeline.
  TrajectoryRecorder *trajectory_recorder;
  // Position-pool mode (91-8 rack injection): when MAGPIE_POSITION_POOL=<file>
  // is set, the worker loop loads mid-game CGPs from the pool instead of
  // starting fresh games, plays the on-turn move + POST, and records via the
  // trajectory recorder (same positions/bag_<NN>.csv format → combines with
  // natural). position_pool_next is the shared work-stealing cursor.
  PositionPool *position_pool;
  _Atomic int position_pool_next;
  // Per-(bag,leave,diff-bin) deficit table gating the position-pool fan-out
  // for the L0-L4 per-leave track (NULL = no gating). Env MAGPIE_PP_LEAVE_*.
  LeaveDeficit *leave_deficit;
  // Targeted rack pool (MAGPIE_PP_RACK_POOL): when set, the fan-out injects an
  // on-turn rack sampled from this list (e.g. vowel-heavy + consonant-heavy
  // racks) via swap_player_rack, instead of a random re-roll — so the rare
  // typed strata (6_vowel, 5_vowel, 6_cons…) get covered efficiently rather
  // than waiting for random draws to produce them.
  PositionPool *rack_pool;
  // Opener-conditioned pool (MAGPIE_OPENER_POOL): for the opening bags, replay
  // a specific opener move on an empty board (sets the starting rack,
  // movegen-matches score+leave, plays it; sign '+' passes the opponent) to
  // synthesize a (bag, diff) board the natural pool never produces (blank-burn
  // openers, rare scores). Then the same inject + fan-out runs at the result.
  // Mutually exclusive with position_pool; reuses position_pool_next as cursor.
  OpenerPool *opener_pool;
} AutoplaySharedData;

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
    // One-shot announcement (runs under iter_mutex, so a plain static
    // is safe). Gives downstream tooling — chain_train.sh — an
    // unambiguous "deficit hit 0" signal instead of parsing per-tick
    // deficit numbers that miss the final partial window.
    static bool exhaust_announced = false;
    if (!exhaust_announced) {
      exhaust_announced = true;
      fprintf(stderr, "force_table: deficit reached 0 — force table "
                      "EXHAUSTED, stopping autoplay (clean completion)\n");
    }
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
  if (shared_data->eb_branch_active || shared_data->vmodel_any_loaded) {
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
  // Optional resume: replay W/L state from a prior run's
  // force_remaining.csv dump. Cross-session iteration without losing
  // drain progress.
  const char *resume_path = getenv("MAGPIE_FORCE_TABLE_RESUME");
  if (resume_path && resume_path[0] != '\0' &&
      shared_data->force_table != NULL) {
    force_table_resume_from_dump(shared_data->force_table, resume_path,
                                  args->game_args->ld);
  }
  shared_data->stop_on_force_exhaust =
      shared_data->force_table != NULL &&
      getenv("MAGPIE_FORCE_STOP_ON_EXHAUST") != NULL;
  // Snapshot initial total deficit for the rare_frac adaptive ramp's
  // completion-ratio denominator. Done once at startup; stays constant.
  if (shared_data->force_table != NULL) {
    shared_data->initial_total_deficit =
        force_table_total_remaining(shared_data->force_table);
  }

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
  shared_data->play_index = NULL;
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
  shared_data->rare_frac_min = 0.20;
  shared_data->rare_frac_max = 0.60;
  // Legacy single-value override: both MIN and MAX set to the same value.
  const char *rare_frac_env = getenv("MAGPIE_EB_RARE_FRAC");
  if (rare_frac_env && rare_frac_env[0] != '\0') {
    double f = atof(rare_frac_env);
    if (f >= 0.0 && f <= 1.0) {
      shared_data->rare_frac_min = f;
      shared_data->rare_frac_max = f;
    } else {
      fprintf(stderr,
              "empty_board: invalid MAGPIE_EB_RARE_FRAC=%s (must be in "
              "[0.0, 1.0]); ignored\n", rare_frac_env);
    }
  }
  // Per-bound overrides take precedence over the legacy single value.
  const char *rfmin = getenv("MAGPIE_EB_RARE_FRAC_MIN");
  if (rfmin && rfmin[0] != '\0') {
    double f = atof(rfmin);
    if (f >= 0.0 && f <= 1.0) shared_data->rare_frac_min = f;
  }
  const char *rfmax = getenv("MAGPIE_EB_RARE_FRAC_MAX");
  if (rfmax && rfmax[0] != '\0') {
    double f = atof(rfmax);
    if (f >= 0.0 && f <= 1.0) shared_data->rare_frac_max = f;
  }
  if (shared_data->rare_frac_max < shared_data->rare_frac_min) {
    shared_data->rare_frac_max = shared_data->rare_frac_min;
  }
  fprintf(stderr,
          "empty_board: rare-pool injection ramp = [%.2f .. %.2f] "
          "(quadratic in completion fraction)\n",
          shared_data->rare_frac_min, shared_data->rare_frac_max);
  shared_data->initial_total_deficit = 0;
  // Snapshot initial deficit AFTER the force_table is loaded; deferred
  // a few lines below where shared_data->force_table is set.
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
  // Optional bag-tile rare pool — merged into the same in-memory pool
  // so the deficit-aware sampler treats leave-cell and bag-cell racks
  // uniformly. If the leave-cell pool wasn't provided, this can stand
  // alone (rare_pool_create handles the first file).
  const char *bag_rare_pool = getenv("MAGPIE_EB_BAG_RARE_POOL");
  if (bag_rare_pool && bag_rare_pool[0] != '\0' && shared_data->force_table) {
    if (shared_data->rare_rack_cells) {
      rare_pool_load_more(shared_data->rare_rack_cells, bag_rare_pool,
                           shared_data->force_table, shared_data->ld);
    } else {
      shared_data->rare_rack_cells = rare_pool_create(
          bag_rare_pool, shared_data->force_table, shared_data->ld);
    }
  }
  // Cell-driven (rack, play) index. When set, supersedes the 5-pool
  // sampler at the recording turn (rack picked via deficit-aware
  // top-K cell scan; existing fanout consumes the chosen rack as
  // before).
  const char *play_index_dir = getenv("MAGPIE_PLAY_INDEX_DIR");
  if (play_index_dir && play_index_dir[0] != '\0' && shared_data->force_table) {
    shared_data->play_index = play_index_create(
        play_index_dir, shared_data->force_table, shared_data->ld);
  }

  // Phase 3: outcome-aware sampler priors. Loaded only when both the
  // priors path is set AND play_index is loaded (priors only have effect
  // through the play_index outcome-aware sampler).
  shared_data->outcome_priors = NULL;
  shared_data->outcome_priors_lambda = 1.0;
  const char *priors_path = getenv("MAGPIE_OUTCOME_PRIORS_PATH");
  if (priors_path && priors_path[0] != '\0' && shared_data->play_index) {
    shared_data->outcome_priors = outcome_priors_load(priors_path);
  }
  const char *priors_lambda_env = getenv("MAGPIE_EB_PRIOR_LAMBDA");
  if (priors_lambda_env && priors_lambda_env[0] != '\0') {
    shared_data->outcome_priors_lambda = atof(priors_lambda_env);
  }
  if (shared_data->outcome_priors) {
    fprintf(stderr,
            "outcome_priors: %d buckets loaded, lambda=%.3f\n",
            outcome_priors_num_buckets(shared_data->outcome_priors),
            shared_data->outcome_priors_lambda);
  }

  // Phase 4: late-stage targeted scheduler flag. Requires play_index.
  atomic_store_explicit(&shared_data->eb_late_stage, false,
                        memory_order_relaxed);
  // Threshold policy lives in the launcher (launch_eb_run.sh sets
  // MAGPIE_EB_AUTO_LATE_THRESHOLD). C default = 0 (disabled). When run
  // bare without the launcher, no auto-switch unless the env var is set.
  shared_data->eb_auto_late_threshold = 0.0;
  const char *late_stage_env = getenv("MAGPIE_EB_LATE_STAGE");
  if (late_stage_env && late_stage_env[0] == '1') {
    if (shared_data->play_index) {
      atomic_store_explicit(&shared_data->eb_late_stage, true,
                            memory_order_relaxed);
      fprintf(stderr, "eb_late_stage: ON (cell-driven targeted picker)\n");
    } else {
      fprintf(stderr,
              "eb_late_stage: requested but play_index not loaded; ignoring\n");
    }
  }
  const char *auto_thr_env = getenv("MAGPIE_EB_AUTO_LATE_THRESHOLD");
  if (auto_thr_env && auto_thr_env[0] != '\0') {
    shared_data->eb_auto_late_threshold = atof(auto_thr_env);
  }
  if (shared_data->play_index &&
      !atomic_load_explicit(&shared_data->eb_late_stage,
                            memory_order_relaxed) &&
      shared_data->eb_auto_late_threshold > 0.0) {
    fprintf(stderr,
            "eb_auto_late_switch: armed (threshold=%.4f credits/game)\n",
            shared_data->eb_auto_late_threshold);
  }

  // Phase 2 baseline: task-driven T6-from-scratch mode.
  shared_data->t6_baseline = NULL;
  const char *t6_task = getenv("MAGPIE_T6_BASELINE_TASK_FILE");
  const char *t6_out = getenv("MAGPIE_T6_BASELINE_OUT_FILE");
  const char *t6_pool = getenv("MAGPIE_T6_BASELINE_POOL_FILE");
  if (t6_task && t6_task[0] && t6_out && t6_out[0] &&
      t6_pool && t6_pool[0]) {
    shared_data->t6_baseline =
        t6_baseline_state_create(t6_task, t6_out, t6_pool);
    if (shared_data->t6_baseline) {
      fprintf(stderr,
              "t6_baseline: tasks=%s out=%s pool=%s\n",
              t6_task, t6_out, t6_pool);
    }
  }

  // Phase 0 capture: pre-T6 P1 rack + bag snapshot file.
  shared_data->pre_t6_capture_file = NULL;
  cpthread_mutex_init(&shared_data->pre_t6_capture_mutex);
  const char *pre_t6_capture = getenv("MAGPIE_PRE_T6_CAPTURE");
  if (pre_t6_capture && pre_t6_capture[0] != '\0') {
    shared_data->pre_t6_capture_file = fopen(pre_t6_capture, "w");
    if (shared_data->pre_t6_capture_file) {
      fprintf(shared_data->pre_t6_capture_file, "p1_rack,bag\n");
      fflush(shared_data->pre_t6_capture_file);
      fprintf(stderr,
              "pre_t6_capture: writing per-game P1-rack+bag to %s\n",
              pre_t6_capture);
    } else {
      fprintf(stderr,
              "pre_t6_capture: cannot open %s for write\n", pre_t6_capture);
    }
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
    // Loud warning if any pool is missing — rack-class distribution at
    // TARGET will be skewed toward natural (pass-cycle pool) for the
    // missing 1/5 share(s).
    int missing = !shared_data->pass_pool + !shared_data->exch_pool +
                  !shared_data->bingo_pool + !shared_data->play_pool +
                  !shared_data->rare_rack_cells;
    if (missing > 0) {
      fprintf(stderr,
              "WARNING: %d of 5 pools NOT loaded — TARGET-turn rack "
              "distribution will be skewed (%d/5 = %.0f%% natural fallback). "
              "Load all 5 of MAGPIE_EB_{PASS,EXCH,BINGO,PLAY,RARE}_POOL "
              "for uniform 20/20/20/20/20 5-way mix.\n",
              missing, missing, missing * 20.0);
    }
  } else {
    fprintf(stderr,
            "WARNING: NO pools loaded — TARGET-turn racks will be 100%% "
            "natural (pass-cycle pool sample). Set "
            "MAGPIE_EB_{PASS,EXCH,BINGO,PLAY,RARE}_POOL for the 5-way mix.\n");
  }

  // V-model inference: load any number of .vmt files. Each declares its
  // own training turn (m->turn) and goes into the matching slot.
  for (int i = 0; i < 7; i++) {
    shared_data->vmodels[i] = NULL;
    shared_data->vmodel_picks[i] = NULL;
  }
  shared_data->vmodel_static_leaves = NULL;
  shared_data->vmodel_any_loaded = 0;
  shared_data->vmodel_picks_any_loaded = 0;
  {
    // Build a working list of paths from MAGPIE_VMODEL_PATHS (colon-
    // separated) plus the back-compat singleton MAGPIE_VMODEL_PATH.
    const char *paths_env = getenv("MAGPIE_VMODEL_PATHS");
    const char *single_env = getenv("MAGPIE_VMODEL_PATH");
    char *combined = NULL;
    size_t cap = 0;
    if (paths_env && paths_env[0] != '\0') {
      cap = strlen(paths_env) + 1;
      combined = malloc(cap);
      snprintf(combined, cap, "%s", paths_env);
    }
    if (single_env && single_env[0] != '\0') {
      size_t add = strlen(single_env) + 2;
      char *grown = realloc(combined, cap + add);
      if (!grown) log_fatal("vmodel: alloc fail");
      combined = grown;
      if (cap == 0) {
        snprintf(combined, add, "%s", single_env);
        cap = strlen(single_env) + 1;
      } else {
        cap--;  // drop terminator
        snprintf(combined + cap, add, ":%s", single_env);
        cap += add - 1;
      }
    }
    if (combined && combined[0] != '\0') {
      const char *l6 = getenv("MAGPIE_VMODEL_LEAVES_6");
      const char *l7 = getenv("MAGPIE_VMODEL_LEAVES_7");
      shared_data->vmodel_static_leaves = static_leaves_create(l6, l7);
      if (!shared_data->vmodel_static_leaves) {
        log_fatal("vmodel: static_leaves_create failed "
                  "(leaves_6=%s leaves_7=%s)",
                  l6 ? l6 : "(null)", l7 ? l7 : "(null)");
      }
      char *saveptr = NULL;
      for (char *p = strtok_r(combined, ":", &saveptr); p;
           p = strtok_r(NULL, ":", &saveptr)) {
        if (p[0] == '\0') continue;
        VModel *m = vmodel_create(p);
        if (!m) log_fatal("vmodel: failed to load %s", p);
        if (m->turn < 1 || m->turn > 6) {
          log_fatal("vmodel: model %s has unsupported turn %d (must be 1..6)",
                    p, m->turn);
        }
        if (shared_data->vmodels[m->turn]) {
          log_fatal("vmodel: duplicate model for turn %d (existing + %s)",
                    m->turn, p);
        }
        shared_data->vmodels[m->turn] = m;
        shared_data->vmodel_any_loaded = 1;
        fprintf(stderr, "vmodel: loaded %s for T%d\n", p, m->turn);
      }
    }
    free(combined);
  }

  // V-model picks (precomputed rack→move lookup tables). Same env-var
  // shape as MAGPIE_VMODEL_PATHS: colon-separated .picks files. Each
  // file declares its training turn in its header; loader places it
  // in the matching slot.
  {
    const char *picks_env = getenv("MAGPIE_VMODEL_PICKS_PATHS");
    if (picks_env && picks_env[0] != '\0') {
      char *combined = strdup(picks_env);
      char *saveptr = NULL;
      for (char *p = strtok_r(combined, ":", &saveptr); p;
           p = strtok_r(NULL, ":", &saveptr)) {
        if (p[0] == '\0') continue;
        VModelPicks *pk = vmodel_picks_create(p);
        if (!pk) log_fatal("vmodel_picks: failed to load %s", p);
        int t = pk->model_turn;
        if (t < 1 || t > 6) log_fatal(
            "vmodel_picks: model_turn %d out of range in %s", t, p);
        if (shared_data->vmodel_picks[t]) {
          log_fatal("vmodel_picks: duplicate file for T%d (existing + %s)",
                    t, p);
        }
        shared_data->vmodel_picks[t] = pk;
        shared_data->vmodel_picks_any_loaded = 1;
        fprintf(stderr,
                "vmodel_picks: loaded %s for T%d (%d entries)\n",
                p, t, pk->n_entries);
      }
      free(combined);
    }
  }

  shared_data->empty_board_recorder = NULL;
  const char *eb_out = getenv("MAGPIE_EMPTY_BOARD_OUT");
  if (eb_out && eb_out[0] != '\0') {
    shared_data->empty_board_recorder = empty_board_recorder_create(eb_out);
  }
  shared_data->trajectory_recorder = NULL;
  const char *traj_dir = getenv("MAGPIE_TRAJECTORY_RECORDER");
  if (traj_dir && traj_dir[0] != '\0') {
    shared_data->trajectory_recorder = trajectory_recorder_create(traj_dir);
  }
  shared_data->position_pool = NULL;
  atomic_store(&shared_data->position_pool_next, 0);
  const char *pos_pool_path = getenv("MAGPIE_POSITION_POOL");
  if (pos_pool_path && pos_pool_path[0] != '\0') {
    shared_data->position_pool = position_pool_create(pos_pool_path);
  }
  shared_data->rack_pool = NULL;
  const char *rack_pool_path = getenv("MAGPIE_PP_RACK_POOL");
  if (rack_pool_path && rack_pool_path[0] != '\0') {
    shared_data->rack_pool = position_pool_create(rack_pool_path);
  }
  shared_data->opener_pool = NULL;
  const char *opener_pool_path = getenv("MAGPIE_OPENER_POOL");
  if (opener_pool_path && opener_pool_path[0] != '\0') {
    shared_data->opener_pool = opener_pool_create(opener_pool_path);
  }
  shared_data->leave_deficit = NULL;
  const char *leave_tgt = getenv("MAGPIE_PP_LEAVE_TARGET");
  if (leave_tgt && leave_tgt[0] != '\0') {
    const char *binw = getenv("MAGPIE_PP_LEAVE_BINW");
    const char *maxlen = getenv("MAGPIE_PP_LEAVE_MAXLEN");
    shared_data->leave_deficit = leave_deficit_create(
        atoi(leave_tgt), binw ? atoi(binw) : 20, maxlen ? atoi(maxlen) : 4);
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
  // Force-table activation is implicit: when EB is on (eb_branch_active),
  // per-emit credit fires at the TARGET turn. When EB is off, legacy
  // game-end credit fires via pending_force_target. No separate env var.
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
  play_index_destroy(shared_data->play_index);
  outcome_priors_destroy(shared_data->outcome_priors);
  if (shared_data->pre_t6_capture_file) {
    fclose(shared_data->pre_t6_capture_file);
  }
  if (shared_data->t6_baseline) {
    t6_baseline_state_destroy(shared_data->t6_baseline);
  }
  empty_board_recorder_destroy(shared_data->empty_board_recorder);
  empty_board_strata_destroy(shared_data->empty_board_strata_recorder);
  trajectory_recorder_destroy(shared_data->trajectory_recorder);
  position_pool_destroy(shared_data->position_pool);
  position_pool_destroy(shared_data->rack_pool);
  opener_pool_destroy(shared_data->opener_pool);
  leave_deficit_destroy(shared_data->leave_deficit);
  void vmodel_log_stats(void);  // defined later in this TU
  if (shared_data->vmodel_any_loaded ||
      shared_data->vmodel_picks_any_loaded) {
    vmodel_log_stats();
  }
  for (int i = 0; i < 7; i++) vmodel_destroy(shared_data->vmodels[i]);
  for (int i = 0; i < 7; i++) vmodel_picks_destroy(shared_data->vmodel_picks[i]);
  static_leaves_destroy(shared_data->vmodel_static_leaves);
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
  // play_index rack_id of the rack picked at TARGET injection (or -1
  // when not from play_index). Consumed in eb_enumerate_actions to
  // build the targeted-play fanout slots.
  int64_t eb_target_rack_id;
  // play_index play_id of the specific play picked by
  // play_index_pick_starved_rack alongside the rack (or -1 when not in
  // late-stage mode). Consumed in eb_enumerate_actions to force-include
  // this play's sig in the fanout set so the scheduler's intent
  // (drain this specific starved cell) survives the sum-aggregated
  // score_play scoring that would otherwise drop it under bias toward
  // plays covering many small-deficit cells.
  int64_t eb_target_play_id;
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
  // Force-table targets attached to each enumerated slot (none if natural).
  // Set by eb_append_force_target_slots; consumed in play_eb_dfs which
  // copies the chosen slot's targets into eb_snap_force_targets[turn][].
  // Multi-credit: a single move can match many leave cells (e.g. each
  // distinct tile in the leave + each distinct pair) at the same
  // (stratum, diff). Storing the FULL set of matches lets game-end
  // credit every matched cell, not just one.
  // Cap chosen ample: leave_length up to 7 → 7 distinct tile cells +
  // C(7,2)=21 pair cells + 1 stratum cell + headroom = 32.
  enum { EB_MAX_LEAVE_TARGETS_PER_MOVE = 32 };
  ForceTarget *eb_force_targets_for_slot[EB_MAX_ACTIONS]
                                        [EB_MAX_LEAVE_TARGETS_PER_MOVE];
  uint8_t eb_force_target_count_for_slot[EB_MAX_ACTIONS];
  // Per-snap leave-target list (indexed by snap turn 1..6). Set when
  // DFS descends into a forced slot; eb_emit_leaf reads to credit each
  // matched leave cell. Reset per game in game_runner_start, overwritten
  // per-fork-iter in play_eb_dfs, and forced to 0 on the natural-only
  // (n_actions==0) branch to prevent stale counts from a previous fork
  // iteration leaking into emit at the natural-only turn.
  ForceTarget *eb_snap_force_targets[7][EB_MAX_LEAVE_TARGETS_PER_MOVE];
  uint8_t eb_snap_force_target_count[7];
  // Per-snap BAG_TILE matches. Computed at snap-capture time using the
  // pre-move rack against the active shape bucket's BAG_TILE slots.
  // Up to MAX_PENDING_BAG_TARGETS per snap (one per tile in bag).
  // Credited at game-end in eb_emit_leaf — count-based, no per-cell
  // win/loss tally.
  ForceTarget *eb_snap_bag_targets[7][MAX_PENDING_BAG_TARGETS];
  uint8_t eb_snap_bag_count[7];
  // Per-game staging buffer for the trajectory recorder. NULL when
  // MAGPIE_TRAJECTORY_RECORDER is not set. Reset at game start, committed
  // (or discarded) at game end based on GAME_END_REASON. One per
  // game-runner so a game pair gets two independent buffers.
  TrajectoryGameBuffer *trajectory_buf;
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
  game_runner->eb_target_rack_id = -1;
  game_runner->eb_target_play_id = -1;
  game_runner->eb_natural_slot = -1;
  game_runner->eb_divergence_turn = -1;
  game_runner->eb_last_divergence_turn = -1;
  game_runner->eb_n_divergences = 0;
  game_runner->eb_n_action_buf = 0;
  for (int i = 0; i < EB_MAX_ACTIONS; i++) {
    game_runner->eb_action_buf[i] = NULL;
    game_runner->eb_force_target_count_for_slot[i] = 0;
  }
  for (int i = 0; i < 7; i++) {
    game_runner->eb_snap_force_target_count[i] = 0;
    game_runner->eb_snap_bag_count[i] = 0;
  }
  if (autoplay_worker->shared_data->eb_branch_active) {
    game_runner->eb_n_action_buf = EB_MAX_ACTIONS;
    for (int i = 0; i < EB_MAX_ACTIONS; i++) {
      game_runner->eb_action_buf[i] = move_create();
    }
  }
  game_runner->trajectory_buf =
      autoplay_worker->shared_data->trajectory_recorder
          ? trajectory_game_buffer_create(autoplay_worker->worker_index)
          : NULL;
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
  trajectory_game_buffer_destroy(game_runner->trajectory_buf);
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
  // Reset the trajectory buffer for the new game. Buffer commits or
  // discards at game end based on game_end_reason.
  if (game_runner->trajectory_buf) {
    trajectory_game_buffer_reset(game_runner->trajectory_buf);
  }
  for (int i = 0; i < 7; i++) {
    game_runner->eb_snap_bag_count[i] = 0;
    game_runner->eb_snap_force_target_count[i] = 0;
  }
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
            if (slot->leave_type != LEAVE_TYPE_ALL) {  // typed cell (L5/L6 only)
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
        if (slot->leave_type != LEAVE_TYPE_ALL) {  // typed cell (L5/L6 only)
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

// Extract tile indices (python-canonical 0='?'..26='Z') from a Rack into
// `out_buf`, returning the count written. Assumes magpie's English ML
// encoding matches the python canonical (blank=0, A=1..Z=26).
static int vmodel_extract_rack_indices(const Rack *rack, uint8_t *out_buf,
                                       int max_len) {
  int n = 0;
  const uint16_t dist_size = rack_get_dist_size(rack);
  for (uint16_t ml = 0; ml < dist_size && n < max_len; ml++) {
    const int c = rack_get_letter(rack, ml);
    for (int k = 0; k < c && n < max_len; k++) {
      // Blanks-in-play retain BLANK_MASK bit; canonical encoding wants
      // index 0 for blanks and 1..26 for letters. ML 0 is the unplayed
      // blank; played blanks have the mask bit set. For leave/rack here
      // we want the underlying tile identity, not the played-as letter,
      // so unblank.
      const int unblanked = get_unblanked_machine_letter(ml);
      out_buf[n++] = (uint8_t)unblanked;
    }
  }
  return n;
}

// V-model picks (precomputed table) dispatch counters. Per-turn and
// total. Logged at shutdown alongside the inference-path stats.
static _Atomic uint64_t g_vmodel_picks_hits = 0;
static _Atomic uint64_t g_vmodel_picks_misses = 0;

// Try the precomputed rack→move table for the current turn. Returns
// a Move* if hit (caller uses verbatim), NULL on miss / no table /
// non-empty board. Same turn + bag=93 board-empty gate as
// vmodel_pick_top_move so the two share the same applicability.
static const Move *try_vmodel_picks_pick(AutoplayWorker *autoplay_worker,
                                          GameRunner *game_runner) {
  AutoplaySharedData *shared = game_runner->shared_data;
  if (!shared->vmodel_picks_any_loaded) return NULL;
  const int game_turn = game_runner->turn_number + 1;
  if (game_turn < 1 || game_turn > 6) return NULL;
  const VModelPicks *picks = shared->vmodel_picks[game_turn];
  if (!picks) return NULL;
  Game *game = game_runner->game;
  if (board_get_tiles_played(game_get_board(game)) != 0) return NULL;

  const int player_on_turn_index = game_get_player_on_turn_index(game);
  Rack *player_rack =
      player_get_rack(game_get_player(game, player_on_turn_index));
  uint8_t key[VMODEL_PICKS_RACK_KEY_LEN];
  vmodel_picks_key_for_rack(player_rack, key);
  const Move *m = vmodel_picks_lookup(picks, key);
  if (m == NULL) {
    atomic_fetch_add_explicit(&g_vmodel_picks_misses, 1,
                              memory_order_relaxed);
    return NULL;
  }
  atomic_fetch_add_explicit(&g_vmodel_picks_hits, 1,
                            memory_order_relaxed);
  // Copy into spare so the caller has a stable Move* even after the
  // picks table is destroyed (defensive — table lives for process
  // lifetime, but matches the existing eb_forced_move pattern).
  MoveList *ml = autoplay_worker->eb_move_list
                     ? autoplay_worker->eb_move_list
                     : autoplay_worker->move_lists[player_on_turn_index];
  Move *spare = move_list_get_spare_move(ml);
  move_copy(spare, m);
  return spare;
}

// Diagnostic counters incremented every time vmodel fires. Atomic so
// concurrent workers don't lose updates. Printed at shutdown via
// vmodel_log_stats().
static _Atomic uint64_t g_vmodel_invocations  = 0;  // function entered (turn matched)
static _Atomic uint64_t g_vmodel_total_calls  = 0;  // best_move successfully picked
static _Atomic uint64_t g_vmodel_no_pick      = 0;  // turn matched but no scoreable move
static _Atomic uint64_t g_vmodel_disagreements = 0; // best_move != equity-top
static _Atomic uint64_t g_vmodel_pick_pass    = 0;  // best_move was a pass
static _Atomic uint64_t g_vmodel_pick_exch    = 0;  // best_move was an exchange
static _Atomic uint64_t g_vmodel_pick_play    = 0;  // best_move was a play
static _Atomic uint64_t g_vmodel_top_pass     = 0;  // top-equity was a pass
static _Atomic uint64_t g_vmodel_top_exch     = 0;  // top-equity was an exchange
static _Atomic uint64_t g_vmodel_top_play     = 0;  // top-equity was a play
// Per-kind histograms over picked-move leave point-sum (0..50, clamped).
// For pass: leave == rack, so this is rack point sum.
// For exch/play: this is the kept-tile point sum.
#define VMDBG_PTS_BUCKETS 51
static _Atomic uint64_t g_vmodel_pts_hist_pass[VMDBG_PTS_BUCKETS];
static _Atomic uint64_t g_vmodel_pts_hist_exch[VMDBG_PTS_BUCKETS];
static _Atomic uint64_t g_vmodel_pts_hist_play[VMDBG_PTS_BUCKETS];
// Distribution of n_tiles_exchanged (1..7) when the pick was an exchange.
static _Atomic uint64_t g_vmodel_exch_n_hist[8];
// Histogram over rack-point-sum across ALL vmodel invocations (so we can
// sanity-check that the rack distribution is what we think it is).
static _Atomic uint64_t g_vmodel_rack_pts_hist[VMDBG_PTS_BUCKETS];

// Rank of the chosen move within its kind, equity-sorted (0 = top equity).
// Tracks whether the V model usually picks near the top of HastyBot's
// equity ranking (=> can be safely truncated to top-N candidates) or
// scatters across the list. One slot per rank up to MAX (last slot
// clamps anything deeper). MAX sized to fit the worst-case opener move
// count comfortably (plays can hit ~2000 candidates with ?-rich racks).
#define VMDBG_RANK_BUCKETS 2048
static _Atomic uint64_t g_vmodel_play_rank_hist[VMDBG_RANK_BUCKETS];
static _Atomic uint64_t g_vmodel_exch_rank_hist[VMDBG_RANK_BUCKETS];
// Total candidates in move list per kind (sum of counts across picks).
static _Atomic uint64_t g_vmodel_n_play_total = 0;
static _Atomic uint64_t g_vmodel_n_exch_total = 0;
static _Atomic uint64_t g_vmodel_n_call_total = 0;
// Cache hit/miss totals.
static _Atomic uint64_t g_vmodel_cache_hits = 0;
static _Atomic uint64_t g_vmodel_cache_misses = 0;

// Per-call cache: dedup vmodel_predict by (kind, canonical-leave, diff).
// Within a single vmodel_pick_top_move call, the rack is fixed, so the
// triple (kind, leave, diff) fully determines the prediction. Many move
// lists have duplicate signatures — e.g. bingos with the same word at
// different board positions all have leave="" and same score → same
// prediction. Caching collapses 100s of inferences to 1 per unique
// signature.
#define VMCACHE_SLOTS 1024
typedef struct VmCacheEntry {
  uint8_t used;
  uint8_t kind;
  uint8_t leave_len;
  uint8_t leave[STATIC_LEAVES_MAX_LEN];
  int32_t diff;
  float   p;
} VmCacheEntry;

static inline uint32_t vmcache_hash(uint8_t kind, const uint8_t *leave,
                                     int leave_len, int diff) {
  uint32_t h = 2166136261u;  // FNV-1a
  h = (h ^ kind) * 16777619u;
  h = (h ^ (uint32_t)diff) * 16777619u;
  h = (h ^ (uint32_t)leave_len) * 16777619u;
  for (int i = 0; i < leave_len; i++) {
    h = (h ^ leave[i]) * 16777619u;
  }
  return h;
}

// Look up cache for (kind, canonical leave, diff); return -1.0f if miss.
static float vmcache_get(const VmCacheEntry *cache, uint8_t kind,
                          const uint8_t *cleave, int leave_len, int diff) {
  uint32_t h = vmcache_hash(kind, cleave, leave_len, diff);
  for (int probe = 0; probe < VMCACHE_SLOTS; probe++) {
    uint32_t idx = (h + probe) & (VMCACHE_SLOTS - 1);
    const VmCacheEntry *e = &cache[idx];
    if (!e->used) return -2.0f;  // empty slot — miss
    if (e->kind == kind && e->leave_len == leave_len && e->diff == diff &&
        memcmp(e->leave, cleave, leave_len) == 0) {
      return e->p;
    }
  }
  return -2.0f;
}

static void vmcache_put(VmCacheEntry *cache, uint8_t kind,
                        const uint8_t *cleave, int leave_len, int diff,
                        float p) {
  uint32_t h = vmcache_hash(kind, cleave, leave_len, diff);
  for (int probe = 0; probe < VMCACHE_SLOTS; probe++) {
    uint32_t idx = (h + probe) & (VMCACHE_SLOTS - 1);
    VmCacheEntry *e = &cache[idx];
    if (!e->used) {
      e->used = 1;
      e->kind = kind;
      e->leave_len = (uint8_t)leave_len;
      memcpy(e->leave, cleave, leave_len);
      e->diff = diff;
      e->p = p;
      return;
    }
  }
  // Full — overwrite hash slot (rare with 1024 slots and ~500 candidates).
  uint32_t idx = h & (VMCACHE_SLOTS - 1);
  VmCacheEntry *e = &cache[idx];
  e->used = 1;
  e->kind = kind;
  e->leave_len = (uint8_t)leave_len;
  memcpy(e->leave, cleave, leave_len);
  e->diff = diff;
  e->p = p;
}
// English tile point values, indexed by python-canonical 0..26.
static const uint8_t k_vmodel_tile_pts[27] = {
    0,1,3,3,2,1,4,2,4,1,8,5,1,3,1,1,3,10,1,1,1,1,4,4,8,4,10};
static int vmodel_indices_to_pts(const uint8_t *idx, int n) {
  int s = 0;
  for (int i = 0; i < n; i++) {
    if (idx[i] < 27) s += k_vmodel_tile_pts[idx[i]];
  }
  return s;
}

void vmodel_log_stats(void) {
  // Always emit picks stats up-front — when picks cover every rack the
  // inference path may have zero invocations, in which case we'd skip
  // the rest of this function but still want to report the picks usage.
  const uint64_t ph_top = atomic_load_explicit(&g_vmodel_picks_hits,
                                                memory_order_relaxed);
  const uint64_t pm_top = atomic_load_explicit(&g_vmodel_picks_misses,
                                                memory_order_relaxed);
  if (ph_top + pm_top) {
    fprintf(stderr,
            "vmodel_picks: %llu hits, %llu misses (%.2f%% hit rate)\n",
            (unsigned long long)ph_top, (unsigned long long)pm_top,
            100.0 * (double)ph_top / (double)(ph_top + pm_top));
  }
  const uint64_t invos = atomic_load_explicit(&g_vmodel_invocations,
                                              memory_order_relaxed);
  if (invos == 0) return;
  const uint64_t total = atomic_load_explicit(&g_vmodel_total_calls,
                                              memory_order_relaxed);
  const uint64_t nop = atomic_load_explicit(&g_vmodel_no_pick,
                                            memory_order_relaxed);
  const uint64_t disagree = atomic_load_explicit(&g_vmodel_disagreements,
                                                  memory_order_relaxed);
  const uint64_t pp = atomic_load_explicit(&g_vmodel_pick_pass, memory_order_relaxed);
  const uint64_t pe = atomic_load_explicit(&g_vmodel_pick_exch, memory_order_relaxed);
  const uint64_t pl = atomic_load_explicit(&g_vmodel_pick_play, memory_order_relaxed);
  const uint64_t tp = atomic_load_explicit(&g_vmodel_top_pass, memory_order_relaxed);
  const uint64_t te = atomic_load_explicit(&g_vmodel_top_exch, memory_order_relaxed);
  const uint64_t tl = atomic_load_explicit(&g_vmodel_top_play, memory_order_relaxed);
  fprintf(stderr,
          "vmodel: %llu invocations | %llu picks | %llu no-pick | "
          "%llu disagreed with top equity (%.2f%% of picks)\n",
          (unsigned long long)invos, (unsigned long long)total,
          (unsigned long long)nop, (unsigned long long)disagree,
          total ? 100.0 * (double)disagree / (double)total : 0.0);
  fprintf(stderr,
          "vmodel: pick distribution: pass=%llu (%.1f%%) exch=%llu (%.1f%%) "
          "play=%llu (%.1f%%)\n",
          (unsigned long long)pp, total ? 100.0*pp/total : 0.0,
          (unsigned long long)pe, total ? 100.0*pe/total : 0.0,
          (unsigned long long)pl, total ? 100.0*pl/total : 0.0);
  fprintf(stderr,
          "vmodel: top-equity dist:  pass=%llu (%.1f%%) exch=%llu (%.1f%%) "
          "play=%llu (%.1f%%)\n",
          (unsigned long long)tp, total ? 100.0*tp/total : 0.0,
          (unsigned long long)te, total ? 100.0*te/total : 0.0,
          (unsigned long long)tl, total ? 100.0*tl/total : 0.0);

  // Histograms — print non-empty buckets.
  fprintf(stderr, "vmodel: ALL-invocation rack-pts hist:\n");
  for (int b = 0; b < VMDBG_PTS_BUCKETS; b++) {
    uint64_t c = atomic_load_explicit(&g_vmodel_rack_pts_hist[b], memory_order_relaxed);
    if (c) fprintf(stderr, "  %s%d: %llu\n", b == VMDBG_PTS_BUCKETS-1 ? ">=" : "",
                   b, (unsigned long long)c);
  }
  fprintf(stderr, "vmodel: PASS leave-pts hist (rack pts when passing):\n");
  for (int b = 0; b < VMDBG_PTS_BUCKETS; b++) {
    uint64_t c = atomic_load_explicit(&g_vmodel_pts_hist_pass[b], memory_order_relaxed);
    if (c) fprintf(stderr, "  %s%d: %llu\n", b == VMDBG_PTS_BUCKETS-1 ? ">=" : "",
                   b, (unsigned long long)c);
  }
  fprintf(stderr, "vmodel: EXCH leave-pts hist (kept-tile pts when exchanging):\n");
  for (int b = 0; b < VMDBG_PTS_BUCKETS; b++) {
    uint64_t c = atomic_load_explicit(&g_vmodel_pts_hist_exch[b], memory_order_relaxed);
    if (c) fprintf(stderr, "  %s%d: %llu\n", b == VMDBG_PTS_BUCKETS-1 ? ">=" : "",
                   b, (unsigned long long)c);
  }
  fprintf(stderr, "vmodel: EXCH n_tiles_exch hist (1..7 throwbacks):\n");
  for (int b = 1; b < 8; b++) {
    uint64_t c = atomic_load_explicit(&g_vmodel_exch_n_hist[b], memory_order_relaxed);
    if (c) fprintf(stderr, "  exch %d tiles: %llu\n", b, (unsigned long long)c);
  }
  fprintf(stderr, "vmodel: PLAY leave-pts hist (kept-tile pts after a play):\n");
  for (int b = 0; b < VMDBG_PTS_BUCKETS; b++) {
    uint64_t c = atomic_load_explicit(&g_vmodel_pts_hist_play[b], memory_order_relaxed);
    if (c) fprintf(stderr, "  %s%d: %llu\n", b == VMDBG_PTS_BUCKETS-1 ? ">=" : "",
                   b, (unsigned long long)c);
  }
  // Equity-rank histogram per kind: how deep into the equity-sorted
  // candidate list does the V model pick? Tight to the top → top-N
  // truncation is safe; long tail → vmodel needs full list.
  fprintf(stderr, "vmodel: PLAY equity-rank hist (picked play's index in"
                  " equity-sorted plays):\n");
  for (int b = 0; b < VMDBG_RANK_BUCKETS; b++) {
    uint64_t c = atomic_load_explicit(&g_vmodel_play_rank_hist[b],
                                       memory_order_relaxed);
    if (c) fprintf(stderr, "  %srank %d: %llu\n",
                   b == VMDBG_RANK_BUCKETS-1 ? ">=" : "",
                   b, (unsigned long long)c);
  }
  fprintf(stderr, "vmodel: EXCH equity-rank hist (picked exch's index in"
                  " equity-sorted exch):\n");
  for (int b = 0; b < VMDBG_RANK_BUCKETS; b++) {
    uint64_t c = atomic_load_explicit(&g_vmodel_exch_rank_hist[b],
                                       memory_order_relaxed);
    if (c) fprintf(stderr, "  %srank %d: %llu\n",
                   b == VMDBG_RANK_BUCKETS-1 ? ">=" : "",
                   b, (unsigned long long)c);
  }
  // Mean candidates per call (helps decide top-N truncation).
  const uint64_t ncall = atomic_load_explicit(&g_vmodel_n_call_total,
                                                memory_order_relaxed);
  const uint64_t npl = atomic_load_explicit(&g_vmodel_n_play_total,
                                              memory_order_relaxed);
  const uint64_t nex = atomic_load_explicit(&g_vmodel_n_exch_total,
                                              memory_order_relaxed);
  if (ncall) {
    fprintf(stderr,
            "vmodel: mean candidates per call: play=%.1f exch=%.1f "
            "(over %llu calls)\n",
            (double)npl / (double)ncall, (double)nex / (double)ncall,
            (unsigned long long)ncall);
  }
  // Cache stats.
  const uint64_t ch = atomic_load_explicit(&g_vmodel_cache_hits,
                                            memory_order_relaxed);
  const uint64_t cm = atomic_load_explicit(&g_vmodel_cache_misses,
                                            memory_order_relaxed);
  if (ch + cm) {
    fprintf(stderr,
            "vmodel: cache: hits=%llu misses=%llu (%.1f%% hit rate)\n",
            (unsigned long long)ch, (unsigned long long)cm,
            100.0 * (double)ch / (double)(ch + cm));
  }
}

// V-model pick: score every move in the move-list with the trained model,
// return the highest-win% move. Strict greater-than scan preserves the
// movegen-sort tiebreak (equity desc). Returns NULL if vmodel is not
// active for this turn or if no move is scoreable.
static const Move *vmodel_pick_top_move(AutoplayWorker *autoplay_worker,
                                        GameRunner *game_runner) {
  AutoplaySharedData *shared = game_runner->shared_data;
  if (!shared->vmodel_any_loaded) return NULL;
  // turn_number is 0-indexed; convert to 1-indexed game turn.
  const int game_turn = game_runner->turn_number + 1;
  if (game_turn < 1 || game_turn > 6) return NULL;
  const VModel *model = shared->vmodels[game_turn];
  if (!model) return NULL;
  Game *game = game_runner->game;
  // V model assumes bag=93 / board empty / both scores 0 (its training
  // distribution). Skip if the game state doesn't match those
  // assumptions — falls through to HastyBot. In EB cycle target=T runs,
  // turns 1..T-1 are forced pass/exchange so board stays empty; once
  // T's recorded move plays tiles, post-T turns no longer satisfy this.
  if (board_get_tiles_played(game_get_board(game)) != 0) return NULL;
  atomic_fetch_add_explicit(&g_vmodel_invocations, 1, memory_order_relaxed);
  {
    Rack *_pr = player_get_rack(game_get_player(game,
                  game_get_player_on_turn_index(game)));
    uint8_t _ri[16];
    int _rl = vmodel_extract_rack_indices(_pr, _ri, 16);
    int _rp = vmodel_indices_to_pts(_ri, _rl);
    if (_rp < 0) _rp = 0;
    if (_rp >= VMDBG_PTS_BUCKETS) _rp = VMDBG_PTS_BUCKETS - 1;
    atomic_fetch_add_explicit(&g_vmodel_rack_pts_hist[_rp], 1,
                              memory_order_relaxed);
  }

  const int player_on_turn_index = game_get_player_on_turn_index(game);
  // Per-player move_lists are sized to num_plays (often 1), so
  // MOVE_RECORD_ALL into them would prune to top-1. eb_move_list is
  // pre-allocated large for exactly this kind of full-enumeration.
  MoveList *ml = autoplay_worker->eb_move_list
                     ? autoplay_worker->eb_move_list
                     : autoplay_worker->move_lists[player_on_turn_index];
  const MoveGenArgs gen_args = {
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
  generate_moves(&gen_args);
  move_list_sort_moves(ml);
  const int n_moves = move_list_get_count(ml);
  if (n_moves == 0) return NULL;

  Rack *player_rack =
      player_get_rack(game_get_player(game, player_on_turn_index));
  uint8_t rack_idx[16];
  int rack_len = vmodel_extract_rack_indices(player_rack, rack_idx, 16);

  // Pre-action diff (this player's score advantage before the move).
  const Player *me  = game_get_player(game, player_on_turn_index);
  const Player *opp = game_get_player(game, 1 - player_on_turn_index);
  const int pre_diff =
      equity_to_int(player_get_score(me)) - equity_to_int(player_get_score(opp));

  Move *best_move = NULL;
  float best_win  = -1.0f;
  int   best_idx  = -1;  // equity-sorted index of best_move (for rank stats)
  int   n_play = 0, n_exch = 0;
  int   play_idx_in_kind = 0, exch_idx_in_kind = 0;
  int   best_kind_idx = -1;  // index within own kind (play or exch)

  // Per-call dedup cache: (kind, canonical-leave, diff) → prediction.
  // Movegen produces many duplicate (leave, diff) signatures (especially
  // for plays — a bingo word has many positional variants all with
  // leave="" and same score). Hash-table dedup collapses these to one
  // inference per unique signature.
  VmCacheEntry cache[VMCACHE_SLOTS];
  memset(cache, 0, sizeof(cache));
  uint64_t local_hits = 0, local_misses = 0;

  Rack leave_rack;
  rack_set_dist_size(&leave_rack, rack_get_dist_size(player_rack));

  for (int i = 0; i < n_moves; i++) {
    Move *m = move_list_get_move(ml, i);
    int kind;
    switch (move_get_type(m)) {
      case GAME_EVENT_PASS:                kind = 0; break;
      case GAME_EVENT_EXCHANGE:            kind = 1; break;
      case GAME_EVENT_TILE_PLACEMENT_MOVE: kind = 2; break;
      default: continue;
    }
    int kind_idx = -1;
    if (kind == 1) { kind_idx = exch_idx_in_kind++; n_exch++; }
    else if (kind == 2) { kind_idx = play_idx_in_kind++; n_play++; }
    rack_reset(&leave_rack);
    get_leave_for_move(m, game, &leave_rack);
    uint8_t leave_idx[16];
    int leave_len = vmodel_extract_rack_indices(&leave_rack, leave_idx, 16);
    int diff = pre_diff;
    if (kind == 2) {
      diff += equity_to_int(move_get_score(m));
    }
    // Cache lookup keyed on canonical leave + kind + diff.
    uint8_t cleave[STATIC_LEAVES_MAX_LEN] = {0};
    int cleave_len = leave_len < STATIC_LEAVES_MAX_LEN ? leave_len
                                                       : STATIC_LEAVES_MAX_LEN;
    static_leaves_canonicalize(leave_idx, cleave_len, cleave);
    float p = vmcache_get(cache, (uint8_t)kind, cleave, cleave_len, diff);
    if (p < -1.5f) {
      // miss
      p = vmodel_predict(
          model,
          rack_idx, rack_len,
          leave_idx, leave_len,
          kind, diff, game_turn,
          shared->vmodel_static_leaves);
      vmcache_put(cache, (uint8_t)kind, cleave, cleave_len, diff, p);
      local_misses++;
    } else {
      local_hits++;
    }
    if (p < 0.0f) continue;  // unscored (stratum/bucket missing)
    // Round to 4 decimal places (0.01%) before tie-breaking: model SE on
    // any single prediction is ~0.005, so numerical jitter finer than
    // 1e-4 isn't real signal. Ties at this granularity fall back to
    // movegen's equity-desc sort order (= HastyBot static equity).
    const float p_round = roundf(p * 10000.0f) / 10000.0f;
    if (p_round > best_win) {  // strict > preserves equity-sorted tie order
      best_win  = p_round;
      best_move = m;
      best_idx  = i;
      best_kind_idx = kind_idx;
    }
  }
  // Movegen does NOT include a pass move alongside plays/exchanges. The V
  // model needs to consider passing too, so score pass explicitly and
  // override `best_move` if its win% beats the best play/exchange.
  {
    uint8_t pass_cleave[STATIC_LEAVES_MAX_LEN] = {0};
    int pass_clen = rack_len < STATIC_LEAVES_MAX_LEN ? rack_len
                                                      : STATIC_LEAVES_MAX_LEN;
    static_leaves_canonicalize(rack_idx, pass_clen, pass_cleave);
    float p_pass = vmcache_get(cache, 0, pass_cleave, pass_clen, pre_diff);
    if (p_pass < -1.5f) {
      p_pass = vmodel_predict(
          model, rack_idx, rack_len,
          rack_idx, rack_len,  // pass: leave == rack
          0, pre_diff, game_turn,
          shared->vmodel_static_leaves);
      vmcache_put(cache, 0, pass_cleave, pass_clen, pre_diff, p_pass);
      local_misses++;
    } else {
      local_hits++;
    }
    if (p_pass >= 0.0f) {
      const float pp_round = roundf(p_pass * 10000.0f) / 10000.0f;
      // Strict > so that if pass ties a play, the equity-sorted play wins
      // (matches the per-move tiebreak convention).
      if (pp_round > best_win) {
        Move *spare = move_list_get_spare_move(ml);
        move_set_as_pass(spare);
        best_win = pp_round;
        best_move = spare;
        best_kind_idx = 0;  // pass is rank 0 in its own (degenerate) kind
      }
    }
  }
  // Roll up cache stats once per call (cheaper than per-move atomics).
  atomic_fetch_add_explicit(&g_vmodel_cache_hits, local_hits,
                            memory_order_relaxed);
  atomic_fetch_add_explicit(&g_vmodel_cache_misses, local_misses,
                            memory_order_relaxed);
  atomic_fetch_add_explicit(&g_vmodel_n_play_total, (uint64_t)n_play,
                            memory_order_relaxed);
  atomic_fetch_add_explicit(&g_vmodel_n_exch_total, (uint64_t)n_exch,
                            memory_order_relaxed);
  atomic_fetch_add_explicit(&g_vmodel_n_call_total, 1, memory_order_relaxed);
  (void)best_idx;  // available for future use
  if (best_move) {
    atomic_fetch_add_explicit(&g_vmodel_total_calls, 1, memory_order_relaxed);
    // Disagreement = picked move is not the top-equity (movegen-sorted index 0).
    Move *top_eq = move_list_get_move(ml, 0);
    if (best_move != top_eq) {
      atomic_fetch_add_explicit(&g_vmodel_disagreements, 1,
                                memory_order_relaxed);
    }
    // Re-derive the best move's leave for histogram bucketing.
    rack_reset(&leave_rack);
    get_leave_for_move(best_move, game, &leave_rack);
    uint8_t bm_leave_idx[16];
    int bm_leave_len = vmodel_extract_rack_indices(&leave_rack, bm_leave_idx, 16);
    int leave_pts = vmodel_indices_to_pts(bm_leave_idx, bm_leave_len);
    if (leave_pts < 0) leave_pts = 0;
    if (leave_pts >= VMDBG_PTS_BUCKETS) leave_pts = VMDBG_PTS_BUCKETS - 1;
    // Record the picked move's rank within its kind (equity-sorted).
    if (best_kind_idx >= 0) {
      int rb = best_kind_idx;
      if (rb >= VMDBG_RANK_BUCKETS) rb = VMDBG_RANK_BUCKETS - 1;
      if (move_get_type(best_move) == GAME_EVENT_EXCHANGE) {
        atomic_fetch_add_explicit(&g_vmodel_exch_rank_hist[rb], 1,
                                  memory_order_relaxed);
      } else if (move_get_type(best_move) == GAME_EVENT_TILE_PLACEMENT_MOVE) {
        atomic_fetch_add_explicit(&g_vmodel_play_rank_hist[rb], 1,
                                  memory_order_relaxed);
      }
    }
    switch (move_get_type(best_move)) {
      case GAME_EVENT_PASS:
        atomic_fetch_add_explicit(&g_vmodel_pick_pass, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_vmodel_pts_hist_pass[leave_pts], 1,
                                  memory_order_relaxed);
        break;
      case GAME_EVENT_EXCHANGE: {
        atomic_fetch_add_explicit(&g_vmodel_pick_exch, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_vmodel_pts_hist_exch[leave_pts], 1,
                                  memory_order_relaxed);
        int n_exch = rack_len - bm_leave_len;
        if (n_exch >= 1 && n_exch <= 7) {
          atomic_fetch_add_explicit(&g_vmodel_exch_n_hist[n_exch], 1,
                                    memory_order_relaxed);
        }
        break;
      }
      case GAME_EVENT_TILE_PLACEMENT_MOVE:
        atomic_fetch_add_explicit(&g_vmodel_pick_play, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_vmodel_pts_hist_play[leave_pts], 1,
                                  memory_order_relaxed);
        break;
      default: break;
    }
    switch (move_get_type(top_eq)) {
      case GAME_EVENT_PASS:
        atomic_fetch_add_explicit(&g_vmodel_top_pass, 1, memory_order_relaxed); break;
      case GAME_EVENT_EXCHANGE:
        atomic_fetch_add_explicit(&g_vmodel_top_exch, 1, memory_order_relaxed); break;
      case GAME_EVENT_TILE_PLACEMENT_MOVE:
        atomic_fetch_add_explicit(&g_vmodel_top_play, 1, memory_order_relaxed); break;
      default: break;
    }
  } else {
    atomic_fetch_add_explicit(&g_vmodel_no_pick, 1, memory_order_relaxed);
  }
  return best_move;
}

// Returns the played move
// Stage one trajectory row for `move` at the current (pre-move) position into
// the game runner's per-game buffer, when the trajectory recorder is active.
// Factored out of game_runner_play_move so the position-pool mode can record
// the same schema without going through the autoplay move-selection path.
static void eb_stage_trajectory_row(GameRunner *game_runner, Game *game,
                                    const Move *move,
                                    int player_on_turn_index) {
  TrajectoryRecorder *traj_r = game_runner->shared_data->trajectory_recorder;
  TrajectoryGameBuffer *traj_buf = game_runner->trajectory_buf;
  if (!(traj_r && traj_buf)) return;
  const Rack *player_rack =
      player_get_rack(game_get_player(game, player_on_turn_index));
  const LetterDistribution *traj_ld = game_get_ld(game);
  // Unseen-total convention: physical bag + opp's actual rack size.
  const int traj_opp_idx = 1 - player_on_turn_index;
  const int traj_opp_rack_size = rack_get_total_letters(
      player_get_rack(game_get_player(game, traj_opp_idx)));
  const int traj_bag =
      bag_get_letters(game_get_bag(game)) + traj_opp_rack_size;
  char p1_rack_str[RACK_SIZE + 2] = {0};
  char p2_rack_str[RACK_SIZE + 2] = {0};
  for (int pp = 0; pp < 2; pp++) {
    const Rack *rk = player_get_rack(game_get_player(game, pp));
    char *out = (pp == 0) ? p1_rack_str : p2_rack_str;
    int n = 0;
    const uint16_t ds = rack_get_dist_size(rk);
    for (uint16_t i = 1; i < ds && n < RACK_SIZE; i++) {
      const int c = rack_get_letter(rk, i);
      for (int k = 0; k < c && n < RACK_SIZE; k++) {
        out[n++] = traj_ld->ld_ml_to_hl[i][0];
      }
    }
    const int nb = rack_get_letter(rk, 0);
    for (int k = 0; k < nb && n < RACK_SIZE; k++) out[n++] = '?';
    out[n] = '\0';
  }
  const game_event_t mt = move_get_type(move);
  int act_kind;
  int act_size;
  if (mt == GAME_EVENT_PASS) {
    act_kind = 0;
    act_size = 0;
  } else if (mt == GAME_EVENT_EXCHANGE) {
    act_kind = 1;
    act_size = move_get_tiles_played(move);
  } else {
    act_kind = 2;
    act_size = move_get_tiles_played(move);
  }
  char act_repr[64];
  StringBuilder *act_sb = string_builder_create();
  string_builder_add_move(act_sb, game_get_board(game), move, traj_ld, false);
  size_t act_len = 0;
  char *act_dump = string_builder_dump(act_sb, &act_len);
  snprintf(act_repr, sizeof(act_repr), "%s", act_dump ? act_dump : "");
  free(act_dump);
  string_builder_destroy(act_sb);
  Rack traj_leave;
  rack_set_dist_size(&traj_leave, rack_get_dist_size(player_rack));
  rack_reset(&traj_leave);
  if (mt == GAME_EVENT_PASS) {
    rack_copy(&traj_leave, player_rack);
  } else {
    get_leave_for_move(move, game, &traj_leave);
  }
  char leave_str[RACK_SIZE + 2] = {0};
  int ln = 0;
  const uint16_t lds = rack_get_dist_size(&traj_leave);
  for (uint16_t i = 1; i < lds && ln < RACK_SIZE; i++) {
    const int c = rack_get_letter(&traj_leave, i);
    for (int k = 0; k < c && ln < RACK_SIZE; k++) {
      leave_str[ln++] = traj_ld->ld_ml_to_hl[i][0];
    }
  }
  const int lvb = rack_get_letter(&traj_leave, 0);
  for (int k = 0; k < lvb && ln < RACK_SIZE; k++) leave_str[ln++] = '?';
  leave_str[ln] = '\0';
  char *cgp_str = game_get_cgp(game, false);
  const int p1_score =
      equity_to_int(player_get_score(game_get_player(game, 0)));
  const int p2_score =
      equity_to_int(player_get_score(game_get_player(game, 1)));
  const int move_score_int =
      (mt == GAME_EVENT_TILE_PLACEMENT_MOVE)
          ? equity_to_int(move_get_score(move))
          : 0;
  const int on_turn_score = (player_on_turn_index == 0) ? p1_score : p2_score;
  const int opp_score = (player_on_turn_index == 0) ? p2_score : p1_score;
  const int score_diff_pre = on_turn_score - opp_score;
  const int score_diff_post = score_diff_pre + move_score_int;
  trajectory_game_buffer_add(
      traj_buf, game_runner->game_number, game_runner->turn_number + 1,
      traj_bag, player_on_turn_index, p1_rack_str, p2_rack_str, p1_score,
      p2_score, cgp_str ? cgp_str : "", act_kind, act_repr, act_size,
      move_score_int, score_diff_pre, score_diff_post, leave_str);
  free(cgp_str);
}

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
  // V-model decision hook (raised ahead of pass-cycle preemption): when
  // the V model is the decision-maker for this turn, it sees the full
  // movegen output INCLUDING pass — so we don't want the pass-cycle
  // classifier to preempt the call. The V model decides pass-vs-play
  // itself based on which has the highest predicted win%.
  //
  // Precomputed picks table tried FIRST (~200ns binary-search lookup),
  // inference fallback (movegen + per-move scoring) tried SECOND
  // (~tens of µs per call). Both gated identically.
  if (!move) {
    move = try_vmodel_picks_pick(autoplay_worker, game_runner);
  }
  if (!move) {
    move = vmodel_pick_top_move(autoplay_worker, game_runner);
  }
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

    // BAG_TILE force-cell matching. Runs only when this snap IS the target
    // turn (only one snap per game gets emitted/credited). The pre-move
    // rack composition is the match key; all matched cells are credited
    // count-based at game-end via eb_emit_leaf.
    game_runner->eb_snap_bag_count[slot] = 0;
    ForceTable *ft_bag = game_runner->shared_data->force_table;
    const int target_turn_check = game_runner->shared_data->eb_target_turn;
    if (ft_bag != NULL && (slot + 1) == target_turn_check) {
      const int bag_count_snap =
          bag_get_letters(game_get_bag(game)) + (RACK_SIZE);
      // Leave length and exchange flag derived from snap fields.
      const int leave_len = (int)strlen(game_runner->eb_snaps[slot].leave);
      const int is_exch_snap =
          (game_runner->eb_snaps[slot].action_kind == 1) ? 1 : 0;
      // Diff at this snap: matches the aggregator's "diff = pre_action_diff
      // + move_score" convention (post-action, from on-turn player's
      // perspective). For passes/exchanges score==0, so eff_diff == pre.
      const int s0_snap =
          equity_to_int(player_get_score(game_get_player(game, 0)));
      const int s1_snap =
          equity_to_int(player_get_score(game_get_player(game, 1)));
      const int my_snap = (player_on_turn_index == 0) ? s0_snap : s1_snap;
      const int opp_snap = (player_on_turn_index == 0) ? s1_snap : s0_snap;
      const int pre_diff = my_snap - opp_snap;
      // Leave-type classification (only matters at L >= 3).
      Rack tmp_leave_for_type;
      rack_set_dist_size(&tmp_leave_for_type, ds_eb);
      rack_reset(&tmp_leave_for_type);
      if (move_get_type(move) == GAME_EVENT_PASS) {
        rack_copy(&tmp_leave_for_type, player_rack);
      } else {
        get_leave_for_move(move, game, &tmp_leave_for_type);
      }
      const LeaveType ltype_snap =
          (leave_len >= 3 && is_exch_snap == 0)
              ? force_classify_leave(&tmp_leave_for_type, ld_eb)
              : LEAVE_TYPE_ALL;
      const int snap_score = game_runner->eb_snaps[slot].move_score;
      const int eff_diff_snap = pre_diff + snap_score;
      // Iterate the shape bucket's slots; collect BAG_TILE matches.
      int bucket_count = 0;
      ForceTargetSlot *slots = force_table_lookup_slots_by_shape(
          ft_bag, bag_count_snap, leave_len, is_exch_snap, &bucket_count);
      if (slots != NULL && bucket_count > 0) {
        uint8_t bag_n = 0;
        for (int t = 0; t < bucket_count &&
             bag_n < MAX_PENDING_BAG_TARGETS; t++) {
          ForceTargetSlot *s = &slots[t];
          if (s->deficit <= 0 ||
              (int)s->kind != FORCE_TARGET_BAG_TILE) {
            continue;
          }
          if (force_target_matches_bag(s->cold, player_rack, leave_len,
                                       ltype_snap, is_exch_snap, eff_diff_snap,
                                       ld_eb)) {
            game_runner->eb_snap_bag_targets[slot][bag_n++] = s->cold;
          }
        }
        game_runner->eb_snap_bag_count[slot] = bag_n;
      }
    }

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

  // Trajectory recorder: stage the pre-move position in this worker's
  // per-game buffer (commits on normal game end; 6-pass terminus discarded).
  // Active only when MAGPIE_TRAJECTORY_RECORDER=<dir> is set at startup.
  eb_stage_trajectory_row(game_runner, game, move, player_on_turn_index);

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
  } else if (shared_data->force_table) {
    string_builder_add_formatted_string(
        status_sb, " Force-table remaining deficit: %ld.",
        (long)force_table_total_remaining(shared_data->force_table));
    // Per-tick diagnostic counters for the multi-credit regression.
    // Snapshot + reset so each line shows in-interval activity (matches
    // the wall-clock-per-interval style of the games-per-tick line).
    ForceCounters fc;
    force_table_get_counters(&fc);
    force_table_reset_counters();

    // Phase 4 auto-switch: if broad efficiency drops below threshold,
    // flip eb_late_stage on. One-shot — once on, never off. Requires
    // play_index loaded (late-stage picker needs it).
    if (shared_data->play_index &&
        shared_data->eb_auto_late_threshold > 0.0 &&
        shared_data->print_interval > 0 &&
        !atomic_load_explicit(&shared_data->eb_late_stage,
                              memory_order_relaxed)) {
      const double window_games = (double)shared_data->print_interval;
      const double broad_eff = (double)fc.decrements_landed / window_games;
      if (broad_eff < shared_data->eb_auto_late_threshold) {
        atomic_store_explicit(&shared_data->eb_late_stage, true,
                              memory_order_release);
        fprintf(stderr,
                "eb_auto_late_switch: TRIGGERED at game %llu — "
                "broad_eff=%.4f credits/game < %.4f threshold; "
                "switching to cell-driven targeted picker\n",
                (unsigned long long)iter_completed_output->iter_count_completed,
                broad_eff,
                shared_data->eb_auto_late_threshold);
      }
    }
    const uint64_t leaves =
        atomic_exchange_explicit(&g_eb_leaves_emitted, 0,
                                 memory_order_relaxed);
    const uint64_t annot =
        atomic_exchange_explicit(&g_eb_annotate_cells_added, 0,
                                 memory_order_relaxed);
    const uint64_t prop =
        atomic_exchange_explicit(&g_eb_dfs_cells_propagated, 0,
                                 memory_order_relaxed);
    const uint64_t emitc =
        atomic_exchange_explicit(&g_eb_emit_cells_credited, 0,
                                 memory_order_relaxed);
    uint64_t s_cred[6], s_bump[6];
    force_table_get_stratum_by_len(s_cred, s_bump);
    force_table_reset_stratum_by_len();
    uint64_t s_annot[6];
    uint64_t s_seen[6];
    for (int i = 0; i < 6; i++) {
      s_annot[i] = atomic_exchange_explicit(&g_eb_annot_stratum_by_len[i], 0,
                                            memory_order_relaxed);
      s_seen[i] = atomic_exchange_explicit(&g_eb_stratum_seen_by_len[i], 0,
                                           memory_order_relaxed);
    }
    const uint64_t post_turns = atomic_exchange_explicit(
        &g_eb_post_turns_total, 0, memory_order_relaxed);
    const uint64_t dfs_movegens = atomic_exchange_explicit(
        &g_eb_dfs_movegen_calls, 0, memory_order_relaxed);
    const uint64_t dfs_copies = atomic_exchange_explicit(
        &g_eb_dfs_game_copies, 0, memory_order_relaxed);
    const uint64_t target_fanout_total = atomic_exchange_explicit(
        &g_eb_target_fanout_total, 0, memory_order_relaxed);
    const uint64_t target_fanout_count = atomic_exchange_explicit(
        &g_eb_target_fanout_count, 0, memory_order_relaxed);
    const uint64_t force_decs = atomic_exchange_explicit(
        &g_eb_force_decrement_calls, 0, memory_order_relaxed);
    const uint64_t exch_gen = atomic_exchange_explicit(
        &g_eb_exch_gen, 0, memory_order_relaxed);
    const uint64_t exch_genbk = atomic_exchange_explicit(
        &g_eb_exch_genbk, 0, memory_order_relaxed);
    const uint64_t exch_slot = atomic_exchange_explicit(
        &g_eb_exch_slot, 0, memory_order_relaxed);
    const uint64_t exch_slotbk = atomic_exchange_explicit(
        &g_eb_exch_slotbk, 0, memory_order_relaxed);
    string_builder_add_formatted_string(
        status_sb,
        " counters: leaves=%llu annot=%llu prop=%llu emitc=%llu "
        "credits=%llu landed=%llu zero=%llu stratum_nb=%llu retries=%llu "
        "strat_cred_L0-5=%llu,%llu,%llu,%llu,%llu,%llu "
        "strat_bump_L0-5=%llu,%llu,%llu,%llu,%llu,%llu "
        "strat_annot_L0-5=%llu,%llu,%llu,%llu,%llu,%llu "
        "strat_seen_L0-5=%llu,%llu,%llu,%llu,%llu,%llu "
        "l3p_outer=%llu l3p_typehit=%llu l3p_typepass=%llu "
        "l3p_iter[STRAT,TILE,PAIR]=%llu,%llu,%llu "
        "l3p_match[STRAT,TILE,PAIR]=%llu,%llu,%llu "
        "perfcnt: post_turns=%llu movegens=%llu copies=%llu "
        "tgt_fanout=%llu/%llu force_decs=%llu "
        "exch[gen=%llu genbk=%llu slot=%llu slotbk=%llu].\n",
        (unsigned long long)leaves,
        (unsigned long long)annot,
        (unsigned long long)prop,
        (unsigned long long)emitc,
        (unsigned long long)fc.credit_calls,
        (unsigned long long)fc.decrements_landed,
        (unsigned long long)fc.noops_already_zero,
        (unsigned long long)fc.stratum_no_bump,
        (unsigned long long)fc.cas_retries,
        (unsigned long long)s_cred[0], (unsigned long long)s_cred[1],
        (unsigned long long)s_cred[2], (unsigned long long)s_cred[3],
        (unsigned long long)s_cred[4], (unsigned long long)s_cred[5],
        (unsigned long long)s_bump[0], (unsigned long long)s_bump[1],
        (unsigned long long)s_bump[2], (unsigned long long)s_bump[3],
        (unsigned long long)s_bump[4], (unsigned long long)s_bump[5],
        (unsigned long long)s_annot[0], (unsigned long long)s_annot[1],
        (unsigned long long)s_annot[2], (unsigned long long)s_annot[3],
        (unsigned long long)s_annot[4], (unsigned long long)s_annot[5],
        (unsigned long long)s_seen[0], (unsigned long long)s_seen[1],
        (unsigned long long)s_seen[2], (unsigned long long)s_seen[3],
        (unsigned long long)s_seen[4], (unsigned long long)s_seen[5],
        (unsigned long long)atomic_exchange_explicit(
            &g_eb_l3plus_outer_loop, 0, memory_order_relaxed),
        (unsigned long long)atomic_exchange_explicit(
            &g_eb_l3plus_type_check_hit, 0, memory_order_relaxed),
        (unsigned long long)atomic_exchange_explicit(
            &g_eb_l3plus_type_check_passed, 0, memory_order_relaxed),
        (unsigned long long)atomic_exchange_explicit(
            &g_eb_l3p_gate_iter_by_prio[0], 0, memory_order_relaxed),
        (unsigned long long)atomic_exchange_explicit(
            &g_eb_l3p_gate_iter_by_prio[1], 0, memory_order_relaxed),
        (unsigned long long)atomic_exchange_explicit(
            &g_eb_l3p_gate_iter_by_prio[2], 0, memory_order_relaxed),
        (unsigned long long)atomic_exchange_explicit(
            &g_eb_l3p_gate_match_by_prio[0], 0, memory_order_relaxed),
        (unsigned long long)atomic_exchange_explicit(
            &g_eb_l3p_gate_match_by_prio[1], 0, memory_order_relaxed),
        (unsigned long long)atomic_exchange_explicit(
            &g_eb_l3p_gate_match_by_prio[2], 0, memory_order_relaxed),
        (unsigned long long)post_turns,
        (unsigned long long)dfs_movegens,
        (unsigned long long)dfs_copies,
        (unsigned long long)target_fanout_total,
        (unsigned long long)target_fanout_count,
        (unsigned long long)force_decs,
        (unsigned long long)exch_gen,
        (unsigned long long)exch_genbk,
        (unsigned long long)exch_slot,
        (unsigned long long)exch_slotbk);
  } else {
    string_builder_add_string(status_sb, "\n");
  }
  thread_control_print(autoplay_worker->args.thread_control,
                       string_builder_peek(status_sb));
  string_builder_destroy(status_sb);

  // SIGUSR1-triggered mid-run dump: `kill -USR1 <pid>` sets a flag; this
  // tick consumes it and writes a numbered snapshot next to the configured
  // MAGPIE_FORCE_TABLE_DUMP path. Numbered .snapN suffix avoids overwriting
  // the final exit dump.
  if (shared_data->force_table &&
      force_table_consume_dump_request()) {
    static _Atomic int dump_seq = 0;
    const char *base = getenv("MAGPIE_FORCE_TABLE_DUMP");
    if (base && base[0] != '\0') {
      int n = atomic_fetch_add_explicit(&dump_seq, 1, memory_order_relaxed);
      char path[1024];
      snprintf(path, sizeof(path), "%s.snap%d", base, n);
      force_table_dump_remaining(shared_data->force_table, path,
                                 shared_data->ld);
      fprintf(stderr, "force_table: dumped snapshot to %s\n", path);
    }
  }
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
  // eb_emit_leaf already handled deficits at the target turn.
  ForceTable *ft = autoplay_worker->shared_data->force_table;
  if (ft != NULL && game_runner->pending_force_target != NULL &&
      !autoplay_worker->shared_data->eb_branch_active) {
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
// Stringify a player's rack into `out`. Sorted alphabetic, blanks
// (machine letter 0) appear first as '?'. Up to RACK_SIZE chars + null.
static void stringify_bag_into(const Game *game, char *out, int max_len) {
  const LetterDistribution *ld = game_get_ld(game);
  const Bag *bag = game_get_bag(game);
  const uint16_t dist_size = ld_get_size(ld);
  int n = 0;
  for (uint16_t i = 0; i < dist_size && n < max_len - 1; i++) {
    const int c = bag_get_letter(bag, i);
    for (int k = 0; k < c && n < max_len - 1; k++) {
      out[n++] = ld->ld_ml_to_hl[i][0];
    }
  }
  out[n] = '\0';
}

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
__attribute__((unused))
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

  // Outer loop: iterate priorities so the slot order still favors
  // PAIR-matching moves first, then TILE, then STRATUM. The per-priority
  // scan only checks for ANY match at that priority — once a move is
  // assigned to a slot, a second scan collects ALL matches (across all
  // priorities) into the slot's multi-target list for game-end credit.
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
      const int eff_diff = cur_diff + score;
      // Inline closure: does slot[t] match the move at the requested kind?
      // (Returns true only when fs->kind matches the kind argument; for
      // the gating scan pass kind=priority; for the collect-all pass
      // pass kind=-1 to accept any kind.)
      bool gate_matched = false;
      for (int t = 0; t < bucket_count; t++) {
        const uint32_t req = bitmaps[t];
        if ((leave_bm & req) != req) continue;
        ForceTargetSlot *fs = &bucket_slots[t];
        if (fs->deficit <= 0 || (int)fs->kind != priority) continue;
        if (eff_diff < fs->diff_min || eff_diff > fs->diff_max) continue;
        if (fs->subleave_count == 2 &&
            fs->subleave_mls[0] == fs->subleave_mls[1] &&
            leaves[m].array[fs->subleave_mls[0]] < 2) continue;
        if (fs->leave_type != LEAVE_TYPE_ALL) {  // typed cell (L5/L6 only)
          if (move_type_cache[m] < 0) {
            move_type_cache[m] = (int8_t)force_classify_leave(&leaves[m], ld);
          }
          if ((LeaveType)move_type_cache[m] != (LeaveType)fs->leave_type)
            continue;
        }
        gate_matched = true;
        break;
      }
      if (!gate_matched) continue;
      // Collect ALL matches for this move (across PAIR/TILE/STRATUM kinds)
      // into the slot's target list. Same predicate set, just no per-kind
      // filter. Cap at EB_MAX_LEAVE_TARGETS_PER_MOVE; in practice a single
      // play matches at most ~15-25 leave cells (tiles + pairs in leave +
      // optional stratum cell).
      uint8_t n_matches = 0;
      for (int t = 0;
           t < bucket_count && n_matches < EB_MAX_LEAVE_TARGETS_PER_MOVE;
           t++) {
        const uint32_t req = bitmaps[t];
        if ((leave_bm & req) != req) continue;
        ForceTargetSlot *fs = &bucket_slots[t];
        if (fs->deficit <= 0) continue;
        if ((int)fs->kind != FORCE_TARGET_PAIR &&
            (int)fs->kind != FORCE_TARGET_TILE &&
            (int)fs->kind != FORCE_TARGET_STRATUM) {
          continue;  // skip BAG_TILE — handled separately at snap-capture
        }
        if (eff_diff < fs->diff_min || eff_diff > fs->diff_max) continue;
        if (fs->subleave_count == 2 &&
            fs->subleave_mls[0] == fs->subleave_mls[1] &&
            leaves[m].array[fs->subleave_mls[0]] < 2) continue;
        if (fs->leave_type != LEAVE_TYPE_ALL) {  // typed cell (L5/L6 only)
          if (move_type_cache[m] < 0) {
            move_type_cache[m] = (int8_t)force_classify_leave(&leaves[m], ld);
          }
          if ((LeaveType)move_type_cache[m] != (LeaveType)fs->leave_type)
            continue;
        }
        gr->eb_force_targets_for_slot[slot][n_matches++] = fs->cold;
      }
      gr->eb_force_target_count_for_slot[slot] = n_matches;
      move_copy(gr->eb_action_buf[slot], move);
      gr->eb_action_present[slot] = true;
      used_move[m] = true;
      slot++;
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
      // Gating scan: does this slot match ANY target at the current priority?
      bool gate_matched = false;
      const int eff_diff = cur_diff + score;
      // BUG: must be plain int, not LeaveType. clang treats LeaveType as
      // unsigned for the `< 0` comparison, so `(LeaveType)-1 < 0` is
      // false and force_classify_leave is never called — leaves
      // cached_leave_type at its sentinel, which then fails the type
      // check against every cell's leave_type, silently rejecting ALL
      // length>=3 force-table cells.
      int cached_leave_type = -1;
      if (leave_len >= 3 && priority >= 0 && priority <= 3) {
        atomic_fetch_add_explicit(&g_eb_l3p_gate_iter_by_prio[priority], 1,
                                   memory_order_relaxed);
      }
      for (int t = 0; t < bucket_count; t++) {
        const uint32_t req = bitmaps[t];
        if ((leave_bm & req) != req) continue;
        ForceTargetSlot *fs = &bucket_slots[t];
        if (fs->deficit <= 0 || (int)fs->kind != priority) continue;
        if (eff_diff < fs->diff_min || eff_diff > fs->diff_max) continue;
        if (fs->subleave_count == 2 &&
            fs->subleave_mls[0] == fs->subleave_mls[1] &&
            leave.array[fs->subleave_mls[0]] < 2) continue;
        if (fs->leave_type != LEAVE_TYPE_ALL) {  // typed cell (L5/L6 only)
          if (cached_leave_type < 0)
            cached_leave_type = force_classify_leave(&leave, ld);
          if ((LeaveType)cached_leave_type != (LeaveType)fs->leave_type)
            continue;
        }
        gate_matched = true;
        break;
      }
      if (gate_matched && leave_len >= 3 && priority >= 0 && priority <= 3) {
        atomic_fetch_add_explicit(&g_eb_l3p_gate_match_by_prio[priority], 1,
                                   memory_order_relaxed);
      }
      if (!gate_matched) continue;
      // Collect ALL matches for this slot's move (across all leave kinds)
      // for game-end credit. Mirrors eb_append_force_target_slots's
      // multi-credit collection.
      uint8_t n_matches = 0;
      for (int t = 0;
           t < bucket_count && n_matches < EB_MAX_LEAVE_TARGETS_PER_MOVE;
           t++) {
        const uint32_t req = bitmaps[t];
        if ((leave_bm & req) != req) continue;
        ForceTargetSlot *fs = &bucket_slots[t];
        // Diagnostic: count every STRATUM slot SEEN at this position,
        // before any further predicate filtering.
        if ((int)fs->kind == FORCE_TARGET_STRATUM) {
          int ll = fs->leave_length;
          if (ll < 0) ll = 0;
          if (ll > 5) ll = 5;
          atomic_fetch_add_explicit(&g_eb_stratum_seen_by_len[ll], 1,
                                     memory_order_relaxed);
        }
        if (fs->deficit <= 0) continue;
        if ((int)fs->kind != FORCE_TARGET_PAIR &&
            (int)fs->kind != FORCE_TARGET_TILE &&
            (int)fs->kind != FORCE_TARGET_STRATUM) continue;
        if (eff_diff < fs->diff_min || eff_diff > fs->diff_max) continue;
        if (fs->subleave_count == 2 &&
            fs->subleave_mls[0] == fs->subleave_mls[1] &&
            leave.array[fs->subleave_mls[0]] < 2) continue;
        if (fs->leave_type != LEAVE_TYPE_ALL) {  // typed cell (L5/L6 only)
          atomic_fetch_add_explicit(&g_eb_l3plus_type_check_hit, 1,
                                     memory_order_relaxed);
          if (cached_leave_type < 0)
            cached_leave_type = force_classify_leave(&leave, ld);
          if ((LeaveType)cached_leave_type != (LeaveType)fs->leave_type)
            continue;
          atomic_fetch_add_explicit(&g_eb_l3plus_type_check_passed, 1,
                                     memory_order_relaxed);
        }
        gr->eb_force_targets_for_slot[s][n_matches++] = fs->cold;
        atomic_fetch_add_explicit(&g_eb_annotate_cells_added, 1,
                                   memory_order_relaxed);
        if ((int)fs->kind == FORCE_TARGET_STRATUM) {
          int ll = fs->leave_length;
          if (ll < 0) ll = 0;
          if (ll > 5) ll = 5;
          atomic_fetch_add_explicit(&g_eb_annot_stratum_by_len[ll], 1,
                                     memory_order_relaxed);
        }
      }
      gr->eb_force_target_count_for_slot[s] = n_matches;
      used_slot[s] = true;
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
    gr->eb_force_target_count_for_slot[i] = 0;
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
  atomic_fetch_add_explicit(&g_eb_dfs_movegen_calls, 1,
                            memory_order_relaxed);
  generate_moves(&gen_args);
  const int n_moves = move_list_get_count(ml);
  move_list_sort_moves(ml);

  const LetterDistribution *ld = game_get_ld(gr->game);
  const uint16_t ld_size = ld_get_size(ld);
  int max_slot = -1;
  int play_slot = -1;  // populated slot index of the best-play action

  // Blank-exchange trace stage 1: count exchange moves straight from
  // generate_moves, and how many throw a blank (thrown tile == ml 0).
  if (role == EB_ROLE_TARGET) {
    for (int m = 0; m < n_moves; m++) {
      const Move *mv = move_list_get_move(ml, m);
      if (move_get_type(mv) != GAME_EVENT_EXCHANGE) continue;
      atomic_fetch_add_explicit(&g_eb_exch_gen, 1, memory_order_relaxed);
      const int tl = move_get_tiles_length(mv);
      for (int i = 0; i < tl; i++) {
        if (move_get_tile(mv, i) == BLANK_MACHINE_LETTER) {
          atomic_fetch_add_explicit(&g_eb_exch_genbk, 1, memory_order_relaxed);
          break;
        }
      }
    }
  }

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

    // Targeted-play fanout: build a set of sigs that the play_index
    // says are deficit-rich for this rack at the current threshold.
    // Plays whose sig is in this set are added to the fanout in
    // addition to all exchanges + the top-equity play. Plays not in
    // the set (and not top-equity) are skipped — that's the savings.
    enum { MAX_TARGETED = 32 };
    static _Thread_local char targeted_sig[MAX_TARGETED][SIG_LEN];
    int n_targeted = 0;
    bool targeted_filter_active = false;
    if (w->shared_data->play_index != NULL && gr->eb_target_rack_id >= 0) {
      const PlayIndex *pi = w->shared_data->play_index;
      const ForceTable *ft = w->shared_data->force_table;
      // Linear threshold decay: 0.01 at completion=0, 0.0 at completion=1.
      double thr = 0.01;
      if (ft && w->shared_data->initial_total_deficit > 0) {
        const int64_t cur = force_table_total_remaining(ft);
        double comp = 1.0 - ((double)cur /
                             (double)w->shared_data->initial_total_deficit);
        if (comp < 0.0) comp = 0.0;
        if (comp > 1.0) comp = 1.0;
        thr = 0.01 * (1.0 - comp);
      }
      uint32_t pids[MAX_TARGETED];
      int npicked = play_index_pick_targeted_plays(
          pi, (uint32_t)gr->eb_target_rack_id, thr, MAX_TARGETED, pids);
      for (int t = 0; t < npicked && n_targeted < MAX_TARGETED; t++) {
        PlayRecord pr;
        if (!play_index_get_play(pi, pids[t], &pr)) continue;
        if (pr.action_kind != 2) continue;  // play moves only
        char leave_buf[RACK_SIZE + 2] = {0};
        int ll = (pr.leave_len < RACK_SIZE) ? pr.leave_len : RACK_SIZE;
        memcpy(leave_buf, pr.leave_str, ll);
        snprintf(targeted_sig[n_targeted], SIG_LEN, "P%d|%s",
                 pr.score, leave_buf);
        n_targeted++;
      }
      // Force-include the play picked by play_index_pick_starved_rack.
      // pick_targeted_plays uses score_play's sum-of-cell_priority
      // ranking, which at N=100 can drop a play covering a high-target
      // stuck cell (low priority differential under the [0.5, 1.0]
      // floor) below plays covering many small-deficit cells. Inserting
      // the picked sig directly guarantees the scheduler's intent
      // reaches the fanout. Pass/exchange targets are already in the
      // fanout unconditionally, so we only handle action_kind==2.
      if (gr->eb_target_play_id >= 0 && n_targeted < MAX_TARGETED) {
        PlayRecord pr;
        if (play_index_get_play(pi, (uint32_t)gr->eb_target_play_id,
                                &pr) && pr.action_kind == 2) {
          char leave_buf[RACK_SIZE + 2] = {0};
          int ll = (pr.leave_len < RACK_SIZE) ? pr.leave_len : RACK_SIZE;
          memcpy(leave_buf, pr.leave_str, ll);
          char tsig[SIG_LEN];
          snprintf(tsig, SIG_LEN, "P%d|%s", pr.score, leave_buf);
          bool dup = false;
          for (int t = 0; t < n_targeted; t++) {
            if (strcmp(targeted_sig[t], tsig) == 0) { dup = true; break; }
          }
          if (!dup) {
            memcpy(targeted_sig[n_targeted++], tsig, SIG_LEN);
          }
        }
      }
      targeted_filter_active = true;
    }

    for (int phase = 0; phase < 2 && slot < EB_MAX_ACTIONS; phase++) {
      const game_event_t kind = (phase == 0)
          ? GAME_EVENT_EXCHANGE
          : GAME_EVENT_TILE_PLACEMENT_MOVE;
      if (kind == GAME_EVENT_TILE_PLACEMENT_MOVE && !has_play) continue;
      bool first_play_added = false;
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
        // Blank-exchange trace stage 3: when a blank-throwing exchange
        // is about to be dropped as a dup, dump its thrown tiles +
        // rendered leave + sig so we can see what it collided with.
        // Capped to the first ~30 occurrences process-wide.
        if (dup && kind == GAME_EVENT_EXCHANGE) {
          int candbk = 0;
          const int ctl = move_get_tiles_length(cand);
          for (int i = 0; i < ctl; i++) {
            if (move_get_tile(cand, i) == BLANK_MACHINE_LETTER) { candbk = 1; break; }
          }
          if (candbk) {
            static _Atomic int g_eb_exch_dup_logged = 0;
            int seen = atomic_fetch_add_explicit(&g_eb_exch_dup_logged, 1,
                                                 memory_order_relaxed);
            if (seen < 30) {
              char thrown[32] = {0};
              int tn = 0;
              for (int i = 0; i < ctl && tn < 30; i++) {
                const MachineLetter t = move_get_tile(cand, i);
                thrown[tn++] = (t == BLANK_MACHINE_LETTER)
                                   ? '?' : (char)('A' + t - 1);
              }
              char rackbuf[RACK_SIZE + 2] = {0};
              eb_canonical_rack(gr->game, p_idx, rackbuf);
              fprintf(stderr,
                      "EXCHDUP: rack=%s thrown=[%s] leave='%s' sig='%s' "
                      "collides-with-existing-slot\n",
                      rackbuf, thrown, leave_str, sig);
            }
          }
        }
        if (dup) continue;
        // For TILE_PLACEMENT plays under the play_index targeted filter:
        // include only the first distinct play (top-equity) PLUS plays
        // whose sig matches the targeted set. Skip everything else.
        if (kind == GAME_EVENT_TILE_PLACEMENT_MOVE &&
            targeted_filter_active && first_play_added) {
          bool match = false;
          for (int k = 0; k < n_targeted; k++) {
            if (strcmp(targeted_sig[k], sig) == 0) { match = true; break; }
          }
          if (!match) continue;
        }
        memcpy(slot_sig[n_sigs++], sig, SIG_LEN);
        move_copy(gr->eb_action_buf[slot], cand);
        gr->eb_action_present[slot] = true;
        // Blank-exchange trace stage 2: count exchange slots that
        // survived dedup, and how many throw a blank.
        if (kind == GAME_EVENT_EXCHANGE) {
          atomic_fetch_add_explicit(&g_eb_exch_slot, 1, memory_order_relaxed);
          const int tl = move_get_tiles_length(cand);
          for (int i = 0; i < tl; i++) {
            if (move_get_tile(cand, i) == BLANK_MACHINE_LETTER) {
              atomic_fetch_add_explicit(&g_eb_exch_slotbk, 1,
                                        memory_order_relaxed);
              break;
            }
          }
        }
        if (kind == GAME_EVENT_TILE_PLACEMENT_MOVE) {
          if (play_slot < 0) play_slot = slot;
          first_play_added = true;
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
  // implicitly at the target turn whenever a force_table is loaded.
  if (role == EB_ROLE_TARGET && w->shared_data->force_table != NULL) {
    eb_annotate_force_targets_to_slots(w, gr, max_slot);
  }

  // Need at least 2 populated slots for the fork to be meaningful.
  int populated = 0;
  for (int i = 0; i <= max_slot; i++) {
    if (gr->eb_action_present[i]) populated++;
  }
  if (role == EB_ROLE_TARGET) {
    atomic_fetch_add_explicit(&g_eb_target_fanout_total, (uint64_t)populated,
                              memory_order_relaxed);
    atomic_fetch_add_explicit(&g_eb_target_fanout_count, 1,
                              memory_order_relaxed);
  }
  return populated >= 2 ? max_slot + 1 : 0;
}

// Emit captured eb_snaps for this leaf branch with its eventual_outcome.
static void eb_emit_leaf(AutoplayWorker *w, GameRunner *gr, uint64_t branch_id) {
  EmptyBoardRecorder *ebr = w->shared_data->empty_board_recorder;
  EmptyBoardStrataRecorder *ebs = w->shared_data->empty_board_strata_recorder;
  if ((!ebr && !ebs) || gr->eb_n_snaps == 0) return;
  atomic_fetch_add_explicit(&g_eb_leaves_emitted, 1, memory_order_relaxed);
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

  // Force-target credit at the target snap. Multi-credit: a single play
  // can match many leave cells (tile + pair + stratum kinds at the same
  // (stratum, diff)) — credit ALL of them, not just one. Mirrors the
  // multi-credit pattern already used for BAG_TILE cells below.
  if (w->shared_data->force_table != NULL &&
      gr->eb_snap_force_target_count[target_turn] > 0) {
    const int p_idx = gr->eb_snaps[idx].player_on_turn;
    const int my_final =
        equity_to_int(player_get_score(game_get_player(gr->game, p_idx)));
    const int opp_final =
        equity_to_int(
            player_get_score(game_get_player(gr->game, 1 - p_idx)));
    const bool is_tie = (my_final == opp_final);
    const bool is_win = (my_final > opp_final);
    const uint8_t cnt = gr->eb_snap_force_target_count[target_turn];
    for (uint8_t k = 0; k < cnt; k++) {
      ForceTarget *ft_target = gr->eb_snap_force_targets[target_turn][k];
      if (ft_target != NULL) {
        atomic_fetch_add_explicit(&g_eb_emit_cells_credited, 1,
                                   memory_order_relaxed);
        atomic_fetch_add_explicit(&g_eb_force_decrement_calls, 1,
                                   memory_order_relaxed);
        force_table_credit_game(
            w->shared_data->force_table, ft_target,
            gr->eb_snaps[idx].move_score, is_win, is_tie);
      }
    }
  }

  // BAG_TILE credit: each cell matched at snap-capture time gets one
  // count-based decrement (force_table_credit_game routes BAG_TILE
  // through the non-STRATUM branch automatically).
  if (w->shared_data->force_table != NULL && gr->eb_snap_bag_count[idx] > 0) {
    for (int b = 0; b < gr->eb_snap_bag_count[idx]; b++) {
      ForceTarget *bt = gr->eb_snap_bag_targets[idx][b];
      if (bt != NULL) {
        // is_win/is_tie irrelevant for count-based credit; pass false/false.
        force_table_credit_game(w->shared_data->force_table, bt,
                                gr->eb_snaps[idx].move_score, false, false);
      }
    }
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

// Inject a rack from a single explicit category. Returns true iff a
// rack was successfully drawn. Used by both the legacy single-pick
// injector and the 5-way fork at TARGET (one branch per category).
// cat: 0=pass, 1=exch, 2=bingo, 3=rare, 4=play.
static bool inject_target_turn_rack_for_category(
    AutoplayWorker *w, GameRunner *gr, int cat, uint64_t seed) {
  const int p = game_get_player_on_turn_index(gr->game);
  if (p != w->shared_data->rare_target_player) return false;
  const char *target = NULL;
  if (cat == 0 && w->shared_data->pass_pool != NULL) {
    pass_cycle_sample_p1(w->shared_data->pass_pool, seed, &target);
  } else if (cat == 1 && w->shared_data->exch_pool != NULL) {
    pass_cycle_sample_p1(w->shared_data->exch_pool, seed, &target);
  } else if (cat == 2 && w->shared_data->bingo_pool != NULL) {
    pass_cycle_sample_p1(w->shared_data->bingo_pool, seed, &target);
  } else if (cat == 3) {
    if (w->shared_data->play_index != NULL) {
      uint32_t rid = 0;
      // Linear K growth: 64 at completion=0 → 1024 at completion=1.
      // Bigger heap late in the run → broader candidate pool, drains
      // long-tail stuck cells the top-64 would have starved.
      int top_k = 64;
      if (w->shared_data->force_table &&
          w->shared_data->initial_total_deficit > 0) {
        const int64_t cur =
            force_table_total_remaining(w->shared_data->force_table);
        double comp = 1.0 - ((double)cur /
                             (double)w->shared_data->initial_total_deficit);
        if (comp < 0.0) comp = 0.0;
        if (comp > 1.0) comp = 1.0;
        top_k = (int)(64.0 + (1024.0 - 64.0) * comp);
      }
      if (w->shared_data->outcome_priors != NULL) {
        target = play_index_sample_rack_outcome_aware(
            w->shared_data->play_index,
            w->shared_data->outcome_priors,
            w->shared_data->outcome_priors_lambda,
            seed, top_k, &rid);
      } else {
        target = play_index_sample_rack_deficit_aware(
            w->shared_data->play_index, seed, top_k, &rid);
      }
      if (target) gr->eb_target_rack_id = (int64_t)rid;
    } else if (w->shared_data->rare_rack_cells != NULL) {
      const int idx = rare_pool_sample_deficit_aware(
          w->shared_data->rare_rack_cells, seed);
      if (idx >= 0) {
        target = rare_pool_get_rack(w->shared_data->rare_rack_cells, idx);
      }
    }
  } else if (cat == 4 && w->shared_data->play_pool != NULL) {
    pass_cycle_sample_p1(w->shared_data->play_pool, seed, &target);
  }
  if (target == NULL) return false;
  return swap_player_rack(gr->game, p, target);
}

// At the start of the target turn, pick a rack source weighted by an
// adaptively-ramped `rare_frac`:
//   completion = 1 - current_deficit / initial_deficit
//   rare_frac  = MIN + (MAX - MIN) * completion^2
// where MIN/MAX come from MAGPIE_EB_RARE_FRAC_{MIN,MAX} (defaults 0.20,
// 0.60). The four non-rare categories share equally in (1 - rare_frac).
// Missing pools leave the natural rack for their slice.
// Only fires when the on-turn player matches rare_target_player.
static void inject_target_turn_rack_by_category(
    AutoplayWorker *w, GameRunner *gr, uint64_t seed) {
  // Reset per-call so stale values from the previous game don't leak
  // into the consumer at eb_enumerate_actions. The late-stage path
  // below sets play_id when it succeeds; other paths leave it at -1.
  gr->eb_target_play_id = -1;
  const int p = game_get_player_on_turn_index(gr->game);
  if (p != w->shared_data->rare_target_player) return;

  // Phase 4: late-stage targeted mode bypasses the category sampler
  // entirely. Pick a starved cell, then a rack reaching it, and capture
  // the specific play_id covering that cell so eb_enumerate_actions can
  // force-include it in the fanout (otherwise sum-aggregated score_play
  // can drop the targeted play under bias toward many-small-cell plays).
  if (atomic_load_explicit(&w->shared_data->eb_late_stage,
                           memory_order_relaxed) &&
      w->shared_data->play_index) {
    uint32_t rid = 0;
    uint32_t pid = 0;
    const char *target = play_index_pick_starved_rack(
        w->shared_data->play_index, seed, &rid, &pid);
    if (target) {
      gr->eb_target_rack_id = (int64_t)rid;
      gr->eb_target_play_id = (int64_t)pid;
      swap_player_rack(gr->game, p, target);
    }
    return;
  }

  // Compute current rare_frac via the adaptive ramp. Reading the deficit
  // is O(num_targets) and runs once per recorded T6 game (~1/game) — a
  // few microseconds per call, negligible vs the rest of the turn.
  double rare_frac = w->shared_data->rare_frac_min;
  if (w->shared_data->rare_frac_max > w->shared_data->rare_frac_min &&
      w->shared_data->initial_total_deficit > 0 &&
      w->shared_data->force_table != NULL) {
    const int64_t cur =
        force_table_total_remaining(w->shared_data->force_table);
    double completion =
        1.0 - ((double)cur / (double)w->shared_data->initial_total_deficit);
    if (completion < 0.0) completion = 0.0;
    if (completion > 1.0) completion = 1.0;
    rare_frac = w->shared_data->rare_frac_min +
                (w->shared_data->rare_frac_max -
                 w->shared_data->rare_frac_min) * completion * completion;
  }

  // Two-stage decision:
  //   1) draw r ∈ [0, 1). If r < rare_frac → cat = 3 (rare).
  //   2) otherwise pick uniformly among {0=pass, 1=exch, 2=bingo, 4=play}.
  // 0=pass, 1=exch, 2=bingo, 3=rare, 4=play.
  const uint64_t h = seed * 0x9e3779b97f4a7c15ULL + 0x123456789abcdef0ULL;
  const double r = ((double)(h >> 11)) / (double)(1ULL << 53);
  int cat;
  if (r < rare_frac) {
    cat = 3;
  } else {
    // Map r ∈ [rare_frac, 1) to one of 4 non-rare categories.
    const uint64_t h2 = h * 0xbf58476d1ce4e5b9ULL;
    cat = (int)((h2 >> 33) % 4ULL);
    if (cat >= 3) cat++;  // skip slot 3 (rare)
  }

  const char *target = NULL;
  if (cat == 0 && w->shared_data->pass_pool != NULL) {
    pass_cycle_sample_p1(w->shared_data->pass_pool, seed, &target);
  } else if (cat == 1 && w->shared_data->exch_pool != NULL) {
    pass_cycle_sample_p1(w->shared_data->exch_pool, seed, &target);
  } else if (cat == 2 && w->shared_data->bingo_pool != NULL) {
    pass_cycle_sample_p1(w->shared_data->bingo_pool, seed, &target);
  } else if (cat == 3) {
    if (w->shared_data->play_index != NULL) {
      uint32_t rid = 0;
      // Linear K growth: 64 at completion=0 → 1024 at completion=1.
      // Bigger heap late in the run → broader candidate pool, drains
      // long-tail stuck cells the top-64 would have starved.
      int top_k = 64;
      if (w->shared_data->force_table &&
          w->shared_data->initial_total_deficit > 0) {
        const int64_t cur =
            force_table_total_remaining(w->shared_data->force_table);
        double comp = 1.0 - ((double)cur /
                             (double)w->shared_data->initial_total_deficit);
        if (comp < 0.0) comp = 0.0;
        if (comp > 1.0) comp = 1.0;
        top_k = (int)(64.0 + (1024.0 - 64.0) * comp);
      }
      if (w->shared_data->outcome_priors != NULL) {
        target = play_index_sample_rack_outcome_aware(
            w->shared_data->play_index,
            w->shared_data->outcome_priors,
            w->shared_data->outcome_priors_lambda,
            seed, top_k, &rid);
      } else {
        target = play_index_sample_rack_deficit_aware(
            w->shared_data->play_index, seed, top_k, &rid);
      }
      if (target) gr->eb_target_rack_id = (int64_t)rid;
    } else if (w->shared_data->rare_rack_cells != NULL) {
      const int idx = rare_pool_sample_deficit_aware(
          w->shared_data->rare_rack_cells, seed);
      if (idx >= 0) {
        target = rare_pool_get_rack(w->shared_data->rare_rack_cells, idx);
      }
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
    // Phase 0 capture: P1's rack and bag at T6 entry, before any rack
    // injection. Once-per-game (only fires when on-turn matches the
    // recording player so each game emits exactly one row). After
    // capture we mark the game over so we don't waste cycles playing
    // out post-T6 (Phase 0 only needs the snapshot).
    if (gr->eb_active &&
        gr->eb_n_snaps == w->shared_data->eb_target_turn - 1 &&
        board_get_tiles_played(game_get_board(gr->game)) == 0 &&
        w->shared_data->pre_t6_capture_file != NULL &&
        game_get_player_on_turn_index(gr->game) ==
            w->shared_data->rare_target_player) {
      const int p1_idx = 1 - w->shared_data->rare_target_player;
      char p1_rack_buf[RACK_SIZE + 2] = {0};
      char bag_buf[256] = {0};
      pass_cycle_stringify_rack(gr->game, p1_idx, p1_rack_buf);
      stringify_bag_into(gr->game, bag_buf, sizeof(bag_buf));
      cpthread_mutex_lock(&w->shared_data->pre_t6_capture_mutex);
      fprintf(w->shared_data->pre_t6_capture_file, "%s,%s\n",
              p1_rack_buf, bag_buf);
      cpthread_mutex_unlock(&w->shared_data->pre_t6_capture_mutex);
      // End the game immediately — Phase 0 doesn't need the playout.
      game_set_game_end_reason(gr->game, GAME_END_REASON_STANDARD);
      return;
    }

    if (gr->eb_active &&
        gr->eb_n_snaps == w->shared_data->eb_target_turn - 1 &&
        board_get_tiles_played(game_get_board(gr->game)) == 0 &&
        (w->shared_data->pass_pool != NULL ||
         w->shared_data->exch_pool != NULL ||
         w->shared_data->bingo_pool != NULL ||
         w->shared_data->play_pool != NULL ||
         w->shared_data->rare_rack_cells != NULL ||
         w->shared_data->play_index != NULL)) {
      // 5-way category sampler at TARGET turn. Slot 3 (rare) uses
      // play_index when loaded (deficit-targeted), else rare_rack_cells.
      // When only play_index is set, all 4 other slots fall through to
      // the natural rack (since their pool fields are NULL) — equivalent
      // to 100% play_index injection.
      inject_target_turn_rack_by_category(w, gr, gr->seed ^ branch_id);
    }
    const int n_actions = eb_enumerate_actions(w, gr);
    if (n_actions == 0) {
      // Not a fork point — play one move using normal selection.
      // Clear any stale leave-target snap count for the about-to-be-played
      // turn. If we don't, a previous fork iteration's count carries over
      // and eb_emit_leaf credits stale ForceTarget* slots that may already
      // have deficit==0 (force_table_decrement_target no-ops, but the call
      // overhead is real and the credit math is wrong).
      const int natural_turn = gr->eb_n_snaps + 1;
      if (natural_turn >= 1 && natural_turn <= 6) {
        gr->eb_snap_force_target_count[natural_turn] = 0;
      }
      if (natural_turn > w->shared_data->eb_target_turn) {
        atomic_fetch_add_explicit(&g_eb_post_turns_total, 1,
                                  memory_order_relaxed);
      }
      game_runner_play_move(w, gr);
      continue;
    }

    // Snapshot enumerated actions to stack BEFORE recursing — eb_action_buf
    // is per-runner and inner forks during recursion will overwrite it.
    // n_actions is max_slot+1; iterate stable slot indices and skip absent.
    Move local_actions[EB_MAX_ACTIONS];
    bool local_present[EB_MAX_ACTIONS];
    ForceTarget *local_force_targets[EB_MAX_ACTIONS]
                                    [EB_MAX_LEAVE_TARGETS_PER_MOVE];
    uint8_t local_force_target_count[EB_MAX_ACTIONS];
    for (int s = 0; s < n_actions; s++) {
      local_present[s] = gr->eb_action_present[s];
      local_force_target_count[s] = gr->eb_force_target_count_for_slot[s];
      for (uint8_t k = 0; k < local_force_target_count[s]; k++) {
        local_force_targets[s][k] = gr->eb_force_targets_for_slot[s][k];
      }
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
    atomic_fetch_add_explicit(&g_eb_dfs_game_copies, 1,
                              memory_order_relaxed);

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
      // Track force-targets attached to this slot at this turn (empty
      // count if natural). eb_emit_leaf consults this per-snap to credit
      // every matched leave cell.
      if (fork_turn >= 1 && fork_turn <= 6) {
        const uint8_t cnt = local_force_target_count[s];
        gr->eb_snap_force_target_count[fork_turn] = cnt;
        for (uint8_t k = 0; k < cnt; k++) {
          gr->eb_snap_force_targets[fork_turn][k] = local_force_targets[s][k];
          atomic_fetch_add_explicit(&g_eb_dfs_cells_propagated, 1,
                                     memory_order_relaxed);
        }
      }
      game_runner_play_move(w, gr);
      gr->eb_forced_move = NULL;
      // Encode the SLOT INDEX (not iteration index) so the action at each
      // fork-depth is recoverable regardless of which slots happened to be
      // populated for that position.
      play_eb_dfs(w, gr, (branch_id << 8) | (uint64_t)(s + 1));
      game_copy(gr->game, saved_game);
      atomic_fetch_add_explicit(&g_eb_dfs_game_copies, 1,
                                memory_order_relaxed);
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
  // Trajectory recorder: commit each game's buffered rows iff it ended
  // normally (GAME_END_REASON_STANDARD). Games ending via consecutive-
  // zeros (6-pass terminus) produce pathological positions; discard.
  {
    TrajectoryRecorder *traj_r_end =
        autoplay_worker->shared_data->trajectory_recorder;
    if (traj_r_end) {
      GameRunner *grs[2] = {game_runner1, game_runner2};
      for (int gi = 0; gi < 2; gi++) {
        GameRunner *gr = grs[gi];
        if (!gr || !gr->trajectory_buf) continue;
        if (game_get_game_end_reason(gr->game) == GAME_END_REASON_STANDARD) {
          // Final scores reflect end-of-game tile-bag penalty adjustment
          // (magpie applies this internally on STANDARD termination).
          const int final_p1 = equity_to_int(
              player_get_score(game_get_player(gr->game, 0)));
          const int final_p2 = equity_to_int(
              player_get_score(game_get_player(gr->game, 1)));
          trajectory_game_buffer_commit(traj_r_end, gr->trajectory_buf,
                                         final_p1, final_p2);
        } else {
          trajectory_game_buffer_discard(gr->trajectory_buf);
        }
      }
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

// Phase 2 baseline worker loop. Runs T6-from-scratch task batches:
// for each task, set both racks, force the target's first move, play
// out the rest with HastyBot, accumulate per-batch (W, L, T) outcome.
// One result row per task.
//
// Naming convention: in the offline/baseline frame, "target" = offline
// player 0 = the rack we're characterizing (= live P2 at T6).
// "opp" = offline player 1 = the opponent rack from pre_t6_pool (=
// live P1 at T6).
static void t6_baseline_run_worker(AutoplayWorker *worker, GameRunner *gr) {
  T6BaselineState *state = worker->shared_data->t6_baseline;
  ThreadControl *thread_control = worker->args.thread_control;
  T6BaselineTask task;
  uint64_t game_seed = (uint64_t)worker->worker_index * 0x9e3779b97f4a7c15ULL;
  while (
      thread_control_get_status(thread_control) !=
          THREAD_CONTROL_STATUS_USER_INTERRUPT &&
      t6_baseline_get_next_task(state, &task)) {
    int n_W = 0, n_L = 0, n_T = 0;
    int n_games_run = 0;
    int64_t target_score_sum = 0, opp_score_sum = 0;
    // Per-task move metadata (leave, blanks_used, score) — derived
    // from the first successful parse + play, then reused for the
    // result record.
    char leave_str[RACK_SIZE + 2] = {0};
    int blanks_used = 0;
    int task_score = 0;
    bool meta_captured = false;
    for (int g = 0; g < task.n_games; g++) {
      game_reset(gr->game);
      // Reset GameRunner fields normally cleared by game_runner_start (we
      // bypass that path). Without these, turn_number / pass_cycle_* /
      // eb_* state from a previous task can poison this game.
      gr->turn_number = 0;
      gr->force_draw = false;
      gr->force_triggered = false;
      gr->pending_force_target = NULL;
      gr->pending_force_diff = 0;
      gr->pending_force_player_index = 0;
      gr->pass_cycle_active = false;
      gr->pass_cycle_abandoned = false;
      gr->pass_cycle_n_moves = 0;
      gr->eb_active = false;
      gr->eb_forced_move = NULL;
      gr->eb_n_snaps = 0;
      gr->eb_actions_p0[0] = '\0';
      gr->eb_actions_p0_off = 0;
      gr->eb_actions_p1[0] = '\0';
      gr->eb_actions_p1_off = 0;
      if (!swap_player_rack(gr->game, 0, task.target_rack)) continue;
      // Retry opp_rack sampling: pool racks may share scarce tiles with
      // the target (e.g. target with 2 Ys leaves none in the bag for opp).
      // Resample until swap_player_rack succeeds; cap at 64 attempts to
      // guard against pathological cases where almost no pool rack fits.
      bool opp_set = false;
      for (int retry = 0; retry < 64; retry++) {
        game_seed += 0x9e3779b97f4a7c15ULL;
        const char *opp_rack = t6_baseline_sample_opp(state, game_seed);
        if (!opp_rack) break;
        if (swap_player_rack(gr->game, 1, opp_rack)) { opp_set = true; break; }
      }
      if (!opp_set) continue;
      ErrorStack *es = error_stack_create();
      ValidatedMoves *vms = validated_moves_create(
          gr->game, 0, task.action_repr, true, true, es);
      if (!error_stack_is_empty(es) ||
          validated_moves_get_number_of_moves(vms) == 0) {
        validated_moves_destroy(vms);
        error_stack_destroy(es);
        continue;
      }
      const Move *forced = validated_moves_get_move(vms, 0);
      // Capture per-task metadata once.
      if (!meta_captured) {
        const LetterDistribution *ld = game_get_ld(gr->game);
        const uint16_t ld_size = ld_get_size(ld);
        Rack tmp_leave;
        rack_set_dist_size(&tmp_leave, ld_size);
        rack_reset(&tmp_leave);
        get_leave_for_move(forced, gr->game, &tmp_leave);
        eb_render_leave(&tmp_leave, ld, leave_str, sizeof(leave_str));
        // Count blanks played: tiles where get_is_blanked is true (=
        // blank-as-letter), plus exact BLANK_MACHINE_LETTER tiles.
        const int n_tiles = move_get_tiles_length(forced);
        int b = 0;
        for (int t = 0; t < n_tiles; t++) {
          MachineLetter ml = move_get_tile(forced, t);
          if (ml == PLAYED_THROUGH_MARKER) continue;
          if (ml == BLANK_MACHINE_LETTER || get_is_blanked(ml)) b++;
        }
        blanks_used = b;
        task_score = equity_to_int(move_get_score(forced));
        meta_captured = true;
      }
      gr->eb_forced_move = (Move *)forced;
      game_runner_play_move(worker, gr);
      gr->eb_forced_move = NULL;
      validated_moves_destroy(vms);
      error_stack_destroy(es);
      while (!game_runner_is_game_over(gr)) {
        game_runner_play_move(worker, gr);
      }
      const int s0 = equity_to_int(
          player_get_score(game_get_player(gr->game, 0)));
      const int s1 = equity_to_int(
          player_get_score(game_get_player(gr->game, 1)));
      target_score_sum += s0;
      opp_score_sum += s1;
      if (s0 > s1) n_W++;
      else if (s0 < s1) n_L++;
      else n_T++;
      n_games_run++;
    }
    t6_baseline_record_result(state, &task, leave_str, blanks_used,
                              task_score, n_games_run, n_W, n_L, n_T,
                              target_score_sum, opp_score_sum);
  }
}

// Position-pool worker (MAGPIE_POSITION_POOL): work-steal mid-game CGPs from
// the pool, load each, play the on-turn move (recorded via the trajectory
// recorder when MAGPIE_TRAJECTORY_RECORDER is also set), then play out POST
// with HastyBot (NOT recorded), and commit/discard the staged row by the
// game-end reason — mirroring the trajectory recorder's two-phase semantics.
// Pilot scope: natural (loaded) rack + best move, to validate the pipeline and
// the synthetic-vs-natural combinability check. Rack injection + counterfactual
// move enumeration are the next increment.
// Render the on-turn player's post-move leave to a canonical string (A..Z
// then '?'). Used to build (kind,score,leave) dedup keys for the fan-out.
static void pp_render_leave(const Move *move, const Game *game, char *out,
                           size_t cap) {
  const int on = game_get_player_on_turn_index(game);
  const Rack *prack = player_get_rack(game_get_player(game, on));
  Rack lv;
  rack_set_dist_size(&lv, rack_get_dist_size(prack));
  rack_reset(&lv);
  if (move_get_type(move) == GAME_EVENT_PASS) {
    rack_copy(&lv, prack);
  } else {
    get_leave_for_move(move, game, &lv);
  }
  const LetterDistribution *ld = game_get_ld(game);
  int n = 0;
  const uint16_t ds = rack_get_dist_size(&lv);
  for (uint16_t i = 1; i < ds && n < (int)cap - 1; i++) {
    const int c = rack_get_letter(&lv, i);
    for (int k = 0; k < c && n < (int)cap - 1; k++) {
      out[n++] = ld->ld_ml_to_hl[i][0];
    }
  }
  const int nb = rack_get_letter(&lv, 0);
  for (int k = 0; k < nb && n < (int)cap - 1; k++) out[n++] = '?';
  out[n] = '\0';
}

#define PP_FANOUT_MAX 512  // max distinct (kind,score,leave) branches per rack
#define PP_FT_MAX 32       // max force cells credited per fan-out branch

// Match a fan-out move's force-table cells: leave cells (stratum/tile/pair via
// the tile-bitmap predicate) + bag_tile cells (via the pre-move-rack predicate).
// Collects targets that still have deficit into out[]; returns the count. 0 =>
// no deficient cell matches, so the fan-out gate skips this branch. Mirrors the
// EB matcher (eb_append_force_target_slots, autoplay.c) for the position pool.
static int pp_match_force_cells(ForceTable *ft, Game *game, const Move *mv,
                                int on_idx, int eff_diff, bool is_exch,
                                ForceTarget **out, int cap) {
  const LetterDistribution *ld = game_get_ld(game);
  const Rack *pre_rack = player_get_rack(game_get_player(game, on_idx));
  Rack leave;
  rack_set_dist_size(&leave, rack_get_dist_size(pre_rack));
  get_leave_for_move(mv, game, &leave);
  uint32_t leave_bm = 0;
  for (uint16_t i = 0; i < leave.dist_size && i < 32; i++) {
    if (leave.array[i] > 0) leave_bm |= ((uint32_t)1) << i;
  }
  const int leave_len = (int)leave.number_of_letters;
  if (leave_len < 0 || leave_len >= 8) return 0;
  const int bag_count = bag_get_letters(game_get_bag(game)) + (RACK_SIZE);
  int count = 0;
  ForceTargetSlot *slots = force_table_lookup_slots_by_shape(
      ft, bag_count, leave_len, is_exch ? 1 : 0, &count);
  if (count == 0 || slots == NULL) return 0;
  uint32_t *bitmaps = force_table_lookup_bitmaps_by_shape(
      ft, bag_count, leave_len, is_exch ? 1 : 0);
  int ltype = -1;
  int n = 0;
  for (int t = 0; t < count && n < cap; t++) {
    ForceTargetSlot *fs = &slots[t];
    if (fs->deficit <= 0) continue;
    if (eff_diff < fs->diff_min || eff_diff > fs->diff_max) continue;
    const int k = (int)fs->kind;
    if (k == FORCE_TARGET_BAG_TILE) {
      if (ltype < 0) ltype = (int)force_classify_leave(&leave, ld);
      if (force_target_matches_bag(fs->cold, pre_rack, leave_len,
                                   (LeaveType)ltype, is_exch, eff_diff, ld)) {
        out[n++] = fs->cold;
      }
      continue;
    }
    if (k != FORCE_TARGET_PAIR && k != FORCE_TARGET_TILE &&
        k != FORCE_TARGET_STRATUM && k != FORCE_TARGET_LEAVE) {
      continue;
    }
    if (bitmaps != NULL) {
      const uint32_t req = bitmaps[t];
      if ((leave_bm & req) != req) continue;
    }
    if (k == FORCE_TARGET_LEAVE) {
      // Enumerated full-leave cell (L3/L4): the bitmap pre-filter only proves
      // the tiles are PRESENT; require an exact multiset match so "AAB" is not
      // credited by an "ABB" leave. The full subleave (up to FORCE_MAX_SUBLEAVE
      // tiles) lives on the cold struct — the hot slot only carries the <=2 MLs
      // the tile/pair same-tile check needs.
      if (!force_subleave_exact_match(fs->cold->subleave_mls,
                                      fs->cold->subleave_count, &leave)) {
        continue;
      }
    } else if (fs->subleave_count == 2 &&
               fs->subleave_mls[0] == fs->subleave_mls[1] &&
               leave.array[fs->subleave_mls[0]] < 2) {
      continue;
    }
    // Type check only for genuinely type-split cells (L5/L6 plays here). Gate
    // on the CELL's type, not the leave length: an "all" cell (L0-L4, L7, and
    // exchanges) is never type-checked, so L3/L4 are treated like L1/L2. Keying
    // on length wrongly type-checked L3/L4/L7 "all" cells against a classified
    // leave, which never matched -> those plays were silently never recorded.
    if (fs->leave_type != LEAVE_TYPE_ALL) {
      if (ltype < 0) ltype = (int)force_classify_leave(&leave, ld);
      if ((LeaveType)ltype != (LeaveType)fs->leave_type) continue;
    }
    out[n++] = fs->cold;
  }
  return n;
}

// Play `dg` out to game end and report the `mover`'s outcome. For the low-bag
// endgame phase (`endgame` true), when the bag empties solve the endgame
// (even-4, diff-gated) for the exact W/L instead of continuing HastyBot — the
// value-level shortcut (solve once, don't play the endgame out). Returns true
// if the game reached a committable end; sets *is_win/*is_tie (mover's view)
// and *f0/*f1 (final scores, sign-correct so the recorder's per-row outcome
// matches the label).
static bool pp_playout_outcome(EndgameSolver *es, EndgameResults *er,
                               ThreadControl *tc, Game *dg, MoveList *post_ml,
                               int worker_index, int mover, bool endgame,
                               int eg_plies, const double *eg_caps,
                               int eg_signstable_k, int eg_diffgate,
                               double *eg_spent, double eg_pos_budget,
                               double eg_overflow_cap, bool *is_win,
                               bool *is_tie, int *f0, int *f1) {
  while (game_get_game_end_reason(dg) == GAME_END_REASON_NONE) {
    if (endgame && bag_get_letters(game_get_bag(dg)) == 0) {
      const int on = game_get_player_on_turn_index(dg);
      const int on_s = equity_to_int(player_get_score(game_get_player(dg, on)));
      const int op_s =
          equity_to_int(player_get_score(game_get_player(dg, 1 - on)));
      const int cur_diff = on_s - op_s;  // on-turn perspective
      int opt = 0;
      if (abs(cur_diff) < eg_diffgate) {  // diff-gate: blowouts skip the solve
        // Per-length wall cap: key on the opponent rack size (1..7) — endgame
        // cost scales with the unseen tiles the search must enumerate.
        int opp_len =
            rack_get_total_letters(player_get_rack(game_get_player(dg, 1 - on)));
        if (opp_len < 1) opp_len = 1;
        if (opp_len > 7) opp_len = 7;
        double cap = eg_caps[opp_len];
        // Over the per-position budget -> tighten to the overflow cap so a
        // heavy position can't tail the worker (still solver-derived sign).
        if (eg_pos_budget > 0 && eg_spent && *eg_spent >= eg_pos_budget &&
            eg_overflow_cap > 0 && eg_overflow_cap < cap) {
          cap = eg_overflow_cap;
        }
        const EndgameArgs ea = {
            .game = dg, .thread_control = tc, .plies = eg_plies,
            .tt_fraction_of_mem = 0.005,
            .initial_small_move_arena_size =
                DEFAULT_INITIAL_SMALL_MOVE_ARENA_SIZE,
            .num_threads = 1, .base_thread_index = worker_index,
            .num_top_moves = 1, .use_heuristics = true,
            .per_ply_callback = NULL, .per_ply_callback_data = NULL,
            .forced_pass_bypass = false, .soft_time_limit = cap,
            .hard_time_limit = cap, .sign_stable_k = eg_signstable_k};
        ErrorStack *err = error_stack_create();
        Timer eg_timer;
        ctimer_start(&eg_timer);
        endgame_solve(es, &ea, er, err);
        if (eg_spent) *eg_spent += ctimer_elapsed_seconds(&eg_timer);
        if (error_stack_is_empty(err)) {
          const PVLine *pv = endgame_results_get_pvline(er, ENDGAME_RESULT_BEST);
          if (pv) opt = pv->score;
        }
        error_stack_destroy(err);
      }
      const int final_on = cur_diff + opt;  // on-turn final spread
      // Sign-correct synthetic finals (opt attributed to on-turn; recorder
      // only uses the sign of f_mover - f_opp for the per-row outcome).
      const int fon = on_s + opt, fop = op_s;
      *f0 = (on == 0) ? fon : fop;
      *f1 = (on == 1) ? fon : fop;
      const bool on_win = final_on > 0, on_tie = final_on == 0;
      *is_tie = on_tie;
      *is_win = (on == mover) ? on_win : (!on_win && !on_tie);
      return true;
    }
    const Move *pm = get_top_equity_move(dg, worker_index, post_ml);
    play_move(pm, dg, NULL);
  }
  if (game_get_game_end_reason(dg) != GAME_END_REASON_STANDARD) return false;
  const int mf = equity_to_int(player_get_score(game_get_player(dg, mover)));
  const int of = equity_to_int(player_get_score(game_get_player(dg, 1 - mover)));
  *f0 = equity_to_int(player_get_score(game_get_player(dg, 0)));
  *f1 = equity_to_int(player_get_score(game_get_player(dg, 1)));
  *is_tie = (mf == of);
  *is_win = (mf > of);
  return true;
}

// Replay an opener on the (freshly reset, seeded) empty board: set the starting
// player's rack, movegen, find the opener move matching (score, leave), play it
// — synthesizing a (bag=93-tiles_played, diff=score) board the natural pool
// never produces (blank-burns, rare scores). sign '+' then passes the opponent
// so the opener is on turn (+diff); '-' leaves the responder on turn (-diff).
// Returns false if the rack can't be drawn or no matching opener exists (caller
// skips the entry). After return the on-turn player is the injection target.
static bool pp_setup_opener(AutoplayWorker *worker, GameRunner *gr,
                            MoveList *fan_ml, MoveList *post_ml,
                            const char *rack, int score, const char *leave,
                            char sign, int variant) {
  if (!swap_player_rack(gr->game, 0, rack)) return false;
  const MoveGenArgs mga = {
      .game = gr->game,
      .move_list = fan_ml,
      .move_record_type = MOVE_RECORD_ALL,
      .move_sort_type = MOVE_SORT_EQUITY,
      .override_kwg = NULL,
      .thread_index = worker->worker_index,
      .eq_margin_movegen = 0,
      .target_equity = EQUITY_MAX_VALUE,
      .target_leave_size_for_exchange_cutoff = UNSET_LEAVE_SIZE,
      .tiles_played_bv = NULL,
      .initial_tiles_bv = 0};
  generate_moves(&mga);
  const int nm = move_list_get_count(fan_ml);
  const Move *match = NULL;
  char lf[RACK_SIZE + 2];
  for (int m = 0; m < nm; m++) {
    const Move *mv = move_list_get_move(fan_ml, m);
    if (move_get_type(mv) != GAME_EVENT_TILE_PLACEMENT_MOVE) continue;
    if (equity_to_int(move_get_score(mv)) != score) continue;
    pp_render_leave(mv, gr->game, lf, sizeof(lf));  // on-turn = P1 (opener rack)
    if (strcmp(lf, leave) == 0) {
      match = mv;
      break;
    }
  }
  if (!match) return false;
  play_move(match, gr->game, NULL);
  (void)fan_ml;
  if (sign == '+') {
    // Put the opener back on turn at +diff by FORCING the opponent to exchange
    // (the realistic passive turn-2 move: fresh rack, board unchanged so bag
    // stays 93-k). A PASS would waste the opponent's whole turn -> tempo
    // artifact inflating win% to ~100%. We construct the exchange explicitly
    // (movegen equity-prunes exchanges for a playable rack), and VARY it across
    // all types (1..ns tiles, rotating which) by `variant` so the +diff
    // positions span how an opponent-exchange actually arises rather than all
    // being "opponent dumped everything".
    const int opp = game_get_player_on_turn_index(gr->game);  // responder
    // The responder's rack is empty here (we bypass game_runner_start, so only
    // the opener's rack was drawn) — fill it from the bag before exchanging.
    draw_to_full_rack(gr->game, opp);
    const Rack *prack = player_get_rack(game_get_player(gr->game, opp));
    MachineLetter tiles[RACK_SIZE];
    int ns = 0;
    const uint16_t ds = rack_get_dist_size(prack);
    for (uint16_t L = 0; L < ds && ns < RACK_SIZE; L++) {
      const int c = rack_get_letter(prack, L);
      for (int k = 0; k < c && ns < RACK_SIZE; k++) {
        tiles[ns++] = (MachineLetter)L;
      }
    }
    if (ns == 0) return false;  // empty rack (shouldn't happen) -> skip
    const int k_ex = 1 + (variant % ns);          // exchange 1..ns tiles
    const int off = (variant / ns) % ns;          // rotate which tiles
    MachineLetter strip[RACK_SIZE];
    for (int j = 0; j < k_ex; j++) strip[j] = tiles[(off + j) % ns];
    Move *ex = move_list_get_spare_move(post_ml);
    move_set_all(ex, strip, 0, k_ex - 1, /*score=*/0, /*row=*/0, /*col=*/0,
                 /*tiles_played=*/k_ex, /*dir=*/0, GAME_EVENT_EXCHANGE,
                 /*leave_value=*/0);
    play_move(ex, gr->game, NULL);
  }
  return true;
}

static void position_pool_run_worker(AutoplayWorker *worker, GameRunner *gr) {
  AutoplaySharedData *sd = worker->shared_data;
  PositionPool *pp = sd->position_pool;
  ThreadControl *tc = worker->args.thread_control;
  TrajectoryRecorder *traj_r = sd->trajectory_recorder;
  OpenerPool *op = sd->opener_pool;
  const int n = op ? opener_pool_count(op) : position_pool_count(pp);
  MoveList *post_ml = move_list_create(1);
  MoveList *fan_ml = move_list_create(8192);  // full move list for fan-out
  // MAGPIE_PP_FANOUT=1: branch a playout per distinct (kind,score,leave) cell of
  // the injected rack's move list (counterfactual coverage), instead of only
  // the best move. Composes with REROLLS (each re-rolled rack is fanned out).
  const char *fanout_env = getenv("MAGPIE_PP_FANOUT");
  const bool fanout = fanout_env && fanout_env[0] == '1';
  // MAGPIE_PP_REROLLS=K: if K>0, inject K randomly re-rolled on-turn racks per
  // position (rack injection for coverage), each a separate playout. K=0 (or
  // unset) = baseline: one playout with the loaded natural rack. The re-roll
  // returns the on-rack to the bag and redraws from it (own+bag pool); opp is
  // left natural (opp-swap for low-bag diversity is a later increment).
  const char *rerolls_env = getenv("MAGPIE_PP_REROLLS");
  const int rerolls = rerolls_env ? atoi(rerolls_env) : 0;
  const int games_per_pos = rerolls > 0 ? rerolls : 1;
  // Opp-swap threshold (unseen tiles): at/below this bag the own+bag pool is
  // too small to draw an injected rack, so the opponent's rack is also
  // returned to the bag (widening the pool) and redrawn afterward. Default 14.
  const char *oppswap_env = getenv("MAGPIE_PP_OPPSWAP_MAXBAG");
  const int oppswap_maxbag = oppswap_env ? atoi(oppswap_env) : 14;
  // MAGPIE_PP_ENDGAME=1: for the low-bag phase (8-14), when a playout empties
  // the bag, solve the endgame (even-4, diff-gated) for exact W/L instead of
  // the HastyBot playout. One solver/results per worker (TT memory-bound;
  // single-thread per solve).
  const char *endgame_env = getenv("MAGPIE_PP_ENDGAME");
  const bool endgame = endgame_env && endgame_env[0] == '1';
  // Endgame POST tuning. Default strategy: deep iterative deepening stopped by
  // sign-stability (K consecutive same-sign depths) — the W/L/T label settles
  // far earlier than the value converges (~0.3s mean vs ~6s). A PER-OPP-LENGTH
  // wall cap backstops the rare oscillating tail. Caps indexed by opponent rack
  // size 1..7 (eg_caps[len]); eg_caps[0] mirrors [1] as a fallback.
  const char *eg_plies_env = getenv("MAGPIE_PP_EG_PLIES");
  const char *eg_ssk_env = getenv("MAGPIE_PP_EG_SIGNSTABLE_K");
  const char *eg_caps_env = getenv("MAGPIE_PP_EG_HARD_BY_LEN");
  const char *eg_gate_env = getenv("MAGPIE_PP_EG_DIFFGATE");
  const char *eg_budget_env = getenv("MAGPIE_PP_EG_POS_BUDGET");
  const char *eg_ovcap_env = getenv("MAGPIE_PP_EG_OVERFLOW_CAP");
  const int eg_plies = eg_plies_env ? atoi(eg_plies_env) : 25;
  const int eg_signstable_k = eg_ssk_env ? atoi(eg_ssk_env) : 2;
  const int eg_diffgate = eg_gate_env ? atoi(eg_gate_env) : 80;
  // Per-position endgame budget: once a position's fan-out branches have spent
  // this many wall-seconds solving, the remaining branches use a tight cap
  // (still the solver, shallower) so one heavy position can't tail a worker.
  // 0 = no budget. Overflow branches keep solver-derived signs (never HastyBot).
  const double eg_pos_budget = eg_budget_env ? atof(eg_budget_env) : 25.0;
  const double eg_overflow_cap = eg_ovcap_env ? atof(eg_ovcap_env) : 1.0;
  // MAGPIE_PP_EG_MAX_SOLVES=K: cap the number of fanned-out playout/solve
  // branches per POSITION (across rerolls). Once K branches have been played
  // out at a position, skip the rest and move to the next position. Bounds the
  // per-position cost — a from-scratch force table matches every branch, so
  // without this one dense board soaks up hundreds of endgame solves (~9 min of
  // one thread). Coverage is preserved by processing more positions: a
  // deficient cell recurs across thousands of boards, so you don't need to
  // drain it at one board. 0 = unlimited (default; e.g. the no-endgame
  // enumerated fill, where each fanout branch is a cheap hasty game worth
  // keeping).
  const char *eg_maxsolves_env = getenv("MAGPIE_PP_EG_MAX_SOLVES");
  const int eg_max_solves = eg_maxsolves_env ? atoi(eg_maxsolves_env) : 0;
  double eg_caps[8] = {0.5, 0.5, 1.0, 1.5, 3.0, 5.0, 8.0, 12.0};
  if (eg_caps_env) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%s", eg_caps_env);
    int li = 1;
    for (char *tok = strtok(buf, ","); tok && li <= 7;
         tok = strtok(NULL, ","), li++) {
      eg_caps[li] = atof(tok);
    }
    eg_caps[0] = eg_caps[1];
  }
  if (endgame) {
    fprintf(stderr,
            "position_pool: endgame POST on (plies=%d sign_stable_k=%d "
            "diff_gate=%d caps[1..7]=%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f "
            "pos_budget=%.1fs overflow_cap=%.2fs max_solves/pos=%d)\n",
            eg_plies, eg_signstable_k, eg_diffgate, eg_caps[1], eg_caps[2],
            eg_caps[3], eg_caps[4], eg_caps[5], eg_caps[6], eg_caps[7],
            eg_pos_budget, eg_overflow_cap, eg_max_solves);
  }
  EndgameSolver *es = endgame ? endgame_solver_create() : NULL;
  EndgameResults *er = endgame ? endgame_results_create() : NULL;
  // Per-worker ThreadControl for the endgame solves: worker->args.thread_control
  // is SHARED across all autoplay workers, so concurrent endgame_solve on it
  // corrupts/crashes. Each worker gets its own (num_threads=1 per solve).
  ThreadControl *eg_tc = endgame ? thread_control_create() : NULL;
  // Force-table gate: when loaded (MAGPIE_FORCE_TABLE), the fan-out fills
  // per-(stratum,bucket,feature) cells to target instead of the per-leaf
  // leave_deficit. ft==NULL falls back to leave_deficit.
  ForceTable *ft = sd->force_table;
  for (;;) {
    if (thread_control_get_status(tc) == THREAD_CONTROL_STATUS_USER_INTERRUPT) {
      break;
    }
    const int i = atomic_fetch_add_explicit(&sd->position_pool_next, 1,
                                             memory_order_relaxed);
    if (i >= n) break;
    const char *cgp = NULL;
    uint64_t pos_id = (uint64_t)i;
    if (!op) {
      cgp = position_pool_get_cgp(pp, i);
      if (!cgp) continue;
      pos_id = position_pool_get_id(pp, i);
    }

    // Per-position playout/solve budget (MAGPIE_PP_EG_MAX_SOLVES). Accumulates
    // across this position's rerolls; when it trips, break to the next position.
    int pos_playouts = 0;
    bool pos_capped = false;
    for (int g = 0; g < games_per_pos; g++) {
      game_reset(gr->game);
      // Reset GameRunner fields normally cleared by game_runner_start, which we
      // bypass (mirrors t6_baseline_run_worker).
      gr->turn_number = 0;
      gr->force_draw = false;
      gr->force_triggered = false;
      gr->pending_force_target = NULL;
      gr->pending_force_diff = 0;
      gr->pending_force_player_index = 0;
      gr->pass_cycle_active = false;
      gr->pass_cycle_abandoned = false;
      gr->pass_cycle_n_moves = 0;
      gr->eb_active = false;
      gr->eb_forced_move = NULL;
      gr->eb_n_snaps = 0;
      gr->eb_actions_p0[0] = '\0';
      gr->eb_actions_p0_off = 0;
      gr->eb_actions_p1[0] = '\0';
      gr->eb_actions_p1_off = 0;

      // Seed per (worker, position, reroll) so re-rolls + POST (and the opener
      // setup's draws) diverge. Seeded before setup so the opener replay's
      // refill draws are deterministic per entry.
      game_seed(gr->game,
                (uint64_t)worker->worker_index * 0x9e3779b97f4a7c15ULL +
                    (uint64_t)i * 1000003ULL + (uint64_t)g);
      if (op) {
        // Opener-conditioned: replay the opener to synthesize the board.
        if (!pp_setup_opener(worker, gr, fan_ml, post_ml,
                             opener_pool_get_rack(op, i),
                             opener_pool_get_score(op, i),
                             opener_pool_get_leave(op, i),
                             opener_pool_get_sign(op, i), i + g)) {
          break;  // rack undrawable or no matching opener: skip entry
        }
      } else {
        ErrorStack *es = error_stack_create();
        game_load_cgp(gr->game, cgp, es);
        if (!error_stack_is_empty(es)) {
          error_stack_destroy(es);
          break;  // unparseable CGP: skip this position entirely
        }
        error_stack_destroy(es);
      }
      gr->game_number = pos_id;

      const int on_idx = game_get_player_on_turn_index(gr->game);
      const int opp_idx = 1 - on_idx;
      // Rack injection. With a targeted rack pool, inject a sampled pool rack
      // via swap_player_rack (retry on undrawable); else if rerolls>0, random
      // re-roll. At low bag the own+bag pool is too small to draw an arbitrary
      // rack, so opp-swap: return the opponent's rack to the bag (widening the
      // pool), inject, then redraw the opponent. Bag-gated so high-bag keeps
      // the opponent natural (no synthetic-opp bias).
      const bool do_inject = (sd->rack_pool != NULL) || (rerolls > 0);
      const int inj_unseen =
          bag_get_letters(game_get_bag(gr->game)) +
          rack_get_total_letters(
              player_get_rack(game_get_player(gr->game, opp_idx)));
      const bool oppswap = do_inject && inj_unseen <= oppswap_maxbag;
      if (oppswap) return_rack_to_bag(gr->game, opp_idx);
      if (sd->rack_pool != NULL) {
        const int rpn = position_pool_count(sd->rack_pool);
        uint64_t rseed = (uint64_t)worker->worker_index * 0x9e3779b97f4a7c15ULL +
                         (uint64_t)i * 1000003ULL + (uint64_t)g;
        for (int retry = 0; retry < 32 && rpn > 0; retry++) {
          rseed = rseed * 6364136223846793005ULL + 1442695040888963407ULL;
          const char *rk =
              position_pool_get_cgp(sd->rack_pool, (int)(rseed % (uint64_t)rpn));
          if (rk && swap_player_rack(gr->game, on_idx, rk)) {
            break;  // injected; on all-retry failure keep the loaded rack
          }
        }
      } else if (rerolls > 0) {
        return_rack_to_bag(gr->game, on_idx);
        draw_to_full_rack(gr->game, on_idx);
      }
      if (oppswap) draw_to_full_rack(gr->game, opp_idx);  // redraw the opponent

      if (fanout) {
        // Branch a playout per distinct (kind,score,leave) cell of the rack's
        // move list. Movegen once on gr->game; each branch clones the position
        // (game_duplicate), plays the move, and POSTs to end on the clone, so
        // gr->game stays intact for the next branch.
        const MoveGenArgs fga = {
            .game = gr->game,
            .move_list = fan_ml,
            .move_record_type = MOVE_RECORD_ALL,
            .move_sort_type = MOVE_SORT_EQUITY,
            .override_kwg = NULL,
            .thread_index = worker->worker_index,
            .eq_margin_movegen = 0,
            .target_equity = EQUITY_MAX_VALUE,
            .target_leave_size_for_exchange_cutoff = UNSET_LEAVE_SIZE,
            .tiles_played_bv = NULL,
            .initial_tiles_bv = 0};
        generate_moves(&fga);
        const int nm = move_list_get_count(fan_ml);
        // Position-level values for the deficit gate (same for all branches):
        // unseen bag and pre-move score diff (move_score added per branch).
        const int g_opp = 1 - on_idx;
        const int gate_bag =
            bag_get_letters(game_get_bag(gr->game)) +
            rack_get_total_letters(player_get_rack(game_get_player(gr->game,
                                                                   g_opp)));
        const int pre_diff =
            equity_to_int(player_get_score(game_get_player(gr->game, on_idx))) -
            equity_to_int(player_get_score(game_get_player(gr->game, g_opp)));
        static _Thread_local char seen[PP_FANOUT_MAX][24];
        int nseen = 0;
        // Per-position endgame budget accumulator (wall-seconds spent solving
        // this position's branches); tightens the cap once exceeded.
        double eg_spent = 0.0;
        for (int m = 0; m < nm && nseen < PP_FANOUT_MAX; m++) {
          const Move *mv = move_list_get_move(fan_ml, m);
          char lf[RACK_SIZE + 2];
          pp_render_leave(mv, gr->game, lf, sizeof(lf));
          char key[24];
          snprintf(key, sizeof(key), "%d_%d_%s", (int)move_get_type(mv),
                   (int)equity_to_int(move_get_score(mv)), lf);
          bool dup_key = false;
          for (int s = 0; s < nseen; s++) {
            if (strcmp(seen[s], key) == 0) {
              dup_key = true;
              break;
            }
          }
          if (dup_key) continue;
          snprintf(seen[nseen++], sizeof(seen[0]), "%s", key);
          const int msc =
              move_get_type(mv) == GAME_EVENT_TILE_PLACEMENT_MOVE
                  ? equity_to_int(move_get_score(mv))
                  : 0;
          const int eff_diff = pre_diff + msc;
          const bool is_exch_mv = move_get_type(mv) == GAME_EVENT_EXCHANGE;
          // Gate: force-table cells (T2-T6 spec) when loaded, else the per-leaf
          // leave_deficit. Skip the branch if no deficient cell would be
          // credited (no point spending a playout on a satisfied cell).
          ForceTarget *ftgts[PP_FT_MAX];
          int nft = 0;
          if (ft != NULL) {
            nft = pp_match_force_cells(ft, gr->game, mv, on_idx, eff_diff,
                                       is_exch_mv, ftgts, PP_FT_MAX);
            if (nft == 0) continue;
          } else if (!leave_deficit_take(sd->leave_deficit, gate_bag, lf,
                                         eff_diff)) {
            continue;
          }
          Game *dg = game_duplicate(gr->game);
          if (traj_r && gr->trajectory_buf) {
            trajectory_game_buffer_reset(gr->trajectory_buf);
          }
          eb_stage_trajectory_row(gr, gr->game, mv, on_idx);
          play_move(mv, dg, NULL);
          bool win = false, tie = false;
          int f0 = 0, f1 = 0;
          const bool committable = pp_playout_outcome(
              es, er, eg_tc, dg, post_ml, worker->worker_index, on_idx, endgame,
              eg_plies, eg_caps, eg_signstable_k, eg_diffgate, &eg_spent,
              eg_pos_budget, eg_overflow_cap, &win, &tie, &f0, &f1);
          if (committable) {
            if (ft != NULL) {
              for (int kk = 0; kk < nft; kk++) {
                force_table_credit_game(ft, ftgts[kk], eff_diff, win, tie);
              }
            }
            if (traj_r && gr->trajectory_buf) {
              trajectory_game_buffer_commit(traj_r, gr->trajectory_buf, f0, f1);
            }
          } else if (traj_r && gr->trajectory_buf) {
            trajectory_game_buffer_discard(gr->trajectory_buf);
          }
          game_destroy(dg);
          if (eg_max_solves > 0 && ++pos_playouts >= eg_max_solves) {
            pos_capped = true;
            break;  // enough branches at this position — move on
          }
        }
      } else {
        if (traj_r && gr->trajectory_buf) {
          trajectory_game_buffer_reset(gr->trajectory_buf);
        }
        // Select the on-turn move via get_top_equity_move (HastyBot) directly —
        // NOT game_runner_play_move, whose autoplay path (sim / EB role forcing)
        // misbehaves on a loaded mid-game board. Stage the row, then play it.
        const Move *target =
            get_top_equity_move(gr->game, worker->worker_index, post_ml);
        eb_stage_trajectory_row(gr, gr->game, target, on_idx);
        play_move(target, gr->game, NULL);
        // POST: play out to game end (endgame solve at bag-empty if enabled).
        bool win = false, tie = false;
        int f0 = 0, f1 = 0;
        const bool committable = pp_playout_outcome(
            es, er, eg_tc, gr->game, post_ml, worker->worker_index, on_idx,
            endgame, eg_plies, eg_caps, eg_signstable_k, eg_diffgate, NULL, 0.0,
            0.0, &win, &tie, &f0, &f1);
        (void)win;
        (void)tie;
        if (traj_r && gr->trajectory_buf) {
          if (committable) {
            trajectory_game_buffer_commit(traj_r, gr->trajectory_buf, f0, f1);
          } else {
            trajectory_game_buffer_discard(gr->trajectory_buf);
          }
        }
      }
      if (pos_capped) break;
    }
  }
  if (es) endgame_solver_destroy(es);
  if (er) endgame_results_destroy(er);
  if (eg_tc) thread_control_destroy(eg_tc);
  move_list_destroy(post_ml);
  move_list_destroy(fan_ml);
}

void *autoplay_worker(void *uncasted_autoplay_worker) {
  AutoplayWorker *autoplay_worker = (AutoplayWorker *)uncasted_autoplay_worker;
  const AutoplayArgs *args = &autoplay_worker->args;
  GameRunner *game_runner1 = game_runner_create(autoplay_worker);
  init_sim_args_for_player(autoplay_worker, 0);
  init_sim_args_for_player(autoplay_worker, 1);

  // Phase 2 baseline mode: run task-driven loop instead of normal
  // autoplay. Returns when task file exhausted.
  if (autoplay_worker->shared_data->t6_baseline != NULL) {
    t6_baseline_run_worker(autoplay_worker, game_runner1);
    game_runner_destroy(game_runner1);
    return NULL;
  }

  // Position-pool mode (91-8 rack injection): load mid-game CGPs, or replay
  // openers (MAGPIE_OPENER_POOL), instead of fresh games.
  if (autoplay_worker->shared_data->position_pool != NULL ||
      autoplay_worker->shared_data->opener_pool != NULL) {
    position_pool_run_worker(autoplay_worker, game_runner1);
    game_runner_destroy(game_runner1);
    return NULL;
  }

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
