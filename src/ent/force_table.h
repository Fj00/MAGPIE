#ifndef FORCE_TABLE_H
#define FORCE_TABLE_H

#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>

#include "../def/board_defs.h"
#include "../def/letter_distribution_defs.h"
#include "letter_distribution.h"
#include "rack.h"

typedef enum {
  FORCE_TARGET_STRATUM = 0,
  FORCE_TARGET_TILE = 1,
  FORCE_TARGET_PAIR = 2,
  // BAG_TILE: cell credits when the player's pre-move RACK lacks at least
  // one of the target's tile (rack.count(tile) < TILE_BAG[tile]). Drives
  // the V model's `bag_exp_<tile>` feature variation in skewed buckets.
  // CSV `subleave` field uses "<TILE>_free" format (e.g. "Z_free").
  // Count-based decrement at game-end (same path as TILE/PAIR).
  FORCE_TARGET_BAG_TILE = 3,
} ForceTargetKind;

// Per-game cap on simultaneous BAG_TILE matches. A move's rack composition
// can satisfy at most 27 cells (one per tile in the bag), so 27 is the
// hard ceiling. GameRunner stores up to this many pending bag targets.
#define MAX_PENDING_BAG_TARGETS 27

typedef enum {
  LEAVE_TYPE_ALL = 0,
  LEAVE_TYPE_CONS = 1,
  LEAVE_TYPE_MIXED = 2,
  LEAVE_TYPE_VOWEL = 3,
} LeaveType;

typedef struct ForceTarget {
  // Hot fields — placed first so a single cache-line load (64 bytes) on the
  // target struct brings all the per-iteration check fields. The matching
  // loop reads deficit, kind, length, exchange, type, diff range, subleave
  // before deciding whether to do a deeper check.
  //
  // deficit is _Atomic so concurrent decrements from many worker threads
  // don't lose updates via plain RMW races. force_table_decrement_target
  // uses a CAS loop; hot-path filter reads use atomic_load_explicit with
  // memory_order_relaxed to avoid the seq_cst fences a plain `_Atomic`
  // read would compile to on x86. (Multi-credit per game makes the same
  // ForceTarget the decrement target for many concurrent games, so plain
  // `deficit--` lost ~half the decrements before this was fixed —
  // see noble-popping-kitten.md / commit log.)
  _Atomic int64_t deficit;
  ForceTargetKind kind;
  int leave_length;
  int exchange;
  LeaveType leave_type;
  int diff_min;
  int diff_max;
  int subleave_count;
  MachineLetter subleave_mls[2];
  // Cold fields — only touched on a candidate match or during table
  // maintenance.
  int bag;
  // Natural-draw probability of any rack producing this cell (sum_p_rack
  // from cell_rarity.csv, summed over the bucket's diff range). Lower =
  // rarer. Used as the secondary sort key in shape-bucket allocator so
  // rare cells get first dibs on the per-game slot budget. Defaults to a
  // very large number when the column is absent (treated as "common").
  double rarity_score;
  // B5: per-bucket index pointers — used to swap-to-back when deficit hits 0.
  // Lookups return only the active prefix of each bucket (entries with
  // deficit > 0 are kept at indices 0..active-1; satisfied entries get pushed
  // to the back).
  int by_bag_idx;
  int by_shape_idx;
  // For STRATUM cells: per-side target T = CSV target value. Cell drains
  // when total_wins >= T AND total_losses >= T. Each credit increments
  // W or L (or both for tie). If credited side was < T before, deficit
  // decrements by 1. Initial in-memory deficit = 2T. Cell drained when
  // deficit hits 0 (= both sides reached T). For non-STRATUM kinds this
  // field is unused.
  int32_t stratum_per_side;
  // Diagnostic: count of force_table_credit_game calls that landed on
  // this cell. Lets us distinguish orphan-by-no-match (counter==0) from
  // orphan-by-credit-failure (counter>0 but deficit unchanged).
  _Atomic uint64_t credit_attempts;
  // Original deficit at load time. Used by the per-cell priority scorer
  // as `base = remaining_deficit / original_deficit` (1.0 untouched,
  // 0 drained).
  int64_t original_deficit;
} ForceTarget;

// Compact per-target hot data co-located with bucket pointer for cache-
// friendly per-iteration matching. Stored inline in shape buckets so the
// inner match loop reads contiguous memory instead of pointer-chasing into
// scattered ForceTarget structs. Same fields the matching loop checks; on a
// candidate match the worker dereferences the cold ForceTarget* for the rest.
typedef struct {
  int32_t deficit;             // sufficient — per-cell deficits fit in int32
  uint8_t kind;                // ForceTargetKind cast to uint8
  uint8_t leave_length;
  uint8_t exchange;
  uint8_t leave_type;
  int32_t diff_min;
  int32_t diff_max;
  MachineLetter subleave_mls[2];
  uint8_t subleave_count;
  uint8_t _pad;
  struct ForceTarget *cold;    // pointer to the full struct (for match)
} ForceTargetSlot;             // 32 bytes, fits 2 per cache line

typedef struct ForceTable ForceTable;

ForceTable *force_table_create(const char *csv_path,
                               const LetterDistribution *ld);

void force_table_destroy(ForceTable *table);

// Returns the array of targets at the given bag count, and sets *count.
// Returned pointer is owned by the table and valid until force_table_destroy.
// NULL is returned and *count set to 0 if no targets exist for the bag.
ForceTarget **force_table_lookup(ForceTable *table, int bag, int *count);

// A2: lookup targets at a specific (bag, leave_length, exchange) shape. Lets
// per-move matching skip iteration over targets whose shape can't match the
// move's leave. Same target pointers as force_table_lookup, just bucketed.
ForceTarget **force_table_lookup_by_shape(ForceTable *table, int bag,
                                          int leave_length, int exchange,
                                          int *count);

// Hot-loop friendly lookup: returns a slot array (inline hot fields + cold
// pointer) for the specified shape bucket. Same active prefix size as the
// pointer version. Slots provide cache-friendly iteration during matching;
// dereference slot->cold only on candidate match.
ForceTargetSlot *force_table_lookup_slots_by_shape(ForceTable *table, int bag,
                                                    int leave_length,
                                                    int exchange, int *count);

// Returns the parallel required-tile-bitmap array for a shape bucket. Workers
// AND each entry with their leave's tile bitmap — if the result != entry,
// the target's required tiles aren't all in the leave and the slot can be
// skipped without further checks. Compact 4 bytes per slot, fits in L1 cache
// for fast pre-filter.
uint32_t *force_table_lookup_bitmaps_by_shape(ForceTable *table, int bag,
                                              int leave_length, int exchange);

// Check whether a candidate move's leave + score + current score-diff matches
// the given target. `leave` is the post-move kept-tile rack. `score` is the
// move's score. `diff` is the current score difference (player_on_turn minus
// opp) at the time of the candidate move; checked against target's
// [diff_min, diff_max]. Blanks are treated per the aggregation spec (ignored
// for type classification, but counted as a distinct tile '?' for subleave
// matching).
bool force_target_matches(const ForceTarget *target, const Rack *leave,
                          int score, int diff);

// BAG_TILE matcher: checks whether the player's pre-move RACK satisfies
// the target's `_free` predicate (rack lacks at least one of the target's
// tile, so unseen_tile > 0). Same (leave_length, exchange, type, diff)
// gates as `force_target_matches`; uses the leave's classification (passed
// by caller via the leave's length and the LeaveType pre-classified).
// `ld` is needed to look up TILE_BAG[tile] for the count check.
bool force_target_matches_bag(const ForceTarget *target,
                              const Rack *pre_move_rack,
                              int leave_length, LeaveType leave_type,
                              int score, int diff,
                              const LetterDistribution *ld);

// Decrement a target's deficit. If the deficit transitions from 1 to 0,
// the table's active-target counter is decremented. Returns true if the
// target is now satisfied.
bool force_table_decrement_target(ForceTable *table, ForceTarget *target);

// Credit a finished game's outcome against a target hit during the game.
// For stratum-kind targets: looks up the per-(target, diff) tally and
// decrements the target's deficit only when the outcome would increase
// min(wins, losses) for that diff bucket. The tally is always updated.
// Race-free across threads via atomic fetch-add of a packed (wins:32,
// losses:32) bucket — each game's pre-increment value uniquely determines
// whether IT was the one that bumped min_wl.
//
// For tile/pair-kind targets: decrements deficit by 1 (their semantics
// are count-based, not min(w,l)-based — original behavior preserved).
//
// `diff` is the score difference at the force turn from the forcing
// player's perspective (same convention as the FJ recorder's score_diff).
// `is_win`/`is_tie` describe the game's final outcome for that player;
// `is_tie` takes precedence over `is_win`.
void force_table_credit_game(ForceTable *table, ForceTarget *target,
                             int diff, bool is_win, bool is_tie);

// True once every target has been satisfied (all deficits are zero). Safe
// to call from multiple threads; the underlying counter is atomic.
bool force_table_is_exhausted(const ForceTable *table);

// Total remaining deficit across all targets (for progress reporting).
int64_t force_table_total_remaining(const ForceTable *table);

// Diagnostic counters for the multi-credit regression investigation. Tracks
// how many credit calls land as decrements vs no-ops, and CAS-loop retries
// under contention. Reset and snapshot at progress ticks.
typedef struct ForceCounters {
  uint64_t credit_calls;        // force_table_credit_game entry count
  uint64_t decrements_landed;   // CAS-decrement that actually fired
  uint64_t noops_already_zero;  // decrement saw deficit==0 on entry
  uint64_t stratum_no_bump;     // stratum credit didn't bump min(w,l)
  uint64_t cas_retries;         // CAS-loop retry count
} ForceCounters;

void force_table_get_counters(ForceCounters *out);
void force_table_reset_counters(void);

// Per-leave-length STRATUM credit/bump counters. Indexed 0..5.
void force_table_get_stratum_by_len(uint64_t credits[6], uint64_t bumps[6]);
void force_table_reset_stratum_by_len(void);

// Dump the remaining deficits to a CSV with the same schema as the input,
// skipping targets whose deficit is 0. Used for progress reporting and to
// produce an updated force_targets.csv for subsequent runs.
void force_table_dump_remaining(const ForceTable *table, const char *csv_path,
                                const LetterDistribution *ld);

// Test-and-clear the SIGUSR1 dump-request flag. Returns true if a dump was
// requested since the previous call. Allows autoplay's progress hook to
// honor `kill -USR1 <pid>` by writing a snapshot CSV.
bool force_table_consume_dump_request(void);

// Per-cell W and L counts for STRATUM cells (other kinds return 0/0).
// Reads the packed atomic with relaxed ordering — values may be stale by
// a few games but bounded by atomic semantics. The scheduler uses this
// to identify which cells are W- or L-deficient.
void force_table_get_stratum_wl(const ForceTable *table, int target_idx,
                                uint32_t *out_w, uint32_t *out_l);

// Same as above but addresses the target by pointer (e.g. resolved via
// force_table_lookup_target_by_key and stored downstream). Phase 3 hot
// path uses this to avoid a redundant index lookup.
void force_table_get_stratum_wl_by_ptr(const ForceTable *table,
                                       const ForceTarget *t,
                                       uint32_t *out_w, uint32_t *out_l);

// Number of targets currently loaded (including satisfied ones).
int force_table_num_targets(const ForceTable *table);

// Classify a leave as all/cons/mixed/vowel per the aggregation rule.
// Blanks (machine letter 0) ignored for the v/c count; leaves of length <= 2
// always return LEAVE_TYPE_ALL.
LeaveType force_classify_leave(const Rack *leave,
                               const LetterDistribution *ld);

// Lookup a target by its full (bag, leave_length, type, kind, exchange,
// subleave, diff) key. `diff` is the single cell diff (e.g. from
// cell_rarity.csv); the function returns the unique target whose
// [diff_min, diff_max] range contains it. Returns NULL if no target matches.
// Used by the rare-rack pool loader to resolve cell keys to ForceTarget*.
ForceTarget *force_table_lookup_target_by_key(
    ForceTable *table, int bag, int leave_length, LeaveType type,
    ForceTargetKind kind, int exchange, MachineLetter sub_ml0,
    MachineLetter sub_ml1, int subleave_count, int diff);

#endif
