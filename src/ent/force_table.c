#include "force_table.h"

#include <ctype.h>
#include <limits.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../util/io_util.h"
#include "../util/string_util.h"

// Maximum bag count indexed directly. Bag sizes we care about are 0..93.
#define FORCE_BAG_MAX 94

// Diagnostic counters (see force_table.h). File-scope atomics; cheap relaxed
// increments on the credit hot path. Used to investigate the multi-credit
// regression: counts how many credit calls land as decrements vs no-ops.
static _Atomic uint64_t g_credit_calls = 0;
static _Atomic uint64_t g_decrements_landed = 0;
static _Atomic uint64_t g_noops_already_zero = 0;
static _Atomic uint64_t g_stratum_no_bump = 0;
static _Atomic uint64_t g_cas_retries = 0;
// Per-length STRATUM credit counters — to investigate why length>=3
// STRATUM cells never drain despite the run emitting millions of leaves.
static _Atomic uint64_t g_stratum_credits_by_len[6] = {0};
static _Atomic uint64_t g_stratum_bumps_by_len[6] = {0};

// SIGUSR1 flag: set by signal handler, consumed by autoplay's progress hook.
// kill -USR1 <pid> requests a snapshot dump on the next progress tick.
static _Atomic int g_dump_request = 0;

static void force_table_sigusr1_handler(int sig) {
  (void)sig;
  atomic_store_explicit(&g_dump_request, 1, memory_order_relaxed);
}

bool force_table_consume_dump_request(void) {
  int expected = 1;
  return atomic_compare_exchange_strong_explicit(
      &g_dump_request, &expected, 0, memory_order_relaxed,
      memory_order_relaxed);
}

void force_table_get_counters(ForceCounters *out) {
  out->credit_calls =
      atomic_load_explicit(&g_credit_calls, memory_order_relaxed);
  out->decrements_landed =
      atomic_load_explicit(&g_decrements_landed, memory_order_relaxed);
  out->noops_already_zero =
      atomic_load_explicit(&g_noops_already_zero, memory_order_relaxed);
  out->stratum_no_bump =
      atomic_load_explicit(&g_stratum_no_bump, memory_order_relaxed);
  out->cas_retries =
      atomic_load_explicit(&g_cas_retries, memory_order_relaxed);
}

void force_table_reset_counters(void) {
  atomic_store_explicit(&g_credit_calls, 0, memory_order_relaxed);
  atomic_store_explicit(&g_decrements_landed, 0, memory_order_relaxed);
  atomic_store_explicit(&g_noops_already_zero, 0, memory_order_relaxed);
  atomic_store_explicit(&g_stratum_no_bump, 0, memory_order_relaxed);
  atomic_store_explicit(&g_cas_retries, 0, memory_order_relaxed);
}

// STRATUM cells use a single packed (wins:32, losses:32) atomic counter.
// Per-bucket subdivision was removed — the spec wants total W and total L
// per cell to each exceed the per-side threshold T (= CSV target column).
// Each credit at the cell increments W or L (or both, for ties).
typedef struct StratumTally {
  _Atomic uint64_t wl;
} StratumTally;

// Kept for backward compat with progress-line counter names — no longer
// per-bucket subdivision, but the FORCE_DIFF_MAX constant is still used
// elsewhere for clamping if needed.
#define FORCE_DIFF_MAX 500
#define FORCE_DIFF_BUCKETS (FORCE_DIFF_MAX * 2 + 1)

// A2 optimization: bucket targets by (bag, leave_length, exchange) so per-move
// matching iterates only the small subset compatible with that move's leave
// shape, instead of the full ~23K targets at the bag.
#define FORCE_LEAVE_LEN_MAX 8 // valid leave_length values 0..7
#define FORCE_EXCHANGE_MAX 2  // exchange flag is 0 or 1

struct ForceTable {
  ForceTarget *targets; // heap-allocated array
  int num_targets;
  int capacity;
  // Count of targets whose deficit is still > 0. Decrements when a target's
  // deficit reaches zero; reaches zero when every target is satisfied.
  atomic_int active_targets;
  // Index: per bag count, an array of pointers into targets[]. The first
  // by_bag_active[bag] entries are active (deficit > 0); entries beyond are
  // already-satisfied targets pushed to the back via swap-on-decrement (B5).
  ForceTarget **by_bag_ptrs[FORCE_BAG_MAX];
  int by_bag_count[FORCE_BAG_MAX]; // total entries (active + dead)
  _Atomic(int) by_bag_active[FORCE_BAG_MAX]; // active prefix size
  // A2: finer-grained index — per (bag, leave_length, exchange) shape. Same
  // active-prefix invariant as by_bag_ptrs.
  ForceTarget **by_shape_ptrs[FORCE_BAG_MAX][FORCE_LEAVE_LEN_MAX]
                             [FORCE_EXCHANGE_MAX];
  int by_shape_count[FORCE_BAG_MAX][FORCE_LEAVE_LEN_MAX][FORCE_EXCHANGE_MAX];
  _Atomic(int) by_shape_active[FORCE_BAG_MAX][FORCE_LEAVE_LEN_MAX]
                              [FORCE_EXCHANGE_MAX];
  // Hot-loop slots: parallel to by_shape_ptrs, contains inline hot fields +
  // cold pointer per slot. Workers iterate slots for cache-friendly matching.
  ForceTargetSlot *by_shape_slots[FORCE_BAG_MAX][FORCE_LEAVE_LEN_MAX]
                                 [FORCE_EXCHANGE_MAX];
  // Per-target required-tile bitmap (parallel to slots). Bit i = subleave
  // requires tile i (0=blank). Workers AND this with their leave's bitmap
  // and skip the slot when the result != required_bitmap. Compact 4 bytes
  // per slot — fits in L1 cache for fast pre-filter.
  uint32_t *by_shape_bitmaps[FORCE_BAG_MAX][FORCE_LEAVE_LEN_MAX]
                            [FORCE_EXCHANGE_MAX];
  // B5: serializes the swap-to-back operation when a target hits deficit==0.
  // Only acquired on the rare deficit-hits-zero transition; readers iterate
  // active prefix without locking (they may briefly see torn order on
  // concurrent swaps, but only ever read valid target pointers).
  pthread_mutex_t bucket_mutex;
  // Per-target diff tally for stratum-kind targets only (NULL otherwise).
  // Indexed by target's offset within targets[]. Used by
  // force_table_credit_game to decrement deficit only on min(w,l) bumps.
  StratumTally **stratum_tallies;
};

static ForceTargetKind parse_kind(const char *s) {
  if (strcmp(s, "stratum") == 0) {
    return FORCE_TARGET_STRATUM;
  }
  if (strcmp(s, "tile") == 0) {
    return FORCE_TARGET_TILE;
  }
  if (strcmp(s, "pair") == 0) {
    return FORCE_TARGET_PAIR;
  }
  if (strcmp(s, "bag_tile") == 0) {
    return FORCE_TARGET_BAG_TILE;
  }
  return -1;
}

static LeaveType parse_leave_type(const char *s) {
  if (strcmp(s, "all") == 0) {
    return LEAVE_TYPE_ALL;
  }
  if (strcmp(s, "cons") == 0) {
    return LEAVE_TYPE_CONS;
  }
  if (strcmp(s, "mixed") == 0) {
    return LEAVE_TYPE_MIXED;
  }
  if (strcmp(s, "vowel") == 0) {
    return LEAVE_TYPE_VOWEL;
  }
  return -1;
}

// Treat Y and the 5 primary vowels as vowels for the type classification.
// Blanks are ignored at type time.
static bool ml_counts_as_vowel(const LetterDistribution *ld, MachineLetter ml) {
  if (ml == 0) {
    return false;
  }
  const char *hl = ld->ld_ml_to_hl[ml];
  if (hl[0] == 'Y' || hl[0] == 'y') {
    return true;
  }
  return ld->is_vowel[ml];
}

// Convert a single-byte letter or '?' to a MachineLetter using the ld map.
// Returns 0xFF on failure.
static MachineLetter char_to_ml(const LetterDistribution *ld, char c) {
  if (c == '?') {
    return 0;
  }
  for (int i = 1; i < MACHINE_LETTER_MAX_VALUE; i++) {
    const char *hl = ld->ld_ml_to_hl[i];
    if (hl[0] == c && hl[1] == '\0') {
      return (MachineLetter)i;
    }
  }
  return (MachineLetter)0xFF;
}

LeaveType force_classify_leave(const Rack *leave,
                               const LetterDistribution *ld) {
  if (leave->number_of_letters <= 2) {
    return LEAVE_TYPE_ALL;
  }
  int vc = 0;
  int cc = 0;
  for (int ml = 1; ml < leave->dist_size; ml++) {
    int count = leave->array[ml];
    if (count <= 0) {
      continue;
    }
    if (ml_counts_as_vowel(ld, ml)) {
      vc += count;
    } else {
      cc += count;
    }
  }
  if (vc == 0) {
    return LEAVE_TYPE_CONS;
  }
  if (cc == 0) {
    return LEAVE_TYPE_VOWEL;
  }
  return LEAVE_TYPE_MIXED;
}

bool force_target_matches(const ForceTarget *target, const Rack *leave,
                          int score, int diff) {
  if (atomic_load_explicit(&target->deficit, memory_order_relaxed) <= 0) {
    return false;
  }
  if (leave->number_of_letters != target->leave_length) {
    return false;
  }
  int is_exchange = (score == 0) ? 1 : 0;
  if (is_exchange != target->exchange) {
    return false;
  }
  if (diff < target->diff_min || diff > target->diff_max) {
    return false;
  }
  // For exchanges or <=2 tile leaves, type is "all". For 3+ tile plays, match
  // the cons/mixed/vowel classification.
  if (target->exchange == 0 && target->leave_length >= 3) {
    // We need to compare against the target's leave type. The leave must be
    // classified the same way. The classifier uses Y as vowel (per agg rule).
    // Caller supplies the leave's Rack; classifier takes ld. Since target
    // carries only LeaveType (not ld), the caller must re-classify before
    // calling us. Here we only validate subleave_mls presence.
    // (Type check is done outside this function to avoid threading ld in.)
  }
  // Subleave checks
  for (int i = 0; i < target->subleave_count; i++) {
    const MachineLetter needed = target->subleave_mls[i];
    if (leave->array[needed] <= 0) {
      return false;
    }
    // For pair where both MLs are the same tile, require count >= 2.
    if (i == 1 && target->subleave_mls[0] == target->subleave_mls[1] &&
        leave->array[needed] < 2) {
      return false;
    }
  }
  return true;
}

// Compute the required-tile bitmap for a target's subleave_mls. Bit i is set
// for each tile type required to be present in the leave. For pair targets
// with same-tile pairs (e.g. "??"), only one bit is set in the bitmap (the
// count check is done downstream).
//
// BAG_TILE targets return 0 — their match predicate is on the rack, not on
// the leave. The leave-bitmap pre-filter `(leave_bitmap & req) == req`
// trivially passes when req == 0 (don't reject), so BAG_TILE slots fall
// through to the per-slot kind check in the matching loop.
static inline uint32_t required_bitmap_for_target(const ForceTarget *target) {
  if (target->kind == FORCE_TARGET_BAG_TILE) {
    return 0;
  }
  uint32_t bm = 0;
  for (int i = 0; i < target->subleave_count; i++) {
    bm |= ((uint32_t)1) << target->subleave_mls[i];
  }
  return bm;
}

bool force_target_matches_bag(const ForceTarget *target,
                              const Rack *pre_move_rack,
                              int leave_length, LeaveType leave_type,
                              int score, int diff,
                              const LetterDistribution *ld) {
  if (atomic_load_explicit(&target->deficit, memory_order_relaxed) <= 0) {
    return false;
  }
  if (target->kind != FORCE_TARGET_BAG_TILE) {
    return false;
  }
  if (leave_length != target->leave_length) {
    return false;
  }
  int is_exchange = (score == 0) ? 1 : 0;
  if (is_exchange != target->exchange) {
    return false;
  }
  if (diff < target->diff_min || diff > target->diff_max) {
    return false;
  }
  // Type check (parallel to force_target_matches): for plays with leave_length
  // >= 3, the leave's cons/mixed/vowel classification must match the target.
  if (target->exchange == 0 && target->leave_length >= 3) {
    if (leave_type != target->leave_type) {
      return false;
    }
  }
  // _free predicate: rack lacks at least one of target's tile.
  // rack.count(t) < TILE_BAG[t] ⇒ unseen > 0 ⇒ bag_exp_t > 0.
  const MachineLetter t = target->subleave_mls[0];
  const int tile_bag_count = ld_get_dist(ld, t);
  if (pre_move_rack->array[t] >= tile_bag_count) {
    return false;
  }
  return true;
}

// Populate a ForceTargetSlot from its full target. Used at load time and
// to refresh slot.deficit when the cold deficit decrements.
static inline void slot_init_from_target(ForceTargetSlot *slot,
                                          ForceTarget *target) {
  slot->deficit = (int32_t)atomic_load_explicit(
      &target->deficit, memory_order_relaxed);
  slot->kind = (uint8_t)target->kind;
  slot->leave_length = (uint8_t)target->leave_length;
  slot->exchange = (uint8_t)(target->exchange ? 1 : 0);
  slot->leave_type = (uint8_t)target->leave_type;
  slot->diff_min = target->diff_min;
  slot->diff_max = target->diff_max;
  slot->subleave_mls[0] = target->subleave_mls[0];
  slot->subleave_mls[1] = target->subleave_mls[1];
  slot->subleave_count = (uint8_t)target->subleave_count;
  slot->cold = target;
}

// B5: when a target hits deficit==0, swap it to the back of each of its
// buckets (by_bag and by_shape) and decrement that bucket's active count.
// Caller must hold table->bucket_mutex. Workers iterate only the active
// prefix, so dead entries swapped to the back never get visited again.
static void swap_to_back_bag(ForceTable *table, ForceTarget *target) {
  const int bag = target->bag;
  if (bag < 0 || bag >= FORCE_BAG_MAX) return;
  ForceTarget **ptrs = table->by_bag_ptrs[bag];
  if (!ptrs) return;
  int last = atomic_load_explicit(&table->by_bag_active[bag],
                                   memory_order_relaxed) - 1;
  int idx = target->by_bag_idx;
  if (idx < 0 || idx > last) return; // already inactive
  if (idx != last) {
    ForceTarget *swapped = ptrs[last];
    ptrs[idx] = swapped;
    ptrs[last] = target;
    swapped->by_bag_idx = idx;
    target->by_bag_idx = last;
  }
  atomic_store_explicit(&table->by_bag_active[bag], last,
                        memory_order_release);
}

static void swap_to_back_shape(ForceTable *table, ForceTarget *target) {
  const int bag = target->bag;
  const int L = target->leave_length;
  const int ex = target->exchange ? 1 : 0;
  if (bag < 0 || bag >= FORCE_BAG_MAX) return;
  if (L < 0 || L >= FORCE_LEAVE_LEN_MAX) return;
  ForceTarget **ptrs = table->by_shape_ptrs[bag][L][ex];
  ForceTargetSlot *slots = table->by_shape_slots[bag][L][ex];
  uint32_t *bitmaps = table->by_shape_bitmaps[bag][L][ex];
  if (!ptrs) return;
  int last = atomic_load_explicit(&table->by_shape_active[bag][L][ex],
                                   memory_order_relaxed) - 1;
  int idx = target->by_shape_idx;
  if (idx < 0 || idx > last) return;
  if (idx != last) {
    ForceTarget *swapped = ptrs[last];
    ptrs[idx] = swapped;
    ptrs[last] = target;
    swapped->by_shape_idx = idx;
    target->by_shape_idx = last;
    if (slots) {
      ForceTargetSlot tmp = slots[idx];
      slots[idx] = slots[last];
      slots[last] = tmp;
    }
    if (bitmaps) {
      uint32_t tmpb = bitmaps[idx];
      bitmaps[idx] = bitmaps[last];
      bitmaps[last] = tmpb;
    }
  }
  atomic_store_explicit(&table->by_shape_active[bag][L][ex], last,
                        memory_order_release);
}

bool force_table_decrement_target(ForceTable *table, ForceTarget *target) {
  // CAS loop: only the thread that observes a positive deficit AND
  // successfully drives it from K to K-1 does the post-decrement
  // bookkeeping. Threads observing 0 return early; CAS losers retry
  // with the freshly-loaded value. Plain `target->deficit--` lost
  // updates under concurrent multi-credit traffic — see the noble-
  // popping-kitten plan / commit log.
  int64_t cur = atomic_load_explicit(&target->deficit, memory_order_relaxed);
  if (cur == 0) {
    atomic_fetch_add_explicit(&g_noops_already_zero, 1, memory_order_relaxed);
    return true;  // already drained
  }
  while (cur > 0) {
    if (atomic_compare_exchange_weak_explicit(
            &target->deficit, &cur, cur - 1,
            memory_order_relaxed, memory_order_relaxed)) {
      atomic_fetch_add_explicit(&g_decrements_landed, 1, memory_order_relaxed);
      const int64_t new_deficit = cur - 1;
      // Mirror to the parallel slot cache so the hot match loop sees
      // the new value without dereferencing the cold target. (Slot
      // deficit is int32, but per-cell deficit always fits. Best-effort
      // — workers re-validate via cold target->deficit on candidate
      // match, so a briefly-stale slot only causes false-positives that
      // the cold check rejects.)
      const int bag = target->bag;
      const int L = target->leave_length;
      const int ex = target->exchange ? 1 : 0;
      if (bag >= 0 && bag < FORCE_BAG_MAX && L >= 0 &&
          L < FORCE_LEAVE_LEN_MAX) {
        ForceTargetSlot *slots = table->by_shape_slots[bag][L][ex];
        if (slots && target->by_shape_idx >= 0) {
          slots[target->by_shape_idx].deficit = (int32_t)new_deficit;
        }
      }
      if (new_deficit == 0) {
        atomic_fetch_sub_explicit(&table->active_targets, 1,
                                   memory_order_relaxed);
        // B5: push the now-satisfied target to the back of each bucket
        // so workers stop iterating it. Brief lock — only contended on
        // the (rare) deficit-hits-zero transition.
        pthread_mutex_lock(&table->bucket_mutex);
        swap_to_back_bag(table, target);
        swap_to_back_shape(table, target);
        pthread_mutex_unlock(&table->bucket_mutex);
        return true;
      }
      return false;
    }
    // CAS failed; cur was reloaded by atomic_compare_exchange_weak.
    atomic_fetch_add_explicit(&g_cas_retries, 1, memory_order_relaxed);
  }
  // Lost the race — another thread drained the cell to 0 while we were
  // looping. Count as a no-op (same outcome as entering with deficit==0).
  atomic_fetch_add_explicit(&g_noops_already_zero, 1, memory_order_relaxed);
  return true;
}

void force_table_credit_game(ForceTable *table, ForceTarget *target,
                             int diff, bool is_win, bool is_tie) {
  atomic_fetch_add_explicit(&g_credit_calls, 1, memory_order_relaxed);
  atomic_fetch_add_explicit(&target->credit_attempts, 1,
                            memory_order_relaxed);
  // Tile/pair: count-based semantics (one matched game = one credit).
  if (target->kind != FORCE_TARGET_STRATUM) {
    force_table_decrement_target(table, target);
    return;
  }
  // STRATUM credit — count by length for diagnosis.
  {
    int ll = target->leave_length;
    if (ll < 0) ll = 0;
    if (ll > 5) ll = 5;
    atomic_fetch_add_explicit(&g_stratum_credits_by_len[ll], 1,
                              memory_order_relaxed);
  }
  // Ties don't change min(wins, losses). Treat them as a non-credit, but
  // still try to drain progress so a stratum with many ties doesn't stall.
  // A tie at the boundary (w == l) would have left min unchanged anyway, so
  // skipping the decrement is correct for the EPV metric. We still update
  // the tally so future games see accurate state.
  const int idx = (int)(target - table->targets);
  StratumTally *st = (table->stratum_tallies != NULL)
                         ? table->stratum_tallies[idx]
                         : NULL;
  if (st == NULL) {
    atomic_fetch_add_explicit(&g_stratum_no_bump, 1, memory_order_relaxed);
    return;
  }
  // Per-side per-cell accounting: track total wins and losses. Cell drains
  // when both totals >= target->stratum_per_side (T). Each credit
  // increments W and/or L; the deficit decrements once per side that
  // crosses T for the first time.
  //   win  → W += 1
  //   loss → L += 1
  //   tie  → W += 1 AND L += 1 (counts as one of each)
  const bool inc_w = is_tie || is_win;
  const bool inc_l = is_tie || !is_win;
  uint64_t increment = 0;
  if (inc_w) increment |= ((uint64_t)1 << 32);
  if (inc_l) increment |= (uint64_t)1;
  if (increment == 0) return;  // defensive — shouldn't happen
  const uint64_t old = atomic_fetch_add_explicit(
      &st->wl, increment, memory_order_relaxed);
  const uint32_t old_w = (uint32_t)(old >> 32);
  const uint32_t old_l = (uint32_t)(old & 0xFFFFFFFFu);
  const int32_t per_side = target->stratum_per_side;
  // Stratum-by-length bump counters (kept for diagnostics — increment per
  // side that crossed threshold).
  int ll = target->leave_length;
  if (ll < 0) ll = 0;
  if (ll > 5) ll = 5;
  // Was W previously below per-side target, and this credit advanced W?
  if (inc_w && per_side > 0 && (int32_t)old_w < per_side) {
    atomic_fetch_add_explicit(&g_stratum_bumps_by_len[ll], 1,
                              memory_order_relaxed);
    force_table_decrement_target(table, target);
  } else if (inc_w) {
    atomic_fetch_add_explicit(&g_stratum_no_bump, 1, memory_order_relaxed);
  }
  if (inc_l && per_side > 0 && (int32_t)old_l < per_side) {
    atomic_fetch_add_explicit(&g_stratum_bumps_by_len[ll], 1,
                              memory_order_relaxed);
    force_table_decrement_target(table, target);
  } else if (inc_l) {
    atomic_fetch_add_explicit(&g_stratum_no_bump, 1, memory_order_relaxed);
  }
}

void force_table_get_stratum_wl(const ForceTable *table, int target_idx,
                                uint32_t *out_w, uint32_t *out_l) {
  *out_w = 0;
  *out_l = 0;
  if (!table || target_idx < 0 || target_idx >= table->num_targets) return;
  if (table->stratum_tallies == NULL) return;
  StratumTally *st = table->stratum_tallies[target_idx];
  if (!st) return;
  uint64_t packed = atomic_load_explicit(&st->wl, memory_order_relaxed);
  *out_w = (uint32_t)(packed >> 32);
  *out_l = (uint32_t)(packed & 0xFFFFFFFFu);
}

void force_table_get_stratum_wl_by_ptr(const ForceTable *table,
                                       const ForceTarget *t,
                                       uint32_t *out_w, uint32_t *out_l) {
  *out_w = 0;
  *out_l = 0;
  if (!table || !t) return;
  // ForceTargets are allocated as a contiguous array in table->targets;
  // pointer subtraction yields the index used by the stratum tally array.
  ptrdiff_t idx = t - table->targets;
  force_table_get_stratum_wl(table, (int)idx, out_w, out_l);
}

void force_table_get_stratum_by_len(uint64_t credits[6], uint64_t bumps[6]) {
  for (int i = 0; i < 6; i++) {
    credits[i] = atomic_load_explicit(&g_stratum_credits_by_len[i],
                                       memory_order_relaxed);
    bumps[i] = atomic_load_explicit(&g_stratum_bumps_by_len[i],
                                     memory_order_relaxed);
  }
}

void force_table_reset_stratum_by_len(void) {
  for (int i = 0; i < 6; i++) {
    atomic_store_explicit(&g_stratum_credits_by_len[i], 0,
                          memory_order_relaxed);
    atomic_store_explicit(&g_stratum_bumps_by_len[i], 0,
                          memory_order_relaxed);
  }
}

bool force_table_is_exhausted(const ForceTable *table) {
  return atomic_load_explicit(&table->active_targets, memory_order_relaxed) ==
         0;
}

ForceTarget **force_table_lookup(ForceTable *table, int bag, int *count) {
  if (bag < 0 || bag >= FORCE_BAG_MAX) {
    *count = 0;
    return NULL;
  }
  // Return only the active prefix size — dead entries (deficit==0) get
  // swapped past this index by force_table_decrement_target.
  *count = atomic_load_explicit(&table->by_bag_active[bag],
                                memory_order_acquire);
  return table->by_bag_ptrs[bag];
}

ForceTarget **force_table_lookup_by_shape(ForceTable *table, int bag,
                                          int leave_length, int exchange,
                                          int *count) {
  if (bag < 0 || bag >= FORCE_BAG_MAX || leave_length < 0 ||
      leave_length >= FORCE_LEAVE_LEN_MAX) {
    *count = 0;
    return NULL;
  }
  const int ex = exchange ? 1 : 0;
  *count = atomic_load_explicit(&table->by_shape_active[bag][leave_length][ex],
                                memory_order_acquire);
  return table->by_shape_ptrs[bag][leave_length][ex];
}

ForceTargetSlot *force_table_lookup_slots_by_shape(ForceTable *table, int bag,
                                                    int leave_length,
                                                    int exchange, int *count) {
  if (bag < 0 || bag >= FORCE_BAG_MAX || leave_length < 0 ||
      leave_length >= FORCE_LEAVE_LEN_MAX) {
    *count = 0;
    return NULL;
  }
  const int ex = exchange ? 1 : 0;
  *count = atomic_load_explicit(&table->by_shape_active[bag][leave_length][ex],
                                memory_order_acquire);
  return table->by_shape_slots[bag][leave_length][ex];
}

uint32_t *force_table_lookup_bitmaps_by_shape(ForceTable *table, int bag,
                                              int leave_length, int exchange) {
  if (bag < 0 || bag >= FORCE_BAG_MAX || leave_length < 0 ||
      leave_length >= FORCE_LEAVE_LEN_MAX) {
    return NULL;
  }
  const int ex = exchange ? 1 : 0;
  return table->by_shape_bitmaps[bag][leave_length][ex];
}

ForceTarget *force_table_lookup_target_by_key(
    ForceTable *table, int bag, int leave_length, LeaveType type,
    ForceTargetKind kind, int exchange, MachineLetter sub_ml0,
    MachineLetter sub_ml1, int subleave_count, int diff) {
  if (!table || bag < 0 || bag >= FORCE_BAG_MAX) return NULL;
  // Use the by_bag bucket — small enough that a linear scan is fine
  // (called only at rare-pool load time, not per game).
  const int ex = exchange ? 1 : 0;
  const int n = table->by_bag_count[bag];
  ForceTarget **arr = table->by_bag_ptrs[bag];
  for (int i = 0; i < n; i++) {
    ForceTarget *t = arr[i];
    if (t->leave_length != leave_length) continue;
    if ((int)t->leave_type != (int)type) continue;
    if ((int)t->kind != (int)kind) continue;
    if (t->exchange != ex) continue;
    if (t->subleave_count != subleave_count) continue;
    if (subleave_count >= 1 && t->subleave_mls[0] != sub_ml0) continue;
    if (subleave_count >= 2 && t->subleave_mls[1] != sub_ml1) continue;
    if (diff < t->diff_min || diff > t->diff_max) continue;
    return t;
  }
  return NULL;
}

int64_t force_table_total_remaining(const ForceTable *table) {
  int64_t total = 0;
  for (int i = 0; i < table->num_targets; i++) {
    total += atomic_load_explicit(&table->targets[i].deficit,
                                    memory_order_relaxed);
  }
  return total;
}

int force_table_num_targets(const ForceTable *table) {
  return table->num_targets;
}

// Simple CSV line split that modifies `line` in place; returns number of
// fields stored in `fields[]` (up to max_fields).
static int split_csv(char *line, char **fields, int max_fields) {
  int n = 0;
  char *p = line;
  fields[n++] = p;
  while (*p && n < max_fields) {
    if (*p == ',') {
      *p = '\0';
      fields[n++] = p + 1;
    }
    p++;
  }
  // Strip trailing newline from the last field
  if (n > 0) {
    char *last = fields[n - 1];
    size_t ll = strlen(last);
    while (ll > 0 && (last[ll - 1] == '\n' || last[ll - 1] == '\r')) {
      last[--ll] = '\0';
    }
  }
  return n;
}

ForceTable *force_table_create(const char *csv_path,
                               const LetterDistribution *ld) {
  FILE *f = fopen(csv_path, "r");
  if (!f) {
    log_warn("force_table: cannot open %s", csv_path);
    return NULL;
  }
  ForceTable *table = (ForceTable *)malloc_or_die(sizeof(ForceTable));
  memset(table, 0, sizeof(*table));
  pthread_mutex_init(&table->bucket_mutex, NULL);
  table->capacity = 1024;
  table->targets =
      (ForceTarget *)malloc_or_die(sizeof(ForceTarget) * table->capacity);

  char buf[1024];
  int line_no = 0;
  while (fgets(buf, sizeof(buf), f)) {
    line_no++;
    if (line_no == 1) {
      continue; // header
    }
    char *fields[16];
    int n = split_csv(buf, fields, 16);
    if (n < 10) {
      continue;
    }
    ForceTargetKind kind = parse_kind(fields[0]);
    if ((int)kind < 0) {
      continue;
    }
    int bag = atoi(fields[1]);
    int length = atoi(fields[2]);
    LeaveType type = parse_leave_type(fields[3]);
    if ((int)type < 0) {
      continue;
    }
    int exchange = atoi(fields[4]);
    const char *subleave = fields[5];
    int64_t deficit = strtoll(fields[8], NULL, 10);
    if (deficit <= 0) {
      continue;
    }
    // Optional diff range (columns 11-12, 1-indexed). Backward-compat:
    // 10-column CSVs default to no constraint.
    int diff_min = INT_MIN;
    int diff_max = INT_MAX;
    if (n >= 12) {
      diff_min = atoi(fields[10]);
      diff_max = atoi(fields[11]);
    }
    // Optional rarity_score column (1-indexed col 13). Lower = rarer cell.
    // Defaults to a very large number so cells without rarity data sort
    // AFTER cells with known small rarity within the same deficit tier.
    double rarity_score = 1e30;
    if (n >= 13) {
      char *endp = NULL;
      double parsed = strtod(fields[12], &endp);
      if (endp != fields[12] && parsed > 0) {
        rarity_score = parsed;
      }
    }

    if (table->num_targets == table->capacity) {
      table->capacity *= 2;
      table->targets = (ForceTarget *)realloc_or_die(
          table->targets, sizeof(ForceTarget) * table->capacity);
    }
    ForceTarget *t = &table->targets[table->num_targets++];
    t->kind = kind;
    t->bag = bag;
    t->leave_length = length;
    t->leave_type = type;
    t->exchange = exchange;
    t->subleave_count = 0;
    t->subleave_mls[0] = 0;
    t->subleave_mls[1] = 0;
    // For STRATUM cells: reinterpret CSV `target` (column 8) as per-side
    // threshold T. Cell needs total_wins >= T AND total_losses >= T.
    // Initial in-memory deficit = 2T (one unit per side per event needed).
    // Decrements 0/1/2 per credit depending on which sides were already
    // satisfied. For non-STRATUM kinds: deficit/target semantics unchanged.
    if (kind == FORCE_TARGET_STRATUM) {
      int64_t target_total = strtoll(fields[7], NULL, 10);
      if (target_total <= 0) target_total = deficit;
      t->stratum_per_side = (int32_t)target_total;
      atomic_store_explicit(&t->deficit, 2 * target_total,
                            memory_order_relaxed);
    } else {
      t->stratum_per_side = 0;
      atomic_store_explicit(&t->deficit, deficit, memory_order_relaxed);
    }
    t->diff_min = diff_min;
    t->diff_max = diff_max;
    t->rarity_score = rarity_score;

    if (kind == FORCE_TARGET_TILE) {
      if (strlen(subleave) != 1) {
        log_warn("force_table: bad tile subleave %s on line %d", subleave,
                 line_no);
        table->num_targets--;
        continue;
      }
      t->subleave_mls[0] = char_to_ml(ld, subleave[0]);
      t->subleave_count = 1;
      if (t->subleave_mls[0] == 0xFF) {
        log_warn("force_table: unknown tile %s on line %d", subleave, line_no);
        table->num_targets--;
        continue;
      }
    } else if (kind == FORCE_TARGET_PAIR) {
      if (strlen(subleave) != 2) {
        log_warn("force_table: bad pair subleave %s on line %d", subleave,
                 line_no);
        table->num_targets--;
        continue;
      }
      t->subleave_mls[0] = char_to_ml(ld, subleave[0]);
      t->subleave_mls[1] = char_to_ml(ld, subleave[1]);
      t->subleave_count = 2;
      if (t->subleave_mls[0] == 0xFF || t->subleave_mls[1] == 0xFF) {
        log_warn("force_table: unknown pair %s on line %d", subleave, line_no);
        table->num_targets--;
        continue;
      }
    } else if (kind == FORCE_TARGET_BAG_TILE) {
      // Expected format: "<TILE>_free" — single tile char + "_free" suffix.
      // The "_free" semantics: cell credits when the rack LACKS at least
      // one of this tile (unseen > 0). The "_held" complement is not
      // needed (handled by bucket merging when impossible, dominant by
      // default when possible).
      const size_t sl_len = strlen(subleave);
      if (sl_len < 6 || strcmp(subleave + sl_len - 5, "_free") != 0) {
        log_warn("force_table: bad bag_tile subleave %s on line %d "
                 "(expected '<TILE>_free')", subleave, line_no);
        table->num_targets--;
        continue;
      }
      if (sl_len != 6) { // single tile char + "_free"
        log_warn("force_table: bag_tile subleave %s must be 1 tile char + "
                 "'_free' on line %d", subleave, line_no);
        table->num_targets--;
        continue;
      }
      t->subleave_mls[0] = char_to_ml(ld, subleave[0]);
      t->subleave_count = 1;
      if (t->subleave_mls[0] == 0xFF) {
        log_warn("force_table: unknown bag_tile %s on line %d", subleave,
                 line_no);
        table->num_targets--;
        continue;
      }
    }
  }
  fclose(f);

  // Build by-bag index
  for (int i = 0; i < table->num_targets; i++) {
    int bag = table->targets[i].bag;
    if (bag >= 0 && bag < FORCE_BAG_MAX) {
      table->by_bag_count[bag]++;
    }
  }
  for (int b = 0; b < FORCE_BAG_MAX; b++) {
    if (table->by_bag_count[b] > 0) {
      table->by_bag_ptrs[b] = (ForceTarget **)malloc_or_die(
          sizeof(ForceTarget *) * table->by_bag_count[b]);
    }
  }
  int by_bag_fill[FORCE_BAG_MAX];
  memset(by_bag_fill, 0, sizeof(by_bag_fill));
  // Initialize all targets' bucket indices to -1 (will be set when placed).
  for (int i = 0; i < table->num_targets; i++) {
    table->targets[i].by_bag_idx = -1;
    table->targets[i].by_shape_idx = -1;
  }
  for (int i = 0; i < table->num_targets; i++) {
    int bag = table->targets[i].bag;
    if (bag >= 0 && bag < FORCE_BAG_MAX) {
      const int idx = by_bag_fill[bag]++;
      table->by_bag_ptrs[bag][idx] = &table->targets[i];
      table->targets[i].by_bag_idx = idx;
    }
  }
  // B5: initialize by_bag_active to full bucket size (all loaded targets are
  // active initially; satisfied ones get swapped to back via swap_to_back_bag
  // at decrement time).
  for (int b = 0; b < FORCE_BAG_MAX; b++) {
    atomic_store_explicit(&table->by_bag_active[b], table->by_bag_count[b],
                          memory_order_relaxed);
  }
  // Note: deficit-asc sort removed. The sort put low-deficit targets at the
  // front of each bucket so rare targets would be matched first. But once a
  // target's deficit hits 0 it stays at the front, and per-turn iteration
  // walks past those dead entries to find active targets — causing per-turn
  // cost to grow linearly with cumulative games played (quadratic total time).
  // Current insertion order is fine; per-turn iteration is now bucket-size
  // bounded and constant. Rare targets still get matched, just not
  // preferentially — fairness over time is preserved by force-table reissue
  // between iterations.

  // A2: build by_shape index per (bag, leave_length, exchange). Iterate the
  // already-sorted by_bag_ptrs so each shape bucket inherits deficit-ascending
  // order. This lets per-move matching iterate only the small subset whose
  // (leave_length, exchange) matches the move, instead of all targets at the
  // bag.
  memset(table->by_shape_count, 0, sizeof(table->by_shape_count));
  for (int b = 0; b < FORCE_BAG_MAX; b++) {
    for (int i = 0; i < table->by_bag_count[b]; i++) {
      ForceTarget *t = table->by_bag_ptrs[b][i];
      if (t->leave_length < 0 || t->leave_length >= FORCE_LEAVE_LEN_MAX) {
        continue;
      }
      const int ex = t->exchange ? 1 : 0;
      table->by_shape_count[b][t->leave_length][ex]++;
    }
  }
  for (int b = 0; b < FORCE_BAG_MAX; b++) {
    for (int L = 0; L < FORCE_LEAVE_LEN_MAX; L++) {
      for (int ex = 0; ex < FORCE_EXCHANGE_MAX; ex++) {
        const int n = table->by_shape_count[b][L][ex];
        if (n > 0) {
          table->by_shape_ptrs[b][L][ex] =
              (ForceTarget **)malloc_or_die(sizeof(ForceTarget *) * n);
          // Parallel slot array — populated below alongside the pointer array.
          table->by_shape_slots[b][L][ex] =
              (ForceTargetSlot *)malloc_or_die(sizeof(ForceTargetSlot) * n);
          // Parallel required-tile-bitmap array — used as fast pre-filter.
          table->by_shape_bitmaps[b][L][ex] =
              (uint32_t *)malloc_or_die(sizeof(uint32_t) * n);
        }
      }
    }
  }
  int by_shape_fill[FORCE_BAG_MAX][FORCE_LEAVE_LEN_MAX][FORCE_EXCHANGE_MAX];
  memset(by_shape_fill, 0, sizeof(by_shape_fill));
  for (int b = 0; b < FORCE_BAG_MAX; b++) {
    for (int i = 0; i < table->by_bag_count[b]; i++) {
      ForceTarget *t = table->by_bag_ptrs[b][i];
      if (t->leave_length < 0 || t->leave_length >= FORCE_LEAVE_LEN_MAX) {
        continue;
      }
      const int ex = t->exchange ? 1 : 0;
      const int idx = by_shape_fill[b][t->leave_length][ex]++;
      table->by_shape_ptrs[b][t->leave_length][ex][idx] = t;
      slot_init_from_target(
          &table->by_shape_slots[b][t->leave_length][ex][idx], t);
      table->by_shape_bitmaps[b][t->leave_length][ex][idx] =
          required_bitmap_for_target(t);
      t->by_shape_idx = idx;
    }
  }
  // Sort each shape bucket by composite key:
  //   primary:   deficit DESC  (largest deficit first — most need attention)
  //   secondary: rarity_score ASC  (rarest cells first within deficit tier)
  //
  // Per-emit force matching iterates from front; rarer targets at the same
  // deficit level get first dibs on the limited per-game slot budget.
  // Without rarity tiebreak, hard-to-construct cells like L5_cons JQ@16
  // got crowded out by easier deficit=100 peers and stayed at deficit=100
  // forever (verified against eb_run_06b residuals).
  //
  // Unlike the prior deficit-asc sort (which caused quadratic-time
  // depleted-entry scans), depleted entries get swapped to the back via
  // the B5 active-count mechanism (see force_table_credit_*), so iteration
  // only touches live entries.
  for (int b = 0; b < FORCE_BAG_MAX; b++) {
    for (int L = 0; L < FORCE_LEAVE_LEN_MAX; L++) {
      for (int ex = 0; ex < FORCE_EXCHANGE_MAX; ex++) {
        const int n = table->by_shape_count[b][L][ex];
        if (n <= 1) continue;
        ForceTarget **arr = table->by_shape_ptrs[b][L][ex];
        // Insertion sort by (deficit DESC, rarity ASC) — bucket sizes small.
        for (int i = 1; i < n; i++) {
          ForceTarget *key = arr[i];
          int j = i - 1;
          while (j >= 0 &&
                 (arr[j]->deficit < key->deficit ||
                  (arr[j]->deficit == key->deficit &&
                   arr[j]->rarity_score > key->rarity_score))) {
            arr[j + 1] = arr[j]; j--;
          }
          arr[j + 1] = key;
        }
        // Rebuild parallel slot/bitmap arrays in sorted order.
        for (int i = 0; i < n; i++) {
          ForceTarget *t = arr[i];
          slot_init_from_target(&table->by_shape_slots[b][L][ex][i], t);
          table->by_shape_bitmaps[b][L][ex][i] = required_bitmap_for_target(t);
          t->by_shape_idx = i;
        }
      }
    }
  }

  // B5: initialize by_shape_active to full bucket size.
  for (int b = 0; b < FORCE_BAG_MAX; b++) {
    for (int L = 0; L < FORCE_LEAVE_LEN_MAX; L++) {
      for (int ex = 0; ex < FORCE_EXCHANGE_MAX; ex++) {
        atomic_store_explicit(&table->by_shape_active[b][L][ex],
                              table->by_shape_count[b][L][ex],
                              memory_order_relaxed);
      }
    }
  }

  atomic_store_explicit(&table->active_targets, table->num_targets,
                        memory_order_relaxed);

  // Allocate per-cell W/L tallies for stratum-kind targets. Single packed
  // (wins:32, losses:32) atomic counter per cell — no per-bucket
  // subdivision. Tile/pair/bag targets keep tally=NULL (count-based).
  table->stratum_tallies = (StratumTally **)malloc_or_die(
      sizeof(StratumTally *) * table->num_targets);
  for (int i = 0; i < table->num_targets; i++) {
    if (table->targets[i].kind == FORCE_TARGET_STRATUM) {
      StratumTally *st = (StratumTally *)malloc_or_die(sizeof(StratumTally));
      atomic_store_explicit(&st->wl, 0, memory_order_relaxed);
      table->stratum_tallies[i] = st;
    } else {
      table->stratum_tallies[i] = NULL;
    }
  }

  // Register SIGUSR1 handler so `kill -USR1 <pid>` requests a mid-run dump
  // via the periodic progress hook in autoplay.
  struct sigaction sa = {0};
  sa.sa_handler = force_table_sigusr1_handler;
  sigemptyset(&sa.sa_mask);
  sigaction(SIGUSR1, &sa, NULL);

  fprintf(stderr, "force_table: loaded %d targets from %s (total deficit=%lld)\n",
          table->num_targets, csv_path,
          (long long)force_table_total_remaining(table));
  return table;
}

void force_table_dump_remaining(const ForceTable *table, const char *csv_path,
                                const LetterDistribution *ld) {
  FILE *f = fopen(csv_path, "w");
  if (!f) {
    fprintf(stderr, "force_table_dump: cannot open %s for write\n", csv_path);
    return;
  }
  fprintf(f, "kind,bag,length,type,exchange,subleave,current,target,deficit,"
             "forced_games_estimate,diff_min,diff_max,rarity_score,"
             "obs_w,obs_l,stratum_per_side,credit_attempts\n");
  const char *kind_names[] = {"stratum", "tile", "pair", "bag_tile"};
  const char *type_names[] = {"all", "cons", "mixed", "vowel"};
  int rows_written = 0;
  for (int i = 0; i < table->num_targets; i++) {
    const ForceTarget *t = &table->targets[i];
    if (t->deficit <= 0) {
      continue;
    }
    char subleave[8] = {0};  // up to "<TILE>_free" = 6 + null
    if (t->kind == FORCE_TARGET_BAG_TILE) {
      const char tile_ch = (t->subleave_mls[0] == 0)
                               ? '?' : ld->ld_ml_to_hl[t->subleave_mls[0]][0];
      snprintf(subleave, sizeof(subleave), "%c_free", tile_ch);
    } else {
      if (t->subleave_count >= 1) {
        subleave[0] = (t->subleave_mls[0] == 0) ? '?' :
                      ld->ld_ml_to_hl[t->subleave_mls[0]][0];
      }
      if (t->subleave_count >= 2) {
        subleave[1] = (t->subleave_mls[1] == 0) ? '?' :
                      ld->ld_ml_to_hl[t->subleave_mls[1]][0];
      }
    }
    // current/target/forced_games_estimate aren't tracked precisely after
    // runtime; emit 0 placeholders except deficit (the reliable field).
    // Diff range and rarity_score echoed back as-loaded.
    // For STRATUM cells also emit per-cell W/L tally and per-side target
    // (Phase 3 diagnostic — shows whether stuck cells are W- or L-skewed).
    uint32_t obs_w = 0, obs_l = 0;
    int per_side = 0;
    if (t->kind == FORCE_TARGET_STRATUM) {
      force_table_get_stratum_wl(table, i, &obs_w, &obs_l);
      per_side = t->stratum_per_side;
    }
    uint64_t cred_attempts = atomic_load_explicit(
        &t->credit_attempts, memory_order_relaxed);
    fprintf(f, "%s,%d,%d,%s,%d,%s,0,0,%lld,0,%d,%d,%.6e,%u,%u,%d,%llu\n",
            kind_names[t->kind], t->bag, t->leave_length,
            type_names[t->leave_type], t->exchange, subleave,
            (long long)t->deficit, t->diff_min, t->diff_max,
            t->rarity_score, obs_w, obs_l, per_side,
            (unsigned long long)cred_attempts);
    rows_written++;
  }
  fclose(f);
  fprintf(stderr, "force_table_dump: wrote %d remaining targets to %s\n",
          rows_written, csv_path);
}

void force_table_destroy(ForceTable *table) {
  if (!table) {
    return;
  }
  for (int b = 0; b < FORCE_BAG_MAX; b++) {
    free(table->by_bag_ptrs[b]);
    for (int L = 0; L < FORCE_LEAVE_LEN_MAX; L++) {
      for (int ex = 0; ex < FORCE_EXCHANGE_MAX; ex++) {
        free(table->by_shape_ptrs[b][L][ex]);
        free(table->by_shape_slots[b][L][ex]);
        free(table->by_shape_bitmaps[b][L][ex]);
      }
    }
  }
  pthread_mutex_destroy(&table->bucket_mutex);
  if (table->stratum_tallies) {
    for (int i = 0; i < table->num_targets; i++) {
      free(table->stratum_tallies[i]);
    }
    free(table->stratum_tallies);
  }
  free(table->targets);
  free(table);
}
