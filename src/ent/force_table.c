#include "force_table.h"

#include <ctype.h>
#include <limits.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../util/io_util.h"
#include "../util/string_util.h"

// Maximum bag count indexed directly. Bag sizes we care about are 0..93.
#define FORCE_BAG_MAX 94

// Per-stratum diff-tally bounds: ±FORCE_DIFF_MAX. Diffs outside this range
// are clamped to the boundary (matches the spreadsheet WINDOW behavior).
// 1001 buckets per stratum target × 8 bytes = 8 KB.
#define FORCE_DIFF_MAX 500
#define FORCE_DIFF_BUCKETS (FORCE_DIFF_MAX * 2 + 1)

// Atomic packed (wins:32, losses:32). Updated with atomic_fetch_add so that
// each game's pre-increment value uniquely tells it whether IT was the one
// that bumped min(w,l).
typedef struct StratumTally {
  _Atomic uint64_t buckets[FORCE_DIFF_BUCKETS];
} StratumTally;

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
  if (target->deficit <= 0) {
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
static inline uint32_t required_bitmap_for_target(const ForceTarget *target) {
  uint32_t bm = 0;
  for (int i = 0; i < target->subleave_count; i++) {
    bm |= ((uint32_t)1) << target->subleave_mls[i];
  }
  return bm;
}

// Populate a ForceTargetSlot from its full target. Used at load time and
// to refresh slot.deficit when the cold deficit decrements.
static inline void slot_init_from_target(ForceTargetSlot *slot,
                                          ForceTarget *target) {
  slot->deficit = (int32_t)target->deficit;
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
  if (target->deficit > 0) {
    target->deficit--;
    // Update the parallel slot's deficit so the hot loop sees the new value
    // without needing to dereference the cold target. (Slot's deficit is
    // int32, but per-cell deficit always fits.)
    {
      const int bag = target->bag;
      const int L = target->leave_length;
      const int ex = target->exchange ? 1 : 0;
      if (bag >= 0 && bag < FORCE_BAG_MAX && L >= 0 &&
          L < FORCE_LEAVE_LEN_MAX) {
        ForceTargetSlot *slots = table->by_shape_slots[bag][L][ex];
        if (slots && target->by_shape_idx >= 0) {
          slots[target->by_shape_idx].deficit = (int32_t)target->deficit;
        }
      }
    }
    if (target->deficit == 0) {
      atomic_fetch_sub_explicit(&table->active_targets, 1,
                                memory_order_relaxed);
      // B5: push the now-satisfied target to the back of each of its buckets
      // so workers stop iterating it. Brief lock — only contended on the
      // (rare) deficit-hits-zero transition.
      pthread_mutex_lock(&table->bucket_mutex);
      swap_to_back_bag(table, target);
      swap_to_back_shape(table, target);
      pthread_mutex_unlock(&table->bucket_mutex);
      return true;
    }
  }
  return target->deficit == 0;
}

void force_table_credit_game(ForceTable *table, ForceTarget *target,
                             int diff, bool is_win, bool is_tie) {
  // Tile/pair: count-based semantics (one matched game = one credit).
  if (target->kind != FORCE_TARGET_STRATUM) {
    force_table_decrement_target(table, target);
    return;
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
  int clamped = diff;
  if (clamped < -FORCE_DIFF_MAX) {
    clamped = -FORCE_DIFF_MAX;
  } else if (clamped > FORCE_DIFF_MAX) {
    clamped = FORCE_DIFF_MAX;
  }
  const int bucket = clamped + FORCE_DIFF_MAX;
  if (is_tie || st == NULL) {
    return;
  }
  // Pack wins in the high 32 bits, losses in the low 32 bits.
  const uint64_t increment = is_win ? ((uint64_t)1 << 32) : (uint64_t)1;
  const uint64_t old = atomic_fetch_add_explicit(
      &st->buckets[bucket], increment, memory_order_relaxed);
  const uint32_t old_w = (uint32_t)(old >> 32);
  const uint32_t old_l = (uint32_t)(old & 0xFFFFFFFFu);
  const uint32_t new_w = old_w + (is_win ? 1u : 0u);
  const uint32_t new_l = old_l + (is_win ? 0u : 1u);
  const uint32_t old_min = (old_w < old_l) ? old_w : old_l;
  const uint32_t new_min = (new_w < new_l) ? new_w : new_l;
  if (new_min > old_min) {
    force_table_decrement_target(table, target);
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

int64_t force_table_total_remaining(const ForceTable *table) {
  int64_t total = 0;
  for (int i = 0; i < table->num_targets; i++) {
    total += table->targets[i].deficit;
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
    t->deficit = deficit;
    t->diff_min = diff_min;
    t->diff_max = diff_max;

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
  // Sort each shape bucket by deficit DESCENDING (rarest cells first).
  // Per-emit force matching iterates from front; rarer targets match
  // preferentially. Unlike the prior deficit-asc sort (which caused
  // quadratic-time depleted-entry scans), depleted entries get swapped to
  // the back via the B5 active-count mechanism (see force_table_credit_*),
  // so iteration only touches live entries.
  // Sort ptrs first, then re-init slots and bitmaps from sorted ptrs.
  for (int b = 0; b < FORCE_BAG_MAX; b++) {
    for (int L = 0; L < FORCE_LEAVE_LEN_MAX; L++) {
      for (int ex = 0; ex < FORCE_EXCHANGE_MAX; ex++) {
        const int n = table->by_shape_count[b][L][ex];
        if (n <= 1) continue;
        ForceTarget **arr = table->by_shape_ptrs[b][L][ex];
        // Insertion sort by deficit desc — bucket sizes are typically small.
        for (int i = 1; i < n; i++) {
          ForceTarget *key = arr[i];
          int j = i - 1;
          while (j >= 0 && arr[j]->deficit < key->deficit) {
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

  // Allocate per-target diff tallies for stratum-kind targets. Tile/pair
  // targets keep tally=NULL since they use count-based decrement.
  table->stratum_tallies = (StratumTally **)malloc_or_die(
      sizeof(StratumTally *) * table->num_targets);
  for (int i = 0; i < table->num_targets; i++) {
    if (table->targets[i].kind == FORCE_TARGET_STRATUM) {
      StratumTally *st = (StratumTally *)malloc_or_die(sizeof(StratumTally));
      for (int b = 0; b < FORCE_DIFF_BUCKETS; b++) {
        atomic_store_explicit(&st->buckets[b], 0, memory_order_relaxed);
      }
      table->stratum_tallies[i] = st;
    } else {
      table->stratum_tallies[i] = NULL;
    }
  }

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
             "forced_games_estimate,diff_min,diff_max\n");
  const char *kind_names[] = {"stratum", "tile", "pair"};
  const char *type_names[] = {"all", "cons", "mixed", "vowel"};
  int rows_written = 0;
  for (int i = 0; i < table->num_targets; i++) {
    const ForceTarget *t = &table->targets[i];
    if (t->deficit <= 0) {
      continue;
    }
    char subleave[3] = {0};
    if (t->subleave_count >= 1) {
      subleave[0] = (t->subleave_mls[0] == 0) ? '?' :
                    ld->ld_ml_to_hl[t->subleave_mls[0]][0];
    }
    if (t->subleave_count >= 2) {
      subleave[1] = (t->subleave_mls[1] == 0) ? '?' :
                    ld->ld_ml_to_hl[t->subleave_mls[1]][0];
    }
    // current/target/forced_games_estimate aren't tracked precisely after
    // runtime; emit 0 placeholders except deficit (the reliable field).
    // Diff range echoed back as-loaded.
    fprintf(f, "%s,%d,%d,%s,%d,%s,0,0,%lld,0,%d,%d\n",
            kind_names[t->kind], t->bag, t->leave_length,
            type_names[t->leave_type], t->exchange, subleave,
            (long long)t->deficit, t->diff_min, t->diff_max);
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
