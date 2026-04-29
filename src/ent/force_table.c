#include "force_table.h"

#include <ctype.h>
#include <limits.h>
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

struct ForceTable {
  ForceTarget *targets; // heap-allocated array
  int num_targets;
  int capacity;
  // Count of targets whose deficit is still > 0. Decrements when a target's
  // deficit reaches zero; reaches zero when every target is satisfied.
  atomic_int active_targets;
  // Index: per bag count, an array of pointers into targets[]. A satisfied
  // target stays in the array (its deficit==0) — lookup callers must filter.
  ForceTarget **by_bag_ptrs[FORCE_BAG_MAX];
  int by_bag_count[FORCE_BAG_MAX];
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

bool force_table_decrement_target(ForceTable *table, ForceTarget *target) {
  if (target->deficit > 0) {
    target->deficit--;
    if (target->deficit == 0) {
      atomic_fetch_sub_explicit(&table->active_targets, 1,
                                memory_order_relaxed);
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
  *count = table->by_bag_count[bag];
  return table->by_bag_ptrs[bag];
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
  for (int i = 0; i < table->num_targets; i++) {
    int bag = table->targets[i].bag;
    if (bag >= 0 && bag < FORCE_BAG_MAX) {
      table->by_bag_ptrs[bag][by_bag_fill[bag]++] = &table->targets[i];
    }
  }
  // Sort each bag's targets by deficit ascending so rare targets get first
  // shot at matching. High-deficit targets with common predicates would
  // otherwise starve the rare ones.
  for (int b = 0; b < FORCE_BAG_MAX; b++) {
    int n = table->by_bag_count[b];
    // Simple insertion sort — n typically ≤ few thousand, called once at load.
    for (int i = 1; i < n; i++) {
      ForceTarget *key = table->by_bag_ptrs[b][i];
      int j = i - 1;
      while (j >= 0 && table->by_bag_ptrs[b][j]->deficit > key->deficit) {
        table->by_bag_ptrs[b][j + 1] = table->by_bag_ptrs[b][j];
        j--;
      }
      table->by_bag_ptrs[b][j + 1] = key;
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
  }
  if (table->stratum_tallies) {
    for (int i = 0; i < table->num_targets; i++) {
      free(table->stratum_tallies[i]);
    }
    free(table->stratum_tallies);
  }
  free(table->targets);
  free(table);
}
