#include "opening_pass.h"

#include <math.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../util/io_util.h"
#include "letter_distribution.h"

#define OPENING_PASS_DEFAULT_TARGET_N 1000
#define OPENING_PASS_MAX_RACK_LEN 16  // 7 chars + slack

typedef struct RackStats {
  // Marginal counters per branch (0 = pass, 1 = play)
  _Atomic int64_t wins[2];
  _Atomic int64_t losses[2];
  _Atomic int64_t ties[2];
  // Paired counters (only updated on pair completion)
  _Atomic int64_t n_pairs;
  // Sum of (pass_outcome_int - play_outcome_int) where outcome_int =
  // 2 * outcome (L=0, T=2, W=4). Diff range: [-4, +4].
  _Atomic int64_t sum_diff;
  _Atomic int64_t sum_sq_diff;
  // SPRT decided flag: once set, this rack is excluded from further
  // assignments. 0 = undecided / still gathering data, 1 = decided.
  _Atomic int decided;
} RackStats;

struct OpeningPassTable {
  char **rack_strings;  // [num_racks], NUL-terminated
  int num_racks;
  RackStats *stats;     // [num_racks]
  int target_n;          // MAX_PAIRS — cap on pairs per rack
  int sprt_min_pairs;    // minimum n_pairs before SPRT can decide a rack
  double sprt_z;         // z-threshold for declaring a rack decided
};

static int read_int_env(const char *name, int default_value, int min_value,
                        int max_value) {
  const char *env = getenv(name);
  if (!env || !*env) {
    return default_value;
  }
  long n = strtol(env, NULL, 10);
  if (n < (long)min_value || n > (long)max_value) {
    return default_value;
  }
  return (int)n;
}

static double read_double_env(const char *name, double default_value) {
  const char *env = getenv(name);
  if (!env || !*env) {
    return default_value;
  }
  char *endp = NULL;
  double v = strtod(env, &endp);
  if (endp == env || v <= 0.0) {
    return default_value;
  }
  return v;
}

static int read_target_n_from_env(void) {
  return read_int_env("MAGPIE_OPENING_PASS_TARGET_N",
                      OPENING_PASS_DEFAULT_TARGET_N, 1, 1000000000);
}

OpeningPassTable *opening_pass_table_create(const char *path,
                                            const LetterDistribution *ld) {
  (void)ld;  // racks are stored as strings; LD only used for validation later
  if (!path || !*path) {
    return NULL;
  }
  FILE *f = fopen(path, "r");
  if (!f) {
    fprintf(stderr, "opening_pass: cannot open %s\n", path);
    return NULL;
  }

  // First pass: count lines.
  int num_racks = 0;
  char line[OPENING_PASS_MAX_RACK_LEN + 4];
  while (fgets(line, sizeof(line), f)) {
    // Strip trailing newline / whitespace.
    size_t len = strlen(line);
    while (len > 0 &&
           (line[len - 1] == '\n' || line[len - 1] == '\r' ||
            line[len - 1] == ' ' || line[len - 1] == '\t')) {
      line[--len] = '\0';
    }
    if (len > 0) {
      num_racks++;
    }
  }
  if (num_racks == 0) {
    fclose(f);
    fprintf(stderr, "opening_pass: %s contains no racks\n", path);
    return NULL;
  }
  rewind(f);

  OpeningPassTable *table = malloc_or_die(sizeof(OpeningPassTable));
  table->num_racks = num_racks;
  table->target_n = read_target_n_from_env();
  table->sprt_min_pairs = read_int_env("MAGPIE_OPENING_PASS_MIN_PAIRS",
                                       50, 1, 1000000000);
  table->sprt_z = read_double_env("MAGPIE_OPENING_PASS_SPRT_Z", 2.326);
  table->rack_strings = malloc_or_die(sizeof(char *) * (size_t)num_racks);
  table->stats = malloc_or_die(sizeof(RackStats) * (size_t)num_racks);

  for (int i = 0; i < num_racks; i++) {
    atomic_store(&table->stats[i].wins[0], 0);
    atomic_store(&table->stats[i].wins[1], 0);
    atomic_store(&table->stats[i].losses[0], 0);
    atomic_store(&table->stats[i].losses[1], 0);
    atomic_store(&table->stats[i].ties[0], 0);
    atomic_store(&table->stats[i].ties[1], 0);
    atomic_store(&table->stats[i].n_pairs, 0);
    atomic_store(&table->stats[i].sum_diff, 0);
    atomic_store(&table->stats[i].sum_sq_diff, 0);
    atomic_store(&table->stats[i].decided, 0);
  }

  // Second pass: load racks.
  int idx = 0;
  while (fgets(line, sizeof(line), f) && idx < num_racks) {
    size_t len = strlen(line);
    while (len > 0 &&
           (line[len - 1] == '\n' || line[len - 1] == '\r' ||
            line[len - 1] == ' ' || line[len - 1] == '\t')) {
      line[--len] = '\0';
    }
    if (len == 0) {
      continue;
    }
    char *copy = malloc_or_die(len + 1);
    memcpy(copy, line, len + 1);
    table->rack_strings[idx++] = copy;
  }
  fclose(f);

  fprintf(stderr,
          "opening_pass: loaded %d racks from %s (target N = %d pairs/rack, "
          "SPRT min_pairs = %d, z = %.3f)\n",
          num_racks, path, table->target_n, table->sprt_min_pairs,
          table->sprt_z);
  return table;
}

int opening_pass_table_resume(OpeningPassTable *table, const char *csv_path) {
  if (!table || !csv_path || !*csv_path) {
    return 0;
  }
  FILE *f = fopen(csv_path, "r");
  if (!f) {
    fprintf(stderr, "opening_pass: resume: cannot open %s\n", csv_path);
    return 0;
  }
  // Build a quick rack -> idx map by linear scan (num_racks is ~200K which
  // is fine to lookup via simple sorted comparison; we use linear scan
  // here for simplicity since resume is a one-time startup cost).
  // For better performance with large tables, sort + bsearch.
  char line[1024];
  int merged = 0;
  // Skip header
  if (!fgets(line, sizeof(line), f)) {
    fclose(f);
    return 0;
  }
  while (fgets(line, sizeof(line), f)) {
    char rack[OPENING_PASS_MAX_RACK_LEN + 1] = {0};
    long long n_pairs = 0;
    long long pw = 0, pl = 0, pt = 0;
    long long yw = 0, yl = 0, yt = 0;
    long long sd = 0, ssd = 0;
    int decided = 0;
    int parsed = sscanf(
        line, "%15[^,],%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%d",
        rack, &n_pairs, &pw, &pl, &pt, &yw, &yl, &yt, &sd, &ssd, &decided);
    // Older CSVs without the decided column also accepted (parsed == 10).
    if (parsed != 10 && parsed != 11) {
      continue;
    }
    int rack_idx = -1;
    for (int i = 0; i < table->num_racks; i++) {
      if (strcmp(table->rack_strings[i], rack) == 0) {
        rack_idx = i;
        break;
      }
    }
    if (rack_idx < 0) {
      continue;
    }
    RackStats *s = &table->stats[rack_idx];
    atomic_fetch_add(&s->wins[0], (int64_t)pw);
    atomic_fetch_add(&s->losses[0], (int64_t)pl);
    atomic_fetch_add(&s->ties[0], (int64_t)pt);
    atomic_fetch_add(&s->wins[1], (int64_t)yw);
    atomic_fetch_add(&s->losses[1], (int64_t)yl);
    atomic_fetch_add(&s->ties[1], (int64_t)yt);
    atomic_fetch_add(&s->n_pairs, (int64_t)n_pairs);
    atomic_fetch_add(&s->sum_diff, (int64_t)sd);
    atomic_fetch_add(&s->sum_sq_diff, (int64_t)ssd);
    if (decided) {
      atomic_store(&s->decided, 1);
    }
    merged++;
  }
  fclose(f);
  fprintf(stderr,
          "opening_pass: resume: merged %d rows from %s into table\n",
          merged, csv_path);
  return merged;
}

void opening_pass_table_destroy(OpeningPassTable *table) {
  if (!table) {
    return;
  }
  for (int i = 0; i < table->num_racks; i++) {
    free(table->rack_strings[i]);
  }
  free(table->rack_strings);
  free(table->stats);
  free(table);
}

int opening_pass_table_num_racks(const OpeningPassTable *table) {
  return table ? table->num_racks : 0;
}

int opening_pass_table_target_n(const OpeningPassTable *table) {
  return table ? table->target_n : 0;
}

const char *opening_pass_table_get_rack(const OpeningPassTable *table,
                                        uint64_t pair_id, int *rack_idx_out) {
  if (!table || table->num_racks == 0) {
    if (rack_idx_out) {
      *rack_idx_out = -1;
    }
    return NULL;
  }
  // Cycle: each rack gets target_n consecutive pairs as the natural slot.
  // If the natural slot is decided (SPRT-stopped) or full (n_pairs >=
  // target_n), scan forward (with wrap) for the next still-active rack.
  // Returns NULL if every rack is done.
  const int n = table->num_racks;
  const int start = (int)((pair_id / (uint64_t)table->target_n) %
                          (uint64_t)n);
  for (int offset = 0; offset < n; offset++) {
    const int idx = (start + offset) % n;
    if (atomic_load_explicit(&table->stats[idx].decided,
                             memory_order_relaxed) != 0) {
      continue;
    }
    if (atomic_load_explicit(&table->stats[idx].n_pairs,
                             memory_order_relaxed) >=
        (int64_t)table->target_n) {
      continue;
    }
    if (rack_idx_out) {
      *rack_idx_out = idx;
    }
    return table->rack_strings[idx];
  }
  if (rack_idx_out) {
    *rack_idx_out = -1;
  }
  return NULL;
}

void opening_pass_record(OpeningPassTable *table, int rack_idx, int branch,
                         int outcome) {
  if (!table || rack_idx < 0 || rack_idx >= table->num_racks) {
    return;
  }
  if (branch < 0 || branch > 1) {
    return;
  }
  RackStats *s = &table->stats[rack_idx];
  if (outcome == 2) {
    atomic_fetch_add_explicit(&s->wins[branch], 1, memory_order_relaxed);
  } else if (outcome == 1) {
    atomic_fetch_add_explicit(&s->ties[branch], 1, memory_order_relaxed);
  } else {
    atomic_fetch_add_explicit(&s->losses[branch], 1, memory_order_relaxed);
  }
}

void opening_pass_record_pair(OpeningPassTable *table, int rack_idx,
                              int pass_outcome, int play_outcome) {
  if (!table || rack_idx < 0 || rack_idx >= table->num_racks) {
    return;
  }
  RackStats *s = &table->stats[rack_idx];
  // outcome_int = 2 * outcome (L=0, T=2, W=4). Diff in [-4, +4].
  const int pass_int = 2 * pass_outcome;
  const int play_int = 2 * play_outcome;
  const int diff = pass_int - play_int;
  atomic_fetch_add_explicit(&s->n_pairs, 1, memory_order_relaxed);
  atomic_fetch_add_explicit(&s->sum_diff, (int64_t)diff, memory_order_relaxed);
  atomic_fetch_add_explicit(&s->sum_sq_diff, (int64_t)(diff * diff),
                            memory_order_relaxed);

  // SPRT early-stop check: once we have enough pairs and the paired
  // |delta| / SE exceeds the z-threshold, mark this rack as decided so
  // future games skip it. Skip the check if already decided.
  if (atomic_load_explicit(&s->decided, memory_order_relaxed) != 0) {
    return;
  }
  const int64_t n =
      atomic_load_explicit(&s->n_pairs, memory_order_relaxed);
  if (n < (int64_t)table->sprt_min_pairs) {
    return;
  }
  const int64_t sd =
      atomic_load_explicit(&s->sum_diff, memory_order_relaxed);
  const int64_t ssd =
      atomic_load_explicit(&s->sum_sq_diff, memory_order_relaxed);
  // sum_diff is in 2x-outcome units (range per pair: [-4, 4]).
  // mean_diff (in win-rate units) = sd / (4 * n)
  // var_diff   (in win-rate units) = ssd / (16 * n) - mean_diff^2
  // paired_se  = sqrt(var_diff / n)
  const double n_d = (double)n;
  const double mean_diff = (double)sd / (4.0 * n_d);
  const double mean_sq = (double)ssd / (16.0 * n_d);
  double var_diff = mean_sq - mean_diff * mean_diff;
  if (var_diff < 0.0) {
    var_diff = 0.0;
  }
  const double se = (var_diff > 0.0) ? sqrt(var_diff / n_d) : 0.0;
  // If SE is essentially zero (all paired diffs identical) and mean_diff
  // is nonzero, the rack is decided regardless of z.
  bool decide = false;
  if (se == 0.0 && mean_diff != 0.0) {
    decide = true;
  } else if (se > 0.0) {
    const double zstat = mean_diff / se;
    if (zstat >= table->sprt_z || zstat <= -table->sprt_z) {
      decide = true;
    }
  }
  if (decide) {
    int expected = 0;
    atomic_compare_exchange_strong(&s->decided, &expected, 1);
  }
}

bool opening_pass_table_is_complete(const OpeningPassTable *table) {
  if (!table) {
    return false;
  }
  const int64_t target = (int64_t)table->target_n;
  for (int i = 0; i < table->num_racks; i++) {
    // A rack is "done" if it's been SPRT-decided OR it's reached MAX_PAIRS.
    if (atomic_load_explicit(&table->stats[i].decided,
                             memory_order_relaxed) != 0) {
      continue;
    }
    if (atomic_load_explicit(&table->stats[i].n_pairs,
                             memory_order_relaxed) < target) {
      return false;
    }
  }
  return true;
}

int64_t opening_pass_table_remaining(const OpeningPassTable *table) {
  if (!table) {
    return 0;
  }
  const int64_t target = (int64_t)table->target_n;
  int64_t total_remaining = 0;
  for (int i = 0; i < table->num_racks; i++) {
    if (atomic_load_explicit(&table->stats[i].decided,
                             memory_order_relaxed) != 0) {
      continue;
    }
    int64_t n = atomic_load_explicit(&table->stats[i].n_pairs,
                                     memory_order_relaxed);
    if (n < target) {
      total_remaining += (target - n);
    }
  }
  return total_remaining;
}

void opening_pass_table_dump(const OpeningPassTable *table,
                             const char *csv_path) {
  if (!table || !csv_path || !*csv_path) {
    return;
  }
  FILE *f = fopen(csv_path, "w");
  if (!f) {
    fprintf(stderr, "opening_pass: cannot write %s\n", csv_path);
    return;
  }
  fprintf(f,
          "rack,n_pairs,"
          "pass_wins,pass_losses,pass_ties,"
          "play_wins,play_losses,play_ties,"
          "sum_diff,sum_sq_diff,decided\n");
  for (int i = 0; i < table->num_racks; i++) {
    const RackStats *s = &table->stats[i];
    fprintf(f, "%s,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%d\n",
            table->rack_strings[i],
            (long long)atomic_load_explicit(&s->n_pairs, memory_order_relaxed),
            (long long)atomic_load_explicit(&s->wins[0], memory_order_relaxed),
            (long long)atomic_load_explicit(&s->losses[0],
                                            memory_order_relaxed),
            (long long)atomic_load_explicit(&s->ties[0], memory_order_relaxed),
            (long long)atomic_load_explicit(&s->wins[1], memory_order_relaxed),
            (long long)atomic_load_explicit(&s->losses[1],
                                            memory_order_relaxed),
            (long long)atomic_load_explicit(&s->ties[1], memory_order_relaxed),
            (long long)atomic_load_explicit(&s->sum_diff,
                                            memory_order_relaxed),
            (long long)atomic_load_explicit(&s->sum_sq_diff,
                                            memory_order_relaxed),
            atomic_load_explicit(&s->decided, memory_order_relaxed));
  }
  fclose(f);
  fprintf(stderr, "opening_pass: wrote %d rows to %s\n", table->num_racks,
          csv_path);
}
