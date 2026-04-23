#include "force_table.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../util/io_util.h"
#include "../util/string_util.h"

// Maximum bag count indexed directly. Bag sizes we care about are 0..93.
#define FORCE_BAG_MAX 94

struct ForceTable {
  ForceTarget *targets; // heap-allocated array
  int num_targets;
  int capacity;
  // Index: per bag count, an array of pointers into targets[]. A satisfied
  // target stays in the array (its deficit==0) — lookup callers must filter.
  ForceTarget **by_bag_ptrs[FORCE_BAG_MAX];
  int by_bag_count[FORCE_BAG_MAX];
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
                          int score) {
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

bool force_target_decrement(ForceTarget *target) {
  if (target->deficit > 0) {
    target->deficit--;
  }
  return target->deficit == 0;
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

  log_info("force_table: loaded %d targets from %s", table->num_targets,
           csv_path);
  return table;
}

void force_table_destroy(ForceTable *table) {
  if (!table) {
    return;
  }
  for (int b = 0; b < FORCE_BAG_MAX; b++) {
    free(table->by_bag_ptrs[b]);
  }
  free(table->targets);
  free(table);
}
