#include "rare_pool.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../def/letter_distribution_defs.h"
#include "../util/io_util.h"
#include "force_table.h"

#define RARE_POOL_MAX_RACK_LEN 16

typedef struct {
  char *rack;                 // canonical sorted rack
  ForceTarget **targets;      // resolved force_table pointers
  int num_targets;
  int cap_targets;
} RareRack;

struct RarePool {
  RareRack *racks;
  int num_racks;
  int cap_racks;
};

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

static LeaveType type_str_to_enum(const char *s) {
  if (strcmp(s, "all") == 0) return LEAVE_TYPE_ALL;
  if (strcmp(s, "cons") == 0) return LEAVE_TYPE_CONS;
  if (strcmp(s, "mixed") == 0) return LEAVE_TYPE_MIXED;
  if (strcmp(s, "vowel") == 0) return LEAVE_TYPE_VOWEL;
  return (LeaveType)-1;
}

static ForceTargetKind kind_str_to_enum(const char *s) {
  if (strcmp(s, "tile") == 0) return FORCE_TARGET_TILE;
  if (strcmp(s, "pair") == 0) return FORCE_TARGET_PAIR;
  if (strcmp(s, "stratum") == 0) return FORCE_TARGET_STRATUM;
  if (strcmp(s, "bag_tile") == 0) return FORCE_TARGET_BAG_TILE;
  return (ForceTargetKind)-1;
}

static int find_or_add_rack(RarePool *rp, const char *rack_str) {
  // Linear scan over recently-added racks. Since the input file is sorted
  // by rack, the rack we want is almost always the last one.
  if (rp->num_racks > 0 &&
      strcmp(rp->racks[rp->num_racks - 1].rack, rack_str) == 0) {
    return rp->num_racks - 1;
  }
  // General scan as fallback (shouldn't happen for sorted input).
  for (int i = 0; i < rp->num_racks; i++) {
    if (strcmp(rp->racks[i].rack, rack_str) == 0) return i;
  }
  // Not found — append.
  if (rp->num_racks == rp->cap_racks) {
    rp->cap_racks = rp->cap_racks ? rp->cap_racks * 2 : 256;
    rp->racks = (RareRack *)realloc(
        rp->racks, sizeof(RareRack) * (size_t)rp->cap_racks);
    if (!rp->racks) {
      fprintf(stderr, "rare_pool: oom on rack alloc\n");
      exit(1);
    }
  }
  RareRack *r = &rp->racks[rp->num_racks++];
  r->rack = strdup(rack_str);
  r->targets = NULL;
  r->num_targets = 0;
  r->cap_targets = 0;
  return rp->num_racks - 1;
}

static void append_target(RareRack *r, ForceTarget *t) {
  if (r->num_targets == r->cap_targets) {
    r->cap_targets = r->cap_targets ? r->cap_targets * 2 : 16;
    r->targets = (ForceTarget **)realloc(
        r->targets, sizeof(ForceTarget *) * (size_t)r->cap_targets);
    if (!r->targets) {
      fprintf(stderr, "rare_pool: oom on target alloc\n");
      exit(1);
    }
  }
  r->targets[r->num_targets++] = t;
}

// Internal: load (rack,cell) rows from a CSV into an existing pool.
// Header is consumed; each row resolved via force_table_lookup and
// appended to the matching rack's target list.
static void rare_pool_load_file(RarePool *rp, const char *csv_path,
                                 ForceTable *force_table,
                                 const LetterDistribution *ld) {
  FILE *f = fopen(csv_path, "r");
  if (!f) {
    fprintf(stderr, "rare_pool: cannot open %s\n", csv_path);
    return;
  }
  char line[512];
  if (!fgets(line, sizeof(line), f)) {
    fclose(f);
    return;
  }
  // Header: rack,length,type,kind,subleave,diff
  int n_rows = 0, n_resolved = 0, n_unresolved = 0;
  while (fgets(line, sizeof(line), f)) {
    n_rows++;
    size_t len = strlen(line);
    while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
      line[--len] = '\0';
    }
    if (len == 0) continue;
    char rack_buf[RARE_POOL_MAX_RACK_LEN + 1] = {0};
    int length = 0;
    char type_str[16] = {0};
    char kind_str[16] = {0};
    char sub_buf[8] = {0};
    int diff = 0;
    if (sscanf(line, "%15[^,],%d,%15[^,],%15[^,],%7[^,],%d", rack_buf,
               &length, type_str, kind_str, sub_buf, &diff) != 6) {
      continue;
    }
    LeaveType type = type_str_to_enum(type_str);
    ForceTargetKind kind = kind_str_to_enum(kind_str);
    if ((int)type < 0 || (int)kind < 0) continue;
    int sub_count = (int)strlen(sub_buf);
    MachineLetter sub_ml0 = 0, sub_ml1 = 0;
    if (kind == FORCE_TARGET_BAG_TILE) {
      // bag_tile subleave: "<TILE>_free" or "<TILE>_<C>" — match parser
      // in force_table.c. For "_free": sub_count=1. For "_<C>": sub_count=2
      // with mls[1] = C, used to look up the per-count cell.
      sub_ml0 = char_to_ml(ld, sub_buf[0]);
      const size_t sl_len = strlen(sub_buf);
      if (sl_len == 6 && strcmp(sub_buf + 2, "free") == 0) {
        sub_count = 1;
      } else if (sl_len == 3 && sub_buf[1] == '_' &&
                 sub_buf[2] >= '0' && sub_buf[2] <= '7') {
        sub_count = 2;
        sub_ml1 = (MachineLetter)(sub_buf[2] - '0');
      } else {
        continue;
      }
    } else {
      if (sub_count >= 1) sub_ml0 = char_to_ml(ld, sub_buf[0]);
      if (sub_count >= 2) sub_ml1 = char_to_ml(ld, sub_buf[1]);
    }
    if (sub_ml0 == 0xFF || (sub_count >= 2 && sub_ml1 == 0xFF)) continue;
    // Force-target lookup: bag=93 (opener), exchange=0 (cells in
    // cell_rarity.csv are play-kind only — built from action_type==play).
    ForceTarget *t = force_table_lookup_target_by_key(
        force_table, /*bag=*/93, length, type, kind, /*exchange=*/0,
        sub_ml0, sub_ml1, sub_count, diff);
    if (!t) {
      n_unresolved++;
      continue;
    }
    int ri = find_or_add_rack(rp, rack_buf);
    append_target(&rp->racks[ri], t);
    n_resolved++;
  }
  fclose(f);
  fprintf(stderr,
          "rare_pool: loaded %s — %d/%d rows resolved (%d unresolved); "
          "pool now has %d racks\n",
          csv_path, n_resolved, n_rows, n_unresolved, rp->num_racks);
}

RarePool *rare_pool_create(const char *csv_path, ForceTable *force_table,
                           const LetterDistribution *ld) {
  if (!csv_path || !force_table || !ld) return NULL;
  RarePool *rp = (RarePool *)malloc_or_die(sizeof(RarePool));
  rp->racks = NULL;
  rp->num_racks = 0;
  rp->cap_racks = 0;
  rare_pool_load_file(rp, csv_path, force_table, ld);
  if (rp->num_racks == 0) {
    rare_pool_destroy(rp);
    return NULL;
  }
  return rp;
}

void rare_pool_load_more(RarePool *rp, const char *csv_path,
                          ForceTable *force_table,
                          const LetterDistribution *ld) {
  if (!rp || !csv_path || !force_table || !ld) return;
  rare_pool_load_file(rp, csv_path, force_table, ld);
}

void rare_pool_destroy(RarePool *rp) {
  if (!rp) return;
  for (int i = 0; i < rp->num_racks; i++) {
    free(rp->racks[i].rack);
    free(rp->racks[i].targets);
  }
  free(rp->racks);
  free(rp);
}

int rare_pool_num_racks(const RarePool *rp) {
  return rp ? rp->num_racks : 0;
}

const char *rare_pool_get_rack(const RarePool *rp, int idx) {
  if (!rp || idx < 0 || idx >= rp->num_racks) return NULL;
  return rp->racks[idx].rack;
}

static uint64_t splitmix64(uint64_t *x) {
  *x += 0x9e3779b97f4a7c15ULL;
  uint64_t z = *x;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

int rare_pool_sample_deficit_aware(const RarePool *rp, uint64_t seed) {
  if (!rp || rp->num_racks == 0) return -1;
  // Single-pass max-with-reservoir: track max score and uniformly pick
  // from the (potentially large) tied set without allocating.
  int best_score = 0;
  int n_tied = 0;
  int picked = -1;
  uint64_t s = seed ^ 0xa1b2c3d4e5f60708ULL;
  for (int i = 0; i < rp->num_racks; i++) {
    const RareRack *r = &rp->racks[i];
    int score = 0;
    for (int j = 0; j < r->num_targets; j++) {
      if (r->targets[j]->deficit > 0) score++;
    }
    if (score > best_score) {
      best_score = score;
      n_tied = 1;
      picked = i;
    } else if (score == best_score && score > 0) {
      n_tied++;
      // Reservoir step: replace `picked` with prob 1/n_tied.
      const uint64_t h = splitmix64(&s);
      if ((h % (uint64_t)n_tied) == 0) picked = i;
    }
  }
  return picked;
}
