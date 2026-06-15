#include "position_pool.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../util/io_util.h"

struct PositionPool {
  char **cgps;
  uint64_t *ids;
  int count;
  int cap;
};

static void pp_push(PositionPool *pp, uint64_t id, const char *cgp) {
  if (pp->count == pp->cap) {
    pp->cap = pp->cap ? pp->cap * 2 : 1024;
    pp->cgps = realloc(pp->cgps, (size_t)pp->cap * sizeof(char *));
    pp->ids = realloc(pp->ids, (size_t)pp->cap * sizeof(uint64_t));
    if (!pp->cgps || !pp->ids) {
      log_fatal("position_pool: realloc failed at cap %d", pp->cap);
    }
  }
  pp->cgps[pp->count] = malloc_or_die(strlen(cgp) + 1);
  strcpy(pp->cgps[pp->count], cgp);
  pp->ids[pp->count] = id;
  pp->count++;
}

PositionPool *position_pool_create(const char *path) {
  if (!path || !path[0]) return NULL;
  FILE *f = fopen(path, "re");
  if (!f) {
    log_fatal("position_pool: cannot open %s", path);
  }
  PositionPool *pp = malloc_or_die(sizeof(PositionPool));
  pp->cgps = NULL;
  pp->ids = NULL;
  pp->count = 0;
  pp->cap = 0;

  char *line = NULL;
  size_t cap = 0;
  ssize_t len;
  uint64_t line_idx = 0;
  while ((len = getline(&line, &cap, f)) != -1) {
    // strip trailing newline / CR
    while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
      line[--len] = '\0';
    }
    if (len == 0 || line[0] == '#') {
      line_idx++;
      continue;
    }
    // optional "<id>\t<cgp>" — only when the field before the first tab is
    // all digits (a CGP itself contains spaces, never a leading numeric+tab).
    char *tab = strchr(line, '\t');
    uint64_t id = line_idx;
    const char *cgp = line;
    if (tab) {
      bool numeric = tab > line;
      for (char *p = line; p < tab; p++) {
        if (*p < '0' || *p > '9') {
          numeric = false;
          break;
        }
      }
      if (numeric) {
        *tab = '\0';
        id = strtoull(line, NULL, 10);
        cgp = tab + 1;
      }
    }
    pp_push(pp, id, cgp);
    line_idx++;
  }
  free(line);
  fclose(f);
  fprintf(stderr, "position_pool: loaded %d positions from %s\n", pp->count,
          path);
  return pp;
}

void position_pool_destroy(PositionPool *pp) {
  if (!pp) return;
  for (int i = 0; i < pp->count; i++) {
    free(pp->cgps[i]);
  }
  free(pp->cgps);
  free(pp->ids);
  free(pp);
}

int position_pool_count(const PositionPool *pp) {
  return pp ? pp->count : 0;
}

const char *position_pool_get_cgp(const PositionPool *pp, int i) {
  if (!pp || i < 0 || i >= pp->count) return NULL;
  return pp->cgps[i];
}

uint64_t position_pool_get_id(const PositionPool *pp, int i) {
  if (!pp || i < 0 || i >= pp->count) return 0;
  return pp->ids[i];
}

// ===== Opener pool =====

struct OpenerPool {
  char **racks;
  char **leaves;
  int *scores;
  char *signs;
  int count;
  int cap;
};

static void op_push(OpenerPool *op, const char *rack, int score,
                    const char *leave, char sign) {
  if (op->count == op->cap) {
    op->cap = op->cap ? op->cap * 2 : 4096;
    op->racks = realloc(op->racks, (size_t)op->cap * sizeof(char *));
    op->leaves = realloc(op->leaves, (size_t)op->cap * sizeof(char *));
    op->scores = realloc(op->scores, (size_t)op->cap * sizeof(int));
    op->signs = realloc(op->signs, (size_t)op->cap * sizeof(char));
    if (!op->racks || !op->leaves || !op->scores || !op->signs) {
      log_fatal("opener_pool: realloc failed at cap %d", op->cap);
    }
  }
  op->racks[op->count] = malloc_or_die(strlen(rack) + 1);
  strcpy(op->racks[op->count], rack);
  op->leaves[op->count] = malloc_or_die(strlen(leave) + 1);
  strcpy(op->leaves[op->count], leave);
  op->scores[op->count] = score;
  op->signs[op->count] = sign;
  op->count++;
}

OpenerPool *opener_pool_create(const char *path) {
  if (!path || !path[0]) return NULL;
  FILE *f = fopen(path, "re");
  if (!f) {
    log_fatal("opener_pool: cannot open %s", path);
  }
  OpenerPool *op = malloc_or_die(sizeof(OpenerPool));
  op->racks = NULL;
  op->leaves = NULL;
  op->scores = NULL;
  op->signs = NULL;
  op->count = 0;
  op->cap = 0;

  char *line = NULL;
  size_t cap = 0;
  ssize_t len;
  while ((len = getline(&line, &cap, f)) != -1) {
    while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
      line[--len] = '\0';
    }
    if (len == 0 || line[0] == '#') continue;
    // Fields: bag \t rack \t score \t leave \t sign  (bag informational; the
    // resulting bag falls out of the replayed move). Split manually — strtok_r
    // collapses consecutive tabs, but the leave field is EMPTY for a 7-tile
    // bingo opener (score \t \t sign), which would misalign the fields.
    char *fields[5];
    int nf = 0;
    char *p = line;
    fields[nf++] = p;
    while (nf < 5) {
      char *t = strchr(p, '\t');
      if (!t) break;
      *t = '\0';
      p = t + 1;
      fields[nf++] = p;
    }
    if (nf < 5 || fields[4][0] == '\0') {
      log_fatal("opener_pool: malformed line in %s", path);
    }
    op_push(op, fields[1], atoi(fields[2]), fields[3], fields[4][0]);
  }
  free(line);
  fclose(f);
  fprintf(stderr, "opener_pool: loaded %d openers from %s\n", op->count, path);
  return op;
}

void opener_pool_destroy(OpenerPool *op) {
  if (!op) return;
  for (int i = 0; i < op->count; i++) {
    free(op->racks[i]);
    free(op->leaves[i]);
  }
  free(op->racks);
  free(op->leaves);
  free(op->scores);
  free(op->signs);
  free(op);
}

int opener_pool_count(const OpenerPool *op) { return op ? op->count : 0; }

const char *opener_pool_get_rack(const OpenerPool *op, int i) {
  if (!op || i < 0 || i >= op->count) return NULL;
  return op->racks[i];
}

const char *opener_pool_get_leave(const OpenerPool *op, int i) {
  if (!op || i < 0 || i >= op->count) return NULL;
  return op->leaves[i];
}

int opener_pool_get_score(const OpenerPool *op, int i) {
  if (!op || i < 0 || i >= op->count) return 0;
  return op->scores[i];
}

char opener_pool_get_sign(const OpenerPool *op, int i) {
  if (!op || i < 0 || i >= op->count) return '-';
  return op->signs[i];
}
