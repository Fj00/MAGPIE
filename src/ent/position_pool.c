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
