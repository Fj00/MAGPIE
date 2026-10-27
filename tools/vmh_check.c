// Equivalence checks for magpie's vmodel_hier.c against the B repo's Python.
//
//   vmh_check model.vmh cases.csv [tol]
//     cases (hier_vmh_testcases.py): rack,leave,kind,diff,unseen,static,p_python
//     For every row: |vmodel_hier_predict - p_python| (must be <= tol, default 1e-6)
//     and |turn-cache prediction - vmodel_hier_predict| (must be <= 1e-12).
//   vmh_check --sixpass sixpass.csv
//     rows rack,unseen,lead,sixpass_python: vmodel_hier_sixpass_diff must match.
// unseen = 27 ';'-separated counts (blank first). Exit 0 iff everything matches.
//
// Build (self-contained):
//   cc -O2 -o vmh_check tools/vmh_check.c src/ent/vmodel_hier.c -lm

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/ent/vmodel_hier.h"

static int counts(const char *s, uint8_t out[VMH_NT]) {
  memset(out, 0, VMH_NT);
  for (; *s; s++) {
    if (*s == '?') out[0]++;
    else if (*s >= 'A' && *s <= 'Z') out[*s - 'A' + 1]++;
    else return -1;
  }
  return 0;
}

static int sorted_idx(const uint8_t c[VMH_NT], uint8_t *idx) {
  int n = 0;
  for (int t = 0; t < VMH_NT; t++)
    for (int k = 0; k < c[t]; k++) idx[n++] = (uint8_t)t;
  return n;
}

static void parse_unseen(char *u, int out[VMH_NT]) {
  for (int t = 0; t < VMH_NT; t++) out[t] = (int)strtol(u, &u, 10), u += (*u == ';');
}

static float row_static(const uint8_t *idx, int len, void *ctx) {
  (void)idx, (void)len;
  return *(float *)ctx;
}

static int split(char *line, char **fld, int cap) {
  int nf = 0;
  char *rest = line;
  for (char *t = strsep(&rest, ","); t && nf < cap; t = strsep(&rest, ",")) fld[nf++] = t;
  return nf;
}

static int check_sixpass(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
  char line[4096], *fld[6];
  if (!fgets(line, sizeof line, f)) return 2;
  long n = 0, bad = 0;
  while (fgets(line, sizeof line, f)) {
    if (split(line, fld, 6) < 4) continue;
    uint8_t rack[VMH_NT];
    int unseen[VMH_NT];
    if (counts(fld[0], rack)) continue;
    parse_unseen(fld[1], unseen);
    int c = vmodel_hier_sixpass_diff(rack, unseen, atoi(fld[2]));
    int p = atoi(fld[3]);
    n++;
    if (c != p) {
      if (bad++ < 5) printf("  MISMATCH %s lead %s: c %d python %d\n", fld[0], fld[2], c, p);
    }
  }
  fclose(f);
  printf("%s: %ld rows, %ld sixpass mismatches\n", path, n, bad);
  return bad ? 1 : 0;
}

int main(int argc, char **argv) {
  if (argc >= 3 && strcmp(argv[1], "--sixpass") == 0) return check_sixpass(argv[2]);
  if (argc < 3) {
    fprintf(stderr, "usage: vmh_check model.vmh cases.csv [tol] | --sixpass file\n");
    return 2;
  }
  double tol = argc > 3 ? atof(argv[3]) : 1e-6;
  VModelHier *m = vmodel_hier_create(argv[1]);
  if (!m) return 2;
  VmhTurnCache *tc = vmh_turn_cache_create();
  FILE *f = fopen(argv[2], "r");
  if (!f) { fprintf(stderr, "cannot open %s\n", argv[2]); return 2; }
  char line[4096];
  if (!fgets(line, sizeof line, f)) return 2;          // header
  long n = 0, bad = 0, bad_tc = 0;
  double worst = 0.0, sum = 0.0, worst_tc = 0.0;
  char worst_rows[5][4096];
  double worst_d[5] = {0};
  while (fgets(line, sizeof line, f)) {
    char copy[4096];
    strcpy(copy, line);
    char *fld[8];
    if (split(line, fld, 8) < 7) continue;
    uint8_t rack[VMH_NT], leave[VMH_NT];
    if (counts(fld[0], rack) || counts(fld[1], leave)) continue;
    int kind = atoi(fld[2]), diff = atoi(fld[3]);
    int unseen[VMH_NT];
    parse_unseen(fld[4], unseen);
    float stat = (float)atof(fld[5]);
    double pp = atof(fld[6]);
    double pc = vmodel_hier_predict(m, rack, leave, kind, diff, unseen, stat);
    // the same move through the per-turn cache
    uint8_t ridx[16], lidx[16];
    int rl = sorted_idx(rack, ridx), ll = sorted_idx(leave, lidx);
    vmh_turn_begin(tc, m, ridx, rl, unseen, row_static, &stat);
    double pt = vmh_turn_predict(tc, lidx, ll, kind, diff);
    double dt = fabs(pt - pc);
    if (dt > worst_tc) worst_tc = dt;
    if (dt > 1e-12) bad_tc++;
    double d = fabs(pc - pp);
    n++;
    sum += d;
    if (d > tol) bad++;
    if (d > worst) worst = d;
    for (int k = 0; k < 5; k++) {
      if (d > worst_d[k]) {
        for (int j = 4; j > k; j--) { worst_d[j] = worst_d[j - 1]; strcpy(worst_rows[j], worst_rows[j - 1]); }
        worst_d[k] = d;
        snprintf(worst_rows[k], sizeof worst_rows[k], "%.9f  c=%.9f  %s", d, pc, copy);
        break;
      }
    }
  }
  fclose(f);
  printf("%s: %ld rows, max |C - python| %.3g, mean %.3g, %ld over tol %.1g; "
         "turn cache vs direct max %.3g, %ld over 1e-12\n", argv[2], n, worst,
         n ? sum / n : 0.0, bad, tol, worst_tc, bad_tc);
  for (int k = 0; k < 5 && worst_d[k] > 0; k++) printf("  %s", worst_rows[k]);
  vmh_turn_cache_destroy(tc);
  vmodel_hier_destroy(m);
  return (bad || bad_tc) ? 1 : 0;
}
