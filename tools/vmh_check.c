// Equivalence check: magpie's vmodel_hier.c vs the B repo's hier_model.py.
//
// Input CSV (from hier_vmh_testcases.py), header then rows:
//   rack,leave,kind,diff,unseen,static,p_python
// unseen = 27 ';'-separated counts (blank first). Prints the max |p_c - p_python|,
// how many rows differ by more than --tol (default 1e-6), and the worst rows.
// Exit status 0 iff no row exceeds the tolerance.
//
// Build (self-contained):
//   cc -O2 -o vmh_check tools/vmh_check.c src/ent/vmodel_hier.c -lm
// Run:
//   ./vmh_check model.vmh cases.csv [tol]

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

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: vmh_check model.vmh cases.csv [tol]\n");
    return 2;
  }
  double tol = argc > 3 ? atof(argv[3]) : 1e-6;
  VModelHier *m = vmodel_hier_create(argv[1]);
  if (!m) return 2;
  FILE *f = fopen(argv[2], "r");
  if (!f) { fprintf(stderr, "cannot open %s\n", argv[2]); return 2; }
  char line[4096];
  if (!fgets(line, sizeof line, f)) return 2;          // header
  long n = 0, bad = 0;
  double worst = 0.0, sum = 0.0;
  char worst_rows[5][4096];
  double worst_d[5] = {0};
  while (fgets(line, sizeof line, f)) {
    char copy[4096];
    strcpy(copy, line);
    char *rest = line, *fld[8];
    int nf = 0;
    for (char *t = strsep(&rest, ","); t && nf < 8; t = strsep(&rest, ",")) fld[nf++] = t;
    if (nf < 7) continue;
    uint8_t rack[VMH_NT], leave[VMH_NT];
    if (counts(fld[0], rack) || counts(fld[1], leave)) continue;
    int kind = atoi(fld[2]), diff = atoi(fld[3]);
    int unseen[VMH_NT];
    char *u = fld[4];
    for (int t = 0; t < VMH_NT; t++) unseen[t] = (int)strtol(u, &u, 10), u += (*u == ';');
    float stat = (float)atof(fld[5]);
    double pp = atof(fld[6]);
    double pc = vmodel_hier_predict(m, rack, leave, kind, diff, unseen, stat);
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
  printf("%s: %ld rows, max |diff| %.3g, mean %.3g, %ld over tol %.1g\n", argv[2], n,
         worst, n ? sum / n : 0.0, bad, tol);
  for (int k = 0; k < 5 && worst_d[k] > 0; k++) printf("  %s", worst_rows[k]);
  vmodel_hier_destroy(m);
  return bad ? 1 : 0;
}
