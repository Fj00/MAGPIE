#include "vmodel_hier.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Mirrors hier_model.py (B repo bots/winpct/scripts). Every rule here has a
// line there; keep them in step. tools/vmh_check.c compares the two on real
// moves.

static const uint8_t TILE_BAG[VMH_NT] = {2, 9, 2, 2, 4, 12, 2, 3, 2, 9, 1, 1, 4, 2,
                                         6, 8, 2, 1, 6, 4, 6, 4, 2, 2, 1, 2, 1};
// AEIOUY (python VOWELS); blank is neither
static const bool IS_VOWEL[VMH_NT] = {
    false, true,  false, false, false, true,  false, false, false, true,
    false, false, false, false, false, true,  false, false, false, false,
    false, true,  false, false, false, true,  false};

enum { T_ALL = 0, T_CONS, T_MIXED, T_VOWEL, T_CONSB, T_MIXEDB, T_VOWELB };
static const char *TYPE_NAME[VMH_N_TYPE] = {"all",   "cons",   "mixed", "vowel",
                                           "cons?", "mixed?", "vowel?"};

static int tile_index(char c) {
  if (c == '?') return 0;
  if (c >= 'A' && c <= 'Z') return c - 'A' + 1;
  return -1;
}

// ------------------------------------------------------------------ parsing
static char *tok(char **rest) {
  char *s = *rest;
  while (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r') s++;
  if (!*s) return NULL;
  char *e = s;
  while (*e && *e != ' ' && *e != '\t' && *e != '\n' && *e != '\r') e++;
  if (*e) *e++ = '\0';
  *rest = e;
  return s;
}

static bool tok_int(char **rest, int *out) {
  char *t = tok(rest);
  if (!t) return false;
  char *end;
  long v = strtol(t, &end, 10);
  if (*end) return false;
  *out = (int)v;
  return true;
}

static bool tok_double(char **rest, double *out) {
  char *t = tok(rest);
  if (!t) return false;
  char *end;
  *out = strtod(t, &end);
  return *end == '\0';
}

static bool read_doubles(char **rest, double *out, int n) {
  for (int i = 0; i < n; i++)
    if (!tok_double(rest, &out[i])) return false;
  return true;
}

static void map_feature(VModelHier *m, const char *name, int row) {
  int a, b;
  size_t n = strlen(name);
  if (strcmp(name, "static_leave") == 0) {
    m->static_row = row;
  } else if (strncmp(name, "cnt_", 4) == 0 && n == 5) {
    if ((a = tile_index(name[4])) >= 0) m->cnt_row[a] = row;
  } else if (strncmp(name, "pairp_", 6) == 0 && n == 8) {
    a = tile_index(name[6]), b = tile_index(name[7]);
    if (a >= 0 && b >= 0) m->pairp_row[a][b] = row;
  } else if (strncmp(name, "pair_", 5) == 0 && n == 7) {
    a = tile_index(name[5]), b = tile_index(name[6]);
    if (a >= 0 && b >= 0) m->pair_row[a][b] = row;
  } else if (strncmp(name, "ucnt_", 5) == 0 && n == 6) {
    if ((a = tile_index(name[5])) >= 0) m->ucnt_row[a] = row;
  } else if (strncmp(name, "beq_", 4) == 0 && n == 8 && name[5] == '_' &&
             name[6] == 'b') {
    a = tile_index(name[4]);
    if (a >= 0 && (name[7] == '0' || name[7] == '1'))
      m->beq_row[a][name[7] - '0'] = row;
  } else if (strncmp(name, "exch_", 5) == 0 && n == 6) {
    if ((a = tile_index(name[5])) >= 0) m->exch_row[a] = row;
  } else if (strncmp(name, "pcnt_", 5) == 0 && n == 6) {
    if ((a = tile_index(name[5])) >= 0) m->pcnt_row[a] = row;
  } else if (strncmp(name, "upair_", 6) == 0 && n == 8) {
    a = tile_index(name[6]), b = tile_index(name[7]);
    if (a >= 0 && b >= 0) m->upair_row[a][b] = row;
  } else {
    fprintf(stderr, "vmodel_hier: unknown feature '%s'\n", name);
  }
}

void vmodel_hier_destroy(VModelHier *m) {
  if (!m) return;
  free(m->a0);
  free(m->a_s);
  free(m->w);
  free(m);
}

#define VMH_LINE_CAP (1 << 16)

VModelHier *vmodel_hier_create(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) {
    fprintf(stderr, "vmodel_hier: cannot open %s\n", path);
    return NULL;
  }
  VModelHier *m = calloc(1, sizeof(VModelHier));
  char *line = malloc(VMH_LINE_CAP);
  bool ok = false;
  int strata_read = 0, rows_read = 0, n_rows = -1;
  m->static_row = -1;
  memset(m->cnt_row, -1, sizeof(m->cnt_row));
  memset(m->pair_row, -1, sizeof(m->pair_row));
  memset(m->pairp_row, -1, sizeof(m->pairp_row));
  memset(m->ucnt_row, -1, sizeof(m->ucnt_row));
  memset(m->beq_row, -1, sizeof(m->beq_row));
  memset(m->exch_row, -1, sizeof(m->exch_row));
  memset(m->pcnt_row, -1, sizeof(m->pcnt_row));
  memset(m->upair_row, -1, sizeof(m->upair_row));
  if (!fgets(line, VMH_LINE_CAP, f) || strncmp(line, "VMH1", 4) != 0) {
    fprintf(stderr, "vmodel_hier: %s: not a VMH1 file\n", path);
    goto done;
  }
  while (fgets(line, VMH_LINE_CAP, f)) {
    char *rest = line, *key = tok(&rest);
    if (!key) continue;
    if (strcmp(key, "end") == 0) {
      ok = true;
      break;
    } else if (strcmp(key, "bag") == 0) {
      if (!tok_int(&rest, &m->bag)) goto done;
    } else if (strcmp(key, "degree") == 0) {
      int d;
      if (!tok_int(&rest, &d) || d != 3) goto done;
    } else if (strcmp(key, "feat_k") == 0) {
      char *t;
      if (!tok_int(&rest, &m->feat_k)) goto done;
      if (!(t = tok(&rest)) || strcmp(t, "int_k") || !tok_int(&rest, &m->int_k))
        goto done;
      if (!(t = tok(&rest)) || strcmp(t, "J") || !tok_int(&rest, &m->J)) goto done;
      if (m->J != m->feat_k + VMH_N_LEN + VMH_N_KIND + VMH_N_TYPE + VMH_N_BLANK +
                      VMH_N_XLEN) {
        fprintf(stderr, "vmodel_hier: J %d does not match this build's layout\n", m->J);
        goto done;
      }
      m->a0 = calloc(m->int_k, sizeof(double));
    } else if (strcmp(key, "groups") == 0) {
      char *t;
      while ((t = tok(&rest))) {
        if (!strcmp(t, "len")) m->g_len = true;
        else if (!strcmp(t, "kind")) m->g_kind = true;
        else if (!strcmp(t, "type")) m->g_type = true;
        else if (!strcmp(t, "blank")) m->g_blank = true;
        else if (!strcmp(t, "xlen")) m->g_xlen = true;
        else { fprintf(stderr, "vmodel_hier: unknown group %s\n", t); goto done; }
      }
    } else if (strcmp(key, "opts") == 0) {
      char *t;
      int v;
      while ((t = tok(&rest))) {
        if (!tok_int(&rest, &v)) goto done;
        if (!strcmp(t, "l2_counts")) m->l2_counts = v;
        else if (!strcmp(t, "blank_types")) m->blank_types = v;
        else if (!strcmp(t, "blank_mixed")) m->blank_mixed = v;
        else if (!strcmp(t, "pass_unseen")) m->pass_unseen = v;
        else { fprintf(stderr, "vmodel_hier: unknown opt %s\n", t); goto done; }
      }
    } else if (strcmp(key, "type_lengths") == 0 || strcmp(key, "type_kinds") == 0) {
      int n, v;
      bool len = key[5] == 'l';
      if (!tok_int(&rest, &n)) goto done;
      for (int i = 0; i < n; i++) {
        if (!tok_int(&rest, &v)) goto done;
        if (len && v >= 0 && v < VMH_N_LEN) m->type_len[v] = true;
        if (!len && v >= 0 && v < VMH_N_KIND) m->type_kind[v] = true;
      }
    } else if (strcmp(key, "knots_f") == 0 || strcmp(key, "knots_i") == 0) {
      bool fk = key[6] == 'f';
      int n;
      if (!tok_int(&rest, &n) || n > VMH_MAX_KNOTS) goto done;
      if (!read_doubles(&rest, fk ? m->knots_f : m->knots_i, n)) goto done;
      if (fk) m->n_kf = n; else m->n_ki = n;
    } else if (strcmp(key, "a0") == 0) {
      if (!m->a0 || !read_doubles(&rest, m->a0, m->int_k)) goto done;
    } else if (strcmp(key, "n_strata") == 0) {
      if (!tok_int(&rest, &m->n_strata) || m->n_strata > VMH_MAX_STRATA) goto done;
      m->a_s = calloc((size_t)m->n_strata * m->int_k, sizeof(double));
    } else if (strcmp(key, "stratum") == 0) {
      char *t = tok(&rest);
      if (!t || strata_read >= m->n_strata || strlen(t) >= 24) goto done;
      strcpy(m->strata_key[strata_read], t);
      if (!read_doubles(&rest, m->a_s + (size_t)strata_read * m->int_k, m->int_k))
        goto done;
      strata_read++;
    } else if (strcmp(key, "n_feat") == 0) {
      if (!tok_int(&rest, &n_rows)) goto done;
      m->n_rows = n_rows;
      m->w = calloc((size_t)n_rows * m->J, sizeof(double));
    } else if (strcmp(key, "feat") == 0) {
      char *name = tok(&rest);
      if (!name || !m->w || rows_read >= n_rows) goto done;
      if (!read_doubles(&rest, m->w + (size_t)rows_read * m->J, m->J)) goto done;
      map_feature(m, name, rows_read);
      rows_read++;
    } else {
      fprintf(stderr, "vmodel_hier: unknown record '%s'\n", key);
      goto done;
    }
  }
  if (ok && (strata_read != m->n_strata || rows_read != n_rows ||
             m->n_kf - 4 != m->feat_k || m->n_ki - 4 != m->int_k)) {
    fprintf(stderr, "vmodel_hier: %s: incomplete (strata %d/%d, rows %d/%d)\n", path,
            strata_read, m->n_strata, rows_read, n_rows);
    ok = false;
  }
done:
  free(line);
  fclose(f);
  if (!ok) {
    fprintf(stderr, "vmodel_hier: failed to load %s\n", path);
    vmodel_hier_destroy(m);
    return NULL;
  }
  return m;
}

// ------------------------------------------------------------------ basis
// Clamped cubic B-spline basis at x (hier_model.basis): x clamped to
// [t[3], t[n-4] - 1e-9 * max(1, |t[n-4]|)], span = largest i in [3, nb-1] with
// t[i] <= x, Cox-de Boor; values rounded to float32 as Python stores them.
// out[nb] zeroed except the 4 active entries; returns the first active index.
static int bspline_basis(const double *t, int n_knots, double x, double *out) {
  const int p = 3, nb = n_knots - p - 1;
  double lo = t[p], hi = t[n_knots - p - 1];
  double cap = hi - 1e-9 * (fabs(hi) > 1.0 ? fabs(hi) : 1.0);
  if (x < lo) x = lo;
  if (x > cap) x = cap;
  int i = p;
  while (i + 1 < nb && t[i + 1] <= x) i++;
  double N[4], left[4], right[4];
  N[0] = 1.0;
  for (int j = 1; j <= p; j++) {
    left[j] = x - t[i + 1 - j];
    right[j] = t[i + j] - x;
    double saved = 0.0;
    for (int r = 0; r < j; r++) {
      double temp = N[r] / (right[r + 1] + left[j - r]);
      N[r] = saved + right[r + 1] * temp;
      saved = left[j - r] * temp;
    }
    N[j] = saved;
  }
  for (int k = 0; k < nb; k++) out[k] = 0.0;
  for (int k = 0; k <= p; k++) out[i - p + k] = (double)(float)N[k];
  return i - p;
}

// ------------------------------------------------------------------ typing
static int classify(const uint8_t leave[VMH_NT], int len) {
  if (len < 3) return T_ALL;
  int vc = 0, cc = 0;
  for (int t = 1; t < VMH_NT; t++) {
    if (IS_VOWEL[t]) vc += leave[t];
    else cc += leave[t];
  }
  if (vc == 0) return T_CONS;
  if (cc == 0) return T_VOWEL;
  return T_MIXED;
}

int vmodel_hier_leave_type(const VModelHier *m, const uint8_t leave[VMH_NT],
                           int kind) {
  int len = 0;
  for (int t = 0; t < VMH_NT; t++) len += leave[t];
  if (kind < 0 || kind >= VMH_N_KIND || len >= VMH_N_LEN || !m->type_kind[kind] ||
      !m->type_len[len])
    return T_ALL;
  bool has_blank = leave[0] > 0;
  if (m->blank_mixed) return has_blank ? T_MIXED : classify(leave, len);
  if (m->blank_types) {
    int t = classify(leave, len);
    if (has_blank && t != T_ALL) return t + 3;   // cons -> cons?, ...
    return t;
  }
  if (len - leave[0] < 2) return T_ALL;
  return classify(leave, len);
}

// ------------------------------------------------------------------ predict
static void stratum_key(char *out, int kind, int len, int typ, bool use_type) {
  snprintf(out, 24, "K%d_L%d_%s", kind, len, use_type ? TYPE_NAME[typ] : "all");
}

static inline double feat_coef(const VModelHier *m, int row, const int *sel_idx,
                               const double *sel_val, int n_sel) {
  const double *w = m->w + (size_t)row * m->J;
  double c = 0.0;
  for (int k = 0; k < n_sel; k++) c += w[sel_idx[k]] * sel_val[k];
  return c;
}

double vmodel_hier_predict(const VModelHier *m, const uint8_t rack[VMH_NT],
                           const uint8_t leave[VMH_NT], int kind, int diff,
                           const int unseen[VMH_NT], float static_leave) {
  int len = 0;
  for (int t = 0; t < VMH_NT; t++) len += leave[t];
  int typ = vmodel_hier_leave_type(m, leave, kind);
  int hb = leave[0] > 0 ? 1 : 0;

  // selector: nonzero entries only (<= 4 + 5)
  double bf[VMH_MAX_KNOTS], bi[VMH_MAX_KNOTS];
  int b0 = bspline_basis(m->knots_f, m->n_kf, (double)diff, bf);
  int sel_idx[16];
  double sel_val[16];
  int n_sel = 0;
  for (int k = b0; k < b0 + 4; k++) {
    if (bf[k] != 0.0) { sel_idx[n_sel] = k; sel_val[n_sel++] = bf[k]; }
  }
  const int K = m->feat_k, o_kind = K + VMH_N_LEN, o_type = o_kind + VMH_N_KIND,
            o_blank = o_type + VMH_N_TYPE, o_xlen = o_blank + VMH_N_BLANK;
  if (m->g_len) { sel_idx[n_sel] = K + len; sel_val[n_sel++] = 1.0; }
  if (m->g_kind) { sel_idx[n_sel] = o_kind + kind; sel_val[n_sel++] = 1.0; }
  if (m->g_type) { sel_idx[n_sel] = o_type + typ; sel_val[n_sel++] = 1.0; }
  if (m->g_blank) { sel_idx[n_sel] = o_blank + hb; sel_val[n_sel++] = 1.0; }
  if (m->g_xlen && kind == 1) { sel_idx[n_sel] = o_xlen + len; sel_val[n_sel++] = 1.0; }

  double z = 0.0;
#define ADD(row, x)                                                       \
  do {                                                                    \
    int r_ = (row);                                                       \
    if (r_ >= 0) z += (double)(x) * feat_coef(m, r_, sel_idx, sel_val, n_sel); \
  } while (0)

  // static leave value (float32, as the design stores it)
  if (len > 0) ADD(m->static_row, (double)(float)static_leave);
  // kept-tile counts: L1, L3+, and L2 with l2_counts
  if (len == 1 || len >= 3 || (len == 2 && m->l2_counts)) {
    for (int t = 0; t < VMH_NT; t++)
      if (leave[t]) ADD(m->cnt_row[t], leave[t]);
  }
  // pair counts + presence
  if (len >= 2) {
    for (int a = 0; a < VMH_NT; a++) {
      if (!leave[a]) continue;
      int na = leave[a];
      if (na >= 2) {
        ADD(m->pair_row[a][a], na * (na - 1) / 2.0);
        ADD(m->pairp_row[a][a], 1.0);
      }
      for (int b = a + 1; b < VMH_NT; b++) {
        if (!leave[b]) continue;
        ADD(m->pair_row[a][b], (double)(na * leave[b]));
        ADD(m->pairp_row[a][b], 1.0);
      }
    }
  }
  // unseen block: draw block for plays / exchanges, pass block for the pass
  if (!(kind == 0 && len == 7)) {
    for (int t = 0; t < VMH_NT; t++) {
      int u = unseen[t];
      if (TILE_BAG[t] != 2) {
        if (u) ADD(m->ucnt_row[t], u);
      } else if (u == 0) {
        ADD(m->beq_row[t][0], 1.0);
      } else if (u == 1) {
        ADD(m->beq_row[t][1], 1.0);
      }
    }
  } else if (m->pass_unseen >= 1) {
    for (int t = 0; t < VMH_NT; t++)
      if (unseen[t]) ADD(m->pcnt_row[t], unseen[t]);
    if (m->pass_unseen >= 2) {
      for (int i = 0; i < VMH_NT; i++) {
        for (int j = i; j < VMH_NT; j++) {
          if (i == j && TILE_BAG[i] < 2) continue;
          int v = (i == j) ? unseen[i] * (unseen[i] - 1) / 2 : unseen[i] * unseen[j];
          if (v) ADD(m->upair_row[i][j], v);
        }
      }
    }
  }
  // exchanged tiles
  if (kind == 1) {
    for (int t = 0; t < VMH_NT; t++) {
      int d = (int)rack[t] - (int)leave[t];
      if (d) ADD(m->exch_row[t], d);
    }
  }
#undef ADD

  // intercept curve: shared + this stratum's deviation
  int c0 = bspline_basis(m->knots_i, m->n_ki, (double)diff, bi);
  char key[24];
  stratum_key(key, kind, len, typ, m->g_type);
  const double *as = NULL;
  for (int s = 0; s < m->n_strata; s++) {
    if (strcmp(m->strata_key[s], key) == 0) {
      as = m->a_s + (size_t)s * m->int_k;
      break;
    }
  }
  for (int k = c0; k < c0 + 4; k++) {
    z += bi[k] * m->a0[k];
    if (as) z += bi[k] * as[k];
  }
  if (z > 40.0) z = 40.0;
  if (z < -40.0) z = -40.0;
  return 1.0 / (1.0 + exp(-z));
}
