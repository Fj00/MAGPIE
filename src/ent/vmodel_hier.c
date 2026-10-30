#include "vmodel_hier.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Mirrors hier_model.py (B repo bots/winpct/scripts). Every rule here has a
// line there; keep them in step. tools/vmh_check.c compares the two on real
// moves.

#define VMH_J_MAX 64

static const uint8_t TILE_BAG[VMH_NT] = {2, 9, 2, 2, 4, 12, 2, 3, 2, 9, 1, 1, 4, 2,
                                         6, 8, 2, 1, 6, 4, 6, 4, 2, 2, 1, 2, 1};
// tile points, same alphabet (blank 0) -- for sixpass_diff
static const uint8_t TILE_VALUE[VMH_NT] = {0, 1, 3, 3, 2, 1, 4, 2, 4, 1, 8, 5, 1, 3,
                                           1, 1, 3, 10, 1, 1, 1, 1, 4, 4, 8, 4, 10};
// AEIOUY (python VOWELS); blank is neither
static const bool IS_VOWEL[VMH_NT] = {
    false, true,  false, false, false, true,  false, false, false, true,
    false, false, false, false, false, true,  false, false, false, false,
    false, true,  false, false, false, true,  false};

enum { T_ALL = 0, T_CONS, T_MIXED, T_VOWEL, T_CONSB, T_MIXEDB, T_VOWELB };
static const char *TYPE_NAME[VMH_N_TYPE] = {"all",   "cons",   "mixed", "vowel",
                                           "cons?", "mixed?", "vowel?"};

static void stratum_key(char *out, int kind, int len, int typ, bool use_type) {
  snprintf(out, 24, "K%d_L%d_%s", kind, len, use_type ? TYPE_NAME[typ] : "all");
}

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
  } else if (strncmp(name, "cstep_", 6) == 0 && n == 9 && name[7] == '_') {
    a = tile_index(name[6]);
    b = name[8] - '0';
    if (a >= 0 && b >= 1 && b <= 3) m->cstep_row[a][b] = row;
  } else if (strncmp(name, "cge_", 4) == 0 && n == 7 && name[5] == '_') {
    a = tile_index(name[4]);
    b = name[6] - '0';
    if (a >= 0 && b >= 1 && b <= 4) m->cge_row[a][b] = row;
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
  memset(m->cstep_row, -1, sizeof(m->cstep_row));
  memset(m->cge_row, -1, sizeof(m->cge_row));
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
    } else if (strcmp(key, "turn") == 0) {
      if (!tok_int(&rest, &m->turn)) goto done;
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
                      VMH_N_XLEN || m->J > VMH_J_MAX) {
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
        else if (!strcmp(t, "pass_sixpass")) m->pass_sixpass = v;
        else if (!strcmp(t, "count_steps")) m->count_steps = v;
        else if (!strcmp(t, "count_cum")) m->count_cum = v;
        else if (!strcmp(t, "play_unseen_pairs")) m->play_unseen_pairs = v;
        else if (!strcmp(t, "int_tails")) m->int_tails = v;
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
             m->n_kf - 4 != m->feat_k ||
             m->n_ki - 4 + (m->int_tails ? VMH_N_TAIL : 0) != m->int_k)) {
    fprintf(stderr, "vmodel_hier: %s: incomplete (strata %d/%d, rows %d/%d)\n", path,
            strata_read, m->n_strata, rows_read, n_rows);
    ok = false;
  }
  if (ok) {   // stratum curve per (kind, len, type): no string compares per move
    char key[24];
    for (int k = 0; k < VMH_N_KIND; k++)
      for (int l = 0; l < VMH_N_LEN; l++)
        for (int t = 0; t < VMH_N_TYPE; t++) {
          stratum_key(key, k, l, t, m->g_type);
          m->sidx[k][l][t] = -1;
          for (int s = 0; s < m->n_strata; s++)
            if (strcmp(m->strata_key[s], key) == 0) { m->sidx[k][l][t] = (int16_t)s; break; }
        }
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
// Writes the 4 active values to out4; returns the first active basis index.
static int bspline_basis4(const double *t, int n_knots, double x, double out4[4]) {
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
  for (int k = 0; k <= p; k++) out4[k] = (double)(float)N[k];
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

int vmodel_hier_sixpass_diff(const uint8_t rack[VMH_NT], const int unseen[VMH_NT],
                             int lead) {
  int mover = 0;
  for (int t = 0; t < VMH_NT; t++) mover += rack[t] * TILE_VALUE[t];
  int need = 7, add = 0;
  for (int v = 0; v <= 10 && need > 0; v++) {
    for (int t = 0; t < VMH_NT && need > 0; t++) {
      if (TILE_VALUE[t] != v) continue;
      int c = unseen[t];
      if (c <= 0) continue;
      int take = c < need ? c : need;
      add += take * v;
      need -= take;
    }
  }
  return lead - mover + add;
}

// ------------------------------------------------------------------ feature blocks
// Each block adds sum_f x_f * W[f] (all J columns) into vec. The prediction is
// then (sum of the move's blocks) . Sel + intercept curve; the turn cache keeps
// the per-turn and per-leave blocks.
static inline void axpy(const VModelHier *m, int row, double x, double *vec) {
  if (row < 0 || x == 0.0) return;
  const double *w = m->w + (size_t)row * m->J;
  for (int j = 0; j < m->J; j++) vec[j] += x * w[j];
}

// kept-tile counts (L1, L3+, L2 with l2_counts) and pair counts + presence
static void block_leave(const VModelHier *m, const uint8_t leave[VMH_NT], int len,
                        double *vec) {
  const bool counts_on = len == 1 || len >= 3 || (len == 2 && m->l2_counts);
  if (counts_on) {
    for (int t = 0; t < VMH_NT; t++) {
      const int n = leave[t];
      if (!n) continue;
      if (m->count_cum) {                    // EEEE: >=1, >=2, >=3, >=4 all on
        for (int k = 1; k <= (n < 4 ? n : 4); k++) axpy(m, m->cge_row[t][k], 1.0, vec);
      } else if (m->count_steps) {
        axpy(m, m->cstep_row[t][n < 3 ? n : 3], 1.0, vec);
      } else {
        axpy(m, m->cnt_row[t], n, vec);
      }
    }
  }
  // Pairs. With count_steps / count_cum the product columns of repeatable pairs
  // were never trained (zero weight, not exported), and the others equal presence,
  // so emitting product + presence reproduces Python without its pair lists.
  if (len >= 2) {
    for (int a = 0; a < VMH_NT; a++) {
      if (!leave[a]) continue;
      int na = leave[a];
      if (na >= 2 && !(m->count_cum && counts_on)) {
        axpy(m, m->pair_row[a][a], na * (na - 1) / 2.0, vec);
        axpy(m, m->pairp_row[a][a], 1.0, vec);
      }
      for (int b = a + 1; b < VMH_NT; b++) {
        if (!leave[b]) continue;
        axpy(m, m->pair_row[a][b], (double)(na * leave[b]), vec);
        axpy(m, m->pairp_row[a][b], 1.0, vec);
      }
    }
  }
}

// draw block (plays / exchanges): unseen counts and bag_eq indicators
static void block_unseen(const VModelHier *m, const int unseen[VMH_NT], double *vec) {
  for (int t = 0; t < VMH_NT; t++) {
    int u = unseen[t];
    if (TILE_BAG[t] != 2) {
      if (u) axpy(m, m->ucnt_row[t], u, vec);
    } else if (u == 0) {
      axpy(m, m->beq_row[t][0], 1.0, vec);
    } else if (u == 1) {
      axpy(m, m->beq_row[t][1], 1.0, vec);
    }
  }
  if (m->play_unseen_pairs) {          // bags 8-9: unseen pair counts (upair_ rows)
    for (int i = 0; i < VMH_NT; i++) {
      for (int j = i; j < VMH_NT; j++) {
        if (i == j && TILE_BAG[i] < 2) continue;
        int v = (i == j) ? unseen[i] * (unseen[i] - 1) / 2 : unseen[i] * unseen[j];
        if (v) axpy(m, m->upair_row[i][j], v, vec);
      }
    }
  }
}

// the pass's own unseen block (pass_unseen 1: counts, 2: + pair counts)
static void block_pass(const VModelHier *m, const int unseen[VMH_NT], double *vec) {
  if (m->pass_unseen < 1) return;
  for (int t = 0; t < VMH_NT; t++)
    if (unseen[t]) axpy(m, m->pcnt_row[t], unseen[t], vec);
  if (m->pass_unseen < 2) return;
  for (int i = 0; i < VMH_NT; i++) {
    for (int j = i; j < VMH_NT; j++) {
      if (i == j && TILE_BAG[i] < 2) continue;
      int v = (i == j) ? unseen[i] * (unseen[i] - 1) / 2 : unseen[i] * unseen[j];
      if (v) axpy(m, m->upair_row[i][j], v, vec);
    }
  }
}

// exchanged tiles (rack - leave)
static void block_exch(const VModelHier *m, const uint8_t rack[VMH_NT],
                       const uint8_t leave[VMH_NT], double *vec) {
  for (int t = 0; t < VMH_NT; t++) {
    int d = (int)rack[t] - (int)leave[t];
    if (d) axpy(m, m->exch_row[t], d, vec);
  }
}

// Sel . vec + intercept curve -> win probability
static double finish(const VModelHier *m, const double *vec, int len, int kind,
                     int typ, int hb, int diff) {
  double bf[4], bi[4];
  int b0 = bspline_basis4(m->knots_f, m->n_kf, (double)diff, bf);
  double z = 0.0;
  for (int k = 0; k < 4; k++) z += bf[k] * vec[b0 + k];
  const int K = m->feat_k, o_kind = K + VMH_N_LEN, o_type = o_kind + VMH_N_KIND,
            o_blank = o_type + VMH_N_TYPE, o_xlen = o_blank + VMH_N_BLANK;
  if (m->g_len) z += vec[K + len];
  if (m->g_kind) z += vec[o_kind + kind];
  if (m->g_type) z += vec[o_type + typ];
  if (m->g_blank) z += vec[o_blank + hb];
  if (m->g_xlen && kind == 1) z += vec[o_xlen + len];
  int c0 = bspline_basis4(m->knots_i, m->n_ki, (double)diff, bi);
  const int si = (len < VMH_N_LEN) ? m->sidx[kind][len][typ] : -1;
  const double *as = si >= 0 ? m->a_s + (size_t)si * m->int_k : NULL;
  for (int k = 0; k < 4; k++) {
    z += bi[k] * m->a0[c0 + k];
    if (as) z += bi[k] * as[c0 + k];
  }
  if (m->int_tails) {
    // Linear tails beyond the knot range (hier_model.int_basis, 2026-10-30): the
    // spline is constant outside [lo, hi], so without these every lead past the
    // 0.2% tails scored the same.
    const int nb = m->n_ki - 4;
    const double lo = m->knots_i[3], hi = m->knots_i[m->n_ki - 4];
    const double tl = (diff < lo ? (double)diff - lo : 0.0) / VMH_TAIL_SCALE;
    const double th = (diff > hi ? (double)diff - hi : 0.0) / VMH_TAIL_SCALE;
    z += m->a0[nb] * tl + m->a0[nb + 1] * th;
    if (as) z += as[nb] * tl + as[nb + 1] * th;
  }
  if (z > 40.0) z = 40.0;
  if (z < -40.0) z = -40.0;
  return 1.0 / (1.0 + exp(-z));
}

double vmodel_hier_predict(const VModelHier *m, const uint8_t rack[VMH_NT],
                           const uint8_t leave[VMH_NT], int kind, int diff,
                           const int unseen[VMH_NT], float static_leave) {
  int len = 0;
  for (int t = 0; t < VMH_NT; t++) len += leave[t];
  double vec[VMH_J_MAX] = {0};
  if (len > 0) axpy(m, m->static_row, (double)(float)static_leave, vec);
  block_leave(m, leave, len, vec);
  if (!(kind == 0 && len == 7)) block_unseen(m, unseen, vec);
  else block_pass(m, unseen, vec);
  if (kind == 1) block_exch(m, rack, leave, vec);
  return finish(m, vec, len, kind, vmodel_hier_leave_type(m, leave, kind),
                leave[0] > 0 ? 1 : 0, diff);
}

// ------------------------------------------------------------------ turn cache
#define VMH_SLOTS 512   // power of two; a rack has <= 127 leaves

typedef struct {
  uint64_t key;          // 0 = empty
  uint8_t len, hb, typ_play, typ_exch, typ_pass;
  bool has_exch;
  double leave_vec[VMH_J_MAX];   // static + counts + pairs
  double exch_vec[VMH_J_MAX];    // exchanged tiles (filled on first exchange use)
} VmhSlot;

struct VmhTurnCache {
  const VModelHier *m;
  uint8_t rack[VMH_NT];
  int unseen[VMH_NT];
  VmhStaticFn static_fn;
  void *static_ctx;
  double unseen_vec[VMH_J_MAX];  // draw block, plays / exchanges
  double pass_vec[VMH_J_MAX];    // pass block
  VmhSlot *slots;
};

VmhTurnCache *vmh_turn_cache_create(void) {
  VmhTurnCache *c = calloc(1, sizeof(VmhTurnCache));
  c->slots = calloc(VMH_SLOTS, sizeof(VmhSlot));
  return c;
}

void vmh_turn_cache_destroy(VmhTurnCache *c) {
  if (!c) return;
  free(c->slots);
  free(c);
}

void vmh_turn_begin(VmhTurnCache *c, const VModelHier *m, const uint8_t *rack_idx,
                    int rack_len, const int unseen[VMH_NT], VmhStaticFn static_fn,
                    void *static_ctx) {
  c->m = m;
  memset(c->rack, 0, sizeof(c->rack));
  for (int i = 0; i < rack_len; i++) c->rack[rack_idx[i]]++;
  memcpy(c->unseen, unseen, sizeof(c->unseen));
  c->static_fn = static_fn;
  c->static_ctx = static_ctx;
  memset(c->unseen_vec, 0, sizeof(c->unseen_vec));
  memset(c->pass_vec, 0, sizeof(c->pass_vec));
  block_unseen(m, c->unseen, c->unseen_vec);
  block_pass(m, c->unseen, c->pass_vec);
  for (int s = 0; s < VMH_SLOTS; s++) c->slots[s].key = 0;
}

static uint64_t leave_key(const uint8_t *idx, int len) {
  uint64_t k = (uint64_t)len + 1;          // never 0
  for (int i = 0; i < len; i++) k = (k << 5) | idx[i];
  return k;
}

static VmhSlot *leave_slot(VmhTurnCache *c, const uint8_t *idx, int len) {
  const uint64_t key = leave_key(idx, len);
  uint64_t h = key * 0x9E3779B97F4A7C15ULL;
  for (int probe = 0; probe < VMH_SLOTS; probe++) {
    VmhSlot *s = &c->slots[((h >> 55) + probe) & (VMH_SLOTS - 1)];
    if (s->key == key) return s;
    if (s->key == 0) {
      const VModelHier *m = c->m;
      uint8_t leave[VMH_NT] = {0};
      for (int i = 0; i < len; i++) leave[idx[i]]++;
      s->key = key;
      s->len = (uint8_t)len;
      s->hb = leave[0] > 0;
      s->typ_play = (uint8_t)vmodel_hier_leave_type(m, leave, 2);
      s->typ_exch = (uint8_t)vmodel_hier_leave_type(m, leave, 1);
      s->typ_pass = (uint8_t)vmodel_hier_leave_type(m, leave, 0);
      s->has_exch = false;
      memset(s->leave_vec, 0, sizeof(s->leave_vec));
      if (len > 0) {
        const float st = c->static_fn ? c->static_fn(idx, len, c->static_ctx) : 0.0f;
        axpy(m, m->static_row, (double)(float)st, s->leave_vec);
      }
      block_leave(m, leave, len, s->leave_vec);
      return s;
    }
  }
  return NULL;   // table full: cannot happen for a 7-tile rack
}

double vmh_turn_predict(VmhTurnCache *c, const uint8_t *leave_idx, int leave_len,
                        int kind, int diff) {
  const VModelHier *m = c->m;
  VmhSlot *s = leave_slot(c, leave_idx, leave_len);
  if (!s) return -1.0;
  double vec[VMH_J_MAX];
  const double *blk = (kind == 0 && leave_len == 7) ? c->pass_vec : c->unseen_vec;
  for (int j = 0; j < m->J; j++) vec[j] = s->leave_vec[j] + blk[j];
  if (kind == 1) {
    if (!s->has_exch) {
      uint8_t leave[VMH_NT] = {0};
      for (int i = 0; i < leave_len; i++) leave[leave_idx[i]]++;
      memset(s->exch_vec, 0, sizeof(s->exch_vec));
      block_exch(m, c->rack, leave, s->exch_vec);
      s->has_exch = true;
    }
    for (int j = 0; j < m->J; j++) vec[j] += s->exch_vec[j];
  }
  const int typ = kind == 2 ? s->typ_play : kind == 1 ? s->typ_exch : s->typ_pass;
  return finish(m, vec, leave_len, kind, typ, s->hb, diff);
}
