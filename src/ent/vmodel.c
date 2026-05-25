#include "vmodel.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../util/io_util.h"

// V-model loader for the flat-text `.vmt` format produced by
// bots/winpct/scripts/export_v_model_text.py.
//
// File grammar (whitespace-separated tokens, # comments and blank lines
// allowed):
//
//   V <version>
//   TURN <int>
//   BAG <int>
//   NSTRATA <int>
//
//   S <stratum_key> <kind> <leave_length> <type>
//   T <n_tiles> <tile1> ... <tilen>
//   P <n_pairs> <pair1> ... <pairn>
//   NBUCKETS <int>
//   B <diff_lo> <diff_hi> <has_diff>
//   I <intercept>
//   C <n_coefs> <c1> ... <cn>
//   [PTS <n> <p1> ...]   (optional)
//   [DRAWN <n> <d1> ...] (optional)
//   [BAG <n> <t1> <c1> <t2> <c2> ...]  (optional)

#define MAX_LINE 65536  // some coef rows are wide (one stratum has 428 floats)

static char *next_token(char **rest) {
    while (**rest && isspace((unsigned char)**rest)) (*rest)++;
    if (**rest == '\0') return NULL;
    char *tok = *rest;
    while (**rest && !isspace((unsigned char)**rest)) (*rest)++;
    if (**rest) {
        **rest = '\0';
        (*rest)++;
    }
    return tok;
}

int vmodel_tile_index_from_char(char c) {
    if (c == '?') return 0;
    if (c >= 'A' && c <= 'Z') return 1 + (c - 'A');
    return -1;
}

static int parse_type_str(const char *s) {
    if (strcmp(s, "all") == 0) return VMODEL_TYPE_ALL;
    if (strcmp(s, "cons") == 0) return VMODEL_TYPE_CONS;
    if (strcmp(s, "mixed") == 0) return VMODEL_TYPE_MIXED;
    if (strcmp(s, "vowel") == 0) return VMODEL_TYPE_VOWEL;
    return -1;
}

// Read next non-blank, non-comment line into `buf`. Returns false at EOF.
static bool read_line(char *buf, size_t cap, FILE *f) {
    while (fgets(buf, cap, f)) {
        char *p = buf;
        while (*p && isspace((unsigned char)*p)) p++;
        if (*p == '\0' || *p == '#') continue;
        return true;
    }
    return false;
}

// Parse helpers — return false on error (caller logs).
static bool expect_token(char **rest, const char *want) {
    char *t = next_token(rest);
    return t && strcmp(t, want) == 0;
}

static bool parse_int(char **rest, int *out) {
    char *t = next_token(rest);
    if (!t) return false;
    char *end = NULL;
    long v = strtol(t, &end, 10);
    if (end == t) return false;
    *out = (int)v;
    return true;
}

static bool parse_float(char **rest, float *out) {
    char *t = next_token(rest);
    if (!t) return false;
    char *end = NULL;
    double v = strtod(t, &end);
    if (end == t) return false;
    *out = (float)v;
    return true;
}

static bool parse_double(char **rest, double *out) {
    char *t = next_token(rest);
    if (!t) return false;
    char *end = NULL;
    double v = strtod(t, &end);
    if (end == t) return false;
    *out = v;
    return true;
}

static bool parse_str(char **rest, char *out, size_t cap) {
    char *t = next_token(rest);
    if (!t) return false;
    size_t n = strlen(t);
    if (n + 1 > cap) return false;
    memcpy(out, t, n + 1);
    return true;
}

static void free_bucket(VBucket *b) {
    free(b->coefs);
    free(b->pts);
    free(b->drawn);
    free(b->bag_tiles);
    free(b->bag_counts);
}

static void free_stratum(VStratum *s) {
    free(s->pairs);
    if (s->buckets) {
        for (int i = 0; i < s->n_buckets; i++) {
            free_bucket(&s->buckets[i]);
        }
        free(s->buckets);
    }
}

void vmodel_destroy(VModel *m) {
    if (!m) return;
    if (m->strata) {
        for (int i = 0; i < m->n_strata; i++) {
            free_stratum(&m->strata[i]);
        }
        free(m->strata);
    }
    free(m);
}

static bool parse_header(FILE *f, VModel *m) {
    char buf[MAX_LINE];
    char *p;

    if (!read_line(buf, sizeof(buf), f)) return false;
    p = buf;
    if (!expect_token(&p, "V") || !parse_int(&p, &m->version)) return false;

    if (!read_line(buf, sizeof(buf), f)) return false;
    p = buf;
    if (!expect_token(&p, "TURN") || !parse_int(&p, &m->turn)) return false;

    if (!read_line(buf, sizeof(buf), f)) return false;
    p = buf;
    if (!expect_token(&p, "BAG") || !parse_int(&p, &m->bag)) return false;

    if (!read_line(buf, sizeof(buf), f)) return false;
    p = buf;
    if (!expect_token(&p, "NSTRATA") || !parse_int(&p, &m->n_strata)) return false;
    return true;
}

// Returns true on success. On error, prints log_warn and returns false;
// caller frees partially-built stratum.
static bool parse_stratum(FILE *f, VStratum *s) {
    char buf[MAX_LINE];
    char *p;

    // S line was already read by caller. Now T, P, NBUCKETS, then each B/I/C/...
    if (!read_line(buf, sizeof(buf), f)) {
        log_warn("vmodel: unexpected EOF before T line"); return false;
    }
    p = buf;
    if (!expect_token(&p, "T")) {
        log_warn("vmodel: expected T line in stratum %s", s->key); return false;
    }
    if (!parse_int(&p, &s->n_tiles) || s->n_tiles > VMODEL_MAX_TILES) {
        log_warn("vmodel: bad n_tiles in %s", s->key); return false;
    }
    for (int i = 0; i < s->n_tiles; i++) {
        char tok[8];
        if (!parse_str(&p, tok, sizeof(tok)) || strlen(tok) != 1) {
            log_warn("vmodel: bad tile in %s", s->key); return false;
        }
        int idx = vmodel_tile_index_from_char(tok[0]);
        if (idx < 0) {
            log_warn("vmodel: bad tile '%s' in %s", tok, s->key); return false;
        }
        s->tiles[i] = (uint8_t)idx;
    }

    if (!read_line(buf, sizeof(buf), f)) {
        log_warn("vmodel: unexpected EOF before P line"); return false;
    }
    p = buf;
    if (!expect_token(&p, "P")) {
        log_warn("vmodel: expected P line in stratum %s", s->key); return false;
    }
    if (!parse_int(&p, &s->n_pairs)) {
        log_warn("vmodel: bad n_pairs in %s", s->key); return false;
    }
    if (s->n_pairs > 0) {
        s->pairs = malloc(sizeof(uint8_t[2]) * s->n_pairs);
        if (!s->pairs) {
            log_warn("vmodel: alloc fail pairs in %s", s->key); return false;
        }
        for (int i = 0; i < s->n_pairs; i++) {
            char tok[8];
            if (!parse_str(&p, tok, sizeof(tok)) || strlen(tok) != 2) {
                log_warn("vmodel: bad pair token in %s", s->key); return false;
            }
            int a = vmodel_tile_index_from_char(tok[0]);
            int b = vmodel_tile_index_from_char(tok[1]);
            if (a < 0 || b < 0) {
                log_warn("vmodel: bad pair '%s' in %s", tok, s->key); return false;
            }
            s->pairs[i][0] = (uint8_t)a;
            s->pairs[i][1] = (uint8_t)b;
        }
    }

    if (!read_line(buf, sizeof(buf), f)) {
        log_warn("vmodel: unexpected EOF before NBUCKETS"); return false;
    }
    p = buf;
    if (!expect_token(&p, "NBUCKETS") || !parse_int(&p, &s->n_buckets)) {
        log_warn("vmodel: bad NBUCKETS in %s", s->key); return false;
    }
    s->buckets = calloc(s->n_buckets, sizeof(VBucket));
    if (!s->buckets) {
        log_warn("vmodel: alloc fail buckets in %s", s->key); return false;
    }

    for (int i = 0; i < s->n_buckets; i++) {
        VBucket *b = &s->buckets[i];

        // B line
        if (!read_line(buf, sizeof(buf), f)) {
            log_warn("vmodel: EOF before B in %s", s->key); return false;
        }
        p = buf;
        int has_diff_int;
        if (!expect_token(&p, "B") ||
            !parse_int(&p, &b->diff_lo) ||
            !parse_int(&p, &b->diff_hi) ||
            !parse_int(&p, &has_diff_int)) {
            log_warn("vmodel: bad B line in %s", s->key); return false;
        }
        b->has_diff = (has_diff_int != 0);

        // I line
        if (!read_line(buf, sizeof(buf), f)) {
            log_warn("vmodel: EOF before I in %s", s->key); return false;
        }
        p = buf;
        if (!expect_token(&p, "I") || !parse_float(&p, &b->intercept)) {
            log_warn("vmodel: bad I line in %s", s->key); return false;
        }

        // C line
        if (!read_line(buf, sizeof(buf), f)) {
            log_warn("vmodel: EOF before C in %s", s->key); return false;
        }
        p = buf;
        if (!expect_token(&p, "C") || !parse_int(&p, &b->n_coefs)) {
            log_warn("vmodel: bad C line in %s", s->key); return false;
        }
        b->coefs = malloc(sizeof(float) * b->n_coefs);
        if (!b->coefs) {
            log_warn("vmodel: alloc fail coefs in %s", s->key); return false;
        }
        for (int j = 0; j < b->n_coefs; j++) {
            if (!parse_float(&p, &b->coefs[j])) {
                log_warn("vmodel: bad coef %d in %s bucket %d", j, s->key, i);
                return false;
            }
        }

        // Optional PTS/DRAWN/BAG records. Peek next non-blank line; if it's
        // one of those, consume it; otherwise leave it for the next B or S.
        long mark = ftell(f);
        while (read_line(buf, sizeof(buf), f)) {
            char *peek = buf;
            char *tok = next_token(&peek);
            if (!tok) break;
            if (strcmp(tok, "PTS") == 0) {
                if (!parse_int(&peek, &b->n_pts)) {
                    log_warn("vmodel: bad PTS n in %s", s->key); return false;
                }
                b->pts = malloc(sizeof(int) * b->n_pts);
                for (int j = 0; j < b->n_pts; j++) {
                    if (!parse_int(&peek, &b->pts[j])) {
                        log_warn("vmodel: bad PTS val in %s", s->key); return false;
                    }
                }
            } else if (strcmp(tok, "DRAWN") == 0) {
                if (!parse_int(&peek, &b->n_drawn)) {
                    log_warn("vmodel: bad DRAWN n in %s", s->key); return false;
                }
                b->drawn = malloc(sizeof(double) * b->n_drawn);
                for (int j = 0; j < b->n_drawn; j++) {
                    if (!parse_double(&peek, &b->drawn[j])) {
                        log_warn("vmodel: bad DRAWN val in %s", s->key); return false;
                    }
                }
            } else if (strcmp(tok, "BAG") == 0) {
                if (!parse_int(&peek, &b->n_bag)) {
                    log_warn("vmodel: bad BAG n in %s", s->key); return false;
                }
                b->bag_tiles = malloc(sizeof(uint8_t) * b->n_bag);
                b->bag_counts = malloc(sizeof(int8_t) * b->n_bag);
                for (int j = 0; j < b->n_bag; j++) {
                    char tok2[8];
                    int cnt;
                    if (!parse_str(&peek, tok2, sizeof(tok2)) ||
                        strlen(tok2) != 1 ||
                        !parse_int(&peek, &cnt)) {
                        log_warn("vmodel: bad BAG entry in %s", s->key); return false;
                    }
                    int idx = vmodel_tile_index_from_char(tok2[0]);
                    if (idx < 0) {
                        log_warn("vmodel: bad BAG tile in %s", s->key); return false;
                    }
                    b->bag_tiles[j] = (uint8_t)idx;
                    b->bag_counts[j] = (int8_t)cnt;
                }
            } else {
                // Not an indicator record — rewind and let the outer loop
                // pick it up. Use fseek with the saved mark.
                fseek(f, mark, SEEK_SET);
                break;
            }
            mark = ftell(f);
        }
    }
    return true;
}

VModel *vmodel_create(const char *vmt_path) {
    FILE *f = fopen(vmt_path, "r");
    if (!f) {
        log_warn("vmodel: cannot open %s", vmt_path);
        return NULL;
    }
    VModel *m = calloc(1, sizeof(VModel));
    if (!m) { fclose(f); return NULL; }

    if (!parse_header(f, m)) {
        log_warn("vmodel: bad header in %s", vmt_path);
        fclose(f); vmodel_destroy(m); return NULL;
    }
    if (m->version != 1) {
        log_warn("vmodel: unsupported version %d in %s", m->version, vmt_path);
        fclose(f); vmodel_destroy(m); return NULL;
    }

    m->strata = calloc(m->n_strata, sizeof(VStratum));
    if (!m->strata) {
        log_warn("vmodel: alloc fail strata");
        fclose(f); vmodel_destroy(m); return NULL;
    }

    char buf[MAX_LINE];
    int s_idx = 0;
    while (read_line(buf, sizeof(buf), f) && s_idx < m->n_strata) {
        char *p = buf;
        if (!expect_token(&p, "S")) {
            log_warn("vmodel: expected S at stratum %d", s_idx);
            fclose(f); vmodel_destroy(m); return NULL;
        }
        VStratum *s = &m->strata[s_idx];
        if (!parse_str(&p, s->key, sizeof(s->key)) ||
            !parse_int(&p, &s->kind) ||
            !parse_int(&p, &s->leave_length)) {
            log_warn("vmodel: bad S line at stratum %d", s_idx);
            fclose(f); vmodel_destroy(m); return NULL;
        }
        char typ_str[16];
        if (!parse_str(&p, typ_str, sizeof(typ_str))) {
            log_warn("vmodel: bad S type at stratum %d", s_idx);
            fclose(f); vmodel_destroy(m); return NULL;
        }
        int t = parse_type_str(typ_str);
        if (t < 0) {
            log_warn("vmodel: unknown type '%s' at stratum %d", typ_str, s_idx);
            fclose(f); vmodel_destroy(m); return NULL;
        }
        s->type = t;
        if (!parse_stratum(f, s)) {
            fclose(f); vmodel_destroy(m); return NULL;
        }
        s_idx++;
    }

    fclose(f);
    if (s_idx != m->n_strata) {
        log_warn("vmodel: expected %d strata, got %d in %s",
                 m->n_strata, s_idx, vmt_path);
        vmodel_destroy(m);
        return NULL;
    }
    log_info("vmodel: loaded %s (%d strata, turn=%d bag=%d)",
             vmt_path, m->n_strata, m->turn, m->bag);
    return m;
}

bool vmodel_terminal_zero_score(int turn, int kind) {
    return turn == 6 && (kind == 0 || kind == 1);
}

VModelLeaveType vmodel_classify_leave(const uint8_t *leave_indices,
                                       int leave_len) {
    // Matches v_model_predict.classify_type: lengths<3 → "all"; else count
    // vowels/consonants excluding blanks. AEIOUY are vowels.
    if (leave_len < 3) return VMODEL_TYPE_ALL;
    int vc = 0, cc = 0;
    for (int i = 0; i < leave_len; i++) {
        uint8_t t = leave_indices[i];
        if (t == 0) continue;  // blank
        char c = 'A' + (t - 1);
        if (c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U' || c == 'Y') {
            vc++;
        } else {
            cc++;
        }
    }
    if (vc == 0) return VMODEL_TYPE_CONS;
    if (cc == 0) return VMODEL_TYPE_VOWEL;
    return VMODEL_TYPE_MIXED;
}

int vmodel_stratum_index(const VModel *m, int kind, int leave_length,
                          VModelLeaveType type, int turn) {
    // Build the stratum_key matching v_model_predict.stratum_key() and
    // train_v_model.stratum_iter():
    //   - L=0,1,2,7 → always "all" (trainer never splits these by type)
    //   - L=3..6: cons/mixed/vowel, unless terminal-zero collapses to "all"
    char key[VMODEL_KEY_MAX];
    const char *typ_str;
    bool tzs = vmodel_terminal_zero_score(turn, kind);
    if (leave_length == 0 || leave_length == 1 ||
        leave_length == 2 || leave_length == 7) {
        typ_str = "all";
    } else if (tzs && leave_length >= 3) {
        typ_str = "all";
    } else {
        switch (type) {
            case VMODEL_TYPE_CONS:  typ_str = "cons";  break;
            case VMODEL_TYPE_MIXED: typ_str = "mixed"; break;
            case VMODEL_TYPE_VOWEL: typ_str = "vowel"; break;
            default:                typ_str = "all";   break;
        }
    }
    snprintf(key, sizeof(key), "K%d_L%d_%s", kind, leave_length, typ_str);
    for (int i = 0; i < m->n_strata; i++) {
        if (strcmp(m->strata[i].key, key) == 0) return i;
    }
    return -1;
}

int vmodel_bucket_index(const VStratum *s, int diff) {
    for (int i = 0; i < s->n_buckets; i++) {
        if (s->buckets[i].diff_lo <= diff && diff <= s->buckets[i].diff_hi) {
            return i;
        }
    }
    return -1;
}
