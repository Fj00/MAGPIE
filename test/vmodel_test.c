#include "../src/ent/static_leaves.h"
#include "../src/ent/vmodel.h"
#include "../src/ent/vmodel_features.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Minimal smoke test for the .vmt text-format loader. Reads a fixed test
// fixture from the env var MAGPIE_VMODEL_TEST_PATH (so the loader can be
// pointed at a regenerated model JSON without re-baking test data into the
// repo). Skipped silently when the env var isn't set.

static void check_stratum_lookup(const VModel *m) {
    // Terminal-zero K1 with L>=3 must collapse to "all" stratum.
    int idx = vmodel_stratum_index(m, 1, 4, VMODEL_TYPE_VOWEL, 6);
    assert(idx >= 0);
    assert(strcmp(m->strata[idx].key, "K1_L4_all") == 0);

    // Non-terminal-zero K2 must NOT collapse.
    idx = vmodel_stratum_index(m, 2, 4, VMODEL_TYPE_VOWEL, 6);
    if (idx >= 0) {
        assert(strcmp(m->strata[idx].key, "K2_L4_vowel") == 0);
    }

    // K0 pass at T6 — leave_length=7 has TZS collapse to "all".
    idx = vmodel_stratum_index(m, 0, 7, VMODEL_TYPE_MIXED, 6);
    if (idx >= 0) {
        assert(strcmp(m->strata[idx].key, "K0_L7_all") == 0);
    }
}

static void check_classify(void) {
    // Empty / short leaves → "all".
    assert(vmodel_classify_leave((const uint8_t[]){}, 0) == VMODEL_TYPE_ALL);
    assert(vmodel_classify_leave((const uint8_t[]){1}, 1) == VMODEL_TYPE_ALL);
    assert(vmodel_classify_leave((const uint8_t[]){1, 2}, 2) == VMODEL_TYPE_ALL);

    // A, E, I = all vowels → "vowel".
    uint8_t aei[] = {1, 5, 9};  // A=1, E=5, I=9
    assert(vmodel_classify_leave(aei, 3) == VMODEL_TYPE_VOWEL);

    // B, C, D = all consonants → "cons".
    uint8_t bcd[] = {2, 3, 4};
    assert(vmodel_classify_leave(bcd, 3) == VMODEL_TYPE_CONS);

    // A, B, ? — vowel + cons + blank → "mixed" (blank ignored, one vowel
    // and one cons).
    uint8_t abq[] = {1, 2, 0};
    assert(vmodel_classify_leave(abq, 3) == VMODEL_TYPE_MIXED);

    // Y treated as vowel: Y, B, ? = vowel + cons → "mixed".
    uint8_t ybq[] = {25, 2, 0};  // Y=25
    assert(vmodel_classify_leave(ybq, 3) == VMODEL_TYPE_MIXED);

    // Y, A, E = all vowels (Y counts as vowel) → "vowel".
    uint8_t yae[] = {25, 1, 5};
    assert(vmodel_classify_leave(yae, 3) == VMODEL_TYPE_VOWEL);
}

static void check_tile_index(void) {
    assert(vmodel_tile_index_from_char('?') == 0);
    assert(vmodel_tile_index_from_char('A') == 1);
    assert(vmodel_tile_index_from_char('Z') == 26);
    assert(vmodel_tile_index_from_char('a') == -1);
    assert(vmodel_tile_index_from_char('!') == -1);
}

void test_vmodel(void) {
    check_classify();
    check_tile_index();

    const char *path = getenv("MAGPIE_VMODEL_TEST_PATH");
    if (!path) {
        printf("test_vmodel: MAGPIE_VMODEL_TEST_PATH not set, "
               "skipping load test\n");
        return;
    }

    VModel *m = vmodel_create(path);
    assert(m != NULL);
    assert(m->version == 1);
    assert(m->bag == 93);
    assert(m->n_strata > 0);
    assert(m->turn == 6);

    int total_buckets = 0;
    int total_coefs = 0;
    for (int i = 0; i < m->n_strata; i++) {
        const VStratum *s = &m->strata[i];
        assert(s->n_tiles > 0);
        assert(s->n_buckets > 0);
        for (int j = 0; j < s->n_buckets; j++) {
            const VBucket *b = &s->buckets[j];
            assert(b->n_coefs > 0);
            assert(b->coefs != NULL);
            assert(b->diff_lo <= b->diff_hi);
            total_buckets += 1;
            total_coefs += b->n_coefs;
        }
    }
    printf("test_vmodel: loaded %d strata, %d buckets, %d total coefs\n",
           m->n_strata, total_buckets, total_coefs);

    check_stratum_lookup(m);

    vmodel_destroy(m);
    printf("test_vmodel: PASS\n");
}

// Parse "AEILORS" (or "?" tokens) into 7-or-fewer python-canonical
// tile indices. Returns length; -1 on error.
static int parse_tile_string(const char *s, uint8_t *out, int max_len) {
    int n = 0;
    for (const char *p = s; *p; p++) {
        if (n >= max_len) return -1;
        int idx = vmodel_tile_index_from_char(*p);
        if (idx < 0) return -1;
        out[n++] = (uint8_t)idx;
    }
    return n;
}

// Verify C predictions against the Python "gold" CSV. Reads:
//   MAGPIE_VMODEL_VERIFY_VMT       — model .vmt path
//   MAGPIE_VMODEL_VERIFY_INPUTS    — verification_inputs.csv
//                                    (rack,leave,kind,diff,turn,bag)
//   MAGPIE_VMODEL_VERIFY_GOLD      — verification_gold.csv
//                                    (rack,leave,kind,diff,turn,bag,win_pct)
//   MAGPIE_VMODEL_VERIFY_LEAVES_6  — CSW24_gen_6.csv
//   MAGPIE_VMODEL_VERIFY_LEAVES_7  — CSW24_7tile_gen_6.csv
//   MAGPIE_VMODEL_VERIFY_TOL       — optional, default 1e-5
//
// Skipped silently if any required env var is unset.
void test_vmodel_verify(void) {
    const char *vmt   = getenv("MAGPIE_VMODEL_VERIFY_VMT");
    const char *gold  = getenv("MAGPIE_VMODEL_VERIFY_GOLD");
    const char *l6    = getenv("MAGPIE_VMODEL_VERIFY_LEAVES_6");
    const char *l7    = getenv("MAGPIE_VMODEL_VERIFY_LEAVES_7");
    if (!vmt || !gold) {
        printf("test_vmodel_verify: env vars not set, skipping\n");
        return;
    }
    const char *tol_str = getenv("MAGPIE_VMODEL_VERIFY_TOL");
    float tol = tol_str ? (float)atof(tol_str) : 1e-5f;

    VModel *m = vmodel_create(vmt);
    assert(m != NULL);
    StaticLeaves *sl = static_leaves_create(l6, l7);
    assert(sl != NULL);

    FILE *f = fopen(gold, "r");
    assert(f != NULL);
    char buf[1024];
    int line_no = 0;
    int n_rows = 0;
    int n_mismatch = 0;
    float max_abs_diff = 0.0f;
    char worst_line[1024] = "";
    fgets(buf, sizeof(buf), f);  // skip header
    while (fgets(buf, sizeof(buf), f)) {
        line_no++;
        // rack,leave,kind,diff,turn,bag,win_pct
        char rack_s[32], leave_s[32];
        int kind, diff, turn;
        float py_win = 0.0f;
        // Handle empty leave (consecutive commas).
        // Robust parse: split on commas.
        char *fields[16];
        int nf = 0;
        char *p = buf;
        fields[nf++] = p;
        while (*p && nf < 16) {
            if (*p == ',') {
                *p = '\0';
                fields[nf++] = p + 1;
            }
            p++;
        }
        if (nf < 7) continue;
        // Strip trailing newline from last field.
        size_t ll = strlen(fields[nf - 1]);
        while (ll > 0 && (fields[nf-1][ll-1] == '\n' || fields[nf-1][ll-1] == '\r')) {
            fields[nf-1][--ll] = '\0';
        }
        snprintf(rack_s, sizeof(rack_s), "%s", fields[0]);
        snprintf(leave_s, sizeof(leave_s), "%s", fields[1]);
        kind = atoi(fields[2]);
        diff = atoi(fields[3]);
        turn = atoi(fields[4]);
        // fields[5] = bag; model carries it, not used in predict.
        py_win = (float)atof(fields[6]);

        uint8_t rack_idx[16], leave_idx[16];
        int rack_len = parse_tile_string(rack_s, rack_idx, 16);
        int leave_len = (leave_s[0] == '\0') ? 0
                        : parse_tile_string(leave_s, leave_idx, 16);
        if (rack_len < 0 || leave_len < 0) {
            printf("test_vmodel_verify: bad parse on line %d: %s\n",
                   line_no, buf);
            continue;
        }
        float c_win = vmodel_predict(m, rack_idx, rack_len,
                                      leave_idx, leave_len,
                                      kind, diff, turn, NULL, sl);
        if (c_win < 0.0f) {
            printf("test_vmodel_verify: predict failed line %d "
                   "rack=%s leave=%s kind=%d diff=%d\n",
                   line_no, rack_s, leave_s, kind, diff);
            n_mismatch++;
            continue;
        }
        n_rows++;
        float d = fabsf(c_win - py_win);
        if (d > max_abs_diff) {
            max_abs_diff = d;
            snprintf(worst_line, sizeof(worst_line),
                     "rack=%s leave=%s kind=%d diff=%d py=%.6f c=%.6f d=%.6e",
                     rack_s, leave_s, kind, diff, py_win, c_win, d);
        }
        if (d > tol) {
            if (n_mismatch < 10) {
                printf("MISMATCH line %d: rack=%s leave=%s kind=%d diff=%d "
                       "py=%.6f c=%.6f d=%.6e\n",
                       line_no, rack_s, leave_s, kind, diff,
                       py_win, c_win, d);
            }
            n_mismatch++;
        }
    }
    fclose(f);
    printf("test_vmodel_verify: %d rows, %d mismatches (tol=%.1e), "
           "max |Δ|=%.6e\n", n_rows, n_mismatch, tol, max_abs_diff);
    printf("test_vmodel_verify: worst: %s\n", worst_line);
    static_leaves_destroy(sl);
    vmodel_destroy(m);
    assert(n_mismatch == 0);
    printf("test_vmodel_verify: PASS\n");
}
