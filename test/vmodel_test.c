#include "../src/ent/vmodel.h"
#include <assert.h>
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
