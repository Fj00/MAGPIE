#include "../src/ent/vmodel_picks.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/def/letter_distribution_defs.h"

// Verification test for vmodel_picks:
//
//   1. Load a .picks file (path via MAGPIE_VMODEL_PICKS_TEST_PATH).
//   2. Sanity: header sane, n_entries > 0, every entry parses to a
//      valid Move (move_type set, fields populated as expected per
//      action_repr semantics).
//   3. Sortedness: entries are sorted by canonical-rack key (required
//      for binary-search lookup correctness).
//   4. Lookup round-trip: for a stratified sample of entries, read
//      the entry's rack key, run vmodel_picks_lookup(rack), and
//      assert it returns the same Move (pointer or byte-for-byte).
//   5. Dump 12 sample entries (varied across pass/exch/play and
//      score range) so the parser output can be eyeball-verified.
//
// Skipped silently when MAGPIE_VMODEL_PICKS_TEST_PATH is unset.

static const char *kind_name(int t) {
    switch (t) {
        case GAME_EVENT_PASS:                return "PASS";
        case GAME_EVENT_EXCHANGE:            return "EXCH";
        case GAME_EVENT_TILE_PLACEMENT_MOVE: return "PLAY";
        default: return "UNK";
    }
}

static void dump_move(const uint8_t *rack_key, const Move *m) {
    char rack[16] = {0};
    for (int i = 0; i < VMODEL_PICKS_RACK_KEY_LEN && rack_key[i]; i++) {
        rack[i] = rack_key[i];
    }
    int score_int = (int)(m->score / 1000);  // Equity → int score
    char tiles[32] = {0};
    int tn = 0;
    for (int i = 0; i < m->tiles_length && tn < (int)sizeof(tiles) - 1; i++) {
        uint8_t t = m->tiles[i];
        if (t == 0) {
            tiles[tn++] = (m->move_type == GAME_EVENT_PASS) ? ' ' : '.';
        } else if (t & BLANK_MASK) {
            uint8_t unb = t & UNBLANK_MASK;
            tiles[tn++] = (unb == 0) ? '?'
                                      : (char)('a' + unb - 1);
        } else {
            tiles[tn++] = (char)('A' + t - 1);
        }
    }
    printf("  rack=%-7s  %s  row=%d col=%d dir=%d  played=%d  len=%d  "
           "score=%d  tiles=[%s]\n",
           rack, kind_name(m->move_type), m->row_start, m->col_start,
           m->dir, m->tiles_played, m->tiles_length, score_int, tiles);
}

void test_vmodel_picks(void) {
    const char *path = getenv("MAGPIE_VMODEL_PICKS_TEST_PATH");
    if (!path) {
        printf("test_vmodel_picks: MAGPIE_VMODEL_PICKS_TEST_PATH not set, "
               "skipping\n");
        return;
    }

    VModelPicks *picks = vmodel_picks_create(path);
    assert(picks != NULL);
    assert(picks->n_entries > 0);
    printf("test_vmodel_picks: loaded %d entries (T%d slot)\n",
           picks->n_entries, picks->model_turn);

    // Sortedness check (required for binary-search lookup correctness).
    int sort_violations = 0;
    for (int i = 1; i < picks->n_entries; i++) {
        if (memcmp(picks->keys[i - 1], picks->keys[i],
                    VMODEL_PICKS_RACK_KEY_LEN) >= 0) {
            sort_violations++;
            if (sort_violations <= 3) {
                printf("SORT VIOLATION at i=%d\n", i);
            }
        }
    }
    printf("test_vmodel_picks: sortedness %s (%d violations)\n",
           sort_violations == 0 ? "OK" : "FAIL", sort_violations);
    assert(sort_violations == 0);

    // Parser correctness: every Move must have a valid move_type. PASS
    // tiles_played == 0; EXCHANGE 1..7; PLAY 1..7 with row,col in [0,14].
    int n_pass = 0, n_exch = 0, n_play = 0, n_bad = 0;
    for (int i = 0; i < picks->n_entries; i++) {
        const Move *m = &picks->moves[i];
        switch (m->move_type) {
            case GAME_EVENT_PASS:
                if (m->tiles_played != 0) n_bad++;
                n_pass++; break;
            case GAME_EVENT_EXCHANGE:
                if (m->tiles_played < 1 || m->tiles_played > 7) n_bad++;
                n_exch++; break;
            case GAME_EVENT_TILE_PLACEMENT_MOVE:
                if (m->tiles_played < 1 || m->tiles_played > 7) n_bad++;
                if (m->row_start > 14 || m->col_start > 14) n_bad++;
                if (m->dir > 1) n_bad++;
                n_play++; break;
            default:
                n_bad++; break;
        }
    }
    printf("test_vmodel_picks: pass=%d exch=%d play=%d bad=%d\n",
           n_pass, n_exch, n_play, n_bad);
    assert(n_bad == 0);
    assert(n_pass + n_exch + n_play == picks->n_entries);

    // Lookup round-trip: 10K random samples must hit + return the
    // same Move as the direct array index.
    int lookup_hits = 0, lookup_mismatches = 0;
    for (int t = 0; t < 10000; t++) {
        int i = (int)(((uint64_t)rand() * picks->n_entries) / RAND_MAX);
        if (i >= picks->n_entries) i = picks->n_entries - 1;
        const Move *expected = &picks->moves[i];
        const Move *got = vmodel_picks_lookup(picks, picks->keys[i]);
        if (got == NULL) {
            printf("LOOKUP MISS at i=%d\n", i);
            lookup_mismatches++;
            continue;
        }
        lookup_hits++;
        // Compare key fields (pointer equality also fine since both
        // come from the picks->moves array).
        if (got->move_type != expected->move_type ||
            got->row_start != expected->row_start ||
            got->col_start != expected->col_start ||
            got->dir != expected->dir ||
            got->tiles_played != expected->tiles_played ||
            got->tiles_length != expected->tiles_length ||
            got->score != expected->score) {
            lookup_mismatches++;
            if (lookup_mismatches <= 3) {
                printf("LOOKUP MISMATCH at i=%d\n", i);
                printf("  expected:"); dump_move(picks->keys[i], expected);
                printf("  got:     "); dump_move(picks->keys[i], got);
            }
        }
    }
    printf("test_vmodel_picks: lookup round-trip %s "
           "(%d hits, %d mismatches over 10K samples)\n",
           lookup_mismatches == 0 ? "OK" : "FAIL",
           lookup_hits, lookup_mismatches);
    assert(lookup_mismatches == 0);

    // Eyeball: 12 sample entries across kinds + score range.
    printf("test_vmodel_picks: sample picks (eyeball check) —\n");
    int stride = picks->n_entries / 12;
    if (stride < 1) stride = 1;
    for (int i = 0; i < picks->n_entries && i < stride * 12; i += stride) {
        dump_move(picks->keys[i], &picks->moves[i]);
    }

    vmodel_picks_destroy(picks);
    printf("test_vmodel_picks: PASS\n");
}
