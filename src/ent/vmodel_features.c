#include "vmodel_features.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../util/io_util.h"

// V-model feature extraction.
//
// Tile encoding throughout: python-canonical 27-tile alphabet with
// blank first: index 0 = '?', 1 = 'A', ..., 26 = 'Z'. All input
// rack/leave arrays use this encoding.

#define VMODEL_RACK_SIZE 7

// English Scrabble tile distribution (initial bag of 100, index by
// python-canonical tile id).
static const uint8_t TILE_BAG[27] = {
    2,  // ?
    9,  // A
    2,  // B
    2,  // C
    4,  // D
    12, // E
    2,  // F
    3,  // G
    2,  // H
    9,  // I
    1,  // J
    1,  // K
    4,  // L
    2,  // M
    6,  // N
    8,  // O
    2,  // P
    1,  // Q
    6,  // R
    4,  // S
    6,  // T
    4,  // U
    2,  // V
    2,  // W
    1,  // X
    2,  // Y
    1,  // Z
};

// Standard Scrabble face values (rack penalty at terminal-zero).
static const uint8_t TILE_VALUE[27] = {
    0,  // ?
    1,  // A
    3,  // B
    3,  // C
    2,  // D
    1,  // E
    4,  // F
    2,  // G
    4,  // H
    1,  // I
    8,  // J
    5,  // K
    1,  // L
    3,  // M
    1,  // N
    1,  // O
    3,  // P
    10, // Q
    1,  // R
    1,  // S
    1,  // T
    1,  // U
    4,  // V
    4,  // W
    8,  // X
    4,  // Y
    10, // Z
};

static bool terminal_zero_score(int turn, int kind) {
    return turn == 6 && (kind == 0 || kind == 1);
}

static void count_tiles(const uint8_t *indices, int len, uint8_t out[27]) {
    memset(out, 0, 27);
    for (int i = 0; i < len; i++) {
        if (indices[i] < 27) out[indices[i]]++;
    }
}

static int leave_pts_total(const uint8_t *leave, int len) {
    int total = 0;
    for (int i = 0; i < len; i++) {
        if (leave[i] < 27) total += TILE_VALUE[leave[i]];
    }
    return total;
}

// Returns tiles_played given unseen_total and leave_size.
static int tiles_played_count(int unseen_total, int leave_size) {
    int v = VMODEL_RACK_SIZE - leave_size;
    int bag_remaining = unseen_total - VMODEL_RACK_SIZE;
    if (bag_remaining < 0) bag_remaining = 0;
    return v < bag_remaining ? v : bag_remaining;
}

// Append a float to buf, advancing the write index.
static int append_feat(float *buf, int cap, int idx, float v) {
    if (idx >= cap) return -1;
    buf[idx] = v;
    return idx + 1;
}

// Fill `unseen[27]` (and its total) for the on-turn player. With `unseen_vec`
// (endgame) copy it directly; otherwise derive `TILE_BAG - own_rack` (opener,
// bag==93 only). Returns false if derivation is requested at bag!=93 (unsafe).
static bool fill_unseen(const int *unseen_vec, const uint8_t *rack_counts,
                        int bag, const char *key, int *unseen, int *total) {
    int t = 0;
    if (unseen_vec) {
        for (t = 0; t < 27; t++) unseen[t] = unseen_vec[t];
    } else {
        if (bag != 93) {
            log_warn("vmodel_features: unseen derivation only correct at "
                     "bag=93; got bag=%d (stratum %s) with no unseen_vec",
                     bag, key);
            return false;
        }
        for (t = 0; t < 27; t++) unseen[t] = TILE_BAG[t] - rack_counts[t];
    }
    int tot = 0;
    for (t = 0; t < 27; t++) tot += unseen[t];
    *total = tot;
    return true;
}

int vmodel_extract_features(float *buf, int cap,
                             const VStratum *s, const VBucket *b,
                             const uint8_t *rack_indices, int rack_len,
                             const uint8_t *leave_indices, int leave_len,
                             int diff, int bag, int turn,
                             const int *unseen_vec,
                             int play_unseen,
                             const StaticLeaves *sl) {
    bool tzs = terminal_zero_score(turn, s->kind);
    int wi = 0;

    if (b->has_diff) {
        wi = append_feat(buf, cap, wi, (float)diff);
        if (wi < 0) return -1;
    }

    if (tzs) {
        // Terminal-zero feature set.
        int lp = leave_pts_total(leave_indices, leave_len);
        bool keep_zero = (b->n_pts > 1);
        for (int i = 0; i < b->n_pts; i++) {
            int p = b->pts[i];
            if (p == 0 && !keep_zero) continue;
            wi = append_feat(buf, cap, wi, (lp == p) ? 1.0f : 0.0f);
            if (wi < 0) return -1;
        }
        if (s->leave_length > 0) {
            wi = append_feat(buf, cap, wi, (float)lp);
            if (wi < 0) return -1;
        }
        if (s->kind == 1) {
            uint8_t rack_counts[27];
            count_tiles(rack_indices, rack_len, rack_counts);
            int unseen_total = 0;
            int unseen[27];
            if (!fill_unseen(unseen_vec, rack_counts, bag, s->key,
                             unseen, &unseen_total)) {
                return -1;
            }
            double weighted = 0.0;
            for (int t = 0; t < 27; t++) {
                weighted += (double)unseen[t] * (double)TILE_VALUE[t];
            }
            double d = 0.0;
            int tp = tiles_played_count(unseen_total, leave_len);
            if (unseen_total > 0 && tp > 0) {
                d = (double)tp * weighted / (double)unseen_total;
            }
            if (b->n_drawn > 0) {
                // Python rounds to 4 decimal places then compares for
                // exact equality; mirror with float64 throughout.
                double d_rounded = round(d * 10000.0) / 10000.0;
                for (int i = 0; i < b->n_drawn; i++) {
                    wi = append_feat(buf, cap, wi,
                                      (d_rounded == b->drawn[i]) ? 1.0f : 0.0f);
                    if (wi < 0) return -1;
                }
            }
            wi = append_feat(buf, cap, wi, (float)d);
            if (wi < 0) return -1;
        }
        if (wi != b->n_coefs) {
            log_warn("vmodel_features: TZS shape mismatch in %s bucket "
                     "[%d,%d]: got %d feats, need %d",
                     s->key, b->diff_lo, b->diff_hi, wi, b->n_coefs);
            return -1;
        }
        return wi;
    }

    // ----- Non-terminal feature set. -----
    if (s->leave_length > 0) {
        uint8_t canon[STATIC_LEAVES_MAX_LEN];
        static_leaves_canonicalize(leave_indices, leave_len, canon);
        float v = static_leaves_lookup(sl, canon, leave_len);
        wi = append_feat(buf, cap, wi, v);
        if (wi < 0) return -1;
    }

    uint8_t leave_counts[27];
    count_tiles(leave_indices, leave_len, leave_counts);

    if (s->leave_length == 1) {
        // 1-hot in the stratum's tile slot.
        for (int i = 0; i < s->n_tiles; i++) {
            uint8_t t = s->tiles[i];
            wi = append_feat(buf, cap, wi,
                              (leave_counts[t] > 0) ? 1.0f : 0.0f);
            if (wi < 0) return -1;
        }
    } else if (s->leave_length >= 2) {
        if (s->leave_length >= 3) {
            for (int i = 0; i < s->n_tiles; i++) {
                uint8_t t = s->tiles[i];
                wi = append_feat(buf, cap, wi, (float)leave_counts[t]);
                if (wi < 0) return -1;
            }
        }
        for (int i = 0; i < s->n_pairs; i++) {
            uint8_t a = s->pairs[i][0];
            uint8_t bb = s->pairs[i][1];
            int n;
            if (a == bb) {
                int na = leave_counts[a];
                n = na * (na - 1) / 2;
            } else {
                n = (int)leave_counts[a] * (int)leave_counts[bb];
            }
            wi = append_feat(buf, cap, wi, (float)n);
            if (wi < 0) return -1;
        }
    }

    // Unseen-pool PAIR block (373), on NON-PASS strata only. Mirrors
    // v_model_features._unseen_pair_names/_unseen_pair_values exactly: the
    // 27-tile upper triangle including the diagonal, SKIPPING doubles of
    // single-copy tiles (J,K,Q,X,Z) -> 378 - 5 = 373, in that iteration order.
    // Values are pair COUNTS over the unseen pool: u[a]*u[b], and
    // u[a](u[a]-1)/2 on the diagonal.
    bool is_pass_stratum = (s->kind == 0 && s->leave_length == VMODEL_RACK_SIZE);
    if (play_unseen >= 1 && !is_pass_stratum) {
        uint8_t rc_up[27];
        count_tiles(rack_indices, rack_len, rc_up);
        int u[27];
        int u_total = 0;
        if (!fill_unseen(unseen_vec, rc_up, bag, s->key, u, &u_total)) return -1;
        for (int a = 0; a < 27; a++) {
            for (int bq = a; bq < 27; bq++) {
                if (a == bq && TILE_BAG[a] < 2) continue;
                float v = (a == bq) ? (float)((double)u[a] * (u[a] - 1) / 2.0)
                                    : (float)((double)u[a] * (double)u[bq]);
                wi = append_feat(buf, cap, wi, v);
                if (wi < 0) return -1;
            }
        }
    }

    // bag_exp: skip for K0_L7 (pass — tiles_played=0).
    bool emit_bag_block = !(s->kind == 0 && s->leave_length == VMODEL_RACK_SIZE);
    if (emit_bag_block) {
        uint8_t rack_counts[27];
        count_tiles(rack_indices, rack_len, rack_counts);
        int unseen[27];
        int unseen_total = 0;
        if (!fill_unseen(unseen_vec, rack_counts, bag, s->key,
                         unseen, &unseen_total)) {
            return -1;
        }
        int tp = (unseen_total > 0) ? tiles_played_count(unseen_total, s->leave_length) : 0;
        // bag_exp for tiles NOT in bag_indicators.
        bool ind_mask[27] = {0};
        for (int i = 0; i < b->n_bag; i++) ind_mask[b->bag_tiles[i]] = true;
        for (int t = 0; t < 27; t++) {
            if (ind_mask[t]) continue;
            float bv;
            if (play_unseen >= 2) {
                // unseen_count_<t>: the RAW count. Within a (bag, stratum) the
                // expectation form tp*u[t]/N is this same feature times a
                // constant, but it sits on a different scale per stratum, so
                // the trainer switched to counts. Must match or every coef is
                // scaled wrong.
                bv = (float)unseen[t];
            } else {
                bv = 0.0f;
                if (unseen_total > 0 && tp > 0) {
                    bv = (float)((double)tp * (double)unseen[t] / (double)unseen_total);
                }
            }
            wi = append_feat(buf, cap, wi, bv);
            if (wi < 0) return -1;
        }
        for (int i = 0; i < b->n_bag; i++) {
            uint8_t t = b->bag_tiles[i];
            int8_t c = b->bag_counts[i];
            // Indicator fires when the UNSEEN pile has exactly c of tile t.
            // (At the opener unseen[t] == TILE_BAG[t] - rack[t]; in the endgame
            // it is bag+opp from unseen_vec — fill_unseen handled the source.)
            wi = append_feat(buf, cap, wi,
                              (unseen[t] == (int)c) ? 1.0f : 0.0f);
            if (wi < 0) return -1;
        }
    }

    if (s->kind == 1) {
        uint8_t rack_counts[27];
        count_tiles(rack_indices, rack_len, rack_counts);
        for (int t = 0; t < 27; t++) {
            int exch = (int)rack_counts[t] - (int)leave_counts[t];
            wi = append_feat(buf, cap, wi, (float)exch);
            if (wi < 0) return -1;
        }
    }

    if (wi != b->n_coefs) {
        // stderr, not log_warn: current_log_level defaults to LOG_FATAL, so
        // every log_warn in this file was suppressed -- which is why a model
        // scoring NOTHING looked identical to one scoring everything.
        static int warned_shape = 0;
        if (warned_shape++ < 8) {
            fprintf(stderr, "vmodel_features: SHAPE MISMATCH in %s bucket "
                    "[%d,%d]: got %d feats, need %d\n",
                    s->key, b->diff_lo, b->diff_hi, wi, b->n_coefs);
        }
        return -1;
    }
    return wi;
}

static float sigmoid(float z) {
    if (z >  50.0f) return 1.0f;
    if (z < -50.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-z));
}

float vmodel_predict(const VModel *m,
                      const uint8_t *rack_indices, int rack_len,
                      const uint8_t *leave_indices, int leave_len,
                      int kind, int diff, int turn,
                      const int *unseen_vec,
                      const StaticLeaves *sl) {
    // The `turn` arg controls the terminal-zero-score gate (which
    // collapses K1_L3..6 to "all" and switches K0/K1 to the compact pts
    // indicator schema). What matters is the MODEL's training turn,
    // not the game's current turn — the schema must match the trained
    // coefs. Override unconditionally so callers can pass game-turn as
    // a documentation hint without affecting inference.
    turn = m->turn;
    // PASS (K0_L7) bucketed on the six-pass tally: convert the incoming raw lead
    // into sixpass_diff so the stratum/bucket lookup AND the move_score feature
    // match how the stratum was fit. sixpass_diff = lead - (mover rack pts) +
    // (sum of the RACK_SIZE lowest-value unseen tiles = opponent's best-case rack
    // in a six-pass standoff). Mirrors v_model_features.sixpass_diff. Gated on the
    // model's PASS6 flag; needs the unseen vector (present for bag != 93).
    if (m->pass_sixpass && kind == 0 && leave_len == VMODEL_RACK_SIZE &&
        unseen_vec) {
        int mover = leave_pts_total(leave_indices, leave_len);
        int need = VMODEL_RACK_SIZE, add = 0;
        for (int v = 0; v <= 10 && need > 0; v++) {
            for (int t = 0; t < 27 && need > 0; t++) {
                if (TILE_VALUE[t] != v) continue;
                int c = unseen_vec[t];
                if (c <= 0) continue;
                int take = c < need ? c : need;
                add += take * v;
                need -= take;
            }
        }
        diff = diff - mover + add;
    }
    VModelLeaveType lt = vmodel_classify_leave(leave_indices, leave_len);
    int si = vmodel_stratum_index(m, kind, leave_len, lt, turn);
    // These two paths returned -1 SILENTLY, so a model that scored nothing at
    // all was indistinguishable from one that scored everything: every fill
    // since bag 8 logged "moves unscored 100.000%" with no other diagnostic.
    // Cap the noise but never let it be silent again.
    if (si < 0) {
        static int warned_stratum = 0;  // benign race; only bounds log volume
        if (warned_stratum++ < 8) {
            fprintf(stderr, "vmodel: NO STRATUM kind=%d leave_len=%d type=%d "
                    "turn=%d bag=%d (n_strata=%d) -> move unscored\n",
                    kind, leave_len, (int)lt, turn, m->bag, m->n_strata);
        }
        return -1.0f;
    }
    const VStratum *s = &m->strata[si];
    int bi = vmodel_bucket_index(s, diff);
    if (bi < 0) {
        static int warned_bucket = 0;
        if (warned_bucket++ < 8) {
            fprintf(stderr, "vmodel: NO BUCKET in %s for diff=%d "
                    "(n_buckets=%d, bag=%d) -> move unscored\n",
                    s->key, diff, s->n_buckets, m->bag);
        }
        return -1.0f;
    }
    const VBucket *b = &s->buckets[bi];

    // Stack buffer — biggest bucket has ~430 features.
    float feats[1024];
    int n = vmodel_extract_features(feats, (int)(sizeof(feats)/sizeof(feats[0])),
                                     s, b, rack_indices, rack_len,
                                     leave_indices, leave_len,
                                     diff, m->bag, turn, unseen_vec,
                                     m->play_unseen, sl);
    if (n < 0) return -1.0f;
    double z = b->intercept;
    for (int i = 0; i < n; i++) {
        z += (double)feats[i] * (double)b->coefs[i];
    }
    return sigmoid((float)z);
}
