#include "vmodel_features.h"

#include <math.h>
#include <stdatomic.h>
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
                             int play_unseen, int pass_unseen,
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
    // PASS block: 27 raw counts, then (level 2) the same 373 pairs. Counts
    // first -- v_model_features._pass_unseen_values builds it that way, which
    // is the OPPOSITE arrangement to the play path below.
    if (pass_unseen >= 1 && is_pass_stratum) {
        uint8_t rc_p[27];
        count_tiles(rack_indices, rack_len, rc_p);
        int u[27];
        int u_total = 0;
        if (!fill_unseen(unseen_vec, rc_p, bag, s->key, u, &u_total)) return -1;
        for (int t2 = 0; t2 < 27; t2++) {
            wi = append_feat(buf, cap, wi, (float)u[t2]);
            if (wi < 0) return -1;
        }
        if (pass_unseen >= 2) {
            for (int a = 0; a < 27; a++) {
                for (int bq = a; bq < 27; bq++) {
                    if (a == bq && TILE_BAG[a] < 2) continue;
                    float v = (a == bq)
                                  ? (float)((double)u[a] * (u[a] - 1) / 2.0)
                                  : (float)((double)u[a] * (double)u[bq]);
                    wi = append_feat(buf, cap, wi, v);
                    if (wi < 0) return -1;
                }
            }
        }
    }
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
                                     m->play_unseen, m->pass_unseen, sl);
    if (n < 0) return -1.0f;
    double z = b->intercept;
    for (int i = 0; i < n; i++) {
        z += (double)feats[i] * (double)b->coefs[i];
    }
    return sigmoid((float)z);
}

// ---------------------------------------------------------------------------
// Per-turn prediction cache. See vmodel_features.h for the derivation.
// ---------------------------------------------------------------------------

// Slots per thread. Distinct (kind, leave) pairs in one turn are bounded by
// 2 x 127 (plays and exchanges over the 7-tile subsets), so 512 keeps the
// load factor under 0.5.
#define VMTC_SLOTS 512
// Buckets cached per slot. Strata carry ~60-95 diff buckets; a stratum with
// more than this falls back to vmodel_predict rather than overflowing.
#define VMTC_MAXB 128

// Nonzero leave-block features per leave: the static leave value, at most 7
// tile counts and 21 pair counts for a 7-tile leave. Room to spare; a leave
// that somehow needs more takes the dense path.
#define VMTC_MAXSP 40

typedef struct {
    uint64_t key;              // packed (kind, len, sorted leave); 0 = empty
    uint32_t gen;              // valid iff == cache->gen
    int16_t  si;               // stratum index, or -1 if none
    // Sparse leave block, filled when the slot is claimed: the NONZERO features
    // among [static leave value, tile counts, pair counts] as (offset into the
    // block, value), in ascending offset order. sp_n < 0: use the dense path.
    int16_t  sp_n;
    uint16_t sp_len;                // full (dense) length of the leave block
    uint16_t sp_off[VMTC_MAXSP];
    float    sp_val[VMTC_MAXSP];
    uint8_t  leave_counts[27];
    uint32_t bgen[VMTC_MAXB];  // bucket entry valid iff == cache->gen
    double   z0[VMTC_MAXB];    // intercept + sum over all features but diff
} VMTCSlot;

struct VModelTurnCache {
    uint32_t gen;
    const VModel *m;
    const StaticLeaves *sl;
    uint8_t rack[16];
    int rack_len;
    int unseen[27];
    bool has_unseen;
    uint8_t rack_counts[27];
    int unseen_total;
    VMTCSlot *slots;
};

static _Thread_local VModelTurnCache *tl_vmtc = NULL;

_Atomic uint64_t g_vmtc_sparse_checks = 0;
_Atomic uint64_t g_vmtc_sparse_mismatch = 0;

VModelTurnCache *vmodel_turn_cache_get(void) {
    if (!tl_vmtc) {
        tl_vmtc = calloc(1, sizeof(VModelTurnCache));
        if (!tl_vmtc) return NULL;
        tl_vmtc->slots = calloc(VMTC_SLOTS, sizeof(VMTCSlot));
        if (!tl_vmtc->slots) { free(tl_vmtc); tl_vmtc = NULL; return NULL; }
    }
    return tl_vmtc;
}

void vmodel_turn_cache_begin(VModelTurnCache *c, const VModel *m,
                             const uint8_t *rack_indices, int rack_len,
                             const int *unseen_vec, const StaticLeaves *sl) {
    // Advance the generation: every slot and bucket stamped with an older value
    // is stale, so nothing needs clearing. On wrap, clear once so a stamp left
    // from 2^32 turns ago can never be mistaken for current.
    c->gen++;
    if (c->gen == 0) {
        memset(c->slots, 0, VMTC_SLOTS * sizeof(VMTCSlot));
        c->gen = 1;
    }
    c->m = m;
    c->sl = sl;
    if (rack_len > 16) rack_len = 16;
    memcpy(c->rack, rack_indices, (size_t)rack_len);
    c->rack_len = rack_len;
    c->has_unseen = unseen_vec != NULL;
    if (unseen_vec) memcpy(c->unseen, unseen_vec, sizeof(c->unseen));
    count_tiles(c->rack, c->rack_len, c->rack_counts);
    c->unseen_total = 0;
    if (unseen_vec) {
        for (int t = 0; t < 27; t++) c->unseen_total += c->unseen[t];
    }
}

// Fill the slot's sparse leave block (see VMTCSlot). Mirrors the leave part of
// vmodel_extract_features feature for feature, keeping only the nonzero ones;
// a zero feature contributes exactly 0 to the dot product, so skipping it
// leaves the sum bit-identical as long as the remaining terms are added in the
// same order.
static void vmtc_fill_sparse(VMTCSlot *slot, const VStratum *s,
                             const uint8_t *leave_indices, int leave_len,
                             const StaticLeaves *sl) {
    count_tiles(leave_indices, leave_len, slot->leave_counts);
    const uint8_t *lc = slot->leave_counts;
    int n = 0, off = 0;
    bool ok = true;
#define VMTC_PUSH(v)                                                  \
    do {                                                              \
        float v_ = (v);                                               \
        if (v_ != 0.0f) {                                             \
            if (n >= VMTC_MAXSP) ok = false;                          \
            else { slot->sp_off[n] = (uint16_t)off; slot->sp_val[n] = v_; n++; } \
        }                                                             \
        off++;                                                        \
    } while (0)
    if (s->leave_length > 0) {
        uint8_t canon[STATIC_LEAVES_MAX_LEN];
        static_leaves_canonicalize(leave_indices, leave_len, canon);
        VMTC_PUSH(static_leaves_lookup(sl, canon, leave_len));
    }
    if (s->leave_length == 1) {
        for (int i = 0; i < s->n_tiles; i++)
            VMTC_PUSH((lc[s->tiles[i]] > 0) ? 1.0f : 0.0f);
    } else if (s->leave_length >= 2) {
        if (s->leave_length >= 3) {
            for (int i = 0; i < s->n_tiles; i++)
                VMTC_PUSH((float)lc[s->tiles[i]]);
        }
        for (int i = 0; i < s->n_pairs; i++) {
            uint8_t a = s->pairs[i][0];
            uint8_t bb = s->pairs[i][1];
            int na = lc[a];
            int v = (a == bb) ? na * (na - 1) / 2 : na * (int)lc[bb];
            VMTC_PUSH((float)v);
        }
    }
#undef VMTC_PUSH
    slot->sp_n = ok ? (int16_t)n : -1;
    slot->sp_len = (uint16_t)off;
}

// Bucket for `diff` by binary search, falling back to the reference linear scan
// whenever the search does not land inside a bucket. vmodel_bucket_index scans
// all ~60-95 buckets per call; buckets are contiguous diff ranges in ascending
// order, so ~7 comparisons suffice. If the search misses (unsorted buckets, or a
// gap) the fallback returns exactly what the linear scan would. Overlapping
// buckets are the one case where the two could differ -- the scan takes the
// FIRST match, the search any match -- and the layout never produces them;
// MAGPIE_VMODEL_CACHE_VERIFY compares against the untouched reference scan.
static int vmtc_bucket_index(const VStratum *s, int diff) {
    int lo = 0, hi = s->n_buckets - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        const VBucket *b = &s->buckets[mid];
        if (diff < b->diff_lo) hi = mid - 1;
        else if (diff > b->diff_hi) lo = mid + 1;
        else return mid;
    }
    return vmodel_bucket_index(s, diff);
}

static uint64_t vmtc_mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// z0 for a (leave, bucket) miss without building the dense feature vector. The
// leave block is ~400 features (tiles + 373 pairs) of which at most ~29 are
// nonzero, and the dense path multiplied every one. Same terms in the same
// order as the dense path, zeros skipped, so the result is bit-identical.
// Returns false where it does not apply (unseen-pair blocks, no unseen vector,
// pass strata, a leave block too dense, a shape mismatch); the caller then takes
// the dense path.
static bool vmtc_z0_sparse(VModelTurnCache *c, const VStratum *s,
                           const VBucket *b, VMTCSlot *slot,
                           const uint8_t *leave_indices, int leave_len,
                           double *out) {
    if (c->m->play_unseen != 0 || !c->has_unseen || s->kind == 0) return false;
    if (slot->sp_n == -2) vmtc_fill_sparse(slot, s, leave_indices, leave_len, c->sl);
    if (slot->sp_n < 0) return false;
    const int off = b->has_diff ? 1 : 0;
    bool ind_mask[27] = {0};
    int n_masked = 0;
    for (int i = 0; i < b->n_bag; i++) {
        uint8_t t = b->bag_tiles[i];
        if (t >= 27) return false;
        if (!ind_mask[t]) { ind_mask[t] = true; n_masked++; }
    }
    const int n_expect = off + slot->sp_len + (27 - n_masked) + b->n_bag +
                         (s->kind == 1 ? 27 : 0);
    if (n_expect != b->n_coefs) return false;

    const float *cf = b->coefs;
    double z0 = b->intercept;
    for (int k = 0; k < slot->sp_n; k++) {
        z0 += (double)slot->sp_val[k] * (double)cf[off + slot->sp_off[k]];
    }
    int wi = off + slot->sp_len;
    const int ut = c->unseen_total;
    const int tp = (ut > 0) ? tiles_played_count(ut, s->leave_length) : 0;
    for (int t = 0; t < 27; t++) {
        if (ind_mask[t]) continue;
        float bv = 0.0f;
        if (ut > 0 && tp > 0) {
            bv = (float)((double)tp * (double)c->unseen[t] / (double)ut);
        }
        z0 += (double)bv * (double)cf[wi++];
    }
    for (int i = 0; i < b->n_bag; i++) {
        float v = (c->unseen[b->bag_tiles[i]] == (int)b->bag_counts[i]) ? 1.0f : 0.0f;
        z0 += (double)v * (double)cf[wi++];
    }
    if (s->kind == 1) {
        for (int t = 0; t < 27; t++) {
            int exch = (int)c->rack_counts[t] - (int)slot->leave_counts[t];
            z0 += (double)(float)exch * (double)cf[wi++];
        }
    }
    *out = z0;
    return true;
}

float vmodel_turn_cache_predict(VModelTurnCache *c,
                                const uint8_t *leave_indices, int leave_len,
                                int kind, int diff) {
    const VModel *m = c->m;
    const int *uv = c->has_unseen ? c->unseen : NULL;
    // Paths whose features are not "leave part + linear diff term":
    //  - passes rewrite diff into sixpass_diff before bucketing;
    //  - terminal-zero-score (turn 6) uses a different, pts-indicator schema.
    if (kind == 0 || m->turn == 6 || leave_len > VMODEL_RACK_SIZE) {
        return vmodel_predict(m, c->rack, c->rack_len, leave_indices, leave_len,
                              kind, diff, m->turn, uv, c->sl);
    }

    // Canonical key: kind (2 bits), length (3 bits), sorted tiles (5 bits each).
    uint8_t srt[VMODEL_RACK_SIZE];
    for (int i = 0; i < leave_len; i++) srt[i] = leave_indices[i];
    for (int i = 1; i < leave_len; i++) {
        uint8_t v = srt[i];
        int j = i - 1;
        while (j >= 0 && srt[j] > v) { srt[j + 1] = srt[j]; j--; }
        srt[j + 1] = v;
    }
    uint64_t key = ((uint64_t)(kind & 3) << 38) | ((uint64_t)leave_len << 35);
    for (int i = 0; i < leave_len; i++) key |= (uint64_t)(srt[i] & 31) << (5 * i);
    key |= 1ULL << 40;  // never 0, so 0 can mark an empty slot

    uint32_t h = (uint32_t)(vmtc_mix(key) & (VMTC_SLOTS - 1));
    VMTCSlot *slot = NULL;
    for (int probe = 0; probe < VMTC_SLOTS; probe++) {
        VMTCSlot *cand = &c->slots[(h + probe) & (VMTC_SLOTS - 1)];
        if (cand->gen != c->gen) {      // empty for this turn: claim it
            cand->gen = c->gen;
            cand->key = key;
            VModelLeaveType lt = vmodel_classify_leave(leave_indices, leave_len);
            cand->si = (int16_t)vmodel_stratum_index(m, kind, leave_len, lt,
                                                     m->turn);
            cand->sp_n = -2;  // sparse leave block not built yet
            slot = cand;
            break;
        }
        if (cand->key == key) { slot = cand; break; }
    }
    if (!slot || slot->si < 0) {
        // Table full (cannot happen at the bound above) or no stratum: let the
        // reference path decide, including its warning on a missing stratum.
        return vmodel_predict(m, c->rack, c->rack_len, leave_indices, leave_len,
                              kind, diff, m->turn, uv, c->sl);
    }
    const VStratum *s = &m->strata[slot->si];
    if (s->n_buckets > VMTC_MAXB) {
        return vmodel_predict(m, c->rack, c->rack_len, leave_indices, leave_len,
                              kind, diff, m->turn, uv, c->sl);
    }
    int bi = vmtc_bucket_index(s, diff);
    if (bi < 0) {
        return vmodel_predict(m, c->rack, c->rack_len, leave_indices, leave_len,
                              kind, diff, m->turn, uv, c->sl);
    }
    const VBucket *b = &s->buckets[bi];
    if (slot->bgen[bi] != c->gen) {
        // MAGPIE_VMODEL_SPARSE=0 forces the dense path for A/B;
        // MAGPIE_VMODEL_CACHE_VERIFY=1 computes both and counts any z0 that is
        // not bit-identical.
        static _Thread_local int sp_on = -1, sp_verify = 0;
        if (sp_on < 0) {
            const char *e = getenv("MAGPIE_VMODEL_SPARSE");
            sp_on = (e && e[0] == '0') ? 0 : 1;
            const char *v = getenv("MAGPIE_VMODEL_CACHE_VERIFY");
            sp_verify = (v && v[0] == '1') ? 1 : 0;
        }
        double z0 = 0.0;
        bool have = sp_on && vmtc_z0_sparse(c, s, b, slot, leave_indices,
                                            leave_len, &z0);
        if (!have || sp_verify) {
            float feats[1024];
            int n = vmodel_extract_features(feats, (int)(sizeof(feats) / sizeof(feats[0])),
                                            s, b, c->rack, c->rack_len,
                                            leave_indices, leave_len,
                                            0, m->bag, m->turn, uv,
                                            m->play_unseen, m->pass_unseen, c->sl);
            if (n < 0) return -1.0f;
            double zd = b->intercept;
            for (int i = b->has_diff ? 1 : 0; i < n; i++) {
                zd += (double)feats[i] * (double)b->coefs[i];
            }
            if (have) {
                atomic_fetch_add_explicit(&g_vmtc_sparse_checks, 1,
                                          memory_order_relaxed);
                if (memcmp(&zd, &z0, sizeof(zd)) != 0)
                    atomic_fetch_add_explicit(&g_vmtc_sparse_mismatch, 1,
                                              memory_order_relaxed);
            }
            z0 = zd;
        }
        slot->z0[bi] = z0;
        slot->bgen[bi] = c->gen;
    }
    double z = slot->z0[bi];
    if (b->has_diff) z += (double)(float)diff * (double)b->coefs[0];
    return sigmoid((float)z);
}
