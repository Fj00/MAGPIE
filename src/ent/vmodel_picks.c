#include "vmodel_picks.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../def/board_defs.h"
#include "../def/game_history_defs.h"
#include "../def/letter_distribution_defs.h"
#include "../util/io_util.h"
#include "rack.h"

// English-only: blank=0, A=1..Z=26 (matches BLANK_MACHINE_LETTER + the
// vmodel pipeline's python-canonical ordering). Picks files are
// English-only (CSW24 opener pipeline).

static inline uint8_t ascii_to_ml(char c) {
  if (c == '?') return 0;
  if (c >= 'A' && c <= 'Z') return (uint8_t)(c - 'A' + 1);
  if (c >= 'a' && c <= 'z') {
    // Lowercase = blank-played-as-letter. Returns ML with BLANK_MASK set
    // (e.g. blank-as-'a' → 1 | 0x80 = 129) so play_move_on_board records
    // the correct letter on the board while keeping the played-tile
    // bookkeeping aware it was a blank.
    return (uint8_t)((c - 'a' + 1) | BLANK_MASK);
  }
  return INVALID_LETTER;
}

int vmodel_picks_key_for_rack(const Rack *rack, uint8_t *out_buf) {
  memset(out_buf, 0, VMODEL_PICKS_RACK_KEY_LEN);
  int n = 0;
  const uint16_t dist_size = rack_get_dist_size(rack);
  // ML 0 = BLANK_MACHINE_LETTER, ML 1..26 = A..Z. Iterating in ML order
  // automatically gives blanks-first canonical form.
  for (uint16_t ml = 0;
       ml < dist_size && n < VMODEL_PICKS_RACK_KEY_LEN; ml++) {
    const int c = rack_get_letter(rack, ml);
    for (int k = 0; k < c && n < VMODEL_PICKS_RACK_KEY_LEN; k++) {
      out_buf[n++] = (ml == 0) ? '?' : (char)('A' + ml - 1);
    }
  }
  return n;
}

const Move *vmodel_picks_lookup(const VModelPicks *picks,
                                 const uint8_t *rack_key) {
  if (!picks || picks->n_entries == 0) return NULL;
  int lo = 0, hi = picks->n_entries;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    int cmp = memcmp(picks->keys[mid], rack_key,
                     VMODEL_PICKS_RACK_KEY_LEN);
    if (cmp == 0) return &picks->moves[mid];
    if (cmp < 0) lo = mid + 1;
    else hi = mid;
  }
  return NULL;
}

// --- Parsing action_repr ---

// Parse "pass" / "(exch ABC)" / "8H FARM" / "H8 FARM" into a Move.
// `score` comes from the .picks entry; equity is left as 0 (unused
// after pick decision). Returns true on success.
static bool parse_action_repr(const char *repr, int repr_len,
                                int16_t score, Move *out) {
  if (repr_len <= 0) return false;

  // PASS
  if (repr_len == 4 && memcmp(repr, "pass", 4) == 0) {
    move_set_as_pass(out);
    return true;
  }

  // EXCHANGE "(exch ABCDEFG)" or "(exch )"  (empty for 0-tile exchange,
  // shouldn't happen in practice but tolerated).
  if (repr_len >= 7 && memcmp(repr, "(exch ", 6) == 0) {
    MachineLetter mls[8] = {0};
    int n = 0;
    for (int i = 6; i < repr_len && repr[i] != ')'; i++) {
      if (n >= 8) return false;  // shouldn't happen for opener (max 7)
      uint8_t ml = ascii_to_ml(repr[i]);
      if (ml == INVALID_LETTER) return false;
      // For exchanges, the inner tiles are raw rack tiles (uppercase or
      // '?'), never blank-as-letter. Strip BLANK_MASK if somehow set.
      mls[n++] = ml & UNBLANK_MASK;
    }
    move_set_all(out, mls, /*leftstrip=*/0, /*rightstrip=*/n - 1,
                  /*score=*/0, /*row=*/0, /*col=*/0, /*tiles_played=*/n,
                  /*dir=*/0, GAME_EVENT_EXCHANGE, /*leave_value=*/0);
    return true;
  }

  // PLAY: position prefix is either "<num><letter>" (horizontal, e.g.
  // "8H") or "<letter><num>" (vertical, e.g. "H8"), followed by a
  // space, then the word (uppercase = real tile, lowercase = blank).
  int row = 0, col = 0, dir = 0;
  int idx = 0;
  if (repr[idx] >= '0' && repr[idx] <= '9') {
    // Horizontal: "8H WORD"
    int n = 0;
    while (idx < repr_len && repr[idx] >= '0' && repr[idx] <= '9') {
      n = n * 10 + (repr[idx] - '0');
      idx++;
    }
    if (idx >= repr_len || repr[idx] < 'A' || repr[idx] > 'O') return false;
    row = n - 1;
    col = repr[idx] - 'A';
    dir = 0;  // horizontal
    idx++;
  } else if (repr[idx] >= 'A' && repr[idx] <= 'O') {
    // Vertical: "H8 WORD"
    col = repr[idx] - 'A';
    idx++;
    int n = 0;
    while (idx < repr_len && repr[idx] >= '0' && repr[idx] <= '9') {
      n = n * 10 + (repr[idx] - '0');
      idx++;
    }
    row = n - 1;
    dir = 1;  // vertical
  } else {
    return false;
  }
  if (idx >= repr_len || repr[idx] != ' ') return false;
  idx++;  // skip space

  // Word tiles. At opener no play-through tiles; tiles_length =
  // tiles_played = word length.
  MachineLetter tiles[BOARD_DIM] = {0};
  int word_len = 0;
  while (idx < repr_len && repr[idx] != '\0') {
    if (word_len >= BOARD_DIM) return false;
    uint8_t ml = ascii_to_ml(repr[idx]);
    if (ml == INVALID_LETTER) return false;
    tiles[word_len++] = ml;
    idx++;
  }
  if (word_len == 0) return false;

  move_set_all(out, tiles, /*leftstrip=*/0, /*rightstrip=*/word_len - 1,
                /*score=*/(Equity)score * 1000,  // score is int; Equity = score * 1000
                /*row=*/row, /*col=*/col,
                /*tiles_played=*/word_len, /*dir=*/dir,
                GAME_EVENT_TILE_PLACEMENT_MOVE, /*leave_value=*/0);
  return true;
}

void vmodel_picks_destroy(VModelPicks *picks) {
  if (!picks) return;
  free(picks->keys);
  free(picks->moves);
  free(picks);
}

VModelPicks *vmodel_picks_create(const char *picks_path) {
  FILE *f = fopen(picks_path, "rb");
  if (!f) {
    log_warn("vmodel_picks: cannot open %s", picks_path);
    return NULL;
  }
  uint32_t magic, version, model_turn, n_entries, entry_size, max_repr_len;
  uint64_t reserved;
  if (fread(&magic, sizeof(magic), 1, f) != 1 ||
      fread(&version, sizeof(version), 1, f) != 1 ||
      fread(&model_turn, sizeof(model_turn), 1, f) != 1 ||
      fread(&n_entries, sizeof(n_entries), 1, f) != 1 ||
      fread(&entry_size, sizeof(entry_size), 1, f) != 1 ||
      fread(&max_repr_len, sizeof(max_repr_len), 1, f) != 1 ||
      fread(&reserved, sizeof(reserved), 1, f) != 1) {
    log_warn("vmodel_picks: short header in %s", picks_path);
    fclose(f); return NULL;
  }
  if (magic != VMODEL_PICKS_MAGIC) {
    log_warn("vmodel_picks: bad magic 0x%08x in %s "
             "(want 0x%08x)", magic, picks_path, VMODEL_PICKS_MAGIC);
    fclose(f); return NULL;
  }
  if (version != VMODEL_PICKS_VERSION) {
    log_warn("vmodel_picks: unsupported version %u in %s",
             version, picks_path);
    fclose(f); return NULL;
  }
  if (model_turn < 1 || model_turn > 6) {
    log_warn("vmodel_picks: bad model_turn %u in %s",
             model_turn, picks_path);
    fclose(f); return NULL;
  }
  const uint32_t expected_entry =
      VMODEL_PICKS_RACK_KEY_LEN + 2 + 1 + max_repr_len;
  if (entry_size != expected_entry) {
    log_warn("vmodel_picks: entry_size %u != expected %u in %s",
             entry_size, expected_entry, picks_path);
    fclose(f); return NULL;
  }

  VModelPicks *picks = calloc(1, sizeof(VModelPicks));
  if (!picks) { fclose(f); return NULL; }
  picks->model_turn = (int)model_turn;
  picks->n_entries = (int)n_entries;
  picks->keys = malloc(sizeof(uint8_t[VMODEL_PICKS_RACK_KEY_LEN])
                       * n_entries);
  picks->moves = calloc(n_entries, sizeof(Move));
  if (!picks->keys || !picks->moves) {
    log_warn("vmodel_picks: alloc fail");
    vmodel_picks_destroy(picks); fclose(f); return NULL;
  }

  // Per-entry buffer for the on-disk row body (16 bytes for entry_size=35).
  uint8_t *row = malloc(entry_size);
  if (!row) {
    vmodel_picks_destroy(picks); fclose(f); return NULL;
  }
  uint64_t parse_failures = 0;
  for (uint32_t i = 0; i < n_entries; i++) {
    if (fread(row, entry_size, 1, f) != 1) {
      log_warn("vmodel_picks: short read at entry %u in %s",
               i, picks_path);
      free(row); vmodel_picks_destroy(picks); fclose(f); return NULL;
    }
    memcpy(picks->keys[i], row, VMODEL_PICKS_RACK_KEY_LEN);
    int16_t score = 0;
    memcpy(&score, row + VMODEL_PICKS_RACK_KEY_LEN, 2);
    uint8_t repr_len = row[VMODEL_PICKS_RACK_KEY_LEN + 2];
    const char *repr = (const char *)
        (row + VMODEL_PICKS_RACK_KEY_LEN + 2 + 1);
    if (repr_len > max_repr_len) {
      log_warn("vmodel_picks: bad repr_len %u at entry %u (max %u)",
               repr_len, i, max_repr_len);
      parse_failures++;
      continue;
    }
    if (!parse_action_repr(repr, repr_len, score, &picks->moves[i])) {
      parse_failures++;
      if (parse_failures <= 5) {
        char tmp[64] = {0};
        int n = repr_len < (int)sizeof(tmp) - 1 ? repr_len
                                                 : (int)sizeof(tmp) - 1;
        memcpy(tmp, repr, n);
        log_warn("vmodel_picks: parse fail at entry %u repr=%s",
                 i, tmp);
      }
    }
  }
  free(row);
  fclose(f);

  if (parse_failures > 0) {
    log_warn("vmodel_picks: %llu parse failures out of %u entries",
             (unsigned long long)parse_failures, n_entries);
  }
  log_info("vmodel_picks: loaded %s — %u entries, T%d slot",
           picks_path, n_entries, picks->model_turn);
  return picks;
}
