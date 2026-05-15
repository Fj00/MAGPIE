#include "outcome_priors.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OUTCOME_PRIORS_MAGIC   0x5450524fU  // "OPRT" little-endian
#define OUTCOME_PRIORS_VERSION 1

// Disk record layout (24 bytes) — must match build_t6_baseline.py
// write_priors_binary:
//   u8  leave_len, n_blanks, n_S, n_J, n_Q, n_X, n_Z, n_vowels;
//   i16 leave_pts_residual, score;
//   f32 p_w, p_l, p_t;
typedef struct __attribute__((packed)) {
  uint8_t  leave_len;
  uint8_t  n_blanks;
  uint8_t  n_S;
  uint8_t  n_J;
  uint8_t  n_Q;
  uint8_t  n_X;
  uint8_t  n_Z;
  uint8_t  n_vowels;
  int16_t  leave_pts_residual;
  int16_t  score;
  float    p_w;
  float    p_l;
  float    p_t;
} OPDiskRow;

typedef struct {
  uint64_t key;   // packed OptionFKey (0 = empty slot)
  float    p_w;
  float    p_l;
  float    p_t;
} OPSlot;

struct OutcomePriors {
  int      num_buckets;
  uint32_t cap_mask;   // capacity - 1 (capacity is power of two)
  OPSlot  *table;      // open-addressed
};

// Pack OptionFKey into a u64 hash key. Layout (low to high):
//   leave_len   : 4 bits  (0..7)
//   n_blanks    : 2 bits  (0..2)
//   n_S         : 3 bits  (0..4)
//   n_J         : 1 bit   (0..1)
//   n_Q         : 1 bit
//   n_X         : 1 bit
//   n_Z         : 1 bit
//   n_vowels    : 3 bits  (0..7)
//   leave_pts_residual : 8 bits  (0..255)
//   score              : 10 bits (signed -512..511, stored unsigned offset)
// Total: 34 bits — well under 64. Reserve 0 as sentinel by ORing the high bit.
static uint64_t pack_key(const OptionFKey *k) {
  uint64_t v = 0;
  v |= ((uint64_t)k->leave_len & 0xF);
  v |= ((uint64_t)k->n_blanks & 0x3) << 4;
  v |= ((uint64_t)k->n_S & 0x7) << 6;
  v |= ((uint64_t)k->n_J & 0x1) << 9;
  v |= ((uint64_t)k->n_Q & 0x1) << 10;
  v |= ((uint64_t)k->n_X & 0x1) << 11;
  v |= ((uint64_t)k->n_Z & 0x1) << 12;
  v |= ((uint64_t)k->n_vowels & 0x7) << 13;
  v |= ((uint64_t)(uint8_t)k->leave_pts_residual) << 16;
  v |= ((uint64_t)((uint16_t)(k->score + 512) & 0x3FF)) << 24;
  v |= 1ULL << 63;  // mark non-empty
  return v;
}

// SplitMix64 for hashing the packed key.
static uint64_t mix64(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

static uint32_t next_pow2(uint32_t v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return v + 1;
}

OutcomePriors *outcome_priors_load(const char *path) {
  FILE *fh = fopen(path, "rb");
  if (!fh) {
    fprintf(stderr, "outcome_priors: cannot open %s\n", path);
    return NULL;
  }
  uint32_t header[3];
  if (fread(header, sizeof(header), 1, fh) != 1) {
    fprintf(stderr, "outcome_priors: short read on header\n");
    fclose(fh);
    return NULL;
  }
  if (header[0] != OUTCOME_PRIORS_MAGIC) {
    fprintf(stderr, "outcome_priors: bad magic 0x%08x (expected 0x%08x)\n",
            header[0], OUTCOME_PRIORS_MAGIC);
    fclose(fh);
    return NULL;
  }
  if (header[1] != OUTCOME_PRIORS_VERSION) {
    fprintf(stderr, "outcome_priors: version mismatch %u (expected %u)\n",
            header[1], OUTCOME_PRIORS_VERSION);
    fclose(fh);
    return NULL;
  }
  uint32_t n = header[2];
  if (n == 0) {
    fprintf(stderr, "outcome_priors: zero buckets in %s\n", path);
    fclose(fh);
    return NULL;
  }
  // Open-addressed table at 2× load factor.
  uint32_t cap = next_pow2(n * 2);
  if (cap < 16) cap = 16;
  OutcomePriors *op = (OutcomePriors *)calloc(1, sizeof(OutcomePriors));
  if (!op) { fclose(fh); return NULL; }
  op->num_buckets = (int)n;
  op->cap_mask = cap - 1;
  op->table = (OPSlot *)calloc(cap, sizeof(OPSlot));
  if (!op->table) { free(op); fclose(fh); return NULL; }

  for (uint32_t i = 0; i < n; i++) {
    OPDiskRow row;
    if (fread(&row, sizeof(row), 1, fh) != 1) {
      fprintf(stderr, "outcome_priors: short read at row %u\n", i);
      outcome_priors_destroy(op);
      fclose(fh);
      return NULL;
    }
    OptionFKey k = {
      .leave_len = row.leave_len, .n_blanks = row.n_blanks,
      .n_S = row.n_S, .n_J = row.n_J, .n_Q = row.n_Q,
      .n_X = row.n_X, .n_Z = row.n_Z, .n_vowels = row.n_vowels,
      .leave_pts_residual = row.leave_pts_residual,
      .score = row.score,
    };
    uint64_t key = pack_key(&k);
    uint32_t slot = (uint32_t)(mix64(key) & op->cap_mask);
    while (op->table[slot].key != 0) {
      // Should never collide with same key (priors CSV is deduped).
      if (op->table[slot].key == key) break;
      slot = (slot + 1) & op->cap_mask;
    }
    op->table[slot].key = key;
    op->table[slot].p_w = row.p_w;
    op->table[slot].p_l = row.p_l;
    op->table[slot].p_t = row.p_t;
  }
  fclose(fh);
  fprintf(stderr,
          "outcome_priors: loaded %d buckets from %s "
          "(table cap=%u)\n", op->num_buckets, path, op->cap_mask + 1);
  return op;
}

void outcome_priors_destroy(OutcomePriors *op) {
  if (!op) return;
  free(op->table);
  free(op);
}

int outcome_priors_num_buckets(const OutcomePriors *op) {
  return op ? op->num_buckets : 0;
}

// Mirrors python build_t6_baseline.py:138-151. leave_str is ASCII
// (e.g. "RY?"). Tile point values for residual:
//   Q,Z = 10; J,X = 8; S = 1.
bool outcome_priors_compute_key(const PlayRecord *pr, OptionFKey *out) {
  static const int8_t TILE_PTS[26] = {
    1,3,3,2,1,4,2,4,1,8,5,1,3,1,1,3,10,1,1,1,1,4,4,8,4,10
  };
  uint8_t nB=0, nS=0, nJ=0, nQ=0, nX=0, nZ=0, nV=0;
  int pts = 0;
  for (uint8_t i = 0; i < pr->leave_len; i++) {
    char c = pr->leave_str[i];
    if (c == '?') { nB++; continue; }
    if (c < 'A' || c > 'Z') return false;
    int idx = c - 'A';
    pts += TILE_PTS[idx];
    switch (c) {
      case 'S': nS++; break;
      case 'J': nJ++; break;
      case 'Q': nQ++; break;
      case 'X': nX++; break;
      case 'Z': nZ++; break;
      case 'A': case 'E': case 'I': case 'O': case 'U': nV++; break;
      default: break;
    }
  }
  out->leave_len = pr->leave_len;
  out->n_blanks  = nB;
  out->n_S = nS; out->n_J = nJ; out->n_Q = nQ;
  out->n_X = nX; out->n_Z = nZ; out->n_vowels = nV;
  out->leave_pts_residual =
      (int16_t)(pts - 10*nQ - 10*nZ - 8*nJ - 8*nX - nS);
  out->score = pr->score;
  return true;
}

bool outcome_priors_lookup(const OutcomePriors *op, const OptionFKey *key,
                            float *out_pw, float *out_pl, float *out_pt) {
  if (!op || !key) return false;
  uint64_t k = pack_key(key);
  uint32_t slot = (uint32_t)(mix64(k) & op->cap_mask);
  for (;;) {
    OPSlot *s = &op->table[slot];
    if (s->key == 0) return false;
    if (s->key == k) {
      if (out_pw) *out_pw = s->p_w;
      if (out_pl) *out_pl = s->p_l;
      if (out_pt) *out_pt = s->p_t;
      return true;
    }
    slot = (slot + 1) & op->cap_mask;
  }
}
