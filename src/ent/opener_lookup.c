// opener_lookup.c — mmap'd binary lookup of opener movegen output.
// Replaces magpie's per-turn movegen on empty board (T2-T6 in EB cycle).
//
// Built by Python tool /Users/eric/B/bots/winpct/scripts/compute_lookup_table.py
// from the 3.2M-rack opener enumeration.

#include "opener_lookup.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../util/io_util.h"
#include "letter_distribution.h"
#include "rack.h"

#define MAGIC "MAGOPNR1"
#define MAGIC_LEN 8
// Version 2: row/col stored in magpie's internal swapped convention for
// vertical plays (matches set_play_for_record), full compare_moves sort
// cascade applied at write time, exchange tile lists in ML-ascending order.
#define VERSION 2

#define HEADER_SIZE 64
#define RACK_REC_SIZE 36   // 28 (key) + 4 (blob_off) + 2 (n_plays) + 2 (pad)
#define PLAY_REC_SIZE 37   // matches OpenerPlay packed layout

typedef struct __attribute__((packed)) {
  uint8_t key[28];
  uint32_t blob_off;
  uint16_t n_plays;
  uint16_t _pad;
} RackIndexEntry;

struct OpenerLookup {
  void *map_base;
  size_t map_size;
  uint32_t num_racks;
  uint64_t total_plays;
  const RackIndexEntry *racks;     // num_racks entries
  const uint8_t *blob;             // total_plays * PLAY_REC_SIZE bytes
};

// FNV-1a 32-bit, matches compute_lookup_table.py's hash.
static uint32_t fnv1a_32(const char *s) {
  uint32_t h = 0x811C9DC5u;
  for (; *s; s++) {
    h ^= (unsigned char)*s;
    h *= 0x01000193u;
  }
  return h;
}

OpenerLookup *opener_lookup_create(const char *path,
                                   const LetterDistribution *ld) {
  (void)ld;  // currently used only for letter encoding consistency check (TODO)
  int fd = open(path, O_RDONLY);
  if (fd < 0) {
    log_warn("opener_lookup: cannot open %s", path);
    return NULL;
  }
  struct stat st;
  if (fstat(fd, &st) != 0) {
    log_warn("opener_lookup: fstat failed on %s", path);
    close(fd);
    return NULL;
  }
  size_t map_size = (size_t)st.st_size;
  if (map_size < HEADER_SIZE) {
    log_warn("opener_lookup: file too small (%zu bytes)", map_size);
    close(fd);
    return NULL;
  }
  // MAP_POPULATE pre-faults pages on Linux (no-op on macOS); first run takes
  // ~10s for 3 GB, but eliminates random page faults during the hot loop.
  int flags = MAP_PRIVATE;
#ifdef MAP_POPULATE
  flags |= MAP_POPULATE;
#endif
  void *base = mmap(NULL, map_size, PROT_READ, flags, fd, 0);
  close(fd);
  if (base == MAP_FAILED) {
    log_warn("opener_lookup: mmap failed");
    return NULL;
  }

  const uint8_t *p = (const uint8_t *)base;
  if (memcmp(p, MAGIC, MAGIC_LEN) != 0) {
    log_warn("opener_lookup: bad magic in %s", path);
    munmap(base, map_size);
    return NULL;
  }
  uint32_t version, lex_hash, num_racks;
  uint64_t total_plays, racks_off, blob_off;
  memcpy(&version, p + 8, 4);
  memcpy(&lex_hash, p + 12, 4);
  memcpy(&num_racks, p + 16, 4);
  memcpy(&total_plays, p + 20, 8);
  memcpy(&racks_off, p + 28, 8);
  memcpy(&blob_off, p + 36, 8);

  if (version != VERSION) {
    log_warn("opener_lookup: version mismatch (got %u, expected %u)",
             version, VERSION);
    munmap(base, map_size);
    return NULL;
  }
  // CSW24 hash check (compute_lookup_table.py defaults to "CSW24").
  uint32_t expected_hash = fnv1a_32("CSW24");
  if (lex_hash != expected_hash) {
    log_warn("opener_lookup: lex_hash mismatch (got 0x%08x, expected 0x%08x); "
             "table may be for a different lexicon", lex_hash, expected_hash);
    // Don't reject — caller may know what they're doing; just warn.
  }

  OpenerLookup *ol = (OpenerLookup *)malloc(sizeof(OpenerLookup));
  ol->map_base = base;
  ol->map_size = map_size;
  ol->num_racks = num_racks;
  ol->total_plays = total_plays;
  ol->racks = (const RackIndexEntry *)(p + racks_off);
  ol->blob = p + blob_off;

  fprintf(stderr,
          "opener_lookup: loaded %s (%u racks, %llu plays, %.2f GB)\n",
          path, num_racks, (unsigned long long)total_plays,
          map_size / 1e9);
  return ol;
}

void opener_lookup_destroy(OpenerLookup *ol) {
  if (!ol) return;
  if (ol->map_base) munmap(ol->map_base, ol->map_size);
  free(ol);
}

const OpenerPlay *opener_lookup_query(const OpenerLookup *ol,
                                      const char *canon_rack,
                                      int *out_count) {
  if (!ol || !canon_rack) {
    if (out_count) *out_count = 0;
    return NULL;
  }
  // Binary search on 28-byte key.
  int lo = 0, hi = (int)ol->num_racks;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    int cmp = memcmp(canon_rack, ol->racks[mid].key, 28);
    if (cmp == 0) {
      const RackIndexEntry *e = &ol->racks[mid];
      if (out_count) *out_count = e->n_plays;
      return (const OpenerPlay *)(ol->blob + e->blob_off);
    }
    if (cmp < 0) hi = mid;
    else lo = mid + 1;
  }
  if (out_count) *out_count = 0;
  return NULL;
}

void opener_lookup_canonical_rack(const Rack *rack,
                                  const LetterDistribution *ld,
                                  char canon_out[28]) {
  // Sorted ASCII (A..Z first), then blanks ('?'), NUL-padded to 28.
  memset(canon_out, 0, 28);
  int idx = 0;
  // ld_ml_to_hl[1]..[26] map to "A".."Z"; ld_ml_to_hl[0] is blank.
  for (MachineLetter ml = 1; ml < (MachineLetter)rack_get_dist_size(rack); ml++) {
    uint16_t n = rack_get_letter(rack, ml);
    char ch = ld->ld_ml_to_hl[ml][0];
    for (uint16_t i = 0; i < n && idx < 28; i++) {
      canon_out[idx++] = ch;
    }
  }
  uint16_t blanks = rack_get_letter(rack, 0);
  for (uint16_t i = 0; i < blanks && idx < 28; i++) {
    canon_out[idx++] = '?';
  }
}
