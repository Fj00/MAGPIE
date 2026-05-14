#include "play_index.h"

#include <fcntl.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../def/letter_distribution_defs.h"
#include "../util/io_util.h"

// Per-cell metadata layout in cells.bin (24 bytes; matches the Python
// builder's struct.pack format: <BBBBBBBB iiI f).
typedef struct __attribute__((packed)) {
  uint8_t  kind;          // 0=stratum 1=tile 2=pair 3=bag_tile
  uint8_t  length;
  uint8_t  type;          // 0=all 1=cons 2=mixed 3=vowel
  uint8_t  exchange;
  uint8_t  sub_count;
  uint8_t  _pad;
  uint8_t  sub_ml0;
  uint8_t  sub_ml1;
  int32_t  diff_min;
  int32_t  diff_max;
  uint32_t target;
  float    rarity;
} CellMeta;

// Per-play header in plays.bin (12 bytes; <I B h B H B B).
// Fields packed, cell_ids and action_repr/leave_str follow.
typedef struct __attribute__((packed)) {
  uint32_t rack_id;
  uint8_t  action_kind;
  int16_t  score;
  uint8_t  ar_len;
  uint16_t n_cells;
  uint8_t  leave_len;
  uint8_t  _pad;
} PlayHeader;

#define RACK_BYTES 8

// Sampler tunables.
#define TOP_CELL_HEAP_K 64
#define MAX_PLAYS_PER_CELL_SCAN 2000

struct PlayIndex {
  // Counts (loaded from meta.json).
  uint32_t num_racks;
  uint32_t num_plays;
  uint32_t num_cells;

  // racks.bin: [num_racks * RACK_BYTES] packed ASCII, NUL-padded.
  const char *racks;        // mmap base
  size_t racks_len;

  // cells.bin: [num_cells * sizeof(CellMeta)].
  const CellMeta *cells;
  size_t cells_len;

  // plays.bin: variable-length records.
  const uint8_t *plays;
  size_t plays_len;

  // plays_by_rack.idx: [num_racks+1 u64 rack_first_play_id]
  //                  + [num_plays u64 play_byte_offset]
  const uint64_t *rack_first_play;   // [num_racks+1]
  const uint64_t *play_byte_offset;  // [num_plays]
  size_t rack_idx_len;

  // plays_by_cell.bin: [total_refs u32 play_id]
  const uint32_t *plays_by_cell;
  size_t plays_by_cell_len;

  // plays_by_cell.idx: [num_cells+1 u64 byte_offset_into_plays_by_cell.bin]
  const uint64_t *cell_play_off;
  size_t cell_idx_len;

  // Resolved at load: cell_id -> ForceTarget* (NULL if cell missing
  // from force_table, e.g. when the index was built against a
  // different force_targets layout).
  ForceTarget **cell_to_target;

  // Per-cell supply weight = 1 / (cell_play_off[c+1] - cell_play_off[c]).
  // Used for rarity-weighted scoring: rare-supply cells dominate.
  // Computed once at load.
  float *inv_supply;
};

// ---------- Helpers ----------

static void *mmap_file(const char *path, size_t *out_len) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) {
    fprintf(stderr, "play_index: cannot open %s\n", path);
    return NULL;
  }
  struct stat st;
  if (fstat(fd, &st) != 0) {
    close(fd);
    return NULL;
  }
  void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (p == MAP_FAILED) {
    fprintf(stderr, "play_index: mmap failed for %s\n", path);
    return NULL;
  }
  *out_len = (size_t)st.st_size;
  return p;
}

// Minimal meta.json parser — looks for "key": NUMBER patterns. Avoids
// pulling in a real JSON dep. Returns 0 on success.
static int parse_meta_json(const char *path, uint32_t *num_racks,
                            uint32_t *num_plays, uint32_t *num_cells) {
  FILE *f = fopen(path, "r");
  if (!f) return -1;
  char buf[4096];
  size_t n = fread(buf, 1, sizeof(buf) - 1, f);
  fclose(f);
  buf[n] = '\0';

  unsigned long long v;
  const char *p;
  if ((p = strstr(buf, "\"num_racks\""))) {
    if (sscanf(p, "\"num_racks\"%*[^0-9]%llu", &v) == 1) *num_racks = (uint32_t)v;
  }
  if ((p = strstr(buf, "\"num_plays\""))) {
    if (sscanf(p, "\"num_plays\"%*[^0-9]%llu", &v) == 1) *num_plays = (uint32_t)v;
  }
  if ((p = strstr(buf, "\"num_cells\""))) {
    if (sscanf(p, "\"num_cells\"%*[^0-9]%llu", &v) == 1) *num_cells = (uint32_t)v;
  }
  return 0;
}

static LeaveType type_int_to_enum(uint8_t t) {
  switch (t) {
    case 0: return LEAVE_TYPE_ALL;
    case 1: return LEAVE_TYPE_CONS;
    case 2: return LEAVE_TYPE_MIXED;
    case 3: return LEAVE_TYPE_VOWEL;
  }
  return LEAVE_TYPE_ALL;
}

// ---------- Loader ----------

PlayIndex *play_index_create(const char *dir_path, ForceTable *force_table,
                              const LetterDistribution *ld) {
  PlayIndex *idx = (PlayIndex *)calloc(1, sizeof(PlayIndex));
  if (!idx) return NULL;

  char path[1024];

  // meta.json
  snprintf(path, sizeof(path), "%s/meta.json", dir_path);
  if (parse_meta_json(path, &idx->num_racks, &idx->num_plays,
                      &idx->num_cells) != 0) {
    fprintf(stderr, "play_index: failed to parse %s\n", path);
    free(idx);
    return NULL;
  }
  fprintf(stderr,
          "play_index: meta = %u racks, %u plays, %u cells\n",
          idx->num_racks, idx->num_plays, idx->num_cells);

  // racks.bin
  snprintf(path, sizeof(path), "%s/racks.bin", dir_path);
  idx->racks = (const char *)mmap_file(path, &idx->racks_len);
  if (!idx->racks) { play_index_destroy(idx); return NULL; }

  // cells.bin
  snprintf(path, sizeof(path), "%s/cells.bin", dir_path);
  idx->cells = (const CellMeta *)mmap_file(path, &idx->cells_len);
  if (!idx->cells) { play_index_destroy(idx); return NULL; }

  // plays.bin
  snprintf(path, sizeof(path), "%s/plays.bin", dir_path);
  idx->plays = (const uint8_t *)mmap_file(path, &idx->plays_len);
  if (!idx->plays) { play_index_destroy(idx); return NULL; }

  // plays_by_rack.idx
  snprintf(path, sizeof(path), "%s/plays_by_rack.idx", dir_path);
  void *rack_idx_buf = mmap_file(path, &idx->rack_idx_len);
  if (!rack_idx_buf) { play_index_destroy(idx); return NULL; }
  idx->rack_first_play = (const uint64_t *)rack_idx_buf;
  idx->play_byte_offset = idx->rack_first_play + (idx->num_racks + 1);

  // plays_by_cell.bin and .idx
  snprintf(path, sizeof(path), "%s/plays_by_cell.bin", dir_path);
  idx->plays_by_cell = (const uint32_t *)mmap_file(path,
                                                    &idx->plays_by_cell_len);
  if (!idx->plays_by_cell) { play_index_destroy(idx); return NULL; }
  snprintf(path, sizeof(path), "%s/plays_by_cell.idx", dir_path);
  idx->cell_play_off = (const uint64_t *)mmap_file(path, &idx->cell_idx_len);
  if (!idx->cell_play_off) { play_index_destroy(idx); return NULL; }

  // Resolve cell_to_target via force_table_lookup_target_by_key.
  idx->cell_to_target = (ForceTarget **)calloc(idx->num_cells,
                                                sizeof(ForceTarget *));
  if (!idx->cell_to_target) { play_index_destroy(idx); return NULL; }
  uint32_t resolved = 0;
  for (uint32_t i = 0; i < idx->num_cells; i++) {
    const CellMeta *c = &idx->cells[i];
    LeaveType lt = type_int_to_enum(c->type);
    ForceTargetKind kind = (ForceTargetKind)c->kind;
    int subleave_count = c->sub_count;
    int diff = c->diff_min;  // any diff in range works for lookup
    ForceTarget *t = force_table_lookup_target_by_key(
        force_table, /*bag*/ 93, c->length, lt, kind, c->exchange,
        c->sub_ml0, c->sub_ml1, subleave_count, diff);
    idx->cell_to_target[i] = t;
    if (t) resolved++;
    (void)ld;
  }
  fprintf(stderr,
          "play_index: resolved %u/%u cells against force_table\n",
          resolved, idx->num_cells);

  // Compute inv_supply per cell for rarity-weighted scoring.
  idx->inv_supply = (float *)malloc(sizeof(float) * idx->num_cells);
  if (!idx->inv_supply) { play_index_destroy(idx); return NULL; }
  for (uint32_t c = 0; c < idx->num_cells; c++) {
    uint64_t supply = idx->cell_play_off[c + 1] - idx->cell_play_off[c];
    idx->inv_supply[c] = (supply > 0) ? (1.0f / (float)supply) : 0.0f;
  }

  return idx;
}

void play_index_destroy(PlayIndex *idx) {
  if (!idx) return;
  if (idx->racks) munmap((void *)idx->racks, idx->racks_len);
  if (idx->cells) munmap((void *)idx->cells, idx->cells_len);
  if (idx->plays) munmap((void *)idx->plays, idx->plays_len);
  if (idx->rack_first_play)
    munmap((void *)idx->rack_first_play, idx->rack_idx_len);
  if (idx->plays_by_cell)
    munmap((void *)idx->plays_by_cell, idx->plays_by_cell_len);
  if (idx->cell_play_off)
    munmap((void *)idx->cell_play_off, idx->cell_idx_len);
  free(idx->cell_to_target);
  free(idx->inv_supply);
  free(idx);
}

int play_index_num_racks(const PlayIndex *idx) {
  return (int)idx->num_racks;
}
int play_index_num_plays(const PlayIndex *idx) {
  return (int)idx->num_plays;
}
int play_index_num_cells(const PlayIndex *idx) {
  return (int)idx->num_cells;
}

// ---------- Play record access ----------

bool play_index_get_play(const PlayIndex *idx, uint32_t play_id,
                          PlayRecord *out) {
  if (play_id >= idx->num_plays) return false;
  uint64_t off = idx->play_byte_offset[play_id];
  const uint8_t *p = idx->plays + off;
  PlayHeader hdr;
  memcpy(&hdr, p, sizeof(hdr));
  out->rack_id = hdr.rack_id;
  out->action_kind = hdr.action_kind;
  out->score = hdr.score;
  out->ar_len = hdr.ar_len;
  out->n_cells = hdr.n_cells;
  out->leave_len = hdr.leave_len;
  out->action_repr = (const char *)(p + sizeof(PlayHeader));
  out->leave_str = (const char *)(p + sizeof(PlayHeader) + hdr.ar_len);
  out->cell_ids = (const uint32_t *)(p + sizeof(PlayHeader) + hdr.ar_len +
                                     hdr.leave_len);
  return true;
}

// Score a play_id by Σ (deficit / supply) over its cell_ids — rarity
// weighted. Returns 0.0 for plays covering only drained cells.
static double score_play(const PlayIndex *idx, uint32_t play_id) {
  PlayRecord pr;
  if (!play_index_get_play(idx, play_id, &pr)) return 0.0;
  double score = 0.0;
  for (uint16_t i = 0; i < pr.n_cells; i++) {
    uint32_t cid = pr.cell_ids[i];
    ForceTarget *t = idx->cell_to_target[cid];
    if (!t) continue;
    int64_t d = atomic_load_explicit(&t->deficit, memory_order_relaxed);
    if (d > 0) score += (double)d * (double)idx->inv_supply[cid];
  }
  return score;
}

// ---------- Min-heap of (deficit, cell_id) ----------

typedef struct {
  double   weight;   // deficit / supply
  uint32_t cell_id;
} CellEntry;

static void heap_push_or_replace(CellEntry *heap, int *size, int cap,
                                  double weight, uint32_t cell_id) {
  if (*size < cap) {
    int i = (*size)++;
    heap[i].weight = weight;
    heap[i].cell_id = cell_id;
    while (i > 0) {
      int parent = (i - 1) / 2;
      if (heap[parent].weight > heap[i].weight) {
        CellEntry tmp = heap[parent];
        heap[parent] = heap[i];
        heap[i] = tmp;
        i = parent;
      } else break;
    }
  } else if (heap[0].weight < weight) {
    heap[0].weight = weight;
    heap[0].cell_id = cell_id;
    int i = 0;
    while (1) {
      int l = 2 * i + 1, r = 2 * i + 2, smallest = i;
      if (l < cap && heap[l].weight < heap[smallest].weight) smallest = l;
      if (r < cap && heap[r].weight < heap[smallest].weight) smallest = r;
      if (smallest != i) {
        CellEntry tmp = heap[smallest];
        heap[smallest] = heap[i];
        heap[i] = tmp;
        i = smallest;
      } else break;
    }
  }
}

// ---------- Samplers ----------

const char *play_index_sample_rack_deficit_aware(const PlayIndex *idx,
                                                  uint64_t seed,
                                                  uint32_t *out_rack_id) {
  // 1) Find top-K cells by rarity-weighted deficit (deficit * inv_supply).
  CellEntry heap[TOP_CELL_HEAP_K];
  int heap_size = 0;
  for (uint32_t cid = 0; cid < idx->num_cells; cid++) {
    ForceTarget *t = idx->cell_to_target[cid];
    if (!t) continue;
    int64_t d = atomic_load_explicit(&t->deficit, memory_order_relaxed);
    if (d <= 0) continue;
    double w = (double)d * (double)idx->inv_supply[cid];
    heap_push_or_replace(heap, &heap_size, TOP_CELL_HEAP_K, w, cid);
  }
  if (heap_size == 0) return NULL;

  // 2) For each top cell, walk its plays_by_cell list and score
  //    candidates. Use a small visited set keyed by play_id mod 1024
  //    as a cheap dedupe (false positives are fine — they just skip
  //    a rescore, no correctness impact).
  uint64_t visited_bitset[16] = {0};  // 1024-bit
  #define VISIT_MARK(pid) ({ \
    uint32_t k = (pid) & 1023; \
    bool was = (visited_bitset[k>>6] >> (k & 63)) & 1ULL; \
    visited_bitset[k>>6] |= (1ULL << (k & 63)); \
    was; \
  })

  double best_score = 0.0;
  uint32_t best_play = 0;
  bool have_best = false;
  uint64_t s = seed ^ 0xa1b2c3d4e5f60708ULL;

  for (int h = 0; h < heap_size; h++) {
    uint32_t cid = heap[h].cell_id;
    uint64_t off_lo = idx->cell_play_off[cid];
    uint64_t off_hi = idx->cell_play_off[cid + 1];
    uint64_t count = off_hi - off_lo;
    uint64_t scan = count;
    uint64_t start = 0;
    if (scan > MAX_PLAYS_PER_CELL_SCAN) {
      s = s * 0x9e3779b97f4a7c15ULL + 0x123456789abcdef0ULL;
      start = s % (count - MAX_PLAYS_PER_CELL_SCAN);
      scan = MAX_PLAYS_PER_CELL_SCAN;
    }
    for (uint64_t i = 0; i < scan; i++) {
      uint32_t pid = idx->plays_by_cell[off_lo + start + i];
      if (VISIT_MARK(pid)) continue;
      double sc = score_play(idx, pid);
      if (sc > best_score ||
          (sc == best_score && have_best && (s ^ pid) & 1)) {
        best_score = sc;
        best_play = pid;
        have_best = true;
      }
    }
  }

  if (!have_best || best_score <= 0.0) return NULL;
  PlayRecord pr;
  if (!play_index_get_play(idx, best_play, &pr)) return NULL;
  if (out_rack_id) *out_rack_id = pr.rack_id;
  return idx->racks + (size_t)pr.rack_id * RACK_BYTES;
}

// Pick top-N play_ids for the given rack_id by rarity-weighted score
// >= threshold. Insertion-sort into a fixed-size descending list.
int play_index_pick_targeted_plays(const PlayIndex *idx,
                                   uint32_t rack_id, double threshold,
                                   int max_n, uint32_t *out_play_ids) {
  if (rack_id >= idx->num_racks || max_n <= 0) return 0;
  uint64_t lo = idx->rack_first_play[rack_id];
  uint64_t hi = idx->rack_first_play[rack_id + 1];

  double best_scores[max_n];
  uint32_t best_pids[max_n];
  int n = 0;
  for (uint64_t pid = lo; pid < hi; pid++) {
    double sc = score_play(idx, (uint32_t)pid);
    if (sc < threshold) continue;
    int pos = n;
    while (pos > 0 && best_scores[pos - 1] < sc) pos--;
    if (pos < max_n) {
      int last = (n < max_n) ? n : max_n - 1;
      for (int j = last; j > pos; j--) {
        best_scores[j] = best_scores[j - 1];
        best_pids[j] = best_pids[j - 1];
      }
      best_scores[pos] = sc;
      best_pids[pos] = (uint32_t)pid;
      if (n < max_n) n++;
    }
  }
  for (int i = 0; i < n; i++) out_play_ids[i] = best_pids[i];
  return n;
}

int play_index_lookup_rack_id(const PlayIndex *idx, const char *rack_str) {
  // Linear scan. Racks are not sorted in racks.bin (insertion order
  // from per-shard merge). Cold path; not perf-critical.
  size_t len = strlen(rack_str);
  if (len > RACK_BYTES) return -1;
  char buf[RACK_BYTES];
  memset(buf, 0, RACK_BYTES);
  memcpy(buf, rack_str, len);
  for (uint32_t i = 0; i < idx->num_racks; i++) {
    if (memcmp(idx->racks + (size_t)i * RACK_BYTES, buf, RACK_BYTES) == 0) {
      return (int)i;
    }
  }
  return -1;
}
