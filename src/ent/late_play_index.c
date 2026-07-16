#include "late_play_index.h"

#include <fcntl.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// Read-only whole-file mmap (play_index.c's mmap_file is static/unexported, so
// this module carries its own). Returns the base + sets *out_len; NULL on fail.
static void *mmap_file(const char *path, size_t *out_len) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) return NULL;
  struct stat st;
  if (fstat(fd, &st) != 0 || st.st_size <= 0) {
    close(fd);
    return NULL;
  }
  void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (p == MAP_FAILED) return NULL;
  *out_len = (size_t)st.st_size;
  return p;
}

// plays.bin per-play header is 18 bytes (build_late_play_index.py PLAY_HDR_FMT
// "<I I B h B H B h B"): rack_id:u32 position_id:u32 action_kind:u8 score:i16
// ar_len:u8 n_cells:u16 leave_len:u8 eff_diff:i16 _pad:u8. The sampler only
// needs the first two u32s (rack_id, position_id).

struct LatePlayIndex {
  uint32_t num_racks;
  uint32_t num_positions;
  uint32_t num_plays;
  uint32_t num_cells;
  ForceTable *ft;

  const char *racks;             size_t racks_len;      // [num_racks * 8] ASCII
  const uint32_t *positions;     size_t positions_len;  // [num_positions] pool_idx
  const uint8_t *plays;          size_t plays_len;      // variable records
  const uint64_t *play_off;      size_t rack_idx_len;   // play byte offsets
  const uint32_t *plays_by_cell; size_t pbc_len;        // [total_refs] play_id
  const uint64_t *cell_off;      size_t cell_idx_len;   // [num_cells+1] CSR
};

static int parse_meta(const char *path, uint32_t *nr, uint32_t *np,
                      uint32_t *npl, uint32_t *nc) {
  FILE *f = fopen(path, "r");
  if (!f) return -1;
  char buf[4096];
  size_t n = fread(buf, 1, sizeof(buf) - 1, f);
  fclose(f);
  buf[n] = '\0';
  const char *k;
  int got = 0;
  if ((k = strstr(buf, "\"num_racks\""))) {
    *nr = (uint32_t)strtoul(strchr(k, ':') + 1, NULL, 10); got++;
  }
  if ((k = strstr(buf, "\"num_positions\""))) {
    *np = (uint32_t)strtoul(strchr(k, ':') + 1, NULL, 10); got++;
  }
  if ((k = strstr(buf, "\"num_plays\""))) {
    *npl = (uint32_t)strtoul(strchr(k, ':') + 1, NULL, 10); got++;
  }
  if ((k = strstr(buf, "\"num_cells\""))) {
    *nc = (uint32_t)strtoul(strchr(k, ':') + 1, NULL, 10); got++;
  }
  return got == 4 ? 0 : -1;
}

LatePlayIndex *late_play_index_create(const char *dir_path, ForceTable *ft) {
  LatePlayIndex *idx = (LatePlayIndex *)calloc(1, sizeof(LatePlayIndex));
  if (!idx) return NULL;
  idx->ft = ft;
  char path[1024];

  snprintf(path, sizeof(path), "%s/meta.json", dir_path);
  if (parse_meta(path, &idx->num_racks, &idx->num_positions, &idx->num_plays,
                 &idx->num_cells) != 0) {
    fprintf(stderr, "late_play_index: cannot parse %s\n", path);
    free(idx);
    return NULL;
  }
  const int nt = force_table_num_targets(ft);
  if ((int)idx->num_cells != nt) {
    fprintf(stderr,
            "late_play_index: num_cells %u != force_table_num_targets %d "
            "(index built against a different force table — rebuild)\n",
            idx->num_cells, nt);
    free(idx);
    return NULL;
  }

  snprintf(path, sizeof(path), "%s/racks.bin", dir_path);
  idx->racks = (const char *)mmap_file(path, &idx->racks_len);
  snprintf(path, sizeof(path), "%s/positions.bin", dir_path);
  idx->positions = (const uint32_t *)mmap_file(path, &idx->positions_len);
  snprintf(path, sizeof(path), "%s/plays.bin", dir_path);
  idx->plays = (const uint8_t *)mmap_file(path, &idx->plays_len);
  snprintf(path, sizeof(path), "%s/plays_by_rack.idx", dir_path);
  const uint64_t *rk = (const uint64_t *)mmap_file(path, &idx->rack_idx_len);
  if (rk) idx->play_off = rk + (idx->num_racks + 1);  // skip rack_first_play
  snprintf(path, sizeof(path), "%s/plays_by_cell.bin", dir_path);
  idx->plays_by_cell = (const uint32_t *)mmap_file(path, &idx->pbc_len);
  snprintf(path, sizeof(path), "%s/plays_by_cell.idx", dir_path);
  idx->cell_off = (const uint64_t *)mmap_file(path, &idx->cell_idx_len);

  if (!idx->racks || !idx->positions || !idx->plays || !idx->play_off ||
      !idx->plays_by_cell || !idx->cell_off) {
    fprintf(stderr, "late_play_index: mmap failed under %s\n", dir_path);
    late_play_index_destroy(idx);
    return NULL;
  }
  fprintf(stderr,
          "late_play_index: %u cells, %u positions, %u racks, %u plays\n",
          idx->num_cells, idx->num_positions, idx->num_racks, idx->num_plays);
  return idx;
}

void late_play_index_destroy(LatePlayIndex *idx) {
  if (!idx) return;
  if (idx->racks) munmap((void *)idx->racks, idx->racks_len);
  if (idx->positions) munmap((void *)idx->positions, idx->positions_len);
  if (idx->plays) munmap((void *)idx->plays, idx->plays_len);
  if (idx->play_off)  // points past rack_first_play; unmap from the base
    munmap((void *)(idx->play_off - (idx->num_racks + 1)), idx->rack_idx_len);
  if (idx->plays_by_cell) munmap((void *)idx->plays_by_cell, idx->pbc_len);
  if (idx->cell_off) munmap((void *)idx->cell_off, idx->cell_idx_len);
  free(idx);
}

// splitmix64 — seed-derived randomness, no shared RNG state.
static inline uint64_t sm64(uint64_t *s) {
  uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

bool late_play_index_sample(const LatePlayIndex *idx, uint64_t seed,
                            uint32_t *out_pool_idx, const char **out_rack) {
  uint64_t s = seed ? seed : 0x1234567ULL;
  ForceTable *ft = idx->ft;
  const uint32_t nc = idx->num_cells;
  // Pass 1: total deficit weight over cells that still have indexed plays. The
  // O(num_cells) scan is negligible vs the ~140 ms endgame solve it feeds.
  double total = 0.0;
  uint32_t last = UINT32_MAX;
  for (uint32_t c = 0; c < nc; c++) {
    if (idx->cell_off[c + 1] <= idx->cell_off[c]) continue;  // no plays indexed
    ForceTarget *t = force_table_target_by_index(ft, (int)c);
    if (!t) continue;
    int64_t d = atomic_load_explicit(&t->deficit, memory_order_relaxed);
    if (d > 0) {
      total += (double)d;
      last = c;
    }
  }
  if (total <= 0.0 || last == UINT32_MAX) return false;  // all drained
  // Pass 2: walk to the weighted-random cell (falls back to `last` if a
  // concurrent decrement shrank total below the draw).
  double r = ((double)(sm64(&s) >> 11) / (double)(1ULL << 53)) * total;
  uint32_t cell = last;
  for (uint32_t c = 0; c < nc; c++) {
    if (idx->cell_off[c + 1] <= idx->cell_off[c]) continue;
    ForceTarget *t = force_table_target_by_index(ft, (int)c);
    if (!t) continue;
    int64_t d = atomic_load_explicit(&t->deficit, memory_order_relaxed);
    if (d > 0) {
      r -= (double)d;
      if (r <= 0.0) {
        cell = c;
        break;
      }
    }
  }
  // Uniform-random play covering the cell.
  const uint64_t lo = idx->cell_off[cell], hi = idx->cell_off[cell + 1];
  if (hi <= lo) return false;
  const uint64_t pick = lo + (sm64(&s) % (hi - lo));
  const uint32_t play_id = idx->plays_by_cell[pick];
  if (play_id >= idx->num_plays) return false;
  const uint8_t *p = idx->plays + idx->play_off[play_id];
  uint32_t rack_id, position_id;
  memcpy(&rack_id, p, 4);
  memcpy(&position_id, p + 4, 4);
  if (rack_id >= idx->num_racks || position_id >= idx->num_positions) {
    return false;
  }
  *out_pool_idx = idx->positions[position_id];
  *out_rack = &idx->racks[(size_t)rack_id * 8];  // NUL-padded => NUL-terminated
  return true;
}
