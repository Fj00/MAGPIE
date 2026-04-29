#include "pass_cycle.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../util/io_util.h"

#define PASS_CYCLE_MAX_RACK_LEN 16
#define PASS_CYCLE_TILE_TYPES 27  // A-Z then ?

// Standard Scrabble bag counts indexed A=0..Z=25, ?=26.
static const uint8_t INITIAL_BAG[PASS_CYCLE_TILE_TYPES] = {
    9, 2, 2, 4, 12, 2, 3, 2, 9, 1, 1, 4, 2, 6, 8, 2, 1, 6, 4, 6, 4, 2, 2, 1,
    2, 1, 2};

typedef struct {
  uint8_t counts[PASS_CYCLE_TILE_TYPES];
} RackTiles;

struct PassCycleTable {
  char **racks;
  RackTiles *tile_counts;  // parsed tile composition per rack
  double *cum_weights;     // cumulative normalized weights for binary search
  uint8_t *is_pass;        // 1 = pass-favorable, 0 = exchange-only
  int num_racks;

  FILE *out_file;
  pthread_mutex_t out_mutex;
};

static void parse_rack_tiles(const char *rack_str, RackTiles *out) {
  memset(out->counts, 0, sizeof(out->counts));
  for (const char *p = rack_str; *p; p++) {
    int idx = (*p == '?') ? 26 : (*p - 'A');
    if (idx >= 0 && idx < PASS_CYCLE_TILE_TYPES) {
      out->counts[idx]++;
    }
  }
}

static bool rack_drawable(const RackTiles *rack, const uint8_t bag[PASS_CYCLE_TILE_TYPES]) {
  for (int i = 0; i < PASS_CYCLE_TILE_TYPES; i++) {
    if (rack->counts[i] > bag[i]) return false;
  }
  return true;
}

static uint64_t mix64(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}

static double uniform01(uint64_t h) {
  return (double)(h >> 11) / (double)(1ULL << 53);
}

PassCycleTable *pass_cycle_table_create(const char *pool_path,
                                        const char *out_path) {
  FILE *f = fopen(pool_path, "r");
  if (!f) {
    fprintf(stderr, "pass_cycle: cannot open pool %s\n", pool_path);
    return NULL;
  }
  char line[256];
  // Skip header.
  if (!fgets(line, sizeof(line), f)) {
    fclose(f);
    return NULL;
  }
  int count = 0;
  while (fgets(line, sizeof(line), f)) {
    if (strlen(line) > 1) count++;
  }
  if (count == 0) {
    fclose(f);
    fprintf(stderr, "pass_cycle: no racks in %s\n", pool_path);
    return NULL;
  }
  rewind(f);
  fgets(line, sizeof(line), f);  // skip header again

  char **racks = malloc_or_die(sizeof(char *) * (size_t)count);
  RackTiles *tiles = malloc_or_die(sizeof(RackTiles) * (size_t)count);
  double *weights = malloc_or_die(sizeof(double) * (size_t)count);
  uint8_t *is_pass_arr = malloc_or_die(sizeof(uint8_t) * (size_t)count);
  int idx = 0;
  while (fgets(line, sizeof(line), f) && idx < count) {
    size_t len = strlen(line);
    while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
      len--;
    if (len == 0) continue;
    char rack_buf[PASS_CYCLE_MAX_RACK_LEN + 1] = {0};
    double w = 0.0;
    int isp = 0;
    if (sscanf(line, "%15[^,],%lf,%d", rack_buf, &w, &isp) < 2) continue;
    char *copy = malloc_or_die(strlen(rack_buf) + 1);
    strcpy(copy, rack_buf);
    racks[idx] = copy;
    parse_rack_tiles(rack_buf, &tiles[idx]);
    weights[idx] = w;
    is_pass_arr[idx] = (uint8_t)(isp ? 1 : 0);
    idx++;
  }
  fclose(f);
  count = idx;

  // Normalize to cumulative weights.
  double total = 0.0;
  for (int i = 0; i < count; i++) total += weights[i];
  double cum = 0.0;
  for (int i = 0; i < count; i++) {
    cum += weights[i] / total;
    weights[i] = cum;
  }
  weights[count - 1] = 1.0;

  FILE *out = fopen(out_path, "w");
  if (!out) {
    fprintf(stderr, "pass_cycle: cannot open output %s\n", out_path);
    for (int i = 0; i < count; i++) free(racks[i]);
    free(racks);
    free(tiles);
    free(weights);
    free(is_pass_arr);
    return NULL;
  }
  fprintf(out,
          "pair_id,p1_rack,p2_rack,p1_final_rack,p2_final_rack,"
          "branch,outcome,num_turns,history\n");
  fflush(out);

  int n_pass = 0;
  for (int i = 0; i < count; i++) n_pass += is_pass_arr[i];

  PassCycleTable *table = malloc_or_die(sizeof(PassCycleTable));
  table->racks = racks;
  table->tile_counts = tiles;
  table->cum_weights = weights;
  table->is_pass = is_pass_arr;
  table->num_racks = count;
  table->out_file = out;
  pthread_mutex_init(&table->out_mutex, NULL);

  fprintf(stderr,
          "pass_cycle: %d racks in combined pool (%d pass-favorable); "
          "output -> %s\n",
          count, n_pass, out_path);
  return table;
}

void pass_cycle_table_destroy(PassCycleTable *table) {
  if (!table) return;
  for (int i = 0; i < table->num_racks; i++) free(table->racks[i]);
  free(table->racks);
  free(table->tile_counts);
  free(table->cum_weights);
  free(table->is_pass);
  if (table->out_file) {
    fflush(table->out_file);
    fclose(table->out_file);
  }
  pthread_mutex_destroy(&table->out_mutex);
  free(table);
}

// Binary search into cumulative weight array.
static int sample_by_weight(const PassCycleTable *table, double u) {
  int lo = 0, hi = table->num_racks - 1;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (table->cum_weights[mid] < u) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

bool pass_cycle_sample_racks(const PassCycleTable *table, uint64_t pair_id,
                             const char **p1_rack_out,
                             const char **p2_rack_out,
                             int *p1_is_pass_out) {
  // P1: hash(pair_id) -> weighted sample from full pool.
  const uint64_t h1 = mix64(pair_id * 2ULL + 1ULL);
  const int p1_idx = sample_by_weight(table, uniform01(h1));
  *p1_rack_out = table->racks[p1_idx];
  if (p1_is_pass_out) *p1_is_pass_out = (int)table->is_pass[p1_idx];

  // Remaining bag after P1's draw.
  uint8_t bag[PASS_CYCLE_TILE_TYPES];
  memcpy(bag, INITIAL_BAG, sizeof(bag));
  const RackTiles *p1_tiles = &table->tile_counts[p1_idx];
  for (int i = 0; i < PASS_CYCLE_TILE_TYPES; i++) {
    bag[i] -= p1_tiles->counts[i];
  }

  // P2: hash(pair_id, attempt) -> weighted sample, retry if not drawable.
  // Expected retry rate is very low since the bag has 93 tiles remaining
  // and most 7-tile racks are easily drawable.
  for (int attempt = 0; attempt < 32; attempt++) {
    const uint64_t h2 = mix64(pair_id * 2ULL + (uint64_t)attempt * 7919ULL);
    const int p2_idx = sample_by_weight(table, uniform01(h2));
    if (rack_drawable(&table->tile_counts[p2_idx], bag)) {
      *p2_rack_out = table->racks[p2_idx];
      return true;
    }
  }
  // Fallback: linear scan from a random starting point for the first
  // drawable rack. Should almost never be reached.
  const uint64_t hf = mix64(pair_id ^ 0xdeadbeefULL);
  const int start = (int)(hf % (uint64_t)table->num_racks);
  for (int i = 0; i < table->num_racks; i++) {
    const int idx = (start + i) % table->num_racks;
    if (rack_drawable(&table->tile_counts[idx], bag)) {
      *p2_rack_out = table->racks[idx];
      return true;
    }
  }
  return false;
}

void pass_cycle_record(PassCycleTable *table, uint64_t pair_id,
                       const char *p1_rack, const char *p2_rack,
                       const char *p1_final_rack, const char *p2_final_rack,
                       int branch, int outcome, int num_turns,
                       const char *history) {
  pthread_mutex_lock(&table->out_mutex);
  fprintf(table->out_file, "%llu,%s,%s,%s,%s,%d,%d,%d,%s\n",
          (unsigned long long)pair_id, p1_rack, p2_rack,
          p1_final_rack ? p1_final_rack : "",
          p2_final_rack ? p2_final_rack : "",
          branch, outcome, num_turns, history ? history : "");
  pthread_mutex_unlock(&table->out_mutex);
}
