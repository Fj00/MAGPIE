#include "leave_deficit.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../util/io_util.h"

#define LD_SHARDS 256

typedef struct LDEntry {
  char *key;
  int remaining;
  struct LDEntry *next;
} LDEntry;

typedef struct {
  pthread_mutex_t mu;
  LDEntry **buckets;
  size_t nbuckets;
  size_t count;
} LDShard;

struct LeaveDeficit {
  int target;
  int binw;
  int max_leave_len;
  LDShard shards[LD_SHARDS];
};

static uint64_t ld_hash(const char *s) {
  uint64_t h = 1469598103934665603ULL;  // FNV-1a
  for (; *s; s++) {
    h ^= (unsigned char)*s;
    h *= 1099511628211ULL;
  }
  return h;
}

LeaveDeficit *leave_deficit_create(int target, int diff_bin_width,
                                   int max_leave_len) {
  if (target <= 0) return NULL;
  LeaveDeficit *ld = malloc_or_die(sizeof(LeaveDeficit));
  ld->target = target;
  ld->binw = diff_bin_width > 0 ? diff_bin_width : 20;
  ld->max_leave_len = max_leave_len > 0 ? max_leave_len : 4;
  for (int i = 0; i < LD_SHARDS; i++) {
    pthread_mutex_init(&ld->shards[i].mu, NULL);
    ld->shards[i].nbuckets = 4096;
    ld->shards[i].count = 0;
    ld->shards[i].buckets = calloc(ld->shards[i].nbuckets, sizeof(LDEntry *));
    if (!ld->shards[i].buckets) {
      log_fatal("leave_deficit: calloc failed");
    }
  }
  fprintf(stderr,
          "leave_deficit: target=%d/cell, diff_bin_width=%d, max_leave_len=%d\n",
          ld->target, ld->binw, ld->max_leave_len);
  return ld;
}

void leave_deficit_destroy(LeaveDeficit *ld) {
  if (!ld) return;
  for (int i = 0; i < LD_SHARDS; i++) {
    for (size_t b = 0; b < ld->shards[i].nbuckets; b++) {
      LDEntry *e = ld->shards[i].buckets[b];
      while (e) {
        LDEntry *nx = e->next;
        free(e->key);
        free(e);
        e = nx;
      }
    }
    free(ld->shards[i].buckets);
    pthread_mutex_destroy(&ld->shards[i].mu);
  }
  free(ld);
}

static void ld_shard_grow(LDShard *sh) {
  size_t new_n = sh->nbuckets * 2;
  LDEntry **nb = calloc(new_n, sizeof(LDEntry *));
  if (!nb) return;  // grow is best-effort; keep old table on OOM
  for (size_t b = 0; b < sh->nbuckets; b++) {
    LDEntry *e = sh->buckets[b];
    while (e) {
      LDEntry *nx = e->next;
      size_t idx = ld_hash(e->key) % new_n;
      e->next = nb[idx];
      nb[idx] = e;
      e = nx;
    }
  }
  free(sh->buckets);
  sh->buckets = nb;
  sh->nbuckets = new_n;
}

bool leave_deficit_take(LeaveDeficit *ld, int bag, const char *leave,
                        int diff) {
  if (!ld) return true;  // gating disabled -> always allow
  const int llen = (int)strlen(leave);
  if (llen > ld->max_leave_len) return false;
  // floor-divide diff into a bin (correct for negatives)
  int bin = diff >= 0 ? (diff / ld->binw) * ld->binw
                      : -(((-diff) + ld->binw - 1) / ld->binw) * ld->binw;
  char key[40];
  snprintf(key, sizeof(key), "%d|%s|%d", bag, leave, bin);
  const uint64_t h = ld_hash(key);
  LDShard *sh = &ld->shards[h % LD_SHARDS];

  pthread_mutex_lock(&sh->mu);
  size_t idx = h % sh->nbuckets;
  for (LDEntry *e = sh->buckets[idx]; e; e = e->next) {
    if (strcmp(e->key, key) == 0) {
      bool ok = e->remaining > 0;
      if (ok) e->remaining--;
      pthread_mutex_unlock(&sh->mu);
      return ok;
    }
  }
  // not found: create at target, take one
  LDEntry *e = malloc_or_die(sizeof(LDEntry));
  e->key = malloc_or_die((size_t)strlen(key) + 1);
  strcpy(e->key, key);
  e->remaining = ld->target - 1;
  e->next = sh->buckets[idx];
  sh->buckets[idx] = e;
  sh->count++;
  if (sh->count > sh->nbuckets * 4) {
    ld_shard_grow(sh);
  }
  pthread_mutex_unlock(&sh->mu);
  return true;
}
