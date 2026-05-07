#ifndef RARE_POOL_H
#define RARE_POOL_H

#include <stdbool.h>
#include <stdint.h>

#include "force_table.h"
#include "letter_distribution.h"

// Rare-rack injection pool with per-rack force-target mapping. Loaded
// from MAGPIE_EB_RARE_POOL_CELLS=path.csv (one row per (rack, cell) pair:
// rack,length,type,kind,subleave,diff). At sample time the deficit-aware
// sampler scans all racks, computes per-rack marginal score = number of
// covered cells with deficit > 0, and uniformly picks among the racks
// tied at the highest score. Ties resolved randomly (per-call seed).
//
// Falls back gracefully when force_table doesn't have a matching target
// for a (rack, cell) pair — just drops that pair from the rack's coverage
// list.

typedef struct RarePool RarePool;

RarePool *rare_pool_create(const char *csv_path, ForceTable *force_table,
                           const LetterDistribution *ld);

void rare_pool_destroy(RarePool *rp);

// Number of distinct racks in the pool.
int rare_pool_num_racks(const RarePool *rp);

// Sample a rack index using deficit-aware scoring with random tie-break.
// `seed` perturbs the per-call random tie-break — pass iter_count or a
// thread-local mix. Returns the rack index in [0, num_racks); use
// rare_pool_get_rack to fetch the canonical rack string. Returns -1 if
// no rack covers any deficient cell (i.e. all rare-pool targets satisfied).
int rare_pool_sample_deficit_aware(const RarePool *rp, uint64_t seed);

// Get the canonical rack string at the given index (do not free).
const char *rare_pool_get_rack(const RarePool *rp, int idx);

#endif
