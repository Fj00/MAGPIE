#ifndef LEAVE_DEFICIT_H
#define LEAVE_DEFICIT_H

#include <stdbool.h>

// Per-(bag, leave, diff-bin) sample-deficit table for the per-leave-enumeration
// track (L0-L4). ("leave" = the tiles kept after a play.) Thread-safe, lazily
// targeted: the first time a cell is seen it is initialized to `target`
// remaining slots; each take() decrements until 0. The position-pool fan-out
// calls take() before branching a move's playout and only branches when a slot
// is available — so common cells stop at target and POST effort flows to rare
// cells (fixing the natural leave skew).
//
// Leaves longer than max_leave_len are rejected (false) — those (L5/L6/L7)
// belong to the parameterized force-table track, not the per-leave table.
//
// Env (read by autoplay): MAGPIE_PP_LEAVE_TARGET (enables gating, target/cell),
// MAGPIE_PP_LEAVE_BINW (diff bin width, default 20), MAGPIE_PP_LEAVE_MAXLEN
// (default 4).

typedef struct LeaveDeficit LeaveDeficit;

LeaveDeficit *leave_deficit_create(int target, int diff_bin_width,
                                   int max_leave_len);
void leave_deficit_destroy(LeaveDeficit *ld);

// Atomically claim one sample slot for (bag, leave, diff). Returns true (and
// decrements) if a slot was available; false if the cell is full or the leave
// exceeds max_leave_len. A NULL table means gating is disabled -> always true.
bool leave_deficit_take(LeaveDeficit *ld, int bag, const char *leave, int diff);

#endif
