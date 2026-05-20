#include "static_leaves.h"
#include "vmodel.h"  // vmodel_tile_index_from_char

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../util/io_util.h"

// Static-leaves CSV loader.

static int compare_entries(const void *a, const void *b) {
    const StaticLeavesEntry *ea = a;
    const StaticLeavesEntry *eb = b;
    if (ea->key_len != eb->key_len) return ea->key_len - eb->key_len;
    return memcmp(ea->key, eb->key, ea->key_len);
}

void static_leaves_canonicalize(const uint8_t *indices, int len,
                                 uint8_t *out_buf) {
    // Blanks (index 0) first, then alphabetical (1..26).
    int wb = 0;
    int wa = 0;
    uint8_t alpha[STATIC_LEAVES_MAX_LEN];
    for (int i = 0; i < len; i++) {
        if (indices[i] == 0) {
            out_buf[wb++] = 0;
        } else {
            alpha[wa++] = indices[i];
        }
    }
    // Insertion sort alpha section.
    for (int i = 1; i < wa; i++) {
        uint8_t v = alpha[i];
        int j = i - 1;
        while (j >= 0 && alpha[j] > v) {
            alpha[j + 1] = alpha[j];
            j--;
        }
        alpha[j + 1] = v;
    }
    memcpy(out_buf + wb, alpha, wa);
    // Zero-pad the rest.
    for (int i = wb + wa; i < STATIC_LEAVES_MAX_LEN; i++) {
        out_buf[i] = 0;
    }
}

// Convert a leave CSV string ("AEIOR" or "??AE") to indices. Returns
// length on success, -1 on error. Caller should still canonicalize.
static int parse_leave_string(const char *s, uint8_t *out) {
    int n = 0;
    for (const char *p = s; *p && *p != ',' && *p != '\n' && *p != '\r'; p++) {
        if (n >= STATIC_LEAVES_MAX_LEN) return -1;
        int idx = vmodel_tile_index_from_char(*p);
        if (idx < 0) return -1;
        out[n++] = (uint8_t)idx;
    }
    return n;
}

// Append an entry to a growing dynamic array.
static bool append_entry(StaticLeavesEntry **arr, int *n, int *cap,
                          const uint8_t *key, int key_len, float value) {
    if (*n == *cap) {
        int new_cap = (*cap == 0) ? 1024 : (*cap) * 2;
        StaticLeavesEntry *new_arr =
            realloc(*arr, new_cap * sizeof(StaticLeavesEntry));
        if (!new_arr) return false;
        *arr = new_arr;
        *cap = new_cap;
    }
    StaticLeavesEntry *e = &(*arr)[*n];
    memset(e->key, 0, sizeof(e->key));
    memcpy(e->key, key, key_len);
    e->key_len = (uint8_t)key_len;
    e->value = value;
    (*n)++;
    return true;
}

// Parse a CSV line. Format depends on `is_full_format`:
//   short: "<leave>,<value>"
//   full:  "<leave>,<count>,<value>[,<extra>]"
// Stores leave indices + value via append_entry. Returns 0 on parse
// error / skip, 1 on success.
static int parse_csv_line(char *line, bool is_full_format,
                           StaticLeavesEntry **arr, int *n, int *cap) {
    // Strip comments and whitespace.
    char *hash = strchr(line, '#');
    if (hash) *hash = '\0';
    while (*line && isspace((unsigned char)*line)) line++;
    if (*line == '\0') return 0;

    uint8_t key[STATIC_LEAVES_MAX_LEN];
    int key_len = parse_leave_string(line, key);
    if (key_len < 0) return 0;

    // Advance to the value column.
    char *p = line;
    while (*p && *p != ',') p++;
    if (*p != ',') return 0;
    p++;
    if (is_full_format) {
        // Skip count column.
        while (*p && *p != ',') p++;
        if (*p != ',') return 0;
        p++;
    }
    char *end = NULL;
    double v = strtod(p, &end);
    if (end == p) return 0;

    // Canonicalize before storing so binary search uses canonical keys.
    uint8_t canon[STATIC_LEAVES_MAX_LEN];
    static_leaves_canonicalize(key, key_len, canon);
    if (!append_entry(arr, n, cap, canon, key_len, (float)v)) return 0;
    return 1;
}

static bool load_csv(const char *path, bool is_full_format,
                      StaticLeavesEntry **arr, int *n, int *cap) {
    FILE *f = fopen(path, "r");
    if (!f) {
        log_warn("static_leaves: cannot open %s", path);
        return false;
    }
    char buf[1024];
    while (fgets(buf, sizeof(buf), f)) {
        parse_csv_line(buf, is_full_format, arr, n, cap);
    }
    fclose(f);
    return true;
}

StaticLeaves *static_leaves_create(const char *short_csv_path,
                                    const char *full_csv_path) {
    StaticLeaves *sl = calloc(1, sizeof(StaticLeaves));
    if (!sl) return NULL;
    int cap = 0;
    int n = 0;
    StaticLeavesEntry *arr = NULL;

    // Always insert empty leave with value 0 (matches Python: leaves[""] = 0.0).
    append_entry(&arr, &n, &cap, NULL, 0, 0.0f);

    if (short_csv_path && !load_csv(short_csv_path, false, &arr, &n, &cap)) {
        free(arr); free(sl); return NULL;
    }
    if (full_csv_path && !load_csv(full_csv_path, true, &arr, &n, &cap)) {
        free(arr); free(sl); return NULL;
    }
    qsort(arr, n, sizeof(StaticLeavesEntry), compare_entries);
    // De-dup: later entry wins (matches Python dict assignment).
    int j = 0;
    for (int i = 0; i < n; i++) {
        if (j > 0 && arr[j-1].key_len == arr[i].key_len &&
            memcmp(arr[j-1].key, arr[i].key, arr[i].key_len) == 0) {
            arr[j-1].value = arr[i].value;
        } else {
            if (j != i) arr[j] = arr[i];
            j++;
        }
    }
    sl->entries = arr;
    sl->n_entries = j;
    log_info("static_leaves: loaded %d entries", sl->n_entries);
    return sl;
}

void static_leaves_destroy(StaticLeaves *sl) {
    if (!sl) return;
    free(sl->entries);
    free(sl);
}

float static_leaves_lookup(const StaticLeaves *sl,
                            const uint8_t *canonical_indices, int len) {
    if (!sl || sl->n_entries == 0) return 0.0f;
    uint8_t key[STATIC_LEAVES_MAX_LEN] = {0};
    memcpy(key, canonical_indices, len);
    // Binary search by (len, key bytes).
    int lo = 0;
    int hi = sl->n_entries;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        const StaticLeavesEntry *e = &sl->entries[mid];
        int cmp;
        if (e->key_len != len) {
            cmp = e->key_len - len;
        } else {
            cmp = memcmp(e->key, key, len);
        }
        if (cmp == 0) return e->value;
        if (cmp < 0) lo = mid + 1;
        else hi = mid;
    }
    return 0.0f;
}
