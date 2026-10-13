/* What the instrumented code sends the tool: one record per warp and memory instruction. */
#pragma once
#include <stdint.h>

#define RW_READ 1
#define RW_WRITE 2

typedef struct {
    uint64_t event;        /* the launch's event number (set at launch, nvbit_set_at_launch) */
    uint64_t addrs[32];    /* per lane */
    uint32_t active_mask;  /* lanes that executed the instruction (its guard predicate true) */
    uint32_t flags;        /* RW_READ | RW_WRITE */
    uint32_t size;         /* bytes per lane */
    uint32_t pad;
} mem_access_t;
