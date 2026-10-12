/* libmpr: allocations served according to MemPrint's per-site reservations.

   A reservation is MemPrint's estimate of how much of a site's requested block the program
   touches. It is not binding: a block always gets the full size the program asked for, and only
   the predicted part is committed. The rest is mapped MAP_NORESERVE, so the kernel backs it if the
   program touches it after all, without a fault the program could see.

   Built as a preload library (LD_PRELOAD=libmpr.so: sites are the callers' return addresses) or
   linked into a program built with the MemPrintResize pass (sites are the ids it passes). */
#ifndef MPR_H
#define MPR_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum mpr_kind { MPR_MALLOC = 1, MPR_CALLOC = 2, MPR_REALLOC = 4, MPR_MEMALIGN = 8, MPR_NEW = 16 };

/* Entry points the MemPrintResize pass calls in place of the allocator (site: its id for the call). */
void *__mpr_alloc(size_t size, uint64_t site, int kind, size_t align);
void *__mpr_realloc(void *ptr, size_t size, uint64_t site);
void *__mpr_calloc(size_t n, size_t size, uint64_t site);
int __mpr_posix_memalign(void **out, size_t align, size_t size, uint64_t site);
/* A program built with the pass in apply mode carries its reservations in this string (the
   MPR_RESERVATIONS file format); MPR_RESERVATIONS overrides it. */
extern const char __mpr_embedded_reservations[] __attribute__((weak));
/* A container's new capacity (len elements of elem bytes, at growth point `site`): at least `len`,
   MemPrint's predicted final size when the call site has one. */
size_t __mpr_capacity(size_t len, size_t size, size_t elem, uint64_t site);
/* The program's call site growing a container on this thread (0: none), set by the pass around calls
   to container members. */
void __mpr_growth(uint64_t site);
/* Zeroing of n bytes at d (memset(d, 0, n), bzero, llvm.memset with a zero value), routed here by the
   pass: pages of a runtime block that were never touched are left alone (they read as zero). */
void __mpr_zero(void *d, size_t n, uint64_t site);
/* Bytes committed for the live blocks the runtime serves, as predicted (a scheduler could read it). */
size_t __mpr_committed(void);

#ifdef __cplusplus
}
#endif
#endif
