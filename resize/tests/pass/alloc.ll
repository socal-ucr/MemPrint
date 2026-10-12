; Allocation calls become libmpr calls with a site id; a call under an invoke becomes a plain call.
; RUN: opt -load-pass-plugin=%plugin -passes=mpr-resize -S %s | FileCheck %s

declare ptr @malloc(i64)
declare ptr @calloc(i64, i64)
declare ptr @realloc(ptr, i64)
declare i32 @posix_memalign(ptr, i64, i64)
declare ptr @_Znwm(i64)
declare ptr @_ZnwmSt11align_val_t(i64, i64)
declare i32 @__gxx_personality_v0(...)

; CHECK-LABEL: define ptr @a(
; CHECK: call ptr @__mpr_alloc(i64 %n, i64 [[ID:[0-9]+]], i32 1, i64 0)
; CHECK-NOT: @malloc
define ptr @a(i64 %n) {
  %p = call ptr @malloc(i64 %n)
  ret ptr %p
}

; CHECK-LABEL: define ptr @b(
; CHECK: call ptr @__mpr_calloc(i64 %n, i64 8, i64 {{[0-9]+}})
; CHECK: call ptr @__mpr_realloc(ptr %p, i64 %m, i64 {{[0-9]+}})
; CHECK: call i32 @__mpr_posix_memalign(ptr %out, i64 64, i64 %m, i64 {{[0-9]+}})
define ptr @b(i64 %n, i64 %m, ptr %out) {
  %p = call ptr @calloc(i64 %n, i64 8)
  %q = call ptr @realloc(ptr %p, i64 %m)
  %r = call i32 @posix_memalign(ptr %out, i64 64, i64 %m)
  ret ptr %q
}

; CHECK-LABEL: define ptr @c(
; CHECK: call ptr @__mpr_alloc(i64 %n, i64 {{[0-9]+}}, i32 16, i64 0)
; CHECK: call ptr @__mpr_alloc(i64 %n, i64 {{[0-9]+}}, i32 16, i64 256)
; CHECK-NOT: invoke
define ptr @c(i64 %n) personality ptr @__gxx_personality_v0 {
entry:
  %p = call ptr @_Znwm(i64 %n)
  %q = invoke ptr @_ZnwmSt11align_val_t(i64 %n, i64 256) to label %ok unwind label %lp
ok:
  ret ptr %q
lp:
  %l = landingpad { ptr, i32 } cleanup
  ret ptr null
}
