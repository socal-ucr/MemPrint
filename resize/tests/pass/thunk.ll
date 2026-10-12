; An allocation function used as a value (stored, passed, or in a table of function pointers) is
; replaced by a thunk that calls the runtime with a site id of its own.
; RUN: opt -load-pass-plugin=%plugin -passes=mpr-resize -S %s | FileCheck %s

declare ptr @malloc(i64)
declare ptr @realloc(ptr, i64)

@table = global [2 x ptr] [ptr @malloc, ptr @realloc]
; CHECK: @table = global [2 x ptr] [ptr @__mpr_thunk_{{[0-9a-f]+}}, ptr @__mpr_thunk_{{[0-9a-f]+}}]

; CHECK-LABEL: define void @keep(
; CHECK: store ptr @__mpr_thunk_[[T:[0-9a-f]+]], ptr %slot
define void @keep(ptr %slot) {
  store ptr @malloc, ptr %slot
  ret void
}

; CHECK: define internal ptr @__mpr_thunk_
; CHECK: call ptr @__mpr_alloc(i64 %0, i64 {{[0-9]+}}, i32 1, i64 0)
; CHECK-NOT: call ptr @malloc
