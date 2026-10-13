; The new capacity of a growing std::vector goes through __mpr_capacity (mpr-resize-growth).
; RUN: opt -load-pass-plugin=%plugin -passes=mpr-resize-growth -S %s | FileCheck %s

declare i64 @_ZNKSt6vectorIlSaIlEE12_M_check_lenEmPKc(ptr, i64, ptr)

; CHECK-LABEL: define i64 @grow(
; CHECK: [[LEN:%.*]] = call i64 @_ZNKSt6vectorIlSaIlEE12_M_check_lenEmPKc(
; CHECK-NEXT: [[CAP:%.*]] = call i64 @__mpr_capacity(i64 [[LEN]], i64 1, i64 1, i64 {{[0-9]+}})
; CHECK-NEXT: ret i64 [[CAP]]
define i64 @grow(ptr %v) {
  %len = call i64 @_ZNKSt6vectorIlSaIlEE12_M_check_lenEmPKc(ptr %v, i64 1, ptr null)
  ret i64 %len
}
