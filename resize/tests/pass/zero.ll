; Zeroing of at least 4 KB, or of a size not known at compile time, goes through __mpr_zero
; (memset with a zero value, bzero, llvm.memset); smaller and non-zero fills stay as they are.
; RUN: opt -load-pass-plugin=%plugin -passes=mpr-resize -S %s | FileCheck %s

declare ptr @memset(ptr, i32, i64)
declare void @bzero(ptr, i64)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)

; CHECK-LABEL: define ptr @z(
; CHECK: call void @__mpr_zero(ptr %p, i64 %n, i64 {{[0-9]+}})
; CHECK: call void @__mpr_zero(ptr %p, i64 8192, i64 {{[0-9]+}})
; CHECK: call void @__mpr_zero(ptr %q, i64 %n, i64 {{[0-9]+}})
; CHECK: call void @llvm.memset.p0.i64(ptr %q, i8 0, i64 64, i1 false)
; CHECK: call void @llvm.memset.p0.i64(ptr %q, i8 1, i64 %n, i1 false)
; CHECK: call ptr @memset(ptr %q, i32 7, i64 %n)
; CHECK: ret ptr %p
define ptr @z(ptr %p, ptr %q, i64 %n) {
  %r = call ptr @memset(ptr %p, i32 0, i64 %n)
  call void @llvm.memset.p0.i64(ptr %p, i8 0, i64 8192, i1 false)
  call void @bzero(ptr %q, i64 %n)
  call void @llvm.memset.p0.i64(ptr %q, i8 0, i64 64, i1 false)
  call void @llvm.memset.p0.i64(ptr %q, i8 1, i64 %n, i1 false)
  %s = call ptr @memset(ptr %q, i32 7, i64 %n)
  ret ptr %r
}
