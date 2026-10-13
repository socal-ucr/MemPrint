; Growth points of other containers (mpr-resize-growth): std::string's _M_create (the capacity is
; passed by reference), a hash table's _M_rehash (its bucket count, 8-byte buckets) and libc++'s
; __recommend (the result).
; RUN: opt -load-pass-plugin=%plugin -passes=mpr-resize-growth -S %s | FileCheck %s

declare ptr @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm(ptr, ptr, i64)
declare void @_ZNSt10_HashtableIllSaIlENSt8__detail9_IdentityESt8equal_toIlESt4hashIlENS1_18_Mod_range_hashingENS1_20_Default_ranged_hashENS1_20_Prime_rehash_policyENS1_17_Hashtable_traitsILb0ELb1ELb1EEEE9_M_rehashEmRKm(ptr, i64, ptr)
declare i64 @_ZNKSt3__16vectorIlNS_9allocatorIlEEE11__recommendEm(ptr, i64)

; CHECK-LABEL: define ptr @s(
; CHECK: [[OLD:%.*]] = load i64, ptr %cap
; CHECK-NEXT: [[NEW:%.*]] = call i64 @__mpr_capacity(i64 [[OLD]], i64 2, i64 1, i64 {{[0-9]+}})
; CHECK-NEXT: store i64 [[NEW]], ptr %cap
; CHECK-NEXT: call ptr @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm(ptr %this, ptr %cap, i64 %old)
define ptr @s(ptr %this, ptr %cap, i64 %old) {
  %p = call ptr @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm(ptr %this, ptr %cap, i64 %old)
  ret ptr %p
}

; CHECK-LABEL: define void @h(
; CHECK: [[B:%.*]] = call i64 @__mpr_capacity(i64 %n, i64 3, i64 8, i64 {{[0-9]+}})
; CHECK-NEXT: call void @_ZNSt10_Hashtable{{.*}}9_M_rehashEmRKm(ptr %t, i64 [[B]], ptr %state)
define void @h(ptr %t, i64 %n, ptr %state) {
  call void @_ZNSt10_HashtableIllSaIlENSt8__detail9_IdentityESt8equal_toIlESt4hashIlENS1_18_Mod_range_hashingENS1_20_Default_ranged_hashENS1_20_Prime_rehash_policyENS1_17_Hashtable_traitsILb0ELb1ELb1EEEE9_M_rehashEmRKm(ptr %t, i64 %n, ptr %state)
  ret void
}

; CHECK-LABEL: define i64 @r(
; CHECK: [[L:%.*]] = call i64 @_ZNKSt3__16vectorIlNS_9allocatorIlEEE11__recommendEm(
; CHECK-NEXT: call i64 @__mpr_capacity(i64 [[L]], i64 1, i64 1, i64 {{[0-9]+}})
define i64 @r(ptr %v, i64 %n) {
  %l = call i64 @_ZNKSt3__16vectorIlNS_9allocatorIlEEE11__recommendEm(ptr %v, i64 %n)
  ret i64 %l
}
