// RUN: xdsl-opt --split-input-file -p canonicalize %s | filecheck %s

%x = "test.op"() : () -> f32
%r = math.roundeven %x : f32
%i = tpu.fptosi %r {rounding_mode = #tpu<rounding_mode towards_zero>} : f32 -> i32
"test.op"(%i) : (i32) -> ()
// CHECK:      %x = "test.op"() : () -> f32
// CHECK-NEXT: %{{.*}} = tpu.fptosi %x
// CHECK-SAME: to_nearest_even

// -----

%x = "test.op"() : () -> f32
%i = tpu.fptosi %x {rounding_mode = #tpu<rounding_mode towards_zero>} : f32 -> i32
"test.op"(%i) : (i32) -> ()
// CHECK:      %x = "test.op"() : () -> f32
// CHECK-NEXT: %{{.*}} = tpu.fptosi %x
// CHECK-SAME: towards_zero
