// RUN: xdsl-opt --split-input-file -p tpu-canonicalize %s | filecheck %s

%c = arith.constant 1.5 : bf16
%e = tpu.extf %c : bf16 -> f32
"test.op"(%e) : (f32) -> ()
// CHECK:      %{{.*}} = arith.constant 1.500000e+00 : f32
// CHECK-NEXT: "test.op"(%{{.*}}) : (f32) -> ()
// CHECK-NOT:  tpu.extf

// -----

%c = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : vector<4xf16>
%e = tpu.extf %c : vector<4xf16> -> vector<4xf32>
"test.op"(%e) : (vector<4xf32>) -> ()
// CHECK:      %{{.*}} = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : vector<4xf32>
// CHECK-NEXT: "test.op"(%{{.*}}) : (vector<4xf32>) -> ()
// CHECK-NOT:  tpu.extf

// -----

%c = arith.constant 1.5 : f64
%t = tpu.truncf %c {rounding_mode = #tpu<rounding_mode to_nearest_even>} : f64 -> f32
"test.op"(%t) : (f32) -> ()
// CHECK:      %{{.*}} = arith.constant 1.500000e+00 : f32
// CHECK-NEXT: "test.op"(%{{.*}}) : (f32) -> ()
// CHECK-NOT:  tpu.truncf

// -----

%c = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : vector<4xf32>
%t = tpu.truncf %c {rounding_mode = #tpu<rounding_mode to_nearest_even>} : vector<4xf32> -> vector<4xf16>
"test.op"(%t) : (vector<4xf16>) -> ()
// CHECK:      %{{.*}} = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : vector<4xf16>
// CHECK-NEXT: "test.op"(%{{.*}}) : (vector<4xf16>) -> ()
// CHECK-NOT:  tpu.truncf

// -----

%v = "test.op"() : () -> vector<4xf32>
%t = tpu.truncf %v {rounding_mode = #tpu<rounding_mode to_nearest_even>} : vector<4xf32> -> vector<4xf16>
"test.op"(%t) : (vector<4xf16>) -> ()
// CHECK:      %v = "test.op"() : () -> vector<4xf32>
// CHECK-NEXT: %{{.*}} = tpu.truncf %v
