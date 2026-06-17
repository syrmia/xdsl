// RUN: xdsl-opt -p canonicalize %s | filecheck %s

// ReshapeOp: reshape of a reshape collapses to a single reshape.
%v3 = "test.op"() : () -> vector<4x8xf32>
%rs_y = tpu.reshape %v3 : vector<4x8xf32> -> vector<8x4xf32>
%rs_z = tpu.reshape %rs_y : vector<8x4xf32> -> vector<2x16xf32>
"test.op"(%rs_z) : (vector<2x16xf32>) -> ()
// CHECK:      %v3 = "test.op"() : () -> vector<4x8xf32>
// CHECK-NEXT: %{{.*}} = tpu.reshape %v3 : vector<4x8xf32> -> vector<2x16xf32>
// CHECK-NEXT: "test.op"(%{{.*}}) : (vector<2x16xf32>) -> ()
