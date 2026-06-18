// RUN: xdsl-opt -p canonicalize %s | filecheck %s

// BitcastVregOp: chain X -> Y -> Z collapses to a single X -> Z.
%v1 = "test.op"() : () -> vector<4x8xf32>
%bc_y = tpu.bitcast_vreg %v1 : vector<4x8xf32> -> vector<8x4xf32>
%bc_z = tpu.bitcast_vreg %bc_y : vector<8x4xf32> -> vector<2x16xf32>
"test.op"(%bc_z) : (vector<2x16xf32>) -> ()
// CHECK:      %v1 = "test.op"() : () -> vector<4x8xf32>
// CHECK-NEXT: %{{.*}} = tpu.bitcast_vreg %v1 : vector<4x8xf32> -> vector<2x16xf32>
// CHECK-NEXT: "test.op"(%{{.*}}) : (vector<2x16xf32>) -> ()
