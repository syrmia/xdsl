// RUN: xdsl-opt --split-input-file -p canonicalize %s | filecheck %s

// identity sublane_offsets [0,1,2,3] -> convert to simple tpu.load
%m = "test.op"() : () -> memref<4x128xf32>
%i0 = "test.op"() : () -> index
%i1 = "test.op"() : () -> index
%r = tpu.shuffled_load %m[%i0, %i1] {sublane_mask = array<i1: true, true, true, true>, sublane_offsets = array<i32: 0, 1, 2, 3>} : memref<4x128xf32>, vector<4x128xf32>
"test.op"(%r) : (vector<4x128xf32>) -> ()
// CHECK:      %{{.*}} = tpu.load %m[%i0, %i1] sublanes [-1, -1, -1, -1] sublane_stride 1
// CHECK-NEXT: "test.op"

// -----

// shuffled offsets [3,2,1,0] -> unchanged
%m = "test.op"() : () -> memref<4x128xf32>
%i0 = "test.op"() : () -> index
%i1 = "test.op"() : () -> index
%r = tpu.shuffled_load %m[%i0, %i1] {sublane_mask = array<i1: true, true, true, true>, sublane_offsets = array<i32: 3, 2, 1, 0>} : memref<4x128xf32>, vector<4x128xf32>
"test.op"(%r) : (vector<4x128xf32>) -> ()
// CHECK:      %{{.*}} = tpu.shuffled_load
// CHECK-SAME: sublane_offsets = array<i32: 3, 2, 1, 0>
