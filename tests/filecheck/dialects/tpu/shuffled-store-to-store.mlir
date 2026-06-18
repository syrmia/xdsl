// RUN: xdsl-opt --split-input-file -p canonicalize %s | filecheck %s

// identity sublane_offsets [0,1,2,3] -> convert to simple tpu.store
%v = "test.op"() : () -> vector<4x128xf32>
%m = "test.op"() : () -> memref<4x128xf32>
%i0 = "test.op"() : () -> index
%i1 = "test.op"() : () -> index
tpu.shuffled_store %m[%i0, %i1], %v {sublane_mask = array<i1: true, true, true, true>, sublane_offsets = array<i32: 0, 1, 2, 3>} : memref<4x128xf32>, vector<4x128xf32>
// CHECK:      tpu.store %m[%i0, %i1], %v sublanes [-1, -1, -1, -1] sublane_stride 1
// CHECK-SAME: add = false

// -----

// shuffled offsets [3,2,1,0] -> unchanged
%v = "test.op"() : () -> vector<4x128xf32>
%m = "test.op"() : () -> memref<4x128xf32>
%i0 = "test.op"() : () -> index
%i1 = "test.op"() : () -> index
tpu.shuffled_store %m[%i0, %i1], %v {sublane_mask = array<i1: true, true, true, true>, sublane_offsets = array<i32: 3, 2, 1, 0>} : memref<4x128xf32>, vector<4x128xf32>
// CHECK:      tpu.shuffled_store
// CHECK-SAME: sublane_offsets = array<i32: 3, 2, 1, 0>
