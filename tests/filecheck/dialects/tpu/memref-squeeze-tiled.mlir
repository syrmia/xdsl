// RUN: xdsl-opt --split-input-file %s | filecheck %s

%m = "test.op"() : () -> memref<1x8x128xf32, #tpu.tiled<(8,128),[1, 1, 1]>>
%s = tpu.memref_squeeze %m : memref<1x8x128xf32, #tpu.tiled<(8,128),[1, 1, 1]>> -> memref<8x128xf32, #tpu.tiled<(8,128),[1, 1]>>
"test.op"(%s) : (memref<8x128xf32, #tpu.tiled<(8,128),[1, 1]>>) -> ()
// CHECK: tpu.memref_squeeze
// CHECK-SAME: #tpu.tiled<(8, 128),[1, 1, 1]>
// CHECK-SAME: #tpu.tiled<(8, 128),[1, 1]>

// -----

%m = "test.op"() : () -> memref<4x1x128xf32, #tpu.tiled<(1,128),[1, 1, 1]>>
%s = tpu.memref_squeeze %m : memref<4x1x128xf32, #tpu.tiled<(1,128),[1, 1, 1]>> -> memref<4x128xf32, #tpu.tiled<(1,128),[1, 1]>>
"test.op"(%s) : (memref<4x128xf32, #tpu.tiled<(1,128),[1, 1]>>) -> ()
// CHECK: tpu.memref_squeeze
