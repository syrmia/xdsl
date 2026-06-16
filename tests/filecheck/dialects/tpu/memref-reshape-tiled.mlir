// RUN: xdsl-opt --split-input-file %s | filecheck %s

%m = "test.op"() : () -> memref<8x128xf32>
%r = tpu.memref_reshape %m : memref<8x128xf32> -> memref<16x64xf32>
"test.op"(%r) : (memref<16x64xf32>) -> ()
// CHECK: tpu.memref_reshape

// -----

%m = "test.op"() : () -> memref<16x128xf32, #tpu.tiled<(8,128),[1, 1]>>
%r = tpu.memref_reshape %m : memref<16x128xf32, #tpu.tiled<(8,128),[1, 1]>> -> memref<16x128xf32, #tpu.tiled<(8,128),[1, 1]>>
"test.op"(%r) : (memref<16x128xf32, #tpu.tiled<(8,128),[1, 1]>>) -> ()
// CHECK: tpu.memref_reshape
// CHECK-SAME: #tpu.tiled<(8, 128),[1, 1]>
