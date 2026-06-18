// RUN: xdsl-opt --split-input-file %s | filecheck %s

%m = "test.op"() : () -> memref<8x128xf32>
%b = tpu.memref_bitcast %m : memref<8x128xf32> -> memref<8x128xi32>
"test.op"(%b) : (memref<8x128xi32>) -> ()
// CHECK: tpu.memref_bitcast %m : memref<8x128xf32> -> memref<8x128xi32>

// -----

%m = "test.op"() : () -> memref<8x128xbf16, #tpu.tiled<(8,128),[1, 1]>>
%b = tpu.memref_bitcast %m : memref<8x128xbf16, #tpu.tiled<(8,128),[1, 1]>> -> memref<4x128xf32, #tpu.tiled<(4,128),[1, 1]>>
"test.op"(%b) : (memref<4x128xf32, #tpu.tiled<(4,128),[1, 1]>>) -> ()
// CHECK: tpu.memref_bitcast
// CHECK-SAME: #tpu.tiled<(8, 128),[1, 1]>
// CHECK-SAME: #tpu.tiled<(4, 128),[1, 1]>
