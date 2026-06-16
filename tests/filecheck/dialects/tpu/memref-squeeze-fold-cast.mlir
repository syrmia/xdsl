// RUN: xdsl-opt --split-input-file -p canonicalize %s | filecheck %s

%m = "test.op"() : () -> memref<4x1x8xf32>
%c = "memref.cast"(%m) : (memref<4x1x8xf32>) -> memref<?x1x8xf32>
%sq = tpu.memref_squeeze %c : memref<?x1x8xf32> -> memref<?x8xf32>
"test.op"(%sq) : (memref<?x8xf32>) -> ()
// CHECK:      %{{.*}} = tpu.memref_squeeze %m : memref<4x1x8xf32> -> memref<4x8xf32>
// CHECK-NEXT: %{{.*}} = "memref.cast"(%{{.*}}) : (memref<4x8xf32>) -> memref<?x8xf32>
// CHECK-NEXT: "test.op"(%{{.*}}) : (memref<?x8xf32>) -> ()

// -----

%m = "test.op"() : () -> memref<?x1x8xf32>
%sq = tpu.memref_squeeze %m : memref<?x1x8xf32> -> memref<?x8xf32>
"test.op"(%sq) : (memref<?x8xf32>) -> ()
// CHECK:      %m = "test.op"() : () -> memref<?x1x8xf32>
// CHECK-NEXT: %{{.*}} = tpu.memref_squeeze %m : memref<?x1x8xf32> -> memref<?x8xf32>
// CHECK-NOT:  memref.cast

// -----

%m = "test.op"() : () -> memref<?x1x8xf32>
%c = "memref.cast"(%m) : (memref<?x1x8xf32>) -> memref<4x1x8xf32>
%sq = tpu.memref_squeeze %c : memref<4x1x8xf32> -> memref<4x8xf32>
"test.op"(%sq) : (memref<4x8xf32>) -> ()
// CHECK:      %m = "test.op"() : () -> memref<?x1x8xf32>
// CHECK-NEXT: %{{.*}} = "memref.cast"(%m) : (memref<?x1x8xf32>) -> memref<4x1x8xf32>
// CHECK-NEXT: %{{.*}} = tpu.memref_squeeze %{{.*}}
