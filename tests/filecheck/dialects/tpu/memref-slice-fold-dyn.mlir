// RUN: xdsl-opt --split-input-file -p canonicalize %s | filecheck %s

%m = "test.op"() : () -> memref<8x16xf32>
%i = arith.constant 0 : i32
%j = arith.constant 0 : i32
%sz = arith.constant 4 : i32
%sl = tpu.memref_slice %m[%i, %j] <%sz> : memref<8x16xf32> -> memref<?x16xf32>
"test.op"(%sl) : (memref<?x16xf32>) -> ()
// CHECK:      %{{.*}} = tpu.memref_slice %m[%i, %j] : memref<8x16xf32> -> memref<4x16xf32>
// CHECK-NEXT: %{{.*}} = "memref.cast"(%{{.*}}) : (memref<4x16xf32>) -> memref<?x16xf32>
// CHECK-NEXT: "test.op"(%{{.*}}) : (memref<?x16xf32>) -> ()

// -----

%m = "test.op"() : () -> memref<8x16xf32>
%i = arith.constant 0 : i32
%j = arith.constant 0 : i32
%sz = "test.op"() : () -> i32
%sl = tpu.memref_slice %m[%i, %j] <%sz> : memref<8x16xf32> -> memref<?x16xf32>
"test.op"(%sl) : (memref<?x16xf32>) -> ()
// CHECK:      %sz = "test.op"() : () -> i32
// CHECK-NEXT: %{{.*}} = tpu.memref_slice %m[%i, %j] <%sz>
// CHECK-NOT:  memref.cast

// -----

%m = "test.op"() : () -> memref<8x16xf32>
%i = arith.constant 0 : i32
%j = arith.constant 0 : i32
%sz = arith.constant -4 : i32
%sl = tpu.memref_slice %m[%i, %j] <%sz> : memref<8x16xf32> -> memref<?x16xf32>
"test.op"(%sl) : (memref<?x16xf32>) -> ()
// CHECK:      %sz = arith.constant -4 : i32
// CHECK-NEXT: %{{.*}} = tpu.memref_slice %m[%i, %j] <%sz>
// CHECK-NOT:  memref.cast
