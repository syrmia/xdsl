// RUN: xdsl-opt --split-input-file -p tpu-canonicalize %s | filecheck %s

// addf(matmul(L, R, 0), x) -> matmul(L, R, x)
%L = "test.op"() : () -> vector<8x16xf32>
%R = "test.op"() : () -> vector<16x8xf32>
%x = "test.op"() : () -> vector<8x8xf32>
%zero = arith.constant dense<0.000000e+00> : vector<8x8xf32>
%mm = tpu.matmul %L, %R, %zero : vector<8x16xf32>, vector<16x8xf32>, vector<8x8xf32> -> vector<8x8xf32>
%r = arith.addf %mm, %x : vector<8x8xf32>
"test.op"(%r) : (vector<8x8xf32>) -> ()
// CHECK:      %L = "test.op"() : () -> vector<8x16xf32>
// CHECK-NEXT: %R = "test.op"() : () -> vector<16x8xf32>
// CHECK-NEXT: %x = "test.op"() : () -> vector<8x8xf32>
// CHECK-NEXT: %{{.*}} = tpu.matmul %L, %R, %x
// CHECK-NEXT: "test.op"(%{{.*}}) : (vector<8x8xf32>) -> ()

// -----

// addi(matmul(L, R, 0), x) -> matmul(L, R, x) # integer variant
%L = "test.op"() : () -> vector<8x16xi32>
%R = "test.op"() : () -> vector<16x8xi32>
%x = "test.op"() : () -> vector<8x8xi32>
%zero = arith.constant dense<0> : vector<8x8xi32>
%mm = tpu.matmul %L, %R, %zero : vector<8x16xi32>, vector<16x8xi32>, vector<8x8xi32> -> vector<8x8xi32>
%r = arith.addi %mm, %x : vector<8x8xi32>
"test.op"(%r) : (vector<8x8xi32>) -> ()
// CHECK:      %L = "test.op"() : () -> vector<8x16xi32>
// CHECK-NEXT: %R = "test.op"() : () -> vector<16x8xi32>
// CHECK-NEXT: %x = "test.op"() : () -> vector<8x8xi32>
// CHECK-NEXT: %{{.*}} = tpu.matmul %L, %R, %x
// CHECK-NEXT: "test.op"(%{{.*}}) : (vector<8x8xi32>) -> ()

// -----

%L = "test.op"() : () -> vector<8x16xf32>
%R = "test.op"() : () -> vector<16x8xf32>
%x = "test.op"() : () -> vector<8x8xf32>
%nonzero = arith.constant dense<1.000000e+00> : vector<8x8xf32>
%mm = tpu.matmul %L, %R, %nonzero : vector<8x16xf32>, vector<16x8xf32>, vector<8x8xf32> -> vector<8x8xf32>
%r = arith.addf %mm, %x : vector<8x8xf32>
"test.op"(%r) : (vector<8x8xf32>) -> ()
// CHECK:      %nonzero = arith.constant dense<1.000000e+00>
// CHECK-NEXT: %{{.*}} = tpu.matmul %L, %R, %nonzero
// CHECK-NEXT: %{{.*}} = arith.addf %{{.*}}, %x

// -----

%L = "test.op"() : () -> vector<8x16xf32>
%R = "test.op"() : () -> vector<16x8xf32>
%x = "test.op"() : () -> vector<8x8xf32>
%zero = arith.constant dense<0.000000e+00> : vector<8x8xf32>
%mm = tpu.matmul %L, %R, %zero : vector<8x16xf32>, vector<16x8xf32>, vector<8x8xf32> -> vector<8x8xf32>
%r = arith.addf %mm, %x : vector<8x8xf32>
"test.op"(%mm) : (vector<8x8xf32>) -> ()
"test.op"(%r) : (vector<8x8xf32>) -> ()
// CHECK:      %mm = tpu.matmul %L, %R, %zero
// CHECK:      %r = arith.addf %mm, %x
