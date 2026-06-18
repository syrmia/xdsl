// RUN: xdsl-opt --split-input-file -p canonicalize %s | filecheck %s

// unroll(roll(v0, v1, v2)) -> direct use of v0, v1, v2
%v0 = "test.op"() : () -> vector<8xf32>
%v1 = "test.op"() : () -> vector<8xf32>
%v2 = "test.op"() : () -> vector<8xf32>
%rolled = tpu.roll_vectors %v0, %v1, %v2 : vector<8xf32>, vector<8xf32>, vector<8xf32> -> vector<3x8xf32>
%a, %b, %c = tpu.unroll_vectors %rolled : vector<3x8xf32> -> vector<8xf32>, vector<8xf32>, vector<8xf32>
"test.op"(%a, %b, %c) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> ()
// CHECK:      %v0 = "test.op"() : () -> vector<8xf32>
// CHECK-NEXT: %v1 = "test.op"() : () -> vector<8xf32>
// CHECK-NEXT: %v2 = "test.op"() : () -> vector<8xf32>
// CHECK-NEXT: "test.op"(%v0, %v1, %v2)

// -----

%v = "test.op"() : () -> vector<3x8xf32>
%a, %b, %c = tpu.unroll_vectors %v : vector<3x8xf32> -> vector<8xf32>, vector<8xf32>, vector<8xf32>
"test.op"(%a, %b, %c) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> ()
// CHECK:      %v = "test.op"() : () -> vector<3x8xf32>
// CHECK-NEXT: %{{.*}}, %{{.*}}, %{{.*}} = tpu.unroll_vectors %v
