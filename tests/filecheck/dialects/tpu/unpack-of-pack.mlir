// RUN: xdsl-opt --split-input-file -p canonicalize %s | filecheck %s

// unpack-of-pack at index 0 should collapse to the first source
%s0 = "test.op"() : () -> vector<8xi32>
%s1 = "test.op"() : () -> vector<8xi32>
%packed = tpu.pack_subelements %s0, %s1 {positions = array<i32: 0, 1>, pack_format = #tpu<pack_format compressed>, unsigned_integers = false} : vector<8xi32>, vector<8xi32> -> vector<8xi16>
%unpacked = tpu.unpack_subelements %packed, 0 {pack_format = #tpu<pack_format compressed>, integer_extended = false, unsigned_integers = false} : vector<8xi16> -> vector<8xi32>
"test.op"(%unpacked) : (vector<8xi32>) -> ()
// CHECK:      %s0 = "test.op"() : () -> vector<8xi32>
// CHECK:      "test.op"(%s0) : (vector<8xi32>) -> ()

// -----

// unpack-of-pack at index 1 should collapse to the second source
%s0 = "test.op"() : () -> vector<8xi32>
%s1 = "test.op"() : () -> vector<8xi32>
%packed = tpu.pack_subelements %s0, %s1 {positions = array<i32: 0, 1>, pack_format = #tpu<pack_format compressed>, unsigned_integers = false} : vector<8xi32>, vector<8xi32> -> vector<8xi16>
%unpacked = tpu.unpack_subelements %packed, 1 {pack_format = #tpu<pack_format compressed>, integer_extended = false, unsigned_integers = false} : vector<8xi16> -> vector<8xi32>
"test.op"(%unpacked) : (vector<8xi32>) -> ()
// CHECK:      %s1 = "test.op"() : () -> vector<8xi32>
// CHECK:      "test.op"(%s1) : (vector<8xi32>) -> ()

// -----

// sign-extended unpack should not be canceled
%s0 = "test.op"() : () -> vector<8xi32>
%s1 = "test.op"() : () -> vector<8xi32>
%packed = tpu.pack_subelements %s0, %s1 {positions = array<i32: 0, 1>, pack_format = #tpu<pack_format compressed>, unsigned_integers = false} : vector<8xi32>, vector<8xi32> -> vector<8xi16>
%unpacked = tpu.unpack_subelements %packed, 0 {pack_format = #tpu<pack_format compressed>, integer_extended = true, unsigned_integers = false} : vector<8xi16> -> vector<8xi32>
"test.op"(%unpacked) : (vector<8xi32>) -> ()
// CHECK:      %unpacked = tpu.unpack_subelements %packed, 0
// CHECK-SAME: integer_extended = true
