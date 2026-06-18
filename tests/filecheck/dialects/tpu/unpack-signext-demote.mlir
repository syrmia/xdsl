// RUN: xdsl-opt --split-input-file -p canonicalize %s | filecheck %s

%s0 = "test.op"() : () -> vector<8xi16>
%s1 = "test.op"() : () -> vector<8xi16>
%packed = tpu.pack_subelements %s0, %s1 {positions = array<i32: 0, 1>, pack_format = #tpu<pack_format compressed>, unsigned_integers = false} : vector<8xi16>, vector<8xi16> -> vector<8xi8>
%unpacked = tpu.unpack_subelements %packed, 0 {pack_format = #tpu<pack_format compressed>, integer_extended = true, unsigned_integers = false} : vector<8xi8> -> vector<8xi16>
%pack_out = tpu.pack_subelements %unpacked, %unpacked {positions = array<i32: 0, 1>, pack_format = #tpu<pack_format compressed>, unsigned_integers = false} : vector<8xi16>, vector<8xi16> -> vector<8xi8>
"test.op"(%pack_out) : (vector<8xi8>) -> ()
// CHECK:      %s0 = "test.op"() : () -> vector<8xi16>
// CHECK:      %{{.*}} = tpu.pack_subelements %s0, %s0
// CHECK-NEXT: "test.op"(%{{.*}})

// -----

%s0 = "test.op"() : () -> vector<8xi16>
%s1 = "test.op"() : () -> vector<8xi16>
%packed = tpu.pack_subelements %s0, %s1 {positions = array<i32: 0, 1>, pack_format = #tpu<pack_format compressed>, unsigned_integers = false} : vector<8xi16>, vector<8xi16> -> vector<8xi8>
%unpacked = tpu.unpack_subelements %packed, 0 {pack_format = #tpu<pack_format compressed>, integer_extended = true, unsigned_integers = false} : vector<8xi8> -> vector<8xi16>
"test.op"(%unpacked) : (vector<8xi16>) -> ()
// CHECK:      %{{.*}} = tpu.unpack_subelements
// CHECK-SAME: integer_extended = true
