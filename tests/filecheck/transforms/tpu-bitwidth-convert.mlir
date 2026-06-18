// RUN: xdsl-opt --split-input-file -p tpu-bitwidth-convert %s | filecheck %s

%s = "test.op"() : () -> vector<128x128xbf16>
%a = "test.op"() : () -> vector<128xbf16>
%r = vector.multi_reduction <add>, %s, %a [1] : vector<128x128xbf16> to vector<128xbf16>
"test.op"(%r) : (vector<128xbf16>) -> ()
// CHECK:      %s = "test.op"() : () -> vector<128x128xbf16>
// CHECK-NEXT: %a = "test.op"() : () -> vector<128xbf16>
// CHECK-NEXT: %{{.*}} = arith.extf %s : vector<128x128xbf16> to vector<128x128xf32>
// CHECK-NEXT: %{{.*}} = arith.extf %a : vector<128xbf16> to vector<128xf32>
// CHECK-NEXT: %{{.*}} = vector.multi_reduction <add>, %{{.*}}, %{{.*}} [1] : vector<128x128xf32> to vector<128xf32>
// CHECK-NEXT: %{{.*}} = arith.truncf %{{.*}} : vector<128xf32> to vector<128xbf16>

// -----

%s = "test.op"() : () -> vector<128x128xf32>
%a = "test.op"() : () -> vector<128xf32>
%r = vector.multi_reduction <add>, %s, %a [1] : vector<128x128xf32> to vector<128xf32>
"test.op"(%r) : (vector<128xf32>) -> ()
// CHECK:      %s = "test.op"() : () -> vector<128x128xf32>
// CHECK-NEXT: %a = "test.op"() : () -> vector<128xf32>
// CHECK-NEXT: %{{.*}} = vector.multi_reduction <add>, %s, %a [1] : vector<128x128xf32> to vector<128xf32>

// -----
%v = "test.op"() : () -> vector<128xbf16>
%r = math.exp %v : vector<128xbf16>
"test.op"(%r) : (vector<128xbf16>) -> ()
// CHECK:      %{{.*}} = arith.extf %v : vector<128xbf16> to vector<128xf32>
// CHECK-NEXT: %{{.*}} = math.exp %{{.*}} : vector<128xf32>
// CHECK-NEXT: %{{.*}} = arith.truncf %{{.*}} : vector<128xf32> to vector<128xbf16>

// -----
%v = "test.op"() : () -> vector<128xbf16>
%r = math.sin %v : vector<128xbf16>
"test.op"(%r) : (vector<128xbf16>) -> ()
// CHECK:      %{{.*}} = arith.extf %v : vector<128xbf16> to vector<128xf32>
// CHECK-NEXT: %{{.*}} = math.sin %{{.*}} : vector<128xf32>
// CHECK-NEXT: %{{.*}} = arith.truncf %{{.*}} : vector<128xf32> to vector<128xbf16>

// -----
%v = "test.op"() : () -> vector<64xbf16>
%r = math.cos %v : vector<64xbf16>
"test.op"(%r) : (vector<64xbf16>) -> ()
// CHECK:      %{{.*}} = arith.extf %v : vector<64xbf16> to vector<64xf32>
// CHECK-NEXT: %{{.*}} = math.cos %{{.*}} : vector<64xf32>
// CHECK-NEXT: %{{.*}} = arith.truncf %{{.*}} : vector<64xf32> to vector<64xbf16>

// -----
%v = "test.op"() : () -> vector<128xbf16>
%r = math.sqrt %v : vector<128xbf16>
"test.op"(%r) : (vector<128xbf16>) -> ()
// CHECK:      %{{.*}} = arith.extf %v
// CHECK-NEXT: %{{.*}} = math.sqrt %{{.*}} : vector<128xf32>
// CHECK-NEXT: %{{.*}} = arith.truncf %{{.*}}

// -----
%v = "test.op"() : () -> vector<128xbf16>
%r = math.log %v : vector<128xbf16>
"test.op"(%r) : (vector<128xbf16>) -> ()
// CHECK:      %{{.*}} = arith.extf %v
// CHECK-NEXT: %{{.*}} = math.log %{{.*}} : vector<128xf32>
// CHECK-NEXT: %{{.*}} = arith.truncf %{{.*}}

// -----
%v = "test.op"() : () -> vector<128xbf16>
%r = math.tanh %v : vector<128xbf16>
"test.op"(%r) : (vector<128xbf16>) -> ()
// CHECK:      %{{.*}} = arith.extf %v
// CHECK-NEXT: %{{.*}} = math.tanh %{{.*}} : vector<128xf32>
// CHECK-NEXT: %{{.*}} = arith.truncf %{{.*}}

// -----
%v = "test.op"() : () -> vector<128xf32>
%r = math.exp %v : vector<128xf32>
"test.op"(%r) : (vector<128xf32>) -> ()
// CHECK:      %v = "test.op"() : () -> vector<128xf32>
// CHECK-NEXT: %{{.*}} = math.exp %v : vector<128xf32>

// -----
%v = "test.op"() : () -> vector<128xbf16>
%r = arith.negf %v : vector<128xbf16>
"test.op"(%r) : (vector<128xbf16>) -> ()
// CHECK:      %{{.*}} = arith.extf %v : vector<128xbf16> to vector<128xf32>
// CHECK-NEXT: %{{.*}} = arith.negf %{{.*}} : vector<128xf32>
// CHECK-NEXT: %{{.*}} = arith.truncf %{{.*}} : vector<128xf32> to vector<128xbf16>

// -----
%a = "test.op"() : () -> vector<128xbf16>
%b = "test.op"() : () -> vector<128xbf16>
%r = arith.addf %a, %b : vector<128xbf16>
"test.op"(%r) : (vector<128xbf16>) -> ()
// CHECK:      %{{.*}} = arith.extf %a : vector<128xbf16> to vector<128xf32>
// CHECK-NEXT: %{{.*}} = arith.extf %b : vector<128xbf16> to vector<128xf32>
// CHECK-NEXT: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<128xf32>
// CHECK-NEXT: %{{.*}} = arith.truncf %{{.*}} : vector<128xf32> to vector<128xbf16>

// -----
%a = "test.op"() : () -> vector<128x128xbf16>
%b = "test.op"() : () -> vector<128x128xbf16>
%r = arith.subf %a, %b : vector<128x128xbf16>
"test.op"(%r) : (vector<128x128xbf16>) -> ()
// CHECK:      %{{.*}} = arith.extf %a : vector<128x128xbf16> to vector<128x128xf32>
// CHECK-NEXT: %{{.*}} = arith.extf %b : vector<128x128xbf16> to vector<128x128xf32>
// CHECK-NEXT: %{{.*}} = arith.subf %{{.*}}, %{{.*}} : vector<128x128xf32>
// CHECK-NEXT: %{{.*}} = arith.truncf %{{.*}} : vector<128x128xf32> to vector<128x128xbf16>

// -----
%a = "test.op"() : () -> vector<128xbf16>
%b = "test.op"() : () -> vector<128xbf16>
%r = arith.mulf %a, %b : vector<128xbf16>
"test.op"(%r) : (vector<128xbf16>) -> ()
// CHECK:      %{{.*}} = arith.extf %a
// CHECK-NEXT: %{{.*}} = arith.extf %b
// CHECK-NEXT: %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : vector<128xf32>
// CHECK-NEXT: %{{.*}} = arith.truncf %{{.*}}

// -----
%a = "test.op"() : () -> vector<128xbf16>
%b = "test.op"() : () -> vector<128xbf16>
%r = arith.divf %a, %b : vector<128xbf16>
"test.op"(%r) : (vector<128xbf16>) -> ()
// CHECK:      %{{.*}} = arith.extf %a
// CHECK-NEXT: %{{.*}} = arith.extf %b
// CHECK-NEXT: %{{.*}} = arith.divf %{{.*}}, %{{.*}} : vector<128xf32>
// CHECK-NEXT: %{{.*}} = arith.truncf %{{.*}}

// -----
%a = "test.op"() : () -> vector<128xbf16>
%b = "test.op"() : () -> vector<128xbf16>
%r = arith.maximumf %a, %b : vector<128xbf16>
"test.op"(%r) : (vector<128xbf16>) -> ()
// CHECK:      %{{.*}} = arith.extf %a
// CHECK-NEXT: %{{.*}} = arith.extf %b
// CHECK-NEXT: %{{.*}} = arith.maximumf %{{.*}}, %{{.*}} : vector<128xf32>
// CHECK-NEXT: %{{.*}} = arith.truncf %{{.*}}

// -----
%a = "test.op"() : () -> vector<128xbf16>
%b = "test.op"() : () -> vector<128xbf16>
%r = arith.minimumf %a, %b : vector<128xbf16>
"test.op"(%r) : (vector<128xbf16>) -> ()
// CHECK:      %{{.*}} = arith.extf %a
// CHECK-NEXT: %{{.*}} = arith.extf %b
// CHECK-NEXT: %{{.*}} = arith.minimumf %{{.*}}, %{{.*}} : vector<128xf32>
// CHECK-NEXT: %{{.*}} = arith.truncf %{{.*}}

// -----
%a = "test.op"() : () -> vector<128xf32>
%b = "test.op"() : () -> vector<128xf32>
%r = arith.subf %a, %b : vector<128xf32>
"test.op"(%r) : (vector<128xf32>) -> ()
// CHECK:      %a = "test.op"() : () -> vector<128xf32>
// CHECK-NEXT: %b = "test.op"() : () -> vector<128xf32>
// CHECK-NEXT: %{{.*}} = arith.subf %a, %b : vector<128xf32>

// -----
%a = "test.op"() : () -> vector<128xbf16>
%b = "test.op"() : () -> vector<128xbf16>
%r = math.powf %a, %b : vector<128xbf16>
"test.op"(%r) : (vector<128xbf16>) -> ()
// CHECK:      %{{.*}} = arith.extf %a
// CHECK-NEXT: %{{.*}} = arith.extf %b
// CHECK-NEXT: %{{.*}} = math.powf %{{.*}}, %{{.*}} : vector<128xf32>
// CHECK-NEXT: %{{.*}} = arith.truncf %{{.*}}

// -----
%c = "test.op"() : () -> i1
%a = "test.op"() : () -> vector<128xbf16>
%b = "test.op"() : () -> vector<128xbf16>
%r = arith.select %c, %a, %b : vector<128xbf16>
"test.op"(%r) : (vector<128xbf16>) -> ()
// CHECK:      %c = "test.op"() : () -> i1
// CHECK-NEXT: %a = "test.op"() : () -> vector<128xbf16>
// CHECK-NEXT: %b = "test.op"() : () -> vector<128xbf16>
// CHECK-NEXT: %{{.*}} = arith.extf %a : vector<128xbf16> to vector<128xf32>
// CHECK-NEXT: %{{.*}} = arith.extf %b : vector<128xbf16> to vector<128xf32>
// CHECK-NEXT: %{{.*}} = arith.select %c, %{{.*}}, %{{.*}} : vector<128xf32>
// CHECK-NEXT: %{{.*}} = arith.truncf %{{.*}} : vector<128xf32> to vector<128xbf16>
