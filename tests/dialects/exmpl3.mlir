module @_flash_attention_kernel {
  func.func @main(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: memref<1x1x128x64xbf16, #tpu.memory_space<vmem>>, %arg5: memref<1x1x128x64xbf16, #tpu.memory_space<vmem>>, %arg6: memref<1x1x128x64xbf16, #tpu.memory_space<vmem>>, %arg7: memref<1x1x128x64xbf16, #tpu.memory_space<vmem>>) attributes {dimension_semantics = [#tpu.dimension_semantics<parallel>, #tpu.dimension_semantics<parallel>, #tpu.dimension_semantics<parallel>, #tpu.dimension_semantics<arbitrary>], iteration_bounds = array<i64: 2, 4, 1, 1>, scalar_prefetch = 0 : i64, scratch_operands = 0 : i64, window_params = [{transform_indices = @transform_0, window_bounds = array<i64: 1, 1, 128, 64>}, {transform_indices = @transform_1, window_bounds = array<i64: 1, 1, 128, 64>}, {transform_indices = @transform_2, window_bounds = array<i64: 1, 1, 128, 64>}, {transform_indices = @transform_3, window_bounds = array<i64: 1, 1, 128, 64>}]} {
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %c0_1 = arith.constant 0 : index
    %c0_2 = arith.constant 0 : index
    %0 = vector.load %arg4[%c0, %c0_0, %c0_1, %c0_2] : memref<1x1x128x64xbf16, #tpu.memory_space<vmem>>, vector<1x1x128x64xbf16>
    %1 = vector.shape_cast %0 : vector<1x1x128x64xbf16> to vector<128x64xbf16>
    %c0_3 = arith.constant 0 : index
    %c0_4 = arith.constant 0 : index
    %c0_5 = arith.constant 0 : index
    %c0_6 = arith.constant 0 : index
    %2 = vector.load %arg5[%c0_3, %c0_4, %c0_5, %c0_6] : memref<1x1x128x64xbf16, #tpu.memory_space<vmem>>, vector<1x1x128x64xbf16>
    %3 = vector.shape_cast %2 : vector<1x1x128x64xbf16> to vector<128x64xbf16>
    %cst = arith.constant dense<0.000000e+00> : vector<128x128xf32>
    %4 = tpu.matmul %1, %3, %cst {dimension_numbers = #tpu.dot_dimension_numbers<[1], [1], [0], [0], [0, 0, 1, 0], [], []>} : vector<128x64xbf16>, vector<128x64xbf16>, vector<128x128xf32> -> vector<128x128xf32>
    %cst_7 = arith.constant dense<0xFF800000> : vector<128xf32>
    %5 = vector.multi_reduction <maximumf>, %4, %cst_7 [1] : vector<128x128xf32> to vector<128xf32>
    %6 = vector.shape_cast %5 : vector<128xf32> to vector<128x1xf32>
    %7 = vector.broadcast %6 : vector<128x1xf32> to vector<128x128xf32>
    %8 = arith.subf %4, %7 : vector<128x128xf32>
    %9 = math.exp %8 : vector<128x128xf32>
    %cst_8 = arith.constant dense<0.000000e+00> : vector<128xf32>
    %10 = vector.multi_reduction <add>, %9, %cst_8 [1] : vector<128x128xf32> to vector<128xf32>
    %11 = vector.shape_cast %10 : vector<128xf32> to vector<128x1xf32>
    %12 = vector.broadcast %11 : vector<128x1xf32> to vector<128x128xf32>
    %13 = arith.divf %9, %12 : vector<128x128xf32>
    %c0_9 = arith.constant 0 : index
    %c0_10 = arith.constant 0 : index
    %c0_11 = arith.constant 0 : index
    %c0_12 = arith.constant 0 : index
    %14 = vector.load %arg6[%c0_9, %c0_10, %c0_11, %c0_12] : memref<1x1x128x64xbf16, #tpu.memory_space<vmem>>, vector<1x1x128x64xbf16>
    %15 = vector.shape_cast %14 : vector<1x1x128x64xbf16> to vector<128x64xbf16>
    %16 = arith.truncf %13 : vector<128x128xf32> to vector<128x128xbf16>
    %cst_13 = arith.constant dense<0.000000e+00> : vector<128x64xf32>
    %17 = tpu.matmul %16, %15, %cst_13 {dimension_numbers = #tpu.dot_dimension_numbers<[1], [0], [0], [1], [0, 0, 1, 1], [], []>} : vector<128x128xbf16>, vector<128x64xbf16>, vector<128x64xf32> -> vector<128x64xf32>
    %18 = arith.truncf %17 : vector<128x64xf32> to vector<128x64xbf16>
    %c0_14 = arith.constant 0 : index
    %c0_15 = arith.constant 0 : index
    %c0_16 = arith.constant 0 : index
    %c0_17 = arith.constant 0 : index
    %19 = vector.load %arg7[%c0_14, %c0_15, %c0_16, %c0_17] : memref<1x1x128x64xbf16, #tpu.memory_space<vmem>>, vector<1x1x128x64xbf16>
    %20 = vector.shape_cast %19 : vector<1x1x128x64xbf16> to vector<128x64xbf16>
    %21 = vector.shape_cast %18 : vector<128x64xbf16> to vector<1x1x128x64xbf16>
    tpu.vector_store %arg7[%c0_14, %c0_15, %c0_16, %c0_17], %21 {strides = array<i32>} : memref<1x1x128x64xbf16, #tpu.memory_space<vmem>>, vector<1x1x128x64xbf16>, 
    return
  }
  func.func @transform_0(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32) -> (i32, i32, i32, i32) {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    return %arg0, %arg1, %arg2, %c0_i32 : i32, i32, i32, i32
  }
  func.func @transform_1(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32) -> (i32, i32, i32, i32) {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    return %arg0, %arg1, %arg3, %c0_i32 : i32, i32, i32, i32
  }
  func.func @transform_2(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32) -> (i32, i32, i32, i32) {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    return %arg0, %arg1, %arg3, %c0_i32 : i32, i32, i32, i32
  }
  func.func @transform_3(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32) -> (i32, i32, i32, i32) {
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    return %arg0, %arg1, %arg2, %c0_i32 : i32, i32, i32, i32
  }
}
