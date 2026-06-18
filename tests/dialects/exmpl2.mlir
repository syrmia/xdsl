module @add_kernel {
  func.func @main(%arg0: memref<128xf32, #tpu.memory_space<vmem>>, %arg1: memref<128xf32, #tpu.memory_space<vmem>>) attributes {dimension_semantics = [], scalar_prefetch = 0 : i64, scratch_operands = 0 : i64} {
    %c0 = arith.constant 0 : index
    %0 = vector.load %arg0[%c0] : memref<128xf32, #tpu.memory_space<vmem>>, vector<128xf32>
    %1 = math.sin %0 : vector<128xf32>
    %c0_0 = arith.constant 0 : index
    %2 = vector.load %arg1[%c0_0] : memref<128xf32, #tpu.memory_space<vmem>>, vector<128xf32>
    tpu.vector_store %arg1[%c0_0], %1 {strides = array<i32>} : memref<128xf32, #tpu.memory_space<vmem>>, vector<128xf32>, 
    return
  }
}
