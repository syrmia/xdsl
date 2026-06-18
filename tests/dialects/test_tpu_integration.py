import pytest

from xdsl.context import Context
from xdsl.dialects.arith import Arith
from xdsl.dialects.builtin import (
    Builtin,
    DenseArrayBase,
    IndexType,
    IntegerType,
    MemRefType,
    VectorType,
    f32,
    i1,
    i32,
)
from xdsl.dialects.func import Func
from xdsl.dialects.tpu import (
    TPU,
    DelayOp,
    PackFormat,
    PackFormatAttr,
    RegionOp,
    TraceOp,
    TraceStartOp,
    TraceStopOp,
    YieldOp,
)
from xdsl.dialects.tpu_conversions import (
    FPToSIOp,
    RoundingMode,
    RoundingModeAttr,
    SIToFPOp,
    TruncFOp,
)
from xdsl.dialects.tpu_memory import (
    LoadOp,
    StridedLoadOp,
    VectorLoadOp,
    VectorStoreOp,
)
from xdsl.dialects.tpu_memref import (
    CoreType,
    MemorySpace,
    MemorySpaceAttr,
    MemRefSliceOp,
    MemRefSqueezeOp,
)
from xdsl.dialects.tpu_pack import (
    CreateMaskOp,
    PackSubelementsOp,
    UnpackSubelementsOp,
)
from xdsl.dialects.tpu_shape import (
    ConcatenateOp,
    IotaOp,
    ReshapeOp,
    RotateOp,
)
from xdsl.ir import Attribute, Block, Region
from xdsl.parser import Parser
from xdsl.utils.test_value import create_ssa_value


def _vmem_memref(shape: list[int], elem_ty: Attribute = f32) -> MemRefType:
    return MemRefType(elem_ty, shape, memory_space=MemorySpaceAttr(MemorySpace.Vmem))


def _vmem_memref_with_core_type(
    shape: list[int],
    core_type: CoreType = CoreType.Tc,
    elem_ty=f32,
) -> MemRefType:
    return MemRefType(
        elem_ty,
        shape,
        memory_space=MemorySpaceAttr(MemorySpace.Vmem, core_type),
    )


def test_region_wrapping_memref_slice():
    mem_ty = _vmem_memref([16, 128])
    mem = create_ssa_value(mem_ty)
    idx0 = create_ssa_value(i32)
    idx1 = create_ssa_value(i32)
    out_ty = _vmem_memref([16, 128])
    slice_op = MemRefSliceOp.build(
        operands=[mem, [idx0, idx1], []],
        result_types=[out_ty],
    )

    region = Region([Block([slice_op, YieldOp()])])
    op = RegionOp(result_types=[], region=region)
    op.verify()


def test_trace_op_wrapping_memref_squeeze():
    in_ty = MemRefType(f32, [4, 1, 8])
    out_ty = MemRefType(f32, [4, 8])
    input = create_ssa_value(in_ty)
    sq_op = MemRefSqueezeOp.build(operands=[input], result_types=[out_ty])
    region = Region([Block([sq_op, YieldOp()])])
    op = TraceOp("dbg", 1, [], region)
    op.verify()


def test_memref_in_two_distinct_core_types():
    tc_ty = _vmem_memref_with_core_type([16, 8], CoreType.Tc)
    sc_ty = _vmem_memref_with_core_type([16, 8], CoreType.Sc_Scalar_Subcore)
    assert tc_ty != sc_ty
    tc_val = create_ssa_value(tc_ty)
    sc_val = create_ssa_value(sc_ty)
    assert tc_val.type == tc_ty
    assert sc_val.type == sc_ty


def test_vector_load_from_vmem_memref():
    mem_ty = _vmem_memref([8, 128])
    mem = create_ssa_value(mem_ty)
    idx0 = create_ssa_value(IndexType())
    idx1 = create_ssa_value(IndexType())
    op = VectorLoadOp(
        mem,
        [idx0, idx1],
        strides=DenseArrayBase.from_list(i32, []),
        result_type=VectorType(f32, [8, 128]),
    )
    op.verify()


def test_vector_store_to_vmem_memref():
    mem_ty = _vmem_memref([8, 128])
    mem = create_ssa_value(mem_ty)
    val = create_ssa_value(VectorType(f32, [8, 128]))
    idx0 = create_ssa_value(IndexType())
    idx1 = create_ssa_value(IndexType())
    op = VectorStoreOp(
        val,
        mem,
        [idx0, idx1],
        strides=DenseArrayBase.from_list(i32, []),
    )
    op.verify()


def test_strided_load_from_vmem():
    mem_ty = _vmem_memref([8, 128])
    mem = create_ssa_value(mem_ty)
    idx0 = create_ssa_value(IndexType())
    idx1 = create_ssa_value(IndexType())
    op = StridedLoadOp(
        mem,
        [idx0, idx1],
        strides=DenseArrayBase.from_list(i32, [1, 2]),
        result_type=VectorType(f32, [8, 128]),
    )
    op.verify()


def test_load_op_with_vmem_memref():
    mem_ty = _vmem_memref([8, 128])
    mem = create_ssa_value(mem_ty)
    idx0 = create_ssa_value(IndexType())
    idx1 = create_ssa_value(IndexType())
    op = LoadOp(
        mem,
        [idx0, idx1],
        sublane_mask=DenseArrayBase.from_list(i1, [1] * 8),
        result_type=VectorType(f32, [8, 128]),
    )
    op.verify()


def test_vector_load_rejects_non_vmem_memref():
    smem_ty = MemRefType(
        f32,
        [8, 128],
        memory_space=MemorySpaceAttr(MemorySpace.Smem),
    )
    mem = create_ssa_value(smem_ty)
    idx0 = create_ssa_value(IndexType())
    idx1 = create_ssa_value(IndexType())
    op = VectorLoadOp(
        mem,
        [idx0, idx1],
        strides=DenseArrayBase.from_list(i32, []),
        result_type=VectorType(f32, [8, 128]),
    )
    from xdsl.utils.exceptions import VerifyException

    with pytest.raises(VerifyException, match="Expected base memref to be in VMEM"):
        op.verify()


def test_load_into_rotate():
    mem_ty = _vmem_memref([8, 128])
    mem = create_ssa_value(mem_ty)
    idx0 = create_ssa_value(IndexType())
    idx1 = create_ssa_value(IndexType())
    load = VectorLoadOp(
        mem,
        [idx0, idx1],
        strides=DenseArrayBase.from_list(i32, []),
        result_type=VectorType(f32, [8, 128]),
    )
    load.verify()

    rotated = RotateOp(load.result, amount=4, dimension=1)
    rotated.verify()


def test_load_into_reshape():
    mem_ty = _vmem_memref([16, 128])
    mem = create_ssa_value(mem_ty)
    idx0 = create_ssa_value(IndexType())
    idx1 = create_ssa_value(IndexType())
    load = VectorLoadOp(
        mem,
        [idx0, idx1],
        strides=DenseArrayBase.from_list(i32, []),
        result_type=VectorType(f32, [16, 128]),
    )

    reshaped = ReshapeOp(load.result, result_type=VectorType(f32, [8, 256]))
    reshaped.verify()


def test_iota_into_store():
    iota = IotaOp(dimensions=[0], result_type=VectorType(i32, [8, 128]))
    iota.verify()

    mem_ty = _vmem_memref([8, 128], elem_ty=i32)
    mem = create_ssa_value(mem_ty)
    idx0 = create_ssa_value(IndexType())
    idx1 = create_ssa_value(IndexType())
    store = VectorStoreOp(
        iota.output,
        mem,
        [idx0, idx1],
        strides=DenseArrayBase.from_list(i32, []),
    )
    store.verify()


def test_region_with_load_store_kernel():
    src_ty = _vmem_memref([8, 128])
    dst_ty = _vmem_memref([8, 128])
    src = create_ssa_value(src_ty)
    dst = create_ssa_value(dst_ty)
    idx0 = create_ssa_value(IndexType())
    idx1 = create_ssa_value(IndexType())

    load = VectorLoadOp(
        src,
        [idx0, idx1],
        strides=DenseArrayBase.from_list(i32, []),
        result_type=VectorType(f32, [8, 128]),
    )
    store = VectorStoreOp(
        load.result,
        dst,
        [idx0, idx1],
        strides=DenseArrayBase.from_list(i32, []),
    )

    region = Region([Block([load, store, YieldOp()])])
    op = RegionOp(result_types=[], region=region)
    op.verify()


def test_trace_around_kernel():
    src_ty = _vmem_memref([8, 128])
    dst_ty = _vmem_memref([8, 128])
    src = create_ssa_value(src_ty)
    dst = create_ssa_value(dst_ty)
    idx0 = create_ssa_value(IndexType())
    idx1 = create_ssa_value(IndexType())

    load = VectorLoadOp(
        src,
        [idx0, idx1],
        strides=DenseArrayBase.from_list(i32, []),
        result_type=VectorType(f32, [8, 128]),
    )
    store = VectorStoreOp(
        load.result,
        dst,
        [idx0, idx1],
        strides=DenseArrayBase.from_list(i32, []),
    )

    region = Region([Block([load, store, YieldOp()])])
    op = TraceOp("kernel_body", 1, [], region)
    op.verify()


def test_trace_start_delay_trace_stop_pattern():
    nanos = create_ssa_value(i32)

    trace_start = TraceStartOp("section_a", 0)
    delay = DelayOp(nanos)
    trace_stop = TraceStopOp()

    trace_start.verify()
    delay.verify()
    trace_stop.verify()


def test_load_truncf_store():
    src_ty = _vmem_memref([8, 128])
    src = create_ssa_value(src_ty)
    idx0 = create_ssa_value(IndexType())
    idx1 = create_ssa_value(IndexType())

    load = VectorLoadOp(
        src,
        [idx0, idx1],
        strides=DenseArrayBase.from_list(i32, []),
        result_type=VectorType(f32, [8, 128]),
    )

    trunc = TruncFOp(
        load.result,
        target_type=VectorType(f32, [8, 128]),
        rounding_mode=RoundingModeAttr(RoundingMode.To_Nearest_Even),
    )
    trunc.verify()


def test_load_fptosi_concatenate():
    src_ty = _vmem_memref([8, 64])
    src = create_ssa_value(src_ty)
    idx0 = create_ssa_value(IndexType())
    idx1 = create_ssa_value(IndexType())

    load = VectorLoadOp(
        src,
        [idx0, idx1],
        strides=DenseArrayBase.from_list(i32, []),
        result_type=VectorType(f32, [8, 64]),
    )

    converted = FPToSIOp(
        load.result,
        target_type=VectorType(i32, [8, 64]),
        rounding_mode=RoundingModeAttr(RoundingMode.To_Nearest_Even),
    )
    converted.verify()

    other_int = create_ssa_value(VectorType(i32, [8, 64]))
    concat = ConcatenateOp(
        [converted.output, other_int],
        dimension=1,
    )
    concat.verify()


def test_sitofp_then_store():
    vec = create_ssa_value(VectorType(i32, [8, 128]))
    converted = SIToFPOp(
        vec,
        target_type=VectorType(f32, [8, 128]),
        rounding_mode=RoundingModeAttr(RoundingMode.To_Nearest_Even),
    )
    converted.verify()

    dst_ty = _vmem_memref([8, 128])
    dst = create_ssa_value(dst_ty)
    idx0 = create_ssa_value(IndexType())
    idx1 = create_ssa_value(IndexType())
    store = VectorStoreOp(
        converted.output,
        dst,
        [idx0, idx1],
        strides=DenseArrayBase.from_list(i32, []),
    )
    store.verify()


def test_iota_into_pack_subelements():
    iota1 = IotaOp(dimensions=[0], result_type=VectorType(IntegerType(32), [8]))
    iota2 = IotaOp(dimensions=[0], result_type=VectorType(IntegerType(32), [8]))

    pack = PackSubelementsOp(
        [iota1.output, iota2.output],
        positions=[0, 1],
        pack_format=PackFormatAttr(PackFormat.Compressed),
        result_type=VectorType(IntegerType(16), [8]),
    )
    pack.verify()


def test_unpack_then_concatenate():
    src1 = create_ssa_value(VectorType(IntegerType(16), [8]))
    src2 = create_ssa_value(VectorType(IntegerType(16), [8]))

    unpack1 = UnpackSubelementsOp(
        src1,
        index=0,
        pack_format=PackFormatAttr(PackFormat.Compressed),
        result_type=VectorType(IntegerType(32), [8]),
    )
    unpack2 = UnpackSubelementsOp(
        src2,
        index=0,
        pack_format=PackFormatAttr(PackFormat.Compressed),
        result_type=VectorType(IntegerType(32), [8]),
    )

    concat = ConcatenateOp(
        [unpack1.output, unpack2.output],
        dimension=0,
    )
    concat.verify()


def test_create_mask_into_rotate():
    low1 = create_ssa_value(IndexType())
    low2 = create_ssa_value(IndexType())
    high1 = create_ssa_value(IndexType())
    high2 = create_ssa_value(IndexType())
    mask = CreateMaskOp(
        [low1, low2],
        [high1, high2],
        result_type=VectorType(i1, [8, 128]),
    )
    mask.verify()

    vec = create_ssa_value(VectorType(f32, [8, 128]))
    rotate = RotateOp(vec, amount=4, dimension=1)
    rotate.verify()


def test_round_trip_full_kernel():
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(TPU)
    ctx.load_dialect(Func)
    ctx.load_dialect(Arith)

    text = """
    builtin.module {
      func.func @kernel(
          %src: memref<8x128xf32, #tpu.memory_space<vmem>>,
          %v: vector<8x128xf32>,
          %dst: memref<8x128xf32, #tpu.memory_space<vmem>>
      ) {
        %c0 = arith.constant 0 : index
        tpu.vector_store %dst[%c0, %c0], %v {strides = array<i32>}
            : memref<8x128xf32, #tpu.memory_space<vmem>>, vector<8x128xf32>
        func.return
      }
    }
    """
    module = Parser(ctx, text).parse_module()
    module.verify()


def test_round_trip_kernel_with_trace():
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(TPU)
    ctx.load_dialect(Func)
    ctx.load_dialect(Arith)

    text = """
    builtin.module {
      func.func @traced_kernel(%v: vector<8x128xf32>) {
        "tpu.trace_start"() {message = "begin", level = 0 : i32} : () -> ()
        "tpu.trace_stop"() : () -> ()
        func.return
      }
    }
    """
    module = Parser(ctx, text).parse_module()
    module.verify()


def _parse_and_verify_kernel(path: str):
    from xdsl.context import Context
    from xdsl.dialects.arith import Arith
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func
    from xdsl.dialects.math import Math
    from xdsl.dialects.scf import Scf
    from xdsl.dialects.tpu import TPU
    from xdsl.dialects.vector import Vector
    from xdsl.parser import Parser

    ctx = Context()
    for d in (Builtin, Func, Arith, Vector, Scf, TPU, Math):
        ctx.load_dialect(d)
    with open(path) as f:
        module = Parser(ctx, f.read()).parse_module()
    module.verify()
    return module


def test_example_kernel_1_parses_and_verifies():
    _parse_and_verify_kernel("tests/dialects/exmpl1.mlir")


def test_example_kernel_2_parses_and_verifies():
    _parse_and_verify_kernel("tests/dialects/exmpl2.mlir")


def test_flash_attention_kernel_parses_and_verifies():
    _parse_and_verify_kernel("tests/dialects/exmpl3.mlir")
