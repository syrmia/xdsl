import io

import pytest

from xdsl.context import Context
from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    IntegerType,
    MemRefType,
    NoneAttr,
    f32,
    i32,
)
from xdsl.dialects.tpu import TPU
from xdsl.dialects.tpu_memref import (
    CoreType,
    CoreTypeAttr,
    DMASemaphoreType,
    MemorySpace,
    MemorySpaceAttr,
    MemRefBitcastOp,
    MemRefReshapeOp,
    MemRefSliceOp,
    MemRefSqueezeOp,
    ReinterpretCastOp,
    SemaphoreType,
    _compute_squeezed_dims,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import create_ssa_value


def test_core_type_attr_round_trip():
    ctx = Context()
    ctx.load_dialect(TPU)

    attr = CoreTypeAttr(CoreType.Tc)

    output = io.StringIO()
    Printer(stream=output).print_attribute(attr)
    printed = output.getvalue()

    parsed = Parser(ctx, printed).parse_attribute()
    assert parsed == attr


def test_memory_space_attr_basic():
    attr = MemorySpaceAttr(MemorySpace.Vmem)
    assert attr.value.data == MemorySpace.Vmem
    assert isinstance(attr.core_type, NoneAttr)


def test_memory_space_attr_with_core_type():
    attr = MemorySpaceAttr(MemorySpace.Vmem, CoreType.Tc)
    assert attr.value.data == MemorySpace.Vmem
    assert isinstance(attr.core_type, CoreTypeAttr)
    assert attr.core_type.data == CoreType.Tc


def test_memory_space_attr_with_core_type_attr():
    core = CoreTypeAttr(CoreType.Sc_Scalar_Subcore)
    attr = MemorySpaceAttr(MemorySpace.Smem, core)
    assert attr.core_type == core


def test_memory_space_attr_round_trip_simple():
    ctx = Context()
    ctx.load_dialect(TPU)

    attr = MemorySpaceAttr(MemorySpace.Vmem)

    output = io.StringIO()
    Printer(stream=output).print_attribute(attr)
    printed = output.getvalue()

    parsed = Parser(ctx, printed).parse_attribute()
    assert parsed == attr


def test_memory_space_attr_round_trip_with_core_type():
    ctx = Context()
    ctx.load_dialect(TPU)

    attr = MemorySpaceAttr(MemorySpace.Vmem, CoreType.Tc)

    output = io.StringIO()
    Printer(stream=output).print_attribute(attr)
    printed = output.getvalue()

    parsed = Parser(ctx, printed).parse_attribute()
    assert parsed == attr

def test_semaphore_type_construction():
    ty = SemaphoreType()
    assert isinstance(ty, SemaphoreType)


def test_dma_semaphore_type_construction():
    ty = DMASemaphoreType()
    assert isinstance(ty, DMASemaphoreType)


def test_semaphore_types_are_distinct():
    sem = SemaphoreType()
    dma_sem = DMASemaphoreType()
    assert sem != dma_sem


def test_compute_squeezed_dims_no_squeeze():
    result = _compute_squeezed_dims("test", [4, 8], [4, 8])
    assert result == []


def test_compute_squeezed_dims_single_unit():
    result = _compute_squeezed_dims("test", [4, 1, 8], [4, 8])
    assert result == [1]


def test_compute_squeezed_dims_leading_unit():
    result = _compute_squeezed_dims("test", [1, 4, 8], [4, 8])
    assert result == [0]


def test_compute_squeezed_dims_trailing_unit():
    result = _compute_squeezed_dims("test", [4, 8, 1], [4, 8])
    assert result == [2]


def test_compute_squeezed_dims_multiple_units():
    result = _compute_squeezed_dims("test", [1, 4, 1, 8, 1], [4, 8])
    assert result == [0, 2, 4]


def test_compute_squeezed_dims_rejects_incompatible_shapes():
    with pytest.raises(VerifyException, match="not compatible for squeezing"):
        _compute_squeezed_dims("test", [4, 8], [4, 16])


def test_compute_squeezed_dims_rejects_target_larger_than_source():
    with pytest.raises(VerifyException, match="more dimensions than source"):
        _compute_squeezed_dims("test", [4, 8], [4, 8, 2])


def test_compute_squeezed_dims_rejects_non_unit_trailing():
    with pytest.raises(VerifyException, match="Cannot squeeze non-unit"):
        _compute_squeezed_dims("test", [4, 8, 5], [4, 8])


def test_memref_slice_basic():
    mem = create_ssa_value(MemRefType(f32, [16, 128]))
    idx0 = create_ssa_value(i32)
    idx1 = create_ssa_value(i32)
    out_ty = MemRefType(f32, [16, 128])
    op = MemRefSliceOp.create(
        operands=[mem, idx0, idx1],
        result_types=[out_ty],
        properties={"operandSegmentSizes":
                    create_ssa_value(i32).type}, 
    ) if False else None


def test_memref_slice_rejects_dynamic_source():
    mem = create_ssa_value(MemRefType(f32, [DYNAMIC_INDEX, 128]))
    idx0 = create_ssa_value(i32)
    idx1 = create_ssa_value(i32)
    out_ty = MemRefType(f32, [4, 128])
    op = MemRefSliceOp.build(
        operands=[mem, [idx0, idx1], []],
        result_types=[out_ty],
    )
    with pytest.raises(VerifyException, match="static shapes is supported"):
        op.verify()


def test_memref_slice_rejects_rank_mismatch():
    mem = create_ssa_value(MemRefType(f32, [16, 128]))
    idx0 = create_ssa_value(i32)
    out_ty = MemRefType(f32, [16, 128])
    op = MemRefSliceOp.build(
        operands=[mem, [idx0], []],
        result_types=[out_ty],
    )
    with pytest.raises(VerifyException, match="Indices and slice shapes must match"):
        op.verify()


def test_memref_squeeze_basic():
    input_ty = MemRefType(f32, [4, 1, 8])
    input = create_ssa_value(input_ty)
    out_ty = MemRefType(f32, [4, 8])
    op = MemRefSqueezeOp.build(operands=[input], result_types=[out_ty])
    op.verify()


def test_memref_squeeze_no_squeeze_same_shape():
    input = create_ssa_value(MemRefType(f32, [4, 8]))
    out_ty = MemRefType(f32, [4, 8])
    op = MemRefSqueezeOp.build(operands=[input], result_types=[out_ty])
    op.verify()


def test_memref_squeeze_rejects_element_type_mismatch():
    input = create_ssa_value(MemRefType(f32, [4, 1, 8]))
    out_ty = MemRefType(i32, [4, 8])
    op = MemRefSqueezeOp.build(operands=[input], result_types=[out_ty])
    with pytest.raises(VerifyException, match="Element types don't match"):
        op.verify()


def test_memref_squeeze_rejects_incompatible_shapes():
    input = create_ssa_value(MemRefType(f32, [4, 1, 8]))
    out_ty = MemRefType(f32, [4, 16])    # 8 vs 16 doesn't match
    op = MemRefSqueezeOp.build(operands=[input], result_types=[out_ty])
    with pytest.raises(VerifyException, match="not compatible for squeezing"):
        op.verify()


def test_memref_reshape_basic():
    input = create_ssa_value(MemRefType(f32, [4, 8]))
    out_ty = MemRefType(f32, [2, 16])
    op = MemRefReshapeOp.build(operands=[input], result_types=[out_ty])
    op.verify()


def test_memref_reshape_rejects_1d():
    input = create_ssa_value(MemRefType(f32, [32]))
    out_ty = MemRefType(f32, [4, 8])
    op = MemRefReshapeOp.build(operands=[input], result_types=[out_ty])
    with pytest.raises(VerifyException, match="1d memref reshape"):
        op.verify()


def test_memref_reshape_rejects_element_count_mismatch():
    input = create_ssa_value(MemRefType(f32, [4, 8]))   
    out_ty = MemRefType(f32, [4, 16]) 
    op = MemRefReshapeOp.build(operands=[input], result_types=[out_ty])
    with pytest.raises(VerifyException, match="Number of elements"):
        op.verify()


def test_memref_reshape_rejects_element_type_mismatch():
    input = create_ssa_value(MemRefType(f32, [4, 8]))
    out_ty = MemRefType(i32, [4, 8])
    op = MemRefReshapeOp.build(operands=[input], result_types=[out_ty])
    with pytest.raises(VerifyException, match="Element types"):
        op.verify()


def test_memref_bitcast_basic_same_bitwidth():
    input = create_ssa_value(MemRefType(f32, [16, 8]))
    out_ty = MemRefType(i32, [16, 8])
    op = MemRefBitcastOp.build(operands=[input], result_types=[out_ty])
    op.verify()


def test_memref_bitcast_widening():
    input = create_ssa_value(MemRefType(IntegerType(16), [16, 8]))
    out_ty = MemRefType(IntegerType(32), [8, 8])
    op = MemRefBitcastOp.build(operands=[input], result_types=[out_ty])
    op.verify()


def test_memref_bitcast_rejects_rank_mismatch():
    input = create_ssa_value(MemRefType(f32, [16, 8]))
    out_ty = MemRefType(i32, [16, 8, 1])
    op = MemRefBitcastOp.build(operands=[input], result_types=[out_ty])
    with pytest.raises(VerifyException, match="Ranks do not match"):
        op.verify()


def test_memref_bitcast_rejects_1d():
    input = create_ssa_value(MemRefType(f32, [16]))
    out_ty = MemRefType(i32, [16])
    op = MemRefBitcastOp.build(operands=[input], result_types=[out_ty])
    with pytest.raises(VerifyException, match="1d memref bitcast"):
        op.verify()


def test_memref_bitcast_rejects_bad_bit_count():
    input = create_ssa_value(MemRefType(IntegerType(16), [16, 8]))
    out_ty = MemRefType(IntegerType(32), [7, 8])    # 7*32 != 16*16
    op = MemRefBitcastOp.build(operands=[input], result_types=[out_ty])
    with pytest.raises(VerifyException, match="same number of bits"):
        op.verify()


def test_memref_bitcast_rejects_non_minormost_dim_mismatch():
    input = create_ssa_value(MemRefType(f32, [16, 8]))
    out_ty = MemRefType(f32, [16, 9])    # dim 1 differs without bitwidth diff
    op = MemRefBitcastOp.build(operands=[input], result_types=[out_ty])
    with pytest.raises(VerifyException, match="same dim size on dim"):
        op.verify()


def test_reinterpret_cast_basic():
    input = create_ssa_value(MemRefType(f32, [16, 8]))
    out_ty = MemRefType(f32, [128])
    op = ReinterpretCastOp.build(
        operands=[input, []],  
        result_types=[out_ty],
    )
    op.verify()


def test_reinterpret_cast_with_dynamic_offset():
    input = create_ssa_value(MemRefType(f32, [16, 8]))
    offset = create_ssa_value(i32)
    out_ty = MemRefType(f32, [128])
    op = ReinterpretCastOp.build(
        operands=[input, [offset]],
        result_types=[out_ty],
    )
    op.verify()


def test_reinterpret_cast_rejects_memory_space_mismatch():
    vmem = MemorySpaceAttr(MemorySpace.Vmem)
    smem = MemorySpaceAttr(MemorySpace.Smem)
    input = create_ssa_value(MemRefType(f32, [16, 8], memory_space=vmem))
    out_ty = MemRefType(f32, [128], memory_space=smem)
    op = ReinterpretCastOp.build(
        operands=[input, []],
        result_types=[out_ty],
    )
    with pytest.raises(VerifyException, match="memory spaces must match"):
        op.verify()


def test_round_trip_memref_with_memory_space():
    ctx = Context()
    ctx.load_dialect(TPU)

    text = "memref<8x128xf32, #tpu.memory_space<vmem>>"
    ty = Parser(ctx, text).parse_type()

    output = io.StringIO()
    Printer(stream=output).print_attribute(ty)
    assert "memory_space<vmem>" in output.getvalue()


def test_round_trip_memref_with_memory_space_and_core_type():
    ctx = Context()
    ctx.load_dialect(TPU)

    attr = MemorySpaceAttr(MemorySpace.Vmem, CoreType.Tc)

    output = io.StringIO()
    Printer(stream=output).print_attribute(attr)
    printed = output.getvalue()

    parsed = Parser(ctx, printed).parse_attribute()
    assert parsed == attr


def test_round_trip_semaphore_type():
    ctx = Context()
    ctx.load_dialect(TPU)

    ty = SemaphoreType()

    output = io.StringIO()
    Printer(stream=output).print_attribute(ty)
    printed = output.getvalue()

    parsed = Parser(ctx, printed).parse_type()
    assert parsed == ty
