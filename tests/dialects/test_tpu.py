

import pytest
from typing import cast

from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.func import FuncOp
from xdsl.dialects.builtin import AnyFloat, ArrayAttr, IntegerAttr, IntegerType, MemRefType, StringAttr, i32, i64, f32, bf16, f64
from xdsl.dialects.tpu import (
    TPU,
    CoreType,
    CoreTypeAttr,
    DelayOp,
    DotDimensionNumbersAttr,
    Float8EXMYType,
    PipelineMode, PipelineModeAttr,
    RegionOp,
    RevisitMode, RevisitModeAttr,
    DimensionSemantics, DimensionSemanticsAttr,
    ContractPrecision, ContractPrecisionAttr,
    PackFormat, PackFormatAttr,
    RoundingMode, RoundingModeAttr,
    SemaphoreType,
    DMASemaphoreType,
    TraceOp,
    TraceStartOp,
    TraceStopOp,
    TraceValueOp,
    YieldOp
  )
from xdsl.ir.core import Block, Region
from xdsl.utils.exceptions import VerifyException



def test_core_type_enum_values():
    assert str(CoreType.Tc) == "tc"
    assert str(CoreType.Sc_Scalar_Subcore) == "sc_scalar_subcore"
    assert str(CoreType.Sc_Vector_Subcore) == "sc_vector_subcore"

def test_core_type_attr_construction():
    attr = CoreTypeAttr(CoreType.Tc)
    assert attr.data == CoreType.Tc

def test_core_type_attr_equality():
    a1 = CoreTypeAttr(CoreType.Tc)
    a2 = CoreTypeAttr(CoreType.Tc)
    a3 = CoreTypeAttr(CoreType.Sc_Vector_Subcore)
 
    assert a1 == a2
    assert a1 != a3
    assert hash(a1) == hash(a2)
    assert {a1, a2, a3} == {a1, a3}
 
 
@pytest.mark.parametrize(
    "core_type",
    [CoreType.Tc, CoreType.Sc_Scalar_Subcore, CoreType.Sc_Vector_Subcore],
)
def test_core_type_attr_roundtrip_each_variant(core_type: CoreType):
    attr = CoreTypeAttr(core_type)
    assert attr.data == core_type
 
def test_from_op_when_attr_present():
    func = FuncOp("not_main", ((), ()))
    func.attributes["tpu.core_type"] = CoreTypeAttr(CoreType.Sc_Vector_Subcore)
    assert CoreTypeAttr.from_op(func) == CoreType.Sc_Vector_Subcore
 
 
def test_from_op_main_fallback():
    func = FuncOp("main", ((), ()))
    assert CoreTypeAttr.from_op(func) == CoreType.Tc
 
 
def test_from_op_non_main_no_attr():
    func = FuncOp("not_main", ((), ()))
    assert CoreTypeAttr.from_op(func) is None
 
 
def test_from_op_wrong_attr_type():
    func = FuncOp("not_main", ((), ()))
    func.attributes["tpu.core_type"] = StringAttr("not_a_core_type")
    assert CoreTypeAttr.from_op(func) is None

def _i64_array(*values: int) -> ArrayAttr[IntegerAttr[IntegerType]]:
    return ArrayAttr(
        [
            cast(IntegerAttr[IntegerType], IntegerAttr(v, i64))
            for v in values
        ]
    )
 
 
def test_dot_dimension_numbers_full():
    dot = DotDimensionNumbersAttr(
        _i64_array(1),
        _i64_array(0),
        _i64_array(0),
        _i64_array(1),
        _i64_array(0, 0, 1, 1),
        _i64_array(0),
        _i64_array(0),
    )
 
    assert [a.value.data for a in dot.lhs_contracting_dims.data] == [1]
    assert [a.value.data for a in dot.rhs_contracting_dims.data] == [0]
    assert [a.value.data for a in dot.lhs_non_contracting_dims.data] == [0]
    assert [a.value.data for a in dot.rhs_non_contracting_dims.data] == [1]
    assert [a.value.data for a in dot.output_dim_order.data] == [0, 0, 1, 1]
    assert [a.value.data for a in dot.lhs_batch_dims.data] == [0]
    assert [a.value.data for a in dot.rhs_batch_dims.data] == [0]
 
 
def test_dot_dimension_numbers_optionals_empty():
    empty = _i64_array()
    dot = DotDimensionNumbersAttr(
        _i64_array(1),
        _i64_array(0),
        _i64_array(0),
        empty,                        
        _i64_array(0, 0, 1, 1),
        empty,                        
        empty,                        
    )
    assert len(dot.rhs_non_contracting_dims.data) == 0
    assert len(dot.lhs_batch_dims.data) == 0
    assert len(dot.rhs_batch_dims.data) == 0
 
 
def test_dot_dimension_numbers_equality():
    a = DotDimensionNumbersAttr(
        _i64_array(1), _i64_array(0), _i64_array(0),
        _i64_array(), _i64_array(0, 1),
        _i64_array(), _i64_array(),
    )
    b = DotDimensionNumbersAttr(
        _i64_array(1), _i64_array(0), _i64_array(0),
        _i64_array(), _i64_array(0, 1),
        _i64_array(), _i64_array(),
    )
    c = DotDimensionNumbersAttr(
        _i64_array(2), _i64_array(0), _i64_array(0),  # first field differs
        _i64_array(), _i64_array(0, 1),
        _i64_array(), _i64_array(),
    )
    assert a == b
    assert a != c
 
def test_float8_exmy_construction_f32():
    ty = Float8EXMYType(f32)
    assert ty.underlying_type == f32
 
 
def test_float8_exmy_construction_other_floats():
    assert Float8EXMYType(bf16).underlying_type == bf16
    assert Float8EXMYType(f64).underlying_type == f64
 
 
def test_float8_exmy_rejects_non_float():
    with pytest.raises(Exception):
    #   Float8EXMYType(i32)
        Float8EXMYType(cast(AnyFloat, i32))
 
def test_float8_exmy_equality():
    assert Float8EXMYType(f32) == Float8EXMYType(f32)
    assert Float8EXMYType(f32) != Float8EXMYType(bf16)

 
def test_dialect_name():
    assert TPU.name == "tpu"
 
def test_dialect_registers_attributes_and_types():
    registered = set(TPU.attributes)
    assert CoreTypeAttr in registered
    assert DotDimensionNumbersAttr in registered
    assert Float8EXMYType in registered
    assert SemaphoreType in registered 
    assert DMASemaphoreType in registered


def test_pipeline_mode_enum_values():
    assert str(PipelineMode.Synchronous) == "synchronous"
    assert str(PipelineMode.Double_Buffered) == "double_buffered"
 
 
@pytest.mark.parametrize(
    "variant", [PipelineMode.Synchronous, PipelineMode.Double_Buffered]
)
def test_pipeline_mode_attr_roundtrip(variant: PipelineMode):
    attr = PipelineModeAttr(variant)
    assert attr.data == variant
 
 
def test_pipeline_mode_attr_equality():
    a = PipelineModeAttr(PipelineMode.Synchronous)
    b = PipelineModeAttr(PipelineMode.Synchronous)
    c = PipelineModeAttr(PipelineMode.Double_Buffered)
    assert a == b
    assert a != c
    assert hash(a) == hash(b)

def test_revisit_mode_enum_values():
    assert str(RevisitMode.Immediate) == "immediate"
    assert str(RevisitMode.Any) == "any"
 
 
@pytest.mark.parametrize("variant", [RevisitMode.Immediate, RevisitMode.Any])
def test_revisit_mode_attr_roundtrip(variant: RevisitMode):
    attr = RevisitModeAttr(variant)
    assert attr.data == variant
 
def test_dimension_semantics_enum_values():
    assert str(DimensionSemantics.Parallel) == "parallel"
    assert str(DimensionSemantics.Arbitrary) == "arbitrary"
    assert str(DimensionSemantics.Core_Parallel) == "core_parallel"
    assert str(DimensionSemantics.Subcore_Parallel) == "subcore_parallel"
 
 
@pytest.mark.parametrize(
    "variant",
    [
        DimensionSemantics.Parallel,
        DimensionSemantics.Arbitrary,
        DimensionSemantics.Core_Parallel,
        DimensionSemantics.Subcore_Parallel,
    ],
)
def test_dimension_semantics_attr_roundtrip(variant: DimensionSemantics):
    attr = DimensionSemanticsAttr(variant)
    assert attr.data == variant
 
def test_contract_precision_enum_values():
    assert str(ContractPrecision.Bf16) == "bf16"
    assert str(ContractPrecision.Fp32) == "fp32"
 
 
@pytest.mark.parametrize(
    "variant", [ContractPrecision.Bf16, ContractPrecision.Fp32]
)
def test_contract_precision_attr_roundtrip(variant: ContractPrecision):
    attr = ContractPrecisionAttr(variant)
    assert attr.data == variant
 
def test_pack_format_enum_values():
    assert str(PackFormat.Compressed) == "compressed"
    assert str(PackFormat.Interleaved) == "interleaved"
 
 
@pytest.mark.parametrize(
    "variant", [PackFormat.Compressed, PackFormat.Interleaved]
)
def test_pack_format_attr_roundtrip(variant: PackFormat):
    attr = PackFormatAttr(variant)
    assert attr.data == variant
 
def test_rounding_mode_enum_values():
    assert str(RoundingMode.Towards_Zero) == "towards_zero"
    assert str(RoundingMode.To_Nearest_Even) == "to_nearest_even"
 
 
@pytest.mark.parametrize(
    "variant", [RoundingMode.Towards_Zero, RoundingMode.To_Nearest_Even]
)
def test_rounding_mode_attr_roundtrip(variant: RoundingMode):
    attr = RoundingModeAttr(variant)
    assert attr.data == variant
 


def test_semaphore_type_construction():
    sem = SemaphoreType()
    dma_sem = DMASemaphoreType()
    assert isinstance(sem, SemaphoreType)
    assert isinstance(dma_sem, DMASemaphoreType)


def test_semaphore_types_distinct():
    assert SemaphoreType() != DMASemaphoreType()


def test_semaphore_type_equality():
    assert SemaphoreType() == SemaphoreType()
    assert DMASemaphoreType() == DMASemaphoreType()


def test_semaphore_as_memref_element():
    from xdsl.dialects.builtin import MemRefType
    m_sem = MemRefType(SemaphoreType(), [8])
    m_dma = MemRefType(DMASemaphoreType(), [4])
    assert isinstance(m_sem, MemRefType)
    assert isinstance(m_dma, MemRefType)
    assert m_sem.get_element_type() == SemaphoreType()
    assert m_dma.get_element_type() == DMASemaphoreType()


def test_yield_op_no_operands():
    """YieldOp with zero operands is valid (variadic accepts 0)."""
    op = YieldOp()
    assert len(op.arguments) == 0
 
 
def test_yield_op_one_operand():
    a = ConstantOp.from_int_and_width(1, i32)
    op = YieldOp(a)
    assert len(op.arguments) == 1
    assert op.arguments[0].type == i32
 
 
def test_yield_op_multiple_operands():
    a = ConstantOp.from_int_and_width(1, i32)
    b = ConstantOp.from_int_and_width(2, i32)
    c = ConstantOp.from_int_and_width(3, i32)
    op = YieldOp(a, b, c)
    assert len(op.arguments) == 3
 
def _make_region_with_yield(*values: int) -> Region:
    """Build a single-block region whose terminator yields the given
    integer constants."""
    block = Block()
    consts = [ConstantOp.from_int_and_width(v, i32) for v in values]
    for c in consts:
        block.add_op(c)
    block.add_op(YieldOp(*consts))
    return Region([block])
 
 
def test_region_op_no_results():
    """RegionOp with empty result types and a yield-with-no-values is
    structurally well-formed."""
    block = Block()
    block.add_op(YieldOp())
    region = Region([block])
    op = RegionOp([], region)
    op.verify() 
    assert len(op.results_) == 0
 
 
def test_region_op_with_int_results():
    region = _make_region_with_yield(1, 2)
    op = RegionOp([i32, i32], region)
    op.verify()
    assert len(op.results_) == 2
    assert all(r.type == i32 for r in op.results_)
 
 
def test_region_op_rejects_memref_result():
    """A memref result type is rejected by the verifier (mirrors the
    C++ check: result must be float/int/index/vector)."""
    region = Region([Block([YieldOp()])])
    op = RegionOp([MemRefType(i32, [4])], region)
    with pytest.raises(VerifyException, match="float, int, index"):
        op.verify()

def test_trace_op_construction():
    region = Region([Block([YieldOp()])])
    op = TraceOp("hello", 0, [], region)
    assert op.message.data == "hello"
    assert op.level.value.data == 0
    assert len(op.results_) == 0
 
 
def test_trace_op_accepts_string_and_int_attr_objects():
    """The constructor accepts both raw str/int and pre-built attrs."""
    region = Region([Block([YieldOp()])])
    op = TraceOp(StringAttr("hi"), IntegerAttr(5, i32), [], region)
    assert op.message.data == "hi"
    assert op.level.value.data == 5

def test_trace_start_op():
    op = TraceStartOp("scope_a", 1)
    assert op.message.data == "scope_a"
    assert op.level.value.data == 1
 
 
def test_trace_stop_op():
    op = TraceStopOp()
    assert len(op.operands) == 0
    assert len(op.results) == 0
 
def test_trace_value_op_with_i32():
    v = ConstantOp.from_int_and_width(42, i32)
    op = TraceValueOp(v, "my_value")
    assert op.value.type == i32
    assert op.label.data == "my_value"

def test_delay_op():
    n = ConstantOp.from_int_and_width(1000, i32)
    op = DelayOp(n)
    assert op.nanos.type == i32
    assert len(op.results) == 0

def test_dialect_registers_structural_ops():
      """The seven Family-1 structural ops are registered."""
      registered = set(TPU.operations)
      assert RegionOp in registered
      assert TraceOp in registered
      assert TraceStartOp in registered
      assert TraceStopOp in registered
      assert TraceValueOp in registered
      assert YieldOp in registered
      assert DelayOp in registered
