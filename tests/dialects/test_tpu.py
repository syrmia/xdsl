import io

import pytest

from xdsl.context import Context
from xdsl.dialects.builtin import (
    ArrayAttr,
    Float32Type,
    IntegerAttr,
    IntegerType,
    StringAttr,
    VectorType,
    f32,
    i32,
    i64,
)
from xdsl.dialects.tpu import (
    TPU,
    DelayOp,
    DotDimensionNumbersAttr,
    Float8EXMYType,
    PackFormat,
    PackFormatAttr,
    PipelineMode,
    PipelineModeAttr,
    RegionOp,
    TraceOp,
    TraceStartOp,
    TraceStopOp,
    TraceValueOp,
    YieldOp,
)
from xdsl.ir import Block, Region
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import create_ssa_value

def _i64_array(*values: int) -> ArrayAttr[IntegerAttr[IntegerType]]:
    return ArrayAttr([IntegerAttr(v, i64) for v in values])

def test_pipeline_mode_attr_round_trip():
    ctx = Context()
    ctx.load_dialect(TPU)

    attr = PipelineModeAttr(PipelineMode.Synchronous)
    
    output = io.StringIO()
    Printer(stream=output).print_attribute(attr)
    printed = output.getvalue()
    
    parsed = Parser(ctx, printed).parse_attribute()
    assert parsed == attr

def test_pack_format_attr_round_trip():
    ctx = Context()
    ctx.load_dialect(TPU)

    attr = PackFormatAttr(PackFormat.Interleaved)
    
    output = io.StringIO()
    Printer(stream=output).print_attribute(attr)
    printed = output.getvalue()
    
    parsed = Parser(ctx, printed).parse_attribute()
    assert parsed == attr

def test_float8_exmy_type_round_trip():
    ctx = Context()
    ctx.load_dialect(TPU)

    ty = Float8EXMYType(Float32Type())
    
    output = io.StringIO()
    Printer(stream=output).print_attribute(ty)
    printed = output.getvalue()
    
    parsed = Parser(ctx, printed).parse_type()
    assert parsed == ty

def test_dot_dimension_numbers_construction():
    attr = DotDimensionNumbersAttr(
        _i64_array(0),
        _i64_array(1),
        _i64_array(2),
        _i64_array(3),
        _i64_array(4),
        _i64_array(5),
        _i64_array(6),
    )
    assert len(attr.lhs_contracting_dims.data) == 1
    assert len(attr.rhs_batch_dims.data) == 1


def test_dot_dimension_numbers_field_values():
    attr = DotDimensionNumbersAttr(
        _i64_array(0, 1),
        _i64_array(2, 3),
        _i64_array(4),
        _i64_array(5),
        _i64_array(6, 7, 8),
        _i64_array(),
        _i64_array(9),
    )
    vals = lambda arr: [a.value.data for a in arr.data]
    assert vals(attr.lhs_contracting_dims) == [0, 1]
    assert vals(attr.rhs_contracting_dims) == [2, 3]
    assert vals(attr.lhs_non_contracting_dims) == [4]
    assert vals(attr.output_dim_order) == [6, 7, 8]
    assert vals(attr.lhs_batch_dims) == []

def test_yield_op_empty():
    op = YieldOp()
    assert isinstance(op, YieldOp)
    assert len(op.arguments) == 0


def test_yield_op_with_one_value():
    val = create_ssa_value(i32)
    op = YieldOp(val)
    assert len(op.arguments) == 1
    assert op.arguments[0].type == i32


def test_yield_op_with_multiple_values():
    v1 = create_ssa_value(i32)
    v2 = create_ssa_value(f32)
    v3 = create_ssa_value(VectorType(i32, [4]))
    op = YieldOp(v1, v2, v3)
    assert len(op.arguments) == 3

def test_region_op_basic_i32_result():
    val = create_ssa_value(i32)
    region = Region([Block([YieldOp(val)])])
    op = RegionOp(result_types=[i32], region=region)
    op.verify()


def test_region_op_basic_f32_result():
    val = create_ssa_value(f32)
    region = Region([Block([YieldOp(val)])])
    op = RegionOp(result_types=[f32], region=region)
    op.verify()


def test_region_op_basic_vector_result():
    vec_ty = VectorType(i32, [8])
    val = create_ssa_value(vec_ty)
    region = Region([Block([YieldOp(val)])])
    op = RegionOp(result_types=[vec_ty], region=region)
    op.verify()


def test_region_op_rejects_invalid_result_type():
    val = create_ssa_value(i32)
    region = Region([Block([YieldOp(val)])])
    op = RegionOp(
        result_types=[StringAttr("not_a_type").type if False else Float8EXMYType(f32)], region=region)
    with pytest.raises(VerifyException, match="must be a float, int"):
        op.verify()

def test_trace_op_construction():
    val = create_ssa_value(i32)
    region = Region([Block([YieldOp(val)])])
    op = TraceOp("entering region", 3, [i32], region)
    assert op.message.data == "entering region"
    assert op.level.value.data == 3
    op.verify()


def test_trace_op_from_string_and_int():
    val = create_ssa_value(f32)
    region = Region([Block([YieldOp(val)])])
    op = TraceOp("hello", 1, [f32], region)
    assert isinstance(op.message, StringAttr)
    assert isinstance(op.level, IntegerAttr)


def test_trace_op_from_attributes():
    val = create_ssa_value(f32)
    region = Region([Block([YieldOp(val)])])
    op = TraceOp(StringAttr("hello"), IntegerAttr(5, i32), [f32], region)
    assert op.message.data == "hello"
    assert op.level.value.data == 5


def test_trace_start_op_construction():
    op = TraceStartOp("start", 1)
    assert op.message.data == "start"
    assert op.level.value.data == 1


def test_trace_stop_op_construction():
    op = TraceStopOp()
    assert isinstance(op, TraceStopOp)


def test_trace_value_op_with_i32():
    val = create_ssa_value(i32)
    op = TraceValueOp(val, "my_int")
    assert op.value.type == i32
    assert op.label.data == "my_int"


def test_trace_value_op_with_f32():
    val = create_ssa_value(f32)
    op = TraceValueOp(val, "my_float")
    assert op.value.type == f32

def test_delay_op_construction():
    nanos = create_ssa_value(i32)
    op = DelayOp(nanos)
    assert op.nanos.type == i32

def test_round_trip_trace_start_stop():
    from xdsl.dialects.builtin import Builtin

    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(TPU)

    text = """
    builtin.module {
      "tpu.trace_start"() {message = "hello", level = 1 : i32} : () -> ()
      "tpu.trace_stop"() : () -> ()
    }
    """
    module = Parser(ctx, text).parse_module()
    module.verify()


def test_round_trip_delay():
    from xdsl.dialects.arith import Arith
    from xdsl.dialects.builtin import Builtin

    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(TPU)
    ctx.load_dialect(Arith)

    text = """
    builtin.module {
      %nanos = arith.constant 100 : i32
      tpu.delay %nanos
    }
    """
    module = Parser(ctx, text).parse_module()
    module.verify()



def test_round_trip_trace_value():
    from xdsl.dialects.arith import Arith
    from xdsl.dialects.builtin import Builtin

    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(TPU)
    ctx.load_dialect(Arith)

    text = """
    builtin.module {
      %v = arith.constant 0 : i32
      tpu.trace_value %v, "my_label" : i32
    }
    """
    module = Parser(ctx, text).parse_module()
    module.verify()
