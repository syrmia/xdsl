import io

from xdsl.context import Context
from xdsl.dialects.builtin import (
    Float16Type,
    VectorType,
    f32,
    i1,
    i32,
)
from xdsl.dialects.tpu import TPU
from xdsl.dialects.tpu_conversions import (
    ExtFOp,
    FPToSIOp,
    FPToUIOp,
    ReciprocalOp,
    RoundingMode,
    RoundingModeAttr,
    SIToFPOp,
    TruncFOp,
    UIToFPOp,
    WeirdOp,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.test_value import create_ssa_value


def test_rounding_mode_enum_members():
    assert RoundingMode.Towards_Zero.value == "towards_zero"
    assert RoundingMode.To_Nearest_Even.value == "to_nearest_even"


def test_rounding_mode_attr_towards_zero():
    attr = RoundingModeAttr(RoundingMode.Towards_Zero)
    assert attr.data == RoundingMode.Towards_Zero


def test_rounding_mode_attr_to_nearest_even():
    attr = RoundingModeAttr(RoundingMode.To_Nearest_Even)
    assert attr.data == RoundingMode.To_Nearest_Even


def test_rounding_mode_attr_round_trip():
    ctx = Context()
    ctx.load_dialect(TPU)

    attr = RoundingModeAttr(RoundingMode.Towards_Zero)

    output = io.StringIO()
    Printer(stream=output).print_attribute(attr)
    printed = output.getvalue()

    parsed = Parser(ctx, printed).parse_attribute()
    assert parsed == attr


def test_fp_to_si_op_basic():
    val = create_ssa_value(f32)
    op = FPToSIOp(val, target_type=i32, rounding_mode=RoundingMode.To_Nearest_Even)
    op.verify()
    assert op.input.type == f32
    assert op.output.type == i32


def test_fp_to_si_op_vector():
    val = create_ssa_value(VectorType(f32, [8, 128]))
    op = FPToSIOp(
        val,
        target_type=VectorType(i32, [8, 128]),
        rounding_mode=RoundingMode.Towards_Zero,
    )
    op.verify()


def test_fp_to_si_op_accepts_attr_for_rounding_mode():
    val = create_ssa_value(f32)
    rm = RoundingModeAttr(RoundingMode.To_Nearest_Even)
    op = FPToSIOp(val, target_type=i32, rounding_mode=rm)
    op.verify()
    assert op.rounding_mode == rm


def test_fp_to_ui_op_basic():
    val = create_ssa_value(f32)
    op = FPToUIOp(val, target_type=i32, rounding_mode=RoundingMode.To_Nearest_Even)
    op.verify()


def test_fp_to_ui_op_vector():
    val = create_ssa_value(VectorType(f32, [8]))
    op = FPToUIOp(
        val, target_type=VectorType(i32, [8]), rounding_mode=RoundingMode.Towards_Zero
    )
    op.verify()


def test_si_to_fp_op_basic():
    val = create_ssa_value(i32)
    op = SIToFPOp(val, target_type=f32, rounding_mode=RoundingMode.To_Nearest_Even)
    op.verify()


def test_si_to_fp_op_vector():
    val = create_ssa_value(VectorType(i32, [8, 128]))
    op = SIToFPOp(
        val,
        target_type=VectorType(f32, [8, 128]),
        rounding_mode=RoundingMode.To_Nearest_Even,
    )
    op.verify()


def test_ui_to_fp_op_basic():
    val = create_ssa_value(i32)
    op = UIToFPOp(val, target_type=f32, rounding_mode=RoundingMode.To_Nearest_Even)
    op.verify()


def test_ui_to_fp_op_vector():
    val = create_ssa_value(VectorType(i32, [16]))
    op = UIToFPOp(
        val, target_type=VectorType(f32, [16]), rounding_mode=RoundingMode.Towards_Zero
    )
    op.verify()


def test_ext_f_op_basic():
    val = create_ssa_value(Float16Type())
    op = ExtFOp(val, target_type=f32)
    op.verify()


def test_ext_f_op_vector():
    val = create_ssa_value(VectorType(Float16Type(), [8, 128]))
    op = ExtFOp(val, target_type=VectorType(f32, [8, 128]))
    op.verify()


def test_trunc_f_op_basic():
    val = create_ssa_value(f32)
    op = TruncFOp(
        val, target_type=Float16Type(), rounding_mode=RoundingMode.To_Nearest_Even
    )
    op.verify()


def test_trunc_f_op_vector():
    val = create_ssa_value(VectorType(f32, [8, 128]))
    op = TruncFOp(
        val,
        target_type=VectorType(Float16Type(), [8, 128]),
        rounding_mode=RoundingMode.Towards_Zero,
    )
    op.verify()


def test_reciprocal_op_basic_f32():
    val = create_ssa_value(VectorType(f32, [8, 128]))
    op = ReciprocalOp(val, target_type=VectorType(f32, [8, 128]))
    op.verify()


def test_reciprocal_op_defaults():
    val = create_ssa_value(VectorType(f32, [8]))
    op = ReciprocalOp(val, target_type=VectorType(f32, [8]))
    assert not bool(op.approx.value.data)
    assert bool(op.full_range.value.data)


def test_reciprocal_op_custom_approx():
    val = create_ssa_value(VectorType(f32, [8]))
    op = ReciprocalOp(val, target_type=VectorType(f32, [8]), approx=True)
    assert bool(op.approx.value.data)


def test_reciprocal_op_custom_full_range_false():
    val = create_ssa_value(VectorType(f32, [8]))
    op = ReciprocalOp(val, target_type=VectorType(f32, [8]), full_range=False)
    assert op.full_range.value.data == 0


def test_weird_op_scalar():
    val = create_ssa_value(f32)
    op = WeirdOp(val, target_type=i1)
    op.verify()


def test_weird_op_vector():
    val = create_ssa_value(VectorType(f32, [8, 128]))
    op = WeirdOp(val, target_type=VectorType(i1, [8, 128]))
    op.verify()


def test_dialect_registers_conversion_attrs():
    registered = set(TPU.attributes)
    assert RoundingModeAttr in registered
