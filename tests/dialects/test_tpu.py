

import pytest
from typing import cast

from xdsl.dialects.func import FuncOp
from xdsl.dialects.builtin import AnyFloat, ArrayAttr, IntegerAttr, IntegerType, StringAttr, i32, i64, f32, bf16, f64
from xdsl.dialects.tpu import TPU, CoreType, CoreTypeAttr, DotDimensionNumbersAttr, Float8EXMYType


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
 
 
def test_dialect_has_no_ops_yet():
    assert list(TPU.operations) == []
 
 
def test_dialect_registers_attributes_and_types():
    registered = set(TPU.attributes)
    assert CoreTypeAttr in registered
    assert DotDimensionNumbersAttr in registered
    assert Float8EXMYType in registered