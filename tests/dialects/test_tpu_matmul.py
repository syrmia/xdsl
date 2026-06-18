import pytest

from xdsl.dialects.builtin import (
    ArrayAttr,
    IntegerAttr,
    VectorType,
    bf16,
    f32,
    i64,
)
from xdsl.dialects.tpu_matmul import (
    ContractPrecision,
    DotDimensionNumbersAttr,
    MatmulAccLhsOp,
    MatmulOp,
    MatmulPopOp,
    MatmulPushRhsOp,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import create_ssa_value


def _i64_arr(values: list[int]) -> ArrayAttr:
    return ArrayAttr([IntegerAttr(v, i64) for v in values])


def _ddn(
    lhs_contracting: list[int],
    rhs_contracting: list[int],
    lhs_non_contracting: list[int],
    rhs_non_contracting: list[int],
    output_dim_order: list[int],
    lhs_batch: list[int] | None = None,
    rhs_batch: list[int] | None = None,
) -> DotDimensionNumbersAttr:
    return DotDimensionNumbersAttr(
        _i64_arr(lhs_contracting),
        _i64_arr(rhs_contracting),
        _i64_arr(lhs_non_contracting),
        _i64_arr(rhs_non_contracting),
        _i64_arr(output_dim_order),
        _i64_arr(lhs_batch or []),
        _i64_arr(rhs_batch or []),
    )


def test_matmul_basic_no_dim_numbers():
    lhs = create_ssa_value(VectorType(bf16, [128, 256]))
    rhs = create_ssa_value(VectorType(bf16, [256, 128]))
    acc = create_ssa_value(VectorType(f32, [128, 128]))
    op = MatmulOp(lhs, rhs, acc, result_type=VectorType(f32, [128, 128]))
    op.verify()


def test_matmul_with_dim_numbers_simple_2d():
    lhs = create_ssa_value(VectorType(bf16, [8, 16]))
    rhs = create_ssa_value(VectorType(bf16, [16, 32]))
    acc = create_ssa_value(VectorType(f32, [8, 32]))
    ddn = _ddn(
        lhs_contracting=[1],
        rhs_contracting=[0],
        lhs_non_contracting=[0],
        rhs_non_contracting=[1],
        output_dim_order=[0, 0, 1, 1],
    )
    op = MatmulOp(
        lhs, rhs, acc, result_type=VectorType(f32, [8, 32]), dimension_numbers=ddn
    )
    op.verify()


def test_matmul_with_batch_dim():
    lhs = create_ssa_value(VectorType(bf16, [4, 8, 16]))
    rhs = create_ssa_value(VectorType(bf16, [4, 16, 32]))
    acc = create_ssa_value(VectorType(f32, [4, 8, 32]))
    ddn = _ddn(
        lhs_contracting=[2],
        rhs_contracting=[1],
        lhs_non_contracting=[1],
        rhs_non_contracting=[2],
        output_dim_order=[0, 0, 0, 1, 1, 2],
        lhs_batch=[0],
        rhs_batch=[0],
    )
    op = MatmulOp(
        lhs, rhs, acc, result_type=VectorType(f32, [4, 8, 32]), dimension_numbers=ddn
    )
    op.verify()


def test_matmul_with_precision():
    lhs = create_ssa_value(VectorType(bf16, [8, 16]))
    rhs = create_ssa_value(VectorType(bf16, [16, 32]))
    acc = create_ssa_value(VectorType(f32, [8, 32]))
    op = MatmulOp(
        lhs,
        rhs,
        acc,
        result_type=VectorType(f32, [8, 32]),
        precision=ContractPrecision.Bf16,
    )
    op.verify()
    assert op.precision is not None


def test_matmul_rejects_acc_result_type_mismatch():
    lhs = create_ssa_value(VectorType(bf16, [8, 16]))
    rhs = create_ssa_value(VectorType(bf16, [16, 32]))
    acc = create_ssa_value(VectorType(f32, [8, 32]))
    op = MatmulOp(lhs, rhs, acc, result_type=VectorType(f32, [16, 32]))
    with pytest.raises(VerifyException, match="acc and result have different types"):
        op.verify()


def test_matmul_rejects_non_32_bit_acc():
    lhs = create_ssa_value(VectorType(bf16, [8, 16]))
    rhs = create_ssa_value(VectorType(bf16, [16, 32]))
    acc = create_ssa_value(VectorType(bf16, [8, 32]))
    op = MatmulOp(lhs, rhs, acc, result_type=VectorType(bf16, [8, 32]))
    with pytest.raises(VerifyException, match="32-bit"):
        op.verify()


def test_matmul_rejects_transpose_lhs():
    lhs = create_ssa_value(VectorType(bf16, [8, 16]))
    rhs = create_ssa_value(VectorType(bf16, [16, 32]))
    acc = create_ssa_value(VectorType(f32, [8, 32]))
    op = MatmulOp(
        lhs, rhs, acc, result_type=VectorType(f32, [8, 32]), transpose_lhs=True
    )
    with pytest.raises(VerifyException, match="Lhs transpose"):
        op.verify()


def test_matmul_rejects_multi_contracting_lhs():
    lhs = create_ssa_value(VectorType(bf16, [8, 16, 4]))
    rhs = create_ssa_value(VectorType(bf16, [16, 32]))
    acc = create_ssa_value(VectorType(f32, [8, 32]))
    ddn = _ddn(
        lhs_contracting=[1, 2],
        rhs_contracting=[0],
        lhs_non_contracting=[0],
        rhs_non_contracting=[1],
        output_dim_order=[0, 0, 1, 1],
    )
    op = MatmulOp(
        lhs, rhs, acc, result_type=VectorType(f32, [8, 32]), dimension_numbers=ddn
    )
    with pytest.raises(VerifyException, match="lhs contracting dims must be of size 1"):
        op.verify()


def test_matmul_rejects_unsorted_lhs_non_contracting():
    lhs = create_ssa_value(VectorType(bf16, [8, 4, 16]))
    rhs = create_ssa_value(VectorType(bf16, [16, 32]))
    acc = create_ssa_value(VectorType(f32, [4, 8, 32]))
    ddn = _ddn(
        lhs_contracting=[2],
        rhs_contracting=[0],
        lhs_non_contracting=[1, 0],
        rhs_non_contracting=[1],
        output_dim_order=[0, 1, 0, 0, 1, 1],
    )
    op = MatmulOp(
        lhs, rhs, acc, result_type=VectorType(f32, [4, 8, 32]), dimension_numbers=ddn
    )
    with pytest.raises(
        VerifyException, match="lhs non contracting dims must be sorted"
    ):
        op.verify()


def test_matmul_rejects_dim_total_mismatch():
    lhs = create_ssa_value(VectorType(bf16, [8, 16, 4]))
    rhs = create_ssa_value(VectorType(bf16, [16, 32]))
    acc = create_ssa_value(VectorType(f32, [8, 4, 32]))
    ddn = _ddn(
        lhs_contracting=[1],
        rhs_contracting=[0],
        lhs_non_contracting=[0],
        rhs_non_contracting=[1],
        output_dim_order=[0, 0, 1, 1],
    )
    op = MatmulOp(
        lhs, rhs, acc, result_type=VectorType(f32, [8, 4, 32]), dimension_numbers=ddn
    )
    with pytest.raises(
        VerifyException, match="contracting \\+ non contracting \\+ batch"
    ):
        op.verify()


def test_matmul_rejects_contracting_size_mismatch():
    lhs = create_ssa_value(VectorType(bf16, [8, 16]))
    rhs = create_ssa_value(VectorType(bf16, [32, 64]))
    acc = create_ssa_value(VectorType(f32, [8, 64]))
    ddn = _ddn(
        lhs_contracting=[1],
        rhs_contracting=[0],
        lhs_non_contracting=[0],
        rhs_non_contracting=[1],
        output_dim_order=[0, 0, 1, 1],
    )
    op = MatmulOp(
        lhs, rhs, acc, result_type=VectorType(f32, [8, 64]), dimension_numbers=ddn
    )
    with pytest.raises(
        VerifyException, match="contracting dims must be of the same size"
    ):
        op.verify()


def test_matmul_rejects_batch_count_mismatch():
    lhs = create_ssa_value(VectorType(bf16, [4, 8, 16]))
    rhs = create_ssa_value(VectorType(bf16, [16, 32]))
    acc = create_ssa_value(VectorType(f32, [4, 8, 32]))
    ddn = _ddn(
        lhs_contracting=[2],
        rhs_contracting=[0],
        lhs_non_contracting=[1],
        rhs_non_contracting=[1],
        output_dim_order=[0, 0, 0, 1, 1, 1],
        lhs_batch=[0],
        rhs_batch=[],
    )
    op = MatmulOp(
        lhs, rhs, acc, result_type=VectorType(f32, [4, 8, 32]), dimension_numbers=ddn
    )
    with pytest.raises(VerifyException, match="same number of batch dims"):
        op.verify()


def test_matmul_rejects_more_than_one_batch_dim():
    lhs = create_ssa_value(VectorType(bf16, [2, 4, 8, 16]))
    rhs = create_ssa_value(VectorType(bf16, [2, 4, 16, 32]))
    acc = create_ssa_value(VectorType(f32, [2, 4, 8, 32]))
    ddn = _ddn(
        lhs_contracting=[3],
        rhs_contracting=[2],
        lhs_non_contracting=[2],
        rhs_non_contracting=[3],
        output_dim_order=[0, 0, 0, 1, 0, 2, 1, 3],
        lhs_batch=[0, 1],
        rhs_batch=[0, 1],
    )
    op = MatmulOp(
        lhs, rhs, acc, result_type=VectorType(f32, [2, 4, 8, 32]), dimension_numbers=ddn
    )
    with pytest.raises(VerifyException, match="Up to 1 batch dim"):
        op.verify()


def test_matmul_rejects_repeated_dim_classification():
    lhs = create_ssa_value(VectorType(bf16, [8, 16, 4]))
    rhs = create_ssa_value(VectorType(bf16, [16, 32]))
    acc = create_ssa_value(VectorType(f32, [8, 32]))
    ddn = _ddn(
        lhs_contracting=[1],
        rhs_contracting=[0],
        lhs_non_contracting=[0, 0],
        rhs_non_contracting=[1],
        output_dim_order=[0, 0, 0, 0, 1, 1],
    )
    op = MatmulOp(
        lhs, rhs, acc, result_type=VectorType(f32, [8, 32]), dimension_numbers=ddn
    )
    with pytest.raises(VerifyException, match="repeats in dimension numbers"):
        op.verify()


def test_matmul_rejects_bad_output_dim_order_odd_length():
    lhs = create_ssa_value(VectorType(bf16, [8, 16]))
    rhs = create_ssa_value(VectorType(bf16, [16, 32]))
    acc = create_ssa_value(VectorType(f32, [8, 32]))
    ddn = _ddn(
        lhs_contracting=[1],
        rhs_contracting=[0],
        lhs_non_contracting=[0],
        rhs_non_contracting=[1],
        output_dim_order=[0, 0, 1],
    )
    op = MatmulOp(
        lhs, rhs, acc, result_type=VectorType(f32, [8, 32]), dimension_numbers=ddn
    )
    with pytest.raises(VerifyException, match="even number of elements"):
        op.verify()


def test_matmul_rejects_bad_output_dim_order_shape():
    lhs = create_ssa_value(VectorType(bf16, [8, 16]))
    rhs = create_ssa_value(VectorType(bf16, [16, 32]))
    acc = create_ssa_value(VectorType(f32, [8, 32]))
    ddn = _ddn(
        lhs_contracting=[1],
        rhs_contracting=[0],
        lhs_non_contracting=[0],
        rhs_non_contracting=[1],
        output_dim_order=[1, 1, 0, 0],
    )
    op = MatmulOp(
        lhs, rhs, acc, result_type=VectorType(f32, [8, 32]), dimension_numbers=ddn
    )
    with pytest.raises(VerifyException, match="output dim order must be in the form"):
        op.verify()


def test_matmul_push_rhs_basic():
    rhs = create_ssa_value(VectorType(bf16, [16, 32]))
    op = MatmulPushRhsOp(rhs, mxu_index=0)
    op.verify()
    assert op.mxu_index.value.data == 0
    assert op.staging_register.value.data == 0
    assert op.transpose.value.data == 0


def test_matmul_push_rhs_with_staging():
    rhs = create_ssa_value(VectorType(bf16, [16, 32]))
    op = MatmulPushRhsOp(rhs, mxu_index=1, staging_register=3, transpose=True)
    op.verify()
    assert op.staging_register.value.data == 3
    assert op.transpose.value.data == -1


def test_matmul_acc_lhs_basic():
    lhs = create_ssa_value(VectorType(bf16, [8, 16]))
    op = MatmulAccLhsOp(lhs, acc=0, mxu_index=0)
    op.verify()
    assert op.acc.value.data == 0
    assert op.load_staged_rhs is None


def test_matmul_acc_lhs_with_load_staged():
    lhs = create_ssa_value(VectorType(bf16, [8, 16]))
    op = MatmulAccLhsOp(lhs, acc=2, mxu_index=1, load_staged_rhs=3)
    op.verify()
    assert op.load_staged_rhs is not None
    assert op.load_staged_rhs.value.data == 3


def test_matmul_pop_basic():
    op = MatmulPopOp(VectorType(f32, [8, 32]), acc=0, mxu_index=0)
    op.verify()
    assert op.acc.value.data == 0
    assert op.mxu_index.value.data == 0


def test_matmul_pop_with_attrs():
    op = MatmulPopOp(VectorType(f32, [8, 32]), acc=2, mxu_index=1)
    op.verify()
