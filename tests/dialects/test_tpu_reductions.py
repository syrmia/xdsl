import pytest

from xdsl.dialects.builtin import (
    VectorType,
    f32,
    i1,
    i32,
)
from xdsl.dialects.tpu_reductions import (
    AllReduceOp,
    ReduceIndexOp,
    ReductionKind,
    ScanOp,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import create_ssa_value


def test_all_reduce_sum_basic():
    src = create_ssa_value(VectorType(f32, [8, 128]))
    op = AllReduceOp(
        src, dim=0, kind=ReductionKind.Sum, result_type=VectorType(f32, [8, 128])
    )
    op.verify()


def test_all_reduce_max_basic():
    src = create_ssa_value(VectorType(f32, [4, 8]))
    op = AllReduceOp(
        src, dim=1, kind=ReductionKind.Max, result_type=VectorType(f32, [4, 8])
    )
    op.verify()


def test_all_reduce_min_basic():
    src = create_ssa_value(VectorType(f32, [4, 8]))
    op = AllReduceOp(
        src, dim=1, kind=ReductionKind.Min, result_type=VectorType(f32, [4, 8])
    )
    op.verify()


def test_all_reduce_sum_rejects_shape_mismatch():
    src = create_ssa_value(VectorType(f32, [8, 128]))
    op = AllReduceOp(
        src, dim=0, kind=ReductionKind.Sum, result_type=VectorType(f32, [4, 128])
    )
    with pytest.raises(VerifyException, match="same input and output type"):
        op.verify()


def test_all_reduce_arg_max_basic():
    src = create_ssa_value(VectorType(f32, [8, 128]))
    op = AllReduceOp(
        src, dim=0, kind=ReductionKind.ArgMax, result_type=VectorType(i32, [8, 128])
    )
    op.verify()


def test_all_reduce_arg_max_rejects_non_f32():
    src = create_ssa_value(VectorType(i32, [8, 128]))
    op = AllReduceOp(
        src, dim=0, kind=ReductionKind.ArgMax, result_type=VectorType(i32, [8, 128])
    )
    with pytest.raises(VerifyException, match="f32 input"):
        op.verify()


def test_all_reduce_arg_max_rejects_shape_change():
    src = create_ssa_value(VectorType(f32, [8, 128]))
    op = AllReduceOp(
        src, dim=0, kind=ReductionKind.ArgMax, result_type=VectorType(i32, [4, 128])
    )
    with pytest.raises(VerifyException, match="same input and output shape"):
        op.verify()


def test_all_reduce_i1_sum():
    src = create_ssa_value(VectorType(i1, [128]))
    op = AllReduceOp(
        src, dim=0, kind=ReductionKind.Sum, result_type=VectorType(i32, [128])
    )
    op.verify()


def test_all_reduce_i1_find_first_set():
    src = create_ssa_value(VectorType(i1, [128]))
    op = AllReduceOp(
        src, dim=0, kind=ReductionKind.FindFirstSet, result_type=VectorType(i32, [128])
    )
    op.verify()


def test_all_reduce_i1_rejects_max():
    src = create_ssa_value(VectorType(i1, [128]))
    op = AllReduceOp(
        src, dim=0, kind=ReductionKind.Max, result_type=VectorType(i32, [128])
    )
    with pytest.raises(VerifyException, match="sum and find_first_set"):
        op.verify()


def test_all_reduce_i1_rejects_non_i32_output():
    src = create_ssa_value(VectorType(i1, [128]))
    op = AllReduceOp(
        src, dim=0, kind=ReductionKind.Sum, result_type=VectorType(i1, [128])
    )
    with pytest.raises(VerifyException, match="i32 output"):
        op.verify()


def test_all_reduce_find_first_set_rejects_non_i1():
    src = create_ssa_value(VectorType(f32, [128]))
    op = AllReduceOp(
        src, dim=0, kind=ReductionKind.FindFirstSet, result_type=VectorType(f32, [128])
    )
    with pytest.raises(VerifyException, match="i1 input is supported"):
        op.verify()


def test_reduce_index_arg_max_basic():
    src = create_ssa_value(VectorType(f32, [8, 128]))
    op = ReduceIndexOp(
        src, axis=0, kind=ReductionKind.ArgMax, result_type=VectorType(i32, [128])
    )
    op.verify()


def test_reduce_index_arg_min_basic():
    src = create_ssa_value(VectorType(f32, [8, 128]))
    op = ReduceIndexOp(
        src, axis=1, kind=ReductionKind.ArgMin, result_type=VectorType(i32, [8])
    )
    op.verify()


def test_reduce_index_rejects_sum():
    src = create_ssa_value(VectorType(f32, [8, 128]))
    op = ReduceIndexOp(
        src, axis=0, kind=ReductionKind.Sum, result_type=VectorType(i32, [128])
    )
    with pytest.raises(VerifyException, match="arg_max or arg_min"):
        op.verify()


def test_reduce_index_rejects_non_f32_input():
    src = create_ssa_value(VectorType(i32, [8, 128]))
    op = ReduceIndexOp(
        src, axis=0, kind=ReductionKind.ArgMax, result_type=VectorType(i32, [128])
    )
    with pytest.raises(VerifyException, match="f32 input"):
        op.verify()


def test_reduce_index_rejects_out_of_range_axis():
    src = create_ssa_value(VectorType(f32, [8, 128]))
    op = ReduceIndexOp(
        src, axis=5, kind=ReductionKind.ArgMax, result_type=VectorType(i32, [128])
    )
    with pytest.raises(VerifyException, match="Axis must be in"):
        op.verify()


def test_reduce_index_rejects_rank_1():
    src = create_ssa_value(VectorType(f32, [128]))
    op = ReduceIndexOp(
        src, axis=0, kind=ReductionKind.ArgMax, result_type=VectorType(i32, [])
    )
    with pytest.raises(VerifyException):
        op.verify()


def test_reduce_index_rejects_wrong_output_rank():
    src = create_ssa_value(VectorType(f32, [8, 128]))
    op = ReduceIndexOp(
        src, axis=0, kind=ReductionKind.ArgMax, result_type=VectorType(i32, [8, 128])
    )
    with pytest.raises(VerifyException, match="one less than input rank"):
        op.verify()


def test_reduce_index_rejects_wrong_non_reduction_shape():
    src = create_ssa_value(VectorType(f32, [8, 128]))
    op = ReduceIndexOp(
        src, axis=0, kind=ReductionKind.ArgMax, result_type=VectorType(i32, [64])
    )
    with pytest.raises(VerifyException, match="non-reduction dimensions"):
        op.verify()


def test_scan_op_f32_sum_basic():
    src = create_ssa_value(VectorType(f32, [128]))
    op = ScanOp(src, kind=ReductionKind.Sum, result_type=VectorType(f32, [128]))
    op.verify()


def test_scan_op_f32_max():
    src = create_ssa_value(VectorType(f32, [8, 128]))
    op = ScanOp(src, kind=ReductionKind.Max, result_type=VectorType(f32, [8, 128]))
    op.verify()


def test_scan_op_f32_with_mask():
    src = create_ssa_value(VectorType(f32, [8, 128]))
    mask = create_ssa_value(VectorType(i1, [128]))
    op = ScanOp(
        src, kind=ReductionKind.Sum, result_type=VectorType(f32, [8, 128]), mask=mask
    )
    op.verify()


def test_scan_op_i1_sum():
    src = create_ssa_value(VectorType(i1, [128]))
    op = ScanOp(src, kind=ReductionKind.Sum, result_type=VectorType(i32, [128]))
    op.verify()


def test_scan_op_rejects_i1_non_sum():
    src = create_ssa_value(VectorType(i1, [128]))
    op = ScanOp(src, kind=ReductionKind.Max, result_type=VectorType(i32, [128]))
    with pytest.raises(VerifyException, match="sum reduction is supported for i1"):
        op.verify()


def test_scan_op_rejects_i1_non_i32_output():
    src = create_ssa_value(VectorType(i1, [128]))
    op = ScanOp(src, kind=ReductionKind.Sum, result_type=VectorType(i1, [128]))
    with pytest.raises(VerifyException, match="i32 vector for i1"):
        op.verify()


def test_scan_op_rejects_element_type_mismatch():
    src = create_ssa_value(VectorType(f32, [128]))
    op = ScanOp(src, kind=ReductionKind.Sum, result_type=VectorType(i32, [128]))
    with pytest.raises(VerifyException, match="element type mismatch"):
        op.verify()


def test_scan_op_rejects_shape_mismatch():
    src = create_ssa_value(VectorType(f32, [128]))
    op = ScanOp(src, kind=ReductionKind.Sum, result_type=VectorType(f32, [64]))
    with pytest.raises(VerifyException, match="shape mismatch"):
        op.verify()


def test_scan_op_rejects_rank_3():
    src = create_ssa_value(VectorType(f32, [2, 4, 8]))
    op = ScanOp(src, kind=ReductionKind.Sum, result_type=VectorType(f32, [2, 4, 8]))
    with pytest.raises(VerifyException, match="rank 1 or 2"):
        op.verify()


def test_scan_op_rejects_i1_with_mask():
    src = create_ssa_value(VectorType(i1, [128]))
    mask = create_ssa_value(VectorType(i1, [128]))
    op = ScanOp(
        src, kind=ReductionKind.Sum, result_type=VectorType(i32, [128]), mask=mask
    )
    with pytest.raises(VerifyException, match="Mask is not supported for i1"):
        op.verify()


def test_scan_op_rejects_wrong_mask_rank():
    src = create_ssa_value(VectorType(f32, [8, 128]))
    mask = create_ssa_value(VectorType(i1, [8, 128]))
    op = ScanOp(
        src, kind=ReductionKind.Sum, result_type=VectorType(f32, [8, 128]), mask=mask
    )
    with pytest.raises(VerifyException, match="rank 1 vector"):
        op.verify()


def test_scan_op_rejects_wrong_mask_length():
    src = create_ssa_value(VectorType(f32, [8, 128]))
    mask = create_ssa_value(VectorType(i1, [64]))
    op = ScanOp(
        src, kind=ReductionKind.Sum, result_type=VectorType(f32, [8, 128]), mask=mask
    )
    with pytest.raises(VerifyException, match="Mask and input mismatch"):
        op.verify()
