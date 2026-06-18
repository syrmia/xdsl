import pytest

from xdsl.dialects.builtin import (
    DenseArrayBase,
    VectorType,
    f32,
    i1,
    i32,
)
from xdsl.dialects.tpu_shape import (
    BitcastOp,
    BitcastVregOp,
    BroadcastInSublanesOp,
    ConcatenateOp,
    DynamicGatherOp,
    DynamicRotateOp,
    GatherOp,
    IotaOp,
    MaskCastOp,
    RepeatOp,
    ReshapeOp,
    RollVectorsOp,
    RotateOp,
    ScanCountOp,
    TransposeOp,
    UnrollVectorsOp,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import create_ssa_value


def test_rotate_op_basic():
    val = create_ssa_value(VectorType(f32, [8, 128]))
    op = RotateOp(val, amount=4, dimension=1)
    op.verify()


def test_rotate_op_with_stride():
    val = create_ssa_value(VectorType(f32, [8, 128]))
    op = RotateOp(val, amount=4, dimension=1, stride=2, stride_dimension=0)
    op.verify()


def test_dynamic_rotate_op_basic():
    val = create_ssa_value(VectorType(f32, [8, 128]))
    amount = create_ssa_value(i32)
    op = DynamicRotateOp(val, amount=amount, dimension=0)
    op.verify()


def test_dynamic_rotate_op_with_stride():
    val = create_ssa_value(VectorType(f32, [8, 128]))
    amount = create_ssa_value(i32)
    op = DynamicRotateOp(
        val,
        amount=amount,
        dimension=0,
        stride=2,
        stride_dimension=1,
    )
    op.verify()


def test_iota_op_single_dimension():
    op = IotaOp(dimensions=[0], result_type=VectorType(i32, [8, 128]))
    op.verify()


def test_iota_op_multiple_dimensions():
    op = IotaOp(dimensions=[0, 1], result_type=VectorType(i32, [8, 128]))
    op.verify()


def test_iota_op_accepts_dense_array():
    dims = DenseArrayBase.from_list(i32, [0])
    op = IotaOp(dimensions=dims, result_type=VectorType(i32, [8]))
    op.verify()


def test_reshape_op_basic():
    val = create_ssa_value(VectorType(f32, [8, 128]))
    op = ReshapeOp(val, result_type=VectorType(f32, [4, 256]))
    op.verify()


def test_reshape_op_rank_change():
    val = create_ssa_value(VectorType(f32, [8, 128]))
    op = ReshapeOp(val, result_type=VectorType(f32, [1024]))
    op.verify()


def test_repeat_op_basic():
    val = create_ssa_value(VectorType(f32, [8, 128]))
    op = RepeatOp(val, dimension=0, times=4, result_type=VectorType(f32, [32, 128]))
    op.verify()


def test_repeat_op_along_second_dim():
    val = create_ssa_value(VectorType(f32, [8, 128]))
    op = RepeatOp(val, dimension=1, times=2, result_type=VectorType(f32, [8, 256]))
    op.verify()


def test_bitcast_op_basic():
    val = create_ssa_value(VectorType(f32, [8, 128]))
    op = BitcastOp(val, result_type=VectorType(i32, [8, 128]))
    op.verify()


def test_bitcast_vreg_op_basic():
    val = create_ssa_value(VectorType(f32, [8, 128]))
    op = BitcastVregOp(val, result_type=VectorType(i32, [8, 128]))
    assert isinstance(op, BitcastVregOp)


def test_mask_cast_op_basic():
    val = create_ssa_value(VectorType(i1, [8, 128]))
    op = MaskCastOp(val, result_type=VectorType(i1, [8, 128, 2]))
    op.verify()


def test_scan_count_op_basic():
    in_mask = create_ssa_value(VectorType(i1, [8]))
    values = create_ssa_value(VectorType(f32, [8]))
    op = ScanCountOp(in_mask, values)
    assert len(op.results) == 2


def test_scan_count_op_with_explicit_types():
    in_mask = create_ssa_value(VectorType(i1, [8]))
    values = create_ssa_value(VectorType(f32, [8]))
    op = ScanCountOp(
        in_mask,
        values,
        out_mask_type=VectorType(i1, [8]),
        counts_type=VectorType(i32, [8]),
    )
    assert len(op.results) == 2


def test_broadcast_in_sublanes_op_basic():
    src = create_ssa_value(VectorType(f32, [8, 128]))
    op = BroadcastInSublanesOp(
        src,
        lane=0,
        result_type=VectorType(f32, [8, 128]),
    )
    assert op.lane.value.data == 0


def test_broadcast_in_sublanes_op_with_nonzero_lane():
    src = create_ssa_value(VectorType(f32, [8, 128]))
    op = BroadcastInSublanesOp(
        src,
        lane=5,
        result_type=VectorType(f32, [8, 128]),
    )
    assert op.lane.value.data == 5


def test_gather_op_basic():
    src = create_ssa_value(VectorType(f32, [16, 128]))
    op = GatherOp(
        src,
        indices=[0, 2, 4, 6],
        dimension=0,
        result_type=VectorType(f32, [4, 128]),
    )
    op.verify()


def test_gather_op_accepts_dense_array():
    src = create_ssa_value(VectorType(f32, [16, 128]))
    idx = DenseArrayBase.from_list(i32, [0, 2])
    op = GatherOp(
        src,
        indices=idx,
        dimension=0,
        result_type=VectorType(f32, [2, 128]),
    )
    op.verify()


def test_dynamic_gather_op_basic():
    src = create_ssa_value(VectorType(f32, [16, 128]))
    indices = create_ssa_value(VectorType(i32, [16, 128]))
    op = DynamicGatherOp(
        src,
        indices,
        dimensions=[0],
        result_type=VectorType(f32, [16, 128]),
    )
    op.verify()


def test_dynamic_gather_op_infers_result_type():
    src = create_ssa_value(VectorType(f32, [16, 128]))
    indices = create_ssa_value(VectorType(i32, [16, 128]))
    op = DynamicGatherOp(
        src,
        indices,
        dimensions=[0],
    )
    result_ty = op.results[0].type
    assert isinstance(result_ty, VectorType)


def test_concatenate_op_basic():
    src1 = create_ssa_value(VectorType(f32, [8, 128]))
    src2 = create_ssa_value(VectorType(f32, [8, 128]))
    op = ConcatenateOp(
        [src1, src2], dimension=0, result_type=VectorType(f32, [16, 128])
    )
    op.verify()


def test_concatenate_op_three_sources():
    src1 = create_ssa_value(VectorType(f32, [8, 64]))
    src2 = create_ssa_value(VectorType(f32, [8, 64]))
    src3 = create_ssa_value(VectorType(f32, [8, 64]))
    op = ConcatenateOp(
        [src1, src2, src3], dimension=1, result_type=VectorType(f32, [8, 192])
    )
    op.verify()


def test_concatenate_op_infers_result_type():
    src1 = create_ssa_value(VectorType(f32, [8, 64]))
    src2 = create_ssa_value(VectorType(f32, [8, 64]))
    op = ConcatenateOp([src1, src2], dimension=1)
    result_ty = op.results[0].type
    assert isinstance(result_ty, VectorType)
    assert list(result_ty.get_shape()) == [8, 128]


def test_roll_vectors_op_basic():
    src1 = create_ssa_value(VectorType(f32, [8, 128]))
    src2 = create_ssa_value(VectorType(f32, [8, 128]))
    op = RollVectorsOp([src1, src2], result_type=VectorType(f32, [8, 128, 2]))
    assert isinstance(op, RollVectorsOp)


def test_unroll_vectors_op_basic():
    src = create_ssa_value(VectorType(f32, [8, 128, 2]))
    op = UnrollVectorsOp(
        src,
        result_type=[VectorType(f32, [8, 128]), VectorType(f32, [8, 128])],
    )
    assert len(op.results) == 2


def test_unroll_vectors_op_single_result():
    src = create_ssa_value(VectorType(f32, [8, 128]))
    op = UnrollVectorsOp(
        src,
        result_type=[VectorType(f32, [8, 128])],
    )
    assert len(op.results) == 1


def test_transpose_op_basic():
    src = create_ssa_value(VectorType(f32, [8, 128]))
    op = TransposeOp(src, permutation=[1, 0], result_type=VectorType(f32, [128, 8]))
    op.verify()


def test_transpose_op_identity():
    src = create_ssa_value(VectorType(f32, [8, 128]))
    op = TransposeOp(src, permutation=[0, 1], result_type=VectorType(f32, [8, 128]))
    op.verify()


def test_transpose_op_3d():
    src = create_ssa_value(VectorType(f32, [4, 8, 16]))
    op = TransposeOp(
        src, permutation=[2, 0, 1], result_type=VectorType(f32, [16, 4, 8])
    )
    op.verify()


def test_transpose_op_rejects_element_type_mismatch():
    src = create_ssa_value(VectorType(f32, [8, 128]))
    op = TransposeOp(src, permutation=[1, 0], result_type=VectorType(i32, [128, 8]))
    with pytest.raises(VerifyException, match="element types to match"):
        op.verify()


def test_transpose_op_rejects_wrong_permutation_rank():
    src = create_ssa_value(VectorType(f32, [8, 128]))
    op = TransposeOp(src, permutation=[0, 1, 2], result_type=VectorType(f32, [128, 8]))
    with pytest.raises(VerifyException, match="permutation rank to match input rank"):
        op.verify()


def test_transpose_op_rejects_out_of_bounds():
    src = create_ssa_value(VectorType(f32, [8, 128]))
    op = TransposeOp(src, permutation=[0, 2], result_type=VectorType(f32, [8, 128]))
    with pytest.raises(VerifyException, match="out of bounds"):
        op.verify()


def test_transpose_op_rejects_repeated():
    src = create_ssa_value(VectorType(f32, [8, 128]))
    op = TransposeOp(src, permutation=[0, 0], result_type=VectorType(f32, [8, 8]))
    with pytest.raises(VerifyException, match="repeated"):
        op.verify()


def test_transpose_op_rejects_shape_mismatch():
    src = create_ssa_value(VectorType(f32, [8, 128]))
    op = TransposeOp(src, permutation=[1, 0], result_type=VectorType(f32, [99, 8]))
    with pytest.raises(VerifyException, match="permuted by the given permutation"):
        op.verify()
