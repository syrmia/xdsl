import pytest

from xdsl.dialects.builtin import (
    DenseArrayBase,
    IndexType,
    MemRefType,
    VectorType,
    f32,
    i1,
    i32,
)
from xdsl.dialects.tpu import (
    MemorySpaceAttr,
)
from xdsl.dialects.tpu_memory import (
    LoadOp,
    ShuffledLoadOp,
    ShuffledStoreOp,
    StoreOp,
    StridedLoadOp,
    StridedStoreOp,
    VectorLoadIdxOp,
    VectorLoadOp,
    VectorStoreIdxOp,
    VectorStoreOp,
    _check_base_is_vmem,
    _check_mask_broadcastable,
    _verify_strided_op_common,
)
from xdsl.dialects.tpu_memref import MemorySpace
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import create_ssa_value


def _vmem_memref(shape: list[int], elem_ty=f32) -> MemRefType:
    return MemRefType(elem_ty, shape, memory_space=MemorySpaceAttr(MemorySpace.Vmem))


def _smem_memref(shape: list[int], elem_ty=f32) -> MemRefType:
    return MemRefType(elem_ty, shape, memory_space=MemorySpaceAttr(MemorySpace.Smem))


def test_check_base_is_vmem_accepts_vmem():
    ref = _vmem_memref([8, 128])
    _check_base_is_vmem("test", ref)


def test_check_base_is_vmem_accepts_unset_memory_space():
    ref = MemRefType(f32, [8, 128])
    _check_base_is_vmem("test", ref)


def test_check_base_is_vmem_rejects_smem():
    ref = _smem_memref([8, 128])
    with pytest.raises(VerifyException, match="VMEM"):
        _check_base_is_vmem("test", ref)


def test_check_mask_broadcastable_same_shape():
    value_ty = VectorType(f32, [8, 128])
    mask_ty = VectorType(i1, [8, 128])
    _check_mask_broadcastable("test", value_ty, mask_ty)


def test_check_mask_broadcastable_1d_broadcastable():
    value_ty = VectorType(f32, [8, 128])
    mask_ty = VectorType(i1, [8])
    _check_mask_broadcastable("test", value_ty, mask_ty)


def test_check_mask_broadcastable_rejects_incompatible():
    value_ty = VectorType(f32, [8, 128])
    mask_ty = VectorType(i1, [16, 64])
    with pytest.raises(VerifyException, match="broadcastable"):
        _check_mask_broadcastable("test", value_ty, mask_ty)


def test_verify_strided_op_common_rejects_rank_mismatch():
    ref_ty = _vmem_memref([8, 128])
    value_ty = VectorType(f32, [8, 128])
    strides = DenseArrayBase.from_list(i32, [1, 1])
    with pytest.raises(VerifyException, match="rank and indices size"):
        _verify_strided_op_common(
            "test",
            ref_ty,
            value_ty,
            indices_len=1,
            strides=strides,
            min_stride=0,
        )


def test_verify_strided_op_common_rejects_bad_stride():
    ref_ty = _vmem_memref([8, 128])
    value_ty = VectorType(f32, [8, 128])
    strides = DenseArrayBase.from_list(i32, [1, 0])
    with pytest.raises(VerifyException, match="must be >="):
        _verify_strided_op_common(
            "test",
            ref_ty,
            value_ty,
            indices_len=2,
            strides=strides,
            min_stride=1,
        )


def test_load_op_basic():
    mem = create_ssa_value(_vmem_memref([8, 128]))
    idx0 = create_ssa_value(IndexType())
    idx1 = create_ssa_value(IndexType())
    sublane_mask = DenseArrayBase.from_list(i1, [1] * 8)
    op = LoadOp(
        mem,
        [idx0, idx1],
        sublane_mask=sublane_mask,
        result_type=VectorType(f32, [8, 128]),
    )
    op.verify()


def test_load_op_custom_sublane_stride():
    mem = create_ssa_value(_vmem_memref([8, 128]))
    idx0 = create_ssa_value(IndexType())
    idx1 = create_ssa_value(IndexType())
    sublane_mask = DenseArrayBase.from_list(i1, [1] * 8)
    op = LoadOp(
        mem,
        [idx0, idx1],
        sublane_mask=sublane_mask,
        result_type=VectorType(f32, [8, 128]),
        sublane_stride=2,
    )
    assert op.sublane_stride.value.data == 2


def test_store_op_basic():
    val = create_ssa_value(VectorType(f32, [8, 128]))
    mem = create_ssa_value(_vmem_memref([8, 128]))
    idx0 = create_ssa_value(IndexType())
    idx1 = create_ssa_value(IndexType())
    sublane_mask = DenseArrayBase.from_list(i1, [1] * 8)
    op = StoreOp(
        val,
        mem,
        [idx0, idx1],
        sublane_mask=sublane_mask,
    )
    assert op.add.value.data == 0


def test_store_op_with_mask():
    val = create_ssa_value(VectorType(f32, [8, 128]))
    mem = create_ssa_value(_vmem_memref([8, 128]))
    idx0 = create_ssa_value(IndexType())
    idx1 = create_ssa_value(IndexType())
    mask = create_ssa_value(VectorType(i1, [8, 128]))
    sublane_mask = DenseArrayBase.from_list(i1, [1] * 8)
    op = StoreOp(
        val,
        mem,
        [idx0, idx1],
        sublane_mask=sublane_mask,
        mask=mask,
    )
    assert op.mask is not None


def test_store_op_add_true():
    val = create_ssa_value(VectorType(f32, [8, 128]))
    mem = create_ssa_value(_vmem_memref([8, 128]))
    idx0 = create_ssa_value(IndexType())
    idx1 = create_ssa_value(IndexType())
    sublane_mask = DenseArrayBase.from_list(i1, [1] * 8)
    op = StoreOp(
        val,
        mem,
        [idx0, idx1],
        sublane_mask=sublane_mask,
        add=True,
    )
    assert op.add.value.data == -1


def test_vector_load_basic():
    mem = create_ssa_value(_vmem_memref([8, 128]))
    idx0 = create_ssa_value(IndexType())
    idx1 = create_ssa_value(IndexType())
    op = VectorLoadOp(
        mem,
        [idx0, idx1],
        strides=DenseArrayBase.from_list(i32, []),
        result_type=VectorType(f32, [8, 128]),
    )
    op.verify()


def test_vector_load_with_mask():
    mem = create_ssa_value(_vmem_memref([8, 128]))
    idx0 = create_ssa_value(IndexType())
    idx1 = create_ssa_value(IndexType())
    mask = create_ssa_value(VectorType(i1, [8, 128]))
    op = VectorLoadOp(
        mem,
        [idx0, idx1],
        strides=DenseArrayBase.from_list(i32, []),
        result_type=VectorType(f32, [8, 128]),
        mask=mask,
    )
    op.verify()


def test_vector_load_rejects_wrong_indices_count():
    mem = create_ssa_value(_vmem_memref([8, 128]))
    idx0 = create_ssa_value(IndexType())
    op = VectorLoadOp(
        mem,
        [idx0],
        strides=DenseArrayBase.from_list(i32, []),
        result_type=VectorType(f32, [8, 128]),
    )
    with pytest.raises(VerifyException, match="Expected 2 indices"):
        op.verify()


def test_vector_load_rejects_element_type_mismatch():
    mem = create_ssa_value(_vmem_memref([8, 128]))
    idx0 = create_ssa_value(IndexType())
    idx1 = create_ssa_value(IndexType())
    op = VectorLoadOp(
        mem,
        [idx0, idx1],
        strides=DenseArrayBase.from_list(i32, []),
        result_type=VectorType(i32, [8, 128]),
    )
    with pytest.raises(VerifyException, match="element type"):
        op.verify()


def test_vector_load_rejects_strides_set():
    mem = create_ssa_value(_vmem_memref([8, 128]))
    idx0 = create_ssa_value(IndexType())
    idx1 = create_ssa_value(IndexType())
    op = VectorLoadOp(
        mem,
        [idx0, idx1],
        strides=DenseArrayBase.from_list(i32, [1, 1]),
        result_type=VectorType(f32, [8, 128]),
    )
    with pytest.raises(VerifyException, match="Not implemented"):
        op.verify()


def test_vector_load_rejects_non_vmem():
    mem = create_ssa_value(_smem_memref([8, 128]))
    idx0 = create_ssa_value(IndexType())
    idx1 = create_ssa_value(IndexType())
    op = VectorLoadOp(
        mem,
        [idx0, idx1],
        strides=DenseArrayBase.from_list(i32, []),
        result_type=VectorType(f32, [8, 128]),
    )
    with pytest.raises(VerifyException, match="VMEM"):
        op.verify()


def test_vector_store_basic():
    val = create_ssa_value(VectorType(f32, [8, 128]))
    mem = create_ssa_value(_vmem_memref([8, 128]))
    idx0 = create_ssa_value(IndexType())
    idx1 = create_ssa_value(IndexType())
    op = VectorStoreOp(
        val,
        mem,
        [idx0, idx1],
        strides=DenseArrayBase.from_list(i32, []),
    )
    op.verify()


def test_vector_store_with_mask():
    val = create_ssa_value(VectorType(f32, [8, 128]))
    mem = create_ssa_value(_vmem_memref([8, 128]))
    idx0 = create_ssa_value(IndexType())
    idx1 = create_ssa_value(IndexType())
    mask = create_ssa_value(VectorType(i1, [8, 128]))
    op = VectorStoreOp(
        val,
        mem,
        [idx0, idx1],
        strides=DenseArrayBase.from_list(i32, []),
        mask=mask,
    )
    op.verify()


def test_vector_store_rejects_strides_set():
    val = create_ssa_value(VectorType(f32, [8, 128]))
    mem = create_ssa_value(_vmem_memref([8, 128]))
    idx0 = create_ssa_value(IndexType())
    idx1 = create_ssa_value(IndexType())
    op = VectorStoreOp(
        val,
        mem,
        [idx0, idx1],
        strides=DenseArrayBase.from_list(i32, [1, 1]),
    )
    with pytest.raises(VerifyException, match="Not implemented"):
        op.verify()


def test_vector_store_rejects_element_type_mismatch():
    val = create_ssa_value(VectorType(i32, [8, 128]))
    mem = create_ssa_value(_vmem_memref([8, 128]))
    idx0 = create_ssa_value(IndexType())
    idx1 = create_ssa_value(IndexType())
    op = VectorStoreOp(
        val,
        mem,
        [idx0, idx1],
        strides=DenseArrayBase.from_list(i32, []),
    )
    with pytest.raises(VerifyException, match="element type"):
        op.verify()


def test_strided_load_basic():
    mem = create_ssa_value(_vmem_memref([8, 128]))
    idx0 = create_ssa_value(IndexType())
    idx1 = create_ssa_value(IndexType())
    op = StridedLoadOp(
        mem,
        [idx0, idx1],
        strides=DenseArrayBase.from_list(i32, [1, 1]),
        result_type=VectorType(f32, [8, 128]),
    )
    op.verify()


def test_strided_load_zero_stride_allowed():
    mem = create_ssa_value(_vmem_memref([8, 128]))
    idx0 = create_ssa_value(IndexType())
    idx1 = create_ssa_value(IndexType())
    op = StridedLoadOp(
        mem,
        [idx0, idx1],
        strides=DenseArrayBase.from_list(i32, [0, 1]),
        result_type=VectorType(f32, [8, 128]),
    )
    op.verify()


def test_strided_load_rejects_wrong_strides_count():
    mem = create_ssa_value(_vmem_memref([8, 128]))
    idx0 = create_ssa_value(IndexType())
    idx1 = create_ssa_value(IndexType())
    op = StridedLoadOp(
        mem,
        [idx0, idx1],
        strides=DenseArrayBase.from_list(i32, [1]),
        result_type=VectorType(f32, [8, 128]),
    )
    with pytest.raises(VerifyException, match="rank and strides size"):
        op.verify()


def test_strided_store_basic():
    val = create_ssa_value(VectorType(f32, [8, 128]))
    mem = create_ssa_value(_vmem_memref([8, 128]))
    idx0 = create_ssa_value(IndexType())
    idx1 = create_ssa_value(IndexType())
    op = StridedStoreOp(
        val,
        mem,
        [idx0, idx1],
        strides=DenseArrayBase.from_list(i32, [1, 1]),
    )
    op.verify()


def test_strided_store_rejects_zero_stride():
    val = create_ssa_value(VectorType(f32, [8, 128]))
    mem = create_ssa_value(_vmem_memref([8, 128]))
    idx0 = create_ssa_value(IndexType())
    idx1 = create_ssa_value(IndexType())
    op = StridedStoreOp(
        val,
        mem,
        [idx0, idx1],
        strides=DenseArrayBase.from_list(i32, [0, 1]),
    )
    with pytest.raises(VerifyException, match="must be >="):
        op.verify()


def test_shuffled_load_basic():
    mem = create_ssa_value(_vmem_memref([8, 128]))
    idx0 = create_ssa_value(IndexType())
    idx1 = create_ssa_value(IndexType())
    sublane_mask = DenseArrayBase.from_list(i1, [1] * 8)
    sublane_offsets = DenseArrayBase.from_list(i32, [0] * 8)
    op = ShuffledLoadOp(
        mem,
        [idx0, idx1],
        sublane_mask=sublane_mask,
        sublane_offsets=sublane_offsets,
        result_type=VectorType(f32, [8, 128]),
    )
    op.verify()


def test_shuffled_load_rejects_mask_size_mismatch():
    mem = create_ssa_value(_vmem_memref([8, 128]))
    idx0 = create_ssa_value(IndexType())
    idx1 = create_ssa_value(IndexType())
    sublane_mask = DenseArrayBase.from_list(i1, [1] * 4)
    sublane_offsets = DenseArrayBase.from_list(i32, [0] * 8)
    op = ShuffledLoadOp(
        mem,
        [idx0, idx1],
        sublane_mask=sublane_mask,
        sublane_offsets=sublane_offsets,
        result_type=VectorType(f32, [8, 128]),
    )
    with pytest.raises(VerifyException, match="sublane mask"):
        op.verify()


def test_shuffled_store_basic():
    val = create_ssa_value(VectorType(f32, [8, 128]))
    mem = create_ssa_value(_vmem_memref([8, 128]))
    idx0 = create_ssa_value(IndexType())
    idx1 = create_ssa_value(IndexType())
    sublane_mask = DenseArrayBase.from_list(i1, [1] * 8)
    sublane_offsets = DenseArrayBase.from_list(i32, [0] * 8)
    op = ShuffledStoreOp(
        val,
        mem,
        [idx0, idx1],
        sublane_mask=sublane_mask,
        sublane_offsets=sublane_offsets,
    )
    op.verify()


def test_shuffled_store_rejects_offsets_size_mismatch():
    val = create_ssa_value(VectorType(f32, [8, 128]))
    mem = create_ssa_value(_vmem_memref([8, 128]))
    idx0 = create_ssa_value(IndexType())
    idx1 = create_ssa_value(IndexType())
    sublane_mask = DenseArrayBase.from_list(i1, [1] * 8)
    sublane_offsets = DenseArrayBase.from_list(i32, [0] * 4)
    op = ShuffledStoreOp(
        val,
        mem,
        [idx0, idx1],
        sublane_mask=sublane_mask,
        sublane_offsets=sublane_offsets,
    )
    with pytest.raises(VerifyException, match="sublane"):
        op.verify()


def test_vector_load_idx_basic():
    mem = create_ssa_value(MemRefType(f32, [16, 32]))
    idx0 = create_ssa_value(VectorType(i32, [8]))
    idx1 = create_ssa_value(VectorType(i32, [8]))
    op = VectorLoadIdxOp(
        mem,
        [idx0, idx1],
        result_type=VectorType(f32, [8]),
    )
    op.verify()


def test_vector_load_idx_with_mask():
    mem = create_ssa_value(MemRefType(f32, [16, 32]))
    idx0 = create_ssa_value(VectorType(i32, [8]))
    idx1 = create_ssa_value(VectorType(i32, [8]))
    mask = create_ssa_value(VectorType(i1, [8]))
    op = VectorLoadIdxOp(
        mem,
        [idx0, idx1],
        result_type=VectorType(f32, [8]),
        mask=mask,
    )
    op.verify()


def test_vector_load_idx_rejects_wrong_indices_count():
    mem = create_ssa_value(MemRefType(f32, [16, 32]))
    idx0 = create_ssa_value(VectorType(i32, [8]))
    op = VectorLoadIdxOp(
        mem,
        [idx0],
        result_type=VectorType(f32, [8]),
    )
    with pytest.raises(VerifyException, match="dimension of the base memref"):
        op.verify()


def test_vector_load_idx_rejects_index_shape_mismatch():
    mem = create_ssa_value(MemRefType(f32, [16, 32]))
    idx0 = create_ssa_value(VectorType(i32, [8]))
    idx1 = create_ssa_value(VectorType(i32, [16]))
    op = VectorLoadIdxOp(
        mem,
        [idx0, idx1],
        result_type=VectorType(f32, [8]),
    )
    with pytest.raises(VerifyException, match="elements in indices"):
        op.verify()


def test_vector_store_idx_basic():
    val = create_ssa_value(VectorType(f32, [8]))
    mem = create_ssa_value(MemRefType(f32, [16, 32]))
    idx0 = create_ssa_value(VectorType(i32, [8]))
    idx1 = create_ssa_value(VectorType(i32, [8]))
    op = VectorStoreIdxOp(
        val,
        mem,
        [idx0, idx1],
    )
    op.verify()


def test_vector_store_idx_rejects_non_1d_value():
    val = create_ssa_value(VectorType(f32, [8, 4]))
    mem = create_ssa_value(MemRefType(f32, [16, 32]))
    idx0 = create_ssa_value(VectorType(i32, [8, 4]))
    idx1 = create_ssa_value(VectorType(i32, [8, 4]))
    op = VectorStoreIdxOp(
        val,
        mem,
        [idx0, idx1],
    )
    with pytest.raises(VerifyException, match="rank 1"):
        op.verify()
