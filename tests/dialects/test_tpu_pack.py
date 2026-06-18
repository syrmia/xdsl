import pytest

from xdsl.dialects.builtin import (
    Float32Type,
    IndexType,
    IntegerType,
    VectorType,
    f32,
    i1,
    i32,
)
from xdsl.dialects.tpu import (
    PackFormat,
    PackFormatAttr,
)
from xdsl.dialects.tpu_pack import (
    CreateMaskOp,
    CreateSubelementMaskOp,
    PackElementwiseOp,
    PackMaskOp,
    PackSubelementsOp,
    SublaneShuffleOp,
    UnpackElementwiseOp,
    UnpackSubelementsOp,
    _get_element_bitwidth,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import create_ssa_value


def mask_packing_factor(vty: VectorType) -> int:
    shape = list(vty.get_shape())
    if len(shape) == 2:
        return 1
    return shape[2]


def test_get_element_bitwidth_scalar_int():
    assert _get_element_bitwidth(IntegerType(32)) == 32


def test_get_element_bitwidth_scalar_f32():
    assert _get_element_bitwidth(Float32Type()) == 32


def test_get_element_bitwidth_vector():
    assert _get_element_bitwidth(VectorType(IntegerType(16), [8])) == 16


def test_unpack_subelements_basic():
    src = create_ssa_value(VectorType(IntegerType(16), [8]))
    op = UnpackSubelementsOp(
        src,
        index=0,
        pack_format=PackFormatAttr(PackFormat.Compressed),
        result_type=VectorType(IntegerType(32), [8]),
    )
    op.verify()


def test_unpack_subelements_with_int_index():
    src = create_ssa_value(VectorType(IntegerType(8), [16]))
    op = UnpackSubelementsOp(
        src,
        index=2,
        pack_format=PackFormatAttr(PackFormat.Interleaved),
        result_type=VectorType(IntegerType(32), [16]),
    )
    op.verify()
    assert op.index.value.data == 2


def test_unpack_subelements_defaults_for_bool_attrs():
    src = create_ssa_value(VectorType(IntegerType(16), [8]))
    op = UnpackSubelementsOp(
        src,
        index=0,
        pack_format=PackFormatAttr(PackFormat.Compressed),
        result_type=VectorType(IntegerType(32), [8]),
    )
    assert bool(op.integer_extended.value.data) is True
    assert bool(op.unsigned_integers.value.data) is False


def test_pack_subelements_basic():
    src1 = create_ssa_value(VectorType(IntegerType(32), [8]))
    src2 = create_ssa_value(VectorType(IntegerType(32), [8]))
    op = PackSubelementsOp(
        [src1, src2],
        positions=[0, 1],
        pack_format=PackFormatAttr(PackFormat.Compressed),
        result_type=VectorType(IntegerType(16), [8]),
    )
    op.verify()


def test_pack_subelements_more_sources():
    sources = [create_ssa_value(VectorType(IntegerType(32), [8])) for _ in range(4)]
    op = PackSubelementsOp(
        sources,
        positions=[0, 1, 2, 3],
        pack_format=PackFormatAttr(PackFormat.Compressed),
        result_type=VectorType(IntegerType(8), [8]),
    )
    op.verify()


def test_pack_elementwise_basic():
    src1 = create_ssa_value(VectorType(i32, [8]))
    src2 = create_ssa_value(VectorType(i32, [8]))
    op = PackElementwiseOp(
        [src1, src2],
        target_type=IntegerType(16),
        result_type=VectorType(i32, [8]),
    )
    op.verify()


def test_unpack_elementwise_basic():
    from xdsl.dialects.builtin import BFloat16Type

    src = create_ssa_value(VectorType(i32, [8]))
    op = UnpackElementwiseOp(
        src,
        source_type=BFloat16Type(),
        index=0,
        result_type=VectorType(f32, [8]),
    )
    op.verify()


def test_unpack_elementwise_index_one():
    src = create_ssa_value(VectorType(i32, [8]))
    op = UnpackElementwiseOp(
        src,
        source_type=IntegerType(16),
        index=1,
        result_type=VectorType(i32, [8]),
    )
    op.verify()
    assert op.index.value.data == 1


def test_pack_mask_basic():
    src1 = create_ssa_value(VectorType(i1, [8, 128]))
    src2 = create_ssa_value(VectorType(i1, [8, 128]))
    op = PackMaskOp(
        [src1, src2],
        positions=[0, 1],
        result_type=VectorType(i1, [8, 128, 2]),
    )
    op.verify()


def test_pack_mask_with_dense_array_positions():
    from xdsl.dialects.builtin import DenseArrayBase

    src1 = create_ssa_value(VectorType(i1, [8, 128]))
    src2 = create_ssa_value(VectorType(i1, [8, 128]))
    positions = DenseArrayBase.from_list(i32, [0, 1])
    op = PackMaskOp(
        [src1, src2],
        positions=positions,
        result_type=VectorType(i1, [8, 128, 2]),
    )
    op.verify()


def test_create_mask_basic():
    low1 = create_ssa_value(IndexType())
    low2 = create_ssa_value(IndexType())
    high1 = create_ssa_value(IndexType())
    high2 = create_ssa_value(IndexType())
    op = CreateMaskOp(
        [low1, low2],
        [high1, high2],
        result_type=VectorType(i1, [8, 128]),
    )
    op.verify()


def test_create_mask_1d():
    low = create_ssa_value(IndexType())
    high = create_ssa_value(IndexType())
    op = CreateMaskOp(
        [low],
        [high],
        result_type=VectorType(i1, [16]),
    )
    op.verify()


def test_create_mask_rejects_size_mismatch():
    low = [create_ssa_value(IndexType())]
    high = [create_ssa_value(IndexType()), create_ssa_value(IndexType())]
    with pytest.raises(ValueError, match="different sizes"):
        CreateMaskOp(low, high, result_type=VectorType(i1, [8, 128]))


def test_create_subelement_mask_basic():
    op = CreateSubelementMaskOp(
        from_value=0,
        to=8,
        result_type=VectorType(i1, [8, 128]),
    )
    op.verify()


def test_create_subelement_mask_attr_inputs():
    from xdsl.dialects.builtin import IntegerAttr

    op = CreateSubelementMaskOp(
        from_value=IntegerAttr(0, i32),
        to=IntegerAttr(16, i32),
        result_type=VectorType(i1, [16, 128]),
    )
    op.verify()


def test_sublane_shuffle_basic():
    lhs = create_ssa_value(VectorType(f32, [8, 128]))
    rhs = create_ssa_value(VectorType(f32, [8, 128]))
    pattern = [0, 1, 2, 3, 8, 9, 10, 11]
    op = SublaneShuffleOp(
        lhs,
        rhs,
        pattern,
        result_type=VectorType(f32, [8, 128]),
    )
    op.verify()


def test_sublane_shuffle_with_dense_array_pattern():
    from xdsl.dialects.builtin import DenseArrayBase

    lhs = create_ssa_value(VectorType(f32, [8, 128]))
    rhs = create_ssa_value(VectorType(f32, [8, 128]))
    pattern = DenseArrayBase.from_list(i32, [0, 1, 2, 3, 4, 5, 6, 7])
    op = SublaneShuffleOp(
        lhs,
        rhs,
        pattern,
        result_type=VectorType(f32, [8, 128]),
    )
    op.verify()


def test_pack_mask_rejects_1d_output():
    src = create_ssa_value(VectorType(i1, [8]))
    op = PackMaskOp(
        [src],
        positions=[0],
        result_type=VectorType(i1, [8]),
    )
    with pytest.raises(VerifyException, match="must be 2D or 3D"):
        op.verify()
