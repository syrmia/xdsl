from collections.abc import Sequence

from xdsl.dialects.builtin import (
    I32,
    BoolAttr,
    DenseArrayBase,
    IndexType,
    IntegerAttr,
    IntegerType,
    MemRefType,
    NoneAttr,
    VectorType,
    f32,
    i1,
    i32,
)
from xdsl.dialects.tpu_memref import MemorySpace
from xdsl.ir import Attribute, Operation, SSAValue
from xdsl.irdl import (
    AnyOf,
    AttrSizedOperandSegments,
    EqAttrConstraint,
    IRDLOperation,
    attr_def,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    opt_operand_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.traits import (
    HasCanonicalizationPatternsTrait,
    MemoryReadEffect,
    MemoryWriteEffect,
)
from xdsl.utils.exceptions import VerifyException


def _check_base_is_vmem(op_name: str, ref_ty: MemRefType) -> None:
    from xdsl.dialects.tpu_memref import MemorySpaceAttr

    mem_sp = ref_ty.memory_space
    if isinstance(mem_sp, NoneAttr):
        return
    if isinstance(mem_sp, MemorySpaceAttr) and mem_sp.value.data != MemorySpace.Vmem:
        raise VerifyException(f"{op_name}: Expected base memref to be in VMEM.")


def _check_mask_broadcastable(
    op_name: str,
    value_ty: VectorType,
    mask_ty: VectorType,
) -> None:
    value_shape = list(value_ty.get_shape())
    mask_shape = list(mask_ty.get_shape())
    if mask_shape == value_shape:
        return
    if len(mask_shape) == 1 and mask_shape[0] in value_shape:
        return
    raise VerifyException(
        f"{op_name}: Expected mask shape to be broadcastable to result shape."
    )


def _verify_load_op_common(
    op_name: str,
    ref_ty: MemRefType,
    value_ty: VectorType,
    mask: SSAValue | None,
) -> None:
    _check_base_is_vmem(op_name, ref_ty)

    if value_ty.element_type != ref_ty.element_type:
        raise VerifyException(
            f"{op_name}: Expected base and result element type to match."
        )

    if mask is not None:
        mask_ty = mask.type
        assert isinstance(mask_ty, VectorType)
        elem = value_ty.element_type
        bw = getattr(elem, "bitwidth", None)
        if bw is None or bw != 32:
            raise VerifyException(
                f"{op_name}: Not implemented: masked load with non-32-bit element type"
            )
        _check_mask_broadcastable(op_name, value_ty, mask_ty)


def _verify_store_op_common(
    op_name: str,
    ref_ty: MemRefType,
    value_ty: VectorType,
    mask: SSAValue | None,
) -> None:
    _check_base_is_vmem(op_name, ref_ty)

    if value_ty.element_type != ref_ty.element_type:
        raise VerifyException(
            f"{op_name}: Expected base and value_to_store element type to match"
        )

    if mask is not None:
        mask_ty = mask.type
        assert isinstance(mask_ty, VectorType)
        elem = value_ty.element_type
        bw = getattr(elem, "bitwidth", None)
        if bw is None or bw != 32:
            raise VerifyException(
                f"{op_name}: Not implemented: masked store with non-32-bit element type"
            )
        _check_mask_broadcastable(op_name, value_ty, mask_ty)


def _verify_strided_op_common(
    op_name: str,
    ref_ty: MemRefType,
    value_ty: VectorType,
    indices_len: int,
    strides: DenseArrayBase,
    min_stride: int,
) -> None:
    rank = len(ref_ty.get_shape())

    if rank != indices_len:
        raise VerifyException(
            f"{op_name}: Base memref's rank and indices size do not match: "
            f"{rank} vs {indices_len}"
        )

    stride_values = list(strides.get_values())
    if rank != len(stride_values):
        raise VerifyException(
            f"{op_name}: Base memref's rank and strides size do not match: "
            f"{rank} vs {len(stride_values)}"
        )

    value_rank = len(value_ty.get_shape())
    if rank != value_rank:
        raise VerifyException(
            f"{op_name}: Base memref's rank and result's rank do not match: "
            f"{rank} vs {value_rank}"
        )

    for i, s in enumerate(stride_values):
        if s < min_stride:
            raise VerifyException(
                f"{op_name}: Strides[{i}]={s} must be >= {min_stride}"
            )


class ShuffledLoadHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls):
        from xdsl.transforms.canonicalization_patterns.tpu import (
            ShuffledLoadToSimpleLoad,
        )

        return (ShuffledLoadToSimpleLoad(),)


class ShuffledStoreHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls):
        from xdsl.transforms.canonicalization_patterns.tpu import (
            ShuffledStoreToSimpleStore,
        )

        return (ShuffledStoreToSimpleStore(),)


@irdl_op_definition
class LoadOp(IRDLOperation):
    name = "tpu.load"
    base = operand_def(MemRefType)
    indices = var_operand_def(IndexType)
    sublane_mask = opt_attr_def(DenseArrayBase.constr(i1))
    sublane_stride = attr_def(IntegerAttr[I32])
    result = result_def(VectorType)

    traits = traits_def(MemoryReadEffect())

    assembly_format = (
        "$base `[` $indices `]` `sublanes` $sublane_mask `sublane_stride` "
        "$sublane_stride attr-dict `:` type($base) `,` type($result)"
    )

    def __init__(
        self,
        base: SSAValue | Operation,
        indices: Sequence[SSAValue | Operation],
        sublane_mask: DenseArrayBase,
        result_type: Attribute,
        sublane_stride: int | IntegerAttr[IntegerType] = 1,
    ):
        if isinstance(sublane_stride, int):
            sublane_stride = IntegerAttr(sublane_stride, i32)
        super().__init__(
            operands=[base, list(indices)],
            result_types=[result_type],
            attributes={"sublane_mask": sublane_mask, "sublane_stride": sublane_stride},
        )


@irdl_op_definition
class StoreOp(IRDLOperation):
    name = "tpu.store"
    value_to_store = operand_def(VectorType)
    base = operand_def(MemRefType)
    indices = var_operand_def(IndexType)
    sublane_mask = attr_def(DenseArrayBase.constr(i1))
    mask = opt_operand_def()
    sublane_stride = opt_attr_def(IntegerAttr[I32])
    add = attr_def(BoolAttr)

    irdl_options = (AttrSizedOperandSegments(),)
    traits = traits_def(MemoryWriteEffect())

    assembly_format = (
        "$base `[` $indices `]` `,` $value_to_store (`masked` $mask^)? `sublanes` "
        "$sublane_mask `sublane_stride` $sublane_stride attr-dict `:` type($base) `,` type($value_to_store) `,` type($mask)"
    )

    def __init__(
        self,
        value_to_store: SSAValue | Operation,
        base: SSAValue | Operation,
        indices: Sequence[SSAValue | Operation],
        sublane_mask: DenseArrayBase,
        mask: SSAValue | Operation | None = None,
        sublane_stride: int | IntegerAttr[IntegerType] = 1,
        add: bool | BoolAttr = False,
    ):
        if isinstance(sublane_stride, int):
            sublane_stride = IntegerAttr(sublane_stride, i32)
        if isinstance(add, bool):
            add = BoolAttr.from_bool(add)
        mask_list: list[SSAValue | Operation] = [mask] if mask is not None else []
        super().__init__(
            operands=[value_to_store, base, list(indices), mask_list],
            result_types=[],
            attributes={
                "sublane_mask": sublane_mask,
                "sublane_stride": sublane_stride,
                "add": add,
            },
        )


@irdl_op_definition
class VectorLoadOp(IRDLOperation):
    name = "tpu.vector_load"

    base = operand_def(MemRefType)
    indices = var_operand_def(IndexType)
    strides = attr_def(DenseArrayBase.constr(i32))
    mask = opt_operand_def(VectorType)
    result = result_def(VectorType)

    irdl_options = (AttrSizedOperandSegments(),)
    traits = traits_def(MemoryReadEffect())

    assembly_format = "$base `[` $indices `]` (`masked` $mask^)? attr-dict `:` type($base) `,` type($result) `,` type($mask)"

    def __init__(
        self,
        base: SSAValue | Operation,
        indices: Sequence[SSAValue | Operation],
        strides: DenseArrayBase,
        result_type: Attribute,
        mask: SSAValue | Operation | None = None,
    ):
        mask_list: list[SSAValue | Operation] = [mask] if mask is not None else []
        super().__init__(
            operands=[base, list(indices), mask_list],
            result_types=[result_type],
            attributes={"strides": strides},
        )

    def verify_(self) -> None:
        ref_ty = self.base.type
        result_ty = self.result.type
        assert isinstance(ref_ty, MemRefType)
        assert isinstance(result_ty, VectorType)

        rank = len(ref_ty.get_shape())
        if len(self.indices) != rank:
            raise VerifyException(f"tpu.vector_load: Expected {rank} indices.")

        strides_values = list(self.strides.get_values())
        if strides_values:
            if len(strides_values) != rank:
                raise VerifyException(f"tpu.vector_load: Expected {rank} strides.")
            raise VerifyException(
                "tpu.vector_load: Not implemented: general vector load with strides."
            )

        mask = self.mask
        _verify_load_op_common("tpu.vector_load", ref_ty, result_ty, mask)


@irdl_op_definition
class VectorStoreOp(IRDLOperation):
    name = "tpu.vector_store"
    value_to_store = operand_def(VectorType)
    base = operand_def(MemRefType)
    indices = var_operand_def(IndexType)
    strides = attr_def(DenseArrayBase.constr(i32))
    mask = opt_operand_def(VectorType)
    add = opt_attr_def(BoolAttr)

    irdl_options = (AttrSizedOperandSegments(),)
    traits = traits_def(MemoryWriteEffect())

    assembly_format = (
        "$base `[` $indices `]` `,` $value_to_store (`masked` $mask^)? attr-dict "
        "`:` type($base) `,` type($value_to_store) `,` type($mask)"
    )

    def __init__(
        self,
        value_to_store: SSAValue | Operation,
        base: SSAValue | Operation,
        indices: Sequence[SSAValue | Operation],
        strides: DenseArrayBase,
        mask: SSAValue | Operation | None = None,
        add: bool | BoolAttr = False,
    ):
        if isinstance(add, bool):
            add = BoolAttr.from_bool(add)
        mask_list: list[SSAValue | Operation] = [mask] if mask is not None else []
        super().__init__(
            operands=[value_to_store, base, list(indices), mask_list],
            result_types=[],
            attributes={"strides": strides, "add": add},
        )

    def verify_(self) -> None:
        strides_values = list(self.strides.get_values())
        if strides_values:
            raise VerifyException(
                "tpu.vector_store: Not implemented: general vector store with strides."
            )

        ref_ty = self.base.type
        value_ty = self.value_to_store.type
        assert isinstance(ref_ty, MemRefType)
        assert isinstance(value_ty, VectorType)

        rank = len(ref_ty.get_shape())
        if len(self.indices) != rank:
            raise VerifyException(f"tpu.vector_store: Expected {rank} indices.")

        mask = self.mask
        _verify_store_op_common("tpu.vector_store", ref_ty, value_ty, mask)


@irdl_op_definition
class StridedLoadOp(IRDLOperation):
    name = "tpu.strided_load"

    base = operand_def(MemRefType)
    indices = var_operand_def(IndexType)
    strides = attr_def(DenseArrayBase.constr(i32))
    result = result_def(VectorType)
    traits = traits_def(MemoryReadEffect())

    assembly_format = (
        "$base `[` $indices `]` attr-dict `:` type($base) `,` type($result)"
    )

    def __init__(
        self,
        base: SSAValue | Operation,
        indices: Sequence[SSAValue | Operation],
        strides: DenseArrayBase,
        result_type: Attribute,
    ):
        super().__init__(
            operands=[base, list(indices)],
            result_types=[result_type],
            attributes={"strides": strides},
        )

    def verify_(self) -> None:
        ref_ty = self.base.type
        result_ty = self.result.type
        assert isinstance(ref_ty, MemRefType)
        assert isinstance(result_ty, VectorType)
        _verify_strided_op_common(
            "tpu.strided_load",
            ref_ty,
            result_ty,
            len(self.indices),
            self.strides,
            min_stride=0,
        )


@irdl_op_definition
class StridedStoreOp(IRDLOperation):
    name = "tpu.strided_store"
    value_to_store = operand_def(VectorType)
    base = operand_def(MemRefType)
    indices = var_operand_def(IndexType)
    strides = attr_def(DenseArrayBase.constr(i32))

    traits = traits_def(MemoryWriteEffect())

    assembly_format = "$base `[` $indices `]` `,` $value_to_store attr-dict `:` type($base) `,` type($value_to_store)"

    def __init__(
        self,
        value_to_store: SSAValue | Operation,
        base: SSAValue | Operation,
        indices: Sequence[SSAValue | Operation],
        strides: DenseArrayBase,
    ):
        super().__init__(
            operands=[value_to_store, base, list(indices)],
            result_types=[],
            attributes={"strides": strides},
        )

    def verify_(self) -> None:
        ref_ty = self.base.type
        value_ty = self.value_to_store.type
        assert isinstance(ref_ty, MemRefType)
        assert isinstance(value_ty, VectorType)
        _verify_strided_op_common(
            "tpu.strided_store",
            ref_ty,
            value_ty,
            len(self.indices),
            self.strides,
            min_stride=1,
        )


@irdl_op_definition
class ShuffledLoadOp(IRDLOperation):
    name = "tpu.shuffled_load"
    base = operand_def(MemRefType)
    indices = var_operand_def(IndexType)
    sublane_mask = attr_def(DenseArrayBase.constr(i1))
    sublane_offsets = attr_def(DenseArrayBase.constr(i32))
    result = result_def(VectorType)

    traits = traits_def(
        MemoryReadEffect(), ShuffledLoadHasCanonicalizationPatternsTrait()
    )

    assembly_format = (
        "$base `[` $indices `]` attr-dict `:` type($base) `,` type($result)"
    )

    def __init__(
        self,
        base: SSAValue | Operation,
        indices: Sequence[SSAValue | Operation],
        sublane_mask: DenseArrayBase,
        sublane_offsets: DenseArrayBase,
        result_type: Attribute,
    ):
        super().__init__(
            operands=[base, list(indices)],
            result_types=[result_type],
            attributes={
                "sublane_mask": sublane_mask,
                "sublane_offsets": sublane_offsets,
            },
        )

    def verify_(self) -> None:
        ref_ty = self.base.type
        result_ty = self.result.type
        assert isinstance(ref_ty, MemRefType)
        assert isinstance(result_ty, VectorType)

        rank = len(ref_ty.get_shape())
        if rank != len(self.indices):
            raise VerifyException(
                f"tpu.shuffled_load: Base memref's rank and indices size "
                f"do not match: {rank} vs {len(self.indices)}"
            )

        first_dim = result_ty.get_shape()[0]
        sublane_mask_values = list(self.sublane_mask.get_values())
        if len(sublane_mask_values) != first_dim:
            raise VerifyException(
                f"tpu.shuffled_load: Expected sublane mask size equal to {first_dim} but got {len(sublane_mask_values)}"
            )

        sublane_offset_vals = list(self.sublane_offsets.get_values())
        if len(sublane_offset_vals) != first_dim:
            raise VerifyException(
                f"tpu.shuffled_load: Expected sublane offsets size equals to {first_dim} but got {len(sublane_offset_vals)}"
            )


@irdl_op_definition
class ShuffledStoreOp(IRDLOperation):
    name = "tpu.shuffled_store"
    value_to_store = operand_def(VectorType)
    base = operand_def(MemRefType)
    indices = var_operand_def(IndexType)
    sublane_mask = attr_def(DenseArrayBase.constr(i1))
    sublane_offsets = attr_def(DenseArrayBase.constr(i32))

    traits = traits_def(
        MemoryWriteEffect(), ShuffledStoreHasCanonicalizationPatternsTrait()
    )

    assembly_format = "$base `[` $indices `]` `,` $value_to_store attr-dict `:` type($base) `,` type($value_to_store)"

    def __init__(
        self,
        value_to_store: SSAValue | Operation,
        base: SSAValue | Operation,
        indices: Sequence[SSAValue | Operation],
        sublane_mask: DenseArrayBase,
        sublane_offsets: DenseArrayBase,
    ):
        super().__init__(
            operands=[value_to_store, base, list(indices)],
            result_types=[],
            attributes={
                "sublane_mask": sublane_mask,
                "sublane_offsets": sublane_offsets,
            },
        )

    def verify_(self) -> None:
        ref_ty = self.base.type
        value_ty = self.value_to_store.type
        assert isinstance(ref_ty, MemRefType)
        assert isinstance(value_ty, VectorType)

        rank = len(ref_ty.get_shape())
        if rank != len(self.indices):
            raise VerifyException(
                f"tpu.shuffled_store: Base memref's rank and indices size do not match: {rank} vs {len(self.indices)}"
            )

        value_rank = len(value_ty.get_shape())
        if value_rank != len(self.indices):
            raise VerifyException(
                f"tpu.shuffled_store: The rank of value to store and indices do not match: {rank} vs {len(self.indices)}"
            )

        first_dim = value_ty.get_shape()[0]
        sublane_mask_vals = list(self.sublane_mask.get_values())
        if len(sublane_mask_vals) != first_dim:
            raise VerifyException(
                f"tpu.shuffled_store: Expected sublane mask size equals to {first_dim} but got {len(sublane_mask_vals)}"
            )

        sublane_offsets_vals = list(self.sublane_offsets.get_values())
        if len(sublane_offsets_vals) != first_dim:
            raise VerifyException(
                f"tpu.shuffled_store: Expected sublane offset mask size equals to {first_dim} but got {len(sublane_mask_vals)}"
            )


@irdl_op_definition
class VectorLoadIdxOp(IRDLOperation):
    name = "tpu.vector_load_idx"
    base = operand_def(
        MemRefType.constr(AnyOf((EqAttrConstraint(i32), EqAttrConstraint(f32))))
    )
    indices = var_operand_def(VectorType.constr(i32))
    mask = opt_operand_def(VectorType.constr(i1))
    value = result_def(
        VectorType.constr(AnyOf((EqAttrConstraint(i32), EqAttrConstraint(f32))))
    )

    irdl_options = (AttrSizedOperandSegments(),)
    traits = traits_def(MemoryReadEffect())

    assembly_format = (
        "$base `[` $indices `]` (`masked` $mask^)? attr-dict `:` type($base) "
        "`[` type($indices) `]` `,` type($value) `,` type($mask)"
    )

    def __init__(
        self,
        base: SSAValue | Operation,
        indices: Sequence[SSAValue | Operation],
        result_type: Attribute,
        mask: SSAValue | Operation | None = None,
    ):
        mask_list: list[SSAValue | Operation] = [mask] if mask is not None else []
        super().__init__(
            operands=[base, list(indices), mask_list], result_types=[result_type]
        )

    def verify_(self) -> None:
        ref_ty = self.base.type
        value_ty = self.value.type
        assert isinstance(ref_ty, MemRefType)
        assert isinstance(value_ty, VectorType)

        rank = len(ref_ty.get_shape())
        if len(self.indices) != rank:
            raise VerifyException(
                f"tpu.vector_load_idx: Expected one index vector for each dimension of the base memref with dimension: {rank}. "
                f"Got: {len(self.indices)}"
            )

        value_shape = list(value_ty.get_shape())
        for i, index in enumerate(self.indices):
            index_ty = index.type
            assert isinstance(index_ty, VectorType)
            if list(index_ty.get_shape()) != value_shape:
                raise VerifyException(
                    f"tpu.vector_load_idx: Expected {value_shape} elements in indices. Got {list(index_ty.get_shape())}"
                    f"in inedx #{i}."
                )

        mask = self.mask
        _verify_load_op_common("tpu.vector_load_idx", ref_ty, value_ty, mask)


@irdl_op_definition
class VectorStoreIdxOp(IRDLOperation):
    name = "tpu.vector_store_idx"
    value_to_store = operand_def(
        VectorType.constr(AnyOf((EqAttrConstraint(i32), EqAttrConstraint(f32))))
    )
    base = operand_def(
        MemRefType.constr(AnyOf((EqAttrConstraint(i32), EqAttrConstraint(f32))))
    )
    indices = var_operand_def(VectorType.constr(i32))
    mask = opt_operand_def(VectorType.constr(i1))
    add = opt_attr_def(BoolAttr)

    irdl_options = (AttrSizedOperandSegments(),)
    traits = traits_def(MemoryWriteEffect())

    assembly_format = (
        "$base `[` $indices `]` `,` $value_to_store (`masked` $mask^)? attr-dict `:`"
        "type($base) `[` type($indices) `]` `,` type($value_to_store) `,` type($mask)"
    )

    def __init__(
        self,
        value_to_store: SSAValue | Operation,
        base: SSAValue | Operation,
        indices: Sequence[SSAValue | Operation],
        mask: SSAValue | Operation | None = None,
        add: bool | BoolAttr = False,
    ):
        if isinstance(add, bool):
            add = BoolAttr.from_bool(add)
        mask_list: list[SSAValue | Operation] = [mask] if mask is not None else []
        super().__init__(
            operands=[value_to_store, base, list(indices), mask_list],
            result_types=[],
            attributes={"add": add},
        )

    def verify_(self) -> None:
        ref_ty = self.base.type
        value_ty = self.value_to_store.type
        assert isinstance(ref_ty, MemRefType)
        assert isinstance(value_ty, VectorType)

        rank = len(ref_ty.get_shape())
        if len(self.indices) != rank:
            raise VerifyException(
                f"tpu.vector_store_idx: Expected one index vector for each dimension of the base memref with dimension: {rank}. "
                f"Got: {len(self.indices)}"
            )

        if len(value_ty.get_shape()) != 1:
            raise VerifyException(
                f"tpu.vector_store_idx: Expected value to have rank 1. Got: {len(value_ty.get_shape())}"
            )

        value_shape = list(value_ty.get_shape())
        for i, index in enumerate(self.indices):
            index_ty = index.type
            assert isinstance(index_ty, VectorType)
            if list(index_ty.get_shape()) != value_shape:
                raise VerifyException(
                    f"tpu.vector_store_idx: Expected {value_shape} elements in indices. Got {list(index_ty.get_shape())}"
                    f"in index #{i}."
                )

        mask = self.mask
        _verify_store_op_common("tpu.vector_store_idx", ref_ty, value_ty, mask)
