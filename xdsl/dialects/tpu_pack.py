from collections.abc import Sequence

from xdsl.dialects.builtin import (
    I32,
    BFloat16Type,
    BoolAttr,
    DenseArrayBase,
    Float32Type,
    IndexType,
    IntegerAttr,
    IntegerType,
    VectorType,
    f32,
    i1,
    i32,
)
from xdsl.ir import Attribute, Operation, SSAValue, TypeAttribute
from xdsl.irdl import (
    AnyOf,
    BaseAttr,
    EqAttrConstraint,
    IRDLOperation,
    SameVariadicOperandSize,
    attr_def,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.traits import (
    HasCanonicalizationPatternsTrait,
    Pure,
    SameOperandsAndResultType,
)
from xdsl.utils.exceptions import VerifyException


class UnpackSubelementsHasCanonicalizationPatternsTrait(
    HasCanonicalizationPatternsTrait
):
    @classmethod
    def get_canonicalization_patterns(cls):
        from xdsl.transforms.canonicalization_patterns.tpu import (
            UnpackOfPackCancel,
            UnpackOfPackSignExtensionDemote,
        )

        return (UnpackOfPackCancel(), UnpackOfPackSignExtensionDemote())


def _verify_pack_op(
    op_name: str,
    sources: Sequence[SSAValue],
    positions: DenseArrayBase,
    max_size: int,
) -> None:
    if len(sources) == 0:
        raise VerifyException(f"{op_name}: At least one source is required")

    first_ty = sources[0].type
    for s in sources:
        if s.type != first_ty:
            raise VerifyException(f"{op_name}: All sources must have the same type")

    position_values: list[int] = list(positions.get_values())

    if len(position_values) != len(sources):
        raise VerifyException(f"{op_name}: Size of sources and positions must match")

    if len(sources) > max_size:
        raise VerifyException(
            f"{op_name}: Number of sources must be less than max_size ({max_size}), got {len(sources)}"
        )

    seen: set[int] = set()
    for p in position_values:
        if p < 0 or p >= max_size:
            raise VerifyException(
                f"{op_name}: Positions must be between 0 and max_size ({max_size}), got {p}"
            )
        if p in seen:
            raise VerifyException(f"{op_name}: Positions must be unique")
        seen.add(p)


def _verify_elementwise_packing(
    op_name: str,
    unpacked_ty: Attribute,
    packed_ty: Attribute,
) -> None:
    if isinstance(unpacked_ty, Float32Type):
        if not isinstance(packed_ty, BFloat16Type):
            raise VerifyException(
                f"{op_name}: Only packing/unpacking between f32 and bf16 is supported for floats"
            )
        return

    if isinstance(unpacked_ty, IntegerType) and unpacked_ty.width.data == 32:
        if isinstance(packed_ty, IntegerType) and packed_ty.width.data in (
            16,
            8,
            4,
        ):
            return
        raise VerifyException(
            f"{op_name}: Only packing/unpacking between i32 and i16/i8/i4 is supported for integers"
        )


def _get_element_bitwidth(ty: Attribute) -> int | None:
    if isinstance(ty, VectorType):
        elt = ty.element_type
    else:
        elt = ty
    bw = getattr(elt, "bitwidth", None)
    if isinstance(bw, int):
        return bw
    if isinstance(elt, IntegerType):
        return elt.width.data
    if isinstance(elt, Float32Type):
        return 32
    if isinstance(elt, BFloat16Type):
        return 16
    return None


@irdl_op_definition
class UnpackSubelementsOp(IRDLOperation):
    name = "tpu.unpack_subelements"

    source = operand_def(VectorType)
    index = attr_def(IntegerAttr[I32])
    pack_format = attr_def(Attribute)
    integer_extended = attr_def(BoolAttr)
    unsigned_integers = attr_def(BoolAttr)

    output = result_def(VectorType)

    traits = traits_def(Pure(), UnpackSubelementsHasCanonicalizationPatternsTrait())

    assembly_format = (
        "$source `,` $index attr-dict `:` type($source) `->` type($output)"
    )

    def __init__(
        self,
        source: SSAValue | Operation,
        index: int | IntegerAttr[IntegerType],
        pack_format: Attribute,
        result_type: Attribute,
        integer_extended: bool | BoolAttr = True,
        unsigned_integers: bool | BoolAttr = False,
    ):
        if isinstance(index, int):
            index = IntegerAttr(index, i32)
        if isinstance(integer_extended, bool):
            integer_extended = BoolAttr.from_bool(integer_extended)
        if isinstance(unsigned_integers, bool):
            unsigned_integers = BoolAttr.from_bool(unsigned_integers)
        super().__init__(
            operands=[source],
            result_types=[result_type],
            attributes={
                "index": index,
                "pack_format": pack_format,
                "integer_extended": integer_extended,
                "unsigned_integers": unsigned_integers,
            },
        )

    def verify_(self) -> None:
        source_ty = self.source.type
        output_ty = self.output.type
        assert isinstance(source_ty, VectorType)
        assert isinstance(output_ty, VectorType)

        src_bw = _get_element_bitwidth(source_ty)
        out_bw = _get_element_bitwidth(output_ty)
        if src_bw is None or out_bw is None or src_bw == 0:
            return

        packing_factor = out_bw // src_bw
        index_val = self.index.value.data
        if index_val >= packing_factor:
            raise VerifyException(
                f"tpu.unpack_subelements: Index must be between 0 and the packing factor ({packing_factor}), gor {index_val}"
            )

        if self.unsigned_integers.value.data and not (
            isinstance(source_ty.element_type, IntegerType)
            and source_ty.element_type.signedness.data.name == "SIGNLESS"
        ):
            raise VerifyException(
                "tpu.unpack_subelements: unsigned_integers can only be set when the source type is an integer"
            )

    # TODO:canonicalizer


@irdl_op_definition
class PackSubelementsOp(IRDLOperation):
    name = "tpu.pack_subelements"

    sources = var_operand_def(VectorType)
    positions = attr_def(DenseArrayBase.constr(i32))
    pack_format = attr_def(Attribute)
    unsigned_integers = opt_attr_def(BoolAttr)
    output = result_def(VectorType)

    traits = traits_def(Pure())

    assembly_format = "$sources attr-dict `:` type($sources) `->` type($output)"

    def __init__(
        self,
        sources: Sequence[SSAValue | Operation],
        positions: DenseArrayBase | Sequence[int],
        pack_format: Attribute,
        result_type: Attribute,
        unsigned_integers: bool | BoolAttr = False,
    ):
        if not isinstance(positions, DenseArrayBase):
            positions = DenseArrayBase.from_list(i32, list(positions))
        attrs: dict[str, Attribute] = {
            "positions": positions,
            "pack_format": pack_format,
        }
        if unsigned_integers is not None:
            if isinstance(unsigned_integers, bool):
                unsigned_integers = BoolAttr.from_bool(unsigned_integers)
            attrs["unsigned_integers"] = unsigned_integers

        super().__init__(
            operands=[list(sources)], result_types=[result_type], attributes=attrs
        )

    def verify_(self) -> None:
        output_ty = self.output.type
        assert isinstance(output_ty, VectorType)
        if len(self.sources) == 0:
            raise VerifyException(
                "tpu.pack_subelements: At least one source is required"
            )
        first_src_ty = self.sources[0].type
        assert isinstance(first_src_ty, VectorType)

        src_bw = _get_element_bitwidth(first_src_ty)
        out_bw = _get_element_bitwidth(output_ty)
        if src_bw is None or out_bw is None or out_bw == 0:
            _verify_pack_op(
                "tpu.pack_subelements",
                list(self.sources),
                self.positions,
                max_size=len(self.sources),
            )
            return

        max_size = src_bw // out_bw
        _verify_pack_op(
            "tpu.pack_subelements",
            list(self.sources),
            self.positions,
            max_size=max_size,
        )


@irdl_op_definition
class PackElementwiseOp(IRDLOperation):
    name = "tpu.pack_elementwise"
    sources = var_operand_def(
        VectorType.constr(AnyOf((EqAttrConstraint(f32), BaseAttr(IntegerType))))
    )
    target_type = attr_def(TypeAttribute)
    output = result_def(VectorType.constr(BaseAttr(IntegerType)))

    traits = traits_def(Pure())

    assembly_format = "$sources attr-dict `:` type($sources) `->` type($output)"

    def __init__(
        self,
        sources: Sequence[SSAValue | Operation],
        target_type: TypeAttribute,
        result_type: Attribute,
    ):
        super().__init__(
            operands=[list(sources)],
            result_types=[result_type],
            attributes={"target_type": target_type},
        )

    def verify_(self) -> None:
        if len(self.sources) == 0:
            raise VerifyException(
                "tpu.pack_elementwise: At least one source is required"
            )
        first_src_ty = self.sources[0].type
        assert isinstance(first_src_ty, VectorType)
        for s in self.sources:
            if s.type != first_src_ty:
                raise VerifyException(
                    "tpu.pack_elementwise: All sources must have the same type"
                )

        output_ty = self.output.type
        assert isinstance(output_ty, VectorType)

        src_bw = _get_element_bitwidth(first_src_ty)
        out_bw = _get_element_bitwidth(output_ty)
        if src_bw is not None and out_bw is not None and src_bw != out_bw:
            raise VerifyException(
                "tpu.pack_elementwise: All sources must have the same bitwidth as the result"
            )

        out_elt = output_ty.element_type
        if not (
            isinstance(out_elt, IntegerType)
            and out_elt.signedness.data.name == "SIGNLESS"
        ):
            raise VerifyException(
                "tpu.pack_elementwise: Output type must be a signless integer type"
            )

        src_elt = first_src_ty.element_type
        tgt_elt = self.target_type
        f32_to_bf16 = isinstance(src_elt, Float32Type) and isinstance(
            tgt_elt, BFloat16Type
        )
        int_to_int = (
            isinstance(src_elt, IntegerType)
            and src_elt.signedness.data.name == "SIGNLESS"
            and isinstance(tgt_elt, IntegerType)
            and tgt_elt.signedness.data.name == "SIGNLESS"
        )
        if not (f32_to_bf16 or int_to_int):
            raise VerifyException(
                "tpu.pack_elementwise: Only packing f32 -> bf16 and integer -> integer is supported"
            )

        tgt_bw = _get_element_bitwidth(tgt_elt)
        if src_bw is not None and tgt_bw is not None and tgt_bw > 0:
            packing_factor = src_bw // tgt_bw
            if packing_factor != len(self.sources):
                raise VerifyException(
                    f"tpu.pack_elementwise: The number of sources must match the packing factor ({packing_factor}), got "
                    f"{len(self.sources)}"
                )


@irdl_op_definition
class UnpackElementwiseOp(IRDLOperation):
    name = "tpu.unpack_elementwise"
    source = operand_def(VectorType.constr(i32))
    source_type = attr_def(TypeAttribute)
    index = attr_def(IntegerAttr[I32])
    output = result_def(
        VectorType.constr(AnyOf((EqAttrConstraint(f32), EqAttrConstraint(i32))))
    )

    traits = traits_def(Pure())

    assembly_format = (
        "$source `,` $index attr-dict `:` type($source) `->` type($output)"
    )

    def __init__(
        self,
        source: SSAValue | Operation,
        source_type: TypeAttribute,
        index: int | IntegerAttr[IntegerType],
        result_type: Attribute,
    ):
        if isinstance(index, int):
            index = IntegerAttr(index, i32)
        super().__init__(
            operands=[source],
            result_types=[result_type],
            attributes={"source_type": source_type, "index": index},
        )

    def verify_(self) -> None:
        output_ty = self.output.type
        assert isinstance(output_ty, VectorType)

        _verify_elementwise_packing(
            "tpu.unpack_elementwise",
            unpacked_ty=output_ty.element_type,
            packed_ty=self.source_type,
        )

        out_bw = _get_element_bitwidth(output_ty)
        src_ty_bw = _get_element_bitwidth(self.source_type)
        if out_bw is None or src_ty_bw is None or src_ty_bw == 0:
            return

        packing_factor = out_bw // src_ty_bw
        index_val = self.index.value.data
        if index_val >= packing_factor:
            raise VerifyException(
                f"tpu.unpack_elementwise: Index must be between 0 and the "
                f"packing factor ({packing_factor}), got {index_val}"
            )


@irdl_op_definition
class PackMaskOp(IRDLOperation):
    name = "tpu.pack_vmsk"

    sources = var_operand_def(VectorType.constr(i1))
    positions = attr_def(DenseArrayBase.constr(i32))
    output = result_def(VectorType.constr(i1))

    traits = traits_def(Pure())

    assembly_format = "$sources attr-dict `:` type($sources) `->` type($output)"

    def __init__(
        self,
        sources: Sequence[SSAValue | Operation],
        positions: DenseArrayBase | Sequence[int],
        result_type: Attribute,
    ):
        if not isinstance(positions, DenseArrayBase):
            positions = DenseArrayBase.from_list(i32, list(positions))
        super().__init__(
            operands=[list(sources)],
            result_types=[result_type],
            attributes={"positions": positions},
        )

    def verify_(self) -> None:
        output_ty = self.output.type
        assert isinstance(output_ty, VectorType)
        first_src_ty = self.sources[0].type if len(self.sources) > 0 else None

        def mask_packing_factor(vty: VectorType) -> int:
            shape = list(vty.get_shape())
            if len(shape) == 2:
                return 1
            if len(shape) == 3:
                return shape[2]
            raise VerifyException(
                f"Mask vector must be 2D or 3D, got rank {len(shape)}: {vty}"
            )

        if first_src_ty is None or not isinstance(first_src_ty, VectorType):
            max_size = 1
        else:
            out_pf = mask_packing_factor(output_ty)
            src_pf = mask_packing_factor(first_src_ty)
            max_size = out_pf // src_pf if src_pf != 0 else 1

        _verify_pack_op("tpu.pack_vmsk", list(self.sources), self.positions, max_size)


@irdl_op_definition
class CreateMaskOp(IRDLOperation):
    name = "tpu.create_mask"
    low = var_operand_def(IndexType)
    high = var_operand_def(IndexType)
    output = result_def()

    irdl_options = (SameVariadicOperandSize(),)
    traits = traits_def(Pure())

    assembly_format = "`[` $low `]``[` $high `]` attr-dict `:` type($output)"

    def __init__(
        self,
        low: Sequence[SSAValue | Operation],
        high: Sequence[SSAValue | Operation],
        result_type: Attribute,
    ):
        super().__init__(operands=[list(low), list(high)], result_types=[result_type])


@irdl_op_definition
class CreateSubelementMaskOp(IRDLOperation):
    name = "tpu.create_subelement_mask"
    from_ = attr_def(IntegerAttr[I32])
    to = attr_def(IntegerAttr[I32])
    output = result_def()

    traits = traits_def(Pure())

    assembly_format = "$from_ `,` $to attr-dict `:` type($output)"

    def __init__(
        self,
        from_value: int | IntegerAttr[IntegerType],
        to: int | IntegerAttr[IntegerType],
        result_type: Attribute,
    ):
        if isinstance(from_value, int):
            from_value = IntegerAttr(from_value, i32)
        if isinstance(to, int):
            to = IntegerAttr(to, i32)
        super().__init__(
            result_types=[result_type], attributes={"from_": from_value, "to": to}
        )


@irdl_op_definition
class SublaneShuffleOp(IRDLOperation):
    name = "tpu.sublane_shuffle"
    lhs = operand_def(VectorType)
    rhs = operand_def(VectorType)
    pattern = attr_def(DenseArrayBase.constr(i32))
    result = result_def(VectorType)

    traits = traits_def(SameOperandsAndResultType())

    assembly_format = "$lhs `,` $rhs `,` $pattern attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)"

    def __init__(
        self,
        lhs: SSAValue | Operation,
        rhs: SSAValue | Operation,
        pattern: DenseArrayBase | Sequence[int],
        result_type: Attribute,
    ):
        if not isinstance(pattern, DenseArrayBase):
            pattern = DenseArrayBase.from_list(i32, list(pattern))
        super().__init__(
            operands=[lhs, rhs],
            result_types=[result_type],
            attributes={"pattern": pattern},
        )

    def verify_(self) -> None:
        lhs_ty = self.lhs.type
        rhs_ty = self.rhs.type
        result_ty = self.result.type
        assert isinstance(lhs_ty, VectorType)
        assert isinstance(rhs_ty, VectorType)
        assert isinstance(result_ty, VectorType)

        if list(lhs_ty.get_shape()) != list(rhs_ty.get_shape()) or list(
            lhs_ty.get_shape()
        ) != list(result_ty.get_shape()):
            raise VerifyException(
                "tpu.sublane_shuffle: Expected lhs, rhs, and result shapes to match"
            )
        if (
            lhs_ty.element_type != rhs_ty.element_type
            or lhs_ty.element_type != result_ty.element_type
        ):
            raise VerifyException(
                "tpu.sublane_shuffle: Expected lhs, rhs, and result element types to match"
            )

        shape = list(result_ty.get_shape())
        if len(shape) < 2 or len(shape) > 3:
            raise VerifyException("tpu.sublane_shuffle: Vreg rank should be 2 or 3")

        sublane_count = shape[0]
        pattern_values = list(self.pattern.get_values())
        if len(pattern_values) != sublane_count:
            raise VerifyException(
                f"tpu.sublane_shuffle: Expected pattern size {len(pattern_values)}) to match result/operand "
                f"sublanes ({sublane_count})"
            )

        total_input_sublanes = sublane_count * 2
        for idx in pattern_values:
            if idx < 0 or idx >= total_input_sublanes:
                raise VerifyException(
                    f"tpu.sublane_shuffle: Pattern index {idx} out of bounds [0, {total_input_sublanes})"
                )
