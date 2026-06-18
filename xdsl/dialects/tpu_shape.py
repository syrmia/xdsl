from collections.abc import Sequence

from xdsl.dialects.builtin import (
    I32,
    DenseArrayBase,
    FixedBitwidthType,
    IntegerAttr,
    IntegerType,
    Signedness,
    VectorType,
    i1,
    i32,
)
from xdsl.interfaces import HasFolderInterface
from xdsl.ir import Attribute, Operation, SSAValue
from xdsl.irdl import (
    EqAttrConstraint,
    IRDLOperation,
    attr_def,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.pattern_rewriter import RewritePattern
from xdsl.traits import (
    HasCanonicalizationPatternsTrait,
    Pure,
    SameOperandsAndResultType,
)
from xdsl.utils.exceptions import VerifyException


class BitcastVregHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls):
        from xdsl.transforms.canonicalization_patterns.tpu import (
            BitcastVregChainCollapse,
        )

        return (BitcastVregChainCollapse(),)


class ReshapeHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.tpu import ReshapeOfReshape

        return (ReshapeOfReshape(),)


class UnrollVectorsHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls):
        from xdsl.transforms.canonicalization_patterns.tpu import UnrollOfRollCancel

        return (UnrollOfRollCancel(),)


class DynamicGatherHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls):
        from xdsl.transforms.canonicalization_patterns.tpu import (
            DynamicGatherToBroadcast,
        )

        return (DynamicGatherToBroadcast(),)


si32 = IntegerType(32, Signedness.SIGNED)


@irdl_op_definition
class RotateOp(IRDLOperation):
    name = "tpu.rotate"
    value = operand_def(VectorType)
    amount = attr_def(IntegerAttr[si32])
    dimension = attr_def(IntegerAttr[si32])
    stride = opt_attr_def(IntegerAttr[si32])
    stride_dimension = opt_attr_def(IntegerAttr[si32])

    result = result_def(VectorType)

    traits = traits_def(Pure(), SameOperandsAndResultType())

    assembly_format = (
        "$value `by` $amount `dim` $dimension (`stride` $stride `stride_dim` "
        "$stride_dimension^)? attr-dict `:` type($value) `->` type($result)"
    )

    def __init__(
        self,
        value: SSAValue | Operation,
        amount: int | IntegerAttr[IntegerType],
        dimension: int | IntegerAttr[IntegerType],
        stride: int | IntegerAttr[IntegerType] | None = None,
        stride_dimension: int | IntegerAttr[IntegerType] | None = None,
    ):
        if isinstance(amount, int):
            amount = IntegerAttr(amount, si32)
        if isinstance(dimension, int):
            dimension = IntegerAttr(dimension, si32)
        if isinstance(stride, int):
            stride = IntegerAttr(stride, si32)
        if isinstance(stride_dimension, int):
            stride_dimension = IntegerAttr(stride_dimension, si32)

        result_type = SSAValue.get(value).type
        attrs: dict[str, Attribute] = {"amount": amount, "dimension": dimension}

        if stride is not None:
            attrs["stride"] = stride
        if stride_dimension is not None:
            attrs["stride_dimension"] = stride_dimension

        super().__init__(operands=[value], result_types=[result_type], attributes=attrs)

    def verify_(self) -> None:
        _verify_rotate_ops(self)


@irdl_op_definition
class DynamicRotateOp(IRDLOperation):
    name = "tpu.dynamic_rotate"
    value = operand_def(VectorType)
    amount = operand_def(i32)
    dimension = attr_def(IntegerAttr[si32])
    stride = opt_attr_def(IntegerAttr[si32])
    stride_dimension = opt_attr_def(IntegerAttr[si32])

    result = result_def(VectorType)

    assembly_format = "$value `by` $amount `dim` $dimension attr-dict `:` type($value) `,` type($amount)  `->` type($result)"

    def __init__(
        self,
        value: SSAValue | Operation,
        amount: SSAValue | Operation,
        dimension: int | IntegerAttr[IntegerType],
        stride: int | IntegerAttr[IntegerType] | None = None,
        stride_dimension: int | IntegerAttr[IntegerType] | None = None,
    ):
        if isinstance(dimension, int):
            dimension = IntegerAttr(dimension, si32)
        if isinstance(stride, int):
            stride = IntegerAttr(stride, si32)
        if isinstance(stride_dimension, int):
            stride_dimension = IntegerAttr(stride_dimension, si32)

        result_type = SSAValue.get(value).type
        attrs: dict[str, Attribute] = {"dimension": dimension}

        if stride is not None:
            attrs["stride"] = stride
        if stride_dimension is not None:
            attrs["stride_dimension"] = stride_dimension

        super().__init__(
            operands=[value, amount], result_types=[result_type], attributes=attrs
        )

    def verify_(self) -> None:
        if self.value.type != self.result.type:
            raise VerifyException(
                f"{self.name}: value and result must have the same type"
            )
        _verify_rotate_ops(self)


def _verify_rotate_ops(op: "RotateOp | DynamicRotateOp") -> None:
    vty = op.result.type
    assert isinstance(vty, VectorType)
    rank = len(vty.get_shape())

    dim = op.dimension.value.data
    if rank <= dim or dim < 0:
        raise VerifyException(f"{op.name}: Invalid dimension: {dim}")

    if op.stride is not None:
        stride_val = op.stride.value.data
        if stride_val < 0:
            raise VerifyException(
                f"{op.name}: Rotate stride must be >= 0 if it is specified"
            )

    if op.stride_dimension is not None:
        sd_val = op.stride_dimension.value.data
        if rank <= sd_val or sd_val < 0:
            raise VerifyException(f"{op.name}: Invalid stride dimension: {sd_val}")

    if (op.stride is None) != (op.stride_dimension is None):
        raise VerifyException(
            f"{op.name}: Expected either none or both stride and stride dimensions are present"
        )


@irdl_op_definition
class IotaOp(IRDLOperation):
    name = "tpu.iota"
    dimensions = attr_def(DenseArrayBase)
    output = result_def(VectorType)

    traits = traits_def(Pure())

    def __init__(
        self, dimensions: DenseArrayBase | Sequence[int], result_type: Attribute
    ):
        if not isinstance(dimensions, DenseArrayBase):
            dimensions = DenseArrayBase.from_list(i32, list(dimensions))

        super().__init__(
            result_types=[result_type], attributes={"dimensions": dimensions}
        )

    def verify_(self) -> None:
        out_ty = self.output.type
        assert isinstance(out_ty, VectorType)
        rank = len(out_ty.get_shape())
        dim_values: list[int] = list(self.dimensions.get_values())

        seen: set[int] = set()
        for dim in dim_values:
            if dim < 0 or dim >= rank:
                raise VerifyException(f"tpu.iota: Invalid dimension: {dim}")
            if dim in seen:
                raise VerifyException("tpu.iota: Dimensions must be unique")
            seen.add(dim)


@irdl_op_definition
class ReshapeOp(IRDLOperation, HasFolderInterface):
    name = "tpu.reshape"
    source = operand_def(VectorType)
    result = result_def(VectorType)

    traits = traits_def(Pure(), ReshapeHasCanonicalizationPatternsTrait())

    assembly_format = "$source attr-dict `:` type($source) `->` type($result)"

    def __init__(self, source: SSAValue | Operation, result_type: Attribute):
        super().__init__(operands=[source], result_types=[result_type])

    def verify_(self) -> None:
        src = self.source.type
        dst = self.result.type
        assert isinstance(src, VectorType)
        assert isinstance(dst, VectorType)

        if src.element_type != dst.element_type:
            raise VerifyException("tpu.reshape: element type must match")
        if src.element_count() != dst.element_count():
            raise VerifyException("tpu.reshape: element count must match")

    def fold(self):
        if self.source.type == self.result.type:
            return (self.source,)
        return None


@irdl_op_definition
class RepeatOp(IRDLOperation):
    name = "tpu.repeat"
    source = operand_def(VectorType)
    dimension = attr_def(IntegerAttr[I32])
    times = attr_def(IntegerAttr[I32])
    output = result_def(VectorType)

    traits = traits_def(Pure())

    assembly_format = "$source `,` $dimension `x` $times attr-dict `:` type($source) `->` type($output)"

    def __init__(
        self,
        source: SSAValue | Operation,
        dimension: int | IntegerAttr[IntegerType],
        times: int | IntegerAttr[IntegerType],
        result_type: Attribute,
    ):
        if isinstance(dimension, int):
            dimension = IntegerAttr(dimension, i32)
        if isinstance(times, int):
            times = IntegerAttr(times, i32)
        super().__init__(
            operands=[source],
            result_types=[result_type],
            attributes={"dimension": dimension, "times": times},
        )


@irdl_op_definition
class BitcastOp(IRDLOperation):
    name = "tpu.bitcast"
    input = operand_def(VectorType)
    output = result_def(VectorType)

    traits = traits_def(Pure())

    assembly_format = "$input attr-dict `:` type($input) `->` type($output)"

    def __init__(self, input_: SSAValue | Operation, result_type: Attribute):
        super().__init__(operands=[input_], result_types=[result_type])

    def verify_(self) -> None:
        in_ty = self.input.type
        out_ty = self.output.type
        assert isinstance(in_ty, VectorType)
        assert isinstance(out_ty, VectorType)

        in_elem: Attribute = in_ty.element_type
        out_elem: Attribute = out_ty.element_type
        assert isinstance(in_elem, FixedBitwidthType)
        assert isinstance(out_elem, FixedBitwidthType)

        in_bw = in_elem.bitwidth
        out_bw = out_elem.bitwidth

        in_shape = list(in_ty.get_shape())
        out_shape = list(out_ty.get_shape())

        if in_bw != out_bw:
            if len(in_shape) < 2 or len(out_shape) < 2:
                raise VerifyException(
                    "tpu.bitcast: Not implemented: bitcast between different bitwidths on a 1D vector."
                )
            scaled_in = in_shape.copy()
            scaled_out = out_shape.copy()
            scaled_in[-2] *= in_bw
            scaled_out[-2] *= out_bw

            if scaled_in != scaled_out:
                raise VerifyException(
                    "tpu.bitcast: Expected input and output shapes are the same after multiplying the "
                    "second-minor dimension by the ratio of bitwidths."
                )
        else:
            if in_shape != out_shape:
                raise VerifyException(
                    "tpu.bitcast: Expected input and output shapes are the same when bitwidth does not change."
                )


@irdl_op_definition
class BitcastVregOp(IRDLOperation, HasFolderInterface):
    name = "tpu.bitcast_vreg"
    input = operand_def(VectorType)
    output = result_def(VectorType)

    traits = traits_def(Pure(), BitcastVregHasCanonicalizationPatternsTrait())

    assembly_format = "$input attr-dict `:` type($input) `->` type($output)"

    def __init__(self, input_: SSAValue | Operation, result_type: Attribute):
        super().__init__(operands=[input_], result_types=[result_type])

    def fold(self):
        if self.input.type == self.output.type:
            return (self.input,)
        return None


@irdl_op_definition
class MaskCastOp(IRDLOperation):
    name = "tpu.mask_cast"
    input = operand_def(VectorType)
    result = result_def(VectorType)

    traits = traits_def(Pure())

    assembly_format = "$input attr-dict `:` type($input) `->` type($result)"

    def __init__(self, input_: SSAValue | Operation, result_type: Attribute):
        super().__init__(operands=[input_], result_types=[result_type])

    def verify_(self) -> None:
        in_ty = self.input.type
        out_ty = self.result.type
        assert isinstance(in_ty, VectorType)
        assert isinstance(out_ty, VectorType)

        in_shape = in_ty.get_shape()
        out_shape = out_ty.get_shape()
        if in_shape[:2] != out_shape[:2]:
            raise VerifyException(
                "tpu.mask_cast: leading two dimensions must match"
                f"(input {list(in_shape[:2])}, result {list(out_shape[:2])})"
            )


@irdl_op_definition
class ScanCountOp(IRDLOperation):
    name = "tpu.scan_count"
    in_mask = operand_def(VectorType.constr(element_type=EqAttrConstraint(i1)))
    values = operand_def(VectorType)
    out_mask = result_def(VectorType.constr(element_type=EqAttrConstraint(i1)))
    counts = result_def(VectorType.constr(element_type=EqAttrConstraint(i32)))

    traits = traits_def(Pure())

    assembly_format = (
        "`mask` `(` $in_mask `:` type($in_mask) `)` `value` `(` $values "
        "`:` type($values) `)` attr-dict `:` type(results)"
    )

    def __init__(
        self,
        in_mask: SSAValue | Operation,
        values: SSAValue | Operation,
        out_mask_type: Attribute | None = None,
        counts_type: Attribute | None = None,
    ):
        if out_mask_type is None:
            out_mask_type = SSAValue.get(in_mask).type
        if counts_type is None:
            values_type = SSAValue.get(values).type
            assert isinstance(values_type, VectorType)
            counts_type = VectorType(i32, values_type.get_shape())
        super().__init__(
            operands=[in_mask, values], result_types=[out_mask_type, counts_type]
        )

    def verify_(self) -> None:
        in_mask_ty = self.in_mask.type
        values_ty = self.values.type
        out_mask_ty = self.out_mask.type
        counts_ty = self.counts.type
        assert isinstance(in_mask_ty, VectorType)
        assert isinstance(values_ty, VectorType)
        assert isinstance(out_mask_ty, VectorType)
        assert isinstance(counts_ty, VectorType)

        ref_shape = list(in_mask_ty.get_shape())
        for name, ty in (
            ("values", values_ty),
            ("out_mask", out_mask_ty),
            ("counts", counts_ty),
        ):
            if list(ty.get_shape()) != ref_shape:
                raise VerifyException(
                    f"tpu.scan_count: {name} shape {list(ty.get_shape())} does not match in_mask shape {ref_shape}"
                )


@irdl_op_definition
class BroadcastInSublanesOp(IRDLOperation):
    name = "tpu.broadcast_in_sublanes"
    source = operand_def(VectorType)
    lane = attr_def(IntegerAttr[I32])
    output = result_def(VectorType)

    traits = traits_def(Pure())

    assembly_format = "$source `,` $lane attr-dict `:` type($source) `->` type($output)"

    def __init__(
        self,
        source: SSAValue | Operation,
        lane: int | IntegerAttr[IntegerType],
        result_type: Attribute,
    ):
        if isinstance(lane, int):
            lane = IntegerAttr(lane, i32)
        super().__init__(
            operands=[source], result_types=[result_type], attributes={"lane": lane}
        )


@irdl_op_definition
class GatherOp(IRDLOperation):
    name = "tpu.gather"
    source = operand_def(VectorType)
    indices = attr_def(DenseArrayBase.constr(i32))
    dimension = attr_def(IntegerAttr[I32])
    output = result_def(VectorType)

    traits = traits_def(Pure())

    assembly_format = "$source `[` $indices `]` `in` $dimension attr-dict `:` type($source) `->` type($output)"

    def __init__(
        self,
        source: SSAValue | Operation,
        indices: DenseArrayBase | Sequence[int],
        dimension: int | IntegerAttr[IntegerType],
        result_type: Attribute,
    ):
        if not isinstance(indices, DenseArrayBase):
            indices = DenseArrayBase.from_list(i32, list(indices))
        if isinstance(dimension, int):
            dimension = IntegerAttr(dimension, i32)
        super().__init__(
            operands=[source],
            result_types=[result_type],
            attributes={"indices": indices, "dimension": dimension},
        )


@irdl_op_definition
class DynamicGatherOp(IRDLOperation):
    name = "tpu.dynamic_gather"
    source = operand_def(VectorType)
    indices = operand_def(VectorType)
    dimensions = attr_def(DenseArrayBase.constr(i32))
    output = result_def(VectorType)

    traits = traits_def(Pure(), DynamicGatherHasCanonicalizationPatternsTrait())

    assembly_format = (
        "$source `[` $indices `]` `in` $dimensions attr-dict `:` "
        "type($source) `,` type($indices) `->` type($output)"
    )

    def __init__(
        self,
        source: SSAValue | Operation,
        indices: SSAValue | Operation,
        dimensions: DenseArrayBase | Sequence[int],
        result_type: Attribute | None = None,
    ):
        if not isinstance(dimensions, DenseArrayBase):
            dimensions = DenseArrayBase.from_list(i32, list(dimensions))
        if result_type is None:
            source_ty = SSAValue.get(source).type
            indices_ty = SSAValue.get(indices).type
            assert isinstance(source_ty, VectorType)
            assert isinstance(indices_ty, VectorType)
            result_type = VectorType(source_ty.element_type, indices_ty.get_shape())
        super().__init__(
            operands=[source, indices],
            result_types=[result_type],
            attributes={"dimensions": dimensions},
        )

    def verify_(self) -> None:
        source_ty = self.source.type
        output_ty = self.output.type
        assert isinstance(source_ty, VectorType)
        assert isinstance(output_ty, VectorType)
        rank = len(source_ty.get_shape())

        dim_values: list[int] = list(self.dimensions.get_values())
        seen: set[int] = set()
        for d in dim_values:
            if d < 0 or d >= rank:
                raise VerifyException(
                    f"tpu.dynamic_gather: Dimensions must be in [0, rank) but got {d}"
                )
            if d in seen:
                raise VerifyException("tpu.dynamic_gather: Dimensions must be unique")
            seen.add(d)

        source_shape = source_ty.get_shape()
        output_shape = output_ty.get_shape()
        if len(source_shape) != len(output_shape):
            raise VerifyException(
                "tpu.dynamic_gather: Source and result shapes must have the same rank"
            )

        for i in range(len(source_shape)):
            if i not in seen and source_shape[i] != output_shape[i]:
                raise VerifyException(
                    "tpu.dynamic_gather: Source and result shpaes must match on non-gather dimensions"
                )


@irdl_op_definition
class ConcatenateOp(IRDLOperation):
    name = "tpu.concatenate"
    sources = var_operand_def(VectorType)
    dimension = attr_def(IntegerAttr[I32])
    output = result_def(VectorType)

    traits = traits_def(Pure())

    assembly_format = (
        "$sources `in` $dimension attr-dict `:` type($sources) `->` type($output)"
    )

    def __init__(
        self,
        sources: Sequence[SSAValue | Operation],
        dimension: int | IntegerAttr[IntegerType],
        result_type: Attribute | None = None,
    ):
        if isinstance(dimension, int):
            dimension = IntegerAttr(dimension, i32)
        if result_type is None:
            dim_int = dimension.value.data
            source_values = [SSAValue.get(s) for s in sources]
            first_ty = source_values[0].type
            assert isinstance(first_ty, VectorType)
            result_shape = list(first_ty.get_shape())
            for v in source_values[1:]:
                t = v.type
                assert isinstance(t, VectorType)
                result_shape[dim_int] += t.get_shape()[dim_int]
            result_type = VectorType(first_ty.element_type, result_shape)
        super().__init__(
            operands=[list(sources)],
            result_types=[result_type],
            attributes={"dimension": dimension},
        )

    def verify_(self) -> None:
        if len(self.sources) < 2:
            raise VerifyException(
                "tpu.concatenate: Expected at least 2 operands for concatenate op."
            )

        first_ty = self.sources[0].type
        assert isinstance(first_ty, VectorType)
        first_shape = first_ty.get_shape()
        first_dtype = first_ty.element_type
        dimension = self.dimension.value.data

        for operand in self.sources:
            vty = operand.type
            assert isinstance(vty, VectorType)
            shape = vty.get_shape()
            if vty.element_type != first_dtype:
                raise VerifyException(
                    "tpu.concatenate: Not implemented: Expected all operands to have the same element type."
                )

            for dim in range(len(shape)):
                if dim != dimension and shape[dim] != first_shape[dim]:
                    raise VerifyException(
                        "tpu.concatenate: Not implemented: Expected all operands "
                        "to have the same shape outside of the concat dim."
                    )


@irdl_op_definition
class RollVectorsOp(IRDLOperation):
    name = "tpu.roll_vectors"
    input = var_operand_def(VectorType)
    output = result_def(VectorType)

    traits = traits_def(Pure())

    assembly_format = "$input attr-dict `:` type($input) `->` type($output)"

    def __init__(self, input_: Sequence[SSAValue | Operation], result_type: Attribute):
        super().__init__(operands=[list(input_)], result_types=[result_type])


@irdl_op_definition
class UnrollVectorsOp(IRDLOperation):
    name = "tpu.unroll_vectors"
    input = operand_def(VectorType)
    output = var_result_def(VectorType)

    traits = traits_def(Pure(), UnrollVectorsHasCanonicalizationPatternsTrait())

    assembly_format = "$input attr-dict `:` type($input) `->` type($output)"

    def __init__(self, input_: SSAValue | Operation, result_type: Sequence[Attribute]):
        super().__init__(operands=[input_], result_types=[list(result_type)])


@irdl_op_definition
class TransposeOp(IRDLOperation):
    name = "tpu.transpose"
    vector = operand_def(VectorType)
    permutation = attr_def(DenseArrayBase)
    result = result_def(VectorType)

    traits = traits_def(Pure())

    assembly_format = (
        " $vector `,` $permutation attr-dict `:` type($vector) `->` type($result)"
    )

    def __init__(
        self,
        vector: SSAValue | Operation,
        permutation: DenseArrayBase | Sequence[int],
        result_type: Attribute,
    ):
        if not isinstance(permutation, DenseArrayBase):
            permutation = DenseArrayBase.from_list(IntegerType(64), list(permutation))
        super().__init__(
            operands=[vector],
            result_types=[result_type],
            attributes={"permutation": permutation},
        )

    def verify_(self) -> None:
        source_ty = self.vector.type
        result_ty = self.result.type
        assert isinstance(source_ty, VectorType)
        assert isinstance(result_ty, VectorType)

        if source_ty.element_type != result_ty.element_type:
            raise VerifyException(
                "tpu.transpose: Expected input and output element types to match"
            )

        permutation_values: list[int] = list(self.permutation.get_values())
        rank = len(source_ty.get_shape())

        if len(permutation_values) != rank:
            raise VerifyException(
                "tpu.transpose: Expected permutation rank to match input rank"
            )
        if len(permutation_values) != len(result_ty.get_shape()):
            raise VerifyException(
                "tpu.transpose: Expected permutation rank to match output rank"
            )

        seen: set[int] = set()
        for dim in permutation_values:
            if dim < 0 or dim >= rank:
                raise VerifyException(
                    f"tpu.transpose: Permutation element out of bounds: {dim}"
                )
            if dim in seen:
                raise VerifyException(
                    f"tpu.transpose: Permutation element repeated: {dim}"
                )
            seen.add(dim)

        input_shape = list(source_ty.get_shape())
        output_shape = list(result_ty.get_shape())
        for i in range(rank):
            if input_shape[permutation_values[i]] != output_shape[i]:
                raise VerifyException(
                    "tpu.transpose: Expected input shape permuted by the given permutation to match the output shape"
                )
