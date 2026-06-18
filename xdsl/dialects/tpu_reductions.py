from enum import auto

from xdsl.dialects.builtin import (
    I32,
    I64,
    IntegerAttr,
    IntegerType,
    VectorType,
    i1,
    i32,
)
from xdsl.ir import Attribute, EnumAttribute, Operation, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    ParsePropInAttrDict,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    prop_def,
    result_def,
    traits_def,
)
from xdsl.traits import Pure
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.str_enum import StrEnum


class ReductionKind(StrEnum):
    Sum = auto()
    Max = auto()
    Min = auto()
    ArgMax = "arg_max"
    ArgMin = "arg_min"
    FindFirstSet = "find_first_set"


@irdl_attr_definition
class ReductionKindAttr(EnumAttribute[ReductionKind]):
    name = "tpu.reduction_kind"
    enum_type = ReductionKind


def _elem_bitwidth(vec_ty: VectorType) -> int:
    elem = vec_ty.element_type
    if isinstance(elem, IntegerType):
        return elem.width.data
    name = elem.name
    if name == "f16" or name == "bf16":
        return 16
    if name == "f32":
        return 32
    if name == "f64":
        return 64
    if name == "f8E5M2" or name == "f8E4M3":
        return 8
    raise VerifyException(f"Unknown element-type width for {elem}")


def _is_f32(elem_ty: Attribute) -> bool:
    return elem_ty.name == "f32"


def _is_signless_int(elem_ty: Attribute, width: int) -> bool:
    if not isinstance(elem_ty, IntegerType):
        return False
    if elem_ty.width.data != width:
        return False
    from xdsl.dialects.builtin import Signedness

    return elem_ty.signedness.data == Signedness.SIGNLESS


@irdl_op_definition
class AllReduceOp(IRDLOperation):
    name = "tpu.all_reduce"
    input = operand_def(VectorType)
    dim = prop_def(IntegerAttr[I64])
    kind = prop_def(ReductionKindAttr)
    output = result_def(VectorType)

    traits = traits_def(Pure())
    irdl_options = (ParsePropInAttrDict(),)

    assembly_format = "$input attr-dict `:` type($input) `->` type($output)"

    def __init__(
        self,
        input_: SSAValue | Operation,
        dim: int | IntegerAttr[IntegerType],
        kind: ReductionKind | ReductionKindAttr,
        result_type: Attribute,
    ):
        if isinstance(dim, int):
            dim = IntegerAttr(dim, IntegerType(64))
        if isinstance(kind, ReductionKind):
            kind = ReductionKindAttr(kind)
        super().__init__(
            operands=[input_],
            result_types=[result_type],
            properties={"dim": dim, "kind": kind},
        )

    def verify_(self) -> None:
        in_ty = self.input.type
        out_ty = self.output.type
        assert isinstance(in_ty, VectorType)
        assert isinstance(out_ty, VectorType)
        in_bitwidth = _elem_bitwidth(in_ty)
        kind = self.kind.data

        if in_bitwidth == 1:
            if not _is_signless_int(out_ty.element_type, 32):
                raise VerifyException(
                    "tpu.all_reduce: Vector mask all-reduce must have i32 output"
                )
            if kind not in (ReductionKind.Sum, ReductionKind.FindFirstSet):
                raise VerifyException(
                    "tpu.all_reduce: Mask all-reduce only supports sum and find_first_set kinds"
                )
            return

        if kind in (ReductionKind.Sum, ReductionKind.Max, ReductionKind.Min):
            if in_ty != out_ty:
                raise VerifyException(
                    "tpu.all_reduce: Sum, max and min reductions must have the same input and output type"
                )
        elif kind in (ReductionKind.ArgMax, ReductionKind.ArgMin):
            if list(in_ty.get_shape()) != list(out_ty.get_shape()):
                raise VerifyException(
                    "tpu.all_reduce: Arg_max and arg_min must have the same input and output shape"
                )
            if not _is_f32(in_ty.element_type):
                raise VerifyException(
                    "tpu.all_reduce: Not implemented: only f32 input is supported for arg_max and arg_min"
                )
            if not _is_signless_int(out_ty.element_type, in_bitwidth):
                raise VerifyException(
                    f"tpu.all_reduce: Arg_max and arg_min must have i{in_bitwidth} output"
                )
        elif kind == ReductionKind.FindFirstSet:
            raise VerifyException(
                "tpu.all_reduce: Only i1 input is supported for find_first_set"
            )


@irdl_op_definition
class ReduceIndexOp(IRDLOperation):
    name = "tpu.reduce_index"
    input = operand_def(VectorType)
    axis = prop_def(IntegerAttr[I32])
    kind = prop_def(ReductionKindAttr)
    output = result_def(VectorType)

    traits = traits_def(Pure())
    irdl_options = (ParsePropInAttrDict(),)

    assembly_format = "$input attr-dict `:` type($input) `->` type($output)"

    def __init__(
        self,
        input_: SSAValue | Operation,
        axis: int | IntegerAttr[IntegerType],
        kind: ReductionKind | ReductionKindAttr,
        result_type: Attribute,
    ):
        if isinstance(axis, int):
            axis = IntegerAttr(axis, i32)
        if isinstance(kind, ReductionKind):
            kind = ReductionKindAttr(kind)
        super().__init__(
            operands=[input_],
            result_types=[result_type],
            properties={"axis": axis, "kind": kind},
        )

    def verify_(self) -> None:
        in_ty = self.input.type
        out_ty = self.output.type
        assert isinstance(in_ty, VectorType)
        assert isinstance(out_ty, VectorType)

        kind = self.kind.data
        if kind not in (ReductionKind.ArgMax, ReductionKind.ArgMin):
            raise VerifyException(
                "tpu.reduce_index: Reduction kind must be arg_max or arg_min"
            )
        if not _is_f32(in_ty.element_type):
            raise VerifyException(
                "tpu.reduce_index: Not Implemented: Only f32 input is supported for arg_max and arg_min"
            )
        bitwidth = _elem_bitwidth(in_ty)
        if not _is_signless_int(out_ty.element_type, bitwidth):
            raise VerifyException(
                f"tpu.reduce_index: Arg_max and arg_min must have i{bitwidth} output"
            )
        in_shape = list(in_ty.get_shape())
        out_shape = list(out_ty.get_shape())
        axis_val = self.axis.value.data

        if axis_val < 0 or axis_val >= len(in_shape):
            raise VerifyException(
                f"tpu.reduce_index: Axis must be in [0, {len(in_shape)}), but got {axis_val}"
            )

        if len(in_shape) < 2:
            raise VerifyException(
                "tpu.reduce_index: Not Implemented: Only input rank > 1 is supported."
            )
        if len(out_shape) != len(in_shape) - 1:
            raise VerifyException(
                "tpu.reduce_index: Output rank must be one less than input rank"
            )
        out_dim = 0
        for i in range(len(in_shape)):
            if i == axis_val:
                continue
            if in_shape[i] != out_shape[out_dim]:
                raise VerifyException(
                    "tpu.reduce_index: Output shape must match input shape on non-reduction dimensions"
                )
            out_dim += 1


@irdl_op_definition
class ScanOp(IRDLOperation):
    name = "tpu.scan"

    input = operand_def(VectorType)  # [I1, I16, I32, BF16, F32]
    kind = prop_def(ReductionKindAttr)
    mask = opt_operand_def(VectorType.constr(i1))  # [I1]
    output = result_def(VectorType)  # [I16, I32, BF16, F32]

    assembly_format = "$kind `,` $input (`masked` $mask^)? attr-dict `:` type($input) `,` type($mask) `->` type($output)"

    def __init__(
        self,
        input_: SSAValue | Operation,
        kind: ReductionKind | ReductionKindAttr,
        result_type: Attribute,
        mask: SSAValue | Operation | None = None,
    ):
        if isinstance(kind, ReductionKind):
            kind = ReductionKindAttr(kind)
        mask_list: list[SSAValue | Operation] = [mask] if mask is not None else []

        super().__init__(
            operands=[input_, mask_list],
            result_types=[result_type],
            properties={"kind": kind},
        )

    def verify_(self) -> None:
        in_ty = self.input.type
        out_ty = self.output.type
        assert isinstance(in_ty, VectorType)
        assert isinstance(out_ty, VectorType)

        kind = self.kind.data

        if _is_signless_int(in_ty.element_type, 1):
            if not _is_signless_int(out_ty.element_type, 32):
                raise VerifyException(
                    "tpu.scan: Output element type must be i32 vector for i1 vector inputs."
                )
        else:
            if in_ty.element_type != out_ty.element_type:
                raise VerifyException(
                    "tpu.scan: Input and output element type mismatch"
                )

        if list(in_ty.get_shape()) != list(out_ty.get_shape()):
            raise VerifyException("tpu.scan: Input and output shape mismatch.")

        if len(in_ty.get_shape()) > 2:
            raise VerifyException("tpu.scan: Input must be a rank 1 or 2 vector.")

        if _is_signless_int(in_ty.element_type, 1):
            if kind != ReductionKind.Sum:
                raise VerifyException(
                    "tpu.scan: Only sum reduction is supported for i1 vector inputs."
                )
        else:
            if kind not in (ReductionKind.Sum, ReductionKind.Max, ReductionKind.Min):
                raise VerifyException(
                    "tpu.scan: Only sum, max and min reductions are supported."
                )

        if self.mask is not None:
            if _is_signless_int(in_ty.element_type, 1):
                raise VerifyException(
                    "tpu.scan: Mask is not supported for i1 vector inputs."
                )
            mask_ty = self.mask.type
            assert isinstance(mask_ty, VectorType)
            if len(mask_ty.get_shape()) != 1:
                raise VerifyException("tpu.scan: Mask must be a rank 1 vector.")
            expected = in_ty.get_shape()[len(in_ty.get_shape()) - 1]
            if mask_ty.get_shape()[0] != expected:
                raise VerifyException(
                    f"tpu.scan: Mask and input mismatch. Expected mask of length: {expected}, but got {mask_ty.get_shape()[0]}."
                )
