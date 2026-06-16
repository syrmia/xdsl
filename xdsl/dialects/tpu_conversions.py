from enum import auto

from xdsl.dialects import arith
from xdsl.dialects.builtin import (
    AnyFloatConstr,
    BoolAttr,
    DenseIntOrFPElementsAttr,
    Float32Type,
    FloatAttr,
    IntegerType,
    ShapedType,
    VectorType,
)
from xdsl.interfaces import HasFolderInterface
from xdsl.ir import (
    Attribute,
    EnumAttribute,
    Operation,
    SpacedOpaqueSyntaxAttribute,
    SSAValue,
)
from xdsl.irdl import (
    AnyOf,
    BaseAttr,
    IRDLOperation,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
    traits_def,
)
from xdsl.traits import (
    HasCanonicalizationPatternsTrait,
    Pure,
    SameOperandsAndResultType,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.str_enum import StrEnum

_AnyFloatLike = AnyOf((AnyFloatConstr, VectorType.constr(element_type=AnyFloatConstr)))

_AnySignlessIntegerLike = AnyOf(
    (BaseAttr(IntegerType), VectorType.constr(element_type=BaseAttr(IntegerType)))
)


class RoundingMode(StrEnum):
    Towards_Zero = auto()
    To_Nearest_Even = auto()


@irdl_attr_definition
class RoundingModeAttr(EnumAttribute[RoundingMode], SpacedOpaqueSyntaxAttribute):
    name = "tpu.rounding_mode"
    enum_type = RoundingMode


def _fold_float_conversion(
    input_value: SSAValue, out_type: Attribute
) -> FloatAttr | DenseIntOrFPElementsAttr | None:
    producer = input_value.owner
    if not isinstance(producer, arith.ConstantOp):
        return None
    attr = producer.value

    if isinstance(out_type, ShapedType):
        target_elem_type = out_type.element_type
    else:
        target_elem_type = out_type

    if isinstance(attr, FloatAttr):
        py_val = attr.value.data
        try:
            return FloatAttr(py_val, target_elem_type)
        except (NotImplementedError, ValueError):
            return None

    if isinstance(attr, DenseIntOrFPElementsAttr):
        if not isinstance(out_type, ShapedType):
            return None
        source_elem_type = attr.type.element_type
        n_elements = len(attr)
        try:
            py_values = list(source_elem_type.unpack(attr.data.data, n_elements))
        except (NotImplementedError, ValueError, AttributeError):
            return None
        try:
            return DenseIntOrFPElementsAttr.from_list(out_type, py_values)
        except (NotImplementedError, ValueError):
            return None

    return None


class FPToSIHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls):
        from xdsl.transforms.canonicalization_patterns.tpu import FPToSISinkRoundEven

        return (FPToSISinkRoundEven(),)


@irdl_op_definition
class FPToSIOp(IRDLOperation):
    name = "tpu.fptosi"
    input = operand_def(_AnyFloatLike)
    rounding_mode = attr_def(RoundingModeAttr)
    output = result_def(_AnySignlessIntegerLike)

    traits = traits_def(Pure(), FPToSIHasCanonicalizationPatternsTrait())

    assembly_format = "$input attr-dict `:` type($input) `->` type($output) "

    def __init__(
        self,
        input_: SSAValue | Operation,
        target_type: Attribute,
        rounding_mode: RoundingMode | RoundingModeAttr,
    ):
        if isinstance(rounding_mode, RoundingMode):
            rounding_mode = RoundingModeAttr(rounding_mode)
        super().__init__(
            operands=[input_],
            result_types=[target_type],
            attributes={"rounding_mode": rounding_mode},
        )


@irdl_op_definition
class FPToUIOp(IRDLOperation):
    name = "tpu.fptoui"
    input = operand_def(_AnyFloatLike)
    rounding_mode = attr_def(RoundingModeAttr)
    output = result_def(_AnySignlessIntegerLike)

    traits = traits_def(Pure())

    assembly_format = "$input attr-dict `:` type($input) `->` type($output)"

    def __init__(
        self,
        input: SSAValue | Operation,
        target_type: Attribute,
        rounding_mode: RoundingMode | RoundingModeAttr,
    ):
        if isinstance(rounding_mode, RoundingMode):
            rounding_mode = RoundingModeAttr(rounding_mode)
        super().__init__(
            operands=[input],
            result_types=[target_type],
            attributes={"rounding_mode": rounding_mode},
        )


@irdl_op_definition
class SIToFPOp(IRDLOperation):
    name = "tpu.sitofp"
    input = operand_def(_AnySignlessIntegerLike)
    rounding_mode = attr_def(RoundingModeAttr)
    output = result_def(_AnyFloatLike)

    traits = traits_def(Pure())

    assembly_format = "$input attr-dict `:` type($input) `->` type($output)"

    def __init__(
        self,
        input_: SSAValue | Operation,
        target_type: Attribute,
        rounding_mode: RoundingMode | RoundingModeAttr,
    ):
        if isinstance(rounding_mode, RoundingMode):
            rounding_mode = RoundingModeAttr(rounding_mode)
        super().__init__(
            operands=[input_],
            result_types=[target_type],
            attributes={"rounding_mode": rounding_mode},
        )


@irdl_op_definition
class UIToFPOp(IRDLOperation):
    name = "tpu.uitofp"
    input = operand_def(_AnySignlessIntegerLike)
    rounding_mode = attr_def(RoundingModeAttr)
    output = result_def(_AnyFloatLike)

    traits = traits_def(Pure())

    assembly_format = "$input attr-dict `:` type($input) `->` type($output)"

    def __init__(
        self,
        input_: SSAValue | Operation,
        target_type: Attribute,
        rounding_mode: RoundingMode | RoundingModeAttr,
    ):
        if isinstance(rounding_mode, RoundingMode):
            rounding_mode = RoundingModeAttr(rounding_mode)
        super().__init__(
            operands=[input_],
            result_types=[target_type],
            attributes={"rounding_mode": rounding_mode},
        )


@irdl_op_definition
class ExtFOp(IRDLOperation, HasFolderInterface):
    name = "tpu.extf"
    input = operand_def(_AnyFloatLike)
    out = result_def(_AnyFloatLike)

    traits = traits_def(Pure())

    assembly_format = "$input attr-dict `:` type($input) `->` type($out)"

    def __init__(self, input_: SSAValue | Operation, target_type: Attribute):
        super().__init__(
            operands=[input_],
            result_types=[target_type],
        )

    def fold(self):
        new_attr = _fold_float_conversion(self.input, self.out.type)
        if new_attr is None:
            return None
        return (new_attr,)


@irdl_op_definition
class TruncFOp(IRDLOperation, HasFolderInterface):
    name = "tpu.truncf"
    input = operand_def(_AnyFloatLike)
    rounding_mode = attr_def(RoundingModeAttr)
    out = result_def(_AnyFloatLike)

    traits = traits_def(Pure())

    assembly_format = "$input attr-dict `:` type($input) `->` type($out)"

    def __init__(
        self,
        input_: SSAValue | Operation,
        target_type: Attribute,
        rounding_mode: RoundingMode | RoundingModeAttr,
    ):
        if isinstance(rounding_mode, RoundingMode):
            rounding_mode = RoundingModeAttr(rounding_mode)
        super().__init__(
            operands=[input_],
            result_types=[target_type],
            attributes={"rounding_mode": rounding_mode},
        )

    def fold(self):
        new_attr = _fold_float_conversion(self.input, self.out.type)
        if new_attr is None:
            return None
        return (new_attr,)


@irdl_op_definition
class ReciprocalOp(IRDLOperation):
    name = "tpu.reciprocal"
    input = operand_def(VectorType.constr(element_type=AnyFloatConstr))
    approx = attr_def(BoolAttr)
    full_range = attr_def(BoolAttr)
    output = result_def(VectorType.constr(element_type=AnyFloatConstr))

    traits = traits_def(Pure(), SameOperandsAndResultType())

    assembly_format = "$input attr-dict `:` type($input) `->` type($output)"

    def __init__(
        self,
        input: SSAValue | Operation,
        target_type: Attribute,
        approx: bool | BoolAttr = False,
        full_range: bool | BoolAttr = True,
    ):
        if isinstance(approx, bool):
            approx = BoolAttr.from_bool(approx)
        if isinstance(full_range, bool):
            full_range = BoolAttr.from_bool(full_range)
        super().__init__(
            operands=[input],
            result_types=[target_type],
            attributes={"approx": approx, "full_range": full_range},
        )

    def verify_(self) -> None:
        out_type = self.output.type
        assert isinstance(out_type, VectorType)
        elem_type: Attribute = out_type.element_type
        if not isinstance(elem_type, Float32Type):
            raise VerifyException(
                "tpu.reciprocal: Not implemented: Reciprocal op for non-f32 dtypes"
            )


@irdl_op_definition
class WeirdOp(IRDLOperation):
    name = "tpu.weird"
    input = operand_def()
    output = result_def()

    traits = traits_def(Pure())

    assembly_format = "$input attr-dict `:` type($input) `->` type($output) "

    def __init__(self, input_: SSAValue | Operation, target_type: Attribute):
        super().__init__(operands=[input_], result_types=[target_type])

    def verify_(self) -> None:
        in_type = self.input.type
        out_type = self.output.type

        if isinstance(in_type, VectorType):
            in_elem: Attribute = in_type.element_type
            if not isinstance(in_elem, Float32Type):
                raise VerifyException("tpu.weird: Input type must be F32")
            if not isinstance(out_type, VectorType):
                raise VerifyException(
                    "tpu.weird: Output must be a vector when input is a vector"
                )
            out_elem: Attribute = out_type.element_type
            if not (isinstance(out_elem, IntegerType) and out_elem.width.data == 1):
                raise VerifyException("tpu.weird: Output type must be I1")
        else:
            if not isinstance(in_type, Float32Type):
                raise VerifyException("tpu:weird: Input type must be F32")
            if not (isinstance(out_type, IntegerType) and out_type.width.data == 1):
                raise VerifyException("tpu.weird: Output type must be I1 scalar")
