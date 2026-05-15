
from enum import auto

from xdsl.dialects.builtin import AnyFloatConstr, BoolAttr, Float32Type, IntegerType, VectorType
from xdsl.ir.core import Attribute, EnumAttribute, Operation, SSAValue, SpacedOpaqueSyntaxAttribute
from xdsl.irdl.attributes import irdl_attr_definition
from xdsl.irdl.constraints import AnyOf, BaseAttr
from xdsl.irdl.operations import IRDLOperation, attr_def, irdl_op_definition, operand_def, result_def, traits_def
from xdsl.traits import Pure, SameOperandsAndResultType
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.str_enum import StrEnum


_AnyFloatLike = AnyOf((
    AnyFloatConstr,
    VectorType.constr(element_type=AnyFloatConstr)
))

_AnySignlessIntegerLike = AnyOf((
    BaseAttr(IntegerType),
    VectorType.constr(element_type=BaseAttr(IntegerType))  
))

class RoundingMode(StrEnum):
    Towards_Zero = auto()
    To_Nearest_Even = auto()

@irdl_attr_definition
class RoundingModeAttr(EnumAttribute[RoundingMode], SpacedOpaqueSyntaxAttribute):
    name = "tpu.rounding_mode"
    enum_type = RoundingMode

@irdl_op_definition
class FPToSIOp(IRDLOperation):
    name = "tpu.fptosi"
    input = operand_def(_AnyFloatLike)
    rounding_mode = attr_def(RoundingModeAttr)
    output = result_def(_AnySignlessIntegerLike)

    traits = traits_def(Pure())

    assembly_format = "$input attr-dict `:` type($input) `->` type($output) "

    def __init__(
        self,
        input_: SSAValue | Operation,
        target_type: Attribute,
        rounding_mode: RoundingMode | RoundingModeAttr
    ):
        if isinstance(rounding_mode, RoundingMode):
            rounding_mode = RoundingModeAttr(rounding_mode)
        super().__init__(
            operands=[input_],
            result_types=[target_type],
            attributes={"rounding_mode": rounding_mode}
        )

        #TODO canonicalizer

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
        rounding_mode: RoundingMode | RoundingModeAttr
    ):
        if isinstance(rounding_mode, RoundingMode):
            rounding_mode = RoundingModeAttr(rounding_mode)
        super().__init__(
            operands=[input],
            result_types=[target_type],
            attributes={"rounding_mode": rounding_mode}
        )

    #TODO: canonicalizer

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
        rounding_mode: RoundingMode | RoundingModeAttr
    ):
        if isinstance(rounding_mode, RoundingMode):
            rounding_mode = RoundingModeAttr(rounding_mode)
        super().__init__(
            operands=[input_],
            result_types=[target_type],
            attributes={"rounding_mode": rounding_mode}
        )

    #TODO: canonicalizer

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
        rounding_mode: RoundingMode | RoundingModeAttr
    ):
        if isinstance(rounding_mode, RoundingMode):
            rounding_mode = RoundingModeAttr(rounding_mode)
        super().__init__(
            operands=[input_],
            result_types=[target_type],
            attributes={"rounding_mode": rounding_mode}
        )

@irdl_op_definition
class ExtFOp(IRDLOperation):
    name = "tpu.extf"
    input = operand_def(_AnyFloatLike)
    out = result_def(_AnyFloatLike)
    
    traits = traits_def(Pure())

    assembly_format = "$input attr-dict `:` type($input) `->` type($out)"

    def __init__(
        self,
        input_: SSAValue | Operation,
        target_type: Attribute
    ):
        super().__init__(
            operands=[input_],
            result_types=[target_type],
        )
    #TODO: fold


@irdl_op_definition
class TruncFOp(IRDLOperation):
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
        rounding_mode: RoundingMode | RoundingModeAttr
    ):
        if isinstance(rounding_mode, RoundingMode):
            rounding_mode = RoundingModeAttr(rounding_mode)
        super().__init__(
            operands=[input_],
            result_types=[target_type],
            attributes={"rounding_mode": rounding_mode}
        )

    #TODO: fold

@irdl_op_definition
class ReciprocalOp (IRDLOperation):
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
        full_range: bool | BoolAttr = True
    ):
        if isinstance(approx, bool):
            approx = BoolAttr.from_bool(approx)
        if isinstance(full_range, bool):
            full_range = BoolAttr.from_bool(full_range)
        super().__init__(
            operands=[input],
            result_types=[target_type],
            attributes={"approx": approx, "full_range": full_range}
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

    def __init__(
        self,
        input_: SSAValue | Operation,
        target_type: Attribute
    ):
        super().__init__(operands=[input_], result_types=[target_type])

    def verify_(self) -> None:
        in_type = self.input.type
        out_type = self.output.type

        if isinstance(in_type, VectorType):
            in_elem: Attribute = in_type.element_type 
            if not isinstance(in_elem, Float32Type):
                raise VerifyException("tpu.weird: Input type must be F32")
            if not isinstance(out_type, VectorType):
                raise VerifyException("tpu.weird: Output must be a vector when input is a vector")
            out_elem: Attribute = out_type.element_type 
            if not (isinstance(out_elem, IntegerType) and out_elem.width.data == 1): 
                raise VerifyException("tpu.weird: Output type must be I1") 
        else:
            if not isinstance(in_type, Float32Type):
                raise VerifyException("tpu:weird: Input type must be F32")
            if not (isinstance(out_type, IntegerType) and out_type.width.data==1):
                raise VerifyException("tpu.weird: Output type must be I1 scalar")
            