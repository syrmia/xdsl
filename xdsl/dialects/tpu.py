from enum import auto
from typing import Sequence

from xdsl.dialects.func import FuncOp
from xdsl.dialects.builtin import (
    AnyFloat,
    AnyFloatConstr,
    ArrayAttr,
    IntegerAttr,
    IntegerType,
    i64
)

from xdsl.ir import (
    Dialect,
    EnumAttribute,
    Operation,
    ParametrizedAttribute,
    SpacedOpaqueSyntaxAttribute,
    TypeAttribute
)

from xdsl.ir.core import Attribute
from xdsl.irdl import (
    irdl_attr_definition,
)

from xdsl.irdl.attributes import param_def
from xdsl.parser.attribute_parser import AttrParser
from xdsl.printer import Printer

from xdsl.utils.hints import isa
from xdsl.utils.str_enum import (
    StrEnum
)

#----------------------------------------------------------
#                            tpu.td
#----------------------------------------------------------

class CoreType(StrEnum):    # strenum jer je tako korisceno u drugim dijalektima u ovom proj
    Tc = auto()
    Sc_Scalar_Subcore = auto()
    Sc_Vector_Subcore = auto()
    # EnumAttribute kojim cemo posle definisati coretpyeattr zahteva strenum sa auto() specificno,
    # ako je korisceno u drugim dijalektima

    # TC = "tc"
    # SC_SCALAR_SUBCORE = "sc_scalar_subcore"
    # SC_VECTOR_SUBCORE = "sc_vector_subcore"

    # def __str__(self) -> str:
    #     return self.value

    # def __repr__(self) -> str:
    #     return self.name

# wrapper za mlir atribute ??
# class CoreTypeAttr(Data[CoreType]):
@irdl_attr_definition
class CoreTypeAttr(EnumAttribute[CoreType], SpacedOpaqueSyntaxAttribute): 
    name = "tpu.core_type"
    enum_type = CoreType
      
    # ovde ubacena helper metoda koja je u td na nivou celog dijalekta
    # ali posto se odnosi bas na coretypeattr dodajem ovde za sad
    @staticmethod
    def from_op(op: Operation) -> CoreType | None:
        attr = op.attributes.get(CoreTypeAttr.name)
        if attr is None:
            if isinstance(op, FuncOp) and op.sym_name.data == "main":
                return CoreType.Tc
            return None
        
        if not isinstance(attr, CoreTypeAttr):
            return None
        
        return attr.data
         
I64ArrayAttr = ArrayAttr[IntegerAttr[IntegerType]]

def _parse_i64_array(parser: AttrParser) -> I64ArrayAttr:
    parser.parse_punctuation("[")
    values: list[IntegerAttr[IntegerType]] = []
    if parser.parse_optional_punctuation("]") is None:
        values.append(IntegerAttr(parser.parse_integer(), i64))
        while parser.parse_optional_punctuation(",") is not None:
            values.append(IntegerAttr(parser.parse_integer(), i64))
        parser.parse_punctuation("]")
    return ArrayAttr(values)

def _print_i64_array(printer: Printer, arr: I64ArrayAttr) -> None:
    printer.print_string("[")
    printer.print_list(arr.data, lambda x: printer.print_string(f"{x.value.data}"))
    printer.print_string("]")

# def TPU_DotDimensionNumbersAttr : TPU_Attr<...> {
@irdl_attr_definition
class DotDimensionNumbersAttr(ParametrizedAttribute):
    name = "tpu.dot_dimension_numbers"

    lhs_contracting_dims: I64ArrayAttr = param_def() #ArrayAttr#[IntAttr]
    rhs_contracting_dims: I64ArrayAttr = param_def() #ArrayAttr#[IntAttr]
    lhs_non_contracting_dims: I64ArrayAttr = param_def() #ArrayAttr#[IntAttr]
    rhs_non_contracting_dims: I64ArrayAttr = param_def() #ArrayAttr | NoneAttr#[IntAttr] | NoneAttr
    output_dim_order: I64ArrayAttr = param_def() #ArrayAttr#[IntAttr]
    lhs_batch_dims: I64ArrayAttr = param_def() #ArrayAttr | NoneAttr#[IntAttr] | NoneAttr
    rhs_batch_dims: I64ArrayAttr = param_def() #ArrayAttr | NoneAttr#[IntAttr] | NoneAttr

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        with parser.in_angle_brackets():
            lhs_contracting = _parse_i64_array(parser)
            parser.parse_punctuation(",")
            rhs_contracting = _parse_i64_array(parser)
            parser.parse_punctuation(",")
            lhs_non_contracting_dims = _parse_i64_array(parser)
            parser.parse_punctuation(",")
            rhs_non_contracting_dims = _parse_i64_array(parser)
            parser.parse_punctuation(",")
            output_dim_order = _parse_i64_array(parser)
            parser.parse_punctuation(",")
            lhs_batch_dims = _parse_i64_array(parser)
            parser.parse_punctuation(",")
            rhs_batch_dims = _parse_i64_array(parser)
        return [
            lhs_contracting,
            rhs_contracting,
            lhs_non_contracting_dims,
            rhs_non_contracting_dims,
            output_dim_order,
            lhs_batch_dims,
            rhs_batch_dims
        ]
    
    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            _print_i64_array(printer, self.lhs_contracting_dims)
            printer.print_string(",")
            _print_i64_array(printer, self.rhs_contracting_dims)
            printer.print_string(",")
            _print_i64_array(printer, self.lhs_non_contracting_dims)
            printer.print_string(",")
            _print_i64_array(printer, self.rhs_non_contracting_dims)
            printer.print_string(",")
            _print_i64_array(printer, self.output_dim_order)
            printer.print_string(",")
            _print_i64_array(printer, self.lhs_batch_dims)
            printer.print_string(",")
            _print_i64_array(printer, self.rhs_batch_dims)
            printer.print_string(",")
    
    # def __init__(
    #     self,
    #     lhs_contracting_dims: ArrayAttr,#[IntAttr],
    #     rhs_contracting_dims: ArrayAttr,#[IntAttr],
    #     lhs_non_contracting_dims: ArrayAttr,#[IntAttr],
    #     rhs_non_contracting_dims: ArrayAttr | NoneAttr = NoneAttr(), #[IntAttr] | NoneAttr = NoneAttr(),
    #     output_dim_order: ArrayAttr = ArrayAttr([]), #[IntAttr] = ArrayAttr([]),
    #     lhs_batch_dims: ArrayAttr | NoneAttr = NoneAttr(),#[IntAttr] | NoneAttr = NoneAttr(),
    #     rhs_batch_dims: ArrayAttr | NoneAttr = NoneAttr()  #[IntAttr] | NoneAttr = NoneAttr(),
    # ):
    #     super().__init__(
    #         lhs_contracting_dims,
    #         rhs_contracting_dims,
    #         lhs_non_contracting_dims,
    #         rhs_non_contracting_dims,
    #         output_dim_order,
    #         lhs_batch_dims,
    #         rhs_batch_dims,
    #     )

# def TPU_Float8EXMYType : TPU_Type<...> {
@irdl_attr_definition
class Float8EXMYType(ParametrizedAttribute, TypeAttribute):
    name = "tpu.float8_exmy"
    underlying_type: AnyFloat = param_def(constraint=AnyFloatConstr)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        with parser.in_angle_brackets():
            pos = parser.pos
            ty = parser.parse_type()
            if not isa(ty, AnyFloat):
                parser.raise_error(
                    "tpu.float8_exmy underlying type must be a float type "
                    f"(got {ty})",
                    pos,
                    parser.pos - 1,)
            return [ty]
        
    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_attribute(self.underlying_type)

    # def __init__(self, underlying_type: AnyFloat):
    #    super().__init__(underlying_type)

TPU = Dialect(
    "tpu",
    [
        # ops
    ],
    [
        CoreTypeAttr,
        DotDimensionNumbersAttr,
        Float8EXMYType,
    ],
    [
        # interface
    ]
)