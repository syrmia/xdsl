from enum import auto
from typing import Sequence

from xdsl.dialects.func import FuncOp
from xdsl.dialects.builtin import (
    I32,
    AnyFloat,
    AnyFloatConstr,
    ArrayAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    StringAttr,
    VectorType,
    i32,
    f32,
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

from xdsl.ir.core import Attribute, Block, Region, SSAValue
from xdsl.irdl import (
    irdl_attr_definition,
    irdl_op_definition
)

from xdsl.irdl.attributes import param_def
from xdsl.irdl.constraints import AnyOf
from xdsl.irdl.operations import IRDLOperation, attr_def, operand_def, region_def, traits_def, var_operand_def, var_result_def
from xdsl.parser.attribute_parser import AttrParser
from xdsl.printer import Printer

from xdsl.traits import IsTerminator, Pure, RecursiveMemoryEffect, ReturnLike, SingleBlockImplicitTerminator
from xdsl.utils.exceptions import VerifyException
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

#----------------------------------------------------------
#                        tpu_ops.td
#----------------------------------------------------------

#----------------------------------------------------------
#                   enums + attr wrappers(small)
#----------------------------------------------------------

class PipelineMode(StrEnum):
    # pipeline scheduling mode for tpu.region and DMA ops
    Synchronous = auto()
    Double_Buffered = auto()

@irdl_attr_definition
class PipelineModeAttr(EnumAttribute[PipelineMode], SpacedOpaqueSyntaxAttribute):
    name = "tpu.pipeline_mode"
    enum_type = PipelineMode

class RevisitMode(StrEnum):
    # how grid cells should be handled during revisiting
    Immediate = auto()
    Any = auto()

@irdl_attr_definition
class RevisitModeAttr(EnumAttribute[RevisitMode], SpacedOpaqueSyntaxAttribute):
    name = "tpu.revisit_mode"
    enum_type = RevisitMode

class DimensionSemantics(StrEnum):
    # za computovanje Pallas gridova i baratanja sa dimenzijama, istraziti
    Parallel = auto()
    Arbitrary = auto()
    Core_Parallel = auto()
    Subcore_Parallel = auto()

@irdl_attr_definition
class DimensionSemanticsAttr(EnumAttribute[DimensionSemantics], SpacedOpaqueSyntaxAttribute):
    name = "tpu.dimension_semantics"
    enum_type = DimensionSemantics

class ContractPrecision(StrEnum):
    # za preciznost koju ce matrica da koristi
    Bf16 = auto()
    Fp32 = auto()

@irdl_attr_definition
class ContractPrecisionAttr(EnumAttribute[ContractPrecision], SpacedOpaqueSyntaxAttribute):
    name = "tpu.contract_precision"
    enum_type = ContractPrecision

class PackFormat(StrEnum):
    Compressed = auto()
    Interleaved = auto()

@irdl_attr_definition
class PackFormatAttr(EnumAttribute[PackFormat], SpacedOpaqueSyntaxAttribute):
    name = "tpu.pack_format"
    enum_type = PackFormat

class RoundingMode(StrEnum):
    # zaokruzivanje izmedju float i int i floatova medjusobno
    Towards_Zero = auto()
    To_Nearest_Even = auto()

@irdl_attr_definition
class RoundingModeAttr(EnumAttribute[RoundingMode], SpacedOpaqueSyntaxAttribute):
    name = "tpu.rounding_mode"
    enum_type = RoundingMode


#----------------------------------------------------------
#                opaque types(sem and dma)
#----------------------------------------------------------

@irdl_attr_definition
class SemaphoreType(ParametrizedAttribute, TypeAttribute):
    # za sinhronizaciju izmedju tpu operacija
    name = "tpu.semaphore"

@irdl_attr_definition
class DMASemaphoreType(ParametrizedAttribute, TypeAttribute):
    name = "tpu.dma_semaphore"

#----------------------------------------------------------
#                       structural ops
#----------------------------------------------------------
# === proveri, pisano sa poluizgenerisanim clause kodom
@irdl_op_definition
class YieldOp(IRDLOperation):
    name = "tpu.yield"
    arguments = var_operand_def()
    traits = traits_def(
        Pure(),
        ReturnLike(),
        IsTerminator()
    )

    def __init__(
            self,
            *yielded: SSAValue | Operation,
    ):
        super().__init__(operands=[yielded])

    assembly_format = "attr-dict ($arguments^ `:` type($arguments))?"
@irdl_op_definition
class RegionOp(IRDLOperation):
    name = "tpu.region"
    results_ = var_result_def()
    region = region_def("single_block")
    
    traits = traits_def(
        RecursiveMemoryEffect(),
        SingleBlockImplicitTerminator(YieldOp)
    )

    def __init__(
            self, 
            result_types: Sequence[Attribute],
            region: Region | Sequence[Block] | Sequence[Operation]
    ):
        super().__init__(operands=[],
                         result_types=[result_types],
                         regions=[region]
        )

    def verify_(self) -> None:
        for r in self.results_.types:
            if(not isinstance(r, (IntegerType, VectorType, IndexType)) and not isa(r, AnyFloat)):
                raise VerifyException("tpu.region kust be a float, int, index or a"
                                    f"vector type(got {r})")

@irdl_op_definition
class TraceOp(IRDLOperation):
    name = "tpu.trace"
    message = attr_def(StringAttr)
    level = attr_def(IntegerAttr[I32])
    results_ = var_result_def()
    region = region_def("single_block")

    traits = traits_def(
        RecursiveMemoryEffect(),
        SingleBlockImplicitTerminator(YieldOp)
    )

    def __init__(
            self,
            message: str | StringAttr,
            level: int | IntegerAttr[IntegerType],
            result_types: Sequence[Attribute],
            region: Region | Sequence[Block] | Sequence[Operation]
    ):
        if isinstance(message, str):
            message = StringAttr(message)
        if isinstance(level, int):
            level = IntegerAttr(level, i32)
        super().__init__(
            operands=[],
            result_types=[result_types],
            regions=[region],
            attributes={"message": message, "level": level}
        )

@irdl_op_definition
class TraceStartOp(IRDLOperation):
    name = "tpu.trace_start"
    message = attr_def(StringAttr)
    level = attr_def(IntegerAttr[IntegerType])

    def __init__(
            self,
            message: str | StringAttr,
            level: int | IntegerAttr[IntegerType]
    ):
        if isinstance(message, str):
            message = StringAttr(message)
        if isinstance(level, int):
            level = IntegerAttr(level, i32)
        super().__init__(attributes={"message": message, "level": level})

@irdl_op_definition
class TraceStopOp(IRDLOperation):
    name = "tpu.trace_stop"

    def __init__(self):
        super().__init__()
    
@irdl_op_definition
class TraceValueOp(IRDLOperation):
    name = "tpu.trace_value"
    value = operand_def(AnyOf((i32, f32)))
    label = attr_def(StringAttr)

    assembly_format = "$value `,` $label attr-dict `:` type($value)"

    def __init__(
            self,
            value: SSAValue | Operation,
            label: str | StringAttr
    ):
        if isinstance(label, str):
            label = StringAttr(label)
        super().__init__(operands=[value], attributes={"label":label})

@irdl_op_definition
class DelayOp(IRDLOperation):
    name = "tpu.delay"
    nanos = operand_def(i32)
    assembly_format = "$nanos attr-dict"

    def __init__(
            self,
            nanos: SSAValue | Operation
    ):
        super().__init__(operands=[nanos])

#----------------------------------------------------------
#                memref stvari
#----------------------------------------------------------




#----------------------------------------------------------
#                tpu.td(dialect registration)
#----------------------------------------------------------

TPU = Dialect(
    "tpu",
    [
        RegionOp,
        YieldOp,
        TraceOp,
        TraceStartOp,
        TraceStopOp,
        TraceValueOp,
        DelayOp
    ],
    [
        # tpu.td
        CoreTypeAttr,
        DotDimensionNumbersAttr,
        Float8EXMYType,
        # tpu_ops.td
        PipelineModeAttr,
        RevisitModeAttr,
        DimensionSemanticsAttr,
        ContractPrecisionAttr,
        PackFormatAttr,
        RoundingModeAttr,
        SemaphoreType,
        DMASemaphoreType
    ],
    [
        # interface
    ]
)