from collections.abc import Sequence
from enum import auto

from xdsl.dialects.builtin import (
    I32,
    I64,
    AnyFloat,
    AnyFloatConstr,
    ArrayAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    NoneAttr,
    StringAttr,
    VectorType,
    f32,
    i32,
    i64,
)
from xdsl.dialects.tpu_conversions import (
    ExtFOp,
    FPToSIOp,
    FPToUIOp,
    ReciprocalOp,
    RoundingModeAttr,
    SIToFPOp,
    TruncFOp,
    UIToFPOp,
    WeirdOp,
)
from xdsl.dialects.tpu_dma_sem import (
    AllocaSemaphoreOp,
    BarrierOp,
    DeviceIdOp,
    EnqueueDMAOp,
    GetBarrierSemaphoreOp,
    SemaphoreReadOp,
    SemaphoreSignalOp,
    SemaphoreWaitOp,
    WaitDMA2Op,
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
)
from xdsl.dialects.tpu_memref import (
    CoreTypeAttr,
    DMASemaphoreType,
    MemorySpaceAttr,
    MemRefBitcastOp,
    MemRefReshapeOp,
    MemRefSliceOp,
    MemRefSqueezeOp,
    ReinterpretCastOp,
    SemaphoreType,
    TiledLayoutAttr,
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
)
from xdsl.dialects.tpu_shape import (
    BitcastOp,
    BitcastVregOp,
    BroadcastInSublanesOp,
    ConcatenateOp,
    DynamicGatherOp,
    DynamicRotateOp,
    GatherOp,
    IotaOp,
    MaskCastOp,
    RepeatOp,
    ReshapeOp,
    RollVectorsOp,
    RotateOp,
    ScanCountOp,
    UnrollVectorsOp,
)
from xdsl.ir import (
    Attribute,
    Block,
    Dialect,
    EnumAttribute,
    Operation,
    ParametrizedAttribute,
    Region,
    SpacedOpaqueSyntaxAttribute,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AnyOf,
    IRDLOperation,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    param_def,
    region_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.traits import (
    IsTerminator,
    Pure,
    RecursiveMemoryEffect,
    ReturnLike,
    SingleBlockImplicitTerminator,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa
from xdsl.utils.str_enum import StrEnum

I64ArrayAttr = ArrayAttr[IntegerAttr[I64]]


def _parse_i64_array(parser: AttrParser) -> I64ArrayAttr:
    parser.parse_punctuation("[")
    values: list[IntegerAttr[I64]] = []
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


@irdl_attr_definition
class DotDimensionNumbersAttr(ParametrizedAttribute):
    name = "tpu.dot_dimension_numbers"

    lhs_contracting_dims: I64ArrayAttr = param_def()
    rhs_contracting_dims: I64ArrayAttr = param_def()
    lhs_non_contracting_dims: I64ArrayAttr = param_def()
    rhs_non_contracting_dims: I64ArrayAttr | NoneAttr = param_def()
    output_dim_order: I64ArrayAttr = param_def()
    lhs_batch_dims: I64ArrayAttr | NoneAttr = param_def()
    rhs_batch_dims: I64ArrayAttr | NoneAttr = param_def()

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
            rhs_batch_dims,
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
                    f"tpu.float8_exmy underlying type must be a float type (got {ty})",
                    pos,
                    parser.pos - 1,
                )
            return [ty]

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_attribute(self.underlying_type)


class PipelineMode(StrEnum):
    Synchronous = auto()
    Double_Buffered = auto()


@irdl_attr_definition
class PipelineModeAttr(EnumAttribute[PipelineMode], SpacedOpaqueSyntaxAttribute):
    name = "tpu.pipeline_mode"
    enum_type = PipelineMode


class RevisitMode(StrEnum):
    Immediate = auto()
    Any = auto()


@irdl_attr_definition
class RevisitModeAttr(EnumAttribute[RevisitMode], SpacedOpaqueSyntaxAttribute):
    name = "tpu.revisit_mode"
    enum_type = RevisitMode


class DimensionSemantics(StrEnum):
    Parallel = auto()
    Arbitrary = auto()
    Core_Parallel = auto()
    Subcore_Parallel = auto()


@irdl_attr_definition
class DimensionSemanticsAttr(
    EnumAttribute[DimensionSemantics], SpacedOpaqueSyntaxAttribute
):
    name = "tpu.dimension_semantics"
    enum_type = DimensionSemantics


class ContractPrecision(StrEnum):
    Bf16 = auto()
    Fp32 = auto()


@irdl_attr_definition
class ContractPrecisionAttr(
    EnumAttribute[ContractPrecision], SpacedOpaqueSyntaxAttribute
):
    name = "tpu.contract_precision"
    enum_type = ContractPrecision


class PackFormat(StrEnum):
    Compressed = auto()
    Interleaved = auto()


@irdl_attr_definition
class PackFormatAttr(EnumAttribute[PackFormat], SpacedOpaqueSyntaxAttribute):
    name = "tpu.pack_format"
    enum_type = PackFormat


@irdl_op_definition
class YieldOp(IRDLOperation):
    name = "tpu.yield"
    arguments = var_operand_def()

    traits = traits_def(Pure(), ReturnLike(), IsTerminator())

    assembly_format = "attr-dict ($arguments^ `:` type($arguments))?"

    def __init__(
        self,
        *yielded: SSAValue | Operation,
    ):
        super().__init__(operands=[yielded])


@irdl_op_definition
class RegionOp(IRDLOperation):
    name = "tpu.region"
    results_ = var_result_def()
    region = region_def("single_block")

    traits = traits_def(RecursiveMemoryEffect(), SingleBlockImplicitTerminator(YieldOp))

    def __init__(
        self,
        result_types: Sequence[Attribute],
        region: Region | Sequence[Block] | Sequence[Operation],
    ):
        super().__init__(operands=[], result_types=[result_types], regions=[region])

    def verify_(self) -> None:
        for r in self.results_.types:
            if not isinstance(r, (IntegerType, VectorType, IndexType)) and not isa(
                r, AnyFloat
            ):
                raise VerifyException(
                    f"tpu.region result must be a float, int, index or a vector type (got {r})"
                )


@irdl_op_definition
class TraceOp(IRDLOperation):
    name = "tpu.trace"
    message = attr_def(StringAttr)
    level = attr_def(IntegerAttr[I32])
    results_ = var_result_def()
    region = region_def("single_block")

    traits = traits_def(RecursiveMemoryEffect(), SingleBlockImplicitTerminator(YieldOp))

    def __init__(
        self,
        message: str | StringAttr,
        level: int | IntegerAttr[IntegerType],
        result_types: Sequence[Attribute],
        region: Region | Sequence[Block] | Sequence[Operation],
    ):
        if isinstance(message, str):
            message = StringAttr(message)
        if isinstance(level, int):
            level = IntegerAttr(level, i32)
        super().__init__(
            operands=[],
            result_types=[result_types],
            regions=[region],
            attributes={"message": message, "level": level},
        )


@irdl_op_definition
class TraceStartOp(IRDLOperation):
    name = "tpu.trace_start"
    message = attr_def(StringAttr)
    level = attr_def(IntegerAttr[I32])

    def __init__(
        self, message: str | StringAttr, level: int | IntegerAttr[IntegerType]
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

    def __init__(self, value: SSAValue | Operation, label: str | StringAttr):
        if isinstance(label, str):
            label = StringAttr(label)
        super().__init__(operands=[value], attributes={"label": label})


@irdl_op_definition
class DelayOp(IRDLOperation):
    name = "tpu.delay"
    nanos = operand_def(i32)
    assembly_format = "$nanos attr-dict"

    def __init__(self, nanos: SSAValue | Operation):
        super().__init__(operands=[nanos])


TPU = Dialect(
    "tpu",
    [
        RegionOp,
        YieldOp,
        TraceOp,
        TraceStartOp,
        TraceStopOp,
        TraceValueOp,
        DelayOp,
        MemRefSqueezeOp,
        MemRefReshapeOp,
        MemRefBitcastOp,
        MemRefSliceOp,
        ReinterpretCastOp,
        FPToSIOp,
        FPToUIOp,
        SIToFPOp,
        UIToFPOp,
        ExtFOp,
        TruncFOp,
        ReciprocalOp,
        WeirdOp,
        RotateOp,
        DynamicRotateOp,
        IotaOp,
        RepeatOp,
        ReshapeOp,
        BitcastOp,
        BitcastVregOp,
        MaskCastOp,
        ScanCountOp,
        BroadcastInSublanesOp,
        GatherOp,
        DynamicGatherOp,
        ConcatenateOp,
        RollVectorsOp,
        UnrollVectorsOp,
        LoadOp,
        StoreOp,
        VectorLoadOp,
        VectorStoreOp,
        StridedLoadOp,
        StridedStoreOp,
        ShuffledLoadOp,
        ShuffledStoreOp,
        VectorLoadIdxOp,
        VectorStoreIdxOp,
        UnpackSubelementsOp,
        PackSubelementsOp,
        PackElementwiseOp,
        UnpackElementwiseOp,
        PackMaskOp,
        CreateMaskOp,
        CreateSubelementMaskOp,
        SublaneShuffleOp,
        SemaphoreReadOp,
        SemaphoreWaitOp,
        AllocaSemaphoreOp,
        GetBarrierSemaphoreOp,
        BarrierOp,
        SemaphoreSignalOp,
        EnqueueDMAOp,
        WaitDMA2Op,
        DeviceIdOp,
    ],
    [
        CoreTypeAttr,
        DotDimensionNumbersAttr,
        Float8EXMYType,
        PipelineModeAttr,
        RevisitModeAttr,
        DimensionSemanticsAttr,
        ContractPrecisionAttr,
        PackFormatAttr,
        RoundingModeAttr,
        SemaphoreType,
        DMASemaphoreType,
        MemorySpaceAttr,
        TiledLayoutAttr,
    ],
    [
        # interface
    ],
)
