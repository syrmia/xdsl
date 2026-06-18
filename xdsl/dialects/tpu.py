from collections.abc import Sequence
from enum import auto

from xdsl.dialect_interfaces.constant_materialization import (
    ConstantMaterializationInterface,
)
from xdsl.dialects import arith
from xdsl.dialects.builtin import (
    I32,
    AnyFloat,
    AnyFloatConstr,
    IndexType,
    IntegerAttr,
    IntegerType,
    StringAttr,
    VectorType,
    f32,
    i32,
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
from xdsl.dialects.tpu_matmul import (
    ContractPrecisionAttr,
    DotDimensionNumbersAttr,
    MatmulAccLhsOp,
    MatmulOp,
    MatmulPopOp,
    MatmulPushRhsOp,
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
    EraseLayoutOp,
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
from xdsl.dialects.tpu_reductions import (
    AllReduceOp,
    ReduceIndexOp,
    ReductionKindAttr,
    ScanOp,
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

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> DimensionSemantics:
        with parser.in_angle_brackets():
            return super().parse_parameter(parser)

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            super().print_parameter(printer)


class PackFormat(StrEnum):
    Compressed = auto()
    Interleaved = auto()


@irdl_attr_definition
class PackFormatAttr(EnumAttribute[PackFormat], SpacedOpaqueSyntaxAttribute):
    name = "tpu.pack_format"
    enum_type = PackFormat


class TpuConstantMaterializationInterface(ConstantMaterializationInterface):
    def materialize_constant(
        self, value: Attribute, type: Attribute
    ) -> Operation | None:
        return arith.ConstantOp.build(properties={"value": value}, result_types=(type,))


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
        MatmulOp,
        MatmulPushRhsOp,
        MatmulAccLhsOp,
        MatmulPopOp,
        EraseLayoutOp,
        AllReduceOp,
        ReduceIndexOp,
        ScanOp,
    ],
    [
        CoreTypeAttr,
        Float8EXMYType,
        PipelineModeAttr,
        RevisitModeAttr,
        DimensionSemanticsAttr,
        PackFormatAttr,
        RoundingModeAttr,
        SemaphoreType,
        DMASemaphoreType,
        MemorySpaceAttr,
        TiledLayoutAttr,
        ContractPrecisionAttr,
        DotDimensionNumbersAttr,
        ReductionKindAttr,
    ],
    [
        TpuConstantMaterializationInterface(),
    ],
)
