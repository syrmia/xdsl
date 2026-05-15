
#----------------------------------------------------
#                memref 
#----------------------------------------------------

from enum import auto
from typing import Sequence

from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    FixedBitwidthType,
    MemRefType,
    NoneAttr,
    i32
)
from xdsl.dialects.func import FuncOp

from xdsl.ir.core import Attribute, Data, EnumAttribute, Operation, ParametrizedAttribute, SpacedOpaqueSyntaxAttribute, TypeAttribute
from xdsl.irdl.attributes import irdl_attr_definition, param_def

from xdsl.irdl.operations import (
    AttrSizedOperandSegments,
    IRDLOperation,
    irdl_op_definition,
    operand_def, 
    opt_operand_def,
    result_def,
    traits_def,
    var_operand_def
)
from xdsl.parser.attribute_parser import AttrParser
from xdsl.printer import Printer
from xdsl.traits import Pure
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.str_enum import StrEnum


class CoreType(StrEnum):
    Tc = auto()
    Sc_Scalar_Subcore = auto()
    Sc_Vector_Subcore = auto()
@irdl_attr_definition
class CoreTypeAttr(EnumAttribute[CoreType], SpacedOpaqueSyntaxAttribute): 
    name = "tpu.core_type"
    enum_type = CoreType

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


@irdl_attr_definition
class SemaphoreType(ParametrizedAttribute, TypeAttribute):
    name = "tpu.semaphore"

@irdl_attr_definition
class DMASemaphoreType(ParametrizedAttribute, TypeAttribute):
    name = "tpu.dma_semaphore"
    
class MemorySpace(StrEnum):
    Any = auto()
    Vmem = auto()
    Smem = auto()
    Hbm = auto()
    Cmem = auto()
    Semaphore_Mem = auto()
    Vmem_Shared = auto()
    Host = auto()


# wrapper za mem space enum, da se koristi kao parametar unutar mem space attr
@irdl_attr_definition
class _MemorySpaceData(Data[MemorySpace]):
    name = "tpu.memory_space_value"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> MemorySpace:
        return parser.parse_str_enum(MemorySpace)
    
    def print_parameter(self, printer: Printer) -> None:
        printer.print_identifier_or_string_literal(str(self.data))

# memref ops

@irdl_attr_definition
class MemorySpaceAttr(ParametrizedAttribute):
    name = "tpu.memory_space"

    value: _MemorySpaceData = param_def()
    core_type: CoreTypeAttr | NoneAttr = param_def()

    def __init__(
        self,
        value: MemorySpace | _MemorySpaceData,
        core_type: CoreType | CoreTypeAttr | None = None
    ):
        if isinstance(value, MemorySpace):
            value = _MemorySpaceData(value)

        core_type_attr: Attribute
        if core_type is None:
            core_type_attr = NoneAttr()
        elif isinstance(core_type, CoreType):
            core_type_attr = CoreTypeAttr(core_type)
        else:
            core_type_attr = core_type

        super().__init__(value, core_type_attr)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        with parser.in_angle_brackets():
            value_enum = _MemorySpaceData.parse_parameter(parser)
            value = _MemorySpaceData(value_enum)

            core_type: Attribute = NoneAttr()
            if parser.parse_optional_punctuation(",") is not None:
                core_type_enum = CoreTypeAttr.parse_parameter(parser)
                core_type = CoreTypeAttr(core_type_enum)
            
        return[value, core_type]
    
    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            self.value.print_parameter(printer)
            if not isinstance(self.core_type, NoneAttr):
                printer.print_string(", ")
                self.core_type.print_parameter(printer)



def _check_memref_memory_spaces_match(
    # za MemRefSqueezeOp::verify, MemRefReshapeOp::verify, MemRefBitcastOp::verify
    # provera da li se target memspace razlikuje od source 
    op_name: str,
    source: MemRefType,
    target: MemRefType
) -> None:
    target_mem = target.memory_space
    if isinstance(target_mem, NoneAttr):
        return
    if target_mem != source.memory_space:
        raise VerifyException(f"{op_name}: Memory spaces do not match")

def _check_semaphore_element_type(
    op_name: str,
    mem_ref_type: MemRefType
) -> None:
    mem_sp = mem_ref_type.memory_space
    if not isinstance(mem_sp, MemorySpaceAttr):
        return
    if mem_sp.value.data != MemorySpace.Semaphore_Mem:
        return
    elem = mem_ref_type.element_type
    if not isinstance(elem, (SemaphoreType, DMASemaphoreType)):
        raise VerifyException(
            f"{op_name}: References to semaphore memory space must have a semaphore element type."
        )
    

def _compute_squeezed_dims(
    op_name: str,
    source_shape: Sequence[int],
    target_shape: Sequence[int],
) -> list[int]:
    squeezed: list[int] = []
    si, ti = 0, 0
    src = list(source_shape)
    tgt = list(target_shape)

    while si < len(src) and ti < len(tgt):
        if src[si] == tgt[ti]:
            si += 1
            ti += 1
        elif src[si] == 1:
            squeezed.append(si)
            si += 1
        else:
            raise VerifyException(
                f"{op_name}: Source and target shapes are not compatible for "
                f"squeezing: source dim {si} = {src[si]} vs target dim {ti} = {tgt[ti]}"
            )

    while si < len(src):
        if src[si] != 1:
            raise VerifyException(
                f"{op_name}: Cannot squeeze non-unit source dim {si} (size {src[si]})"
            )
        squeezed.append(si)
        si += 1

    if ti < len(tgt):
        raise VerifyException(
            f"{op_name}: Target shape has more dimensions than source can provide via squeezing"
        )

    return squeezed

@irdl_op_definition
class MemRefSliceOp(IRDLOperation):
    name = "tpu.memref_slice"
    mem_ref = operand_def(MemRefType)
    base_idx = var_operand_def(i32)
    dynamic_sizes =  var_operand_def(i32)
    result = result_def(MemRefType)

    irdl_options = (AttrSizedOperandSegments(),)
    # opcije: kako se IR struktura enkoduje i parsira, printuje
    # u ovom slucaju imamo vise variadic operanada, kroz attr sized op seg pratimo njihove velicine 
    traits = traits_def(Pure())
    # traitovi su za semantiku, sta operacija znaci i radi, dok su irdl opcije za to
    # traitovi ucestvuju u optimiacijama
    # kako se operacija predstavlja

    assembly_format = (
        "$mem_ref `[` $base_idx `]` (`<` $dynamic_sizes^ `>`)? attr-dict `:` type($mem_ref) `->` type($result)"
    )

    def verify_(self) -> None:
        source_type = self.mem_ref.type
        target_type = self.result.type
        assert isinstance(source_type, MemRefType)
        assert isinstance(target_type, MemRefType)

        if not source_type.has_static_shape():
            raise VerifyException("tpu.memref_slice: Only slicing of memrefs with static shapes is supported.")
        
        target_dynamic_dim_count = sum(
            1 for d in target_type.get_shape() if d == DYNAMIC_INDEX
        )

        if len(self.dynamic_sizes) != target_dynamic_dim_count:
            raise VerifyException(
                "tpu.memref_slice: Number of provided dynamic dimensions sizes must match the number of dynamic dimensions in the target type"
            )
        
        _check_semaphore_element_type("tpu.memref_slice", source_type)
        
        source_shape = source_type.get_shape()
        slice_shape = target_type.get_shape()
        if (
            len(self.base_idx) != len(slice_shape)
            or len(self.base_idx) != len(source_shape)
        ):
            raise VerifyException(
                "tpu.memref_slice: Indices and slice shapes must match."
            )
 
        _check_memref_memory_spaces_match(
            "tpu.memref_slice", source_type, target_type
        )

        src_layout = source_type.layout
        tgt_layout = target_type.layout
        if not isinstance(src_layout, NoneAttr) or not isinstance(tgt_layout, NoneAttr):
            raise VerifyException(
                "tpu.memref_slice: Not implemented: slice with non-identity "
                "layouts (TiledLayoutAttr support is pending)."
            )

        # TODO fold, canonicalizer
        # TODO: TiledLayoutAttr provere u verify_, kada se TiledLayoutAttr bude odradio


@irdl_op_definition
class MemRefSqueezeOp(IRDLOperation):
    name = "tpu.memref_squeeze"
    input = operand_def(MemRefType)
    result = result_def(MemRefType)
 
    traits = traits_def(Pure())
 
    assembly_format = ("$input attr-dict `:` type($input) `->` type($result)")
 
    def verify_(self) -> None:
        source_type = self.input.type
        target_type = self.result.type
        assert isinstance(source_type, MemRefType)
        assert isinstance(target_type, MemRefType)
 
        _check_memref_memory_spaces_match(
            "tpu.memref_squeeze", source_type, target_type 
        )
 
        if source_type.element_type != target_type.element_type: 
            raise VerifyException(
                "tpu.memref_squeeze: Element types don't match."
            )
        
        source_shape = list(source_type.get_shape())
        target_shape = list(target_type.get_shape())
        squeezed = _compute_squeezed_dims("tpu.memref_squeeze", source_shape, target_shape)

        if len(squeezed) == 0 and source_shape != target_shape:
            raise VerifyException(
                "tpu.memref_squeeze: Source and target shapes must be the same if no dimensions are squeezed."
            )

    # TODO TiledLayoutAttr provere, canonicalizer

@irdl_op_definition
class MemRefReshapeOp(IRDLOperation):
    name = "tpu.memref_reshape"
    input = operand_def(MemRefType)
    result = result_def(MemRefType)

    traits = traits_def(Pure())
 
    assembly_format = ("$input attr-dict `:` type($input) `->` type($result)")
 
    def verify_(self) -> None:
        source_type = self.input.type
        target_type = self.result.type
        assert isinstance(source_type, MemRefType)
        assert isinstance(target_type, MemRefType)
 
        _check_memref_memory_spaces_match(
            "tpu.memref_reshape", source_type, target_type 
        )
 
        if (len(source_type.get_shape()) < 2
                or len(target_type.get_shape()) < 2):
            raise VerifyException("tpu.memref_reshape: Not implemented: 1d memref reshape.")
 
        if source_type.element_type != target_type.element_type:
            raise VerifyException("tpu.memref_reshape: Element types don't match.")
 
        src_n = source_type.element_count()
        tgt_n = target_type.element_count()
        if src_n != tgt_n:
            raise VerifyException(
                "tpu.memref_reshape: Number of elements doesn't match between input and output memref type."
            )
        
        src_layout = source_type.layout
        tgt_layout = target_type.layout
        if not isinstance(src_layout, NoneAttr) or not isinstance(tgt_layout, NoneAttr):
            raise VerifyException(
                "tpu.memref_reshape: Not implemented: reshape with non-identity layouts"
                # TODO with tiled layout attr
            )


@irdl_op_definition
class MemRefBitcastOp(IRDLOperation):
    name = "tpu.memref_bitcast"
    input = operand_def(MemRefType)
    result = result_def(MemRefType)
 
    traits = traits_def(Pure())
 
    assembly_format = ("$input attr-dict `:` type($input) `->` type($result)")
 
    def verify_(self) -> None:
        source_type = self.input.type
        target_type = self.result.type
        assert isinstance(source_type, MemRefType)
        assert isinstance(target_type, MemRefType)
 
        _check_memref_memory_spaces_match(
            "tpu.memref_bitcast", source_type, target_type
        )
 
        source_shape = source_type.get_shape()
        target_shape = target_type.get_shape()
 
        if len(source_shape) != len(target_shape):
            raise VerifyException("tpu.memref_bitcast: Ranks do not match.")
 
        if len(source_shape) <= 1:
            raise VerifyException("tpu.memref_bitcast: Not implemented: 1d memref bitcast.")

        src_elem = source_type.element_type 
        tgt_elem = target_type.element_type
        # C++ getElementTypeBitwidth je u xDSL element_type.bitwidth iz FixedBitwidthType
        assert isinstance(src_elem, FixedBitwidthType), (
            f"tpu.memref_bitcast: source element type {src_elem} has no fixed bitwidth"
        )
        assert isinstance(tgt_elem, FixedBitwidthType), (
            f"tpu.memref_bitcast: target element type {tgt_elem} has no fixed bitwidth"
        )
        src_bitwidth = src_elem.bitwidth
        tgt_bitwidth = tgt_elem.bitwidth
 
        rank = len(source_shape)
        second_minormost = rank - 2
        for i in range(rank):
            src_dim = source_shape[i]
            tgt_dim = target_shape[i]
            if i == second_minormost:
                src_bits = src_dim * src_bitwidth
                tgt_bits = tgt_dim * tgt_bitwidth
                if src_bits != tgt_bits:
                    raise VerifyException(
                        f"tpu.memref_bitcast: Expected the same number of bits on the 2nd minormost dim: ({src_dim} * "
                        f"{src_bitwidth}) vs ({tgt_dim} * {tgt_bitwidth})"
                    )
            else:
                if src_dim != tgt_dim:
                    raise VerifyException(
                        f"tpu.memref_bitcast: Expected the same dim size on dim {i}: {src_dim} vs {tgt_dim}"
                    )
                
        if not isinstance(target_type.layout, NoneAttr):
            raise VerifyException(
                "tpu.memref_bitcast: Not implemented: bitcast to non-identity layout (TiledLayoutAttr support is pending)."
            )
        # TODO canonicalizer, TiledLayAttr

@irdl_op_definition
class ReinterpretCastOp(IRDLOperation):
    name = "tpu.reinterpret_cast"
    input = operand_def(MemRefType)
    dynamic_offset = opt_operand_def(i32)
    result = result_def(MemRefType)
 
    traits = traits_def(Pure())
 
    assembly_format = ("$input ($dynamic_offset^)? attr-dict `:` type($input) `->` type($result)")
 
    def verify_(self) -> None:
        source_type = self.input.type
        target_type = self.result.type
        assert isinstance(source_type, MemRefType)
        assert isinstance(target_type, MemRefType)
 
        if source_type.memory_space != target_type.memory_space:
            raise VerifyException(
                f"tpu.reinterpret_cast: Source and target memory spaces must match, "
                f"but got {source_type.memory_space} and {target_type.memory_space}"
            )
 