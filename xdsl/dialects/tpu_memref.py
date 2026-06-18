from collections.abc import Sequence
from enum import auto

from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    ArrayAttr,
    FixedBitwidthType,
    IntAttr,
    MemRefLayoutAttr,
    MemRefType,
    NoneAttr,
    i32,
)
from xdsl.dialects.func import FuncOp
from xdsl.interfaces import HasFolderInterface
from xdsl.ir import (
    Attribute,
    Data,
    EnumAttribute,
    Operation,
    ParametrizedAttribute,
    SSAValue,
    TypeAttribute,
)
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineMap
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    param_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.parser import AttrParser
from xdsl.pattern_rewriter import RewritePattern
from xdsl.printer import Printer
from xdsl.traits import HasCanonicalizationPatternsTrait, Pure
from xdsl.transforms.canonicalization_patterns.utils import const_evaluate_operand
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.str_enum import StrEnum


class CoreType(StrEnum):
    Tc = auto()
    Sc_Scalar_Subcore = auto()
    Sc_Vector_Subcore = auto()


@irdl_attr_definition
class CoreTypeAttr(EnumAttribute[CoreType]):
    name = "tpu.core_type"
    enum_type = CoreType

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> CoreType:
        with parser.in_angle_brackets():
            return super().parse_parameter(parser)

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            super().print_parameter(printer)

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


@irdl_attr_definition
class _MemorySpaceData(Data[MemorySpace]):
    name = "tpu.memory_space_value"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> MemorySpace:
        return parser.parse_str_enum(MemorySpace)

    def print_parameter(self, printer: Printer) -> None:
        printer.print_identifier_or_string_literal(str(self.data))


@irdl_attr_definition
class TiledLayoutAttr(MemRefLayoutAttr, ParametrizedAttribute):
    name = "tpu.tiled"

    tiles: ArrayAttr[ArrayAttr[IntAttr]]
    tile_strides: ArrayAttr[IntAttr]

    def __init__(
        self,
        tiles: Sequence[Sequence[int]] | ArrayAttr[ArrayAttr[IntAttr]],
        tile_strides: Sequence[int] | ArrayAttr[IntAttr],
    ):
        if not isinstance(tiles, ArrayAttr):
            tiles = ArrayAttr(
                [ArrayAttr([IntAttr(dim) for dim in tile]) for tile in tiles]
            )
        if not isinstance(tile_strides, ArrayAttr):
            tile_strides = ArrayAttr([IntAttr(s) for s in tile_strides])
        super().__init__(tiles, tile_strides)

    @classmethod
    def parse_parameters(cls, parser: AttrParser):
        parser.parse_punctuation("<")
        tiles_list: list[list[int]] = []
        while parser.parse_optional_punctuation("(") is not None:
            dims: list[int] = [parser.parse_integer()]
            while parser.parse_optional_punctuation(",") is not None:
                dims.append(parser.parse_integer())
            parser.parse_punctuation(")")
            tiles_list.append(dims)
        if not tiles_list:
            parser.raise_error("Expected at least one tile in TiledLayoutAttr")
        parser.parse_punctuation(",")
        strides_list = parser.parse_comma_separated_list(
            parser.Delimiter.SQUARE, parser.parse_integer
        )
        parser.parse_punctuation(">")
        tiles_attr = ArrayAttr([ArrayAttr([IntAttr(d) for d in t]) for t in tiles_list])
        strides_attr = ArrayAttr([IntAttr(s) for s in strides_list])
        return [tiles_attr, strides_attr]

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            for tile in self.tiles.data:
                printer.print_string("(")
                printer.print_list(
                    tile.data,
                    lambda d: printer.print_string(str(d.data)),
                )
                printer.print_string(")")
            printer.print_string(",")
            with printer.in_square_brackets():
                printer.print_list(
                    self.tile_strides.data,
                    lambda s: printer.print_string(str(s.data)),
                )

    def get_rank(self) -> int:
        return len(self.tile_strides.data)

    def get_num_trailing_dims_with_contiguous_tiles(self, shape: Sequence[int]) -> int:
        from xdsl.dialects.builtin import DYNAMIC_INDEX

        tiles = self.tiles.data
        tile_strides = [s.data for s in self.tile_strides.data]
        n = len(shape)

        first_tile_dims: list[int] = []
        if tiles:
            first_tile_dims = [d.data for d in tiles[0].data]
        first_tile_rank = len(first_tile_dims)

        stride = 1
        stride_known = True
        d = n - 1
        while d >= 0:
            in_tiled_region = d >= n - first_tile_rank
            if in_tiled_region and shape[d] != DYNAMIC_INDEX:
                tile_d = d - (n - first_tile_rank)
                tile_size = first_tile_dims[tile_d]
                size_tiles = (shape[d] + tile_size - 1) // tile_size
                size_tiles_known = True
            else:
                size_tiles = shape[d]
                size_tiles_known = shape[d] != DYNAMIC_INDEX

            if stride_known and size_tiles_known and size_tiles != 1:
                if stride != tile_strides[d]:
                    break

            if not stride_known or not size_tiles_known:
                stride_known = False
            else:
                stride *= size_tiles
            d -= 1

        return n - 1 - d

    def tiles_are_known_contiguous(self, shape: Sequence[int]) -> bool:
        return (
            self.get_num_trailing_dims_with_contiguous_tiles(shape) == self.get_rank()
        )

    def get_affine_map(self) -> AffineMap:
        if len(self.tiles.data) != 1:
            raise NotImplementedError(
                "TiledLayoutAttr.get_affine_map: multi-level tiling is not implemented."
            )

        tile = [d.data for d in self.tiles.data[0].data]
        strides = [s.data for s in self.tile_strides.data]
        rank = len(tile)

        if len(strides) != rank:
            raise NotImplementedError(
                f"TiledLayoutAttr.get_affine_map: tile rank {rank} does not match tile_strides rank {len(strides)}. "
                "This implementation supports only equal-rank tile and strides."
            )

        inner_prods = [1] * rank
        for d in range(rank - 2, -1, -1):
            inner_prods[d] = inner_prods[d + 1] * tile[d + 1]

        result = AffineConstantExpr(0)
        for d in range(rank):
            t_d = tile[d]
            s_d = strides[d]
            ip = inner_prods[d]

            i_d = AffineDimExpr(d)
            tile_idx = i_d // AffineConstantExpr(t_d)
            inner_idx = i_d % AffineConstantExpr(t_d)

            result = result + tile_idx * AffineConstantExpr(s_d * ip)
            result = result + inner_idx * AffineConstantExpr(ip)

        return AffineMap(rank, 0, (result,))


class EraseLayoutHasCanonicalizerPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.tpu import (
            EraseLayoutChainCollapse,
        )

        return ((EraseLayoutChainCollapse()),)


class MemRefSliceHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls):
        from xdsl.transforms.canonicalization_patterns.tpu import (
            MemRefSliceFoldConstantDynamicDim,
        )

        return (MemRefSliceFoldConstantDynamicDim(),)


class MemRefSqueezeHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls):
        from xdsl.transforms.canonicalization_patterns.tpu import (
            MemRefSqueezeFoldCast,
        )

        return (MemRefSqueezeFoldCast(),)


@irdl_attr_definition
class MemorySpaceAttr(ParametrizedAttribute):
    name = "tpu.memory_space"

    value: _MemorySpaceData = param_def()
    core_type: CoreTypeAttr | NoneAttr = param_def()

    def __init__(
        self,
        value: MemorySpace | _MemorySpaceData,
        core_type: CoreType | CoreTypeAttr | None = None,
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

        return [value, core_type]

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            self.value.print_parameter(printer)
            if not isinstance(self.core_type, NoneAttr):
                printer.print_string(", ")
                self.core_type.print_parameter(printer)


def _check_memref_memory_spaces_match(
    op_name: str, source: MemRefType, target: MemRefType
) -> None:
    target_mem = target.memory_space
    if isinstance(target_mem, NoneAttr):
        return
    if target_mem != source.memory_space:
        raise VerifyException(f"{op_name}: Memory spaces do not match")


def _check_semaphore_element_type(op_name: str, mem_ref_type: MemRefType) -> None:
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


def _is_acceptable_layout(layout: Attribute) -> bool:
    return isinstance(layout, NoneAttr) or isinstance(layout, TiledLayoutAttr)


@irdl_op_definition
class MemRefSliceOp(IRDLOperation, HasFolderInterface):
    name = "tpu.memref_slice"
    mem_ref = operand_def(MemRefType)
    base_idx = var_operand_def(i32)
    dynamic_sizes = var_operand_def(i32)
    result = result_def(MemRefType)

    irdl_options = (AttrSizedOperandSegments(),)
    traits = traits_def(
        Pure(),
        MemRefSliceHasCanonicalizationPatternsTrait(),
    )

    assembly_format = "$mem_ref `[` $base_idx `]` (`<` $dynamic_sizes^ `>`)? attr-dict `:` type($mem_ref) `->` type($result)"

    def verify_(self) -> None:
        source_type = self.mem_ref.type
        target_type = self.result.type
        assert isinstance(source_type, MemRefType)
        assert isinstance(target_type, MemRefType)

        if not source_type.has_static_shape():
            raise VerifyException(
                "tpu.memref_slice: Only slicing of memrefs with static shapes is supported."
            )

        target_dynamic_dim_count = sum(
            1 for d in target_type.get_shape() if d == DYNAMIC_INDEX
        )

        if len(self.dynamic_sizes) != target_dynamic_dim_count:
            raise VerifyException(
                "tpu.memref_slice: Number of provided dynamic dimensions sizes "
                "must match the number of dynamic dimensions in the target type"
            )

        _check_semaphore_element_type("tpu.memref_slice", source_type)

        source_shape = source_type.get_shape()
        slice_shape = target_type.get_shape()
        if len(self.base_idx) != len(slice_shape) or len(self.base_idx) != len(
            source_shape
        ):
            raise VerifyException(
                "tpu.memref_slice: Indices and slice shapes must match."
            )

        _check_memref_memory_spaces_match("tpu.memref_slice", source_type, target_type)

        src_layout = source_type.layout
        tgt_layout = target_type.layout

        if not _is_acceptable_layout(src_layout) or not _is_acceptable_layout(
            tgt_layout
        ):
            raise VerifyException(
                "tpu.memref_slice: Only NoneAttr or TiledLayoutAttr layouts are supported"
            )

    def fold(self):
        if len(self.dynamic_sizes) != 0:
            return None
        if self.mem_ref.type != self.result.type:
            return None
        if not all(const_evaluate_operand(idx) == 0 for idx in self.base_idx):
            return None
        return (self.mem_ref,)


@irdl_op_definition
class MemRefSqueezeOp(IRDLOperation):
    name = "tpu.memref_squeeze"
    input = operand_def(MemRefType)
    result = result_def(MemRefType)

    traits = traits_def(
        Pure(),
        MemRefSqueezeHasCanonicalizationPatternsTrait(),
    )

    assembly_format = "$input attr-dict `:` type($input) `->` type($result)"

    def verify_(self) -> None:
        source_type = self.input.type
        target_type = self.result.type
        assert isinstance(source_type, MemRefType)
        assert isinstance(target_type, MemRefType)

        _check_memref_memory_spaces_match(
            "tpu.memref_squeeze", source_type, target_type
        )

        if source_type.element_type != target_type.element_type:
            raise VerifyException("tpu.memref_squeeze: Element types don't match.")

        source_shape = list(source_type.get_shape())
        target_shape = list(target_type.get_shape())
        squeezed = _compute_squeezed_dims(
            "tpu.memref_squeeze", source_shape, target_shape
        )

        if len(squeezed) == 0 and source_shape != target_shape:
            raise VerifyException(
                "tpu.memref_squeeze: Source and target shapes must be the same if no dimensions are squeezed."
            )

        src_layout = source_type.layout
        if not _is_acceptable_layout(src_layout):
            raise VerifyException(
                "tpu.memref_squeeze: Only NoneAttr or TiledLayoutAttr layouts are supported."
            )

        if isinstance(src_layout, TiledLayoutAttr):
            tiles = src_layout.tiles.data
            if len(tiles) == 1:
                first_tile_dims = [d.data for d in tiles[0].data]
                first_tiled = len(source_shape) - len(first_tile_dims)
                for dim in squeezed:
                    if dim >= first_tiled:
                        tile_idx = dim - first_tiled
                        if first_tile_dims[tile_idx] != 1:
                            raise VerifyException(
                                f"tpu.memref_squeeze: All tiled squeezed dimensions must be of size 1, but dim "
                                f"{dim} has tile size {first_tile_dims[tile_idx]}."
                            )
            elif len(tiles) >= 2:
                first_tile_dims = [d.data for d in tiles[0].data]
                first_tiled = len(source_shape) - len(first_tile_dims)
                for dim in squeezed:
                    if dim >= first_tiled:
                        raise VerifyException(
                            "tpu.memref_squeeze: When multiple tiles are present, no tiled dimensions "
                            f"can be squeezed, but dim {dim} is in the tiled region."
                        )


@irdl_op_definition
class MemRefReshapeOp(IRDLOperation):
    name = "tpu.memref_reshape"
    input = operand_def(MemRefType)
    result = result_def(MemRefType)

    traits = traits_def(Pure())

    assembly_format = "$input attr-dict `:` type($input) `->` type($result)"

    def verify_(self) -> None:
        source_type = self.input.type
        target_type = self.result.type
        assert isinstance(source_type, MemRefType)
        assert isinstance(target_type, MemRefType)

        _check_memref_memory_spaces_match(
            "tpu.memref_reshape", source_type, target_type
        )

        if len(source_type.get_shape()) < 2 or len(target_type.get_shape()) < 2:
            raise VerifyException(
                "tpu.memref_reshape: Not implemented: 1d memref reshape."
            )

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

        if not _is_acceptable_layout(src_layout) or not _is_acceptable_layout(
            tgt_layout
        ):
            raise VerifyException(
                "tpu.memref_reshape: Only NoneAttr or TiledLayoutAttr layouts are supported"
            )

        src_is_tiled = isinstance(src_layout, TiledLayoutAttr)
        tgt_is_tiled = isinstance(tgt_layout, TiledLayoutAttr)
        if src_is_tiled != tgt_is_tiled:
            raise VerifyException(
                "tpu.memref_reshape: Source and target must both have a tiled layout, or both have none."
            )

        if src_is_tiled:
            if src_layout.tiles != tgt_layout.tiles:
                raise VerifyException(
                    "tpu.memref_reshape: Expected the same tiling for the input and output memref."
                )
            if len(src_layout.tiles.data) > 0:
                tile_dims = [d.data for d in src_layout.tiles.data[0].data]
                if len(tile_dims) != 2:
                    raise VerifyException(
                        "tpu.memref_reshape: Not implemented: memref reshape with 1D tiling."
                    )
                src_shape = source_type.get_shape()
                tgt_shape = target_type.get_shape()
                if not src_layout.tiles_are_known_contiguous(
                    src_shape
                ) or not tgt_layout.tiles_are_known_contiguous(tgt_shape):
                    raise VerifyException(
                        "tpu.memref_reshape: Not implemented: reshape on a non-contiguous memref."
                    )
                src_tiled = src_shape[-2:]
                tgt_tiled = tgt_shape[-2:]
                is_src_align_2nd_minor = src_tiled[0] % tile_dims[0] == 0
                is_src_align_minor = src_tiled[1] % tile_dims[1] == 0
                is_tgt_align_2nd_minor = tgt_tiled[0] % tile_dims[0] == 0
                is_tgt_align_minor = tgt_tiled[1] % tile_dims[1] == 0

                if tile_dims[0] == 1 and is_src_align_minor and is_tgt_align_minor:
                    pass
                elif tgt_tiled[1] != src_tiled[1]:
                    raise VerifyException(
                        "tpu.memref_reshape: Expected the minormost dimension to be unchanged."
                    )
                elif tgt_tiled[0] != src_tiled[0]:
                    if not is_src_align_2nd_minor or not is_tgt_align_2nd_minor:
                        raise VerifyException(
                            "tpu.memref_reshape: Expected the 2nd minor dimension to be aligned to the tile."
                        )


@irdl_op_definition
class MemRefBitcastOp(IRDLOperation, HasFolderInterface):
    name = "tpu.memref_bitcast"
    input = operand_def(MemRefType)
    result = result_def(MemRefType)

    traits = traits_def(Pure())

    assembly_format = "$input attr-dict `:` type($input) `->` type($result)"

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
            raise VerifyException(
                "tpu.memref_bitcast: Not implemented: 1d memref bitcast."
            )

        src_elem = source_type.element_type
        tgt_elem = target_type.element_type
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

        src_layout = source_type.layout
        tgt_layout = target_type.layout

        if not _is_acceptable_layout(src_layout) or not _is_acceptable_layout(
            tgt_layout
        ):
            raise VerifyException(
                "tpu.memref_bitcast: Only NoneAttr or TiledLayoutAttr layouts are supported."
            )

        src_is_tiled = isinstance(src_layout, TiledLayoutAttr)
        tgt_is_tiled = isinstance(tgt_layout, TiledLayoutAttr)
        if src_is_tiled != tgt_is_tiled:
            raise VerifyException(
                "tpu.memref_bitcast: Source and target must both have a tiled layout, or both have none."
            )

        if src_is_tiled:
            src_tile_dims = [d.data for d in src_layout.tiles.data[0].data]
            tgt_tile_dims = [d.data for d in tgt_layout.tiles.data[0].data]
            if src_tile_dims[0] * src_bitwidth != tgt_tile_dims[0] * tgt_bitwidth:
                raise VerifyException(
                    f"tpu.memref_bitcast: Invalid memref bitcast. "
                    f"First tile dim mismatch: ({src_tile_dims[0]} * {src_bitwidth}) "
                    f"vs ({tgt_tile_dims[0]} * {tgt_bitwidth})."
                )

    def fold(self):
        if self.input.type == self.result.type:
            return (self.input,)
        return None


@irdl_op_definition
class ReinterpretCastOp(IRDLOperation):
    name = "tpu.reinterpret_cast"
    input = operand_def(MemRefType)
    dynamic_offset = opt_operand_def(i32)
    result = result_def(MemRefType)

    traits = traits_def(Pure())

    assembly_format = (
        "$input ($dynamic_offset^)? attr-dict `:` type($input) `->` type($result)"
    )

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


@irdl_op_definition
class EraseLayoutOp(IRDLOperation, HasFolderInterface):
    name = "tpu.erase_memref_layout"
    operand = operand_def(MemRefType)
    result = result_def(MemRefType)

    traits = traits_def(Pure(), EraseLayoutHasCanonicalizerPatternsTrait())

    assembly_format = "$operand attr-dict `:` type($operand) `->` type($result)"

    def __init__(self, operand: SSAValue | Operation, result_type: Attribute):
        super().__init__(operands=[operand], result_types=[result_type])

    def verify_(self) -> None:
        operand_ty = self.operand.type
        result_ty = self.result.type
        assert isinstance(operand_ty, MemRefType)
        assert isinstance(result_ty, MemRefType)

        if operand_ty.element_type != result_ty.element_type:
            raise VerifyException(
                "tpu.erase_memref_layout: Cannot change the memref element type"
            )

        _check_memref_memory_spaces_match(
            "tpu.erase_memref_layout", operand_ty, result_ty
        )

    def fold(self):
        if self.operand.type == self.result.type:
            return (self.operand,)
        return None
