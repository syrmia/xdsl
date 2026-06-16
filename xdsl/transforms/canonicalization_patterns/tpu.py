from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    IntegerType,
    MemRefType,
    Signedness,
    VectorType,
)
from xdsl.dialects.math import RoundEvenOp
from xdsl.dialects.memref import CastOp
from xdsl.dialects.tpu_conversions import FPToSIOp, RoundingMode
from xdsl.dialects.tpu_memory import LoadOp, ShuffledLoadOp, ShuffledStoreOp, StoreOp
from xdsl.dialects.tpu_memref import (
    EraseLayoutOp,
    MemRefSliceOp,
    MemRefSqueezeOp,
    _compute_squeezed_dims,
)
from xdsl.dialects.tpu_pack import PackSubelementsOp, UnpackSubelementsOp
from xdsl.dialects.tpu_shape import (
    BitcastVregOp,
    DynamicGatherOp,
    ReshapeOp,
    RollVectorsOp,
    UnrollVectorsOp,
)
from xdsl.dialects.vector import BroadcastOp
from xdsl.ir import Attribute, SSAValue
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.canonicalization_patterns.utils import const_evaluate_operand
from xdsl.utils.exceptions import VerifyException


class EraseLayoutChainCollapse(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: EraseLayoutOp, rewriter: PatternRewriter) -> None:
        defining_op = op.operand.owner
        if not isinstance(defining_op, EraseLayoutOp):
            return
        new_op = EraseLayoutOp(defining_op.operand, op.result.type)
        rewriter.replace_matched_op(new_op)


class MemRefSliceFoldConstantDynamicDim(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MemRefSliceOp, rewriter: PatternRewriter) -> None:
        if not any(const_evaluate_operand(ds) is not None for ds in op.dynamic_sizes):
            return

        old_type = op.result.type
        if not isinstance(old_type, MemRefType):
            return
        old_shape = list(old_type.get_shape())
        new_shape = list(old_shape)

        new_dynamic_sizes: list[SSAValue] = []
        dynamic_dim_index = 0
        for dynamic_size in op.dynamic_sizes:
            while (
                dynamic_dim_index < len(new_shape)
                and new_shape[dynamic_dim_index] != DYNAMIC_INDEX
            ):
                dynamic_dim_index += 1
            if dynamic_dim_index >= len(new_shape):
                return
            const_val = const_evaluate_operand(dynamic_size)
            if const_val is not None:
                if const_val <= 0:
                    new_dynamic_sizes.append(dynamic_size)
                else:
                    new_shape[dynamic_dim_index] = const_val
            else:
                new_dynamic_sizes.append(dynamic_size)
            dynamic_dim_index += 1

        if new_shape == old_shape:
            return

        new_type = MemRefType(
            old_type.element_type,
            new_shape,
            old_type.layout,
            old_type.memory_space,
        )

        new_slice = MemRefSliceOp.build(
            operands=[op.mem_ref, list(op.base_idx), new_dynamic_sizes],
            result_types=[new_type],
        )

        cast = CastOp.get(new_slice.result, old_type)

        rewriter.replace_matched_op([new_slice, cast])


class MemRefSqueezeFoldCast(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MemRefSqueezeOp, rewriter: PatternRewriter) -> None:
        producer = op.input.owner
        if not isinstance(producer, CastOp):
            return

        cast_source = producer.source
        cast_source_type = cast_source.type
        cast_result_type = producer.dest.type

        if not (
            isinstance(cast_source_type, MemRefType)
            and isinstance(cast_result_type, MemRefType)
        ):
            return
        if cast_source_type.get_num_dims() != cast_result_type.get_num_dims():
            return

        for src_dim, dst_dim in zip(
            cast_source_type.get_shape(), cast_result_type.get_shape()
        ):
            if src_dim == dst_dim:
                continue
            if src_dim == DYNAMIC_INDEX and dst_dim != DYNAMIC_INDEX:
                return

        squeeze_result_type = op.result.type
        try:
            squeezed_dims = _compute_squeezed_dims(
                "MemRefSqueezeFoldCast",
                list(cast_result_type.get_shape()),
                list(squeeze_result_type.get_shape()),
            )
        except VerifyException:
            return

        new_result_shape = [
            dim
            for i, dim in enumerate(cast_source_type.get_shape())
            if i not in squeezed_dims
        ]
        if len(new_result_shape) != squeeze_result_type.get_num_dims():
            return
        if list(squeeze_result_type.get_shape()) == new_result_shape:
            return

        new_result_type = MemRefType(
            squeeze_result_type.element_type,
            new_result_shape,
            squeeze_result_type.layout,
            squeeze_result_type.memory_space,
        )
        new_squeeze = MemRefSqueezeOp.build(
            operands=[cast_source],
            result_types=[new_result_type],
        )
        cast = CastOp.get(new_squeeze.result, squeeze_result_type)
        rewriter.replace_matched_op([new_squeeze, cast])


class FPToSISinkRoundEven(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: FPToSIOp, rewriter: PatternRewriter) -> None:
        producer = op.input.owner
        if not isinstance(producer, RoundEvenOp):
            return
        new_op = FPToSIOp(
            input_=producer.operand,
            target_type=op.output.type,
            rounding_mode=RoundingMode.To_Nearest_Even,
        )
        rewriter.replace_matched_op(new_op)


class ShuffledLoadToSimpleLoad(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ShuffledLoadOp, rewriter: PatternRewriter) -> None:
        offsets = list(op.sublane_offsets.get_values())
        for i, offset in enumerate(offsets):
            if offset != i:
                return
        new_op = LoadOp(
            base=op.base,
            indices=list(op.indices),
            sublane_mask=op.sublane_mask,
            result_type=op.result.type,
        )
        rewriter.replace_matched_op(new_op)


class ShuffledStoreToSimpleStore(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ShuffledStoreOp, rewriter: PatternRewriter) -> None:
        offsets = list(op.sublane_offsets.get_values())
        for i, offset in enumerate(offsets):
            if offset != i:
                return
        new_op = StoreOp(
            op.value_to_store,
            op.base,
            list(op.indices),
            op.sublane_mask,
            None,
        )
        rewriter.replace_matched_op(new_op)


def _fill_positions(
    values: list[SSAValue], positions: list[int], size: int
) -> list[SSAValue | None]:
    result: list[SSAValue[Attribute] | None] = [None] * size
    for value, position in zip(values, positions):
        result[position] = value
    return result


class UnpackOfPackCancel(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: UnpackSubelementsOp, rewriter: PatternRewriter
    ) -> None:
        src_ty = op.source.type
        dst_ty = op.output.type
        if not (isinstance(src_ty, VectorType)):
            return
        src_elem = src_ty.element_type
        dst_elem = dst_ty.element_type
        if not isinstance(src_elem, IntegerType):
            return
        if not isinstance(dst_elem, IntegerType):
            return
        if src_elem.signedness.data != Signedness.SIGNLESS:
            return
        if dst_elem.signedness.data != Signedness.SIGNLESS:
            return

        if op.integer_extended.value.data:
            return

        producer = op.source.owner
        if not isinstance(producer, PackSubelementsOp):
            return

        if producer.pack_format != op.pack_format:
            return

        if len(producer.sources) == 0:
            return
        if producer.sources[0].type != op.output.type:
            return

        src_bw = src_elem.width.data
        dst_bw = dst_elem.width.data
        if dst_bw == 0 or dst_bw % src_bw != 0:
            return
        packing_factor = dst_bw // src_bw

        positions = list(producer.positions.get_values())
        filled = _fill_positions(list(producer.sources), positions, packing_factor)

        index = op.index.value.data
        if index >= len(filled):
            return
        source = filled[index]
        if source is None:
            return

        rewriter.replace_matched_op([], new_results=[source])


class UnpackOfPackSignExtensionDemote(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: UnpackSubelementsOp, rewriter: PatternRewriter
    ) -> None:
        if not op.integer_extended.value.data:
            return
        src_ty = op.source.type
        if not isinstance(src_ty, VectorType):
            return
        src_elem = src_ty.element_type
        if not isinstance(src_elem, IntegerType):
            return
        src_bw = src_elem.width.data

        users = list(op.output.uses)
        if len(users) == 0:
            return

        for use in users:
            user_op = use.operation
            if not isinstance(user_op, PackSubelementsOp):
                return
            packed_ty = user_op.output.type
            if not isinstance(packed_ty, VectorType):
                return
            packed_elem = packed_ty.element_type
            if not isinstance(packed_elem, IntegerType):
                return
            if packed_elem.signedness.data != Signedness.SIGNLESS:
                return
            if packed_elem.width.data > src_bw:
                return

        new_unpack = UnpackSubelementsOp(
            source=op.source,
            index=op.index,
            pack_format=op.pack_format,
            result_type=op.output.type,
            integer_extended=False,
            unsigned_integers=op.unsigned_integers.value.data,
        )
        rewriter.replace_matched_op(new_unpack)


class BitcastVregChainCollapse(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: BitcastVregOp, rewriter: PatternRewriter) -> None:
        defining_op = op.input.owner
        if not isinstance(defining_op, BitcastVregOp):
            return
        new_op = BitcastVregOp(defining_op.input, op.output.type)
        rewriter.replace_matched_op(new_op)


class ReshapeOfReshape(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ReshapeOp, rewriter: PatternRewriter) -> None:
        defining_op = op.source.owner
        if not isinstance(defining_op, ReshapeOp):
            return
        new_op = ReshapeOp(defining_op.source, op.result.type)
        rewriter.replace_matched_op(new_op)


class UnrollOfRollCancel(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: UnrollVectorsOp, rewriter: PatternRewriter) -> None:
        producer = op.input.owner
        if not isinstance(producer, RollVectorsOp):
            return
        if len(producer.input) != len(op.output):
            return
        for roll_operand, unroll_result in zip(producer.input, op.output):
            if roll_operand.type != unroll_result.type:
                return
        rewriter.replace_matched_op([], new_results=list(producer.input))


class DynamicGatherToBroadcast(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DynamicGatherOp, rewriter: PatternRewriter) -> None:
        src_ty = op.source.type
        if not isinstance(src_ty, VectorType):
            return
        src_shape = src_ty.get_shape()
        dimensions = list(op.dimensions.get_values())
        for d in dimensions:
            if d < 0 or d >= len(src_shape) or src_shape[d] != 1:
                return
        new_op = BroadcastOp(op.source, op.output.type)
        rewriter.replace_matched_op(new_op)
