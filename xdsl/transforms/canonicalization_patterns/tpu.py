from xdsl.dialects.builtin import DYNAMIC_INDEX, MemRefType
from xdsl.dialects.math import RoundEvenOp
from xdsl.dialects.memref import CastOp
from xdsl.dialects.tpu_conversions import FPToSIOp, RoundingMode
from xdsl.dialects.tpu_memref import (
    EraseLayoutOp,
    MemRefSliceOp,
    MemRefSqueezeOp,
    _compute_squeezed_dims,
)
from xdsl.ir import SSAValue
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
