from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.dialects.arith import (
    AddfOp,
    DivfOp,
    ExtFOp,
    MaximumfOp,
    MinimumfOp,
    MulfOp,
    NegfOp,
    SelectOp,
    SubfOp,
    TruncFOp,
)
from xdsl.dialects.builtin import VectorType, bf16, f32
from xdsl.dialects.math import (
    AbsFOp,
    CosOp,
    ExpOp,
    Log1pOp,
    LogOp,
    PowFOp,
    RoundEvenOp,
    RoundOp,
    RsqrtOp,
    SinOp,
    SqrtOp,
    TanhOp,
)
from xdsl.dialects.vector import MultiDimReductionOp
from xdsl.ir import Attribute, Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa


def _is_bf16_vector(t: Attribute) -> bool:
    return isa(t, VectorType) and t.element_type == bf16


class MultiReductionBitwidthConvert(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: MultiDimReductionOp, rewriter: PatternRewriter
    ) -> None:
        src_ty = op.source.type
        if not _is_bf16_vector(src_ty):
            return
        res_ty = op.dest.type
        if not isa(res_ty, VectorType):
            return

        src_f32_ty = VectorType(f32, src_ty.get_shape())
        ext_src = ExtFOp(op.source, src_f32_ty)
        acc_f32_ty = VectorType(f32, res_ty.get_shape())
        ext_acc = ExtFOp(op.acc, acc_f32_ty)
        new_reduction = MultiDimReductionOp(
            ext_src.result,
            ext_acc.result,
            op.kind,
            op.reduction_dims,
            acc_f32_ty,
        )
        trunc = TruncFOp(new_reduction.dest, res_ty)
        rewriter.replace_matched_op([ext_src, ext_acc, new_reduction, trunc])


_UNARY_BF16_OPS = (
    AbsFOp,
    CosOp,
    ExpOp,
    Log1pOp,
    LogOp,
    RoundEvenOp,
    RoundOp,
    RsqrtOp,
    SinOp,
    SqrtOp,
    TanhOp,
    NegfOp,
)

_BINARY_BF16_OPS = (
    AddfOp,
    SubfOp,
    MulfOp,
    DivfOp,
    MaximumfOp,
    MinimumfOp,
    PowFOp,
)


class GenericBitwidthConvert(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
        result_ty = getattr(op, "result", None)
        result_ty = result_ty.type if result_ty is not None else None
        if not _is_bf16_vector(result_ty):
            return

        if isinstance(op, _UNARY_BF16_OPS):
            operand_ty = op.operand.type
            if not _is_bf16_vector(operand_ty):
                return
            ext = ExtFOp(op.operand, VectorType(f32, operand_ty.get_shape()))
            new_op = type(op)(ext.result)
            trunc = TruncFOp(new_op.result, result_ty)
            rewriter.replace_matched_op([ext, new_op, trunc])
            return

        if isinstance(op, _BINARY_BF16_OPS):
            lhs_ty = op.lhs.type
            rhs_ty = op.rhs.type
            if not (_is_bf16_vector(lhs_ty) and _is_bf16_vector(rhs_ty)):
                return
            ext_lhs = ExtFOp(op.lhs, VectorType(f32, lhs_ty.get_shape()))
            ext_rhs = ExtFOp(op.rhs, VectorType(f32, rhs_ty.get_shape()))
            new_op = type(op)(ext_lhs.result, ext_rhs.result)
            trunc = TruncFOp(new_op.result, result_ty)
            rewriter.replace_matched_op([ext_lhs, ext_rhs, new_op, trunc])
            return

        if isinstance(op, SelectOp):
            lhs_ty = op.lhs.type
            rhs_ty = op.rhs.type
            if not (_is_bf16_vector(lhs_ty) and _is_bf16_vector(rhs_ty)):
                return
            ext_lhs = ExtFOp(op.lhs, VectorType(f32, lhs_ty.get_shape()))
            ext_rhs = ExtFOp(op.rhs, VectorType(f32, rhs_ty.get_shape()))
            new_sel = SelectOp(op.cond, ext_lhs.result, ext_rhs.result)
            trunc = TruncFOp(new_sel.result, result_ty)
            rewriter.replace_matched_op([ext_lhs, ext_rhs, new_sel, trunc])
            return


@dataclass(frozen=True)
class TpuBitwidthConvertPass(ModulePass):
    name = "tpu-bitwidth-convert"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    MultiReductionBitwidthConvert(),
                    GenericBitwidthConvert(),
                ]
            )
        ).rewrite_module(op)
