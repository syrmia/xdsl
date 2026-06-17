from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
)
from xdsl.transforms.canonicalization_patterns.tpu import (
    CanonicalizeAddFOfMatmul,
    CanonicalizeAddIOfMatmul,
)
from xdsl.transforms.canonicalize import CanonicalizationRewritePattern
from xdsl.transforms.dead_code_elimination import (
    RemoveUnusedOperations,
    region_dce,
)


class TpuCanonicalizePass(ModulePass):
    name = "tpu-canonicalize"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    RemoveUnusedOperations(),
                    CanonicalizationRewritePattern(),
                    CanonicalizeAddFOfMatmul(),
                    CanonicalizeAddIOfMatmul(),
                ],
                folding_enabled=True,
                ctx=ctx,
            ),
            post_walk_func=region_dce,
        ).rewrite_module(op)
