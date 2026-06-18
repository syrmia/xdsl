from collections.abc import Sequence
from enum import auto

from xdsl.dialects.builtin import (
    I32,
    I64,
    ArrayAttr,
    BoolAttr,
    IntegerAttr,
    IntegerType,
    VectorType,
    i32,
    i64,
)
from xdsl.ir import (
    Attribute,
    EnumAttribute,
    Operation,
    ParametrizedAttribute,
    SpacedOpaqueSyntaxAttribute,
    SSAValue,
)
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    param_def,
    result_def,
    traits_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.traits import Pure
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.str_enum import StrEnum


class ContractPrecision(StrEnum):
    Bf16 = auto()
    Fp32 = auto()


@irdl_attr_definition
class ContractPrecisionAttr(
    EnumAttribute[ContractPrecision], SpacedOpaqueSyntaxAttribute
):
    name = "tpu.contract_precision"
    enum_type = ContractPrecision


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
    rhs_non_contracting_dims: I64ArrayAttr = param_def()
    output_dim_order: I64ArrayAttr = param_def()
    lhs_batch_dims: I64ArrayAttr = param_def()
    rhs_batch_dims: I64ArrayAttr = param_def()

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
            printer.print_string(", ")
            _print_i64_array(printer, self.rhs_contracting_dims)
            printer.print_string(", ")
            _print_i64_array(printer, self.lhs_non_contracting_dims)
            printer.print_string(", ")
            _print_i64_array(printer, self.rhs_non_contracting_dims)
            printer.print_string(", ")
            _print_i64_array(printer, self.output_dim_order)
            printer.print_string(", ")
            _print_i64_array(printer, self.lhs_batch_dims)
            printer.print_string(", ")
            _print_i64_array(printer, self.rhs_batch_dims)

    def get_lhs_contracting_dims(self) -> list[int]:
        return [a.value.data for a in self.lhs_contracting_dims.data]

    def get_rhs_contracting_dims(self) -> list[int]:
        return [a.value.data for a in self.rhs_contracting_dims.data]

    def get_lhs_non_contracting_dims(self) -> list[int]:
        return [a.value.data for a in self.lhs_non_contracting_dims.data]

    def get_rhs_non_contracting_dims(self) -> list[int]:
        return [a.value.data for a in self.rhs_non_contracting_dims.data]

    def get_output_dim_order(self) -> list[int]:
        return [a.value.data for a in self.output_dim_order.data]

    def get_lhs_batch_dims(self) -> list[int]:
        return [a.value.data for a in self.lhs_batch_dims.data]

    def get_rhs_batch_dims(self) -> list[int]:
        return [a.value.data for a in self.rhs_batch_dims.data]


def _element_bitwidth(elem: Attribute) -> int:
    if isinstance(elem, IntegerType):
        return elem.width.data
    name = elem.name
    if name in ("f16", "bf16"):
        return 16
    if name == "f32":
        return 32
    if name == "f64":
        return 64
    if name in ("f8E5M2", "f8E4M3"):
        return 8
    raise VerifyException(f"Unknown element-type bitwidth for {elem}")


@irdl_op_definition
class MatmulOp(IRDLOperation):
    name = "tpu.matmul"

    lhs = operand_def(VectorType)
    rhs = operand_def(VectorType)
    acc = operand_def(VectorType)
    transpose_lhs = opt_attr_def(BoolAttr)
    transpose_rhs = opt_attr_def(BoolAttr)
    precision = opt_attr_def(ContractPrecisionAttr)
    dimension_numbers = opt_attr_def(DotDimensionNumbersAttr)
    result = result_def(VectorType)

    traits = traits_def(Pure())

    assembly_format = "$lhs `,` $rhs `,` $acc attr-dict `:` type($lhs) `,` type($rhs) `,` type($acc) `->` type($result)"

    def __init__(
        self,
        lhs: SSAValue | Operation,
        rhs: SSAValue | Operation,
        acc: SSAValue | Operation,
        result_type: Attribute,
        transpose_lhs: bool | BoolAttr = False,
        transpose_rhs: bool | BoolAttr = False,
        precision: ContractPrecision | ContractPrecisionAttr | None = None,
        dimension_numbers: DotDimensionNumbersAttr | None = None,
    ):
        if isinstance(transpose_lhs, bool):
            transpose_lhs = BoolAttr.from_bool(transpose_lhs)
        if isinstance(transpose_rhs, bool):
            transpose_rhs = BoolAttr.from_bool(transpose_rhs)
        if isinstance(precision, ContractPrecision):
            precision = ContractPrecisionAttr(precision)
        attrs: dict[str, Attribute] = {
            "transpose_lhs": transpose_lhs,
            "transpose_rhs": transpose_rhs,
        }
        if precision is not None:
            attrs["precision"] = precision
        if dimension_numbers is not None:
            attrs["dimension_numbers"] = dimension_numbers
        super().__init__(
            operands=[lhs, rhs, acc], result_types=[result_type], attributes=attrs
        )

    def verify_(self) -> None:
        lhs_ty = self.lhs.type
        rhs_ty = self.rhs.type
        acc_ty = self.acc.type
        res_ty = self.result.type
        assert isinstance(lhs_ty, VectorType)
        assert isinstance(rhs_ty, VectorType)
        assert isinstance(acc_ty, VectorType)
        assert isinstance(res_ty, VectorType)

        if acc_ty != res_ty:
            raise VerifyException(
                "tpu.matmul: Not implemented: matmul acc and result have "
                "different types"
            )

        if _element_bitwidth(acc_ty.element_type) != 32:
            raise VerifyException("tpu.matmul: Expected matmul acc to be 32-bit")

        if self.transpose_lhs is not None and self.transpose_lhs.value.data != 0:
            raise VerifyException(
                "tpu.matmul: Lhs transpose not supported via this API - please use the dimension numbers API."
            )

        if self.dimension_numbers is None:
            return

        dn = self.dimension_numbers
        lhs_contracting = dn.get_lhs_contracting_dims()
        rhs_contracting = dn.get_rhs_contracting_dims()
        lhs_batch = dn.get_lhs_batch_dims()
        rhs_batch = dn.get_rhs_batch_dims()
        lhs_non_contracting = dn.get_lhs_non_contracting_dims()
        rhs_non_contracting = dn.get_rhs_non_contracting_dims()

        if len(lhs_contracting) != 1:
            raise VerifyException(
                "tpu.matmul: Not implemented: lhs contracting dims must be of size 1"
            )
        if len(rhs_contracting) != 1:
            raise VerifyException(
                "tpu.matmul: Not implemented: rhs contracting dims must be of size 1"
            )

        lhs_contracting_dim = lhs_contracting[0]
        rhs_contracting_dim = rhs_contracting[0]

        if list(lhs_non_contracting) != sorted(lhs_non_contracting):
            raise VerifyException(
                "tpu.matmul: Not implemented: lhs non contracting dims must be sorted"
            )
        if list(rhs_non_contracting) != sorted(rhs_non_contracting):
            raise VerifyException(
                "tpu.matmul: Not implemented: rhs non contracting dims must be sorted"
            )

        lhs_rank = len(lhs_ty.get_shape())
        rhs_rank = len(rhs_ty.get_shape())
        if len(lhs_contracting) + len(lhs_non_contracting) + len(lhs_batch) != lhs_rank:
            raise VerifyException(
                "tpu.matmul: Not implemented: lhs contracting + non contracting"
                " + batch dims must be of the same size as the lhs shape"
            )
        if len(rhs_contracting) + len(rhs_non_contracting) + len(rhs_batch) != rhs_rank:
            raise VerifyException(
                "tpu.matmul: Not implemented: rhs contracting + non contracting"
                " + batch dims must be of the same size as the rhs shape"
            )

        lhs_shape = list(lhs_ty.get_shape())
        rhs_shape = list(rhs_ty.get_shape())
        if lhs_shape[lhs_contracting_dim] != rhs_shape[rhs_contracting_dim]:
            raise VerifyException(
                "tpu.matmul: Not implemented: lhs and rhs contracting dims must be of the same size"
            )

        if len(lhs_batch) != len(rhs_batch):
            raise VerifyException(
                "tpu.matmul: Not implemented: lhs and rhs should have the same number of batch dims"
            )
        if len(lhs_batch) > 1:
            raise VerifyException(
                "tpu.matmul: Not implemented: Up to 1 batch dim supported"
            )

        def _check_and_mark(
            dims: Sequence[int],
            seen: list[bool],
            operand: str,
        ) -> None:
            for dim in dims:
                if seen[dim]:
                    raise VerifyException(
                        f"tpu.matmul: Illegal: Dim {dim} repeats in dimension numbers of {operand}"
                    )
                seen[dim] = True

        seen_lhs = [False] * lhs_rank
        _check_and_mark(lhs_contracting, seen_lhs, "lhs")
        _check_and_mark(lhs_non_contracting, seen_lhs, "lhs")
        _check_and_mark(lhs_batch, seen_lhs, "lhs")
        for dim in range(lhs_rank):
            if not seen_lhs[dim]:
                raise VerifyException(
                    f"tpu.matmul: Illegal: Dim {dim} is not seen in lhs dimension numbers"
                )

        seen_rhs = [False] * rhs_rank
        _check_and_mark(rhs_contracting, seen_rhs, "rhs")
        _check_and_mark(rhs_non_contracting, seen_rhs, "rhs")
        _check_and_mark(rhs_batch, seen_rhs, "rhs")
        for dim in range(rhs_rank):
            if not seen_rhs[dim]:
                raise VerifyException(
                    f"tpu.matmul: Illegal: Dim {dim} is not seen in rhs dimension numbers"
                )

        if lhs_batch:
            batch_size = lhs_shape[lhs_batch[0]]
            rhs_batch_size = rhs_shape[rhs_batch[0]]
            if batch_size != rhs_batch_size:
                raise VerifyException(
                    "tpu.matmul: Not Implemented: batch dims must be equal"
                )
            if batch_size == 0:
                raise VerifyException("tpu.matmul: Illegal: batch size must be > 0")

        output_dim_order = dn.get_output_dim_order()
        if len(output_dim_order) % 2 != 0:
            raise VerifyException(
                "tpu.matmul: Illegal: output dim order must have an even "
                "number of elements."
            )

        expected: list[int] = []
        for dim in lhs_batch:
            expected.append(0)
            expected.append(dim)
        for dim in lhs_non_contracting:
            expected.append(0)
            expected.append(dim)
        for dim in rhs_non_contracting:
            expected.append(1)
            expected.append(dim)
        if list(output_dim_order) != expected:
            raise VerifyException(
                "tpu.matmul: Illegal: output dim order must be in the form of [0, lhs_batch_dims,"
                " 0, lhs_non_contracting_dims, 1, rhs_non_contracting_dims]"
            )


@irdl_op_definition
class MatmulPushRhsOp(IRDLOperation):
    name = "tpu.matmul_push_rhs"
    rhs = operand_def(VectorType)
    mxu_index = attr_def(IntegerAttr[I32])
    staging_register = attr_def(IntegerAttr[I32])
    transpose = attr_def(BoolAttr)

    assembly_format = "$rhs attr-dict `:` type($rhs)"

    def __init__(
        self,
        rhs: SSAValue | Operation,
        mxu_index: int | IntegerAttr[IntegerType],
        staging_register: int | IntegerAttr[IntegerType] = 0,
        transpose: bool | BoolAttr = False,
    ):
        if isinstance(mxu_index, int):
            mxu_index = IntegerAttr(mxu_index, i32)
        if isinstance(staging_register, int):
            staging_register = IntegerAttr(staging_register, i32)
        if isinstance(transpose, bool):
            transpose = BoolAttr.from_bool(transpose)
        super().__init__(
            operands=[rhs],
            attributes={
                "mxu_index": mxu_index,
                "staging_register": staging_register,
                "transpose": transpose,
            },
        )


@irdl_op_definition
class MatmulAccLhsOp(IRDLOperation):
    name = "tpu.matmul_acc_lhs"
    acc = attr_def(IntegerAttr[I32])
    lhs = operand_def(VectorType)
    mxu_index = attr_def(IntegerAttr[I32])
    load_staged_rhs = opt_attr_def(IntegerAttr[I32])

    assembly_format = "$acc `,` $lhs attr-dict `:` type($lhs)"

    def __init__(
        self,
        lhs: SSAValue | Operation,
        acc: int | IntegerAttr[IntegerType],
        mxu_index: int | IntegerAttr[IntegerType],
        load_staged_rhs: int | IntegerAttr[IntegerType] | None = None,
    ):
        if isinstance(acc, int):
            acc = IntegerAttr(acc, i32)
        if isinstance(mxu_index, int):
            mxu_index = IntegerAttr(mxu_index, i32)
        if isinstance(load_staged_rhs, int):
            load_staged_rhs = IntegerAttr(load_staged_rhs, i32)
        attrs: dict[str, Attribute] = {"acc": acc, "mxu_index": mxu_index}
        if load_staged_rhs is not None:
            attrs["load_staged_rhs"] = load_staged_rhs
        super().__init__(operands=[lhs], attributes=attrs)


@irdl_op_definition
class MatmulPopOp(IRDLOperation):
    name = "tpu.matmul_pop"
    acc = attr_def(IntegerAttr[I32])
    mxu_index = attr_def(IntegerAttr[I32])
    result = result_def(VectorType)

    assembly_format = "$acc attr-dict `:` type($result)"

    def __init__(
        self,
        result_type: Attribute,
        acc: int | IntegerAttr[IntegerType],
        mxu_index: int | IntegerAttr[IntegerType],
    ):
        if isinstance(acc, int):
            acc = IntegerAttr(acc, i32)
        if isinstance(mxu_index, int):
            mxu_index = IntegerAttr(mxu_index, i32)
        super().__init__(
            result_types=[result_type], attributes={"acc": acc, "mxu_index": mxu_index}
        )
