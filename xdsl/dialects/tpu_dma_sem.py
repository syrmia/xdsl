from xdsl.dialects.builtin import (
    I32,
    BoolAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    MemRefType,
    i32,
)
from xdsl.dialects.tpu_memref import DMASemaphoreType, SemaphoreType
from xdsl.ir import Attribute, Operation, SSAValue
from xdsl.irdl import (
    AnyOf,
    AttrSizedOperandSegments,
    BaseAttr,
    IRDLOperation,
    attr_def,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    result_def,
    traits_def,
)
from xdsl.traits import Pure
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class SemaphoreReadOp(IRDLOperation):
    name = "tpu.sem_read"
    semaphore = operand_def(
        MemRefType.constr(AnyOf((BaseAttr(SemaphoreType), BaseAttr(DMASemaphoreType))))
    )
    result = result_def(i32)

    assembly_format = "$semaphore attr-dict `:` type($semaphore) `->` type($result)"

    def __init__(self, semaphore: SSAValue | Operation):
        super().__init__(operands=[semaphore], result_types=[i32])


@irdl_op_definition
class SemaphoreWaitOp(IRDLOperation):
    name = "tpu.sem_wait"

    semaphore = operand_def(MemRefType.constr(BaseAttr(SemaphoreType)))
    amount = operand_def(i32)

    assembly_format = "$semaphore `,` $amount attr-dict `:` type($semaphore)"

    def __init__(self, semaphore: SSAValue | Operation, amount: SSAValue | Operation):
        super().__init__(operands=[semaphore, amount])

    def verify_(self) -> None:
        sem_ty = self.semaphore.type
        assert isinstance(sem_ty, MemRefType)
        if len(sem_ty.get_shape()) != 0:
            raise VerifyException("tpu.sem_wait: Semaphore reference must be rank 0")


@irdl_op_definition
class AllocaSemaphoreOp(IRDLOperation):
    name = "tpu.sem_alloc"
    result = result_def(
        MemRefType.constr(AnyOf((BaseAttr(SemaphoreType), BaseAttr(DMASemaphoreType))))
    )

    assembly_format = "attr-dict `:` type($result)"

    def __init__(self, result_type: Attribute):
        super().__init__(result_types=[result_type])


@irdl_op_definition
class GetBarrierSemaphoreOp(IRDLOperation):
    name = "tpu.sem_barrier"
    semaphore = result_def(MemRefType.constr(BaseAttr(SemaphoreType)))

    assembly_format = "attr-dict `:` type($semaphore)"

    def __init__(self, result_type: Attribute):
        super().__init__(result_types=[result_type])

    def verify_(self) -> None:
        sem_ty = self.semaphore.type
        assert isinstance(sem_ty, MemRefType)
        if len(sem_ty.get_shape()) != 0:
            raise VerifyException(
                "tpu.sem_barier: Barrier emaphore reference must be rank 0"
            )


@irdl_op_definition
class BarrierOp(IRDLOperation):
    name = "tpu.barrier"

    barrier_id = operand_def(IndexType)

    assembly_format = "`barrier_id` `(` $barrier_id `)` attr-dict"

    def __init__(self, barrier_id: SSAValue | Operation):
        super().__init__(operands=[barrier_id])


@irdl_op_definition
class SemaphoreSignalOp(IRDLOperation):
    name = "tpu.sem_signal"
    semaphore = operand_def(MemRefType.constr(BaseAttr(SemaphoreType)))
    amount = operand_def(i32)
    device_id = opt_operand_def(i32)
    core_id = opt_operand_def(i32)

    irdl_options = (AttrSizedOperandSegments(),)

    assembly_format = "$semaphore `,` $amount (`device_id` $device_id^)? (`core_id` $core_id^)? attr-dict `:` type($semaphore)"

    def __init__(
        self,
        semaphore: SSAValue | Operation,
        amount: SSAValue | Operation,
        device_id: SSAValue | Operation | None = None,
        core_id: SSAValue | Operation | None = None,
    ):
        device_list: list[SSAValue | Operation] = (
            [device_id] if device_id is not None else []
        )

        core_list: list[SSAValue | Operation] = [core_id] if core_id is not None else []

        super().__init__(operands=[semaphore, amount, device_list, core_list])

    def verify_(self) -> None:
        sem_ty = self.semaphore.type
        assert isinstance(sem_ty, MemRefType)
        if len(sem_ty.get_shape()) != 0:
            raise VerifyException("tpu.sem_signal: Semaphore reference must be rank 0")


@irdl_op_definition
class EnqueueDMAOp(IRDLOperation):
    name = "tpu.enqueue_dma"
    source = operand_def(MemRefType)
    source_semaphore = opt_operand_def(
        MemRefType.constr(AnyOf((BaseAttr(SemaphoreType), BaseAttr(DMASemaphoreType))))
    )
    target = operand_def(MemRefType)
    target_semaphore = operand_def(
        MemRefType.constr(AnyOf((BaseAttr(SemaphoreType), BaseAttr(DMASemaphoreType))))
    )
    device_id = opt_operand_def(i32)
    core_id = opt_operand_def(i32)
    priority = attr_def(IntegerAttr[I32])
    strict_ordering = attr_def(BoolAttr)

    irdl_options = (AttrSizedOperandSegments(),)

    assembly_format = (
        "`source` `(` $source `:` type($source) `)`"
        "`target` `(` $target `:` type($target) `)`"
        "(`source_semaphore` `(` $source_semaphore^ `:` type($source_semaphore) `)`)?"
        "`target_semaphore` `(` $target_semaphore `:` type($target_semaphore) `)`"
        " (`device_id` `(` $device_id^ `)`)?"
        "(`core_id` `(` $core_id^ `)`)?"
        "attr-dict"
    )

    def __init__(
        self,
        source: SSAValue | Operation,
        target: SSAValue | Operation,
        target_semaphore: SSAValue | Operation,
        source_semaphore: SSAValue | Operation | None = None,
        device_id: SSAValue | Operation | None = None,
        core_id: SSAValue | Operation | None = None,
        priority: int | IntegerAttr[IntegerType] = 0,
        strict_ordering: bool | BoolAttr = False,
    ):
        if isinstance(priority, int):
            priority = IntegerAttr(priority, i32)
        if isinstance(strict_ordering, bool):
            strict_ordering = BoolAttr.from_bool(strict_ordering)
        source_sem_list: list[SSAValue | Operation] = (
            [source_semaphore] if source_semaphore is not None else []
        )
        device_list: list[SSAValue | Operation] = (
            [device_id] if device_id is not None else []
        )
        core_list: list[SSAValue | Operation] = [core_id] if core_id is not None else []
        super().__init__(
            operands=[
                source,
                source_sem_list,
                target,
                target_semaphore,
                device_list,
                core_list,
            ],
            attributes={"priority": priority, "strict_ordering": strict_ordering},
        )

    def verify_(self) -> None:
        target_sem_ty = self.target_semaphore.type
        assert isinstance(target_sem_ty, MemRefType)
        if len(target_sem_ty.get_shape()) != 0:
            raise VerifyException(
                "tpu.enqueue_dma: DMA target semaphore must be rank 0"
            )

        source_sem = self.source_semaphore
        if source_sem is not None:
            source_sem_ty = source_sem.type
            assert isinstance(source_sem_ty, MemRefType)
            if len(source_sem_ty.get_shape()) != 0:
                raise VerifyException(
                    "tpu.enqueue_dma: DMA source semaphore reference must be rank 0"
                )
            if source_sem_ty.element_type != target_sem_ty.element_type:
                raise VerifyException(
                    "tpu.enqueue_dma: DMA source and target semaphore must have the same type"
                )

        source_ty = self.source.type
        target_ty = self.target.type
        assert isinstance(source_ty, MemRefType)
        assert isinstance(target_ty, MemRefType)

        if source_ty.element_type != target_ty.element_type:
            raise VerifyException(
                "tpu.enqueue_dma: DMA source and target element type mismatch"
            )
        if list(source_ty.get_shape()) != list(target_ty.get_shape()):
            raise VerifyException(
                "tpu.enqueue_dma: DMA source and target shape mismatch."
            )

        has_device_or_core = self.device_id is not None or self.core_id is not None

        if has_device_or_core and self.source_semaphore is None:
            raise VerifyException(
                "tpu.enqueue_dma: DMA source semaphore must be specified when device_id or core_id is specified"
            )
        if self.source_semaphore is not None and not has_device_or_core:
            raise VerifyException(
                "tpu.enqueue_dma: DMA destination device_id or core_id must be specified when source semaphore is specified"
            )

        priority_val = self.priority.value.data
        if priority_val < 0 or priority_val > 1:
            raise VerifyException(
                f"tpu.enqueue_dma: Not implemented: only support priority 0 or 1, but got {priority_val}"
            )
        if priority_val != 0 and has_device_or_core:
            raise VerifyException(
                "tpu.enqueue_dma: Not implemented: non-zero priority is not supported for remote DMA"
            )


@irdl_op_definition
class WaitDMA2Op(IRDLOperation):
    name = "tpu.wait_dma2"

    semaphore = operand_def(MemRefType.constr(BaseAttr(DMASemaphoreType)))
    src = operand_def(MemRefType)
    dst = operand_def(MemRefType)
    device_id = opt_operand_def(i32)
    core_id = opt_operand_def(i32)
    strict_ordering = attr_def(BoolAttr)

    irdl_options = (AttrSizedOperandSegments(),)

    assembly_format = (
        "`semaphore` `(` $semaphore `:` type($semaphore) `)`"
        "`src` `(` $src `:` type($src) `)`"
        "`dst` `(` $dst `:` type($dst) `)`"
        "(`device_id` `(` $device_id^ `)`)?"
        "(`core_id` `(` $core_id^ `)`)?"
        "attr-dict"
    )

    def __init__(
        self,
        semaphore: SSAValue | Operation,
        src: SSAValue | Operation,
        dst: SSAValue | Operation,
        device_id: SSAValue | Operation | None = None,
        core_id: SSAValue | Operation | None = None,
        strict_ordering: bool | BoolAttr = False,
    ):
        if isinstance(strict_ordering, bool):
            strict_ordering = BoolAttr.from_bool(strict_ordering)
        device_list: list[SSAValue | Operation] = (
            [device_id] if device_id is not None else []
        )
        core_list: list[SSAValue | Operation] = [core_id] if core_id is not None else []
        super().__init__(
            operands=[
                semaphore,
                src,
                dst,
                device_list,
                core_list,
            ],
            attributes={"strict_ordering": strict_ordering},
        )

    def verify_(self) -> None:
        sem_ty = self.semaphore.type
        assert isinstance(sem_ty, MemRefType)
        if len(sem_ty.get_shape()) != 0:
            raise VerifyException("tpu.wait_dma2: DMA wait semaphore must be rank 0")


@irdl_op_definition
class DeviceIdOp(IRDLOperation):
    name = "tpu.devide_id"
    result = result_def(i32)

    traits = traits_def(Pure())

    assembly_format = "attr-dict `:` type($result)"

    def __init__(self):
        super().__init__(result_types=[i32])
