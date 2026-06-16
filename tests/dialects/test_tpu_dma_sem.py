import pytest

from xdsl.dialects.builtin import (
    IndexType,
    MemRefType,
    i32,
)
from xdsl.dialects.tpu import TPU
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
from xdsl.dialects.tpu_memref import (
    DMASemaphoreType,
    SemaphoreType,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import create_ssa_value


def _semaphore_memref(rank: int = 0) -> MemRefType:
    shape = [1] * rank
    return MemRefType(SemaphoreType(), shape)


def _dma_semaphore_memref(rank: int = 0) -> MemRefType:
    shape = [1] * rank
    return MemRefType(DMASemaphoreType(), shape)


def test_semaphore_read_basic():
    sem = create_ssa_value(_semaphore_memref())
    op = SemaphoreReadOp(sem)
    op.verify()
    assert op.result.type == i32


def test_semaphore_read_from_dma_semaphore():
    sem = create_ssa_value(_dma_semaphore_memref())
    op = SemaphoreReadOp(sem)
    op.verify()


def test_semaphore_wait_basic():
    sem = create_ssa_value(_semaphore_memref())
    amount = create_ssa_value(i32)
    op = SemaphoreWaitOp(sem, amount)
    op.verify()


def test_semaphore_wait_rejects_non_rank_0():
    sem = create_ssa_value(_semaphore_memref(rank=1))
    amount = create_ssa_value(i32)
    op = SemaphoreWaitOp(sem, amount)
    with pytest.raises(VerifyException, match="must be rank 0"):
        op.verify()


def test_alloca_semaphore_basic():
    result_ty = _semaphore_memref()
    op = AllocaSemaphoreOp(result_ty)
    op.verify()
    assert op.result.type == result_ty


def test_alloca_semaphore_dma():
    result_ty = _dma_semaphore_memref()
    op = AllocaSemaphoreOp(result_ty)
    op.verify()


def test_get_barrier_semaphore_basic():
    result_ty = _semaphore_memref()
    op = GetBarrierSemaphoreOp(result_ty)
    op.verify()


def test_get_barrier_semaphore_rejects_non_rank_0():
    result_ty = _semaphore_memref(rank=1)
    op = GetBarrierSemaphoreOp(result_ty)
    with pytest.raises(VerifyException, match="must be rank 0"):
        op.verify()


def test_barrier_op_basic():
    barrier_id = create_ssa_value(IndexType())
    op = BarrierOp(barrier_id)
    op.verify()
    assert op.barrier_id.type == IndexType()


def test_semaphore_signal_basic():
    sem = create_ssa_value(_semaphore_memref())
    amount = create_ssa_value(i32)
    op = SemaphoreSignalOp(sem, amount)
    op.verify()


def test_semaphore_signal_with_device_id():
    sem = create_ssa_value(_semaphore_memref())
    amount = create_ssa_value(i32)
    device_id = create_ssa_value(i32)
    op = SemaphoreSignalOp(sem, amount, device_id=device_id)
    op.verify()
    assert op.device_id is not None


def test_semaphore_signal_with_core_id():
    sem = create_ssa_value(_semaphore_memref())
    amount = create_ssa_value(i32)
    core_id = create_ssa_value(i32)
    op = SemaphoreSignalOp(sem, amount, core_id=core_id)
    op.verify()
    assert op.core_id is not None


def test_semaphore_signal_with_both_ids():
    sem = create_ssa_value(_semaphore_memref())
    amount = create_ssa_value(i32)
    device_id = create_ssa_value(i32)
    core_id = create_ssa_value(i32)
    op = SemaphoreSignalOp(sem, amount, device_id=device_id, core_id=core_id)
    op.verify()


def test_semaphore_signal_rejects_non_rank_0():
    sem = create_ssa_value(_semaphore_memref(rank=1))
    amount = create_ssa_value(i32)
    op = SemaphoreSignalOp(sem, amount)
    with pytest.raises(VerifyException, match="must be rank 0"):
        op.verify()


def _basic_dma_memref():
    from xdsl.dialects.builtin import f32

    return MemRefType(f32, [8, 128])


def test_enqueue_dma_basic_local():
    src = create_ssa_value(_basic_dma_memref())
    tgt = create_ssa_value(_basic_dma_memref())
    target_sem = create_ssa_value(_dma_semaphore_memref())
    op = EnqueueDMAOp(src, tgt, target_sem)
    op.verify()
    assert op.priority.value.data == 0
    assert op.strict_ordering.value.data == 0


def test_enqueue_dma_remote():
    src = create_ssa_value(_basic_dma_memref())
    tgt = create_ssa_value(_basic_dma_memref())
    target_sem = create_ssa_value(_dma_semaphore_memref())
    source_sem = create_ssa_value(_dma_semaphore_memref())
    device_id = create_ssa_value(i32)
    op = EnqueueDMAOp(
        src,
        tgt,
        target_sem,
        source_semaphore=source_sem,
        device_id=device_id,
    )
    op.verify()


def test_enqueue_dma_with_strict_ordering_and_priority():
    src = create_ssa_value(_basic_dma_memref())
    tgt = create_ssa_value(_basic_dma_memref())
    target_sem = create_ssa_value(_dma_semaphore_memref())
    op = EnqueueDMAOp(
        src,
        tgt,
        target_sem,
        priority=1,
        strict_ordering=True,
    )
    op.verify()
    assert op.priority.value.data == 1
    assert op.strict_ordering.value.data == -1


def test_enqueue_dma_rejects_non_rank_0_target_semaphore():
    src = create_ssa_value(_basic_dma_memref())
    tgt = create_ssa_value(_basic_dma_memref())
    target_sem = create_ssa_value(_dma_semaphore_memref(rank=1))
    op = EnqueueDMAOp(src, tgt, target_sem)
    with pytest.raises(VerifyException, match="target semaphore must be rank 0"):
        op.verify()


def test_enqueue_dma_rejects_non_rank_0_source_semaphore():
    src = create_ssa_value(_basic_dma_memref())
    tgt = create_ssa_value(_basic_dma_memref())
    target_sem = create_ssa_value(_dma_semaphore_memref())
    source_sem = create_ssa_value(_dma_semaphore_memref(rank=1))
    device_id = create_ssa_value(i32)
    op = EnqueueDMAOp(
        src,
        tgt,
        target_sem,
        source_semaphore=source_sem,
        device_id=device_id,
    )
    with pytest.raises(
        VerifyException, match="source semaphore reference must be rank 0"
    ):
        op.verify()


def test_enqueue_dma_rejects_semaphore_type_mismatch():
    from xdsl.dialects.tpu_memref import SemaphoreType

    src = create_ssa_value(_basic_dma_memref())
    tgt = create_ssa_value(_basic_dma_memref())
    target_sem = create_ssa_value(_dma_semaphore_memref())
    source_sem = create_ssa_value(MemRefType(SemaphoreType(), []))
    device_id = create_ssa_value(i32)
    op = EnqueueDMAOp(
        src,
        tgt,
        target_sem,
        source_semaphore=source_sem,
        device_id=device_id,
    )
    with pytest.raises(VerifyException, match="same type"):
        op.verify()


def test_enqueue_dma_rejects_element_type_mismatch():
    from xdsl.dialects.builtin import f32

    src = create_ssa_value(MemRefType(f32, [8, 128]))
    tgt = create_ssa_value(MemRefType(i32, [8, 128]))
    target_sem = create_ssa_value(_dma_semaphore_memref())
    op = EnqueueDMAOp(src, tgt, target_sem)
    with pytest.raises(VerifyException, match="element type mismatch"):
        op.verify()


def test_enqueue_dma_rejects_shape_mismatch():
    from xdsl.dialects.builtin import f32

    src = create_ssa_value(MemRefType(f32, [8, 128]))
    tgt = create_ssa_value(MemRefType(f32, [16, 128]))
    target_sem = create_ssa_value(_dma_semaphore_memref())
    op = EnqueueDMAOp(src, tgt, target_sem)
    with pytest.raises(VerifyException, match="shape mismatch"):
        op.verify()


def test_enqueue_dma_rejects_device_id_without_source_semaphore():
    src = create_ssa_value(_basic_dma_memref())
    tgt = create_ssa_value(_basic_dma_memref())
    target_sem = create_ssa_value(_dma_semaphore_memref())
    device_id = create_ssa_value(i32)
    op = EnqueueDMAOp(src, tgt, target_sem, device_id=device_id)
    with pytest.raises(VerifyException, match="source semaphore must be specified"):
        op.verify()


def test_enqueue_dma_rejects_source_semaphore_without_remote():
    src = create_ssa_value(_basic_dma_memref())
    tgt = create_ssa_value(_basic_dma_memref())
    target_sem = create_ssa_value(_dma_semaphore_memref())
    source_sem = create_ssa_value(_dma_semaphore_memref())
    op = EnqueueDMAOp(
        src,
        tgt,
        target_sem,
        source_semaphore=source_sem,
    )
    with pytest.raises(VerifyException, match="device_id or core_id must be specified"):
        op.verify()


def test_enqueue_dma_rejects_priority_out_of_range():
    src = create_ssa_value(_basic_dma_memref())
    tgt = create_ssa_value(_basic_dma_memref())
    target_sem = create_ssa_value(_dma_semaphore_memref())
    op = EnqueueDMAOp(src, tgt, target_sem, priority=2)
    with pytest.raises(VerifyException, match="priority"):
        op.verify()


def test_enqueue_dma_rejects_remote_with_priority_1():
    src = create_ssa_value(_basic_dma_memref())
    tgt = create_ssa_value(_basic_dma_memref())
    target_sem = create_ssa_value(_dma_semaphore_memref())
    source_sem = create_ssa_value(_dma_semaphore_memref())
    device_id = create_ssa_value(i32)
    op = EnqueueDMAOp(
        src,
        tgt,
        target_sem,
        source_semaphore=source_sem,
        device_id=device_id,
        priority=1,
    )
    with pytest.raises(
        VerifyException, match="priority is not supported for remote DMA"
    ):
        op.verify()


def test_wait_dma2_basic():
    sem = create_ssa_value(_dma_semaphore_memref())
    src = create_ssa_value(_basic_dma_memref())
    dst = create_ssa_value(_basic_dma_memref())
    op = WaitDMA2Op(sem, src, dst)
    op.verify()


def test_wait_dma2_with_device_and_core_id():
    sem = create_ssa_value(_dma_semaphore_memref())
    src = create_ssa_value(_basic_dma_memref())
    dst = create_ssa_value(_basic_dma_memref())
    device_id = create_ssa_value(i32)
    core_id = create_ssa_value(i32)
    op = WaitDMA2Op(sem, src, dst, device_id=device_id, core_id=core_id)
    op.verify()


def test_wait_dma2_strict_ordering():
    sem = create_ssa_value(_dma_semaphore_memref())
    src = create_ssa_value(_basic_dma_memref())
    dst = create_ssa_value(_basic_dma_memref())
    op = WaitDMA2Op(sem, src, dst, strict_ordering=True)
    op.verify()
    assert op.strict_ordering.value.data == -1


def test_wait_dma2_rejects_non_rank_0_semaphore():
    sem = create_ssa_value(_dma_semaphore_memref(rank=1))
    src = create_ssa_value(_basic_dma_memref())
    dst = create_ssa_value(_basic_dma_memref())
    op = WaitDMA2Op(sem, src, dst)
    with pytest.raises(VerifyException, match="must be rank 0"):
        op.verify()


def test_dialect_registers_dma_batch2_ops():
    registered = set(TPU.operations)
    assert SemaphoreSignalOp in registered
    assert EnqueueDMAOp in registered
    assert WaitDMA2Op in registered


def test_device_id_op_basic():
    op = DeviceIdOp()
    op.verify()
    assert op.result.type == i32
