# SPDX-License-Identifier: Apache-2.0

from awex.transfer.nccl_bounded_stream import (
    BoundedMemoryNcclColocateStreamBatchTransport,
)

from areal.engine.awex.colocate_reader import _DeviceBoundWeightsReader


def test_reader_selects_bounded_awex_transport(monkeypatch):
    import awex.transfer.nccl_stream_batch as stream_batch

    monkeypatch.setattr(stream_batch.device_util, "create_stream", lambda: object())
    reader = object.__new__(_DeviceBoundWeightsReader)
    reader.transfer_rank = 7
    reader.infer_world_size = 64

    transport = reader.create_colocate_transport()

    assert isinstance(transport, BoundedMemoryNcclColocateStreamBatchTransport)
    assert transport.transfer_rank == 7
    assert transport.world_size == 64
