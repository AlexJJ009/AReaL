# SPDX-License-Identifier: Apache-2.0

import gc
import threading
from concurrent.futures import ThreadPoolExecutor

from areal.engine.awex.metadata import serialize_metadata_gc


def test_gc_scan_waits_until_metadata_tuple_is_complete():
    building = threading.Event()
    scan_requested = threading.Event()
    events = []
    marker = object()

    @serialize_metadata_gc
    def build_metadata():
        def dimensions():
            yield marker
            building.set()
            assert scan_requested.wait(5)
            yield 2

        result = tuple(dimensions())
        events.append("metadata complete")
        return result

    @serialize_metadata_gc
    def freeze_gc():
        # Holding references to a growing tuple here would cause SystemError
        # when tuple(dimensions()) resizes its allocation after iteration.
        retained = gc.get_referrers(marker)
        events.append("GC scan")
        return retained

    def request_scan():
        assert building.wait(5)
        scan_requested.set()
        return freeze_gc()

    with ThreadPoolExecutor(max_workers=2) as pool:
        metadata = pool.submit(build_metadata)
        scan = pool.submit(request_scan)
        assert metadata.result(timeout=10) == (marker, 2)
        scan.result(timeout=10)
    assert events == ["metadata complete", "GC scan"]


def test_metadata_gc_guard_releases_lock_after_failure():
    import pytest

    @serialize_metadata_gc
    def fail():
        raise ValueError("metadata failed")

    @serialize_metadata_gc
    def succeed():
        return "ready"

    with pytest.raises(ValueError, match="metadata failed"):
        fail()
    assert succeed() == "ready"
