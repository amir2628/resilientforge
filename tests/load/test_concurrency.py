"""Load/concurrency test (Phase 5): N threads hammering ONE shared
Oracle with realistic wrap()/invoke() traffic. @pytest.mark.load — opt-in,
deselected by default (slower, hardware-dependent — not something every
PR should pay for). Correctness (no lost writes under concurrent access)
is asserted; throughput/latency are reported, not asserted against a
fixed number, since those are honestly hardware-dependent — same "real
numbers, not marketing copy" discipline as the failure-injection report.

Run: pytest -m load -v -s tests/load/test_concurrency.py
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from resilientforge import wrap
from resilientforge.oracle import Oracle
from resilientforge.oracle.recipes import RecipeManager

pytestmark = pytest.mark.load

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

N_THREADS = 16
CALLS_PER_THREAD = 25


def flaky_create_event(date: str, title: str = "Event") -> dict:
    if not _ISO_DATE_RE.match(date):
        raise ValueError(f"could not parse date '{date}'")
    return {"date": date, "title": title, "status": "created"}


def date_fixing_reflect(context) -> dict:
    return {
        "strategy": "reformat_argument",
        "transforms": [{"argument": "date", "transform": "parse_relative_date_to_iso"}],
    }


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = min(int(len(ordered) * p), len(ordered) - 1)
    return ordered[index]


def test_concurrent_writers_do_not_lose_updates_and_reports_real_throughput(tmp_path):
    oracle = Oracle(tmp_path / "oracle")

    def worker(thread_id: int) -> list[float]:
        # Every thread wraps the SAME tool/signature — deliberately the
        # worst case: all N_THREADS threads contend to update the exact
        # same recipe row, not N_THREADS independent rows.
        wrapped = wrap(
            flaky_create_event,
            oracle=oracle,
            reflect=date_fixing_reflect,
            tool_name="create_event",
            # Disabled deliberately: with guards enabled (the default),
            # a standing guard promotes after 3 successes and then
            # PREVENTS the failure outright — a real, correct, and
            # already-proven Phase 2 behavior (see recurring_date_guard's
            # prevention_rate), but one that makes later calls succeed
            # via the GUARD's own counter, not the recipe's, which would
            # confound what THIS test is isolating: whether the
            # recipe's counter update itself is race-free under
            # concurrent access. (Found by getting this wrong first —
            # times_applied landed at 17, not 400 — and realizing the
            # "missing" 383 had legitimately gone through the guard
            # instead, not been lost.)
            enable_standing_guards=False,
        )
        latencies = []
        for i in range(CALLS_PER_THREAD):
            start = time.monotonic()
            result = wrapped.invoke(date="next Friday", title=f"t{thread_id}-{i}")
            latencies.append(time.monotonic() - start)
            assert result["status"] == "created"
        return latencies

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
        futures = [executor.submit(worker, i) for i in range(N_THREADS)]
        all_latencies: list[float] = []
        for future in as_completed(futures):
            all_latencies.extend(future.result())  # re-raises if any thread hit an exception
    elapsed = time.monotonic() - start

    total_calls = N_THREADS * CALLS_PER_THREAD

    # Correctness: the ONE shared recipe's times_applied/times_succeeded
    # must exactly equal the total number of calls made — no update lost
    # to a race between concurrent readers/writers of the same row.
    recipes = RecipeManager(oracle)
    recipe = recipes.get(recipes.list()[0].signature)
    assert recipe.times_applied == total_calls
    assert recipe.times_succeeded == total_calls
    assert recipe.success_rate == 1.0

    throughput = total_calls / elapsed
    p50 = _percentile(all_latencies, 0.50)
    p99 = _percentile(all_latencies, 0.99)
    print(
        f"\n  {N_THREADS} threads x {CALLS_PER_THREAD} calls = {total_calls} total, "
        f"{elapsed:.2f}s wall-clock\n"
        f"  throughput: {throughput:.0f} calls/sec\n"
        f"  latency p50: {p50 * 1000:.1f}ms, p99: {p99 * 1000:.1f}ms"
    )

    oracle.close()
