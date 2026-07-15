#!/usr/bin/env python3
"""Round 3 validation driver — see ../../docs/real_world_validation_round3.md.
NOT new exploration: reuses round 2's exact setup (same react-agent commit,
same tools.py/graph.py patches, same prompts.py, same invariants) verbatim,
pointed at the now-fixed resilientforge (Part A/B of the round 3 task).
Independent of rounds 1/2: own oracle path, own metrics log, own session
log directory, all inside validation/round3/.

Call this at least twice. Oracle and metrics log persist at fixed paths
across invocations.

No oracle pre-seeding, no prompts hand-crafted to trigger a specific
failure — see prompts.py's module docstring (copied from round 2).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

VALIDATION_DIR = Path(__file__).resolve().parent
ORACLE_PATH = VALIDATION_DIR / ".resilientforge"
METRICS_LOG_PATH = VALIDATION_DIR / "metrics_log.jsonl"
SESSION_LOG_DIR = VALIDATION_DIR / "session_logs"

os.environ.setdefault("MODEL", "ollama/qwen2.5:7b")
os.environ["RESILIENTFORGE_ORACLE_PATH"] = str(ORACLE_PATH)
os.environ["RESILIENTFORGE_METRICS_LOG_PATH"] = str(METRICS_LOG_PATH)

sys.path.insert(0, str(VALIDATION_DIR / "react-agent" / "src"))
sys.path.insert(0, str(VALIDATION_DIR))

from react_agent.context import Context  # noqa: E402
from react_agent.graph import graph  # noqa: E402

from prompts import PROMPTS  # noqa: E402


def _next_session_number() -> int:
    SESSION_LOG_DIR.mkdir(exist_ok=True)
    existing = sorted(SESSION_LOG_DIR.glob("session_*.jsonl"))
    if not existing:
        return 1
    nums = [int(p.stem.split("_")[1]) for p in existing]
    return max(nums) + 1


async def _run_one_prompt(index: int, prompt: str, log_f) -> None:
    start = time.time()
    record: dict = {
        "index": index,
        "prompt": prompt,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        result = await graph.ainvoke(
            {"messages": [("user", prompt)]},
            context=Context(),
        )
        final = result["messages"][-1]
        record["final_answer"] = str(final.content)
        record["num_messages"] = len(result["messages"])
        record["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - a real, uncaught crash is itself a finding
        record["status"] = "graph_exception"
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["elapsed_seconds"] = round(time.time() - start, 2)
    log_f.write(json.dumps(record) + "\n")
    log_f.flush()
    print(f"[{index + 1}/{len(PROMPTS)}] ({record['elapsed_seconds']}s, {record['status']}) {prompt[:80]}")


async def main() -> None:
    session_num = _next_session_number()
    session_log_path = SESSION_LOG_DIR / f"session_{session_num}.jsonl"
    print(f"=== Round 3 validation session {session_num} ===")
    print(f"oracle: {ORACLE_PATH}")
    print(f"metrics log: {METRICS_LOG_PATH}")
    print(f"session log: {session_log_path}")
    print(f"{len(PROMPTS)} prompts\n")

    with open(session_log_path, "w") as log_f:
        for index, prompt in enumerate(PROMPTS):
            await _run_one_prompt(index, prompt, log_f)

    print(f"\n=== Session {session_num} complete ===")


if __name__ == "__main__":
    asyncio.run(main())
