"""Unit tests for the `resilientforge` CLI (cli/main.py): list, inspect,
prune, stats — PROJECT_SPEC.md §8's "CLI can list, inspect, and prune
oracle contents" acceptance criterion.

Not in PROJECT_SPEC.md §6's file tree, which doesn't list a CLI test file
at all — added for the same reason as prior additions this build
(harness.py, test_engine.py): the CLI needs its own coverage.
"""

from __future__ import annotations

from typer.testing import CliRunner

from resilientforge.cli.main import app
from resilientforge.oracle import Oracle
from resilientforge.oracle.recipes import RecipeManager
from resilientforge.oracle.store import ResolutionStatus

runner = CliRunner()


def _seed(oracle_path, tool_name="create_event", signature=None, **recipe_kwargs):
    signature = signature or f"tool:{tool_name}|error_type:ValueError|args:{{date:<STR>}}"
    with Oracle(oracle_path) as oracle:
        RecipeManager(oracle).record_success(
            signature=signature,
            tool_name=tool_name,
            fix_detail={"strategy": "reformat_argument"},
            root_cause="natural-language date string",
            fix_strategy="reformat_argument",
        )
        for _ in range(recipe_kwargs.get("extra_failures", 0)):
            oracle.record_failure(tool_name=tool_name, signature=signature, sanitized_args={})
    return signature


# -- list ------------------------------------------------------------------


def test_list_on_empty_oracle_shows_no_recipes(tmp_path):
    result = runner.invoke(app, ["list", "-p", str(tmp_path / "oracle")])
    assert result.exit_code == 0
    assert "No recipes found." in result.output


def test_list_shows_seeded_recipe(tmp_path):
    oracle_path = tmp_path / "oracle"
    signature = _seed(oracle_path)

    result = runner.invoke(app, ["list", "-p", str(oracle_path)])

    assert result.exit_code == 0
    assert "create_event" in result.output
    assert "100%" in result.output
    assert signature[:30] in result.output


def test_list_filters_by_tool_name(tmp_path):
    oracle_path = tmp_path / "oracle"
    _seed(oracle_path, tool_name="create_event", signature="sig-a")
    _seed(oracle_path, tool_name="send_email", signature="sig-b")

    result = runner.invoke(app, ["list", "-p", str(oracle_path), "--tool-name", "send_email"])

    assert "send_email" in result.output
    assert "create_event" not in result.output


def test_list_failures_flag_shows_failure_records(tmp_path):
    oracle_path = tmp_path / "oracle"
    with Oracle(oracle_path) as oracle:
        oracle.record_failure(tool_name="create_event", signature="sig-a", sanitized_args={})

    result = runner.invoke(app, ["list", "-p", str(oracle_path), "--failures"])

    assert result.exit_code == 0
    assert "create_event" in result.output
    assert "unresolved" in result.output


# -- inspect -----------------------------------------------------------------


def test_inspect_exact_match_shows_full_detail(tmp_path):
    oracle_path = tmp_path / "oracle"
    signature = _seed(oracle_path)

    result = runner.invoke(app, ["inspect", signature, "-p", str(oracle_path)])

    assert result.exit_code == 0
    assert f"signature:       {signature}" in result.output
    assert "fix_detail:" in result.output
    assert "reformat_argument" in result.output


def test_inspect_prefix_match(tmp_path):
    oracle_path = tmp_path / "oracle"
    signature = _seed(oracle_path)

    result = runner.invoke(app, ["inspect", signature[:20], "-p", str(oracle_path)])

    assert result.exit_code == 0
    assert signature in result.output


def test_inspect_ambiguous_prefix_shows_all_matches(tmp_path):
    oracle_path = tmp_path / "oracle"
    _seed(oracle_path, tool_name="t", signature="tool:t|shared-prefix-a")
    _seed(oracle_path, tool_name="t", signature="tool:t|shared-prefix-b")

    result = runner.invoke(app, ["inspect", "tool:t|shared-prefix", "-p", str(oracle_path)])

    assert result.exit_code == 0
    assert "2 recipes match" in result.output
    assert "tool:t|shared-prefix-a" in result.output
    assert "tool:t|shared-prefix-b" in result.output


def test_inspect_no_match_exits_nonzero(tmp_path):
    result = runner.invoke(app, ["inspect", "no-such-signature", "-p", str(tmp_path / "oracle")])

    assert result.exit_code == 1
    assert "No recipe found" in result.output


# -- prune ---------------------------------------------------------------------


def test_prune_dry_run_reports_without_deleting(tmp_path):
    oracle_path = tmp_path / "oracle"
    signature = _seed(oracle_path)

    result = runner.invoke(
        app, ["prune", "-p", str(oracle_path), "--dry-run", "--min-success-rate", "1.1"]
    )

    assert result.exit_code == 0
    assert "Would prune 1 recipe(s)" in result.output
    assert signature in result.output

    with Oracle(oracle_path) as oracle:
        assert oracle.get_recipe(signature) is not None  # still there


def test_prune_actually_deletes(tmp_path):
    oracle_path = tmp_path / "oracle"
    signature = _seed(oracle_path)

    result = runner.invoke(app, ["prune", "-p", str(oracle_path), "--min-success-rate", "1.1"])

    assert result.exit_code == 0
    assert "Pruned 1 recipe(s)" in result.output

    with Oracle(oracle_path) as oracle:
        assert oracle.get_recipe(signature) is None


def test_prune_nothing_qualifies(tmp_path):
    oracle_path = tmp_path / "oracle"
    _seed(oracle_path)

    result = runner.invoke(app, ["prune", "-p", str(oracle_path), "--min-success-rate", "0.0"])

    assert result.exit_code == 0
    assert "Pruned 0 recipes." in result.output


# -- stats -----------------------------------------------------------------


def test_stats_reports_counts_and_status_breakdown(tmp_path):
    # _seed() calls record_success() directly, which only touches the
    # recipes table — no `failures` row comes from it. Add two failure
    # records explicitly (one for each status) to exercise the breakdown.
    oracle_path = tmp_path / "oracle"
    _seed(oracle_path)
    with Oracle(oracle_path) as oracle:
        oracle.record_failure(tool_name="create_event", signature="sig-unresolved", sanitized_args={})
        f = oracle.record_failure(tool_name="create_event", signature="sig-recovered", sanitized_args={})
        oracle.update_failure_resolution(f.id, ResolutionStatus.RECOVERED)

    result = runner.invoke(app, ["stats", "-p", str(oracle_path)])

    assert result.exit_code == 0
    assert "recipes:         1" in result.output
    assert "failure records: 2" in result.output
    assert "recovered" in result.output
    assert "unresolved" in result.output
