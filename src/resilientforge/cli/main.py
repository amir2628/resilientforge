"""`resilientforge` CLI: inspect, list, and prune oracle contents
(PROJECT_SPEC.md §4.1, §8's "CLI can list, inspect, and prune oracle
contents" acceptance criterion).

No `rich` dependency — plain column-aligned text output, consistent with
the dependency list actually declared in §5.2.
"""

from __future__ import annotations

import json
from collections import Counter

import typer

from resilientforge.oracle import Oracle
from resilientforge.oracle.recipes import Recipe, RecipeManager
from resilientforge.oracle.store import FailureRecord

app = typer.Typer(
    name="resilientforge",
    help="Inspect, list, and prune a ResilientForge failure oracle.",
    no_args_is_help=True,
)

_ORACLE_PATH_OPTION = typer.Option(
    ".resilientforge", "--oracle-path", "-p", help="Path to the oracle directory."
)


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


def _print_recipes_table(recipes: list[Recipe]) -> None:
    if not recipes:
        typer.echo("No recipes found.")
        return
    typer.echo(f"{'SIGNATURE':<57} {'TOOL':<18} {'APPLIED':>7} {'SUCCESS':>8}  LAST USED")
    for r in recipes:
        typer.echo(
            f"{_truncate(r.signature, 57):<57} {_truncate(r.tool_name, 18):<18} "
            f"{r.times_applied:>7} {r.success_rate:>7.0%}  {r.last_used}"
        )


def _print_recipe_detail(recipe: Recipe) -> None:
    typer.echo(f"signature:       {recipe.signature}")
    typer.echo(f"tool_name:       {recipe.tool_name}")
    typer.echo(f"root_cause:      {recipe.root_cause or '-'}")
    typer.echo(f"fix_strategy:    {recipe.fix_strategy or '-'}")
    typer.echo(f"fix_detail:      {json.dumps(recipe.fix_detail, indent=2)}")
    typer.echo(f"times_applied:   {recipe.times_applied}")
    typer.echo(f"times_succeeded: {recipe.times_succeeded}")
    typer.echo(f"success_rate:    {recipe.success_rate:.0%}")
    typer.echo(f"created_at:      {recipe.created_at}")
    typer.echo(f"last_used:       {recipe.last_used}")
    typer.echo("")


def _print_failures_table(failures: list[FailureRecord]) -> None:
    if not failures:
        typer.echo("No failure records found.")
        return
    typer.echo(f"{'ID':>5} {'TOOL':<18} {'STATUS':<11} {'ERROR TYPE':<18}  CREATED")
    for f in failures:
        typer.echo(
            f"{f.id:>5} {_truncate(f.tool_name, 18):<18} "
            f"{f.resolution_status.value:<11} {_truncate(f.error_type or '-', 18):<18}  {f.created_at}"
        )


@app.command("list")
def list_contents(
    oracle_path: str = _ORACLE_PATH_OPTION,
    tool_name: str = typer.Option(
        None, "--tool-name", "-t", help="Filter recipes by tool name."
    ),
    limit: int = typer.Option(50, "--limit", "-n", help="Maximum rows to show."),
    failures: bool = typer.Option(
        False, "--failures", help="List raw failure records instead of recipes."
    ),
) -> None:
    """List recipes (or, with --failures, raw failure records) in the oracle."""
    with Oracle(oracle_path) as oracle:
        if failures:
            _print_failures_table(oracle.list_failures(limit=limit))
        else:
            _print_recipes_table(RecipeManager(oracle).list(tool_name=tool_name, limit=limit))


@app.command()
def inspect(
    signature: str = typer.Argument(
        ..., help="A recipe signature, or a prefix of one — ambiguous prefixes show all matches."
    ),
    oracle_path: str = _ORACLE_PATH_OPTION,
) -> None:
    """Show full detail for one or more recipes matching SIGNATURE."""
    with Oracle(oracle_path) as oracle:
        matches = [
            r for r in RecipeManager(oracle).list(limit=10_000) if r.signature.startswith(signature)
        ]
    if not matches:
        typer.echo(f"No recipe found matching {signature!r}.")
        raise typer.Exit(code=1)
    if len(matches) > 1:
        typer.echo(f"{len(matches)} recipes match {signature!r}:\n")
    for recipe in matches:
        _print_recipe_detail(recipe)


@app.command()
def prune(
    oracle_path: str = _ORACLE_PATH_OPTION,
    min_success_rate: float = typer.Option(
        0.0, "--min-success-rate", help="Prune recipes below this success rate."
    ),
    min_times_applied: int = typer.Option(
        1, "--min-times-applied", help="Only judge success rate once applied at least this many times."
    ),
    max_age_days: float = typer.Option(
        None, "--max-age-days", help="Prune recipes not used within this many days."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be pruned without deleting anything."
    ),
) -> None:
    """Prune unreliable and/or stale recipes from the oracle."""
    with Oracle(oracle_path) as oracle:
        pruned = RecipeManager(oracle).prune(
            min_success_rate=min_success_rate,
            min_times_applied=min_times_applied,
            max_age_days=max_age_days,
            dry_run=dry_run,
        )
    verb = "Would prune" if dry_run else "Pruned"
    if not pruned:
        typer.echo(f"{verb} 0 recipes.")
        return
    typer.echo(f"{verb} {len(pruned)} recipe(s):")
    for signature in pruned:
        typer.echo(f"  - {signature}")


@app.command()
def stats(oracle_path: str = _ORACLE_PATH_OPTION) -> None:
    """Show a summary of the oracle's contents."""
    with Oracle(oracle_path) as oracle:
        recipes = oracle.list_recipes(limit=10_000)
        failures = oracle.list_failures(limit=10_000)

    typer.echo(f"oracle path:     {oracle_path}")
    typer.echo(f"recipes:         {len(recipes)}")
    typer.echo(f"failure records: {len(failures)}")
    by_status = Counter(f.resolution_status.value for f in failures)
    for status in sorted(by_status):
        typer.echo(f"  {status:<12} {by_status[status]}")


if __name__ == "__main__":
    app()
