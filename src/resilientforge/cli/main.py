"""`resilientforge` CLI: inspect, list, and prune oracle contents.

No `rich` dependency — plain column-aligned text output, consistent with
the project's declared dependency list.
"""

from __future__ import annotations

import json
from collections import Counter

import typer

from resilientforge.oracle import Oracle
from resilientforge.oracle.guards import GuardManager, StandingGuard
from resilientforge.oracle.recipes import Recipe, RecipeManager
from resilientforge.oracle.store import FailureRecord

app = typer.Typer(
    name="resilientforge",
    help="Inspect, list, and prune a ResilientForge failure oracle.",
    no_args_is_help=True,
)

guards_app = typer.Typer(
    name="guards",
    help="Inspect, list, and revoke standing guards (Phase 2).",
    no_args_is_help=True,
)
app.add_typer(guards_app, name="guards")

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
        guards = oracle.list_guards(active_only=False, limit=10_000)

    typer.echo(f"oracle path:     {oracle_path}")
    typer.echo(f"recipes:         {len(recipes)}")
    typer.echo(f"failure records: {len(failures)}")
    typer.echo(f"guards:          {len(guards)} ({sum(1 for g in guards if g.active)} active)")
    by_status = Counter(f.resolution_status.value for f in failures)
    for status in sorted(by_status):
        typer.echo(f"  {status:<12} {by_status[status]}")


# -- guards (Phase 2) ---------------------------------------------------------


def _print_guards_table(guards: list[StandingGuard]) -> None:
    if not guards:
        typer.echo("No guards found.")
        return
    typer.echo(
        f"{'TOOL':<18} {'ARGUMENT':<14} {'KIND':<10} {'DETAIL':<28} "
        f"{'APPLIED':>7} {'SUCCESS':>8} {'ACTIVE':<6}"
    )
    for g in guards:
        detail = g.transform if g.kind == "transform" else repr(g.patch_value)
        typer.echo(
            f"{_truncate(g.tool_name, 18):<18} {_truncate(g.argument, 14):<14} "
            f"{g.kind:<10} {_truncate(detail, 28):<28} "
            f"{g.times_applied:>7} {g.success_rate:>7.0%} {'yes' if g.active else 'no':<6}"
        )


def _print_guard_detail(guard: StandingGuard) -> None:
    typer.echo(f"tool_name:        {guard.tool_name}")
    typer.echo(f"argument:         {guard.argument}")
    typer.echo(f"kind:             {guard.kind}")
    if guard.kind == "transform":
        typer.echo(f"transform:        {guard.transform}")
    else:
        typer.echo(f"patch_value:      {guard.patch_value!r}")
    typer.echo(f"source_signature: {guard.source_signature}")
    typer.echo(f"root_cause:       {guard.root_cause or '-'}")
    typer.echo(f"active:           {guard.active}")
    typer.echo(f"times_applied:    {guard.times_applied}")
    typer.echo(f"times_succeeded:  {guard.times_succeeded}")
    typer.echo(f"success_rate:     {guard.success_rate:.0%}")
    typer.echo(f"created_at:       {guard.created_at}")
    typer.echo(f"last_applied:     {guard.last_applied or '-'}")
    typer.echo("")


@guards_app.command("list")
def guards_list(
    oracle_path: str = _ORACLE_PATH_OPTION,
    tool_name: str = typer.Option(None, "--tool-name", "-t", help="Filter guards by tool name."),
    all_: bool = typer.Option(False, "--all", help="Include revoked guards (default: active only)."),
    limit: int = typer.Option(50, "--limit", "-n", help="Maximum rows to show."),
) -> None:
    """List standing guards in the oracle."""
    with Oracle(oracle_path) as oracle:
        guards = GuardManager(oracle).list(tool_name=tool_name, active_only=not all_, limit=limit)
    _print_guards_table(guards)


@guards_app.command("inspect")
def guards_inspect(
    tool_name: str = typer.Argument(..., help="The tool name the guard applies to."),
    argument: str = typer.Argument(..., help="The argument name the guard applies to."),
    kind: str = typer.Option(
        None, "--kind", help='Restrict to "transform" or "patch" — omit to show all matches.'
    ),
    oracle_path: str = _ORACLE_PATH_OPTION,
) -> None:
    """Show full detail for the guard(s) matching TOOL_NAME and ARGUMENT."""
    with Oracle(oracle_path) as oracle:
        matches = [
            g
            for g in GuardManager(oracle).list(tool_name=tool_name, active_only=False, limit=10_000)
            if g.argument == argument and (kind is None or g.kind == kind)
        ]
    if not matches:
        typer.echo(f"No guard found matching tool_name={tool_name!r}, argument={argument!r}.")
        raise typer.Exit(code=1)
    if len(matches) > 1:
        typer.echo(f"{len(matches)} guards match — showing all:\n")
    for guard in matches:
        _print_guard_detail(guard)


@guards_app.command("revoke")
def guards_revoke(
    tool_name: str = typer.Argument(..., help="The tool name the guard applies to."),
    argument: str = typer.Argument(..., help="The argument name the guard applies to."),
    kind: str = typer.Option(
        None, "--kind", help='Restrict to "transform" or "patch" — omit to revoke both.'
    ),
    oracle_path: str = _ORACLE_PATH_OPTION,
) -> None:
    """Deactivate the standing guard(s) matching TOOL_NAME and ARGUMENT.

    Revocation is sticky: a revoked guard will not be silently reactivated
    by future automatic promotion (see oracle/guards.py's GuardManager.promote)."""
    with Oracle(oracle_path) as oracle:
        revoked = GuardManager(oracle).revoke(tool_name, argument, kind=kind)
    if not revoked:
        typer.echo(f"No active guard found matching tool_name={tool_name!r}, argument={argument!r}.")
        return
    typer.echo(f"Revoked {len(revoked)} guard(s):")
    for guard in revoked:
        typer.echo(f"  - {guard.tool_name}({guard.argument}) [{guard.kind}]")


@guards_app.command("describe")
def guards_describe(
    oracle_path: str = _ORACLE_PATH_OPTION,
    tool_name: str = typer.Option(None, "--tool-name", "-t", help="Restrict to one tool."),
) -> None:
    """Print active guards as text — splice this into YOUR OWN system
    prompt if you want the model to see them. ResilientForge never does
    this automatically (see integrations/*.py, neither of which has
    any system-prompt access to begin with)."""
    with Oracle(oracle_path) as oracle:
        typer.echo(GuardManager(oracle).describe(tool_name=tool_name))


if __name__ == "__main__":
    app()
