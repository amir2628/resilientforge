"""The dashboard's entire front end: one inlined HTML/CSS/vanilla-JS
string, no build step, no CDN. Kept in its own module (not a separate
static file) purely for readability — `app.py` returning this constant
avoids any hatchling package-data/MANIFEST configuration, keeping
`pip install` exactly as simple as every other phase's fresh-venv
verification cared about."""

from __future__ import annotations

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ResilientForge Dashboard</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 2rem auto; max-width: 1100px; padding: 0 1rem; }
  h1 { margin-bottom: 0.25rem; }
  .subtitle { color: #888; margin-top: 0; font-size: 0.9rem; }
  .stats { display: flex; gap: 1.25rem; margin: 1.5rem 0; flex-wrap: wrap; }
  .stat { border: 1px solid #88888855; border-radius: 8px; padding: 0.6rem 1.1rem; }
  .stat .n { font-size: 1.5rem; font-weight: 600; display: block; }
  .stat .label { font-size: 0.75rem; color: #888; }
  table { border-collapse: collapse; width: 100%; margin-bottom: 2rem; font-size: 0.88rem; }
  th, td { text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid #88888833; }
  th { color: #888; font-weight: 600; font-size: 0.75rem; text-transform: uppercase; }
  .bar { background: #88888833; border-radius: 4px; height: 8px; overflow: hidden;
         width: 70px; display: inline-block; vertical-align: middle; margin-right: 6px; }
  .bar-fill { background: #44aa77; height: 100%; }
  code { font-size: 0.85em; }
  section h2 { border-bottom: 1px solid #88888833; padding-bottom: 0.3rem; font-size: 1.1rem; }
</style>
</head>
<body>
<h1>ResilientForge Dashboard</h1>
<p class="subtitle" id="oracle-path">loading…</p>

<div class="stats" id="stats"></div>

<section>
  <h2>Recipes</h2>
  <table id="recipes-table"><thead><tr>
    <th>Signature</th><th>Tool</th><th>Applied</th><th>Success rate</th><th>Last used</th>
  </tr></thead><tbody></tbody></table>
</section>

<section>
  <h2>Guards</h2>
  <table id="guards-table"><thead><tr>
    <th>Tool</th><th>Argument</th><th>Kind</th><th>Applied</th><th>Success rate</th><th>Active</th>
  </tr></thead><tbody></tbody></table>
</section>

<section>
  <h2>Recent failures</h2>
  <table id="failures-table"><thead><tr>
    <th>ID</th><th>Tool</th><th>Status</th><th>Error type</th><th>Created</th>
  </tr></thead><tbody></tbody></table>
</section>

<script>
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}
function pct(x) { return Math.round((x || 0) * 100) + "%"; }
function bar(x) {
  return `<span class="bar"><span class="bar-fill" style="width:${Math.round((x || 0) * 100)}%"></span></span>${pct(x)}`;
}

async function load() {
  const stats = await (await fetch("/api/stats")).json();
  document.getElementById("oracle-path").textContent = stats.oracle_path;
  document.getElementById("stats").innerHTML = [
    ["Recipes", stats.recipe_count],
    ["Failures", stats.failure_count],
    ["Guards", `${stats.active_guard_count} / ${stats.guard_count} active`],
  ].map(([label, n]) => `<div class="stat"><span class="n">${escapeHtml(n)}</span><span class="label">${label}</span></div>`).join("");

  const recipes = await (await fetch("/api/recipes?limit=200")).json();
  document.querySelector("#recipes-table tbody").innerHTML = recipes.map(r => `<tr>
    <td><code>${escapeHtml(r.signature)}</code></td>
    <td>${escapeHtml(r.tool_name)}</td>
    <td>${r.times_applied}</td>
    <td>${bar(r.success_rate)}</td>
    <td>${escapeHtml(r.last_used)}</td>
  </tr>`).join("") || "<tr><td colspan=5>No recipes yet.</td></tr>";

  const guards = await (await fetch("/api/guards?active_only=false&limit=200")).json();
  document.querySelector("#guards-table tbody").innerHTML = guards.map(g => `<tr>
    <td>${escapeHtml(g.tool_name)}</td>
    <td>${escapeHtml(g.argument)}</td>
    <td>${escapeHtml(g.kind)}</td>
    <td>${g.times_applied}</td>
    <td>${bar(g.success_rate)}</td>
    <td>${g.active ? "yes" : "no"}</td>
  </tr>`).join("") || "<tr><td colspan=6>No guards yet.</td></tr>";

  const failures = await (await fetch("/api/failures?limit=200")).json();
  document.querySelector("#failures-table tbody").innerHTML = failures.map(f => `<tr>
    <td>${f.id}</td>
    <td>${escapeHtml(f.tool_name)}</td>
    <td>${escapeHtml(f.resolution_status)}</td>
    <td>${escapeHtml(f.error_type || "-")}</td>
    <td>${escapeHtml(f.created_at)}</td>
  </tr>`).join("") || "<tr><td colspan=5>No failures recorded.</td></tr>";
}
load();
</script>
</body>
</html>
"""
