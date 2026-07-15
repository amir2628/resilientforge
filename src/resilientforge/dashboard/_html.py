"""The dashboard's entire front end: one inlined HTML/CSS/vanilla-JS
string, no build step, no CDN. Kept in its own module (not a separate
static file) purely for readability — `app.py` returning this constant
avoids any hatchling package-data/MANIFEST configuration, keeping
`pip install` exactly as simple as every other phase's fresh-venv
verification cared about.

Table search/sort/pagination is done client-side over the fetched rows
(see `makeTable` in the script below), not via new backend offset/sort/
search query params: this dashboard is deliberately read-only and reuses
the exact same `oracle.list_*`/`RecipeManager`/`GuardManager` read paths
`cli/main.py` already uses (see `app.py`'s docstring) rather than adding
new SQL query surface. Recipes/guards are naturally small (deduped by
signature / by (tool, argument, kind)); failures is the only table that
grows with usage, so it's fetched up to `FETCH_LIMIT` rows — instant to
sort/filter client-side at that size for a local single-oracle tool.
"""

from __future__ import annotations

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ResilientForge Dashboard</title>
<style>
  :root {
    color-scheme: light;
    --page:            #f9f9f7;
    --surface-1:       #fcfcfb;
    --text-primary:    #0b0b0b;
    --text-secondary:  #52514e;
    --text-muted:      #898781;
    --gridline:        #e1e0d9;
    --border:          rgba(11,11,11,0.10);
    --accent:          #2a78d6;
    --good:            #0ca30c;
    --warning:         #fab219;
    --serious:         #ec835a;
    --critical:        #d03b3b;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      color-scheme: dark;
      --page:            #0d0d0d;
      --surface-1:       #1a1a19;
      --text-primary:    #ffffff;
      --text-secondary:  #c3c2b7;
      --text-muted:      #898781;
      --gridline:        #2c2c2a;
      --border:          rgba(255,255,255,0.10);
      --accent:          #3987e5;
      --good:            #0ca30c;
      --warning:         #fab219;
      --serious:         #ec835a;
      --critical:        #e66767;
    }
  }

  * { box-sizing: border-box; }
  html, body { overflow-x: hidden; }
  body {
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    background: var(--page);
    color: var(--text-primary);
    margin: 0 auto;
    max-width: 1100px;
    padding: clamp(1rem, 4vw, 2.5rem) clamp(1rem, 4vw, 1.5rem);
    line-height: 1.45;
  }

  header { display: flex; align-items: flex-start; justify-content: space-between;
           gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
  h1 { margin: 0 0 0.35rem; font-size: clamp(1.3rem, 3vw, 1.6rem); font-weight: 650; }
  .oracle-chip { display: inline-flex; align-items: center; gap: 0.4em; font-size: 0.8rem;
         color: var(--text-secondary); background: var(--surface-1); border: 1px solid var(--border);
         border-radius: 999px; padding: 0.3rem 0.75rem; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
         max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  #refresh-btn { flex: 0 0 auto; display: inline-flex; align-items: center; gap: 0.4em;
         background: var(--surface-1); color: var(--text-secondary); border: 1px solid var(--border);
         border-radius: 8px; padding: 0.45rem 0.85rem; font-size: 0.82rem; cursor: pointer;
         font-family: inherit; }
  #refresh-btn:hover { color: var(--text-primary); border-color: var(--text-muted); }
  #refresh-btn .spin { display: inline-block; }
  #refresh-btn.loading .spin { animation: spin 0.7s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  #last-updated { font-size: 0.72rem; color: var(--text-muted); margin-top: 0.3rem; text-align: right; }

  #error-banner { display: none; background: color-mix(in srgb, var(--critical) 12%, var(--surface-1));
         border: 1px solid var(--critical); color: var(--text-primary); border-radius: 8px;
         padding: 0.7rem 1rem; margin-bottom: 1.25rem; font-size: 0.88rem; }

  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
           gap: 0.75rem; margin: 0 0 2rem; }
  .stat { border: 1px solid var(--border); background: var(--surface-1); border-radius: 12px;
          padding: 0.85rem 1.1rem; }
  .stat .n { font-size: 1.7rem; font-weight: 650; display: block; }
  .stat .label { font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase;
                 letter-spacing: 0.03em; }
  .stat .sub { font-size: 0.78rem; color: var(--text-secondary); margin-top: 0.15rem; }

  section { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px;
            padding: 1.1rem 1.25rem 0.9rem; margin-bottom: 1.5rem; }
  section h2 { margin: 0 0 0.85rem; font-size: 1rem; font-weight: 650;
               border-bottom: 1px solid var(--gridline); padding-bottom: 0.6rem; }

  .table-toolbar { display: flex; justify-content: flex-end; margin-bottom: 0.6rem; }
  .table-search { width: 100%; max-width: 240px; font: inherit; font-size: 0.85rem;
         color: var(--text-primary); background: var(--page); border: 1px solid var(--border);
         border-radius: 8px; padding: 0.4rem 0.7rem; }
  .table-search:focus { outline: 2px solid var(--accent); outline-offset: -1px; }

  .table-scroll { overflow-x: auto; margin: 0 -0.1rem; }
  table { border-collapse: collapse; width: 100%; min-width: 480px; font-size: 0.86rem; }
  th, td { text-align: left; padding: 0.5rem 0.6rem; border-bottom: 1px solid var(--gridline);
           white-space: nowrap; }
  td.wrap { white-space: normal; word-break: break-word; min-width: 220px; }
  th { color: var(--text-muted); font-weight: 600; font-size: 0.7rem; text-transform: uppercase;
       letter-spacing: 0.03em; }
  th.sortable { cursor: pointer; user-select: none; }
  th.sortable:hover { color: var(--text-primary); }
  th .sort-arrow { color: var(--accent); font-size: 0.75em; margin-left: 0.2em; display: inline-block; width: 0.8em; }
  tbody tr:hover { background: color-mix(in srgb, var(--text-primary) 4%, transparent); }
  td.num, th.num { font-variant-numeric: tabular-nums; }
  .empty-row td { color: var(--text-muted); font-style: italic; white-space: normal; }

  .table-pager { display: flex; align-items: center; justify-content: space-between; gap: 0.75rem;
         flex-wrap: wrap; margin-top: 0.65rem; font-size: 0.78rem; color: var(--text-muted); }
  .pager-btns { display: flex; align-items: center; gap: 0.5rem; }
  .pager-btns button { font: inherit; font-size: 0.78rem; color: var(--text-secondary);
         background: var(--page); border: 1px solid var(--border); border-radius: 6px;
         padding: 0.3rem 0.65rem; cursor: pointer; }
  .pager-btns button:hover:not(:disabled) { color: var(--text-primary); border-color: var(--text-muted); }
  .pager-btns button:disabled { opacity: 0.4; cursor: not-allowed; }

  code { font-size: 0.85em; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }

  .meter { display: flex; align-items: center; gap: 0.5rem; }
  .meter-track { flex: 0 0 68px; width: 68px; height: 7px; border-radius: 4px;
                 background: var(--gridline); overflow: hidden; }
  .meter-fill { display: block; height: 100%; border-radius: 4px; min-width: 2px; }
  .meter-fill.good { background: var(--good); }
  .meter-fill.warning { background: var(--warning); }
  .meter-fill.critical { background: var(--critical); }
  .meter-pct { font-variant-numeric: tabular-nums; color: var(--text-secondary); min-width: 2.6em; }

  .status { display: inline-flex; align-items: center; gap: 0.45em; color: var(--text-secondary); }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; flex: 0 0 8px; }
  .status-dot.good { background: var(--good); }
  .status-dot.warning { background: var(--warning); }
  .status-dot.serious { background: var(--serious); }
  .status-dot.critical { background: var(--critical); }
  .status-dot.muted { background: var(--text-muted); }

  @media (max-width: 480px) {
    body { font-size: 0.94rem; }
    .stats { grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); }
    .stat .n { font-size: 1.4rem; }
    section { padding: 0.9rem 0.9rem 0.7rem; }
    .table-toolbar { justify-content: stretch; }
    .table-search { max-width: none; }
  }
</style>
</head>
<body>
<header>
  <div>
    <h1>ResilientForge Dashboard</h1>
    <span class="oracle-chip" id="oracle-path">loading…</span>
  </div>
  <div>
    <button id="refresh-btn" onclick="load()"><span class="spin">&#8635;</span> Refresh</button>
    <div id="last-updated"></div>
  </div>
</header>

<div id="error-banner"></div>

<div class="stats" id="stats"></div>

<section>
  <h2>Recipes</h2>
  <div id="recipes-table"></div>
</section>

<section>
  <h2>Guards</h2>
  <div id="guards-table"></div>
</section>

<section>
  <h2>Recent failures</h2>
  <div id="failures-table"></div>
</section>

<script>
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}

function meterSeverity(x) {
  if (x >= 0.8) return "good";
  if (x >= 0.5) return "warning";
  return "critical";
}

function meter(x) {
  const v = x || 0;
  const pct = Math.round(v * 100);
  const sev = meterSeverity(v);
  return `<div class="meter"><div class="meter-track"><div class="meter-fill ${sev}" style="width:${pct}%"></div></div><span class="meter-pct">${pct}%</span></div>`;
}

const STATUS_SEVERITY = { recovered: "good", exhausted: "critical", aborted: "serious", unresolved: "warning" };

function statusBadge(status) {
  const sev = STATUS_SEVERITY[status] || "muted";
  return `<span class="status"><span class="status-dot ${sev}"></span>${escapeHtml(status)}</span>`;
}

function activeBadge(active) {
  return `<span class="status"><span class="status-dot ${active ? "good" : "muted"}"></span>${active ? "yes" : "no"}</span>`;
}

// A small, dependency-free table controller: client-side search, click-to-sort
// columns, and pagination over whatever rows setRows() is given. Shared by all
// three dashboard tables instead of duplicating this logic per table.
function makeTable({ mountId, columns, searchText, emptyText, initialSort, pageSize = 10 }) {
  const root = document.getElementById(mountId);
  root.innerHTML = `
    <div class="table-toolbar"><input type="search" class="table-search" placeholder="Search…"></div>
    <div class="table-scroll"><table><thead><tr>${columns.map(c => `
      <th class="${c.numeric ? "num" : ""} ${c.sortable ? "sortable" : ""}" data-key="${c.key}">${escapeHtml(c.label)}<span class="sort-arrow"></span></th>
    `).join("")}</tr></thead><tbody></tbody></table></div>
    <div class="table-pager"></div>`;

  const searchInput = root.querySelector(".table-search");
  const tbody = root.querySelector("tbody");
  const pager = root.querySelector(".table-pager");

  let rows = [];
  let sortKey = initialSort ? initialSort.key : null;
  let sortDir = initialSort ? initialSort.dir : 1;
  let page = 1;

  root.querySelectorAll("th.sortable").forEach(th => {
    th.addEventListener("click", () => {
      const key = th.dataset.key;
      sortDir = sortKey === key ? sortDir * -1 : 1;
      sortKey = key;
      page = 1;
      render();
    });
  });
  searchInput.addEventListener("input", () => { page = 1; render(); });

  function render() {
    const term = searchInput.value.trim().toLowerCase();
    let view = term ? rows.filter(r => r._search.includes(term)) : rows.slice();

    if (sortKey) {
      const col = columns.find(c => c.key === sortKey);
      view.sort((a, b) => {
        const av = col.value(a), bv = col.value(b);
        return av < bv ? -sortDir : av > bv ? sortDir : 0;
      });
    }

    root.querySelectorAll("th .sort-arrow").forEach(el => el.textContent = "");
    if (sortKey) {
      const arrow = root.querySelector(`th[data-key="${sortKey}"] .sort-arrow`);
      if (arrow) arrow.textContent = sortDir === 1 ? "▲" : "▼";
    }

    const total = view.length;
    const totalPages = Math.max(1, Math.ceil(total / pageSize));
    page = Math.min(page, totalPages);
    const start = (page - 1) * pageSize;
    const pageRows = view.slice(start, start + pageSize);

    tbody.innerHTML = pageRows.length
      ? pageRows.map(r => `<tr>${columns.map(c => `<td class="${c.numeric ? "num" : ""} ${c.wrap ? "wrap" : ""}">${c.html(r)}</td>`).join("")}</tr>`).join("")
      : `<tr class="empty-row"><td colspan="${columns.length}">${rows.length ? "No matching rows." : emptyText}</td></tr>`;

    const from = total === 0 ? 0 : start + 1;
    const to = Math.min(start + pageSize, total);
    pager.innerHTML = `
      <span>${from}–${to} of ${total}</span>
      <div class="pager-btns">
        <button data-act="prev" ${page <= 1 ? "disabled" : ""}>Prev</button>
        <span>Page ${page} / ${totalPages}</span>
        <button data-act="next" ${page >= totalPages ? "disabled" : ""}>Next</button>
      </div>`;
    pager.querySelector('[data-act="prev"]').addEventListener("click", () => { page--; render(); });
    pager.querySelector('[data-act="next"]').addEventListener("click", () => { page++; render(); });
  }

  return {
    setRows(newRows) {
      rows = newRows.map(r => ({ ...r, _search: searchText(r).toLowerCase() }));
      page = 1;
      render();
    },
  };
}

const recipesTable = makeTable({
  mountId: "recipes-table",
  emptyText: "No recipes yet.",
  initialSort: { key: "last_used", dir: -1 },
  searchText: r => [r.signature, r.tool_name, r.root_cause, r.fix_strategy].filter(Boolean).join(" "),
  columns: [
    { key: "signature", label: "Signature", sortable: true, wrap: true, value: r => r.signature.toLowerCase(), html: r => `<code>${escapeHtml(r.signature)}</code>` },
    { key: "tool_name", label: "Tool", sortable: true, value: r => r.tool_name.toLowerCase(), html: r => escapeHtml(r.tool_name) },
    { key: "times_applied", label: "Applied", sortable: true, numeric: true, value: r => r.times_applied, html: r => r.times_applied },
    { key: "success_rate", label: "Success rate", sortable: true, value: r => r.success_rate, html: r => meter(r.success_rate) },
    { key: "last_used", label: "Last used", sortable: true, value: r => r.last_used, html: r => escapeHtml(r.last_used) },
  ],
});

const guardsTable = makeTable({
  mountId: "guards-table",
  emptyText: "No guards yet.",
  initialSort: { key: "tool_name", dir: 1 },
  searchText: g => [g.tool_name, g.argument, g.kind].filter(Boolean).join(" "),
  columns: [
    { key: "tool_name", label: "Tool", sortable: true, value: g => g.tool_name.toLowerCase(), html: g => escapeHtml(g.tool_name) },
    { key: "argument", label: "Argument", sortable: true, value: g => g.argument.toLowerCase(), html: g => escapeHtml(g.argument) },
    { key: "kind", label: "Kind", sortable: true, value: g => g.kind.toLowerCase(), html: g => escapeHtml(g.kind) },
    { key: "times_applied", label: "Applied", sortable: true, numeric: true, value: g => g.times_applied, html: g => g.times_applied },
    { key: "success_rate", label: "Success rate", sortable: true, value: g => g.success_rate, html: g => meter(g.success_rate) },
    { key: "active", label: "Active", sortable: true, value: g => (g.active ? 1 : 0), html: g => activeBadge(g.active) },
  ],
});

const failuresTable = makeTable({
  mountId: "failures-table",
  emptyText: "No failures recorded.",
  initialSort: { key: "id", dir: -1 },
  searchText: f => [f.tool_name, f.resolution_status, f.error_type, f.error_message].filter(Boolean).join(" "),
  columns: [
    { key: "id", label: "ID", sortable: true, numeric: true, value: f => f.id, html: f => f.id },
    { key: "tool_name", label: "Tool", sortable: true, value: f => f.tool_name.toLowerCase(), html: f => escapeHtml(f.tool_name) },
    { key: "resolution_status", label: "Status", sortable: true, value: f => f.resolution_status, html: f => statusBadge(f.resolution_status) },
    { key: "error_type", label: "Error type", sortable: true, value: f => (f.error_type || "").toLowerCase(), html: f => escapeHtml(f.error_type || "-") },
    { key: "created_at", label: "Created", sortable: true, value: f => f.created_at, html: f => escapeHtml(f.created_at) },
  ],
});

const FETCH_LIMIT = 500;

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} -> HTTP ${res.status}`);
  return res.json();
}

async function load() {
  const btn = document.getElementById("refresh-btn");
  const errorBanner = document.getElementById("error-banner");
  btn.classList.add("loading");
  try {
    const stats = await fetchJson("/api/stats");
    document.getElementById("oracle-path").textContent = stats.oracle_path;

    const failureBreakdown = Object.entries(stats.failures_by_status || {})
      .map(([status, n]) => `${n} ${status}`).join(" · ") || "none yet";

    document.getElementById("stats").innerHTML = [
      ["Recipes", stats.recipe_count, "learned fixes"],
      ["Failures", stats.failure_count, failureBreakdown],
      ["Guards", `${stats.active_guard_count} / ${stats.guard_count}`, "active"],
    ].map(([label, n, sub]) => `<div class="stat">
        <span class="n">${escapeHtml(n)}</span>
        <span class="label">${escapeHtml(label)}</span>
        <div class="sub">${escapeHtml(sub)}</div>
      </div>`).join("");

    recipesTable.setRows(await fetchJson(`/api/recipes?limit=${FETCH_LIMIT}`));
    guardsTable.setRows(await fetchJson(`/api/guards?active_only=false&limit=${FETCH_LIMIT}`));
    failuresTable.setRows(await fetchJson(`/api/failures?limit=${FETCH_LIMIT}`));

    errorBanner.style.display = "none";
    document.getElementById("last-updated").textContent = "updated " + new Date().toLocaleTimeString();
  } catch (err) {
    errorBanner.textContent = "Couldn't load dashboard data: " + err.message;
    errorBanner.style.display = "block";
  } finally {
    btn.classList.remove("loading");
  }
}

load();
setInterval(load, 8000);
</script>
</body>
</html>
"""
