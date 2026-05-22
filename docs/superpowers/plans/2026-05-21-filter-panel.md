# Filter & Overlay Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Filters & Overlays" section to both 2D (Cosmograph) and 3D (3d-force-graph) viewers, expand search to match oracle text, and upgrade cluster labels to use an LLM. All on the existing 5k card set; no scale changes.

**Architecture:** Three Python changes (load_and_filter adds fields, visualize emits them in `point_meta`, cluster adds an LLM-label step), then per-viewer JS additions for the new UI. Filter state lives in JS as a single mutable object; each filter change recomputes per-card `visible` and `colorRGBA` arrays which get pushed to the viewer's API.

**Tech Stack:** Python 3.11 (uv), OpenAI `gpt-4o-mini` for cluster names, JS in browser (Cosmograph `@cosmos.gl/graph@2.6.4`, `3d-force-graph@1.73.4`, `three@0.160.0`).

**Spec:** `docs/superpowers/specs/2026-05-21-filter-panel-design.md`

**Testing approach:** Per established convention, no unit tests — verification is "run pipeline, open HTML, eyeball each filter."

---

## File Structure

Files this plan creates or modifies:

```
src/mtg_graph/load_and_filter.py        # add cmc, legalities, set code
src/mtg_graph/visualize.py              # include new fields in point_meta
src/mtg_graph/cluster.py                # optional LLM-label final step
src/mtg_graph/templates/viewer_2d.html  # filter panel UI + JS
src/mtg_graph/templates/viewer_3d.html  # filter panel UI + JS
```

No new files. All changes are additions to existing modules; each module retains its single responsibility.

---

## Task 1: Add cmc / legalities / set fields to load_and_filter

**Files:**
- Modify: `src/mtg_graph/load_and_filter.py`

- [ ] **Step 1: Update the kept columns list**

Open `src/mtg_graph/load_and_filter.py`. Find the `cols = [...]` list inside `run()` and replace it with:

```python
    cols = [
        "oracle_id", "name", "mana_cost", "cmc", "type_line", "oracle_text",
        "flavor_text", "keywords", "colors", "color_identity", "rarity",
        "set", "set_name", "legalities", "edhrec_rank",
        "image_small", "image_normal", "image_art_crop",
        "price_usd", "price_usd_foil",
        "scryfall_uri",
    ]
```

(Three additions: `cmc`, `set` short code, `legalities` dict.)

- [ ] **Step 2: Run the stage and verify the parquet has the new fields**

Run: `uv run python -m mtg_graph.load_and_filter`

Then:

```bash
uv run python -c "
import pandas as pd
df = pd.read_parquet('output/cards_top5k.parquet')
print('Columns:', list(df.columns))
print()
print('Sample (Sol Ring):')
sample = df[df['name'] == 'Sol Ring'].iloc[0]
print('  cmc:', sample['cmc'])
print('  set:', sample['set'])
print('  legalities formats:', list(sample['legalities'].keys())[:5])
print('  modern legality:', sample['legalities'].get('modern'))
"
```

Expected: columns include `cmc`, `set`, `legalities`. Sol Ring's cmc = 1.0, set = a short code like "lea" or "c14", legalities is a dict with format keys.

- [ ] **Step 3: Re-run downstream stages so the changes flow through**

Run: `uv run python -m mtg_graph.build_text_profiles`

Expected: completes; the text_profile column doesn't depend on the new fields (it uses name/type/mana_cost/oracle_text only).

- [ ] **Step 4: Commit**

```bash
git add src/mtg_graph/load_and_filter.py
git commit -m "Keep cmc, set code, and legalities in the cards parquet

Needed by the filter panel (Phase 1) — format filter reads legalities,
set filter reads set, mana value filter and overlay read cmc.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Include new fields in point_meta serialization

**Files:**
- Modify: `src/mtg_graph/visualize.py`

- [ ] **Step 1: Update Cosmograph meta_fields list**

In `_build_cosmograph_data()`, find:

```python
    meta_fields = [
        "oracle_id", "name", "mana_cost", "type_line", "oracle_text",
        "flavor_text", "rarity", "set_name", "edhrec_rank",
        "image_small", "image_normal", "image_art_crop",
        "price_usd", "price_usd_foil", "scryfall_uri",
    ]
```

Replace with:

```python
    meta_fields = [
        "oracle_id", "name", "mana_cost", "cmc", "type_line", "oracle_text",
        "flavor_text", "rarity", "set", "set_name", "legalities", "edhrec_rank",
        "image_small", "image_normal", "image_art_crop",
        "price_usd", "price_usd_foil", "scryfall_uri",
    ]
```

- [ ] **Step 2: Same update inside `_build_forcegraph_data()`**

Find the identical `meta_fields = [...]` list inside `_build_forcegraph_data()` and apply the same replacement.

- [ ] **Step 3: Handle the legalities dict serialization**

The `point_meta` builder currently does `isinstance(val, (list, np.ndarray))` for list-like values but `legalities` is a dict — falls through to the `else` branch. That's fine (`val` gets passed through as-is and JSON-serializes the dict cleanly). Just verify by running:

```bash
uv run python -m mtg_graph.visualize 2>&1 | tail -5
```

Then:

```bash
uv run python -c "
import json, re
html = open('output/mtg_graph_poc.html').read()
m = re.search(r'<script id=\"point-meta\" type=\"application/json\">(.+?)</script>', html, re.DOTALL)
data = json.loads(m.group(1).replace('<\\\\/','</'))
print('First card:', data[0]['name'])
print('  cmc:', data[0]['cmc'])
print('  set:', data[0]['set'])
print('  legalities modern:', data[0]['legalities'].get('modern'))
"
```

Expected: First card (likely "Sol Ring") prints with cmc=1.0, set as a short string, modern legality string.

- [ ] **Step 4: Commit**

```bash
git add src/mtg_graph/visualize.py
git commit -m "Include cmc, set, legalities in point_meta JSON

These flow through to the filter panel JS (Phase 1).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: LLM cluster labels in cluster.py

**Files:**
- Modify: `src/mtg_graph/cluster.py`

- [ ] **Step 1: Add the LLM relabeling function**

In `src/mtg_graph/cluster.py`, add this function above the existing `run()`:

```python
LLM_MODEL = "gpt-4o-mini"
LLM_PROMPT = """You are labeling clusters of Magic: The Gathering cards.

Given the cards below from one cluster, produce a short label (3–7 words) that
captures what they have in common. Prefer mechanic names, tribal names, or
specific effect categories. Avoid generic terms like "cards" or "spells".

Examples of good labels:
- "Counterspells"
- "Ninja tribal & Ninjutsu"
- "Treasure-token generators"
- "Mass exile sweepers"
- "Cycling lands"
- "Fetch lands (search & sacrifice)"

Cards in this cluster:
{cards}

Hint terms from this cluster (may be noisy): {hints}

Reply with ONLY the label, no preamble, no quotes."""


def _llm_label_clusters(
    df: pd.DataFrame,
    cluster_ids: np.ndarray,
    tfidf_labels: dict[int, str],
) -> dict[int, str]:
    """Use an LLM to give each cluster a natural-language name.
    Returns dict[int, str]. Falls back to TF-IDF label on per-cluster API error."""
    try:
        from openai import OpenAI
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        return tfidf_labels

    import os
    if not os.environ.get("OPENAI_API_KEY"):
        print("    LLM labels skipped (OPENAI_API_KEY not set) — using TF-IDF")
        return tfidf_labels

    client = OpenAI()
    out: dict[int, str] = {}
    unique = [c for c in np.unique(cluster_ids) if c != -1 and c in {int(k) for k in tfidf_labels.keys()}]
    print(f"    LLM-labeling {len(unique)} clusters via {LLM_MODEL}...")
    for cid in unique:
        sub = df[cluster_ids == cid].sort_values("edhrec_rank").head(8)
        cards_block = "\n".join(
            f"- {r['name']} ({r['type_line']}): {(r.get('oracle_text') or '').replace(chr(10), ' ')[:160]}"
            for _, r in sub.iterrows()
        )
        prompt = LLM_PROMPT.format(cards=cards_block, hints=tfidf_labels.get(int(cid), ""))
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=24,
            )
            label = resp.choices[0].message.content.strip().strip('"').strip("'")
            out[int(cid)] = label or tfidf_labels[int(cid)]
        except Exception as e:
            print(f"    cluster {cid}: LLM call failed ({e}); falling back to TF-IDF")
            out[int(cid)] = tfidf_labels[int(cid)]
    return out
```

- [ ] **Step 2: Wire the LLM step into `run()`**

In `src/mtg_graph/cluster.py`, find the existing loop:

```python
    for label in ("local", "openai"):
        x_col, y_col = f"x_{label}", f"y_{label}"
        if x_col not in merged.columns:
            print(f"  No {label} UMAP coords — skipping")
            continue
        xy = merged[[x_col, y_col]].to_numpy()
        cids = _run_hdbscan(xy)
        clusters_df[f"cluster_{label}"] = cids
        cluster_labels = _label_clusters(merged["text_profile"].tolist(), cids)
        all_labels[label] = {str(k): v for k, v in cluster_labels.items()}
        print(f"    {len(cluster_labels)} labeled clusters")
```

Replace the body of the loop's bottom with the LLM relabeling step. Replace from `cluster_labels = _label_clusters(...)` through `print(f"    {len(cluster_labels)} labeled clusters")` with:

```python
        cluster_labels = _label_clusters(merged["text_profile"].tolist(), cids)
        cluster_labels = _llm_label_clusters(merged, cids, cluster_labels)
        all_labels[label] = {str(k): v for k, v in cluster_labels.items()}
        print(f"    {len(cluster_labels)} labeled clusters")
```

- [ ] **Step 3: Run the cluster stage with API key set**

Run: `uv run python -m mtg_graph.cluster`

Expected:
```
  HDBSCAN on shape (5000, 2)...
    -> ~100 clusters, ~1500 noise points (~30%)
    LLM-labeling ~78 clusters via gpt-4o-mini...
    78 labeled clusters
  ...
  Wrote output/clusters.parquet
  Wrote output/cluster_labels.json
```

This takes ~30–60 seconds (one API call per cluster, ~78 clusters).

- [ ] **Step 4: Eyeball the new labels**

Run:

```bash
uv run python -c "
import json
labels = json.load(open('output/cluster_labels.json'))['local']
print(f'Total: {len(labels)}')
print()
for cid, label in list(labels.items())[:25]:
    print(f'  cluster {cid}: {label}')
"
```

Expected: labels read more like coherent phrases ("Ninjutsu & ninja tribal" vs the prior "ninja / ninjutsu"). Some clusters may keep TF-IDF labels (cards with weird text — fine).

- [ ] **Step 5: Commit**

```bash
git add src/mtg_graph/cluster.py
git commit -m "Add LLM-named clusters via gpt-4o-mini

Cluster labels go from TF-IDF terms ('ninja / ninjutsu') to natural
phrases ('Ninjutsu & ninja tribal'). Falls back to TF-IDF when no
OPENAI_API_KEY is set or per-cluster API call fails.

Cost: ~$0.01 for ~80 clusters at gpt-4o-mini pricing.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: 2D Cosmograph — filter panel UI + JS

**Files:**
- Modify: `src/mtg_graph/templates/viewer_2d.html`

This task adds the entire filter UI + logic to the 2D viewer. It's a big task — break it into the substeps below.

- [ ] **Step 1: Add filter panel CSS**

In `src/mtg_graph/templates/viewer_2d.html`, find the CSS block. Locate this line:

```css
.search-row { position: relative; margin-bottom: 16px; }
```

Insert these styles directly before that line:

```css
.filter-section { margin-bottom: 14px; padding-bottom: 10px; border-bottom: 1px solid var(--panel-border); }
.filter-header { display: flex; justify-content: space-between; align-items: center; font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-dim); cursor: pointer; padding: 4px 0; user-select: none; }
.filter-header:hover { color: var(--text); }
.filter-header .chev { font-size: 10px; font-family: monospace; }
.filter-section.collapsed .filter-body { display: none; }
.filter-body { padding: 8px 0 4px; }
.filter-row { margin-bottom: 12px; }
.filter-row:last-child { margin-bottom: 0; }
.filter-row label { display: block; font-size: 11px; color: var(--text-dim); margin-bottom: 4px; }
.filter-row select, .filter-row input[type=text] {
  width: 100%; background: var(--input-bg); color: var(--text);
  border: 1px solid var(--panel-border); padding: 5px 8px;
  font-size: 12px; border-radius: 3px; outline: none;
}
.filter-row select:focus, .filter-row input[type=text]:focus { border-color: var(--accent); }
.format-chips, .color-chips { display: flex; flex-wrap: wrap; gap: 4px; }
.chip {
  font-size: 11px; padding: 3px 8px; border-radius: 11px;
  background: var(--input-bg); color: var(--text-dim);
  border: 1px solid var(--panel-border); cursor: pointer;
  user-select: none;
}
.chip.active { color: var(--text); border-color: var(--accent); background: rgba(201,162,39,0.1); }
.chip.color-chip { width: 22px; height: 22px; padding: 0; border-radius: 50%; border: 2px solid transparent; position: relative; }
.chip.color-chip.off { opacity: 0.25; border-color: var(--panel-border); }
.chip.color-chip[title]:hover::after {
  content: attr(title); position: absolute; bottom: 110%; left: 50%; transform: translateX(-50%);
  background: var(--input-bg); color: var(--text); padding: 3px 7px; border-radius: 3px;
  font-size: 10px; white-space: nowrap; border: 1px solid var(--panel-border);
}
.cmc-row { display: flex; align-items: center; gap: 8px; }
.cmc-row input[type=range] { flex: 1; accent-color: var(--accent); }
.cmc-row .cmc-vals { font-size: 11px; color: var(--text-dim); min-width: 32px; text-align: right; }
.set-options { max-height: 140px; overflow-y: auto; background: var(--input-bg); border: 1px solid var(--panel-border); border-radius: 3px; margin-top: 4px; display: none; }
.set-options.open { display: block; }
.set-option { padding: 4px 8px; font-size: 11px; cursor: pointer; }
.set-option:hover { background: var(--panel-border); }
.set-option.selected { color: var(--accent-bright); }
.selected-sets { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; }
.selected-set-chip { background: rgba(201,162,39,0.15); color: var(--accent-bright); border: 1px solid var(--accent); padding: 2px 6px; border-radius: 3px; font-size: 10px; cursor: pointer; }
.selected-set-chip:hover { background: rgba(201,162,39,0.25); }
.sr-text-hit { color: var(--accent-bright); font-size: 10px; margin-left: 6px; }
```

- [ ] **Step 2: Add filter panel HTML**

In the same file, find the `<div class="search-row">` line. Insert this block directly before it:

```html
      <div class="filter-section" id="filter-section">
        <div class="filter-header" id="filter-header">
          <span>▾ Filters &amp; Overlays</span>
        </div>
        <div class="filter-body">
          <div class="filter-row">
            <label for="color-axis">Color by</label>
            <select id="color-axis">
              <option value="identity">Color identity</option>
              <option value="edhrec">EDHREC rank (popularity)</option>
              <option value="price">Price (USD)</option>
              <option value="type">Card type</option>
              <option value="cmc">Mana value</option>
            </select>
          </div>
          <div class="filter-row">
            <label>Format</label>
            <div class="format-chips" id="format-chips"></div>
          </div>
          <div class="filter-row">
            <label for="set-input">Set</label>
            <input type="text" id="set-input" placeholder="Type to filter sets…" autocomplete="off" />
            <div class="set-options" id="set-options"></div>
            <div class="selected-sets" id="selected-sets"></div>
          </div>
          <div class="filter-row">
            <label>Mana value <span class="cmc-vals" id="cmc-vals">0–10+</span></label>
            <div class="cmc-row">
              <input type="range" id="cmc-min" min="0" max="11" step="1" value="0" />
              <input type="range" id="cmc-max" min="0" max="11" step="1" value="11" />
            </div>
          </div>
          <div class="filter-row">
            <label>Color identity visible</label>
            <div class="color-chips" id="color-chips"></div>
          </div>
        </div>
      </div>
```

- [ ] **Step 3: Update the search placeholder**

In the same file, find:

```html
<input type="text" id="search" placeholder="Search card name…" autocomplete="off" />
```

Replace with:

```html
<input type="text" id="search" placeholder="Search name or oracle text…" autocomplete="off" />
```

- [ ] **Step 4: Add the JS filter framework — state, helpers, and color computation**

In the `<script type="module">` block, find this line:

```js
let neighborMode = 'semantic';
```

Insert this entire block directly after that line:

```js
// ===== Filter & overlay state =====
const COLOR_BUCKETS = ['White', 'Blue', 'Black', 'Red', 'Green', 'Multicolor', 'Colorless'];
const COLOR_HEX = {
  'White': '#F5EEDC', 'Blue': '#3B82F6', 'Black': '#7A6FA0', 'Red': '#EF4444',
  'Green': '#22C55E', 'Multicolor': '#FBBF24', 'Colorless': '#9CA3AF',
};
const FORMATS = ['standard', 'pioneer', 'modern', 'legacy', 'vintage', 'commander', 'pauper', 'brawl'];

const filters = {
  colorAxis: 'identity',
  formats: new Set(),        // empty = no format filter
  sets: new Set(),
  cmcMin: 0,
  cmcMax: 11,                // 11 = "10+" = no upper limit
  visibleColors: new Set(COLOR_BUCKETS),  // all visible by default
};

function colorBucket(ci) {
  if (!ci || ci.length === 0) return 'Colorless';
  if (ci.length > 1) return 'Multicolor';
  return {W: 'White', U: 'Blue', B: 'Black', R: 'Red', G: 'Green'}[ci[0]] || 'Colorless';
}

function hexToRgb(hex) {
  const h = hex.replace('#', '');
  return [parseInt(h.slice(0, 2), 16) / 255, parseInt(h.slice(2, 4), 16) / 255, parseInt(h.slice(4, 6), 16) / 255];
}

// Primary card type extracted from "Legendary Creature — Elf Druid"
function primaryType(typeLine) {
  if (!typeLine) return 'Other';
  const t = typeLine.toLowerCase();
  for (const k of ['land', 'creature', 'planeswalker', 'instant', 'sorcery', 'enchantment', 'artifact']) {
    if (t.includes(k)) return k;
  }
  return 'Other';
}
const TYPE_COLOR = {
  'land': '#9CA3AF', 'creature': '#22C55E', 'planeswalker': '#FBBF24',
  'instant': '#3B82F6', 'sorcery': '#7A6FA0', 'enchantment': '#F5EEDC',
  'artifact': '#A5A5A5', 'Other': '#666666',
};

function viridis(t) {
  // 0..1 → roughly viridis. Hand-tuned 5-stop ramp.
  t = Math.max(0, Math.min(1, t));
  const stops = [
    [0.267, 0.005, 0.329], [0.231, 0.318, 0.545],
    [0.128, 0.567, 0.551], [0.369, 0.788, 0.382],
    [0.993, 0.906, 0.144],
  ];
  const i = Math.min(Math.floor(t * (stops.length - 1)), stops.length - 2);
  const f = t * (stops.length - 1) - i;
  return [
    stops[i][0] + (stops[i+1][0] - stops[i][0]) * f,
    stops[i][1] + (stops[i+1][1] - stops[i][1]) * f,
    stops[i][2] + (stops[i+1][2] - stops[i][2]) * f,
  ];
}

// Precompute min/max for numeric overlays.
const edhrecRanks = POINT_META.map(c => c.edhrec_rank).filter(r => r != null);
const EDHREC_MIN = Math.min(...edhrecRanks);
const EDHREC_MAX = Math.max(...edhrecRanks);
const prices = POINT_META.map(c => parseFloat(c.price_usd)).filter(p => !isNaN(p) && p > 0);
const PRICE_LOG_MIN = Math.log10(Math.min(...prices));
const PRICE_LOG_MAX = Math.log10(Math.max(...prices));
const CMC_OVERLAY_MAX = 8;

function pointRgbFor(card) {
  switch (filters.colorAxis) {
    case 'identity':
      return hexToRgb(COLOR_HEX[colorBucket(card.color_identity)]);
    case 'edhrec': {
      const r = card.edhrec_rank;
      if (r == null) return hexToRgb('#444444');
      const t = 1 - (r - EDHREC_MIN) / (EDHREC_MAX - EDHREC_MIN); // 1 = top-played
      return viridis(t);
    }
    case 'price': {
      const p = parseFloat(card.price_usd);
      if (isNaN(p) || p <= 0) return hexToRgb('#333333');
      const t = (Math.log10(p) - PRICE_LOG_MIN) / (PRICE_LOG_MAX - PRICE_LOG_MIN);
      return viridis(t);
    }
    case 'type':
      return hexToRgb(TYPE_COLOR[primaryType(card.type_line)] || TYPE_COLOR.Other);
    case 'cmc': {
      const c = card.cmc;
      if (c == null) return hexToRgb('#444444');
      const t = Math.min(c, CMC_OVERLAY_MAX) / CMC_OVERLAY_MAX;
      return viridis(t);
    }
  }
  return hexToRgb('#888888');
}

function cardVisible(card) {
  // Color identity visibility
  if (!filters.visibleColors.has(colorBucket(card.color_identity))) return false;
  // Format filter (cards must be legal in AT LEAST ONE selected format)
  if (filters.formats.size > 0) {
    const leg = card.legalities || {};
    let anyOk = false;
    for (const f of filters.formats) {
      if (leg[f] === 'legal') { anyOk = true; break; }
    }
    if (!anyOk) return false;
  }
  // Set filter
  if (filters.sets.size > 0 && !filters.sets.has(card.set)) return false;
  // Mana value range
  const cmc = card.cmc;
  if (cmc != null) {
    if (cmc < filters.cmcMin) return false;
    if (filters.cmcMax < 11 && cmc > filters.cmcMax) return false;
  }
  return true;
}

function applyFilters() {
  const colors = new Float32Array(POINT_META.length * 4);
  for (let i = 0; i < POINT_META.length; i++) {
    const card = POINT_META[i];
    const [r, g, b] = pointRgbFor(card);
    const visible = cardVisible(card);
    colors[i * 4] = r;
    colors[i * 4 + 1] = g;
    colors[i * 4 + 2] = b;
    colors[i * 4 + 3] = visible ? 1.0 : 0.0;
  }
  graph.setPointColors(colors);
}
```

- [ ] **Step 5: Wire up the filter UI controls**

In the same `<script type="module">` block, find the `// Search` comment that begins the search wiring. Insert this entire block directly *before* that comment:

```js
// ===== Filter UI wiring =====

// Color axis dropdown
document.getElementById('color-axis').addEventListener('change', (e) => {
  filters.colorAxis = e.target.value;
  applyFilters();
});

// Format chips
const formatChipsEl = document.getElementById('format-chips');
formatChipsEl.innerHTML = FORMATS.map(f =>
  `<span class="chip" data-format="${f}">${f}</span>`
).join('');
formatChipsEl.querySelectorAll('.chip').forEach(el => {
  el.addEventListener('click', () => {
    const f = el.dataset.format;
    if (filters.formats.has(f)) { filters.formats.delete(f); el.classList.remove('active'); }
    else { filters.formats.add(f); el.classList.add('active'); }
    applyFilters();
  });
});

// Set filter
const SETS_IN_DATA = [...new Set(POINT_META.map(c => c.set).filter(Boolean))].sort();
const setInput = document.getElementById('set-input');
const setOptionsEl = document.getElementById('set-options');
const selectedSetsEl = document.getElementById('selected-sets');

function renderSetOptions(query) {
  const q = query.trim().toLowerCase();
  const matches = SETS_IN_DATA.filter(s => s.includes(q) && !filters.sets.has(s)).slice(0, 30);
  if (!matches.length) { setOptionsEl.classList.remove('open'); return; }
  setOptionsEl.innerHTML = matches.map(s => `<div class="set-option" data-set="${s}">${s}</div>`).join('');
  setOptionsEl.classList.add('open');
  setOptionsEl.querySelectorAll('.set-option').forEach(el => {
    el.addEventListener('click', () => {
      filters.sets.add(el.dataset.set);
      setInput.value = '';
      setOptionsEl.classList.remove('open');
      renderSelectedSets();
      applyFilters();
    });
  });
}
function renderSelectedSets() {
  selectedSetsEl.innerHTML = [...filters.sets].sort().map(s =>
    `<span class="selected-set-chip" data-set="${s}">${s} ×</span>`
  ).join('');
  selectedSetsEl.querySelectorAll('.selected-set-chip').forEach(el => {
    el.addEventListener('click', () => {
      filters.sets.delete(el.dataset.set);
      renderSelectedSets();
      applyFilters();
    });
  });
}
setInput.addEventListener('input', () => renderSetOptions(setInput.value));
setInput.addEventListener('focus', () => renderSetOptions(setInput.value));
document.addEventListener('click', (e) => {
  if (!setInput.contains(e.target) && !setOptionsEl.contains(e.target)) setOptionsEl.classList.remove('open');
});

// Mana value range slider
const cmcMin = document.getElementById('cmc-min');
const cmcMax = document.getElementById('cmc-max');
const cmcVals = document.getElementById('cmc-vals');
function updateCmcLabel() {
  const lo = parseInt(cmcMin.value);
  const hi = parseInt(cmcMax.value);
  cmcVals.textContent = `${lo}–${hi >= 11 ? '10+' : hi}`;
}
function onCmcChange() {
  let lo = parseInt(cmcMin.value);
  let hi = parseInt(cmcMax.value);
  if (lo > hi) { if (this === cmcMin) { hi = lo; cmcMax.value = lo; } else { lo = hi; cmcMin.value = hi; } }
  filters.cmcMin = lo;
  filters.cmcMax = hi;
  updateCmcLabel();
  applyFilters();
}
cmcMin.addEventListener('input', onCmcChange);
cmcMax.addEventListener('input', onCmcChange);
updateCmcLabel();

// Color identity chips
const colorChipsEl = document.getElementById('color-chips');
colorChipsEl.innerHTML = COLOR_BUCKETS.map(b =>
  `<span class="chip color-chip" style="background:${COLOR_HEX[b]}" data-color="${b}" title="${b}"></span>`
).join('');
colorChipsEl.querySelectorAll('.chip').forEach(el => {
  el.addEventListener('click', () => {
    const b = el.dataset.color;
    if (filters.visibleColors.has(b)) { filters.visibleColors.delete(b); el.classList.add('off'); }
    else { filters.visibleColors.add(b); el.classList.remove('off'); }
    applyFilters();
  });
});

// Filter section collapse
document.getElementById('filter-header').addEventListener('click', () => {
  document.getElementById('filter-section').classList.toggle('collapsed');
  const span = document.getElementById('filter-header').querySelector('span');
  span.textContent = span.textContent.startsWith('▾')
    ? span.textContent.replace('▾', '▸') : span.textContent.replace('▸', '▾');
});
```

- [ ] **Step 6: Extend search to match oracle text**

In the same file, find the search input handler block. Replace the existing `searchInput.addEventListener('input', ...)` handler with:

```js
const ORACLE_LOWER = POINT_META.map(c => (c.oracle_text || '').toLowerCase());
searchInput.addEventListener('input', (e) => {
  const q = e.target.value.trim().toLowerCase();
  if (!q) { searchResults.classList.remove('open'); searchResults.innerHTML = ''; return; }
  const words = q.split(/\s+/).filter(Boolean);
  const matches = [];
  for (let i = 0; i < NAME_INDEX.length; i++) {
    const nm = NAME_INDEX[i].name.toLowerCase();
    const txt = ORACLE_LOWER[i];
    const nameHit = words.every(w => nm.includes(w));
    const textHit = !nameHit && words.every(w => txt.includes(w));
    if (nameHit || textHit) matches.push({...NAME_INDEX[i], textHit, score: nm.length});
  }
  if (!matches.length) { searchResults.classList.remove('open'); return; }
  matches.sort((a, b) => a.score - b.score);
  const top = matches.slice(0, 15);
  searchResults.innerHTML = top.map(m =>
    `<div class="search-result" data-oid="${m.oid}">${escapeHtml(m.name)}<span class="sr-type">${escapeHtml(m.type)}</span>${m.textHit ? '<span class="sr-text-hit">[text]</span>' : ''}</div>`
  ).join('');
  searchResults.classList.add('open');
  searchResults.querySelectorAll('.search-result').forEach(el => {
    el.addEventListener('click', () => {
      const oid = el.dataset.oid;
      selectCard(oid);
      searchInput.value = POINT_META[OID_TO_IDX[oid]].name;
      searchResults.classList.remove('open');
    });
  });
});
```

- [ ] **Step 7: Render the page**

Run: `uv run python -m mtg_graph.visualize 2>&1 | tail -3`

Expected:
```
  Wrote output/mtg_graph_poc.html
  Wrote output/mtg_graph_poc_3d.html
```

- [ ] **Step 8: Open and verify in browser**

Run: `open output/mtg_graph_poc.html`

Manual checks:
- Filter section visible at top of side panel; collapsible
- Color axis dropdown switches the entire graph's colors (try EDHREC rank → should see a clear popularity gradient)
- Format chips: click "modern" → only Modern-legal cards remain visible
- Set: type "blb" in the box → Bloomburrow set chip appears in dropdown → click adds it
- Mana value sliders shrink visible cards to CMC range
- Color chip click hides that color's cards
- Search "flying" → dropdown shows cards with "flying" in name OR oracle text, text-only matches tagged "[text]"

- [ ] **Step 9: Commit**

```bash
git add src/mtg_graph/templates/viewer_2d.html
git commit -m "2D viewer: filter & overlay panel, oracle text search

Adds collapsible Filters & Overlays section: color overlay (identity,
EDHREC rank, price, type, mana value), format multi-select, searchable
set picker, mana value range slider, hide-by-color chips. Search input
now also matches oracle text substrings (AND across words).

All filter state lives in JS; Cosmograph.setPointColors() pushes both
color + visibility (via alpha) on every change.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: 3D 3d-force-graph — filter panel UI + JS

**Files:**
- Modify: `src/mtg_graph/templates/viewer_3d.html`

Mostly mirrors Task 4 but using 3d-force-graph's API (`nodeColor` accessor, `nodeVisibility` accessor) instead of Cosmograph's `setPointColors`.

- [ ] **Step 1: Add filter panel CSS**

In `src/mtg_graph/templates/viewer_3d.html`, find:

```css
.search-row { position: relative; margin-bottom: 16px; }
```

Insert these styles directly before that line:

```css
.filter-section { margin-bottom: 14px; padding-bottom: 10px; border-bottom: 1px solid var(--panel-border); }
.filter-header { display: flex; justify-content: space-between; align-items: center; font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-dim); cursor: pointer; padding: 4px 0; user-select: none; }
.filter-header:hover { color: var(--text); }
.filter-section.collapsed .filter-body { display: none; }
.filter-body { padding: 8px 0 4px; }
.filter-row { margin-bottom: 12px; }
.filter-row:last-child { margin-bottom: 0; }
.filter-row label { display: block; font-size: 11px; color: var(--text-dim); margin-bottom: 4px; }
.filter-row select, .filter-row input[type=text] {
  width: 100%; background: var(--input-bg); color: var(--text);
  border: 1px solid var(--panel-border); padding: 5px 8px;
  font-size: 12px; border-radius: 3px; outline: none;
}
.filter-row select:focus, .filter-row input[type=text]:focus { border-color: var(--accent); }
.format-chips, .color-chips { display: flex; flex-wrap: wrap; gap: 4px; }
.chip {
  font-size: 11px; padding: 3px 8px; border-radius: 11px;
  background: var(--input-bg); color: var(--text-dim);
  border: 1px solid var(--panel-border); cursor: pointer;
  user-select: none;
}
.chip.active { color: var(--text); border-color: var(--accent); background: rgba(201,162,39,0.1); }
.chip.color-chip { width: 22px; height: 22px; padding: 0; border-radius: 50%; border: 2px solid transparent; position: relative; }
.chip.color-chip.off { opacity: 0.25; border-color: var(--panel-border); }
.cmc-row { display: flex; align-items: center; gap: 8px; }
.cmc-row input[type=range] { flex: 1; accent-color: var(--accent); }
.cmc-row .cmc-vals { font-size: 11px; color: var(--text-dim); min-width: 32px; text-align: right; }
.set-options { max-height: 140px; overflow-y: auto; background: var(--input-bg); border: 1px solid var(--panel-border); border-radius: 3px; margin-top: 4px; display: none; }
.set-options.open { display: block; }
.set-option { padding: 4px 8px; font-size: 11px; cursor: pointer; }
.set-option:hover { background: var(--panel-border); }
.selected-sets { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; }
.selected-set-chip { background: rgba(201,162,39,0.15); color: var(--accent-bright); border: 1px solid var(--accent); padding: 2px 6px; border-radius: 3px; font-size: 10px; cursor: pointer; }
.sr-text-hit { color: var(--accent-bright); font-size: 10px; margin-left: 6px; }
```

- [ ] **Step 2: Add filter panel HTML**

Find the `<div class="search-row">` line. Insert this block directly before it:

```html
      <div class="filter-section" id="filter-section">
        <div class="filter-header" id="filter-header">
          <span>▾ Filters &amp; Overlays</span>
        </div>
        <div class="filter-body">
          <div class="filter-row">
            <label for="color-axis">Color by</label>
            <select id="color-axis">
              <option value="identity">Color identity</option>
              <option value="edhrec">EDHREC rank (popularity)</option>
              <option value="price">Price (USD)</option>
              <option value="type">Card type</option>
              <option value="cmc">Mana value</option>
            </select>
          </div>
          <div class="filter-row">
            <label>Format</label>
            <div class="format-chips" id="format-chips"></div>
          </div>
          <div class="filter-row">
            <label for="set-input">Set</label>
            <input type="text" id="set-input" placeholder="Type to filter sets…" autocomplete="off" />
            <div class="set-options" id="set-options"></div>
            <div class="selected-sets" id="selected-sets"></div>
          </div>
          <div class="filter-row">
            <label>Mana value <span class="cmc-vals" id="cmc-vals">0–10+</span></label>
            <div class="cmc-row">
              <input type="range" id="cmc-min" min="0" max="11" step="1" value="0" />
              <input type="range" id="cmc-max" min="0" max="11" step="1" value="11" />
            </div>
          </div>
          <div class="filter-row">
            <label>Color identity visible</label>
            <div class="color-chips" id="color-chips"></div>
          </div>
        </div>
      </div>
```

- [ ] **Step 3: Update the search placeholder**

Find:

```html
<input type="text" id="search" placeholder="Search card name…" autocomplete="off" />
```

Replace with:

```html
<input type="text" id="search" placeholder="Search name or oracle text…" autocomplete="off" />
```

- [ ] **Step 4: Add the JS filter framework — state, helpers, color computation**

In the `<script type="module">` block, find:

```js
const NAME_INDEX = POINT_META.map(c => ({oid: c.oracle_id, name: c.name, type: c.type_line || ''}));
```

Insert this entire block directly after that line:

```js

// ===== Filter & overlay state =====
const COLOR_BUCKETS = ['White', 'Blue', 'Black', 'Red', 'Green', 'Multicolor', 'Colorless'];
const COLOR_HEX = {
  'White': '#F5EEDC', 'Blue': '#3B82F6', 'Black': '#7A6FA0', 'Red': '#EF4444',
  'Green': '#22C55E', 'Multicolor': '#FBBF24', 'Colorless': '#9CA3AF',
};
const FORMATS = ['standard', 'pioneer', 'modern', 'legacy', 'vintage', 'commander', 'pauper', 'brawl'];
const filters = {
  colorAxis: 'identity',
  formats: new Set(),
  sets: new Set(),
  cmcMin: 0,
  cmcMax: 11,
  visibleColors: new Set(COLOR_BUCKETS),
};
function colorBucket(ci) {
  if (!ci || ci.length === 0) return 'Colorless';
  if (ci.length > 1) return 'Multicolor';
  return {W: 'White', U: 'Blue', B: 'Black', R: 'Red', G: 'Green'}[ci[0]] || 'Colorless';
}
function primaryType(typeLine) {
  if (!typeLine) return 'Other';
  const t = typeLine.toLowerCase();
  for (const k of ['land', 'creature', 'planeswalker', 'instant', 'sorcery', 'enchantment', 'artifact']) {
    if (t.includes(k)) return k;
  }
  return 'Other';
}
const TYPE_COLOR = {
  'land': '#9CA3AF', 'creature': '#22C55E', 'planeswalker': '#FBBF24',
  'instant': '#3B82F6', 'sorcery': '#7A6FA0', 'enchantment': '#F5EEDC',
  'artifact': '#A5A5A5', 'Other': '#666666',
};
function viridis(t) {
  t = Math.max(0, Math.min(1, t));
  const stops = [
    [0.267, 0.005, 0.329], [0.231, 0.318, 0.545],
    [0.128, 0.567, 0.551], [0.369, 0.788, 0.382],
    [0.993, 0.906, 0.144],
  ];
  const i = Math.min(Math.floor(t * (stops.length - 1)), stops.length - 2);
  const f = t * (stops.length - 1) - i;
  return [
    stops[i][0] + (stops[i+1][0] - stops[i][0]) * f,
    stops[i][1] + (stops[i+1][1] - stops[i][1]) * f,
    stops[i][2] + (stops[i+1][2] - stops[i][2]) * f,
  ];
}
function rgbToHex(rgb) {
  return '#' + rgb.map(c => Math.round(c * 255).toString(16).padStart(2, '0')).join('');
}
const edhrecRanks = POINT_META.map(c => c.edhrec_rank).filter(r => r != null);
const EDHREC_MIN = Math.min(...edhrecRanks);
const EDHREC_MAX = Math.max(...edhrecRanks);
const prices = POINT_META.map(c => parseFloat(c.price_usd)).filter(p => !isNaN(p) && p > 0);
const PRICE_LOG_MIN = Math.log10(Math.min(...prices));
const PRICE_LOG_MAX = Math.log10(Math.max(...prices));
const CMC_OVERLAY_MAX = 8;

function nodeColorForCard(card) {
  switch (filters.colorAxis) {
    case 'identity':
      return COLOR_HEX[colorBucket(card.color_identity)];
    case 'edhrec': {
      const r = card.edhrec_rank;
      if (r == null) return '#444444';
      const t = 1 - (r - EDHREC_MIN) / (EDHREC_MAX - EDHREC_MIN);
      return rgbToHex(viridis(t));
    }
    case 'price': {
      const p = parseFloat(card.price_usd);
      if (isNaN(p) || p <= 0) return '#333333';
      const t = (Math.log10(p) - PRICE_LOG_MIN) / (PRICE_LOG_MAX - PRICE_LOG_MIN);
      return rgbToHex(viridis(t));
    }
    case 'type':
      return TYPE_COLOR[primaryType(card.type_line)] || TYPE_COLOR.Other;
    case 'cmc': {
      const c = card.cmc;
      if (c == null) return '#444444';
      const t = Math.min(c, CMC_OVERLAY_MAX) / CMC_OVERLAY_MAX;
      return rgbToHex(viridis(t));
    }
  }
  return '#888888';
}

function cardVisible(card) {
  if (!filters.visibleColors.has(colorBucket(card.color_identity))) return false;
  if (filters.formats.size > 0) {
    const leg = card.legalities || {};
    let anyOk = false;
    for (const f of filters.formats) {
      if (leg[f] === 'legal') { anyOk = true; break; }
    }
    if (!anyOk) return false;
  }
  if (filters.sets.size > 0 && !filters.sets.has(card.set)) return false;
  const cmc = card.cmc;
  if (cmc != null) {
    if (cmc < filters.cmcMin) return false;
    if (filters.cmcMax < 11 && cmc > filters.cmcMax) return false;
  }
  return true;
}

// Refresh by re-applying the (same) accessor — 3d-force-graph re-evaluates closures.
function applyFilters() {
  graph.nodeColor(graph.nodeColor()).nodeVisibility(graph.nodeVisibility());
}
```

- [ ] **Step 5: Update the graph initialization to use the filter-aware accessors**

In the same file, find this block:

```js
const graph = new ForceGraph3D(chartEl)
  .graphData({nodes, links})
  .backgroundColor('#0a0a0a')
  .showNavInfo(false)
  .nodeRelSize(1.2)
  .nodeOpacity(1.0)
  .enableNodeDrag(false)
  .nodeColor(node => {
    if (state.selectedOid === null) return node.color;
    if (node.id === state.selectedOid) return '#FBBF24';
    if (state.highlightSet.has(node.id)) return '#FFFFFF';
    return dim(node.color, 0.12);
  })
  .nodeLabel(node => `<div style="font-weight:600">${escapeHtml(node.name)}</div>`)
```

Replace the `nodeColor(...)` portion with the filter-aware version. Specifically, change:

```js
  .nodeColor(node => {
    if (state.selectedOid === null) return node.color;
    if (node.id === state.selectedOid) return '#FBBF24';
    if (state.highlightSet.has(node.id)) return '#FFFFFF';
    return dim(node.color, 0.12);
  })
```

To:

```js
  .nodeColor(node => {
    const card = POINT_META[OID_TO_IDX[node.id]];
    if (state.selectedOid === null) return nodeColorForCard(card);
    if (node.id === state.selectedOid) return '#FBBF24';
    if (state.highlightSet.has(node.id)) return '#FFFFFF';
    return dim(nodeColorForCard(card), 0.12);
  })
  .nodeVisibility(node => cardVisible(POINT_META[OID_TO_IDX[node.id]]))
```

- [ ] **Step 6: Wire up the filter UI controls**

In the same file, find the `// Search` comment. Insert this block directly *before* it:

```js
// ===== Filter UI wiring =====
document.getElementById('color-axis').addEventListener('change', (e) => {
  filters.colorAxis = e.target.value;
  applyFilters();
});

const formatChipsEl = document.getElementById('format-chips');
formatChipsEl.innerHTML = FORMATS.map(f =>
  `<span class="chip" data-format="${f}">${f}</span>`
).join('');
formatChipsEl.querySelectorAll('.chip').forEach(el => {
  el.addEventListener('click', () => {
    const f = el.dataset.format;
    if (filters.formats.has(f)) { filters.formats.delete(f); el.classList.remove('active'); }
    else { filters.formats.add(f); el.classList.add('active'); }
    applyFilters();
  });
});

const SETS_IN_DATA = [...new Set(POINT_META.map(c => c.set).filter(Boolean))].sort();
const setInput = document.getElementById('set-input');
const setOptionsEl = document.getElementById('set-options');
const selectedSetsEl = document.getElementById('selected-sets');
function renderSetOptions(query) {
  const q = query.trim().toLowerCase();
  const matches = SETS_IN_DATA.filter(s => s.includes(q) && !filters.sets.has(s)).slice(0, 30);
  if (!matches.length) { setOptionsEl.classList.remove('open'); return; }
  setOptionsEl.innerHTML = matches.map(s => `<div class="set-option" data-set="${s}">${s}</div>`).join('');
  setOptionsEl.classList.add('open');
  setOptionsEl.querySelectorAll('.set-option').forEach(el => {
    el.addEventListener('click', () => {
      filters.sets.add(el.dataset.set);
      setInput.value = '';
      setOptionsEl.classList.remove('open');
      renderSelectedSets();
      applyFilters();
    });
  });
}
function renderSelectedSets() {
  selectedSetsEl.innerHTML = [...filters.sets].sort().map(s =>
    `<span class="selected-set-chip" data-set="${s}">${s} ×</span>`
  ).join('');
  selectedSetsEl.querySelectorAll('.selected-set-chip').forEach(el => {
    el.addEventListener('click', () => {
      filters.sets.delete(el.dataset.set);
      renderSelectedSets();
      applyFilters();
    });
  });
}
setInput.addEventListener('input', () => renderSetOptions(setInput.value));
setInput.addEventListener('focus', () => renderSetOptions(setInput.value));
document.addEventListener('click', (e) => {
  if (!setInput.contains(e.target) && !setOptionsEl.contains(e.target)) setOptionsEl.classList.remove('open');
});

const cmcMin = document.getElementById('cmc-min');
const cmcMax = document.getElementById('cmc-max');
const cmcVals = document.getElementById('cmc-vals');
function updateCmcLabel() {
  const lo = parseInt(cmcMin.value);
  const hi = parseInt(cmcMax.value);
  cmcVals.textContent = `${lo}–${hi >= 11 ? '10+' : hi}`;
}
function onCmcChange() {
  let lo = parseInt(cmcMin.value);
  let hi = parseInt(cmcMax.value);
  if (lo > hi) { if (this === cmcMin) { hi = lo; cmcMax.value = lo; } else { lo = hi; cmcMin.value = hi; } }
  filters.cmcMin = lo;
  filters.cmcMax = hi;
  updateCmcLabel();
  applyFilters();
}
cmcMin.addEventListener('input', onCmcChange);
cmcMax.addEventListener('input', onCmcChange);
updateCmcLabel();

const colorChipsEl = document.getElementById('color-chips');
colorChipsEl.innerHTML = COLOR_BUCKETS.map(b =>
  `<span class="chip color-chip" style="background:${COLOR_HEX[b]}" data-color="${b}" title="${b}"></span>`
).join('');
colorChipsEl.querySelectorAll('.chip').forEach(el => {
  el.addEventListener('click', () => {
    const b = el.dataset.color;
    if (filters.visibleColors.has(b)) { filters.visibleColors.delete(b); el.classList.add('off'); }
    else { filters.visibleColors.add(b); el.classList.remove('off'); }
    applyFilters();
  });
});

document.getElementById('filter-header').addEventListener('click', () => {
  document.getElementById('filter-section').classList.toggle('collapsed');
  const span = document.getElementById('filter-header').querySelector('span');
  span.textContent = span.textContent.startsWith('▾')
    ? span.textContent.replace('▾', '▸') : span.textContent.replace('▸', '▾');
});
```

- [ ] **Step 7: Extend search to match oracle text**

In the same file, find the `searchInput.addEventListener('input', ...)` handler. Replace the entire handler body (the function passed to `addEventListener`) with:

```js
searchInput.addEventListener('input', (e) => {
  const q = e.target.value.trim().toLowerCase();
  if (!q) { searchResults.classList.remove('open'); searchResults.innerHTML = ''; return; }
  const words = q.split(/\s+/).filter(Boolean);
  const matches = [];
  for (let i = 0; i < NAME_INDEX.length; i++) {
    const nm = NAME_INDEX[i].name.toLowerCase();
    const txt = (POINT_META[i].oracle_text || '').toLowerCase();
    const nameHit = words.every(w => nm.includes(w));
    const textHit = !nameHit && words.every(w => txt.includes(w));
    if (nameHit || textHit) matches.push({...NAME_INDEX[i], textHit, score: nm.length});
  }
  if (!matches.length) { searchResults.classList.remove('open'); return; }
  matches.sort((a, b) => a.score - b.score);
  const top = matches.slice(0, 15);
  searchResults.innerHTML = top.map(m =>
    `<div class="search-result" data-oid="${m.oid}">${escapeHtml(m.name)}<span class="sr-type">${escapeHtml(m.type)}</span>${m.textHit ? '<span class="sr-text-hit">[text]</span>' : ''}</div>`
  ).join('');
  searchResults.classList.add('open');
  searchResults.querySelectorAll('.search-result').forEach(el => {
    el.addEventListener('click', () => {
      const oid = el.dataset.oid;
      selectCard(oid);
      searchInput.value = POINT_META[OID_TO_IDX[oid]].name;
      searchResults.classList.remove('open');
    });
  });
});
```

- [ ] **Step 8: Render and verify in browser**

Run: `uv run python -m mtg_graph.visualize 2>&1 | tail -3 && open output/mtg_graph_poc_3d.html`

Manual checks (same as Task 4 Step 8 but in 3D):
- Filter section appears in the side panel
- Color overlay dropdown changes node colors throughout the scene
- Format chips hide non-matching cards (they disappear, leaving holes in the cloud)
- Set / mana value / hide-by-color all work
- Search "flying" returns results with [text] tag for oracle-text-only matches
- Orbit / click / camera-fly-to-selected all still work after each filter change (no Plotly-style controls breaking)

- [ ] **Step 9: Commit**

```bash
git add src/mtg_graph/templates/viewer_3d.html
git commit -m "3D viewer: filter & overlay panel, oracle text search

Mirrors the 2D filter panel using 3d-force-graph's nodeColor/nodeVisibility
accessors. Refresh pattern is re-invoking the accessors so the closures
re-evaluate against updated filter state.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: End-to-end pipeline run + final verification

- [ ] **Step 1: Full pipeline rerun**

Run: `uv run mtg-graph-poc --open 2d`

Expected: all stages complete (load, profiles, embed cached, reduce, cluster + LLM, knn, visualize). Total time: ~2-3 minutes due to LLM cluster labeling.

- [ ] **Step 2: Spot-check the new cluster labels in the 2D view**

In the open browser tab, scroll/zoom across the graph. Verify cluster labels read naturally — e.g. you should see things like "Counterspells", "Ninja tribal", "Treasure tokens & ramp" instead of "instant counter / countered", "ninja / ninjutsu", "treasure / token add".

- [ ] **Step 3: Smoke-test each filter feature in 2D**

In the browser:
1. **Color overlay** — switch to EDHREC rank, observe popularity heatmap; switch to Price, observe expensive clusters
2. **Format filter** — toggle Modern, observe big drop in visible cards (a lot of Commander staples aren't Modern-legal); toggle Commander, restore them
3. **Set filter** — type "blb", select Bloomburrow, observe ~80 cards remain visible
4. **Mana value slider** — drag low end to 0, drag high end to 2, see only 0/1/2 cmc cards
5. **Hide-by-color** — click Green chip, all green cards fade
6. **Search "lifelink"** — dropdown should show all lifelink cards, with [text] indicators on cards that have lifelink but not in name
7. **Click a card** — selection still works, neighbors light up

- [ ] **Step 4: Same smoke test in 3D**

Run: `open output/mtg_graph_poc_3d.html`

Repeat checks 1–7 in the 3D viewer. Pay special attention to: filter changes don't break the orbit controls, search still works, click + camera-fly-to-card still works.

- [ ] **Step 5: Final commit** (only if anything was tweaked during the smoke test)

If you ended up making small fixes (CSS nudges, missing event handlers, etc.), squash them into one commit:

```bash
git add -A
git commit -m "Polish: filter panel verified end-to-end in 2D and 3D"
```

Otherwise the prior commits already cover the work; just confirm `git status` is clean.

---

## Done

Phase 1 ships:
1. Filter & Overlays panel in both viewers (color overlays × 5 axes, format / set / mana value filters, hide-by-color)
2. Oracle text substring search
3. LLM-named clusters

Phase 2 (scale to 37k + OpenAI re-embed) and Phase 3 (semantic search, deck-aware, multi-select, RAG) are separate plans.
