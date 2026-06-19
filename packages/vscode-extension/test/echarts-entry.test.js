// Guards echarts-entry.mjs against drift: every echarts series `type: '...'`
// the viewer uses must have a matching chart registered in the entry's use([])
// list. A mismatch would make that chart silently no-op in the webview.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const VIEWER_SRC = path.join(__dirname, '..', '..', 'viewer', 'src');
const ENTRY = path.join(__dirname, '..', 'echarts-entry.mjs');

// echarts series type -> the chart class that must be registered.
const SERIES_TO_CHART = {
  line: 'LineChart', bar: 'BarChart', scatter: 'ScatterChart',
  heatmap: 'HeatmapChart', custom: 'CustomChart',
};

function walk(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
    const p = path.join(dir, e.name);
    return e.isDirectory() ? walk(p) : [p];
  });
}

test('echarts-entry registers every series type the viewer uses', () => {
  const entry = fs.readFileSync(ENTRY, 'utf8');
  const used = new Set();
  for (const f of walk(VIEWER_SRC).filter((f) => /\.(ts|js|mjs)$/.test(f))) {
    const src = fs.readFileSync(f, 'utf8');
    for (const m of src.matchAll(/type:\s*['"](\w+)['"]/g)) {
      if (m[1] in SERIES_TO_CHART) used.add(m[1]);
    }
  }
  assert.ok(used.size > 0, 'expected to find echarts series types in viewer src');
  for (const series of used) {
    const chart = SERIES_TO_CHART[series];
    assert.ok(
      entry.includes(chart),
      `viewer uses series '${series}' but echarts-entry.mjs does not register ${chart}`,
    );
  }
});
