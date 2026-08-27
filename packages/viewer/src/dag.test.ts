// @ts-nocheck — test file; compiled separately by node --test (not by tsc)
import { registerHooks } from 'node:module';
import { test } from 'node:test';
import assert from 'node:assert/strict';

// viewer/src uses bundler-style .js extensions in imports (resolved by esbuild
// at build time). Register a resolver hook so Node --test can load .ts source
// directly without a build step.
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier.endsWith('.js') && context.parentURL?.includes('/packages/viewer/src/')) {
      return nextResolve(specifier.slice(0, -3) + '.ts', context);
    }
    return nextResolve(specifier, context);
  }
});

const {
  resolveDroppedChain,
  rewriteEdgesForCollapse,
  anchorGlyphSuffix,
  hitsAnchorGlyph,
  glyphStartOffset,
  GLYPH_COLLAPSED,
  GLYPH_EXPANDED,
} = await import('./dag.ts');

// Regression for #176: the graph-view compactor collapsed a reification
// bubble whose anchor was ITSELF a member of an outer collapsed bubble.
// dminus-to-3pi-amplitude.flatppl nests three deep —
// phase_weight ⊃ intensity_fn ⊃ angular_tensors, each anchor directly
// listed as a member of the one enclosing it. A single-hop lookup left an
// edge pointing at `intensity_fn` even though intensity_fn itself was
// dropped under `phase_weight`, and cytoscape threw on the dangling
// endpoint, killing the whole DAG render.

test('resolveDroppedChain follows a multi-level nested-anchor chain to the surviving anchor', () => {
  // angular_tensors dropped under intensity_fn, itself dropped under
  // phase_weight — phase_weight is never a value, so it's the chain's end.
  const dropped = new Map([
    ['angular_tensors', 'intensity_fn'],
    ['intensity_fn', 'phase_weight'],
    ['p1', 'angular_tensors'],
  ]);
  assert.equal(resolveDroppedChain(dropped, 'angular_tensors'), 'phase_weight');
  assert.equal(resolveDroppedChain(dropped, 'p1'), 'phase_weight');
  assert.equal(resolveDroppedChain(dropped, 'phase_weight'), 'phase_weight');
  assert.equal(resolveDroppedChain(dropped, 'unrelated'), 'unrelated');
});

test('rewriteEdgesForCollapse re-routes every endpoint to a surviving anchor, never leaving a dangling edge', () => {
  const dropped = new Map([
    ['angular_tensors', 'intensity_fn'],
    ['intensity_fn', 'phase_weight'],
    ['p1', 'angular_tensors'],
  ]);
  const survivingNodeIds = new Set(['phase_weight', 'p2', 'q_of']);
  const edges = [
    // p1 -> angular_tensors: both endpoints nested under phase_weight,
    // collapses to a self-edge and is dropped.
    { source: 'p1', target: 'angular_tensors', edgeType: 'data' },
    // Straddles the collapsed bubble: p2 (surviving) into intensity_fn
    // (dropped, chain-resolves to phase_weight, which survives).
    { source: 'p2', target: 'intensity_fn', edgeType: 'data' },
    // Duplicate of the above once both re-route to the same pair —
    // deduped.
    { source: 'p2', target: 'angular_tensors', edgeType: 'data' },
    // Untouched by any drop.
    { source: 'q_of', target: 'p2', edgeType: 'call' },
  ];
  const result = rewriteEdgesForCollapse(edges, dropped, survivingNodeIds);
  assert.deepEqual(
    result.map((e) => [e.source, e.target, e.edgeType]).sort(),
    [
      ['p2', 'phase_weight', 'data'],
      ['q_of', 'p2', 'call'],
    ],
  );
  // Every surviving edge's endpoints are, by construction, in
  // survivingNodeIds — this is the property whose absence crashed
  // cytoscape in #176.
  for (const e of result) {
    assert.ok(survivingNodeIds.has(e.source), `dangling source ${e.source}`);
    assert.ok(survivingNodeIds.has(e.target), `dangling target ${e.target}`);
  }
});

test('rewriteEdgesForCollapse drops an edge whose resolved endpoint still is not a surviving node (defensive net)', () => {
  // No entry for 'ghost' at all — simulates a future gap in the drop
  // accounting rather than the nested-chain case above.
  const dropped = new Map();
  const survivingNodeIds = new Set(['a']);
  const edges = [{ source: 'a', target: 'ghost', edgeType: 'data' }];
  assert.deepEqual(rewriteEdgesForCollapse(edges, dropped, survivingNodeIds), []);
});

// Glyph hit-test geometry. The region is now MEASURED, so these tests
// supply a proportional fake measure whose per-character widths are the
// ones 13px "Helvetica Neue" actually reports (letters ~6.5px, digits
// ~7.2px, space ~3.6px, and the ⊞/⊟ glyphs ~9.9px — wider than a letter,
// which is half of why the old per-character estimate drifted).
const FAKE_CHAR_PX = { ' ': 3.61, '⊞': 9.88, '⊟': 9.88 };
function fakeMeasure(s) {
  let w = 0;
  for (const ch of s) {
    if (FAKE_CHAR_PX[ch] != null) w += FAKE_CHAR_PX[ch];
    else if (ch >= '0' && ch <= '9') w += 7.23;
    else if (ch === '_') w += 7.33;
    else w += 6.5;
  }
  return w;
}

// renderDAG still sizes the node box off the generous per-character
// estimate, so the box is WIDER than the text it centres — reproduce that
// here rather than assuming box width and text width agree.
function anchorBox(label, cx = 0, borderPx = 1.5) {
  const w = Math.max(label.length * 9 + 24, 60);
  const half = borderPx / 2;
  return { x1: cx - w / 2 - half, x2: cx + w / 2 + half, y1: -18 - half, y2: 18 + half };
}

// Where the glyph really is: the label is centred, so its right edge sits
// at centre + labelWidth/2 and the glyph is the tail of it.
function trueGlyphSpan(label, glyph, cx = 0) {
  const right = cx + fakeMeasure(label) / 2;
  return { x1: right - fakeMeasure(glyph), x2: right };
}

function regionLeft(label, glyph, cx = 0) {
  return cx + glyphStartOffset(fakeMeasure(label), fakeMeasure(glyph));
}

test('anchorGlyphSuffix shows ⊞ with the drop count when collapsed and ⊟ when expanded', () => {
  assert.equal(anchorGlyphSuffix(true, 3), '  ' + GLYPH_COLLAPSED + '3');
  assert.equal(anchorGlyphSuffix(false, 3), '  ' + GLYPH_EXPANDED);
  // Every anchor gets a glyph, including one that dropped nothing.
  assert.equal(anchorGlyphSuffix(true, 0), '  ' + GLYPH_COLLAPSED + '0');
});

test('glyphStartOffset opens the region just left of the measured glyph, clear of the name', () => {
  const glyph = GLYPH_COLLAPSED + '3';
  const label = 'prior  ' + glyph;
  const span = trueGlyphSpan(label, glyph);
  const left = regionLeft(label, glyph);
  // Region starts at or before the glyph, so the whole glyph is clickable.
  assert.ok(left <= span.x1, `region left ${left} must not cut into the glyph at ${span.x1}`);
  // And it starts after the name ends, so a click on the name still selects.
  const nameRight = span.x1 - fakeMeasure('  ');
  assert.ok(left > nameRight, `region left ${left} must sit right of the name end ${nameRight}`);
});

test('hitsAnchorGlyph accepts the whole measured glyph and rejects the name, short label', () => {
  const glyph = GLYPH_COLLAPSED + '3';
  const label = 'prior  ' + glyph;               // 'prior  ⊞3'
  const box = anchorBox(label);
  const start = glyphStartOffset(fakeMeasure(label), fakeMeasure(glyph));
  const span = trueGlyphSpan(label, glyph);
  for (const x of [span.x1, (span.x1 + span.x2) / 2, span.x2]) {
    assert.equal(hitsAnchorGlyph({ x, y: 0 }, box, start), true, `glyph x=${x} must hit`);
  }
  // The node is wider than its text; blank space right of the glyph out to
  // the node edge stays part of the target.
  assert.equal(hitsAnchorGlyph({ x: box.x2, y: 0 }, box, start), true);
  // The name half keeps select-and-plot.
  assert.equal(hitsAnchorGlyph({ x: 0, y: 0 }, box, start), false);
  assert.equal(hitsAnchorGlyph({ x: box.x1 + 4, y: 0 }, box, start), false);
});

test('hitsAnchorGlyph accepts the measured glyph on an expanded anchor', () => {
  const glyph = GLYPH_EXPANDED;
  const label = 'prior  ' + glyph;               // 'prior  ⊟'
  const box = anchorBox(label);
  const start = glyphStartOffset(fakeMeasure(label), fakeMeasure(glyph));
  const span = trueGlyphSpan(label, glyph);
  assert.equal(hitsAnchorGlyph({ x: (span.x1 + span.x2) / 2, y: 0 }, box, start), true);
  assert.equal(hitsAnchorGlyph({ x: 0, y: 0 }, box, start), false);
});

// Regression for the shipped defect (#206 follow-up): a per-character
// estimate of 9px overshoots a proportional font by ~50 % on a long label,
// so the region landed entirely PAST the glyph and clicking the visible
// glyph did nothing. Measured live before the fix on
// 'forward_kernel  ⊟': glyph at [382.3, 392.1], region at [408.0, 430.8].
test('hitsAnchorGlyph covers the glyph on a LONG label, where the per-character estimate misses it', () => {
  const glyph = GLYPH_COLLAPSED + '7';
  const label = 'forward_kernel  ' + glyph;      // 17 characters
  const box = anchorBox(label);
  const start = glyphStartOffset(fakeMeasure(label), fakeMeasure(glyph));
  const span = trueGlyphSpan(label, glyph);
  for (const x of [span.x1, (span.x1 + span.x2) / 2, span.x2]) {
    assert.equal(hitsAnchorGlyph({ x, y: 0 }, box, start), true, `glyph x=${x} must hit`);
  }
  assert.equal(hitsAnchorGlyph({ x: 0, y: 0 }, box, start), false);
  // Pin the drift itself: the old estimate's region opened beyond the
  // glyph's right edge, so no click on the glyph could ever land in it.
  const estimatedLeft = (label.length * 9) / 2 - glyph.length * 9;
  assert.ok(estimatedLeft > span.x2,
    `the per-character estimate should open past the glyph (${estimatedLeft} vs ${span.x2})`);
});

test('hitsAnchorGlyph keeps the glyph clickable on a collapsed anchor with a 7px double border', () => {
  const glyph = GLYPH_COLLAPSED + '12';
  const label = 'phase_weight  ' + glyph;
  // Collapsed anchors draw border-width 7; boundingBox() reports the box
  // inflated by half of it on every side.
  const box = anchorBox(label, 0, 7);
  const start = glyphStartOffset(fakeMeasure(label), fakeMeasure(glyph));
  const span = trueGlyphSpan(label, glyph);
  assert.equal(hitsAnchorGlyph({ x: (span.x1 + span.x2) / 2, y: 0 }, box, start), true);
  // The border is inside the region, not shaved off it.
  assert.equal(hitsAnchorGlyph({ x: box.x2, y: 0 }, box, start), true);
  assert.equal(hitsAnchorGlyph({ x: 0, y: 0 }, box, start), false);
});

test('hitsAnchorGlyph rejects a tap outside the node box vertically', () => {
  const glyph = GLYPH_COLLAPSED + '3';
  const label = 'prior  ' + glyph;
  const box = anchorBox(label);
  const start = glyphStartOffset(fakeMeasure(label), fakeMeasure(glyph));
  const span = trueGlyphSpan(label, glyph);
  const x = (span.x1 + span.x2) / 2;
  assert.equal(hitsAnchorGlyph({ x, y: 40 }, box, start), false);
  assert.equal(hitsAnchorGlyph({ x, y: -40 }, box, start), false);
});

test('hitsAnchorGlyph rejects every tap on a node with no glyph region', () => {
  const box = anchorBox('theta');
  // null is what an unmeasured node carries (no canvas, or not an anchor).
  assert.equal(hitsAnchorGlyph({ x: box.x2, y: 0 }, box, null), false);
  assert.equal(hitsAnchorGlyph({ x: 0, y: 0 }, box, null), false);
});

test('hitsAnchorGlyph is independent of where the node sits on the canvas', () => {
  const glyph = GLYPH_COLLAPSED + '12';
  const label = 'phase_weight  ' + glyph;
  const start = glyphStartOffset(fakeMeasure(label), fakeMeasure(glyph));
  const near = anchorBox(label, 0);
  const far = anchorBox(label, 4000);
  const nearSpan = trueGlyphSpan(label, glyph, 0);
  const farSpan = trueGlyphSpan(label, glyph, 4000);
  assert.equal(hitsAnchorGlyph({ x: (nearSpan.x1 + nearSpan.x2) / 2, y: 0 }, near, start), true);
  assert.equal(hitsAnchorGlyph({ x: (farSpan.x1 + farSpan.x2) / 2, y: 0 }, far, start), true);
  // The near node's glyph coordinate is nowhere near the far node's box.
  assert.equal(hitsAnchorGlyph({ x: (nearSpan.x1 + nearSpan.x2) / 2, y: 0 }, far, start), false);
});
