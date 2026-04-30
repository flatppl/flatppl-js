'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, findEnclosingRanges } = require('../index');

function rangesAt(src, line, col) {
  const { ast } = processSource(src);
  return findEnclosingRanges(ast, line, col);
}

function spans(ranges) {
  return ranges.map(r => `${r.start.line}:${r.start.col}-${r.end.line}:${r.end.col}`);
}

test('findEnclosingRanges: empty file returns empty array', () => {
  assert.deepEqual(rangesAt('', 0, 0), []);
});

test('findEnclosingRanges: cursor on identifier in expression', () => {
  // 'x = a + b'
  //   col: 0   4 8
  const src = 'x = a + b\n';
  // Cursor on 'a' (col 4)
  const r = rangesAt(src, 0, 4);
  // Expected innermost-to-outermost:
  //   - 'a' (Identifier)
  //   - 'a + b' (BinaryExpr)
  //   - 'x = a + b' (AssignStatement)
  //   - whole Program (no .loc) — so 3 ranges
  assert.equal(r.length, 3);
  // Innermost is the Identifier 'a'
  assert.equal(r[0].start.col, 4);
  assert.equal(r[0].end.col, 5);
});

test('findEnclosingRanges: progressively larger ranges', () => {
  const src = 'x = f(a + b * c)\n';
  // Cursor on 'b' (col 10)
  const r = rangesAt(src, 0, 10);
  // Innermost first; each range should be a non-strict superset of the previous.
  for (let i = 1; i < r.length; i++) {
    const inner = r[i - 1];
    const outer = r[i];
    assert.ok(
      outer.start.col <= inner.start.col && outer.end.col >= inner.end.col,
      `range ${i} should contain range ${i - 1}`
    );
  }
});

test('findEnclosingRanges: cursor outside any range returns empty', () => {
  const src = 'x = 1\n';
  // Cursor far past end of line
  const r = rangesAt(src, 0, 100);
  // Program has no loc; everything else fails inLoc check
  assert.equal(r.length, 0);
});

test('findEnclosingRanges: AST node ranges only (no whitespace)', () => {
  const src = 'x = 1\n';
  const r = rangesAt(src, 0, 0); // cursor on 'x'
  // Should hit: 'x' (Identifier), 'x = 1' (AssignStatement)
  assert.equal(r.length, 2);
});

test('findEnclosingRanges: nested calls expand correctly', () => {
  const src = 'x = f(g(h(a)))\n';
  // Cursor on 'a' (col 11)
  const r = rangesAt(src, 0, 11);
  // a, h(a), g(h(a)), f(g(h(a))), AssignStatement → 5 ranges
  assert.equal(r.length, 5);
});
