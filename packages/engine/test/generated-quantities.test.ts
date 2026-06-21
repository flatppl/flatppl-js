'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const GQ = require('../generated-quantities.ts');

function recMeasure(fields: Record<string, number[]>, logWeights?: any) {
  const out: any = { fields: {} as Record<string, any>, logWeights: logWeights || null };
  for (const k in fields) out.fields[k] = { samples: Float64Array.from(fields[k]) };
  return out;
}
// Lowered IR for `mu1 - mu2` referencing record fields by self-name.
// The engine uses lowered op names: '-' → 'sub' (see lower.ts line 154).
const subIR = { kind: 'call', op: 'sub', args: [
  { kind: 'ref', ns: 'self', name: 'mu1' },
  { kind: 'ref', ns: 'self', name: 'mu2' },
] };

test('deriveColumn computes a binding samplewise over record field columns', () => {
  const m = recMeasure({ mu1: [3, 5, 7], mu2: [1, 2, 3] });
  const col = GQ.deriveColumn(subIR, m, {});
  assert.deepEqual(Array.from(col), [2, 3, 4]);
});

test('appendGeneratedQuantities adds a field sharing logWeights', () => {
  const lw = Float64Array.from([-0.1, -0.2, -0.3]);
  const m = recMeasure({ mu1: [3, 5, 7], mu2: [1, 2, 3] }, lw);
  const out = GQ.appendGeneratedQuantities(m, [{ name: 'diff', ir: subIR }], {});
  assert.ok(out.fields.diff, 'diff field appended');
  assert.deepEqual(Array.from(out.fields.diff.samples), [2, 3, 4]);
  assert.equal(out.fields.diff.logWeights, lw, 'derived field shares the measure logWeights');
});
