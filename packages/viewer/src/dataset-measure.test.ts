// @ts-nocheck — test file; compiled separately by node --test (not by tsc)
// A dataset binding (iid over a record law) is ONE k-row table, drawn at one
// atom. The viewer shows it as a record measure over its rows: a field per
// scalar column, a field per component of a vector column, never constant.
import { registerHooks } from 'node:module';
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier.endsWith('.js') && context.parentURL?.includes('/packages/viewer/src/')) {
      return nextResolve(specifier.slice(0, -3) + '.ts', context);
    }
    return nextResolve(specifier, context);
  }
});

const require_ = createRequire(import.meta.url);
const engine = require_('@flatppl/engine');
globalThis.FlatPPLEngine = engine;

const { datasetMeasure } = await import('./engine-facade.ts');
const { measureIsConstant } = await import('./render-record.ts');

test('a dataset table becomes a record over its rows, vector columns split', () => {
  const t = {
    shape: 'table', nrows: 3, n_eff: 123.5, sir: { K: 4, poolRows: 12, ess: 123.5 },
    columns: {
      s: { shape: [3], data: new Float64Array([1, 2, 3]) },
      p: { shape: [3, 2], data: new Float64Array([10, 11, 20, 21, 30, 31]) },
    },
  };
  const m = datasetMeasure(t);
  assert.equal(m.shape, 'record');
  assert.deepEqual(Object.keys(m.fields), ['s', 'p[1]', 'p[2]']);
  assert.deepEqual(Array.from(m.fields.s.samples), [1, 2, 3]);
  assert.deepEqual(Array.from(m.fields['p[1]'].samples), [10, 20, 30]);
  assert.deepEqual(Array.from(m.fields['p[2]'].samples), [11, 21, 31]);
  assert.deepEqual(m.dataset, { rows: 3, n_eff: 123.5, sir: t.sir });
  // Its rows are draws, so it is not shown as a constant record, whatever
  // its row count against the sample count.
  assert.equal(measureIsConstant({ SAMPLE_COUNT: 50000 }, m), false);
});

test('the engine names a record iid as a dataset and nothing else', () => {
  const src = `
P = normalize(weighted((x, y) -> x * y, Lebesgue(cartprod(x = interval(0.0, 1.0), y = interval(0.0, 1.0)))))
T ~ iid(P, 10)
U ~ iid(Normal(0, 1), 10)
`;
  const proc = engine.processSource(src);
  const built = engine.orchestrator.buildDerivations(proc.linkedBindings);
  const ctx = { derivations: built.derivations, bindings: built.bindings };
  assert.equal(engine.materialiser.isRecordIidDataset('T', ctx), true);
  assert.equal(engine.materialiser.isRecordIidDataset('U', ctx), false);
  assert.equal(engine.materialiser.isRecordIidDataset('P', ctx), false);
});
