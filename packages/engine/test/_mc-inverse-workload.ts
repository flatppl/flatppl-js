'use strict';
// The simple-transport MC inverse, 20 x 6000 points: the perf workload of the
// fused-loop compiler and the interpreter's scalar kernels. The leading
// underscore keeps this file outside the `test/**/*.test.ts` glob.

const path = require('node:path');
const fs = require('node:fs');

function mcInverseWorkload() {
  const ENG = path.resolve(__dirname, '..') + '/';
  const { processSource, orchestrator } = require(ENG + 'index.ts');
  const mcr = require(ENG + 'mc-recipe.ts');
  const { createWorkerHandler } = require(ENG + 'worker.ts');
  const src = fs.readFileSync(path.join(ENG, 'test/fixtures/simple-transport1.flatppl'), 'utf8');
  const { bindings } = processSource(src);
  const built = orchestrator.buildDerivations(bindings);
  const expand = (nm: any) => orchestrator.expandMeasure(
    nm, { derivations: built.derivations, bindings: built.bindings });
  const form = mcr.buildMcMarginalForm(expand('zs'), built.bindings, expand);
  if (!(form && form.args && form.args[0] && form.args[0].op === 'mcmarginal')) {
    throw new Error('transport zs must re-express as iid(mcmarginal, n)');
  }
  const mc = form.args[0];
  const baseEnv = { pars: { a: 0.1, b: 0.3, mu: 1.0 }, sigma: 0.2 };
  const w = createWorkerHandler(); w.handle({ type: 'init', seed: [1, 2, 3] });
  w.handle({ type: 'setEnv', env: baseEnv, merge: true });
  const data = [3.8, 2.9, 3.2, 3.1, 3.3, 5.0, 0.84, 2.3, 3.1, 2.6, 2.5, 5.4, 1.2, 1.6, 1.4, 0.77, 0.26, 1.6, 0.56, 4.5];
  const M = 6000, D = data.length, count = D * M;
  const xs = w.handle({ type: 'sampleN', ir: mc.marginalDistIR, count: M, refArrays: {}, seed: [1, 2, 3, 4] }).samples;
  const zcol = new Float64Array(count), xcol = new Float64Array(count);
  for (let d = 0; d < D; d++) for (let m = 0; m < M; m++) { zcol[d * M + m] = data[d]; xcol[d * M + m] = xs[m]; }
  const refArrays: any = { [mc.outName]: zcol, [mc.retainedRef]: xcol, x: xcol };
  return { ir: mc.inverseIR, refArrays, count, baseEnv };
}

module.exports = { mcInverseWorkload };
