// packages/engine/mh-sample.ts
'use strict';
// Backward-compatible facade: MH now runs through the shared ensemble driver
// with the independent-walker kernel. Returns the historical shape
// ({ drawsByName, chains, acceptRate }) so existing callers/tests are unaffected.
const { runMcmc } = require('./mcmc-driver.ts');
const { mhKernel } = require('./mh-kernel.ts');

function mhSample(mv: any, opts: any) {
  const r = runMcmc(mv, mhKernel, {
    nWalkers: opts.chains ?? 4,
    warmup:   opts.warmup ?? 1000,
    draws:    opts.draws ?? 1000,
    seed:     opts.seed ?? 0,
    initStep: opts.initStep ?? 1,
  });
  return { drawsByName: r.drawsByName, chains: r.walkers, acceptRate: r.acceptRate };
}

module.exports = { mhSample };
