'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

function setupCtx(src: string, N: number) {
  const lifted = processSource(src);
  const errs = (lifted.diagnostics || []).filter((d: any) => d.severity === 'error');
  if (errs.length > 0) return { errs, ctx: null, built: null, lifted };
  const built = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler(); worker.handle({ type: 'init', seed: 42 });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    moduleRegistry: lifted.loweredModule.moduleRegistry || null,
    getMeasure: (n: string) => { if (cache.has(n)) return cache.get(n); const p = materialiser.materialiseMeasure(n, ctx); cache.set(n, p); return p; },
    sendWorker: (m: any) => { const r = worker.handle(m); return r && r.type === 'error' ? Promise.reject(new Error(r.message)) : Promise.resolve(r); },
    sampleCount: N, rootSeed: 42,
  };
  return { errs: [], ctx, built, lifted };
}
const DATA = '[1.2, 3.4, 5.1, 2.8, 4.0, 3.7, 5.5, 2.1, 4.3, 3.9]';
const ORACLE = -37.99959196868393;
async function score(src: string) {
  const { errs, ctx, lifted } = setupCtx(src, 1);
  if (errs.length) throw new Error('DIAG: ' + JSON.stringify((lifted.diagnostics || []).map((d: any) => d.message)));
  const m = await ctx!.getMeasure('__score__');
  return m.value ? m.value.data[0] : m.samples[0];
}

test('C1: joint(relabel(...)) prior classifies + materialises to a record variate', async () => {
  const { errs, ctx, built } = setupCtx(`
flatppl_compat = "0.1"
prior = joint(relabel(Normal(0, 1), ["theta1"]), relabel(Exponential(1), ["theta2"]))
`, 20);
  assert.equal(errs.length, 0);
  assert.ok(built.derivations.prior, 'prior classifies to a derivation');
  const m = await ctx.getMeasure('prior');
  assert.ok(m && m.fields && m.fields.theta1 && m.fields.theta2,
    `prior is a record variate with fields theta1, theta2 (got keys ${m && Object.keys(m)})`);
});

test('C1: relabel(joint(...)) prior classifies + materialises to a record variate', async () => {
  const { errs, ctx, built } = setupCtx(`
flatppl_compat = "0.1"
prior = relabel(joint(Normal(0, 1), Exponential(1)), ["theta1", "theta2"])
`, 20);
  assert.equal(errs.length, 0);
  assert.ok(built.derivations.prior, 'prior classifies to a derivation');
  const m = await ctx.getMeasure('prior');
  assert.ok(m && m.fields && m.fields.theta1 && m.fields.theta2,
    `prior is a record variate with fields theta1, theta2 (got keys ${m && Object.keys(m)})`);
});
