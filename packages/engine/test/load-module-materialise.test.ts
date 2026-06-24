'use strict';

// Stage 3 — cross-module MATERIALISATION (spec §04 Module composition).
// A loaded module's value / measure bindings materialise and score
// correctly when referenced from the loading module, with load-time
// substitutions flowing through. The design is a derivation-time "module
// linking" pass (`processSource` exposes `result.linkedBindings`): loaded
// bindings are spliced under namespaced names with refs rewritten and
// substituted inputs rewired, so the EXISTING by-name materialiser /
// density / sampler handle everything unchanged (spec §11 "tooling may
// flatten internally").
//
// Oracles are closed-form (Normal density by hand); the harness runs the
// real main-thread materialiser against the in-process worker.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const fs = require('node:fs');
const path = require('node:path');

const { processSource, orchestrator, materialiser, moduleDeps, moduleResolve } = require('..');
const { createWorkerHandler } = require('../worker.ts');

const SAMPLE_COUNT = 4096;
const ROOT_SEED = 0x10AD11; // "loadll"
const FIXTURES = path.join(__dirname, 'fixtures');

// Build a materialiser context over a processSource result (the linked
// graph), backed by an in-process worker.
function buildCtx(r: any) {
  const built = orchestrator.buildDerivations(r.linkedBindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: ROOT_SEED });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations,
    bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure: (name: any) => {
      if (cache.has(name)) return cache.get(name);
      const p = materialiser.materialiseMeasure(name, ctx);
      cache.set(name, p);
      return p;
    },
    sendWorker: (msg: any) => {
      const reply = worker.handle(msg);
      if (reply && reply.type === 'error') return Promise.reject(new Error(reply.message));
      return Promise.resolve(reply);
    },
    sampleCount: SAMPLE_COUNT,
    rootSeed: ROOT_SEED,
  };
  return ctx;
}

function makeCrossCtx(model: string, deps: Record<string, string>) {
  const r = processSource(model, { bundle: { sources: deps } });
  return { r, ctx: buildCtx(r) };
}

// Read a fixture `.flatppl` plus its transitive load_module dependencies
// from disk into a bundle — mirroring exactly what a host resolver does
// (walk `moduleDeps` + `moduleResolve.resolveModulePath`, fetch each).
function fixtureBundle(primaryRel: string) {
  const primarySource = fs.readFileSync(path.join(FIXTURES, primaryRel), 'utf8');
  const sources: Record<string, string> = {};
  (function walk(importerRel: string, text: string) {
    for (const rel of moduleDeps(text)) {
      const resolved = moduleResolve.resolveModulePath(importerRel, rel);
      if (sources[resolved]) continue;
      const depText = fs.readFileSync(path.join(FIXTURES, resolved), 'utf8');
      sources[resolved] = depText;
      walk(resolved, depText);
    }
  })(primaryRel, primarySource);
  return { primarySource, primaryPath: primaryRel, sources };
}

function makeFixtureCtx(primaryRel: string) {
  const b = fixtureBundle(primaryRel);
  const r = processSource(b.primarySource, { path: b.primaryPath, bundle: { sources: b.sources } });
  return { r, ctx: buildCtx(r) };
}

function normalLogpdf(x: number, mu: number, sigma: number) {
  return -0.5 * Math.log(2 * Math.PI) - Math.log(sigma)
    - 0.5 * ((x - mu) / sigma) ** 2;
}

function mean(s: Float64Array) {
  let t = 0; for (let i = 0; i < s.length; i++) t += s[i]; return t / s.length;
}
function std(s: Float64Array) {
  const m = mean(s); let t = 0;
  for (let i = 0; i < s.length; i++) t += (s[i] - m) ** 2;
  return Math.sqrt(t / (s.length - 1));
}

test('cross-module value ref materialises (no substitution)', async () => {
  const { r, ctx } = makeCrossCtx('m = load_module("h.flatppl")\ny = m.c * 2', {
    'h.flatppl': 'c = 2.5',
  });
  assert.equal(r.diagnostics.filter((d: any) => d.severity === 'error').length, 0);
  const y = await ctx.getMeasure('y');
  assert.ok(Math.abs(y.samples[0] - 5.0) < 1e-9,
    'm.c (2.5) * 2 = 5.0, materialised cross-module');
});

test('cross-module measure density (no substitution) matches closed form', async () => {
  const { r, ctx } = makeCrossCtx(
    'm = load_module("h.flatppl")\nlp = logdensityof(m.dist, 0.5)', {
      'h.flatppl': 'dist = Normal(0.0, 1.0)',
    });
  assert.equal(r.diagnostics.filter((d: any) => d.severity === 'error').length, 0);
  const lp = await ctx.getMeasure('lp');
  assert.ok(Math.abs(lp.samples[0] - normalLogpdf(0.5, 0, 1)) < 1e-9,
    'logdensityof(m.dist, 0.5) = N(0,1) logpdf at 0.5');
});

test('substitution flows: loaded measure density reflects the substituted input', async () => {
  // helpers.dist = Normal(mu, 1); the loader pins mu = 0.5 via load-time
  // substitution, so logdensityof(mm.dist, 1.5) = N(0.5, 1) logpdf at 1.5.
  const { r, ctx } = makeCrossCtx(
    'theta = 0.5\nmm = load_module("h.flatppl", mu = theta)\nlp = logdensityof(mm.dist, 1.5)', {
      'h.flatppl': 'mu = elementof(reals)\ndist = Normal(mu, 1.0)',
    });
  assert.equal(r.diagnostics.filter((d: any) => d.severity === 'error').length, 0);
  const lp = await ctx.getMeasure('lp');
  assert.ok(Math.abs(lp.samples[0] - normalLogpdf(1.5, 0.5, 1)) < 1e-9,
    'substituted mu = 0.5 flows into the loaded Normal density');
});

test('substitution flows: sampling a loaded measure honours the substituted input', async () => {
  const { r, ctx } = makeCrossCtx(
    'mm = load_module("h.flatppl", mu = 3.0)\nx = draw(mm.dist)', {
      'h.flatppl': 'mu = elementof(reals)\ndist = Normal(mu, 0.5)',
    });
  assert.equal(r.diagnostics.filter((d: any) => d.severity === 'error').length, 0);
  const x = await ctx.getMeasure('x');
  assert.ok(Math.abs(mean(x.samples) - 3.0) < 0.05, 'sample mean ≈ substituted mu (3.0)');
  assert.ok(Math.abs(std(x.samples) - 0.5) < 0.05, 'sample std ≈ loaded sigma (0.5)');
});

// ---------------------------------------------------------------------
// File-based fixtures (test/fixtures/load-module/): a shared dependency
// `channel.flatppl` loaded by two primaries — one WITH a load-time
// substitution, one WITHOUT — read from disk and resolved exactly as a
// host would. Closed-form (Normal) oracles.
// ---------------------------------------------------------------------

test('FIXTURE with substitution: model-subst scores N(theta, sigma) cross-module', async () => {
  const { r, ctx } = makeFixtureCtx('load-module/model-subst.flatppl');
  assert.equal(r.diagnostics.filter((d: any) => d.severity === 'error').length, 0);
  // The dep was fetched into the bundle by walking moduleDeps off disk.
  assert.ok(r.modules.get('load-module/channel.flatppl'), 'channel.flatppl compiled');
  const lp = await ctx.getMeasure('lp');
  // ch.dist = Normal(mu, 0.7); mu substituted with theta = 1.5; scored at 2.0.
  assert.ok(Math.abs(lp.samples[0] - normalLogpdf(2.0, 1.5, 0.7)) < 1e-9,
    'substituted mu = 1.5 flows from model-subst into the loaded measure');
});

test('FIXTURE without substitution: model-nosubst scores N(0, sigma) cross-module', async () => {
  const { r, ctx } = makeFixtureCtx('load-module/model-nosubst.flatppl');
  assert.equal(r.diagnostics.filter((d: any) => d.severity === 'error').length, 0);
  const lp = await ctx.getMeasure('lp');
  // ch.dist0 = Normal(0, 0.7), concrete (no substitution); scored at 0.5.
  assert.ok(Math.abs(lp.samples[0] - normalLogpdf(0.5, 0.0, 0.7)) < 1e-9,
    'the loaded concrete measure scores correctly with no load-time substitution');
});

test('transitive load: model → A → B materialises a B value through both hops', async () => {
  const { r, ctx } = makeCrossCtx(
    'a = load_module("a.flatppl")\ny = a.va + 1', {
      'a.flatppl': 'b = load_module("b.flatppl")\nva = b.vb * 10',
      'b.flatppl': 'vb = 4',
    });
  assert.equal(r.diagnostics.filter((d: any) => d.severity === 'error').length, 0);
  const y = await ctx.getMeasure('y');
  assert.ok(Math.abs(y.samples[0] - 41.0) < 1e-9, 'b.vb(4)*10 + 1 = 41 across two module hops');
});

test('mcmcRun resolves a cross-module bayesupdate posterior (load_module) via the linked graph', async () => {
  // The off-thread MCMC worker re-processes the model SOURCE and must build
  // derivations from the LINKED (cross-module-resolved) graph + the bundle —
  // else a load_module posterior has "no derivation for 'posterior'" (the
  // primary graph leaves `common.*` refs unresolved). This mirrors exactly
  // what runMcmcPool ships to each worker (source + processOpts.bundle/path).
  const b = fixtureBundle('bayesian_inference_3.flatppl');
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 1 });
  const reply: any = await Promise.resolve(worker.handle({
    type: 'mcmcRun',
    source: b.primarySource,
    name: 'posterior',
    processOpts: { bundle: { sources: b.sources }, path: b.primaryPath },
    inferenceOpts: { backend: 'mh' },
    sampleCount: 256, seed: 1,
  }));
  assert.notEqual(reply.type, 'error', 'mcmcRun errored: ' + (reply && reply.message));
  assert.equal(reply.type, 'mcmcResult');
  assert.ok(reply.measure, 'returns a posterior measure');
});
