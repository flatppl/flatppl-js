'use strict';

// #360: a load_module posterior must score its own default_pars.
//
// load_module linking (module-link.ts) prefixes binding names + refs with the
// instance alias, so a posterior's kernel boundary inputs become `model$theta1`
// while the module's own `joint(theta1=…)` variate labels and
// `record(theta1=…)` default_pars keys stay bare. The posterior θ-feed
// (mat-density) matched θ fields to the namespaced input names and rejected the
// bare point ("θ must be a record with a model$theta2 field"), while the prior
// (bare↔bare) scored fine.
//
// Spec (§04 "Module composition": load-time substitution addresses inputs by
// bare keyword names; §04 sec:functionof: input names are the leaf-node names,
// keyword-only; §06 "Likelihood construction": logdensityof(L,θ) feeds by the
// kernel's input-parameter names by field name; §11: the alias prefix is an
// internal flat-graph device): a module's boundary-input INTERFACE is its bare
// public names. A module must be able to score its own default_pars against its
// own posterior.
//
// ORACLE = the SAME model scored WITHOUT load_module (aliasing is referentially
// transparent, §04) — load_module must not change the number. No baked engine
// constant; the invariant is load_module == inline.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

const MODEL = `flatppl_compat = "0.1"
theta1_dist = Normal(0, 1)
theta2_dist = Exponential(1)
prior = joint(theta1 = theta1_dist, theta2 = theta2_dist)
theta1 = elementof(reals)
theta2 = elementof(reals)
a = 5 * theta2
b = abs(theta1) * theta2
obs ~ iid(Normal(mu = a, sigma = b), 3)
forward_kernel = kernelof(record(obs = obs))
observed_data = [1.2, 3.4, 5.1]
L = likelihoodof(forward_kernel, record(obs = observed_data))
posterior = bayesupdate(L, prior)
default_pars = record(theta1 = 0.5, theta2 = 1.0)
`;

function buildCtx(r: any) {
  const built = orchestrator.buildDerivations(r.linkedBindings || r.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 0x360 });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(), sampleCount: 1,
    rootKey: 0x360, rootSeed: 0x360, marginalizationCount: 32,
    moduleRegistry: r.loweredModule && r.loweredModule.moduleRegistry,
    getMeasure: (n: any) => {
      if (cache.has(n)) return cache.get(n);
      const p = materialiser.materialiseMeasure(n, ctx); cache.set(n, p); return p;
    },
    sendWorker: (m: any) => Promise.resolve(worker.handle(m)),
  };
  return ctx;
}

async function score(ctx: any, name: string): Promise<number> {
  const m = await ctx.getMeasure(name);
  return m.samples ? m.samples[0] : m.value;
}

test('#360: load_module posterior scores its own default_pars (bare inputs), == inline', async () => {
  // Inline reference: the same model, no load_module (bare boundary inputs).
  const inlineCtx = buildCtx(processSource(
    MODEL + 'lp_prior = logdensityof(prior, default_pars)\n'
          + 'lp_post = logdensityof(posterior, default_pars)\n'));
  const inlinePrior = await score(inlineCtx, 'lp_prior');
  const inlinePost = await score(inlineCtx, 'lp_post');
  assert.ok(Number.isFinite(inlinePost), `inline posterior finite; got ${inlinePost}`);

  // load_module path: score the module's posterior at its own default_pars.
  const root = `flatppl_compat = "0.1"
model = load_module("m.flatppl")
prior = model.prior
posterior = model.posterior
x = model.default_pars
lp_prior = logdensityof(prior, x)
lp_post = logdensityof(posterior, x)
`;
  const modCtx = buildCtx(processSource(root, { bundle: { sources: { 'm.flatppl': MODEL } } }));
  const modPrior = await score(modCtx, 'lp_prior');
  const modPost = await score(modCtx, 'lp_post');

  // Posterior no longer throws and equals the un-linked value (transparency).
  assert.ok(Number.isFinite(modPost),
    `load_module posterior must score finite (was: "θ must have a model$theta2 field"); got ${modPost}`);
  assert.ok(Math.abs(modPost - inlinePost) < 1e-9,
    `load_module posterior ${modPost} must equal inline ${inlinePost}`);
  // Prior worked before and after (bare↔bare) — guard it stays consistent too.
  assert.ok(Math.abs(modPrior - inlinePrior) < 1e-9,
    `load_module prior ${modPrior} must equal inline ${inlinePrior}`);
});
