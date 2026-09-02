// @ts-nocheck — test file; compiled separately by node --test (not by tsc)
import { registerHooks } from 'node:module';
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

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

const require_ = createRequire(import.meta.url);
const engine = require_('../../engine/index.ts');
// buildPlotPlan reads the engine off the host global, the way the webview and
// the playground provide it.
globalThis.FlatPPLEngine = engine;

const { buildPlotPlan } = await import('./plot-plan.ts');

// A binding with no derivation gets an IMPLICIT reification (spec §04: clicking
// `x` is plotting `kernelof(x)` / `functionof(x)` with no boundary kwargs). The
// dispatch used to read PHASE alone, so a parameterized MEASURE — a truncate
// over a latent bound, a posterior over free `elementof` boundaries — was
// handed to the implicit FUNCTION builder, which refuses a measure-typed
// subject by contract. Both builders then declined and the viewer reported the
// whole chain "Not plottable". A measure samples rather than evaluates, so it
// takes the kernel path regardless of phase.

function ctxFor(src) {
  const lifted = engine.processSource(src);
  const errs = (lifted.diagnostics || []).filter((d) => d.severity === 'error');
  assert.deepEqual(errs.map((d) => d.message), [], 'source analyses cleanly');
  const built = engine.orchestrator.buildDerivations(lifted.bindings);
  return {
    ctx: {
      derivationsState: {
        derivations: built.derivations,
        fixedValues: built.fixedValues || new Map(),
        discrete: built.discrete || {},
        bindings: built.bindings,
      },
      currentBindings: built.bindings,
      currentLoweredModule: lifted.loweredModule,
    },
    bindings: built.bindings,
    derivations: built.derivations,
  };
}

const LATENT_TRUNCATE = `
rate = elementof(posreals)
tau = elementof(posreals)
m = truncate(Exponential(rate = rate), interval(0.0, tau))
`;

test('a parameterized MEASURE routes to the implicit kernel plan', () => {
  const { ctx, bindings, derivations } = ctxFor(LATENT_TRUNCATE);
  const b = bindings.get('m');
  assert.equal(b.phase, 'parameterized');
  assert.equal(b.inferredType.kind, 'measure');
  assert.equal(derivations.m, undefined, 'no derivation — the implicit path applies');
  const plan = buildPlotPlan(ctx, b);
  assert.ok(plan, 'm is plottable (was "Not plottable")');
  assert.equal(plan.mode, 'kernel-sample');
  assert.deepEqual(plan.signature.inputs.map((i) => i.paramName).sort(), ['rate', 'tau']);
});

test('a parameterized VALUE still routes to the implicit profile plan', () => {
  const { ctx, bindings } = ctxFor(`
mu = elementof(reals)
mu2 = mu^2
`);
  const plan = buildPlotPlan(ctx, bindings.get('mu2'));
  assert.ok(plan);
  assert.equal(plan.mode, 'profile', 'a value evaluates, so it keeps the function path');
});

test('a bare elementof boundary stays unplottable', () => {
  // A free parameter has no measure of its own; nothing to plot is the right
  // answer, and render-plot says why.
  const { ctx, bindings } = ctxFor(LATENT_TRUNCATE);
  assert.equal(buildPlotPlan(ctx, bindings.get('rate')), null);
});

test('a posterior over a joint likelihood is plottable as a measure', () => {
  const { ctx, bindings, derivations } = ctxFor(`
obs_a = 1.5
obs_b = 3.2
mu = elementof(reals)
model_a = functionof(Normal(mu = mu, sigma = 1.0))
model_b = functionof(Normal(mu = 2.0 * mu, sigma = 0.5))
L = joint_likelihood(likelihoodof(model_a, obs_a), likelihoodof(model_b, obs_b))
prior = joint(mu = Normal(mu = 0.0, sigma = 2.0))
posterior = bayesupdate(L, prior)
`);
  // Classified rather than reached through an implicit reification: the
  // per-term joint resolution gives it a real bayesupdate derivation, so the
  // pane samples the reweighted prior instead of substituting kernel inputs.
  assert.equal(derivations.posterior.kind, 'bayesupdate');
  assert.equal(derivations.posterior.subs.length, 2, 'both instrument terms carried');
  const plan = buildPlotPlan(ctx, bindings.get('posterior'));
  assert.ok(plan, 'posterior is plottable (was "Not plottable")');
  assert.equal(plan.mode, 'samples');
});

test('a joint_likelihood binding gets a profile plan with the unioned axes', () => {
  const { ctx, bindings } = ctxFor(`
obs_a = 1.5
obs_b = 3.2
mu = elementof(reals)
model_a = functionof(Normal(mu = mu, sigma = 1.0))
model_b = functionof(Normal(mu = 2.0 * mu, sigma = 0.5))
L_a = likelihoodof(model_a, obs_a)
L_b = likelihoodof(model_b, obs_b)
L = joint_likelihood(L_a, L_b)
`);
  const plan = buildPlotPlan(ctx, bindings.get('L'));
  assert.ok(plan, 'the joint likelihood is plottable');
  assert.equal(plan.mode, 'profile');
  assert.deepEqual(plan.axes.map((a) => a.key), ['mu']);
});

test('a point-free likelihood over an inline reification carries its axes', () => {
  // §04's implicit-boundary rule reaches an INLINE `functionof`, so the
  // likelihood's own signature has the parametric leaves as inputs. Without it
  // the profile plan had zero axes and the viewer refused the likelihood.
  const { ctx, bindings } = ctxFor(`
t_obs = [0.3, 1.1, 0.7]
rate = elementof(posreals)
tau = elementof(posreals)
m = truncate(Exponential(rate = rate), interval(0.0, tau))
L = likelihoodof(functionof(iid(m, 3)), t_obs)
`);
  const plan = buildPlotPlan(ctx, bindings.get('L'));
  assert.ok(plan, 'L is plottable (had 0 axes)');
  assert.equal(plan.mode, 'profile');
  assert.deepEqual(plan.axes.map((a) => a.key).sort(), ['rate', 'tau']);
});
