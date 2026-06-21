'use strict';

// A `bayesupdate` posterior whose prior is NOT the law of the likelihood's
// boundary draws is cascade-pruned (the H1 boundary-conflation scar zone —
// flatppl-dev/measure-algebra-audit.md; the engine materialises only
// bayesupdate(L, lawof(draws)) + the disintegration idiom). That prune used to
// be SILENT — the posterior vanished with no diagnostic. buildDerivations now
// emits a loud error diagnostic for any bayesupdate-typed binding left without
// a derivation, so a dropped spec-canonical posterior is reported, not hidden.

const test = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator } = require('../index.ts');

// Verbatim from flatppl-design/docs/06-measure-algebra.md "Posterior construction".
const SPEC_POSTERIOR = `
mu = elementof(reals)
model = Normal(mu = mu, sigma = 1.0)
obs = 2.5
L = likelihoodof(functionof(model), obs)
prior = joint(mu = Normal(mu = 0, sigma = 2.0))
posterior = bayesupdate(L, prior)
`;

test('pruned bayesupdate posterior emits a loud diagnostic (not silent)', () => {
  const ds = orchestrator.buildDerivations(processSource(SPEC_POSTERIOR).bindings);
  // Still pruned (the materialisation fix is the separate unification work).
  assert.equal(ds.derivations.posterior, undefined, 'posterior is cascade-pruned');
  // But no longer silent: an error diagnostic names the posterior.
  const diags = (ds.diagnostics || []).filter((d: any) => d.severity === 'error');
  const hit = diags.find((d: any) => /posterior/i.test(d.message) && /bayesupdate/i.test(d.message));
  assert.ok(hit, 'a loud error diagnostic is emitted for the pruned bayesupdate posterior');
  assert.match(hit.message, /disintegrat/i, 'diagnostic points at the disintegration idiom');
});
