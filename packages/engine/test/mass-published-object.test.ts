'use strict';

// =====================================================================
// mass-published-object.test.ts — the published `inferredType` object
// must not be mutated by a downstream consumer (spec §11 `%mass`)
// =====================================================================
//
// `typeinfer.ts`'s mass pass (`massOfExpr` / `fillMasses`) used to WRITE
// `.mass` directly onto whatever object `ir.meta.type` / `b.inferredType`
// pointed at. Several ops return an argument's own type object verbatim
// rather than a fresh one — `inferWeighted` ("result is the same measure
// type"), `inferLawof`'s identity law, and a bare alias (`r = j`) all leave
// two different bindings' `inferredType` pointing at the exact SAME object.
// Writing `.mass` through that shared object stamps whichever binding is
// classified LAST over every earlier binding's already-correct class —
// silently, with no diagnostic, because the gate verdicts computed from the
// CORRECT class earlier in the pass are unaffected.
//
// Two reproductions, both from a point-in-time architectural review:
//
//   1. `n1 = Normal(mu = 0, sigma = 1)` reports `%normalized` standalone,
//      but flips to `%finite` — n1's OWN class, not a copy of it — the
//      moment a `weighted(0.3, n1)` binding exists ANYWHERE ELSE in the
//      module. n1 is never itself weighted.
//   2. `r = j` (a bare alias of a mass-`%deferred` measure) reports
//      `%deferred` standalone, same as `j`; both flip to `%unknown` once a
//      `weighted(2.0, r)` exists elsewhere.
//
// Both are read off the BINDING's own `inferredType.mass`, which is exactly
// what a viewer, LSP hover, or the conformance harness reads. The fix is
// copy-on-write at the two write sites (`massOfExpr`, `fillMasses` pass 1):
// each binding gets its OWN fresh type object with its OWN computed mass,
// so no binding's published class depends on what else happens to exist in
// the module. Gate verdicts (draw / lawof) were already correct before the
// fix — this defect never changed what the engine accepted or refused, only
// what it REPORTED — so this file also pins that the fix changes nothing
// about them.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index.ts');

function massOf(src: string, name: string): any {
  const r = processSource(src);
  const b = r.loweredModule.bindings.get(name);
  return b && b.inferredType ? b.inferredType.mass : undefined;
}

function errorsOf(src: string): string[] {
  const r = processSource(src);
  return (r.diagnostics || [])
    .filter((d: any) => d.severity === 'error')
    .map((d: any) => d.message);
}

// ── repro 1: weighted(0.3, n1) elsewhere must not touch n1's own class ─────

const N1_STANDALONE = 'n1 = Normal(mu = 0.0, sigma = 1.0)\n';
const N1_WITH_WEIGHTED_ELSEWHERE = N1_STANDALONE + 'w = weighted(0.3, n1)\n';

test('repro 1 — weighted(0.3, n1) elsewhere does not mutate n1\'s own '
  + 'reported mass', () => {
  assert.equal(massOf(N1_STANDALONE, 'n1'), 'normalized');
  assert.equal(massOf(N1_WITH_WEIGHTED_ELSEWHERE, 'n1'), 'normalized',
    'n1 is never itself weighted; its class must not depend on w existing');
});

test('repro 1 — the weighted binding still gets its OWN correct class', () => {
  assert.equal(massOf(N1_WITH_WEIGHTED_ELSEWHERE, 'w'), 'finite');
});

test('repro 1 — the draw gate verdict on n1 is unaffected either way', () => {
  assert.deepEqual(errorsOf(N1_STANDALONE + 'x ~ n1\n'), []);
  assert.deepEqual(errorsOf(N1_WITH_WEIGHTED_ELSEWHERE + 'x ~ n1\n'), []);
});

// ── repro 2: a %deferred alias pair must not flip to %unknown when a ──────
// ── weighted use of the alias appears elsewhere ────────────────────────────

// `KD`'s output measure has no mass rule (`relabel` over a named base — same
// shape mass-class.test.ts's "a shape with no rule is deferred" test uses),
// so `jointchain`'s base-carries-through rule leaves `j` itself `%deferred`.
const DEFERRED_BASE = 'mu = elementof(reals)\n'
  + 'jj = joint(Normal(mu = mu, sigma = 1.0), Beta(alpha = 1.0, beta = 1.0))\n'
  + 'KD = functionof(relabel(jj, ["a", "b"]), mu = mu)\n'
  + 'j = jointchain(Normal(mu = 0.0, sigma = 1.0), KD)\n'
  + 'r = j\n';
const DEFERRED_WITH_WEIGHTED_ELSEWHERE = DEFERRED_BASE + 'w = weighted(2.0, r)\n';

test('repro 2 — standalone, the alias pair both report %deferred', () => {
  assert.equal(massOf(DEFERRED_BASE, 'j'), 'deferred');
  assert.equal(massOf(DEFERRED_BASE, 'r'), 'deferred');
});

test('repro 2 — weighted(2.0, r) elsewhere must not flip j OR r to '
  + '%unknown', () => {
  assert.equal(massOf(DEFERRED_WITH_WEIGHTED_ELSEWHERE, 'j'), 'deferred',
    'j is r\'s base and is never itself weighted');
  assert.equal(massOf(DEFERRED_WITH_WEIGHTED_ELSEWHERE, 'r'), 'deferred',
    'r is only READ by weighted, not redefined by it');
});

test('repro 2 — the weighted binding still gets its OWN correct class '
  + '(a non-identity scale over a deferred base settles %unknown)', () => {
  assert.equal(massOf(DEFERRED_WITH_WEIGHTED_ELSEWHERE, 'w'), 'unknown');
});

test('repro 2 — the draw gate rejects drawing from the deferred-mass alias '
  + 'the same way with or without the weighted use elsewhere', () => {
  // `%deferred` passes the gate (spec §11: "not yet inferred" is not a
  // proven non-probability measure) — same verdict both ways, pinning that
  // the fix changes only what is REPORTED, not what is ACCEPTED.
  assert.deepEqual(errorsOf(DEFERRED_BASE + 'x ~ r\n'), []);
  assert.deepEqual(errorsOf(DEFERRED_WITH_WEIGHTED_ELSEWHERE + 'x ~ r\n'), []);
});

// ── the published object is genuinely independent, not just correct by luck

test('two bindings that alias the SAME base measure get independently '
  + 'correct classes even when weighted differently', () => {
  // Both `p` and `q` alias `n1`; `p` is weighted by a non-identity scale
  // elsewhere, `q` is not. Every one of the three must report its own
  // class, none contaminating another.
  const src = N1_STANDALONE
    + 'p = n1\nq = n1\nwp = weighted(2.0, p)\n';
  assert.equal(massOf(src, 'n1'), 'normalized');
  assert.equal(massOf(src, 'p'), 'normalized');
  assert.equal(massOf(src, 'q'), 'normalized');
  assert.equal(massOf(src, 'wp'), 'finite');
});

// ── the pre-existing bayesupdate witness (TODO-flatppl-js.md, wave JSSMALL
// review round 2) — the aliasing bug via a THIRD inferrer, not weighted/lawof

test('bayesupdate\'s posterior class does not depend on whether the prior is '
  + 'named or written inline', () => {
  // `inferBayesupdate`'s result-type construction is a fourth site that used
  // to leave the posterior's `inferredType` aliased to the (inline) prior
  // expression's own node. Pre-fix: the inline spelling read `%normalized`
  // (the aliased Normal literal's own class, stamped over the posterior's
  // actual `%unknown` evidence-integral class), while the named spelling
  // read `%unknown` correctly — two spellings of one model disagreeing, and
  // the wrong one claiming a posterior is a probability measure.
  const LL = 'mu = elementof(reals)\n'
    + 'K = kernelof(Normal(mu = mu, sigma = 1.0), mu = mu)\n'
    + 'LL = likelihoodof(K, 0.5)\n';
  const inlineSrc = LL + 'm = bayesupdate(LL, Normal(mu = 0.0, sigma = 1.0))\n';
  const namedSrc = LL + 'pr = Normal(mu = 0.0, sigma = 1.0)\nm = bayesupdate(LL, pr)\n';
  assert.equal(massOf(inlineSrc, 'm'), 'unknown');
  assert.equal(massOf(namedSrc, 'm'), 'unknown');
});
