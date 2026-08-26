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

function bindingOf(src: string, name: string): any {
  const r = processSource(src);
  return r.loweredModule.bindings.get(name);
}

function massOf(src: string, name: string): any {
  const b = bindingOf(src, name);
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

// ── AGENTS.md:280's "structural ops still fall through" list, per op ───────
//
// The four ops split FOUR ways at the mass layer, not the two or three ways
// earlier drafts of the AGENTS.md clause claimed — measured here with a
// WELL-FORMED call for each, so a malformed call's own argument error can't
// stand in for a type gap that isn't actually there (this is exactly how
// `restrict` got misclassified as "fully deferred" in an earlier round: the
// probe that produced that reading had an unrelated observation-argument
// error).

const MASS_LAYER_OPS: Array<{
  name: string; src: string; kind: string;
  mass?: string; resultMass?: string; drawErrorSubstring: string | null;
}> = [
  {
    // A single-target destructure (`m = disintegrate(...)`, silently
    // discarding the forward kernel) gives `kind: 'deferred'` with ZERO
    // diagnostics instead — the malformed-call trap that caught `restrict`
    // in an earlier round, reproduced here deliberately as a regression
    // guard rather than left as a discovery someone else has to repeat.
    name: 'disintegrate',
    src: 'u ~ Normal(mu = 0.0, sigma = 1.0)\n'
      + 'x ~ Normal(mu = u, sigma = 1.0)\n'
      + 'joint_model = lawof(record(obs = x, u = u))\n'
      + 'fk, m = disintegrate("obs", joint_model)\n',
    kind: 'measure',
    mass: 'normalized', // the prior's own mass, via the ordinary joint/record rule
    drawErrorSubstring: null, // %normalized passes the gate cleanly
  },
  {
    name: 'kernelof',
    src: 'mu = elementof(reals)\n'
      + 'x ~ Normal(mu = mu, sigma = 1.0)\n'
      + 'm = kernelof(x, mu = mu)\n',
    kind: 'kernel',
    resultMass: 'normalized',
    drawErrorSubstring: 'expects measure, got kernel', // kind short-circuit, not mass
  },
  {
    name: 'relabel',
    src: 'm = relabel(joint(Normal(mu = 0.0, sigma = 1.0), '
      + 'Beta(alpha = 1.0, beta = 1.0)), ["a", "b"])\n',
    kind: 'measure',
    mass: 'deferred',
    drawErrorSubstring: null, // %deferred passes the gate (§11: not yet inferred)
  },
  {
    name: 'restrict',
    src: 'jj = joint(a = Normal(mu = 0.0, sigma = 1.0), '
      + 'b = Beta(alpha = 1.0, beta = 1.0))\n'
      + 'm = restrict(jj, record(a = 0.5))\n',
    kind: 'measure',
    mass: 'unknown',
    drawErrorSubstring: 'total mass is %unknown', // REFUSES at the mass layer
  },
];

for (const spec of MASS_LAYER_OPS) {
  test(`${spec.name}: well-formed call types kind: '${spec.kind}'`
    + (spec.mass ? `, mass: '${spec.mass}'` : '')
    + (spec.resultMass ? `, result.mass: '${spec.resultMass}'` : ''), () => {
    assert.deepEqual(errorsOf(spec.src), [], 'the probe itself must be clean');
    const t = bindingOf(spec.src, 'm').inferredType;
    assert.equal(t.kind, spec.kind);
    if (spec.mass !== undefined) assert.equal(t.mass, spec.mass);
    if (spec.resultMass !== undefined) {
      assert.equal(t.result && t.result.mass, spec.resultMass);
    }
  });

  test(`${spec.name}: 'y ~ m' ` + (spec.drawErrorSubstring
    ? `refuses at the mass layer ("${spec.drawErrorSubstring}")`
    : 'passes the draw gate'), () => {
    const errors = errorsOf(spec.src + 'y ~ m\n');
    if (spec.drawErrorSubstring) {
      assert.ok(errors.some((e: string) => e.includes(spec.drawErrorSubstring!)),
        'got: ' + errors.join(' | '));
    } else {
      assert.deepEqual(errors, []);
    }
  });
}

test('disintegrate: a single-target destructure is now a located arity '
  + 'error, not the silent malformed-call trap', () => {
  // Discarding the forward kernel by binding only one name used to give
  // `kind: 'deferred'` with ZERO diagnostics — silently wrong, and the
  // exact shape that made an earlier AGENTS.md draft call `disintegrate`
  // "fully deferred at both layers". It now raises a located diagnostic
  // naming the op, the produced arity, and the target count instead.
  const src = 'u ~ Normal(mu = 0.0, sigma = 1.0)\n'
    + 'x ~ Normal(mu = u, sigma = 1.0)\n'
    + 'joint_model = lawof(record(obs = x, u = u))\n'
    + 'm = disintegrate("obs", joint_model)\n';
  const errors = errorsOf(src);
  assert.equal(errors.length, 1);
  assert.match(errors[0], /disintegrate/);
  assert.match(errors[0], /2/);
  assert.match(errors[0], /1/);
});
