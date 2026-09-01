'use strict';

// =====================================================================
// FlatPDL kernel TAGS vs source-level distribution NAMES.
// =====================================================================
//
// A determinised module names its kernel by the bare constructor symbol, and
// for a §09 standard-module member the module qualification is already
// discharged: `hep.CrystalBall(m0 = …)` in FlatPPL becomes
// `builtin_logdensityof(CrystalBall, record(m0 = …), x)` in FlatPDL. So
// `lower.ts`'s three `builtin_*` kernel gates admit `FLATPDL_KERNEL_TAGS`,
// which is wider than `DISTRIBUTIONS`.
//
// The separation is the point. §09 "Standard modules" gives a member no
// unqualified spelling, so `CrystalBall(...)` written directly in FlatPPL must
// stay an undefined variable — which is what widening `DISTRIBUTIONS` (and
// through it `ALL_KNOWN` and the analyzer's bare-name resolution) would have
// broken. Before this split, every §09 particle-physics member was
// unscoreable: the densities are REGISTRY-resident and evaluate fine, but the
// name was rejected at lowering.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const engine = require('../index.ts');
const builtins = require('../builtins.ts');
const sampler = require('../sampler.ts');
const lowerMod = require('../lower.ts');
const stdlibErfc = require('@stdlib/math-base-special-erfc');

const { DISTRIBUTIONS, FLATPDL_KERNEL_TAGS, ALL_KNOWN } = builtins;

const SEP = ['CrystalBall', 'DoubleSidedCrystalBall', 'Argus',
             'RelativisticBreitWigner', 'Voigtian', 'Landau',
             'BifurcatedNormal', 'ContinuedPoisson'];

/** Lower one bare RHS expression to FlatPIR-JSON. */
function lowerRHS(expr: string): any {
  const ctx = engine.processSource(`flatppl_compat = "0.1"\n__x__ = ${expr}\n`);
  const b = ctx.bindings.get('__x__');
  assert.ok(b, 'test helper: binding __x__ not found');
  return lowerMod.lowerExpr(b.node.value);
}

/** The evaluated value of a fixed-phase binding. */
function fixedValue(src: string, name: string): any {
  const r = engine.processSource(src);
  const errs = r.diagnostics.filter((d: any) => d.severity === 'error');
  assert.deepEqual(errs.map((d: any) => d.message), [],
    'the FlatPDL module must analyze cleanly');
  const orchestrator = require('../orchestrator.ts');
  const derivs = orchestrator.buildDerivations(r.bindings, r.loweredModule);
  return derivs.fixedValues.get(name);
}

// =====================================================================
// The set relations
// =====================================================================

// Every kernel the evaluator can resolve must also LOWER. Pinning the two
// directions together is what keeps a newly registered density from
// evaluating fine while still failing at lowering.
test('FLATPDL_KERNEL_TAGS covers the whole sampler REGISTRY', () => {
  for (const name of Object.keys(sampler._internal.REGISTRY)) {
    assert.ok(
      FLATPDL_KERNEL_TAGS.has(name),
      `REGISTRY distribution '${name}' is not a FlatPDL kernel tag, so a `
      + `determinised module naming it cannot lower`);
  }
});

test('FLATPDL_KERNEL_TAGS is a superset of the §08 DISTRIBUTIONS', () => {
  for (const name of DISTRIBUTIONS) {
    assert.ok(FLATPDL_KERNEL_TAGS.has(name),
      `§08 name '${name}' must stay a valid kernel tag`);
  }
});

// The §09 names are tags ONLY. Leaking them into the source-level namespace is
// the mistake this split exists to prevent.
test('a §09 member is a kernel tag but not a source-level name', () => {
  for (const name of SEP) {
    assert.ok(FLATPDL_KERNEL_TAGS.has(name), `${name} must be a kernel tag`);
    assert.ok(!DISTRIBUTIONS.has(name),
      `${name} must NOT be in DISTRIBUTIONS — §09 gives it no unqualified spelling`);
    assert.ok(!ALL_KNOWN.has(name),
      `${name} must NOT be in ALL_KNOWN — a bare use has to stay undefined`);
  }
});

test('bare CrystalBall(...) in FlatPPL source is still an undefined variable', () => {
  const src = 'flatppl_compat = "0.1"\n'
    + 'x = CrystalBall(m0 = 5.279, sigma = 0.003, alpha = 1.5, n = 3.0)\n';
  const diags = engine.processSource(src).diagnostics
    .filter((d: any) => d.severity === 'error');
  assert.ok(
    diags.some((d: any) => /ndefined variable 'CrystalBall'/.test(d.message)),
    'expected an undefined-variable error for the unqualified §09 name; got '
    + JSON.stringify(diags.map((d: any) => d.message)));
});

// =====================================================================
// The three gated positions
// =====================================================================

// A §09 tag in each gate `lower.ts` guards: the kernel argument of
// `builtin_logdensityof` (arg 0), of a transport primitive (arg 0), and of
// `builtin_sample` (arg 1).
test('a §09 tag lowers in all three builtin_* kernel positions', () => {
  const ld = lowerRHS('builtin_logdensityof(Argus, record(resonance = 5.29, '
    + 'slope = -20.0, power = 0.5), 5.28)');
  assert.equal(ld.args[0].kind, 'lit');
  assert.equal(ld.args[0].value, 'Argus');

  const tu = lowerRHS('builtin_touniform(CrystalBall, record(m0 = 5.279, '
    + 'sigma = 0.003, alpha = 1.5, n = 3.0), 5.28)');
  assert.equal(tu.args[0].value, 'CrystalBall');

  const sm = lowerRHS('builtin_sample(rngstate(1), Landau, '
    + 'record(loc = 0.0, scale = 1.0))');
  assert.equal(sm.args[1].value, 'Landau');
});

// The DOTTED form is the determiniser's axis-native emission for `iid` and for
// a broadcast kernel — `hepphys.ContinuedPoisson.(rates)` becomes
// `builtin_logdensityof.(ContinuedPoisson, broadcast(record, rate = rates),
// obs)`, which parses as `broadcast(builtin_logdensityof, ContinuedPoisson, …)`
// (spec §05 "Broadcasting syntax"). The tag therefore sits one slot right, and
// the analyzer has to exempt it there too — before this it scored correctly but
// still reported the tag as an undefined variable, a wrong error verdict on a
// legal FlatPDL module.
test('a §09 tag in the dotted broadcast form analyzes cleanly and scores', () => {
  const src = 'flatppl_compat = "0.1"\n'
    + 'rates = [2.0, 3.0]\n'
    + 'obs = [1.0, 2.0]\n'
    + 'lp = sum(builtin_logdensityof.(ContinuedPoisson, '
    + 'broadcast(record, rate = rates), obs))\n';

  // §09 ContinuedPoisson: log f = x·ln(rate) - rate - lnΓ(x+1), summed. The
  // observations are integers, where the continuous extension agrees with
  // Poisson's log-pmf, so lnΓ(x+1) = ln(x!) exactly.
  const terms = [[1.0, 2.0], [2.0, 3.0]].map(
    ([x, rate]) => x * Math.log(rate) - rate - Math.log(factorial(x)));
  const expected = terms[0] + terms[1];

  const lp = fixedValue(src, 'lp');
  assert.ok(typeof lp === 'number', `lp should be a number, got ${JSON.stringify(lp)}`);
  assert.ok(Math.abs(lp - expected) < 1e-12,
    `§09 ContinuedPoisson sum: got ${lp}, closed form ${expected}`);
});

function factorial(n: number): number {
  let f = 1;
  for (let i = 2; i <= n; i++) f *= i;
  return f;
}

// Refuse-don't-mislower is unchanged: a name no registry resolves is still
// rejected at lowering rather than reaching the evaluator as an unknown kernel.
test('an unregistered kernel tag still throws at lowering, in all three gates', () => {
  const want = /kernel argument must be a built-in or standard-module distribution name/;
  assert.throws(
    () => lowerRHS('builtin_logdensityof(NotADistribution, record(a = 1.0), 0.5)'),
    want);
  assert.throws(
    () => lowerRHS('builtin_touniform(NotADistribution, record(a = 1.0), 0.5)'),
    want);
  assert.throws(
    () => lowerRHS('builtin_sample(rngstate(1), NotADistribution, record(a = 1.0))'),
    want);
});

// =====================================================================
// End to end, against a closed form
// =====================================================================

// Score a FlatPDL module naming a §09 kernel, the shape flatppl-rust's
// determiniser emits for `hep.CrystalBall(...)`. The oracle is §09's own
// CrystalBall density with a CLOSED-FORM normalizer, computed here — not taken
// from any engine (§09 leaves `M` as "a normalizing constant"):
//
//   for |t| <= alpha:  log f = -t²/2 - log M,     t = (x - m0)/sigma
//   M = sigma·( (n/a)·exp(-a²/2)/(n-1) + sqrt(pi/2)·(1 + erf(a/sqrt 2)) ),  a = |alpha|
//
// The tail piece integrates A·(B-t)^(-n) over t < -a (substituting u = B - t,
// whose lower limit is B + a = n/a); the core piece integrates exp(-t²/2) over
// t >= -a. scipy.stats.crystalball(beta=alpha, m=n, loc=m0, scale=sigma).logpdf
// reproduces this to 0-9e-16 absolute at the b_mass_peak corpus masses.
test('a §09 kernel scores through the FlatPDL pipeline', () => {
  const m0 = 5.279, sigma = 0.003, alpha = 1.5, n = 3.0, x = 5.28;
  const src = 'flatppl_compat = "0.1"\n'
    + `lp = builtin_logdensityof(CrystalBall, record(m0 = ${m0}, `
    + `sigma = ${sigma}, alpha = ${alpha}, n = ${n}), ${x})\n`;

  const t = (x - m0) / sigma;
  const a = Math.abs(alpha);
  const tail = (n / a) * Math.exp(-a * a / 2) / (n - 1);
  // erf(z) = 1 - erfc(z).
  const core = Math.sqrt(Math.PI / 2) * (1 + (1 - stdlibErfc(a / Math.SQRT2)));
  const expected = -0.5 * t * t - Math.log(sigma * (tail + core));

  const lp = fixedValue(src, 'lp');
  assert.ok(typeof lp === 'number', `lp should be a number, got ${JSON.stringify(lp)}`);
  assert.ok(Math.abs(lp - expected) < 1e-12,
    `§09 CrystalBall density: got ${lp}, closed form ${expected}`);
});

// The Argus half of the same corpus row, likewise against a closed form. §09's
// density is (1/M)·x·u^p·exp(c·u) with u = 1 - (x/m0)², and for p = 0.5
//
//   M = (m0²/2)·s^(-3/2)·( (sqrt(pi)/2)·erf(sqrt s) - sqrt(s)·e^(-s) ),  s = -slope
//
// from gamma(3/2, s), the lower incomplete gamma at p + 1 = 3/2. Matches a
// direct quadrature of §09's formula to 9e-10 relative or better.
test('a §09 Argus kernel scores through the FlatPDL pipeline', () => {
  const res = 5.29, slope = -20.0, power = 0.5, x = 5.28;
  const src = 'flatppl_compat = "0.1"\n'
    + `lp = builtin_logdensityof(Argus, record(resonance = ${res}, `
    + `slope = ${slope}, power = ${power}), ${x})\n`;

  const u = 1 - (x / res) * (x / res);
  const s = -slope;
  const bracket = (Math.sqrt(Math.PI) / 2) * (1 - stdlibErfc(Math.sqrt(s)))
    - Math.sqrt(s) * Math.exp(-s);
  const mass = (res * res / 2) * Math.pow(s, -1.5) * bracket;
  const expected = Math.log(x) + power * Math.log(u) + slope * u - Math.log(mass);

  const lp = fixedValue(src, 'lp');
  assert.ok(typeof lp === 'number', `lp should be a number, got ${JSON.stringify(lp)}`);
  assert.ok(Math.abs(lp - expected) < 1e-10,
    `§09 Argus density: got ${lp}, closed form ${expected}`);
});
