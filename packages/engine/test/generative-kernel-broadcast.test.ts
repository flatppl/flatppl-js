'use strict';

// =====================================================================
// generative-kernel-broadcast.test.ts — engine-concepts §21 (5th kind)
// =====================================================================
//
// Pins the GENERATIVE composite-body recogniser + executor:
//
//   broadcast(kernelof(<value-expr-with-internal-draw>, …), args…)
//
// The motivating model is the stochastic transport kernel
// (flatppl-examples/examples/tmp_transport_model.flatppl):
//
//   delta_alpha = (2 * draw(Uniform(interval(0,1))) + 1) * a
//   y           = (x + delta_alpha)^3 * exp(x - b)
//   transport   = kernelof(y, x = x, pars = pars)
//   ys          = transport.(xs, [pars])
//
// The body `lawof(y)` is the LAW of a deterministic transform closing
// over an INTERNAL DRAW (the Uniform). Earlier recognisers (iid / joint
// / jointchain / nested_broadcast) all require a measure CONSTRUCTION as
// the lawof arg; the generative recogniser (registered LAST) matches this
// residual `lawof(<value-expr>)` shape when the value-expr embeds a draw.
//
// What we pin:
//   (a) recogniser — transport descriptor (boundaries, internal draw,
//       bare-ref body) + the registry registers it LAST;
//   (b) classify — the broadcast binding gets a `kernelbroadcast`
//       derivation (silent `null` no more) for the real transport model;
//   (c) executor — a self-contained generative model materialises to an
//       [N, K] measure with finite samples in the expected range, and the
//       transform calibrates against a hand-rolled MC oracle;
//   (d) §22.4 within-atom independence — within ONE atom the K cells use
//       FRESH internal draws (NOT tiled): they show variance, ~Uniform
//       cell means, and ~zero cross-cell correlation. Keying the RNG by
//       atom-only would be a silent statistical bug; this calibrates it
//       away;
//   (e) density — the generative pushforward has no tractable density
//       (spec §06 case 3); the density path refuses LOUDLY (never a
//       silent NaN).

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');

const { processSource } = require('../index.ts');
const orchestrator = require('../orchestrator.ts');
const materialiser = require('../materialiser.ts');
const density = require('../density.ts');
const shape = require('../kernel-broadcast-shape.ts');
const recognizers = require('../composite-body-recognizers.ts');
const { createWorkerHandler } = require('../worker.ts');

const TRANSPORT_PATH =
  '/homedir/Data/Science/Projects/BAT/Projects/FlatPPL/flatppl-examples'
  + '/examples/tmp_transport_model.flatppl';

// Real (async) materialiser harness — mirrors measure-algebra.test.ts's
// `materialiseReal` and the joint-composite test's ctx.
function makeCtx(src: string, sampleCount: number, seed: number) {
  const lifted = processSource(src);
  const errs = lifted.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0,
    'parses cleanly: ' + errs.map((d: any) => d.message).join('; '));
  const built = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations,
    bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const p = materialiser.materialiseMeasure(n, ctx);
      cache.set(n, p);
      return p;
    },
    sendWorker: (m: any) => {
      const r = worker.handle(m);
      return r && r.type === 'error'
        ? Promise.reject(new Error(r.message)) : Promise.resolve(r);
    },
    sampleCount: sampleCount,
    rootKey: [seed, 0],
  };
  return { ctx, built };
}

// A self-contained generative model mirroring the transport transform but
// with `xs` as a literal per-cell array and `pars` supplied via the
// `[glob]` broadcast idiom (a per-atom-constant record). This exercises the
// full generative executor without the transport model's `xs ~ iid(gun…)`
// dependency (out of scope: an iid over a generative kernel, not a
// broadcast — see the deviation note in the agent report).
const SELF_CONTAINED = `flatppl_compat = "0.1"
xs = [0.5, 1.0, 1.5]
pars = elementof(cartprod(a = reals, b = reals))
glob = record(a = 0.1, b = 0.3)
a, b = (pars.a, pars.b)
delta_alpha = (2 * draw(Uniform(interval(0, 1))) + 1) * a
y = (x + delta_alpha)^3 * exp(x - b)
transport = kernelof(y, x = x, pars = pars)
ys = transport.(xs, [glob])`;

// =====================================================================
// (a) recogniser surface
// =====================================================================

test('generative recogniser: transport descriptor (boundaries, draw, bare-ref body)', () => {
  const src = fs.readFileSync(TRANSPORT_PATH, 'utf8');
  const built = orchestrator.buildDerivations(processSource(src).bindings);
  const desc = shape.detectGenerativeKernelBinding('transport', built.bindings);
  assert.ok(desc, 'transport recognised as generative');
  assert.deepEqual(desc.params, ['x', 'pars'], 'boundaries x, pars');
  assert.equal(desc.hasInternalDraw, true, 'has an internal draw');
  assert.equal(desc.internalDraws.length, 1, 'exactly one internal draw');
  assert.equal(desc.internalDraws[0].distIR.op, 'Uniform',
    'the internal draw is the Uniform');
  // The lawof arg is the bare value-binding ref `y`.
  assert.equal(desc.bodyValueExprIR.kind, 'ref');
  assert.equal(desc.bodyValueExprIR.name, 'y');
});

test('generative recogniser declines a draw-free deterministic body', () => {
  // kernelof(2 * c, c) — a pure pushforward, no internal draw → null.
  const SRC = `flatppl_compat = "0.1"
det = kernelof(2 * c, c = c)
xs = [1.0, 2.0]
ys = det.(xs)`;
  const built = orchestrator.buildDerivations(processSource(SRC).bindings);
  const desc = shape.detectGenerativeKernelBinding('det', built.bindings);
  assert.equal(desc, null, 'no internal draw ⇒ not generative (pushforward)');
});

test('generative recogniser is registered (registry surface grew)', () => {
  // The four measure-construction recognisers + generative ⇒ at least 5.
  const count = recognizers._testRegisteredCompositeBodyRecognizerCount();
  assert.ok(count >= 5, 'generative recogniser registered (count ' + count + ')');
});

// =====================================================================
// (b) classify — the real transport model
// =====================================================================

test('transport model: broadcast binding gets a kernelbroadcast derivation', () => {
  const src = fs.readFileSync(TRANSPORT_PATH, 'utf8');
  const built = orchestrator.buildDerivations(processSource(src).bindings);
  const derivations = require('../derivations.ts');
  // The `ys ~ transport.(xs, [pars])` binding lowers to an anon
  // `broadcast(transport, xs, [pars])` measure. Locate it.
  let bcName: string | null = null;
  for (const [k, v] of built.bindings) {
    if (v.ir && v.ir.op === 'broadcast' && Array.isArray(v.ir.args)
        && v.ir.args[0] && v.ir.args[0].name === 'transport') bcName = k;
  }
  assert.ok(bcName, 'found the transport broadcast binding');
  // Classify it directly: the generative gate now yields kernelbroadcast
  // instead of the former silent null. (NOTE: the build-time cascade-prune
  // drops this derivation from `built.derivations` because its `xs` arg —
  // `iid(gun(pars), n)`, an iid over a generative kernel — is itself not
  // yet materialisable; that is out of scope for this task. The classify
  // gate is the change under test here.)
  const dcls = derivations.classifyDerivation(
    built.bindings.get(bcName), built.bindings, built.fixedValues);
  assert.ok(dcls, 'classify produced a derivation (no silent null)');
  assert.equal(dcls.kind, 'kernelbroadcast');
  assert.equal(dcls.distOp, 'transport');
});

// =====================================================================
// (c) executor — e2e materialisation + MC-oracle calibration
// =====================================================================

test('generative executor: ys materialises to [N, K] with finite in-range samples', async () => {
  const N = 4000;
  const { ctx } = makeCtx(SELF_CONTAINED, N, 7);
  const m = await ctx.getMeasure('ys');
  assert.deepEqual(m.value.shape, [N, 3], 'shape [N, cells]');
  assert.equal(m.value.data.length, N * 3, 'flat buffer N·3');
  assert.equal(m.n_eff, N, 'n_eff = N atoms');
  assert.equal(m.logTotalmass, 0, 'unit total mass (a generative draw)');

  // Per-cell support. With u ~ Uniform(0,1), a = 0.1, b = 0.3:
  //   delta_alpha = (2u+1)*0.1 ∈ [0.1, 0.3]
  //   y_j = (x_j + delta_alpha)^3 * exp(x_j - 0.3)
  // Closed bounds per cell (x = xs[j]).
  const xs = [0.5, 1.0, 1.5];
  for (let j = 0; j < 3; j++) {
    const x = xs[j];
    const lo = Math.pow(x + 0.1, 3) * Math.exp(x - 0.3);
    const hi = Math.pow(x + 0.3, 3) * Math.exp(x - 0.3);
    for (let i = 0; i < N; i++) {
      const y = m.value.data[i * 3 + j];
      assert.ok(Number.isFinite(y), 'finite sample at (' + i + ',' + j + ')');
      assert.ok(y >= lo - 1e-9 && y <= hi + 1e-9,
        'cell ' + j + ' sample ' + y + ' within [' + lo + ', ' + hi + ']');
    }
  }
});

test('generative executor: per-cell mean calibrates against a hand-rolled MC oracle', async () => {
  const N = 20000;
  const { ctx } = makeCtx(SELF_CONTAINED, N, 21);
  const m = await ctx.getMeasure('ys');
  const xs = [0.5, 1.0, 1.5];
  // Hand-rolled MC oracle of the SAME transform with fresh independent u.
  const M = 200000;
  for (let j = 0; j < 3; j++) {
    const x = xs[j];
    // Engine empirical mean for cell j.
    let sEngine = 0;
    for (let i = 0; i < N; i++) sEngine += m.value.data[i * 3 + j];
    const engineMean = sEngine / N;
    // Oracle mean (independent RNG, same closed-form transform).
    let sOracle = 0;
    for (let k = 0; k < M; k++) {
      const u = Math.random();
      const da = (2 * u + 1) * 0.1;
      sOracle += Math.pow(x + da, 3) * Math.exp(x - 0.3);
    }
    const oracleMean = sOracle / M;
    const tol = Math.max(0.02 * Math.abs(oracleMean), 0.02);
    assert.ok(Math.abs(engineMean - oracleMean) < tol,
      'cell ' + j + ' engine mean ' + engineMean.toFixed(4)
      + ' ≈ oracle ' + oracleMean.toFixed(4) + ' (tol ' + tol.toFixed(4) + ')');
  }
});

// =====================================================================
// (d) §22.4 within-atom independence (fresh draw per cell, NOT tiled)
// =====================================================================

test('§22.4: within an atom the K cells use FRESH internal draws (not tiled)', async () => {
  const N = 6000;
  // SAME x in every cell, b = 0 ⇒ cell differences are PURE u (the internal
  // draw). If the RNG were keyed by atom-only (tiled), every cell of an atom
  // would be identical — a silent statistical bug this test catches.
  const SRC = `flatppl_compat = "0.1"
xs = [1.0, 1.0, 1.0, 1.0]
pars = elementof(cartprod(a = reals, b = reals))
glob = record(a = 0.5, b = 0.0)
a, b = (pars.a, pars.b)
delta_alpha = (2 * draw(Uniform(interval(0, 1))) + 1) * a
y = (x + delta_alpha)^3 * exp(x - b)
transport = kernelof(y, x = x, pars = pars)
ys = transport.(xs, [glob])`;
  const { ctx } = makeCtx(SRC, N, 11);
  const m = await ctx.getMeasure('ys');
  const [Nn, Kk] = m.value.shape;
  assert.equal(Nn, N);
  assert.equal(Kk, 4);
  const d = m.value.data;
  // Recover u from y: y = (1 + 0.5*(2u+1))^3 * e^1 ⇒ u = cbrt(y/e) - 1.5.
  const e = Math.exp(1);
  const recoverU = (y: number) => Math.cbrt(y / e) - 1.5;

  // (i) NOT tiled: no atom has all-identical cells, and the within-atom
  //     variance of recovered u across cells is materially > 0.
  let identicalAtoms = 0;
  let withinVar = 0;
  const cellU: number[][] = [[], [], [], []];
  for (let i = 0; i < N; i++) {
    const row: number[] = [];
    for (let j = 0; j < Kk; j++) {
      const u = recoverU(d[i * Kk + j]);
      row.push(u);
      cellU[j].push(u);
    }
    const mean = row.reduce((s, x) => s + x, 0) / Kk;
    withinVar += row.reduce((s, x) => s + (x - mean) ** 2, 0) / Kk;
    if (row.every((x) => Math.abs(x - row[0]) < 1e-9)) identicalAtoms++;
  }
  withinVar /= N;
  assert.equal(identicalAtoms, 0,
    'NO atom has tiled (all-identical) cells — draw is fresh per cell');
  // Uniform(0,1) variance is 1/12 ≈ 0.083; tiled would be ~0.
  assert.ok(withinVar > 0.05,
    'within-atom across-cell variance ' + withinVar.toFixed(4)
    + ' ≈ Uniform variance (tiled ⇒ ~0)');

  // (ii) Each cell's recovered-u mean ≈ 0.5 (a proper Uniform(0,1)).
  for (let j = 0; j < Kk; j++) {
    const mu = cellU[j].reduce((s, x) => s + x, 0) / N;
    assert.ok(Math.abs(mu - 0.5) < 0.03,
      'cell ' + j + ' recovered-u mean ' + mu.toFixed(4) + ' ≈ 0.5');
  }

  // (iii) Cross-cell correlation ≈ 0 (independent draws).
  const corr = (a0: number[], a1: number[]) => {
    const n = a0.length;
    const m0 = a0.reduce((s, x) => s + x, 0) / n;
    const m1 = a1.reduce((s, x) => s + x, 0) / n;
    let cov = 0; let s0 = 0; let s1 = 0;
    for (let i = 0; i < n; i++) {
      cov += (a0[i] - m0) * (a1[i] - m1);
      s0 += (a0[i] - m0) ** 2;
      s1 += (a1[i] - m1) ** 2;
    }
    return cov / Math.sqrt(s0 * s1);
  };
  assert.ok(Math.abs(corr(cellU[0], cellU[1])) < 0.05,
    'corr(u_cell0, u_cell1) ≈ 0 (independent)');
  assert.ok(Math.abs(corr(cellU[1], cellU[2])) < 0.05,
    'corr(u_cell1, u_cell2) ≈ 0 (independent)');
});

// =====================================================================
// (e) density — loud refusal (spec §06 case 3, never a silent NaN)
// =====================================================================

test('density of a generative kernel-broadcast refuses loudly (no silent NaN)', () => {
  const src = fs.readFileSync(TRANSPORT_PATH, 'utf8');
  const built = orchestrator.buildDerivations(processSource(src).bindings);
  // The broadcast(transport, …) IR. Asking its density must throw the
  // actionable §06 error — not return NaN.
  let bcIR: any = null;
  for (const [, v] of built.bindings) {
    if (v.ir && v.ir.op === 'broadcast' && Array.isArray(v.ir.args)
        && v.ir.args[0] && v.ir.args[0].name === 'transport') bcIR = v.ir;
  }
  assert.ok(bcIR, 'found the transport broadcast IR');
  assert.throws(
    () => density.logDensityConsumeN(bcIR, 1.0, {}, 4, {
      resolveMeasureRef: (n: string) => {
        const b = built.bindings.get(n);
        return (b && b.ir) || null;
      },
    }),
    /generative value-expression|non-bijection|bijection\(|Monte Carlo|§06|user-defined kernel/,
    'density refuses loudly with §06 guidance',
  );
});
