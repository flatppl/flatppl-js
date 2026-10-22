'use strict';
// MERGED IMPORTANCE WEIGHTS ENTER A CONSUMER ONCE.
//
// THE DEFECT. `propagateLogWeights` used to recognise a shared weighting event
// by reference identity of the `logWeights` array. Combining two streams
// allocates a fresh array, so both ancestors became unrecoverable, and a
// consumer that met the constituents ALONGSIDE their merged descendant counted
// each constituent twice. `record(a = a, b = b, y = y)` with `y ~ Normal(a + b,
// 1)` measured `2A + 2B` where the joint importance weight is `A + B`; the
// centered record weights were exactly twice the centered `y` weights.
//
// THE ORACLE, and it is closed form. `weighted(fn(exp(c * _)), Normal(0, 1))`
// normalized is `Normal(c, 1)`: completing the square gives
// `exp(cx)·φ(x) = exp(c²/2)·φ(x − c)`, so the tilt only shifts the mean. The
// two latents are therefore exactly `Normal(1, 1)` and `Normal(0.5, 1)`, every
// downstream mean below is an exact number, and §04 "Reification to measures"
// says `record(...)` reifies the total law of those existing draws — recording
// them must not change their law. §06 "Density reweighting", "Normalization and
// mass" and "Joint composition" govern the weights themselves.
//
// Each witness pins the correct mean AND excludes the doubled one, so a green
// test cannot mean "the number moved somewhere plausible".

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const lineage = require('../weight-lineage.ts');

const H = 'flatppl_compat = "0.1"\n';

// Two independently tilted standard normals: exactly Normal(1, 1) and
// Normal(0.5, 1).
const LATENTS = H
  + 'ma = normalize(weighted(fn(exp(_)), Normal(0.0, 1.0)))\n'
  + 'mb = normalize(weighted(fn(exp(0.5 * _)), Normal(0.0, 1.0)))\n'
  + 'a ~ ma\n'
  + 'b ~ mb\n';

const N = 60000;

// Flatten a measure into one labelled column per scalar coordinate: a record's
// fields, a tuple's elements, and an `iid` atom's k coordinates (atom-major).
function columns(m: any, prefix: string, out: [string, Float64Array][] = []) {
  if (m.shape === 'record') {
    for (const k of Object.keys(m.fields)) columns(m.fields[k], prefix + '.' + k, out);
    return out;
  }
  if (m.shape === 'tuple') {
    m.elems.forEach((e: any, i: number) => columns(e, prefix + '.' + i, out));
    return out;
  }
  const s: Float64Array = m.samples;
  const k = (m.dims && m.dims.length === 1 && m.dims[0] > 1) ? m.dims[0] : 1;
  if (k === 1) { out.push([prefix, s]); return out; }
  const n = s.length / k;
  for (let j = 0; j < k; j++) {
    const col = new Float64Array(n);
    for (let i = 0; i < n; i++) col[i] = s[i * k + j];
    out.push([prefix + '[' + j + ']', col]);
  }
  return out;
}

// Self-normalised weighted mean and its Monte-Carlo standard error of one
// column, with the weights `w` (sum `sw`, Kish ESS `ess`).
//
// Keep this a top-level function that returns to a plain `for` loop. The same
// loop inside an arrow function handed to `Array.prototype.map`, returning an
// object literal, is miscompiled by V8's on-stack replacement (Node 24's V8
// 13.6 and Node 26's 14.6): on roughly 1 run in 25 the first columns came back
// with the LAST column's mean and error, so `r.a` read the mean of `r.u`.
// `--no-use-osr`, `--no-maglev` and `--no-turbofan` each removed it.
function columnMoments(label: string, col: Float64Array, w: Float64Array,
  sw: number, ess: number) {
  const n = col.length;
  let mu = 0;
  for (let i = 0; i < n; i++) mu += w[i] * col[i];
  mu /= sw;
  let v = 0;
  for (let i = 0; i < n; i++) v += w[i] * (col[i] - mu) * (col[i] - mu);
  v /= sw;
  return { label, mean: mu, se: Math.sqrt(v / ess), ess };
}

// Self-normalised weighted mean and its Monte-Carlo standard error per column.
// The error uses Kish's ESS, since these ensembles are importance-weighted.
function moments(m: any, prefix: string) {
  const cols = columns(m, prefix);
  const lw: Float64Array | null = m.logWeights;
  const n = cols[0][1].length;
  let mx = -Infinity;
  if (lw) for (let i = 0; i < n; i++) if (lw[i] > mx) mx = lw[i];
  const w = new Float64Array(n);
  let sw = 0;
  let sw2 = 0;
  for (let i = 0; i < n; i++) {
    w[i] = lw ? Math.exp(lw[i] - mx) : 1;
    sw += w[i];
    sw2 += w[i] * w[i];
  }
  const ess = (sw * sw) / sw2;
  const out = [];
  for (let j = 0; j < cols.length; j++) {
    out.push(columnMoments(cols[j][0], cols[j][1], w, sw, ess));
  }
  return out;
}

async function measures(src: string, names: string[], n = N) {
  const { proc, ctx } = ctxFor(src, n);
  const errs = proc.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0, errs.map((e: any) => e.message).join(' | '));
  const out: Record<string, any> = {};
  for (const name of names) out[name] = await ctx.getMeasure(name);
  return out;
}

// Every mean within four standard errors of its exact value, and nowhere near
// the value the double count produced.
function assertMeans(m: any, prefix: string, oracle: number[]) {
  const got = moments(m, prefix);
  assert.equal(got.length, oracle.length,
    `expected ${oracle.length} columns, got ${got.map((g) => g.label).join(', ')}`);
  got.forEach((g, j) => {
    assert.ok(Math.abs(g.mean - oracle[j]) < 4 * g.se,
      `${g.label} = ${g.mean}, oracle ${oracle[j]}, 4se ${4 * g.se} (ESS ${g.ess})`);
    assert.ok(Math.abs(g.mean - 2 * oracle[j]) > 0.2 * Math.max(Math.abs(oracle[j]), 1),
      `${g.label} = ${g.mean} is the DOUBLE-COUNTED ${2 * oracle[j]}`);
  });
  return got;
}

test('moments reads each column against the same weights', () => {
  // Weights exp(0) : exp(log 3) = 1 : 3, so the weighted mean of [0, 4] is
  // (0 + 3 * 4) / 4 = 3, of [10, 2] is (10 + 3 * 2) / 4 = 4, and the Kish ESS
  // is (1 + 3)^2 / (1 + 9) = 1.6.
  const m = {
    shape: 'record',
    fields: {
      a: { samples: Float64Array.from([0, 4]) },
      b: { samples: Float64Array.from([10, 2]) },
    },
    logWeights: Float64Array.from([0, Math.log(3)]),
  };
  const got = moments(m, 'r');
  assert.deepEqual(got.map((g) => g.label), ['r.a', 'r.b']);
  assert.ok(Math.abs(got[0].mean - 3) < 1e-12);
  assert.ok(Math.abs(got[1].mean - 4) < 1e-12);
  assert.ok(Math.abs(got[0].ess - 1.6) < 1e-12);
  // Variance of [0, 4] under (1/4, 3/4) is 1/4 * 9 + 3/4 * 1 = 3; se = sqrt(3 / 1.6).
  assert.ok(Math.abs(got[0].se - Math.sqrt(3 / 1.6)) < 1e-12);
});

// =====================================================================
// The witness: a record over a draw AND the latents it was drawn from
// =====================================================================

test('record over a merged draw and its own constituents keeps their law',
  async () => {
    const ms = await measures(LATENTS
      + 'y ~ Normal(a + b, 1.0)\n'
      + 'r = record(a = a, b = b, y = y)\n', ['a', 'b', 'y', 'r']);
    // The separate marginals were always right; they must stay right.
    assertMeans(ms.a, 'a', [1.0]);
    assertMeans(ms.b, 'b', [0.5]);
    assertMeans(ms.y, 'y', [1.5]);
    assertMeans(ms.r, 'r', [1.0, 0.5, 1.5]);
    // `y` already sums every weighting event the record's parents bring in, so
    // the merge hands back that exact array rather than re-adding a and b.
    assert.equal(ms.r.logWeights, ms.y.logWeights,
      'the record must inherit the covering stream by reference');
  });

test('lawof(record) inherits the corrected record law', async () => {
  const ms = await measures(LATENTS
    + 'y ~ Normal(a + b, 1.0)\n'
    + 'r = record(a = a, b = b, y = y)\n'
    + 'M = lawof(r)\n', ['M']);
  assertMeans(ms.M, 'M', [1.0, 0.5, 1.5]);
});

test('the defect signature is gone: record weights are not twice y\'s',
  async () => {
    const ms = await measures(LATENTS
      + 'y ~ Normal(a + b, 1.0)\n'
      + 'r = record(a = a, b = b, y = y)\n', ['y', 'r'], 4096);
    // Centered, because a shared constant shift is not a double count. Under
    // the defect this residual was exactly 0 at every atom.
    let resid = 0;
    for (let i = 0; i < 4096; i++) {
      const d = (ms.r.logWeights[i] - ms.r.logWeights[0])
        - 2 * (ms.y.logWeights[i] - ms.y.logWeights[0]);
      if (Math.abs(d) > resid) resid = Math.abs(d);
    }
    assert.ok(resid > 1e-6,
      `centered record weights are still twice y's (residual ${resid})`);
  });

// =====================================================================
// The same overlap through the other assembly shapes
// =====================================================================

test('a normalize over a joint of the latents keeps its components\' law',
  async () => {
    // §06 "Joint composition" gives the inner joint mass 1 at every atom, so
    // `normalize` leaves the law alone and the field means are the latents'.
    // This shape defeats a propagation-only patch: `normalize` allocates a
    // fresh array, so the merged joint's ancestry has to survive it.
    const ms = await measures(LATENTS
      + 'mm = normalize(joint(aa = Normal(a, 1.0), bb = Normal(b, 1.0)))\n'
      + 'y ~ mm\n'
      + 'r = record(a = a, b = b, y = y)\n', ['r']);
    assertMeans(ms.r, 'r', [1.0, 0.5, 1.0, 0.5]);
  });

test('a record over an iid fold of the merged kernel keeps the law', async () => {
  // Both coordinates are drawn from Normal(a + b, 1), so each has mean 1.5 and
  // the fold's own weights already carry both latents' events.
  const ms = await measures(LATENTS
    + 'z ~ iid(Normal(a + b, 1.0), 2)\n'
    + 'r = record(a = a, b = b, z = z)\n', ['r']);
  assertMeans(ms.r, 'r', [1.0, 0.5, 1.5, 1.5]);
});

test('a record over a pushforward of a merged law keeps the law', async () => {
  const ms = await measures(LATENTS
    + 'c = a + b\n'
    + 'Mc = pushfwd(fn(2.0 * _), lawof(c))\n'
    + 'u ~ Mc\n'
    + 'r = record(a = a, b = b, u = u)\n', ['r']);
  assertMeans(ms.r, 'r', [1.0, 0.5, 3.0]);
});

// =====================================================================
// weight-lineage.ts itself
// =====================================================================

test('an unregistered array is one opaque event, memoised by identity', () => {
  const arr = Float64Array.from([-1, -2, -3]);
  const first = lineage.lineageOf(arr);
  assert.equal(first.events.length, 1);
  assert.equal(first.events[0].values, arr);
  assert.equal(first.events[0].offset, 0);
  // The same id on a second visit is what preserves the old
  // reference-identity dedupe for sites that register nothing.
  assert.equal(lineage.lineageOf(arr).events[0].id, first.events[0].id);
});

test('two events with equal numbers stay distinct', () => {
  const a = lineage.newEvent(null, -1.5);
  const b = lineage.newEvent(null, -1.5);
  assert.notEqual(a.id, b.id);
});

test('unionEvents keeps first-seen order and drops repeats', () => {
  const e1 = lineage.newEvent(Float64Array.from([1, 2]), 0);
  const e2 = lineage.newEvent(Float64Array.from([3, 4]), 0);
  const left = lineage.register(new Float64Array(2), [e1, e2]);
  const right = lineage.register(new Float64Array(2), [e2]);
  assert.deepEqual(lineage.unionEvents([right, left]).map((e: any) => e.id),
    [e2.id, e1.id]);
});

test('coversExactly needs the whole event set and nothing more', () => {
  const e1 = lineage.newEvent(null, -1);
  const e2 = lineage.newEvent(null, -2);
  const both = lineage.register(new Float64Array(2), [e1, e2]);
  const one = lineage.register(new Float64Array(2), [e1]);
  assert.equal(lineage.coversExactly(both, [e2, e1]), true);
  assert.equal(lineage.coversExactly(one, [e1, e2]), false);
  assert.equal(lineage.coversExactly(both, [e1]), false);
  assert.equal(lineage.coversExactly(both, [e1, lineage.newEvent(null, -3)]), false);
});

test('sumEvents adds per-atom values, offsets and constants, and registers',
  () => {
    const perAtom = lineage.newEvent(Float64Array.from([1, 2, 3]), 0);
    const shifted = lineage.newEvent(Float64Array.from([10, 20, 30]), 0.5);
    const constant = lineage.newEvent(null, -1);
    const zero = lineage.newEvent(null, 0);
    const out = lineage.sumEvents([perAtom, shifted, constant, zero], 3);
    assert.deepEqual(Array.from(out), [10.5, 21.5, 32.5]);
    assert.equal(lineage.coversExactly(out, [perAtom, shifted, constant, zero]),
      true);
  });

test('eventLength reports the atom axis, -1 for a constant', () => {
  assert.equal(lineage.eventLength(lineage.newEvent(new Float64Array(7), 0)), 7);
  assert.equal(lineage.eventLength(lineage.newEvent(null, 3)), -1);
});

test('derive appends an operation\'s own event to its parent\'s', () => {
  const parentEvent = lineage.newEvent(null, -2);
  const parent = lineage.register(new Float64Array(2), [parentEvent]);
  const own = lineage.newEvent(Float64Array.from([1, 1]), -0.5);
  const out = lineage.derive(new Float64Array(2), parent, [own]);
  assert.deepEqual(lineage.lineageOf(out).events.map((e: any) => e.id),
    [parentEvent.id, own.id]);
  // A null parent starts a lineage from the operation's own events alone.
  const fresh = lineage.derive(new Float64Array(2), null, [own]);
  assert.deepEqual(lineage.lineageOf(fresh).events.map((e: any) => e.id), [own.id]);
});
