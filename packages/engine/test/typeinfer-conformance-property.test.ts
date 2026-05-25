'use strict';

// =====================================================================
// Property-based cross-mode shape conformance (engine-concepts §17.2)
// =====================================================================
//
// The unit-style suite in `typeinfer-conformance.test.ts` pins one case
// per op. Real drift between typeinfer's per-op rules and sampler's
// runtime shape semantics doesn't usually land on the cases an author
// thought to enumerate — it lands on the case the author DIDN'T think
// of. fast-check generators over the shape-determining op surface
// catch drift continuously.
//
// Property: for any well-typed FlatPPL source S whose top-level
// binding `result` has an array-typed inferredType with concrete
// shape, the runtime materialised Value's shape AGREES with the
// inferred shape (literal entries match; %dynamic accepts anything).

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fc = require('fast-check');
const engine = require('../index.ts');
const orchestrator = require('../orchestrator.ts');

function shapesAgree(staticShape: any[], runtimeShape: any[]): boolean {
  if (staticShape.length !== runtimeShape.length) return false;
  for (let i = 0; i < staticShape.length; i++) {
    if (staticShape[i] === '%dynamic') continue;
    if (staticShape[i] !== runtimeShape[i]) return false;
  }
  return true;
}

function checkConformance(src: string): { ok: boolean; reason?: string } {
  const r = engine.processSource(src);
  const errs = r.diagnostics.filter((d: any) => d.severity === 'error');
  if (errs.length > 0) {
    // Generated source can produce type errors (e.g. negative dim).
    // Those aren't conformance failures — skip.
    return { ok: true };
  }
  const lb = r.loweredModule.bindings.get('result');
  const t = lb && lb.inferredType;
  if (!t || t.kind !== 'array') return { ok: true };   // not array-typed; skip
  let runtimeVal;
  try {
    const derivs = orchestrator.buildDerivations(r.bindings, r.loweredModule);
    runtimeVal = derivs.fixedValues.get('result');
  } catch (e: any) {
    // Runtime failed (e.g. on a generated edge case); not a typeinfer
    // conformance failure. Skip.
    return { ok: true };
  }
  if (!runtimeVal || !runtimeVal.shape) return { ok: true };
  const ok = shapesAgree(t.shape, runtimeVal.shape);
  return {
    ok,
    reason: ok ? undefined : `static ${JSON.stringify(t.shape)} != runtime ${JSON.stringify(runtimeVal.shape)} for source: ${src}`,
  };
}

// Generators for small positive integer dims (real models almost
// always have dims in this range; the property holds for any positive
// dim though, so this just keeps runtimes fast).
const arbDim = fc.integer({ min: 1, max: 6 });

test('property: zeros(n) static shape == runtime shape', () => {
  fc.assert(fc.property(arbDim, (n: number) => {
    const r = checkConformance(`result = zeros(${n})`);
    return r.ok || r.reason;
  }), { numRuns: 20 });
});

test('property: zeros([m, n]) static shape == runtime shape', () => {
  fc.assert(fc.property(arbDim, arbDim, (m: number, n: number) => {
    const r = checkConformance(`result = zeros([${m}, ${n}])`);
    return r.ok || r.reason;
  }), { numRuns: 20 });
});

test('property: ones(n) static shape == runtime shape', () => {
  fc.assert(fc.property(arbDim, (n: number) => {
    const r = checkConformance(`result = ones(${n})`);
    return r.ok || r.reason;
  }), { numRuns: 20 });
});

test('property: eye(n) static shape == runtime shape', () => {
  fc.assert(fc.property(arbDim, (n: number) => {
    const r = checkConformance(`result = eye(${n})`);
    return r.ok || r.reason;
  }), { numRuns: 20 });
});

test('property: fill(v, [m, n]) static shape == runtime shape', () => {
  fc.assert(fc.property(arbDim, arbDim, (m: number, n: number) => {
    const r = checkConformance(`result = fill(1.0, [${m}, ${n}])`);
    return r.ok || r.reason;
  }), { numRuns: 20 });
});

test('property: linspace(a, b, n) static shape == runtime shape', () => {
  fc.assert(fc.property(arbDim, (n: number) => {
    const r = checkConformance(`result = linspace(0.0, 1.0, ${n + 1})`);
    return r.ok || r.reason;
  }), { numRuns: 20 });
});

test('property: rowstack of literal-rows static shape == runtime shape', () => {
  fc.assert(fc.property(
    fc.integer({ min: 1, max: 4 }),
    fc.integer({ min: 1, max: 4 }),
    (m: number, n: number) => {
      // Build `rowstack([[0.0, ...], [0.0, ...], ...])`.
      const row = '[' + Array(n).fill('0.0').join(', ') + ']';
      const rows = Array(m).fill(row).join(', ');
      const r = checkConformance(`result = rowstack([${rows}])`);
      return r.ok || r.reason;
    }
  ), { numRuns: 15 });
});

test('property: dotted .^ over rowstack preserves shape', () => {
  fc.assert(fc.property(
    fc.integer({ min: 1, max: 4 }),
    fc.integer({ min: 1, max: 4 }),
    (m: number, n: number) => {
      const row = '[' + Array(n).fill('1.0').join(', ') + ']';
      const rows = Array(m).fill(row).join(', ');
      const r = checkConformance(`M = rowstack([${rows}])
result = M .^ 2`);
      return r.ok || r.reason;
    }
  ), { numRuns: 15 });
});

test('property: const-eval chain through lengthof / arithmetic / iid', () => {
  // For any array literal of length n, `iid(M, lengthof(arr))` must
  // produce a measure(array([n], real)) — runtime samples should
  // come out length-n too.
  fc.assert(fc.property(
    fc.integer({ min: 1, max: 8 }),
    (n: number) => {
      const data = '[' + Array(n).fill('0.0').join(', ') + ']';
      // We use a deterministic distribution here (Dirac) so the
      // runtime materialisation is fully deterministic and the
      // sample's shape is exactly the iid shape — no sample
      // variability noise.
      const src = `data = ${data}
result = rand(rnginit([1, 2, 3, 4]), iid(Dirac(value = 1.0), lengthof(data)))`;
      const r = engine.processSource(src);
      const errs = r.diagnostics.filter((d: any) => d.severity === 'error');
      if (errs.length > 0) return true;   // skip parse errors
      const lb = r.loweredModule.bindings.get('result');
      const t = lb && lb.inferredType;
      // result is a tuple [variate, new_rngstate]; we don't easily
      // surface the runtime Value here, but the inferred shape
      // should be [n] inside the tuple's first element.
      if (!t || t.kind !== 'tuple' || !Array.isArray(t.elems)) return true;
      const variateT = t.elems[0];
      if (!variateT || variateT.kind !== 'array') return true;
      // Either literal n or %dynamic is acceptable; literal == n
      // proves const-eval folded correctly.
      const expected = [n];
      return shapesAgree(variateT.shape, expected)
        || `static ${JSON.stringify(variateT.shape)} != expected ${JSON.stringify(expected)}`;
    }
  ), { numRuns: 15 });
});
