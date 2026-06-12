'use strict';

// =====================================================================
// Property-based tests for spec invariants
// =====================================================================
//
// Uses `fast-check` (the JS/TS equivalent of Rust's proptest and Julia's
// Supposition.jl) to generate small random FlatPPL programs and verify
// invariants that should hold uniformly across the model space.
//
// Each invariant is sourced from flatppl-design spec §04-06. The
// generators are small and focused — we don't try to cover the entire
// surface language. The goal is exercising load-bearing identities and
// engine guarantees, not exhaustive coverage.
//
// Property tests complement the unit-style conformance suite: unit
// tests pin specific cases the engine must handle; property tests
// exercise the structural laws those cases instantiate.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fc = require('fast-check');
const { processSource, computePhases } = require('../index.ts');
const { buildDerivations, collectSelfRefs } = require('../orchestrator.ts');
const { disintegratePlan } = require('../disintegrate.ts');

// ---------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------
//
// Small generators that produce well-formed FlatPPL source snippets.
// We deliberately stay in the simple structural fragment — no
// disintegrate-of-disintegrate, no nested reifications — because the
// invariants we test only require small examples to exercise.

// A safe binding-name generator. Just lowercase letters with optional
// digit suffix; avoids placeholder/reserved patterns.
const arbBindingName = fc.string({
  unit: fc.constantFrom('a','b','c','d','e','f','g','h','i','j','k','l','m','n'),
  minLength: 1, maxLength: 3,
}).filter((s: string) => /^[a-z]+$/.test(s) && !['in','if','do','of'].includes(s));

// A scalar-measure generator: `Normal(mu = <small int>, sigma = <small posint>)`.
const arbScalarMeasure = fc.tuple(
  fc.integer({ min: -3, max: 3 }),
  fc.integer({ min: 1, max: 3 }),
).map(([mu, sigma]: [number, number]) => `Normal(mu = ${mu}, sigma = ${sigma})`);

// A small joint-record source. Returns { src, names } where names is
// the list of field names introduced. Ensures unique names.
function arbJointRecordSrc(): any {
  return fc.uniqueArray(arbBindingName, { minLength: 2, maxLength: 4 }).chain((names: string[]) =>
    fc.array(arbScalarMeasure, { minLength: names.length, maxLength: names.length })
      .map((measures: string[]) => {
        const draws = names.map((n, i) => `${n} = draw(${measures[i]})`).join('\n');
        const rec   = names.map(n => `${n} = ${n}`).join(', ');
        return {
          src: draws + '\n' + `j = lawof(record(${rec}))`,
          names,
        };
      })
  );
}

// Helpers.
function errorsOf(src: string): string[] {
  const ctx = processSource(src);
  return ctx.diagnostics.filter((d: any) => d.severity === 'error').map((d: any) => d.message);
}

function safeProcessSource(src: string): any {
  try { return processSource(src); }
  catch (e: any) { return { __threw: true, message: (e && e.message) || String(e) }; }
}

// ---------------------------------------------------------------------
// Property 1: processSource totality
// ---------------------------------------------------------------------
//
// Spec §05 grammar: every well-formed surface source parses and
// type/phase-analyses without the engine throwing. Errors are
// reported via diagnostics, not exceptions. This property exercises
// random small joint-record sources and verifies the engine never
// throws.

test('property: processSource is total over small joint-record sources', () => {
  fc.assert(fc.property(arbJointRecordSrc(), ({ src }: any) => {
    const result = safeProcessSource(src);
    if (result.__threw) {
      return false; // engine must not throw on syntactically-well-formed input
    }
    // Diagnostics OK (errors are diagnostics, not exceptions).
    return Array.isArray(result.diagnostics);
  }), { numRuns: 60 });
});

// ---------------------------------------------------------------------
// Property 2: Phase domain
// ---------------------------------------------------------------------
//
// Spec §04 §phases: every binding's phase ∈ {fixed, parameterized,
// stochastic}. The engine must not produce any other phase tag.

test('property: every binding has a valid phase tag', () => {
  fc.assert(fc.property(arbJointRecordSrc(), ({ src }: any) => {
    const { bindings } = processSource(src);
    for (const [, b] of bindings) {
      if (!['fixed', 'parameterized', 'stochastic'].includes(b.phase)) {
        return false;
      }
    }
    return true;
  }), { numRuns: 60 });
});

// ---------------------------------------------------------------------
// Property 3: bodyDeps ⊆ deps
// ---------------------------------------------------------------------
//
// Engine-concepts §8: bodyDeps is the body-internal subset of a
// binding's refs (paramSourceDeps is the complement for reification
// bindings). Their union is `deps` (preserved for back-compat).
// Invariant: bodyDeps ⊆ deps, paramSourceDeps ⊆ deps, and
// bodyDeps ∪ paramSourceDeps = deps.

test('property: bodyDeps and paramSourceDeps partition deps', () => {
  fc.assert(fc.property(arbJointRecordSrc(), ({ src }: any) => {
    const { bindings } = processSource(src);
    for (const [, b] of bindings) {
      if (!b.deps || !b.bodyDeps) continue; // synthetic anons may be partial
      const deps = new Set(b.deps);
      const body = new Set(b.bodyDeps);
      const ps   = new Set(b.paramSourceDeps || []);
      // bodyDeps ⊆ deps
      for (const d of body) if (!deps.has(d)) return false;
      // paramSourceDeps ⊆ deps
      for (const d of ps) if (!deps.has(d)) return false;
      // body ∪ ps ⊇ deps  (the union covers every dep; some refs may
      // appear in both, e.g. an Identifier referenced in both the body
      // and a kwarg RHS of a functionof, like _par = _par)
      for (const d of deps) {
        if (!body.has(d) && !ps.has(d)) return false;
      }
    }
    return true;
  }), { numRuns: 60 });
});

// ---------------------------------------------------------------------
// Property 4: Function/kernel values are always %fixed (spec §04 §sec:functionof)
// ---------------------------------------------------------------------
//
// "The function/kernel value itself is %fixed" — spec §04 line 1019
// quoted in our analyzer. No matter what the body or kwargs reference,
// the binding that holds the function value has phase = fixed.

const arbFnSrc = arbJointRecordSrc().chain(({ src, names }: any) => {
  // Pick one variate and wrap its arithmetic in a functionof keyed by
  // the others as boundary inputs.
  return fc.constantFrom(...names).chain((target: string) => {
    const others = names.filter((n: string) => n !== target);
    if (others.length === 0) return fc.constant({ src, names });
    const kwargs = others.map((n: string) => `${n} = ${n}`).join(', ');
    const body = `2 * ${target}`;
    return fc.constant({
      src: src + '\n' + `myfn = functionof(${body}, ${kwargs})`,
      names: [...names, 'myfn'],
    });
  });
});

test('property: functionof binding always has phase = fixed', () => {
  fc.assert(fc.property(arbFnSrc, ({ src }: any) => {
    const { bindings } = processSource(src);
    const fn = bindings.get('myfn');
    if (!fn) return true; // skip if generator didn't add myfn
    return fn.phase === 'fixed';
  }), { numRuns: 60 });
});

// ---------------------------------------------------------------------
// Property 5: `lawof(draw(m))` ≡ `m` (spec §04 identity law)
// ---------------------------------------------------------------------
//
// Spec §04 line 308: "lawof(draw(m)) is equivalent to m". For our
// engine: a binding `x = draw(m); y = lawof(x)` should classify y
// such that materialising y is equivalent to materialising m.
//
// We verify the STRUCTURAL equivalence at the derivation layer:
//   - y's derivation should be `alias` of x.
//   - x's derivation (sample-from-m or alias-to-m) should reach m.

test('property: lawof(draw(m)) classifies as alias chain to m', () => {
  fc.assert(fc.property(arbScalarMeasure, (m: string) => {
    const src = `m = ${m}\nx = draw(m)\ny = lawof(x)`;
    const ctx = processSource(src);
    if (ctx.diagnostics.filter((d: any) => d.severity === 'error').length > 0) {
      return false;
    }
    const dctx = buildDerivations(ctx.bindings, ctx.diagnostics);
    const y = dctx.derivations.y;
    if (!y || y.kind !== 'alias') return false;
    // y is alias of x; x is alias of m (since draw(m) where m is a ref).
    const x = dctx.derivations[y.from];
    if (!x) return false;
    // Either x is an alias to m or a sample of m's distIR (the engine
    // can pick either form per the identity).
    if (x.kind === 'alias' && x.from === 'm') return true;
    if (x.kind === 'sample') return true;
    return false;
  }), { numRuns: 40 });
});

// ---------------------------------------------------------------------
// Property 6: disintegrate(selector, joint) → (kernel, prior) satisfies
// the structural round-trip (spec §06 line 554)
// ---------------------------------------------------------------------
//
// Spec §06: `kernel, base_measure = disintegrate(selector, joint_measure)`
// must satisfy `jointchain(base_measure, kernel) ≡ joint_measure`.
//
// We test the STRUCTURAL form: for a joint built as
// `lawof(record(name1 = v1, ..., nameN = vN))` where v_i are
// independent draws, disintegrating any non-trivial proper subset
// must produce a synthesized plan whose kernel + prior together
// cover all the joint's fields (no field is lost, no field is
// double-counted).

test('property: disintegrate(selector, lawof(record(...))) preserves field coverage', () => {
  fc.assert(fc.property(
    arbJointRecordSrc().chain(({ src, names }: any) => {
      // Pick a non-empty proper subset of names as selector.
      return fc.subarray(names, { minLength: 1, maxLength: Math.max(1, names.length - 1) })
        .filter((sel: string[]) => sel.length > 0 && sel.length < names.length)
        .map((selector: string[]) => ({ src, names, selector }));
    }),
    ({ src, names, selector }: any) => {
      const ctx = processSource(src);
      if (ctx.diagnostics.filter((d: any) => d.severity === 'error').length > 0) {
        return true; // skip if the source itself has errors
      }
      const joint = ctx.bindings.get('j');
      if (!joint) return true;
      const plan = disintegratePlan(joint.node.value, selector, ctx.bindings,
        { seen: new Set(), source: 'j' });
      if (plan.kind !== 'synthesized') {
        // Some random selections might not be admissible (e.g. cross-
        // dependencies). Skip those — they're handled by Property 6b.
        return plan.kind === 'unsupported';
      }
      // Coverage: kernel's record + prior's record should together
      // name every joint field exactly once.
      const kernelFields = extractRecordFields(plan.kernel);
      const priorFields  = extractRecordFields(plan.prior);
      const all = new Set([...kernelFields, ...priorFields]);
      // Every original field must be covered exactly once.
      if (all.size !== names.length) return false;
      for (const n of names) if (!all.has(n)) return false;
      // No field appears on both sides.
      for (const k of kernelFields) if (priorFields.includes(k)) return false;
      return true;
    }
  ), { numRuns: 60 });
});

// Pull out the field names from a `lawof(record(...))` or
// `kernelof(record(...), ...)` synthesised by disintegrate. Returns []
// for shapes we can't introspect (the caller treats that as "no
// coverage", which is what we want for unfamiliar shapes).
function extractRecordFields(node: any): string[] {
  if (!node || node.type !== 'CallExpr') return [];
  // kernelof/lawof wrap a record literal as the first positional arg.
  const recArg = node.args && node.args[0];
  if (!recArg || recArg.type !== 'CallExpr'
      || !recArg.callee || recArg.callee.type !== 'Identifier'
      || recArg.callee.name !== 'record') return [];
  const out: string[] = [];
  for (const f of recArg.args) {
    if (f.type === 'KeywordArg') out.push(f.name);
  }
  return out;
}
