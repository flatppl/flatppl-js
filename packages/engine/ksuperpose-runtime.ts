'use strict';

// ksuperpose-runtime.ts — spec §06 `ksuperpose` with a RUNTIME component count
// ============================================================================
//
// §06 `ksuperpose`: "The number of components N is the length of `weights`,
// WHICH NEED NOT BE STATICALLY KNOWN."
//
// `ksuperpose-expand.ts` rewrites an applied lift to §06's own variadic
// spelling
//
//   ksuperpose(K, w)(p = vec) ≡ superpose(weighted(w[1], K(p = vec[1])), …)
//
// and needs N while it runs, on the AST, before any value exists. A weight
// vector whose length no TYPE carries — `reverse(v)`, `linspace(a, b, n)`,
// `cumsum(v)`, anything the shape inference leaves `%dynamic` — therefore
// could not be rewritten there, and §06's runtime N was a located refusal.
//
// This pass closes that at the one point where the value IS available:
// `buildDerivations`, after `fixedValues` exists. `fixedValues` is the lazy
// fixed-phase resolver (fixed-values.ts), so asking it for the weight vector
// evaluates that binding's subgraph and returns the real vector — its length
// is N.
//
// WHAT IT SYNTHESISES is the derivation graph the static rewrite produces,
// component for component: per component i a `sample` binding for `κ(θᵢ)`
// with each family argument indexed to row i, a `weighted` binding carrying
// the weight EXPRESSION `w[i]`, and a `superpose` derivation over the N
// weighted bindings. That is deliberate, and it is the whole design:
//
//   - DENSITY is `walkSuperpose` → `walkSelect`, so §06's
//     `logsumexp_i(log wᵢ + logdensityof(κ(θᵢ), x))` is the reviewed
//     implementation, not a second one.
//   - PER-WEIGHT RULES are `addLogW` (density.ts) on the weight expression:
//     negative refuses, zero is legal and drops the component out, NaN gets
//     its own message. Inherited verbatim rather than restated, so the two
//     spellings cannot drift on the sign rule.
//   - MASS is `additiveMass` / `normalize-mass.totalMassExpr` over the
//     `weighted` components, which is §06's `Σᵢ wᵢ·totalmass(κ(θᵢ))`, and
//     `Σᵢ wᵢ` for a Markov kernel.
//   - SAMPLING is `matSuperpose`'s PER-OUTPUT-INDEX component selection. That
//     is the property this pass exists to inherit: under `iid(…, k)` each of
//     the k coordinates must select its own component (§06's product measure;
//     flatppl-engine-concepts §22.4 "The repeat axis"), and a parallel
//     sampler would have had to re-derive it. It was hard-won once.
//
// The weight is carried as the EXPRESSION `w[i]`, not as the resolved number,
// for the same reason the static rewrite does: the walker then evaluates it at
// the scored point, so a weight that is recomputed per θ stays correct and the
// per-weight refusals fire at the value.
//
// WHAT IT STILL REFUSES, and why the refusal lives here rather than in the
// analyzer: only whoever holds the value knows whether N is readable.
//
//   - a weight vector with no resolvable value (a host-supplied `external`,
//     a latent draw) — N is genuinely unavailable;
//   - a weight vector written inline rather than bound to a name;
//   - a REIFIED or user-defined kernel — this builds a distribution CALL,
//     so `κ` must be a builtin measure kernel. The static rewrite handles a
//     reification, so the remedy is a statically-known N;
//   - a family argument whose size is neither N nor one, and a table family.
//
// Runs after the second classification pass in `buildDerivations`, before the
// cascade-prune: the synthesised bindings must be prunable like any other.

const AST = require('./ast.ts');

// The `ksuperpose(K, w)` lift a binding's IR applies, or null. The applied
// form lowers to the expression-headed user call (lower.ts) — the callee is
// the lift, the arguments are the family — because `ksuperpose` is not an
// Identifier-headed call at that position.
function _appliedLift(ir: any): any {
  if (!ir || ir.kind !== 'call' || !ir.callee) return null;
  const c = ir.callee;
  if (c.kind !== 'call' || c.op !== 'ksuperpose') return null;
  return c;
}

function _isSelfRef(ir: any): boolean {
  return !!(ir && ir.kind === 'ref' && ir.ns === 'self');
}

// The outer length of a resolved fixed value, or null when it is not a
// one-axis vector. `fixedValues` hands back the engine's Value shape
// (`{shape, data}`) for a vector and a bare number for a scalar.
function _vectorLength(v: any): number | null {
  if (!v || typeof v !== 'object') return null;
  if (Array.isArray(v.shape) && v.shape.length === 1) {
    const n = v.shape[0];
    return (typeof n === 'number' && Number.isInteger(n) && n > 0) ? n : null;
  }
  if (Array.isArray(v) && v.length > 0) return v.length;
  return null;
}

// Is the resolved value a scalar — a §06 "non-collection argument … held
// constant across the components"?
function _isScalarValue(v: any): boolean {
  if (typeof v === 'number' || typeof v === 'boolean') return true;
  return !!(v && typeof v === 'object' && Array.isArray(v.shape)
    && v.shape.length === 0);
}

function _litInt(i: number): any {
  return { kind: 'lit', value: i, numType: 'integer' };
}

// Index the FAMILY axis only. §07 gives a multi-axis row slice as
// `get(M, i, all)`, so an argument whose outer shape also carries the
// parameter's own axes takes one `all` per remaining outer axis (`lead`).
function _getRow(ir: any, row: number, lead?: number): any {
  const args: any[] = [ir, _litInt(row)];
  for (let k = 1; k < (lead || 1); k++) args.push({ kind: 'const', name: 'all' });
  return { kind: 'call', op: 'get', args };
}

// A synthesised binding. It carries no AST: this pass sets the derivation
// itself, so nothing downstream re-classifies it from source. `phase: null`
// keeps it out of the fixed-phase dead-end sweep, which would otherwise
// report every component as an engine gap.
function _synthBinding(name: string, ir: any, deps: string[], loc: any): any {
  return {
    name,
    names: [name],
    line: 0,
    rhs: null,
    type: 'call',
    deps,
    callDeps: [],
    bodyDeps: [],
    paramSourceDeps: [],
    node: AST.AssignStatement([name], null, loc),
    nameLoc: loc,
    phase: null,
    inferredType: null,
    ir,
  };
}

// One family argument of the application, classified against N.
//
//   { kind: 'const' }            — held constant across the components
//   { kind: 'axis', length: n }  — a size-n collection, read row-wise
//
// The classification reads the inferred TYPE first, by the same rule the
// static rewrite uses (`classifyNamedFamilyArg`), and falls back to the
// resolved VALUE's shape when the type carries no length — which is the whole
// reason this pass exists. A `null` return is a refusal the caller reports.
function _classifyArg(
  ir: any, label: string, bindings: any, fixedValues: any, refuse: any,
  dist?: string | null, param?: string | null, known?: boolean,
): any {
  if (ir && ir.kind === 'lit') return { kind: 'const' };
  if (!_isSelfRef(ir)) {
    refuse(`ksuperpose: family argument ${label} is an inline expression, so `
      + 'neither its type nor its value can be read to tell a size-N '
      + 'collection from a held-constant scalar (spec §06). Bind it to a name.');
    return null;
  }
  const name = ir.name;
  const byType = require('./ksuperpose-expand.ts')
    .classifyNamedFamilyArg(name, bindings);
  const complaint = require('./ksuperpose-expand.ts')
    .familyAxisComplaint(byType, dist || null, param || null, label, !!known);
  if (complaint) {
    refuse(complaint);
    return null;
  }
  if (byType && byType.kind === 'table') {
    refuse('ksuperpose: §06 allows a TABLE parameter family (one axis, its '
      + 'rows), but this engine does not lower one — the per-column family '
      + 'extraction is not built. Pass the columns as keyword vectors instead.');
    return null;
  }
  if (byType && byType.kind === 'const') return { kind: 'const' };
  if (byType && byType.kind === 'axis' && byType.length != null) {
    return { kind: 'axis', length: byType.length, lead: byType.lead || 1 };
  }
  // The type carries no length. Read the value.
  const v = fixedValues.has(name) ? fixedValues.get(name) : null;
  if (_isScalarValue(v)) return { kind: 'const' };
  const len = _vectorLength(v);
  if (len != null) return { kind: 'axis', length: len };
  refuse(`ksuperpose: family argument ${label} has no statically-known length `
    + 'and no resolvable value, so its component rows cannot be read '
    + '(spec §06). Bind it to a name whose length is known, or give the '
    + 'mixture a statically-known N.');
  return null;
}

// The distribution op a `ksuperpose` kernel argument denotes, or null.
//
// The synthesised component is a distribution CALL, so the kernel must be a
// builtin measure kernel. A reified kernel (`m -> Normal(mu = m, …)`, a
// `kernelof`, a user function) would need its body substituted per component,
// which is what the static rewrite gets for free by rewriting the AST before
// lowering — hence the remedy in the refusal is a statically-known N, not a
// different kernel.
function _kernelOp(kernelIR: any, bindings: any): string | null {
  if (!_isSelfRef(kernelIR)) return null;
  const name = kernelIR.name;
  // A ref to a real binding is a user-defined or reified kernel, never a
  // builtin: builtin distribution names lower to a self-ref with no binding.
  if (bindings.has(name)) return null;
  const orchestrator = require('./orchestrator.ts');
  const known = orchestrator.SAMPLEABLE_DISTRIBUTIONS;
  return (known && known.has(name)) ? name : null;
}

// Expand one binding whose IR is an applied `ksuperpose` lift with a runtime
// N. Mutates `bindings` and `derivations`; pushes at most one diagnostic.
// Returns true when the binding was expanded.
function _expandOne(
  name: string, binding: any, lift: any,
  bindings: any, derivations: any, fixedValues: any, diagnostics: any[],
): boolean {
  const loc = (binding.node && binding.node.loc) || null;
  let refused = false;
  const refuse = (message: string) => {
    if (refused) return;
    refused = true;
    diagnostics.push({ severity: 'error', message, loc });
  };

  const liftArgs = lift.args || [];
  if (liftArgs.length !== 2) return false;   // arity is the analyzer's refusal
  const kernelIR = liftArgs[0];
  const weightsIR = liftArgs[1];

  // N is the length of the weight vector, read from its value.
  if (!_isSelfRef(weightsIR)) {
    refuse('ksuperpose: the component count N is the length of `weights` '
      + '(spec §06), and this weight vector has neither a statically-known '
      + 'length nor a name whose value can be read. Bind it to a name.');
    return false;
  }
  const wName = weightsIR.name;
  const wValue = fixedValues.has(wName) ? fixedValues.get(wName) : null;
  const N = _vectorLength(wValue);
  if (N == null) {
    refuse('ksuperpose: the component count N is the length of `weights` '
      + `(spec §06), and \`${wName}\` has neither a statically-known length `
      + 'nor a value this engine can resolve — a host-supplied `external` '
      + 'input or a latent draw has no length at build time. §06 allows a '
      + 'runtime N, and this engine reads it from a DETERMINISTIC weight '
      + 'vector; give the vector a statically-known length instead.');
    return false;
  }

  const kernelOp = _kernelOp(kernelIR, bindings);
  if (kernelOp == null) {
    refuse('ksuperpose: with a runtime component count the mixture is built '
      + 'as a call to a builtin measure kernel, so a reified or user-defined '
      + 'kernel cannot be used here (spec §06 allows it, and the '
      + 'statically-known-N spelling supports it). Give the weight vector a '
      + 'statically-known length.');
    return false;
  }

  // Family arguments, in the application's own order and naming — the
  // component call takes each one in the position or under the keyword it
  // already has (§05 lets a distribution take its parameters positionally).
  const { known, params } = require('./ksuperpose-expand.ts')
    .componentParams(kernelOp);
  const specs: any[] = [];
  const posArgs = binding.ir.args || [];
  for (let i = 0; i < posArgs.length; i++) {
    const cls = _classifyArg(posArgs[i], '#' + (i + 1), bindings, fixedValues,
      refuse, kernelOp, params[i] || null, known);
    if (cls == null) return false;
    specs.push({ key: null, ir: posArgs[i], cls });
  }
  const kwargs = binding.ir.kwargs || {};
  for (const k of Object.keys(kwargs)) {
    const cls = _classifyArg(kwargs[k], '`' + k + '`', bindings, fixedValues,
      refuse, kernelOp, k, known);
    if (cls == null) return false;
    specs.push({ key: k, ir: kwargs[k], cls });
  }

  // §06: "each collection argument has size N or is singular (size one,
  // expanded by repetition)".
  for (const s of specs) {
    if (s.cls.kind !== 'axis') continue;
    if (s.cls.length !== N && s.cls.length !== 1) {
      refuse('ksuperpose: family argument '
        + `${s.key == null ? 'position' : '`' + s.key + '`'} has size `
        + `${s.cls.length}, but the mixture has N = ${N} components; each `
        + 'collection argument must have size N or be singular (spec §06)');
      return false;
    }
  }

  const weightedNames: string[] = [];
  for (let i = 1; i <= N; i++) {
    const compIR: any = { kind: 'call', op: kernelOp };
    const cArgs: any[] = [];
    const cKwargs: Record<string, any> = {};
    let hasKw = false;
    for (const s of specs) {
      // §06: a singular collection is "size one, expanded by repetition", so
      // it reads row 1 for every component.
      const value = s.cls.kind === 'const'
        ? s.ir
        : _getRow(s.ir, s.cls.length === 1 ? 1 : i, s.cls.lead);
      if (s.key == null) cArgs.push(value);
      else { cKwargs[s.key] = value; hasKw = true; }
    }
    if (cArgs.length > 0) compIR.args = cArgs;
    if (hasKw) compIR.kwargs = cKwargs;

    const compName = `__ksrt_${name}_c${i}`;
    const wtName = `__ksrt_${name}_w${i}`;
    const weightIR = _getRow(weightsIR, i);
    const compDeps = specs.filter((s: any) => _isSelfRef(s.ir)).map((s: any) => s.ir.name);
    bindings.set(compName, _synthBinding(compName, compIR, compDeps, loc));
    bindings.set(wtName, _synthBinding(
      wtName, { kind: 'call', op: 'weighted', args: [weightIR, { kind: 'ref', ns: 'self', name: compName }] },
      [compName, wName], loc));
    derivations[compName] = { kind: 'sample', distIR: compIR };
    derivations[wtName] = { kind: 'weighted', from: compName, weightIR };
    weightedNames.push(wtName);
  }
  derivations[name] = { kind: 'superpose', fromNames: weightedNames };
  return true;
}

// Expand every applied `ksuperpose` lift the static rewrite left behind.
//
// Returns `{ changed, refused }`. `refused` names the mixture bindings that
// got a located refusal; the caller suppresses the generic fixed-phase
// dead-end message for them and for everything above them, which would
// otherwise follow a precise diagnosis with "this is an engine gap".
function expandRuntimeKsuperpose(
  bindings: any, derivations: any, fixedValues: any, diagnostics: any[],
): { changed: boolean; refused: Set<string> } {
  // Snapshot the names: the loop adds bindings, and a synthesised one is never
  // itself an applied lift.
  const names: string[] = Array.from(bindings.keys()).map(String);
  let changed = false;
  const refused = new Set<string>();
  for (const name of names) {
    if (derivations[name]) continue;
    const binding = bindings.get(name);
    if (!binding || !binding.ir) continue;
    const lift = _appliedLift(binding.ir);
    if (!lift) continue;
    const before = diagnostics.length;
    if (_expandOne(name, binding, lift, bindings, derivations,
      fixedValues, diagnostics)) {
      changed = true;
    } else if (diagnostics.length > before) {
      refused.add(name);
    }
  }
  return { changed, refused };
}

module.exports = { expandRuntimeKsuperpose };
