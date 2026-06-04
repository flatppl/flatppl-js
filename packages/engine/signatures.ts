'use strict';

// signatures.js — callable introspection (for the profile-plot UI).
// =====================================================================
//
// signatureOf(name, bindings) returns the canonical input/output
// signature of any reified callable — function, kernel, or
// likelihood — by combining typeinfer's structural types
// (binding.inferredType) with the lowering's paramSources backrefs
// (binding.ir.paramSources, populated in lower.js _lowerReification).
//
// distributeAxes(signature) flattens the input types into the list
// of atomic scalar leaves the profile-plot UI uses for its axis
// dropdown — distributing cartprod (records / tuples) and cartpow
// (static-shape arrays) recursively.
//
// Both return plain JS objects; no mutation, no global state. The
// viewer treats them as a stable shape contract.
//
// Likelihood handling:
//   L = likelihoodof(K, obs) is treated as a function with K's input
//   signature and a real-valued output (the log-likelihood at obs).
//   We dereference to K to get the structural inputs but rewrite the
//   output type and tag kind='likelihood' so the profile evaluator
//   uses logDensityN rather than evaluateExpr.
//
// A leaf w.r.t. the orchestrator split: depends only on ir-shared
// (isSelfRef, resolveIRToValue).

import type { IRNode } from './engine-types';

const { isSelfRef, resolveIRToValue, SAMPLEABLE_DISTRIBUTIONS } = require('./ir-shared.ts');
const { mapIR } = require('./ir-walk.ts');

function signatureOf(name: string, bindings: any): any {
  if (!bindings) return null;
  const b = bindings.get(name);
  if (!b) return null;

  if (b.type === 'likelihood') return signatureOfLikelihood(b, bindings);

  if (b.type !== 'functionof' && b.type !== 'fn' && b.type !== 'kernelof') {
    return null;
  }
  const ir = b.ir;
  if (!ir || ir.op !== 'functionof') return null;

  // kind comes from inferredType when typeinfer resolved it
  // (function: value body, kernel: measure body). For bindings whose
  // inferredType is deferred — disintegrate-derived bindings, synthesised
  // analyses, etc. — fall back to inspecting the body: a self-ref body
  // pointing at a measure-typed binding makes this a kernel; same for
  // a body that's itself a measure-op call. Otherwise (or when the
  // body resolution fails) default to function.
  const t = b.inferredType;
  let kind: string;
  if (t && t.kind === 'function')      kind = 'function';
  else if (t && t.kind === 'kernel')   kind = 'kernel';
  else if (b.type === 'kernelof')      kind = 'kernel';
  else                                  kind = bodyImpliesKernel(ir.body, bindings) ? 'kernel' : 'function';

  const params      = ir.params      || [];
  const paramKwargs = ir.paramKwargs || [];
  const sources     = ir.paramSources|| [];
  const inputTypes  = (t && t.inputs)|| [];

  const inputs: any[] = [];
  for (let i = 0; i < params.length; i++) {
    // typeinfer's inferReification stores boundary types via
    // expr.kwargs lookup, but the lowered functionof IR doesn't carry
    // the boundary expressions, so all reification params come back
    // typed 'any'. Recover the actual boundary type from the
    // paramSources backref: a binding source's type is its
    // inferredType (e.g. theta1 → real); a placeholder source's
    // type is the corresponding `<name> = elementof(<set>)`
    // binding's inferredType.
    const inferredHere = inputTypes[i] && inputTypes[i].type;
    const fromSource   = resolveSourceType(sources[i] || null, bindings);
    inputs.push({
      paramName: params[i],
      kwargName: paramKwargs[i],
      type:      fromSource || inferredHere || null,
      source:    sources[i] || null,
    });
  }
  // Per spec §04 (sec:functionof), no-kwargs functionof / kernelof
  // auto-promotes elementof ancestors as inputs. That rewrite now
  // happens once in canonicalizeImplicitBoundaries (called from
  // liftInlineSubexpressions before lowering), so ir.params here is
  // already complete — no second auto-promote pass needed.
  return { kind, inputs, output: { type: t && t.result }, body: ir.body || null };
}

// Decide if a reified body returns a measure (→ kernel) by walking
// the body IR. Used as a fallback when typeinfer left the binding
// type as 'deferred'. Conservative: only returns true when we can
// see a measure-shaped op or a self-ref to a measure-typed binding.
const KNOWN_MEASURE_OPS = new Set([
  'joint', 'record', 'iid', 'weighted', 'logweighted', 'normalize',
  'superpose', 'lawof', 'pushfwd', 'truncate', 'mixture',
  // leaf distributions are measures too — we don't enumerate them
  // here; they'd need the full SAMPLEABLE_DISTRIBUTIONS set, but a
  // reified callable typically has its leaf wrapped in a record /
  // joint / lawof anyway.
]);

function bodyImpliesKernel(body: IRNode | null | undefined, bindings: any): boolean {
  if (!body) return false;
  if (body.kind === 'call' && body.op != null) {
    if (KNOWN_MEASURE_OPS.has(body.op))                  return true;
    if (SAMPLEABLE_DISTRIBUTIONS.has(body.op))           return true;
  }
  if (body.kind === 'ref' && body.ns === 'self' && bindings) {
    const target = bindings.get(body.name);
    if (target && target.inferredType
        && target.inferredType.kind === 'measure') {
      return true;
    }
    // Some derived measure bindings have inferredType='deferred' too;
    // recurse into THEIR body / IR if it's a call we recognise.
    if (target && target.ir) return bodyImpliesKernel(target.ir, bindings);
  }
  return false;
}

// Resolve a paramSources entry to the value type it references.
// Returns null if the source can't be resolved (e.g. placeholder
// without a corresponding elementof binding) — caller falls back
// to typeinfer's per-call type (typically 'any').
function resolveSourceType(source: any, bindings: any) {
  if (!source || !bindings) return null;
  const target = bindings.get(source.name);
  if (!target || !target.inferredType) return null;
  // Both binding-source and placeholder-source ultimately want the
  // value type of the bound expression — for a placeholder the
  // typeinfer pass walks elementof's set and tags the binding with
  // the corresponding value type, so the lookup is uniform.
  return target.inferredType;
}

// Likelihood inspection: walk likelihoodof(K, obs) → resolve K to a
// kernel binding → reuse its signature with the output overridden to
// REAL (the log-likelihood at obs) and the obs IR stored alongside.
// The viewer calls resolveIRToValue at materialise time, after pre-
// eval has populated fixedValues, so an observation that's a draw
// / rand result / any other dynamically-computed value works the
// same as an inline literal.
function signatureOfLikelihood(b: any, bindings: any): any {
  const ir = b.ir;
  if (!ir || ir.op !== 'likelihoodof' || !Array.isArray(ir.args) || ir.args.length !== 2) {
    return null;
  }
  const Kref  = ir.args[0];
  const obsIR = ir.args[1];
  if (!isSelfRef(Kref) || !bindings.has(Kref.name)) return null;
  const inner: any = signatureOf(Kref.name, bindings);
  if (!inner) return null;
  return {
    kind: 'likelihood',
    inputs: inner.inputs,
    output: { type: { kind: 'scalar', prim: 'real' } },
    obsIR,
    kernelName: Kref.name,
    // Likelihood evaluation reuses the kernel's body (peeled of any
    // lawof wrapper) at logdensity-mode evaluation time. Carrying the
    // body here lets the viewer dispatch without re-resolving the
    // kernel ref.
    body: inner.body,
  };
}

// Distribute a signature's inputs over their structural shape: one
// axis per scalar leaf. Records emit one axis per field
// (recursively, dot-separated), tuples per element (1-indexed),
// static-shape arrays per slot. Dynamic-shape arrays surface as a
// single non-scalar axis (the UI either rejects them or the user
// has to constrain shape elsewhere).
//
// Each axis carries:
//   - key:        unique stable id within the signature
//                 (e.g. "theta", "theta.mu", "obs[1]", "x[2,3].phi")
//   - label:      same string, used as display
//   - kwargName:  the input this axis belongs to (for fixed-value
//                 substitution at evaluation time)
//   - path:       array of segments — strings for record/tuple-named
//                 fields, numbers (1-indexed) for tuple/array slots
//   - leafType:   the scalar / dynamic-array type at the leaf
//   - source:     paramSources entry for the input (the *whole*
//                 input — UI may need to drill in for record sources)
function distributeAxes(signature: any) {
  if (!signature || !Array.isArray(signature.inputs)) return [];
  const out: any[] = [];
  for (const input of signature.inputs) {
    walkType(input.type || null, [], (path: any, leafType: any) => {
      const label = formatAxisLabel(input.kwargName, path);
      out.push({
        key: label,
        label,
        kwargName: input.kwargName,
        path,
        leafType,
        source: input.source,
      });
    });
  }
  return out;
}

function walkType(type: any, path: any[], emit: (p: any[], t: any) => void) {
  if (!type) return;
  if (type.kind === 'scalar') return emit(path, type);
  // Unrestricted placeholder boundaries (`fn(_)` / `_par_ =
  // elementof(anything)`) infer to 'any'. From the UI's standpoint
  // these are unknown scalars — emit a single axis so the user can
  // still profile-sweep along the input. Defaults fall back to the
  // generic-real handling.
  if (type.kind === 'any' || type.kind === 'deferred' || type.kind === 'var') {
    return emit(path, type);
  }
  if (type.kind === 'record' && type.fields) {
    for (const f in type.fields) walkType(type.fields[f], path.concat([f]), emit);
    return;
  }
  if (type.kind === 'tuple' && Array.isArray(type.elems)) {
    for (let i = 0; i < type.elems.length; i++) {
      walkType(type.elems[i], path.concat([i + 1]), emit);
    }
    return;
  }
  if (type.kind === 'array' && Array.isArray(type.shape)) {
    if (type.shape.some((d: any) => d === '%dynamic')) {
      // Dynamic shape: surface a single axis at this level, marked
      // by leafType so the UI can refuse to plot or ask the user
      // for a concrete shape via a preset.
      return emit(path, type);
    }
    walkArraySlots(type, path, emit);
    return;
  }
  // function / kernel / measure / failed / set: not axis-emitting
  // (the UI shouldn't be asked to sweep them).
}

function walkArraySlots(arrayType: any, path: any[], emit: (p: any[], t: any) => void) {
  const dims = arrayType.shape;
  let total = 1;
  for (const d of dims) total *= d;
  for (let s = 0; s < total; s++) {
    let r = s;
    const idx = new Array(dims.length);
    for (let d = dims.length - 1; d >= 0; d--) {
      idx[d] = (r % dims[d]) + 1;  // FlatPPL is 1-indexed
      r = Math.floor(r / dims[d]);
    }
    walkType(arrayType.elem, path.concat([{ idx }]), emit);
  }
}

/**
 * Convert a concrete env value (the kind the viewer's kernel/profile
 * input-supply path builds for a callable input) into an IR node:
 *
 *   - scalar (number/boolean)        → `lit`
 *   - array / typed array            → `vector(elem, …)`
 *   - record (plain object)          → `record(%field name value, …)`
 *   - pre-built IR node (`.kind`)    → passed through unchanged
 *
 * The pre-built-IR case is how a profile sweep threads a varying slot
 * through an otherwise-fixed structured input: the per-slot / per-field
 * value is a `(ref %local sweepName)` IR so `profileN` sweeps just that
 * leaf while the rest of the array/record stays fixed.
 *
 * Records compose recursively (a record input nested inside another, or
 * a record field that is itself an array), so the same routine handles
 * arbitrarily-nested structured inputs — e.g. `pars = elementof(cartprod(
 * a = reals, b = reals, mu = reals))` used as a single kernel input
 * binds `env.pars = {a, b, mu}` here and lowers to a `record(...)` IR,
 * so the kernel body's `pars.mu` (a `get_field`) resolves at sample time.
 */
function envValueToIR(v: any, loc: any): any {
  // Pre-built IR node (sweep ref, or an already-lowered sub-expression).
  if (v != null && typeof v === 'object' && (v as any).kind != null) return v;
  // Array / typed-array → vector(...).
  if (Array.isArray(v) || (v != null && typeof v === 'object'
      && typeof (v as any).length === 'number'
      && (typeof (v as any).BYTES_PER_ELEMENT === 'number'
          || (v as any)[0] !== undefined))) {
    const len = (v as any).length;
    const args: any[] = [];
    for (let i = 0; i < len; i++) args.push(envValueToIR((v as any)[i], loc));
    return { kind: 'call', op: 'vector', args, loc };
  }
  // Record (plain object, not array-like, not an IR node) → record(...)
  // with FIELD_FORM entries (matching lower's `record` shape). Field
  // order follows the object's own key order — the env-assembly path
  // builds it in the input's declared field order.
  if (v != null && typeof v === 'object') {
    const fields: any[] = [];
    for (const k in v) {
      if (!Object.prototype.hasOwnProperty.call(v, k)) continue;
      fields.push({ name: k, value: envValueToIR((v as any)[k], loc) });
    }
    return { kind: 'call', op: 'record', fields, loc };
  }
  // Scalar leaf.
  return { kind: 'lit', value: v, numType: 'real', loc };
}

/**
 * Walk an IR replacing every (ref %local <name>) with the IR form of
 * env[name] when env contains a value for that name. Used by the
 * kernel-plot path to substitute preset parameter values into a
 * kernel body before sampling it as a concrete measure.
 *
 * Leaves self-refs intact (those go through the normal materialiser
 * path via expandMeasureRefsInIR + the binding's derivation) and
 * leaves %local refs not in env intact too (defensive — caller
 * should populate env for every param before calling this).
 */
function substituteLocals(ir: any, env: any): any {
  return mapIR(ir, function(node: any): any {
    if (node && node.kind === 'ref' && node.ns === '%local'
        && Object.prototype.hasOwnProperty.call(env, node.name)) {
      return envValueToIR(env[node.name], node.loc);
    }
    return node;
  });
}

function formatAxisLabel(kwargName: string, path: any[]) {
  let s = kwargName || '';
  for (const seg of path) {
    if (typeof seg === 'string')          s += '.' + seg;
    else if (typeof seg === 'number')     s += '[' + seg + ']';
    else if (seg && Array.isArray(seg.idx)) s += '[' + seg.idx.join(',') + ']';
  }
  return s;
}

/**
 * Enumerate the scalar leaves of an output type. Mirrors
 * distributeAxes but for the OUTPUT side of a callable: each
 * leaf is a scalar component the viewer's profile plot can
 * select as its evaluated y-value.
 *
 * For a scalar output type returns a single entry with empty path;
 * the caller reads it as "there's only one output, no Output:
 * dropdown needed".
 *
 *   record { a: real, b: integer }   → [{path:['a'], label:'a',  leafType:real},
 *                                       {path:['b'], label:'b',  leafType:integer}]
 *   tuple (real, real)               → [{path:[1],  label:'[1]', leafType:real},
 *                                       {path:[2],  label:'[2]', leafType:real}]
 *   array<real, [3]>                  → [{path:[{idx:[1]}], label:'[1]', …}, …]
 *   real                              → [{path:[], label:'', leafType:real}]
 *
 * Path segments use the same shape as distributeAxes' input paths
 * — string field name, integer (1-indexed) tuple/array slot, or
 * `{idx:[…]}` for multi-dim array indices. extractOutputIR
 * consumes the same path shape to pull the matching sub-IR out
 * of the body.
 */
function enumerateOutputLeaves(outputType: any) {
  if (!outputType) return [];
  const out: any[] = [];
  walkType(outputType, [], (path: any, leafType: any) => {
    out.push({
      key: formatAxisLabel('', path) || '<scalar>',
      label: formatAxisLabel('', path),
      path: path.slice(),
      leafType,
    });
  });
  return out;
}

/**
 * Extract the sub-IR of a body expression at the given output
 * path. Reverses the path navigation enumerateOutputLeaves built:
 * record path segments (field names) descend through `body.fields`,
 * tuple / array integer indices descend through `body.args`,
 * array `{idx:[…]}` segments traverse `body.args` flattened in
 * row-major order.
 *
 *   body = record(a = X, b = Y)        → path ['a']  →  X
 *   body = (call tuple X Y)             → path [1]    →  X
 *   body = (call vector X Y Z)          → path [2]    →  Y  (1-indexed)
 *
 * Returns the original body when path is empty (scalar output).
 * Returns null if the path can't be resolved (e.g. body shape
 * doesn't match the type's shape — should never happen on a
 * type-checked module, but defensive).
 */
function extractOutputIR(bodyIR: any, path: any[]) {
  if (!path || path.length === 0) return bodyIR || null;
  let cur = bodyIR;
  for (const seg of path) {
    if (!cur) return null;
    if (typeof seg === 'string') {
      // Record field. body.fields = [{name, value}, …]
      if (!Array.isArray(cur.fields)) return null;
      const f = cur.fields.find((x: any) => x && x.name === seg);
      if (!f) return null;
      cur = f.value;
      continue;
    }
    if (typeof seg === 'number') {
      // Tuple / 1-D array slot, 1-indexed.
      if (!Array.isArray(cur.args)) return null;
      cur = cur.args[seg - 1];
      continue;
    }
    if (seg && Array.isArray(seg.idx)) {
      // Multi-dim array. Flatten the index in row-major order
      // matching walkArraySlots' (FlatPPL 1-indexed) traversal.
      // Without the array's static shape here we can't fully
      // generalise, but the common case is a single positional
      // index into a 1-D vector body.
      if (!Array.isArray(cur.args)) return null;
      if (seg.idx.length === 1) {
        cur = cur.args[seg.idx[0] - 1];
        continue;
      }
      // For multi-dim arrays we'd need the shape from the type to
      // do the same row-major flatten as walkArraySlots; defer
      // until a concrete model exercises this.
      return null;
    }
    return null;
  }
  return cur || null;
}

module.exports = {
  signatureOf,
  KNOWN_MEASURE_OPS,
  bodyImpliesKernel,
  resolveSourceType,
  signatureOfLikelihood,
  distributeAxes,
  walkType,
  walkArraySlots,
  substituteLocals,
  formatAxisLabel,
  enumerateOutputLeaves,
  extractOutputIR,
};
