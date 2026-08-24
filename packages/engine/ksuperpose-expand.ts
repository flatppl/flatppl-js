'use strict';

// ksuperpose-expand.ts — spec §06 `ksuperpose` weighted-superposition lift.
// =====================================================================
//
// §06 `ksuperpose`: "lifts a kernel to a weighted superposition: the
// result is itself a kernel, and applying it to a parameter family
// yields the mixture ν = Σᵢ wᵢ·κ(θᵢ), with θᵢ read from row i of the
// family."
//
// The applied form is therefore EXACTLY the variadic spelling §06 already
// gives for a mixture:
//
//   ksuperpose(K, w)(p = vec)  ≡  superpose(weighted(w[1], K(p = vec[1])),
//                                          …,
//                                          weighted(w[N], K(p = vec[N])))
//
// and this pass rewrites the AST to the right-hand side. Every §06
// consequence then falls out of machinery that is already conformant and
// already reviewed, rather than out of a second implementation of it:
//
//   - density: §06's `logsumexp_i(log wᵢ + logdensityof(κ(θᵢ), x))` is
//     what `walkSuperpose` → `walkSelect` computes over `weighted`
//     components (density.ts). A zero weight enters as `log 0 = -Infinity`
//     and drops out exactly; all-zero weights give -Infinity, which is
//     §06's "log-density −∞".
//   - mass: `Σᵢ wᵢ·totalmass(κ(θᵢ))` is `additiveMass` over
//     `weighted` (typeinfer), and `normalize` resolves Z per-θ through
//     `normalize-mass.ts`'s existing superpose-of-weighted arm — which is
//     `Σᵢ wᵢ` for a Markov component, §06's Markov specialization.
//   - SAMPLING: the component selection is drawn per output index by
//     `matSuperpose`, so under `iid(ksuperpose(…), k)` every coordinate
//     selects its own component. That is the iid(superpose) branch-
//     freshness invariant (flatppl-engine-concepts §22.4 "The repeat
//     axis"), and inheriting it is the reason this pass expands instead
//     of adding a parallel handler that would have to re-derive it.
//     `REPLICATED_ARG_OPS` (typeinfer) already lists `superpose` for the
//     same reason, so shared ancestors — parameterized weights, family
//     entries from outer draws — stay shared while the selection freshens.
//
// The cost is that N must be resolvable when this pass runs. §06 says N
// "need not be statically known", so a weight vector of genuinely dynamic
// length is a located refusal here rather than a lowering. Nothing else in
// §06 is restricted: the weights themselves may be latent (the common
// mixture spelling), because the rewrite indexes the weight EXPRESSION.
//
// Runs after type inference (it reads `inferredType` to classify each
// family argument as size-N, singular, or held constant) and before
// `buildDerivations`, whose lift hoists each synthesised component into
// its own binding.

const AST = require('./ast.ts');

// Structural deep clone of an AST subtree. Each of the N components gets
// its own copy of the kernel head and of every held-constant argument, so
// no node object is shared between two components.
function _clone(node: any): any {
  if (node == null || typeof node !== 'object') return node;
  if (Array.isArray(node)) return node.map(_clone);
  const out: any = {};
  for (const k of Object.keys(node)) out[k] = _clone(node[k]);
  return out;
}

// The RHS of a top-level binding, or null.
function _rhsOf(name: string, body: any[]): any {
  for (const stmt of body) {
    if (stmt.type !== 'AssignStatement' || !stmt.names) continue;
    for (const n of stmt.names) if (n.name === name) return stmt.value;
  }
  return null;
}

function _isKsuperposeCall(node: any): boolean {
  return !!(node && node.type === 'CallExpr' && node.callee
    && node.callee.type === 'Identifier'
    && node.callee.name === 'ksuperpose');
}

// The `ksuperpose(K, w)` lift node a call's callee denotes, or null. Covers
// the inline spelling `ksuperpose(K, w)(…)`, the named one
// (`lift = ksuperpose(K, w)` then `lift(…)`) and an alias CHAIN of any depth
// (`l1 = ksuperpose(…)`, `l2 = l1`, then `l2(…)`) — §04 "Aliasing is just
// assignment" makes all of those the same program.
//
// The chain is walked here rather than inherited from `alias-resolution.ts`:
// that pass canonicalises the LOWERED module, and this rewrite runs on the
// AST. `seen` is the cycle guard — a self- or mutually-referential binding is
// a different error, not this pass's to report, so it stops rather than
// looping.
function _liftNodeOf(callee: any, body: any[]): any {
  if (!callee) return null;
  if (_isKsuperposeCall(callee)) return callee;
  let node = callee;
  const seen = new Set<string>();
  while (node && node.type === 'Identifier' && !seen.has(node.name)) {
    seen.add(node.name);
    node = _rhsOf(node.name, body);
    if (_isKsuperposeCall(node)) return node;
  }
  return null;
}

// Outer length of an ARRAY LITERAL, or null. Read straight off the AST so a
// literal family or weight vector needs no inferred type.
function _literalLength(node: any): number | null {
  if (node && node.type === 'ArrayLiteral' && Array.isArray(node.elements)) {
    return node.elements.length;
  }
  return null;
}

function _typeOf(node: any, bindings: any): any {
  if (!node || node.type !== 'Identifier' || !bindings || !bindings.get) return null;
  const b = bindings.get(node.name);
  return (b && b.inferredType) || null;
}

// §06 classifies each family argument as a collection (size N or singular,
// "expanded by repetition") or a non-collection ("held constant across the
// components"). Return one of:
//   { kind: 'const' }                 — held constant
//   { kind: 'axis', length: n|null }  — one axis; null length = unknown
//   { kind: 'multiaxis' }             — §06 static error
//   { kind: 'table' }                 — one axis, but not lowered here
//   null                              — cannot classify
//
// Axis counting follows the TYPE, not `shape.length`: FlatPPL spells a
// nested array as an array whose element is an array, so a rank-1 type with
// an array element is two axes and must not pass as one.
function _classify(node: any, bindings: any): any {
  const lit = _literalLength(node);
  if (lit != null) {
    const nested = (node.elements || []).some(
      (e: any) => e && (e.type === 'ArrayLiteral' || e.type === 'TupleLiteral'));
    return nested ? { kind: 'multiaxis' } : { kind: 'axis', length: lit };
  }
  if (node && (node.type === 'NumberLiteral' || node.type === 'BoolLiteral'
      || node.type === 'StringLiteral')) {
    return { kind: 'const' };
  }
  const t = _typeOf(node, bindings);
  if (!t) return null;
  if (t.kind === 'table') return { kind: 'table' };
  if (t.kind === 'tvector') return { kind: 'axis', length: _dim(t.length) };
  if (t.kind === 'array') {
    const rank = Array.isArray(t.shape) ? t.shape.length : t.rank;
    if (rank !== 1) return { kind: 'multiaxis' };
    if (t.elem && (t.elem.kind === 'array' || t.elem.kind === 'table'
        || t.elem.kind === 'tvector')) {
      return { kind: 'multiaxis' };
    }
    return { kind: 'axis', length: _dim(t.shape && t.shape[0]) };
  }
  if (t.kind === 'measure' || t.kind === 'kernel' || t.kind === 'function'
      || t.kind === 'module' || t.kind === 'failed' || t.kind === 'deferred') {
    return null;
  }
  // Scalars, sets, records and everything else §06 does not read row-wise.
  return { kind: 'const' };
}

function _dim(d: any): number | null {
  return (typeof d === 'number' && Number.isInteger(d) && d > 0) ? d : null;
}

function _err(diagnostics: any[], message: string, loc: any) {
  diagnostics.push({ severity: 'error', message, loc });
}

// One family argument, specialised to component `i` (1-based).
function _componentArg(spec: any, i: number, sloc: any): any {
  const node = _clone(spec.node);
  let value = node;
  if (spec.cls.kind !== 'const') {
    // §06: a singular collection is "size one, expanded by repetition", so
    // it reads row 1 for every component.
    const row = spec.cls.length === 1 ? 1 : i;
    value = AST.IndexExpr(
      node, [AST.NumberLiteral(row, String(row), sloc)], sloc, 'get');
  }
  return spec.name == null ? value : AST.KeywordArg(spec.name, value, sloc);
}

// Rewrite one `ksuperpose(K, w)(family…)` application. Returns the
// `superpose(...)` replacement, or `call` unchanged when §06 makes the
// application a static error or when the rewrite is refused.
function _expandApplication(call: any, lift: any, bindings: any, diagnostics: any[]): any {
  const sloc = { ...call.loc, synthetic: true, source: 'ksuperpose-expand' };
  const liftArgs = lift.args || [];
  if (liftArgs.length !== 2 || liftArgs.some((a: any) => a && a.type === 'KeywordArg')) {
    _err(diagnostics, 'ksuperpose expects 2 positional arguments '
      + `(kernel, weights), got ${liftArgs.length} (spec §06)`, lift.loc);
    return call;
  }
  const kernelNode = liftArgs[0];
  const weightsNode = liftArgs[1];

  // §06: "`weights` is a distinguished input, not a member of the family,
  // and never expands." A non-vector weight argument has no components.
  const wCls = _classify(weightsNode, bindings);
  if (wCls && (wCls.kind === 'multiaxis' || wCls.kind === 'const'
      || wCls.kind === 'table')) {
    _err(diagnostics, 'ksuperpose: `weights` must be a one-axis vector of '
      + 'non-negative weights (spec §06); its length is the component '
      + 'count N', weightsNode.loc || lift.loc);
    return call;
  }

  // Family arguments, in the application's own order and naming. Nothing
  // here needs the component's parameter names: each argument is handed to
  // the component call in the position or under the keyword it already has
  // (§05 lets a distribution take its parameters positionally).
  const specs: any[] = [];
  for (const a of call.args || []) {
    const isKw = a && a.type === 'KeywordArg';
    const node = isKw ? a.value : a;
    const cls = _classify(node, bindings);
    if (cls == null) {
      _err(diagnostics, 'ksuperpose: cannot tell whether family argument '
        + `${isKw ? '`' + a.name + '`' : '#' + (specs.length + 1)} is a `
        + 'size-N collection or a held-constant scalar, so the component '
        + 'rows cannot be read (spec §06). Bind it to a name with a '
        + 'statically-known vector length.', (node && node.loc) || call.loc);
      return call;
    }
    if (cls.kind === 'multiaxis') {
      _err(diagnostics, 'ksuperpose: the parameter family is restricted to a '
        + `single axis, and family argument ${isKw ? '`' + a.name + '`' : '#' + (specs.length + 1)} `
        + 'has more than one (spec §06)', (node && node.loc) || call.loc);
      return call;
    }
    if (cls.kind === 'table') {
      _err(diagnostics, 'ksuperpose: §06 allows a TABLE parameter family (one '
        + 'axis, its rows), but this engine does not lower one — the '
        + 'per-column family extraction is not built. Pass the columns as '
        + 'keyword vectors instead.', (node && node.loc) || call.loc);
      return call;
    }
    specs.push({ name: isKw ? a.name : null, node, cls });
  }

  // N is the length of `weights`. A family collection pins it too (§06
  // gives each one "size N or … one"), which covers a latent weight vector
  // whose own length inference did not reach.
  let N: number | null = (wCls && wCls.kind === 'axis') ? wCls.length : null;
  if (N == null) {
    for (const s of specs) {
      if (s.cls.kind === 'axis' && s.cls.length != null && s.cls.length > 1) {
        N = s.cls.length;
        break;
      }
    }
  }
  if (N == null) {
    _err(diagnostics, 'ksuperpose: the component count N is not statically '
      + 'known. §06 allows a runtime N, but this engine expands the mixture '
      + 'into its N components, so the weight vector (or a size-N family '
      + 'argument) needs a statically-known length.',
      (weightsNode && weightsNode.loc) || lift.loc);
    return call;
  }

  // §06: "each collection argument has size N or is singular (size one,
  // expanded by repetition)".
  for (const s of specs) {
    if (s.cls.kind !== 'axis') continue;
    if (s.cls.length != null && s.cls.length !== N && s.cls.length !== 1) {
      _err(diagnostics, 'ksuperpose: family argument '
        + `${s.name == null ? 'position' : '`' + s.name + '`'} has size `
        + `${s.cls.length}, but the mixture has N = ${N} components; each `
        + 'collection argument must have size N or be singular (spec §06)',
        (s.node && s.node.loc) || call.loc);
      return call;
    }
  }

  const components: any[] = [];
  for (let i = 1; i <= N; i++) {
    const args = specs.map((s: any) => _componentArg(s, i, sloc));
    const component = AST.CallExpr(_clone(kernelNode), args, sloc);
    const wi = AST.IndexExpr(
      _clone(weightsNode), [AST.NumberLiteral(i, String(i), sloc)], sloc, 'get');
    components.push(AST.CallExpr(
      AST.Identifier('weighted', sloc), [wi, component], sloc));
  }
  return AST.CallExpr(AST.Identifier('superpose', sloc), components, sloc);
}

// Recursively rewrite every `ksuperpose` application in `node`. Returns
// `node` unchanged (same reference) when nothing was rewritten, so a model
// without `ksuperpose` pays no clone cost. The structural recursion is
// required rather than speculative: `x ~ M` desugars to `x = draw(M)` at
// parse time, so an application written `x ~ normalize(ksuperpose(…)(…))`
// arrives nested two levels down.
function _rewrite(node: any, bindings: any, diagnostics: any[], body: any[]): any {
  if (node == null || typeof node !== 'object') return node;
  if (Array.isArray(node)) {
    let out: any = node;
    for (let i = 0; i < node.length; i++) {
      const r = _rewrite(node[i], bindings, diagnostics, body);
      if (r !== node[i]) {
        if (out === node) out = node.slice();
        out[i] = r;
      }
    }
    return out;
  }
  if (node.type === 'CallExpr') {
    const lift = _liftNodeOf(node.callee, body);
    if (lift) {
      // Rewrite the family arguments first: they may themselves contain a
      // nested application.
      const inner = _rewrite(node.args, bindings, diagnostics, body);
      const call = inner === node.args ? node : { ...node, args: inner };
      return _expandApplication(call, lift, bindings, diagnostics);
    }
  }
  let out: any = node;
  for (const k of Object.keys(node)) {
    if (k === 'loc' || k === 'type') continue;
    const r = _rewrite(node[k], bindings, diagnostics, body);
    if (r !== node[k]) {
      if (out === node) out = { ...node };
      out[k] = r;
    }
  }
  return out;
}

// Expand every `ksuperpose` application in `ast`, in place. `bindings` is
// the analyzer binding map, carrying `inferredType`, so this must run after
// type inference. Returns true when something was rewritten, which is the
// caller's signal to lower and infer the module again.
//
// Both AST slots `pir.lowerToModule` reads are rewritten: `effectiveValue`
// (set for destructured / disintegrated bindings) takes precedence there, so
// rewriting only `node.value` would silently skip those.
//
// `done` MEMOISES the rewrite by input node rather than marking nodes as
// visited. Two slots can share one AST object — `attachDelegate` assigns
// `binding.effectiveValue = target.node.value` — and a visited-set would then
// rewrite the object through the first slot and skip the second, leaving the
// UN-EXPANDED node in the slot `pir.lowerToModule` actually prefers. Memoising
// assigns the same rewritten node to every slot that shared the original, and
// keeps the pass idempotent besides.
function expandKsuperposeApplications(ast: any, bindings: any, diagnostics: any[]): boolean {
  if (!ast || !Array.isArray(ast.body)) return false;
  let changed = false;
  const done = new Map<any, any>();
  const rewriteSlot = (holder: any, key: string) => {
    const node = holder && holder[key];
    if (!node || typeof node !== 'object') return;
    let out;
    if (done.has(node)) {
      out = done.get(node);
    } else {
      out = _rewrite(node, bindings, diagnostics, ast.body);
      done.set(node, out);
    }
    if (out !== node) { holder[key] = out; changed = true; }
  };
  for (const stmt of ast.body) {
    if (stmt.type === 'AssignStatement') rewriteSlot(stmt, 'value');
  }
  if (bindings && bindings.values) {
    for (const b of bindings.values()) {
      if (!b) continue;
      if (b.effectiveValue) rewriteSlot(b, 'effectiveValue');
      if (b.node) rewriteSlot(b.node, 'value');
    }
  }
  return changed;
}

module.exports = { expandKsuperposeApplications };
