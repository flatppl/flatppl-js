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
// length cannot be rewritten HERE. That case is handed to the RUNTIME ARM
// (`ksuperpose-runtime.ts`), which runs at derivation-build time where a
// deterministic vector's VALUE — and so its length — is available, and which
// synthesises the same component graph this pass would have written. This
// pass therefore leaves such an application untouched and SILENT: the
// refusal, if one is still owed, belongs to whoever knows the value.
//
// Nothing else in §06 is restricted: the weights themselves may be latent
// (the common mixture spelling), because the rewrite indexes the weight
// EXPRESSION. Latent weights keep this path — a latent vector has a static
// length but no value, so the runtime arm could not read N from it.
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
//   { kind: 'const' }                            — held constant
//   { kind: 'axis', length, axes, lead }         — a collection
//   { kind: 'table', length, columns }           — one axis, its rows
//   null                                         — cannot classify
//
// `axes` is the argument's TOTAL axis count, which §06's family rule measures
// against the rank of the parameter it feeds ([`familyAxisComplaint`]). `lead`
// is how many of those axes live in the OUTER array shape, which is how many
// selectors read one family row: FlatPPL spells a nested array as an array
// whose element is an array, so an N-vector of d x d matrices takes one
// selector while a flat N x d x d array takes three.
function _classify(node: any, bindings: any): any {
  const lit = _literalLength(node);
  if (lit != null) {
    // A literal's nesting is its own axes; the outer literal is one axis.
    const inner = (node.elements || [])[0];
    const nested = inner && (inner.type === 'ArrayLiteral' || inner.type === 'TupleLiteral');
    const innerCls = nested ? _classify(inner, bindings) : null;
    const innerAxes = innerCls && innerCls.kind === 'axis' ? innerCls.axes : (nested ? 1 : 0);
    return { kind: 'axis', length: lit, axes: 1 + innerAxes, lead: 1 };
  }
  if (node && (node.type === 'NumberLiteral' || node.type === 'BoolLiteral'
      || node.type === 'StringLiteral')) {
    return { kind: 'const' };
  }
  const t = _typeOf(node, bindings);
  if (!t) return null;
  if (t.kind === 'table') {
    return { kind: 'table', length: _dim(t.nrows), columns: t.columns || null };
  }
  if (t.kind === 'tvector') {
    return {
      kind: 'axis', length: _dim(t.length), lead: 1,
      axes: 1 + _typeAxes(t.elem),
    };
  }
  if (t.kind === 'array') {
    const lead = Array.isArray(t.shape) ? t.shape.length : (t.rank || 1);
    return {
      kind: 'axis', length: _dim(t.shape && t.shape[0]), lead,
      axes: lead + _typeAxes(t.elem),
    };
  }
  if (t.kind === 'measure' || t.kind === 'kernel' || t.kind === 'function'
      || t.kind === 'module' || t.kind === 'failed' || t.kind === 'deferred') {
    return null;
  }
  // Scalars, sets, records and everything else §06 does not read row-wise.
  return { kind: 'const' };
}

// Axes of a value TYPE, counting a nested array's own axes.
function _typeAxes(t: any): number {
  if (!t) return 0;
  if (t.kind === 'array') {
    const rank = Array.isArray(t.shape) ? t.shape.length : (t.rank || 1);
    return rank + _typeAxes(t.elem);
  }
  if (t.kind === 'tvector') return 1 + _typeAxes(t.elem);
  return 0;
}

function _dim(d: any): number | null {
  return (typeof d === 'number' && Number.isInteger(d) && d > 0) ? d : null;
}

// The builtin measure-constructor name a `ksuperpose` kernel argument denotes,
// or null for a reified or user-defined kernel. Follows an alias chain (§04
// "Aliasing is just assignment").
//
// A §09 module member (`hepphys.Landau`) also answers null, which the family
// rule reads as rank-polymorphic. That is not a gap in the rank rule: every §09
// module distribution's parameters are scalars, and such a component does not
// type as a measure in this engine yet, so no mixture reaches the check.
function _componentName(node: any, body: any[]): string | null {
  let n = node;
  const seen = new Set<string>();
  while (n && n.type === 'Identifier' && !seen.has(n.name)) {
    seen.add(n.name);
    const rhs = _rhsOf(n.name, body);
    // No binding: a builtin constructor name.
    if (rhs == null) return n.name;
    n = rhs;
  }
  return null;
}

// The component's declared parameter names in order, plus whether its
// parameter list is declared at all. A positional family argument feeds the
// parameter at its own position (§06 passes the family "as to `broadcast`",
// which binds a positional data-arg to the head's ordered parameter name).
function _componentParams(name: string | null): { known: boolean, params: string[] } {
  if (!name) return { known: false, params: [] };
  const sig: any = require('./types.ts').signatureOf(name);
  const kwargs = sig && sig.kwargs;
  const isMeasure = !!(sig && sig.result && sig.result.kind === 'measure');
  if (!kwargs || !isMeasure) return { known: false, params: [] };
  return { known: true, params: Object.keys(kwargs) };
}

// §06's family-axis rule for ONE classified argument, or null when it holds.
//
// > with one family axis per collection argument: an argument's family axes are
// > its leading axes in excess of the rank (number of axes) of the parameter it
// > feeds, and any count other than one is a static error
//
// A parameter whose rank is unknown (a reified component, `Dirac`'s
// rank-polymorphic `value`) is read as rank-polymorphic: the rank that leaves
// exactly one family axis, so the argument holds. That is the only reading
// available without a declared rank.
//
// Shared with the runtime arm (`ksuperpose-runtime.ts`) so the two spellings of
// one mixture apply the same rule.
function familyAxisComplaint(
  cls: any, dist: string | null, param: string | null, label: string,
  known: boolean,
): string | null {
  const { paramRankOf } = require('./distribution-param-ranks.ts');
  // A TABLE family works by ROW axis with per-column element rank (§06 makes a
  // table's columns the collection arguments), so the rank comparison is
  // against the column ELEMENT's own axes and the row axis is the family axis.
  if (cls && cls.kind === 'table' && cls.columns) {
    for (const col of Object.keys(cls.columns)) {
      const colRank = paramRankOf(dist, col, known);
      if (colRank == null) continue;
      const colAxes = _typeAxes(cls.columns[col]);
      if (colAxes === colRank) continue;
      return `ksuperpose: family argument ${label} must carry exactly one `
        + 'family axis (spec §06: "an argument\'s family axes are its leading '
        + 'axes in excess of the rank (number of axes) of the parameter it '
        + 'feeds, and any count other than one is a static error"); the '
        + `table's rows are one axis and \`${col}\` has rank ${colRank}, so a `
        + `column of rank-${colAxes} elements does not leave exactly one`;
    }
    return null;
  }
  if (!cls || cls.kind !== 'axis') return null;
  const rank = paramRankOf(dist, param, known);
  if (rank == null) return null;
  const axes = typeof cls.axes === 'number' ? cls.axes : 1;
  if (axes === rank + 1) return null;
  return `ksuperpose: family argument ${label} must carry exactly one family `
    + 'axis (spec §06: "an argument\'s family axes are its leading axes in '
    + 'excess of the rank (number of axes) of the parameter it feeds, and any '
    + `count other than one is a static error"); \`${param}\` has rank ${rank}, `
    + `so a collection with ${axes} axes gives ${Math.max(0, axes - rank)} `
    + 'family axes';
}

// A numeric AST node as its value, or null when it is not a constant this
// reads. `-0.3` parses as a unary minus over a NumberLiteral, so the sign
// lives outside the literal and a NumberLiteral-only check would miss every
// negative weight written the obvious way.
// `node` is always a real node: the only caller walks a parsed ArrayLiteral's
// elements, and the parser never leaves a hole there — `[0.3, ]` drops the
// trailing comma and `[0.3,, 1.2]` yields the `__error__` identifier (both
// measured). So there is no null guard to take.
function _constNumber(node: any): number | null {
  if (node.type === 'NumberLiteral' && typeof node.value === 'number') return node.value;
  // The parser spells unary minus `op: '-'` (measured), so there is no `neg`
  // alternative to test for. A minus over a NON-literal (`[-x, 1.2]`) folds to
  // null and is left to the density walker.
  if (node.type === 'UnaryExpr' && node.op === '-') {
    const inner = _constNumber(node.operand);
    return inner == null ? null : -inner;
  }
  // No unary-plus case: FlatPPL has no prefix `+`, and `[+2.0]` is a parse
  // error (the element comes back as the `__error__` identifier).
  return null;
}

// The array-literal weight vector, reached inline or through the named
// binding (and an alias chain). Returns its element nodes, or null.
function _weightElements(node: any, body: any[]): any[] | null {
  let n = node;
  const seen = new Set<string>();
  while (n && n.type === 'Identifier' && !seen.has(n.name)) {
    seen.add(n.name);
    n = _rhsOf(n.name, body);
  }
  return (n && n.type === 'ArrayLiteral' && Array.isArray(n.elements))
    ? n.elements : null;
}

// §06 `ksuperpose`: `weights` "must be non-negative but need not be
// normalized". PER WEIGHT — a non-negative TOTAL is no defence, because a
// negative component makes the superposition a signed set function with no
// density, and `normalize` would divide by a mass that never was one. Zero
// stays legal: §06 gives it a meaning ("when every weight is zero it is the
// zero measure").
//
// Only a written CONSTANT is read here. There is no NaN case: FlatPPL has no
// NaN literal, so a NaN weight can only be computed (`0.0 / 0.0`) and belongs
// to the density walker's per-atom check, which owns every weight this cannot
// fold.
function _checkWeightSigns(weightsNode: any, body: any[], lift: any, diagnostics: any[]): boolean {
  const els = _weightElements(weightsNode, body);
  if (els == null) return true;
  for (let i = 0; i < els.length; i++) {
    const v = _constNumber(els[i]);
    if (v == null) continue;
    if (v < 0) {
      _err(diagnostics, `ksuperpose: weight #${i + 1} is ${v}, but §06 requires `
        + 'the weights to be non-negative ("It must be non-negative but need '
        + 'not be normalized") — a negative weight makes the component a '
        + 'signed measure, not a measure, whatever the other weights sum to. '
        + 'A ZERO weight is legal and drops the component out.',
        // Every parsed array element carries its own loc, so the refusal
        // points at the offending weight and not at the whole vector.
        els[i].loc);
      return false;
    }
  }
  return true;
}

function _err(diagnostics: any[], message: string, loc: any) {
  diagnostics.push({ severity: 'error', message, loc });
}

// One family argument, specialised to component `i` (1-based).
//
// Only the FAMILY axis is indexed; the parameter's own axes ride along. §07
// gives the row slice of a multi-axis array as `get(M, i, all)`, so an argument
// whose outer shape carries the parameter's axes takes one `all` per remaining
// outer axis. A nested spelling (an N-vector of matrices) carries them in the
// ELEMENT instead, so one selector already lands on the whole matrix.
function _componentArg(spec: any, i: number, sloc: any): any {
  const node = _clone(spec.node);
  let value = node;
  if (spec.cls.kind !== 'const') {
    // §06: a singular collection is "size one, expanded by repetition", so
    // it reads row 1 for every component.
    const row = spec.cls.length === 1 ? 1 : i;
    const sels: any[] = [AST.NumberLiteral(row, String(row), sloc)];
    const lead = typeof spec.cls.lead === 'number' ? spec.cls.lead : 1;
    for (let k = 1; k < lead; k++) sels.push(AST.Identifier('all', sloc));
    value = AST.IndexExpr(node, sels, sloc, 'get');
  }
  return spec.name == null ? value : AST.KeywordArg(spec.name, value, sloc);
}

// Rewrite one `ksuperpose(K, w)(family…)` application. Returns the
// `superpose(...)` replacement, or `call` unchanged when §06 makes the
// application a static error or when the rewrite is refused.
function _expandApplication(call: any, lift: any, bindings: any, diagnostics: any[], body: any[]): any {
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
  if (!_checkWeightSigns(weightsNode, body, lift, diagnostics)) return call;

  // Family arguments, in the application's own order and naming. Nothing
  // here needs the component's parameter names: each argument is handed to
  // the component call in the position or under the keyword it already has
  // (§05 lets a distribution take its parameters positionally).
  //
  // An argument this cannot classify does NOT refuse here. Whether it is a
  // refusal at all depends on N: with a static N the rows must be read now,
  // so the message below stands; with a runtime N the whole application goes
  // to the runtime arm, which reads each argument's shape from its value and
  // needs no static classification. So the error is HELD until N is known.
  const dist = _componentName(kernelNode, body);
  const { known, params } = _componentParams(dist);
  const specs: any[] = [];
  let unclassified: any = null;
  for (const a of call.args || []) {
    const isKw = a && a.type === 'KeywordArg';
    const node = isKw ? a.value : a;
    const param = isKw ? a.name : (params[specs.length] || null);
    const cls = _classify(node, bindings);
    if (cls == null) {
      if (unclassified == null) {
        unclassified = {
          message: 'ksuperpose: cannot tell whether family argument '
            + `${isKw ? '`' + a.name + '`' : '#' + (specs.length + 1)} is a `
            + 'size-N collection or a held-constant scalar, so the component '
            + 'rows cannot be read (spec §06). Bind it to a name with a '
            + 'statically-known vector length.',
          loc: (node && node.loc) || call.loc,
        };
      }
      specs.push({ name: isKw ? a.name : null, node, cls: null });
      continue;
    }
    const label = isKw ? '`' + a.name + '`' : '#' + (specs.length + 1);
    const complaint = familyAxisComplaint(cls, dist, param, label, known);
    if (complaint) {
      _err(diagnostics, complaint, (node && node.loc) || call.loc);
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
      if (s.cls && s.cls.kind === 'axis' && s.cls.length != null && s.cls.length > 1) {
        N = s.cls.length;
        break;
      }
    }
  }
  // §06: N "need not be statically known". Hand the application to the
  // runtime arm rather than refusing, and say nothing — a diagnostic here
  // would fire on a program the runtime arm goes on to score.
  if (N == null) return call;
  if (unclassified != null) {
    _err(diagnostics, unclassified.message, unclassified.loc);
    return call;
  }

  // §06: "each collection argument has size N or is singular (size one,
  // expanded by repetition)".
  for (const s of specs) {
    if (!s.cls || s.cls.kind !== 'axis') continue;
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
      return _expandApplication(call, lift, bindings, diagnostics, body);
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

// The §06 family classification for a NAMED binding, read off its inferred
// type. Exported so the runtime arm classifies by this rule and not a second
// one: the two spellings of the same mixture must agree on which arguments are
// size-N collections, which are singular, and which are held constant.
function classifyNamedFamilyArg(name: string, bindings: any): any {
  return _classify({ type: 'Identifier', name }, bindings);
}

module.exports = {
  expandKsuperposeApplications, classifyNamedFamilyArg, familyAxisComplaint,
  componentName: _componentName, componentParams: _componentParams,
};
