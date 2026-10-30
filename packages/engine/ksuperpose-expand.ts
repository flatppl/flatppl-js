'use strict';

// Direct ksuperpose shares broadcast's outer axes, then lowers to weighted
// superpose components. Density, normalization and per-output component draws
// therefore remain on the ordinary mixture paths. Unknown shapes wait until
// derivation building, where fixed operand values are available.
const AST = require('./ast.ts');
const { bindMixedArgs } = require('./broadcast-bind.ts');
const { mergeBroadcastShapes } = require('./broadcast-shape.ts');

function _clone(node: any): any {
  if (node == null || typeof node !== 'object') return node;
  if (Array.isArray(node)) return node.map(_clone);
  return Object.fromEntries(Object.entries(node).map(([k, v]) => [k, _clone(v)]));
}

function _isKsuperposeCall(node: any): boolean {
  return node?.type === 'CallExpr' && node.callee?.type === 'Identifier'
    && node.callee.name === 'ksuperpose';
}

function classifyType(t: any): any {
  if (!t || t.kind === 'deferred' || t.kind === 'any' || t.kind === 'failed') return null;
  if (t.kind === 'array') return { shape: t.shape?.slice() || null };
  if (t.kind === 'tvector') return { shape: [t.length] };
  if (t.kind === 'table') return { shape: [t.nrows], table: true, columns: t.columns };
  if (t.kind === 'record' || t.kind === 'tuple') return { invalid: t.kind };
  return { shape: [] };
}

function _classify(node: any, bindings: any, types?: { get(key: string): any }): any {
  const inferred = types?.get(JSON.stringify(node?.loc));
  if (inferred) return classifyType(inferred);
  if (node?.type === 'Identifier') return classifyType(bindings.get(node.name)?.inferredType);
  if (node?.type === 'ArrayLiteral') return { shape: [node.elements.length] };
  if (node?.type === 'TupleLiteral') return { invalid: 'tuple' };
  if (['NumberLiteral', 'BoolLiteral', 'StringLiteral'].includes(node?.type)) return { shape: [] };
  if (node?.type === 'UnaryExpr' && node.operand?.type === 'NumberLiteral') return { shape: [] };
  return null;
}

function componentIndex(shape: number[], index: number[]): number[] {
  return shape.map((n, i) => n === 1 ? 1 : index[i] + 1);
}

function _cell(node: any, shape: number[], index: number[], loc: any): any {
  return shape.length ? AST.IndexExpr(_clone(node), componentIndex(shape, index)
    .map(i => AST.NumberLiteral(i, String(i), loc)), loc, 'get') : _clone(node);
}

function _checkWeightSigns(node: any, body: any[], diagnostics: any[]): boolean {
  const seen = new Set<string>();
  while (node?.type === 'Identifier' && !seen.has(node.name)) {
    seen.add(node.name);
    node = body.find(stmt => stmt.type === 'AssignStatement'
      && stmt.names?.some((n: any) => n.name === node.name))?.value;
  }
  const values = node?.type === 'ArrayLiteral' ? node.elements : [node];
  for (const [i, value] of values.entries()) {
    const negative = value?.type === 'UnaryExpr' && value.op === '-'
      && value.operand?.type === 'NumberLiteral' && value.operand.value > 0;
    if (negative) {
      diagnostics.push({ severity: 'error', loc: value.loc,
        message: `ksuperpose: weight #${i + 1} is ${-value.operand.value}; weights must be non-negative (spec §06)` });
      return false;
    }
  }
  return true;
}

function _expandApplication(call: any, bindings: any, diagnostics: any[], body: any[], types: { get(key: string): any; resolve(node: any): any }): any {
  const args = call.args || [];
  const fail = (message: string) => {
    diagnostics.push({ severity: 'error', message, loc: call.loc });
    return call;
  };
  if (args.length < 2 || args.slice(0, 2).some((a: any) => a.type === 'KeywordArg')) {
    return fail('ksuperpose expects positional kernel and weights, followed by kernel arguments (spec §06)');
  }
  if (!_checkWeightSigns(args[1], body, diagnostics)) return call;
  let params: any[];
  try { params = bindMixedArgs(args.slice(2), args[0], bindings); }
  catch (e: any) { return fail(e.message); }
  const specs = [args[1], ...params].map(a => ({
    name: a.type === 'KeywordArg' ? a.name : null,
    node: a.type === 'KeywordArg' ? a.value : a,
    cls: _classify(a.type === 'KeywordArg' ? a.value : a, bindings, types),
  }));
  if (specs.some(s => s.cls?.invalid)) return fail('ksuperpose: records and tuples are not allowed as broadcast operands (spec §04)');
  // Preserve static expansion when a fixed operand has an imprecise type.
  if (specs[0].cls?.shape?.every((n: any) => typeof n === 'number')) {
    for (const spec of specs.slice(1)) {
      if (spec.cls?.shape?.some((n: any) => typeof n !== 'number')) {
        const value = types.resolve(spec.node);
        if (value !== undefined && typeof value !== 'symbol') {
          const slot = require('./broadcast-shape.ts').classifyAxisStructure(value);
          spec.cls = { ...spec.cls, shape: slot.coll ? slot.outerShape : [] };
        }
      }
    }
  }
  // Keep unresolved dimensions for the value-aware expansion, including a
  // known weight shape with an unresolved parameter shape.
  if (specs.some(s => !s.cls?.shape || s.cls.shape.some((n: any) => typeof n !== 'number'))) return call;
  let shape: number[];
  try { shape = mergeBroadcastShapes(specs.map(s => s.cls.shape)); }
  catch (e: any) { return fail('ksuperpose: ' + e.message); }
  const loc = { ...call.loc, synthetic: true, source: 'ksuperpose-expand' };
  const components: any[] = [];
  const index = shape.map(() => 0);
  const total = shape.reduce((a, b) => a * b, 1);
  // Keep the component domain on an empty mixture. The value-aware pass
  // builds the empty superposition without inventing a component parameter.
  if (total === 0) return call;
  for (let i = 0; i < total; i++) {
    const cells = specs.map(s => {
      const v = _cell(s.node, s.cls.shape, index, loc);
      return s.name == null ? v : AST.KeywordArg(s.name, v, loc);
    });
    let cellArgs = cells.slice(1);
    if (params.length === 1 && params[0].type !== 'KeywordArg' && specs[1].cls.table) {
      const iface = args[0].type === 'Identifier'
        ? bindings.get(args[0].name)?.inferredType || require('./types.ts').builtinInterface(args[0].name) : null;
      if (iface && !require('./types.ts').acceptsPositional(iface, 1)) {
        cellArgs = Object.keys(specs[1].cls.columns).map(name =>
          AST.KeywordArg(name, AST.FieldAccess(_clone(cells[1]), name, loc), loc));
      }
    }
    components.push(AST.CallExpr(AST.Identifier('weighted', loc), [cells[0],
      AST.CallExpr(_clone(args[0]), cellArgs, loc)], loc));
    for (let a = shape.length - 1; a >= 0; a--) {
      if (++index[a] < shape[a]) break;
      index[a] = 0;
    }
  }
  return AST.CallExpr(AST.Identifier('superpose', loc), components, loc);
}

// Recursively rewrite every `ksuperpose` application in `node`. Returns
// `node` unchanged (same reference) when nothing was rewritten, so a model
// without `ksuperpose` pays no clone cost. The structural recursion is
// required rather than speculative: `x ~ M` desugars to `x = draw(M)` at
// parse time, so an application written `x ~ normalize(ksuperpose(…, …))`
// arrives nested two levels down.
function _rewrite(node: any, bindings: any, diagnostics: any[], body: any[], types: { get(key: string): any; resolve(node: any): any }): any {
  if (node == null || typeof node !== 'object') return node;
  if (Array.isArray(node)) {
    let out: any = node;
    for (let i = 0; i < node.length; i++) {
      const r = _rewrite(node[i], bindings, diagnostics, body, types);
      if (r !== node[i]) {
        if (out === node) out = node.slice();
        out[i] = r;
      }
    }
    return out;
  }
  if (node.type === 'CallExpr') {
    if (_isKsuperposeCall(node)) {
      // Rewrite the family arguments first: they may themselves contain a
      // nested application.
      const inner = _rewrite(node.args, bindings, diagnostics, body, types);
      const call = inner === node.args ? node : { ...node, args: inner };
      return _expandApplication(call, bindings, diagnostics, body, types);
    }
  }
  let out: any = node;
  for (const k of Object.keys(node)) {
    if (k === 'loc' || k === 'type') continue;
    const r = _rewrite(node[k], bindings, diagnostics, body, types);
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
function expandKsuperposeApplications(ast: any, bindings: any, diagnostics: any[], loweredModule?: any): boolean {
  if (!ast || !Array.isArray(ast.body)) return false;
  // Most models contain no ksuperpose; do not scan their IR for types.
  let inferred: Map<string, any> | null = null;
  let resolve: any;
  const types = { resolve(node: any): any {
    if (node.type !== 'Identifier') return undefined;
    resolve ||= require('./fixed-eval.ts').makeResolver({ loweredModule });
    return resolve({ kind: 'ref', ns: 'self', name: node.name });
  }, get(key: string): any {
    if (!inferred) {
      inferred = new Map<string, any>();
      const collect = (ir: any) => {
        if (!ir || typeof ir !== 'object') return;
        if (ir.loc && ir.meta?.type) inferred!.set(JSON.stringify(ir.loc), ir.meta.type);
        require('./ir-walk.ts').forEachIRChild(ir, collect);
      };
      for (const b of loweredModule?.bindings?.values() || []) collect(b.rhs);
    }
    return inferred.get(key);
  } };
  let changed = false;
  const done = new Map<any, any>();
  const rewriteSlot = (holder: any, key: string) => {
    const node = holder && holder[key];
    if (!node || typeof node !== 'object') return;
    let out;
    if (done.has(node)) {
      out = done.get(node);
    } else {
      out = _rewrite(node, bindings, diagnostics, ast.body, types);
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

module.exports = { expandKsuperposeApplications, classifyType, componentIndex };
