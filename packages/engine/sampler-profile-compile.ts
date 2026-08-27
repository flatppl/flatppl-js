'use strict';

// =====================================================================
// sampler-profile-compile.ts — single-point compiler for profile bodies
// =====================================================================
//
// A profile plot evaluates ONE inlined body at many points of a swept
// axis. `worker.profileN` mode 'function' calls `sampler.evaluateExpr`
// once per point, so it re-walks the whole tree every point: on the
// Dalitz `intensity_fn` body that is 5283 node visits of which only 176
// are structurally distinct. This module compiles the body ONCE per
// sweep into a tree of closures and calls it per point.
//
// NOT the same component as `sampler-eval-compile.ts`. That one serves
// `evaluateExprN`, the per-atom BATCHED path, emits one fused numeric
// loop, and bails whole-tree on any op outside the real scalar subset.
// A function-mode profile never enters it. This one is single-point,
// never bails (an op it does not specialise becomes an opaque closure
// calling the interpreter's own `evaluateCall`), and its main lever is
// common-subexpression elimination across a point, which a fused numeric
// loop cannot express.
//
// Two levers, both measured separately (see flatppl-dev/decisions-log.md):
//   1. Closure conversion — the per-node `switch (ir.kind)`, the
//      `ir.target.ns` probes, the `op in ARITH_OPS` test, the
//      `ops.isDeclared` lookup and the arity branch all move from run
//      time to compile time.
//   2. Per-point memoisation of repeated pure subtrees.
//
// Dependencies (ARITH_OPS, evaluateExpr, evaluateCall, resolveConst) are
// injected by `initProfileCompiler()` from sampler.ts — same pattern as
// sampler-eval-compile, and for the same reason: it avoids a require
// cycle and keeps those internals unexported.

const opsModule = require('./ops.ts');

let _OPS: any = null;
let _evaluateExpr: any = null;
let _evaluateCall: any = null;
let _resolveConst: any = null;

function initProfileCompiler(deps: any): void {
  _OPS = deps.ARITH_OPS;
  _evaluateExpr = deps.evaluateExpr;
  _evaluateCall = deps.evaluateCall;
  _resolveConst = deps.resolveConst;
}

// Ops that read or advance RNG state, or whose value otherwise depends
// on something other than the env. A subtree containing one of these is
// never memoised. A call with a `target` (a user function or a
// cross-module call) is treated the same way: its body is resolved from
// the env at run time, so nothing here can see whether it is pure.
const _IMPURE_OPS: Record<string, true> = {
  rand: true, rand_succ: true, rnginit: true, rngstate: true,
  builtin_sample: true,
};

// =====================================================================
// Structural identity
// =====================================================================
//
// CSE needs to know when two DISTINCT node objects denote the same
// expression. Each subtree gets an integer id: equal ids mean deep-equal
// subtrees (`loc` excluded).
//
// The id of a node is interned from a signature covering every own
// enumerable field except `loc`: an integer tuple of a leading tag, then
// interned field-name / child-id pairs in field-name order. Enumerating
// the fields generically rather than from a per-op list of "the fields
// that matter" is deliberate — a forgotten field would make two
// different expressions share an id, and the failure mode is a silently
// wrong number rather than an error.
//
// Two properties of that encoding are load-bearing, and both were paid
// for in measurement:
//   - The signature holds child IDS, not child signatures. Nesting the
//     signatures makes the root's O(tree), so interning every node costs
//     O(tree^2) — on the inlined Dalitz body that measured EIGHT TIMES
//     SLOWER than the interpreter this exists to beat.
//   - The signature is an integer tuple, not a string. Hashing the tuple
//     and comparing candidates elementwise allocates nothing, and the
//     analysis runs inside the sweep it is shortening.
//
// Purity and free-axis are properties of the structure, so they are
// cached per id and computed in the same bottom-up pass.
type Analysis = {
  names: Map<string, number>;    // field name / primitive → small int
  buckets: Map<number, number[]>;// tuple hash → candidate ids
  parts: number[][];             // id → its tuple
  byNode: WeakMap<object, number>;
  next: number;
  pure: boolean[];
  freeAxis: boolean[];
  // Occurrences of each CALL structure, indexed by id and pre-set to 0 by
  // `_intern`. An array rather than a Map so the read in pass 2 needs no
  // absent-key default.
  callCounts: number[];
};

function newAnalysis(): Analysis {
  return { names: new Map(), buckets: new Map(), parts: [],
           byNode: new WeakMap(), next: 0,
           pure: [], freeAxis: [], callCounts: [] };
}

const _TAG_ARRAY = -1;
const _TAG_OBJECT = -2;

function _nameId(An: Analysis, s: string): number {
  const hit = An.names.get(s);
  if (hit !== undefined) return hit;
  const id = An.names.size;
  An.names.set(s, id);
  return id;
}

// A payload primitive interns through the same name table. `-0` needs
// its own key: `String(-0)` is "0", and the two are not interchangeable
// in a literal (1/-0 is -Infinity).
function _primId(An: Analysis, v: any): number {
  const key = typeof v === 'number' && Object.is(v, -0)
    ? 'p#number:-0'
    : 'p#' + typeof v + ':' + String(v);
  return _nameId(An, key);
}

// Field names of `v` worth signing, sorted. `loc` is excluded so two
// occurrences of the same expression at different source positions share
// an id.
function _sortedFields(v: any): string[] {
  const names: string[] = [];
  for (const k in v) {
    if (!Object.prototype.hasOwnProperty.call(v, k)) continue;
    if (k === 'loc') continue;
    names.push(k);
  }
  return names.sort();
}

// Intern `v` (an IR node, an array, or a payload primitive) and return
// its id, filling in the purity and free-axis facts for that id.
function _idOf(v: any, An: Analysis): number {
  if (v === null || typeof v !== 'object') {
    const pid = _primId(An, v);
    return _intern(An, [_TAG_OBJECT - 1, pid], true, false);
  }
  const seen = An.byNode.get(v);
  if (seen !== undefined) {
    if (v.kind === 'call') An.callCounts[seen]++;
    return seen;
  }
  let pure = true;
  let freeAxis = false;
  const parts: number[] = [];
  if (Array.isArray(v)) {
    parts.push(_TAG_ARRAY);
    for (let i = 0; i < v.length; i++) {
      const cid = _idOf(v[i], An);
      parts.push(cid);
      if (!An.pure[cid]) pure = false;
      if (An.freeAxis[cid]) freeAxis = true;
    }
  } else {
    parts.push(_TAG_OBJECT);
    const names = _sortedFields(v);
    for (let i = 0; i < names.length; i++) {
      const cid = _idOf(v[names[i]], An);
      parts.push(_nameId(An, names[i]), cid);
      if (!An.pure[cid]) pure = false;
      if (An.freeAxis[cid]) freeAxis = true;
    }
    if (v.kind === 'call') {
      // A targeted call resolves its body from the env at run time, so
      // nothing here can see whether that body is pure.
      if (v.target || _IMPURE_OPS[v.op] === true) pure = false;
      // An `aggregate` consumes the axis labels in its body structurally
      // (the lowering permutes and reduces whole tensors; nothing sets
      // `env.__axisEnv`), so those labels are bound, not free.
      if (v.op === 'aggregate') freeAxis = false;
    } else if (v.kind === 'axis') {
      // Outside an aggregate an axis node throws — "axes are not
      // values", spec §05. Never a memoisation candidate.
      freeAxis = true;
    }
  }
  const id = _intern(An, parts, pure, freeAxis);
  An.byNode.set(v, id);
  if (v.kind === 'call') An.callCounts[id]++;
  return id;
}

function _hashParts(parts: number[]): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < parts.length; i++) {
    h ^= parts[i] + 0x9e3779b9;
    h = Math.imul(h, 0x01000193);
  }
  return h | 0;
}

function _samePartsAs(An: Analysis, id: number, parts: number[]): boolean {
  const other = An.parts[id];
  if (other.length !== parts.length) return false;
  for (let i = 0; i < parts.length; i++) if (other[i] !== parts[i]) return false;
  return true;
}

function _intern(An: Analysis, parts: number[], pure: boolean, freeAxis: boolean): number {
  const h = _hashParts(parts);
  let bucket = An.buckets.get(h);
  if (bucket !== undefined) {
    for (let i = 0; i < bucket.length; i++) {
      if (_samePartsAs(An, bucket[i], parts)) return bucket[i];
    }
  } else {
    bucket = [];
    An.buckets.set(h, bucket);
  }
  const id = An.next++;
  An.parts[id] = parts;
  An.pure[id] = pure;
  An.freeAxis[id] = freeAxis;
  An.callCounts[id] = 0;
  bucket.push(id);
  return id;
}

// Pass 1. `_idOf` tallies a call node once per edge that reaches it, so
// the id walk IS the occurrence count and no second traversal is needed
// (the tree is large enough that a second walk showed up in the sweep
// this compiler exists to shorten). A slot then goes only to a structure
// that occurs more than once. Leaf kinds are never counted: a memo read
// costs more than re-reading a literal or an env slot.
//
// The count under-reports a subtree hanging off a REPEATED node object,
// because the walk stops at the repeat. That only ever withholds a slot,
// and withholding it costs nothing: the ancestor's own slot already
// stops the descendant from being re-evaluated.
function _analyse(ir: any, An: Analysis): void {
  _idOf(ir, An);
}

// =====================================================================
// Session
// =====================================================================
//
// One session per sweep. It owns the memo slots, so slots are shared by
// every tree compiled in the session — the outer body and any aggregate
// body reached through `env.__bodyEval`.
//
// A slot hit requires BOTH the current generation (bumped once per
// point) and the same env OBJECT the value was computed under. The env
// guard is what makes sharing across trees safe without a scope
// analysis: a binder (a user-function call, a `reduce` element env)
// builds its env with `Object.assign({}, env)`, so a subtree evaluated
// under a different set of bindings arrives with a different env object
// and recomputes. Env objects are never mutated in place during a point
// — the only in-place write is `worker.profileN` setting the swept name
// between points, which the generation bump covers.
type Session = {
  g: number;
  slots: number;
  idToSlot: Map<number, number>;
  An: Analysis;
  gen: number[];
  envs: any[];
  vals: any[];
  kinds: number[];
  compiled: WeakMap<object, any> | null;   // built on first aggregate re-entry
};

function newSession(): Session {
  return {
    g: 1, slots: 0, idToSlot: new Map(), An: newAnalysis(),
    gen: [], envs: [], vals: [], kinds: [], compiled: null,
  };
}

// =====================================================================
// Memo wrapper
// =====================================================================
//
// Three value kinds are cached: a number or boolean, a scalar complex
// `{re, im}`, and a Value (`{shape, data, …}`). Nothing else — a record,
// a tuple, a JS array or a `__table__` is left uncached rather than
// reasoned about.
//
// Every cached OBJECT is handed out as a fresh wrapper, so each
// occurrence receives its own object exactly as the interpreter's
// per-occurrence construction would. That matters twice: an in-place
// write through one occurrence cannot reach the others, and a reference
// comparison downstream still sees two distinct objects.
//
// A Value's wrapper shares the `data` (and `im`) buffer and copies only
// the small `shape` array. Sharing the buffer is the engine's own
// convention for a non-copying result — `value.transpose` is defined to
// alias it (value.test.ts, "transpose is laziness-only") — and no engine
// module writes into a buffer it received.
const _CACHE_NONE = 0, _CACHE_PRIM = 1, _CACHE_COMPLEX = 2, _CACHE_VALUE = 3;

function _cacheKind(v: any): number {
  const t = typeof v;
  if (t === 'number' || t === 'boolean') return _CACHE_PRIM;
  if (v === null || t !== 'object') return _CACHE_NONE;
  if (Array.isArray(v.shape) && v.data instanceof Float64Array) return _CACHE_VALUE;
  if (typeof v.re === 'number' && typeof v.im === 'number'
      && Object.keys(v).length === 2) return _CACHE_COMPLEX;
  return _CACHE_NONE;
}

function _rewrap(kind: number, c: any): any {
  if (kind === _CACHE_PRIM) return c;
  if (kind === _CACHE_COMPLEX) return { re: c.re, im: c.im };
  const out = Object.assign({}, c);
  out.shape = c.shape.slice();
  return out;
}

function _memoise(inner: any, slot: number, S: Session): any {
  return function (env: any): any {
    if (S.gen[slot] === S.g && S.envs[slot] === env) {
      return _rewrap(S.kinds[slot], S.vals[slot]);
    }
    const v = inner(env);
    const kind = _cacheKind(v);
    if (kind === _CACHE_NONE) return v;
    S.gen[slot] = S.g; S.envs[slot] = env; S.kinds[slot] = kind;
    S.vals[slot] = kind === _CACHE_PRIM ? v : _rewrap(kind, v);
    return v;
  };
}

// =====================================================================
// Compilation
// =====================================================================

function _compileNode(node: any, S: Session): any {
  if (node === null || typeof node !== 'object' || typeof node.kind !== 'string') {
    // Not an IR node — let the interpreter produce its own error.
    return function (env: any) { return _evaluateExpr(node, env); };
  }
  const raw = _compileByKind(node, S);
  if (node.kind !== 'call') return raw;
  const An = S.An;
  // Read the id WITHOUT `_idOf`: that call tallies an occurrence, so
  // asking it here would push every unique node's count to 2 and hand
  // out a slot to subtrees that never repeat. Pass 1 has already visited
  // every node reachable from the root, so the entry is present — and a
  // missing one falls out through the count test below anyway.
  const id = An.byNode.get(node) as number;
  if (An.callCounts[id] < 2) return raw;
  if (!An.pure[id] || An.freeAxis[id]) return raw;
  let slot = S.idToSlot.get(id);
  if (slot === undefined) { slot = S.slots++; S.idToSlot.set(id, slot); }
  return _memoise(raw, slot, S);
}

function _compileByKind(node: any, S: Session): any {
  switch (node.kind) {
    case 'lit': {
      const v = node.value;
      return function () { return v; };
    }
    case 'const': {
      const name = node.name;
      // `pi` / `e` / `inf` / `-inf` are numbers and fold. `im` returns a
      // fresh `{re, im}` per call in the interpreter, so keep the call.
      const probe = _resolveConst(name);
      if (typeof probe === 'number') return function () { return probe; };
      return function () { return _resolveConst(name); };
    }
    case 'ref': {
      const name = node.name;
      const ns = node.ns;
      return function (env: any) {
        if (env == null || !(name in env)) {
          throw new Error(
            `evaluateExpr: unbound ${ns} reference '${name}' — env must ` +
            `provide values for all upstream-resolved names`
          );
        }
        return env[name];
      };
    }
    case 'axis': {
      const name = node.name;
      return function (env: any) {
        const axisEnv = env && env.__axisEnv;
        if (!axisEnv || !(name in axisEnv)) {
          throw new Error(`evaluateExpr: axis '.${name}' is not in scope ` +
            `(legal only inside aggregate(...))`);
        }
        return axisEnv[name];
      };
    }
    case 'call':
      return _compileCall(node, S);
    default:
      return function (env: any) { return _evaluateExpr(node, env); };
  }
}

function _compileCall(node: any, S: Session): any {
  // A targeted call (user function or cross-module) resolves its body
  // from the env at run time — opaque.
  if (node.target && node.target.ns) return _opaque(node);
  const op = node.op;
  // Anything without an ARITH_OPS entry is opaque. That INCLUDES the
  // higher-order ops — `aggregate`, `broadcast`, `reduce`, `scan`,
  // `filter` — which `evaluateCall` dispatches through
  // `ops.dispatchHigherOrder` before its arithmetic arm and which are
  // deliberately absent from ARITH_OPS.
  if (!(op in _OPS)) return _opaque(node);
  const rawArgs = node.args;
  const n = rawArgs == null ? 0 : rawArgs.length;
  // `evaluateCall`'s arithmetic arm builds its argument list from
  // `ir.args` alone and ignores kwargs, so compile the same way.
  const declared = opsModule.isDeclared(op) === true;
  if (n === 1) {
    const a0 = _compileNode(rawArgs[0], S);
    if (declared) {
      return function (env: any) { return opsModule.dispatch(op, [a0(env)]); };
    }
    const f = _OPS[op];
    return function (env: any) { return f(a0(env)); };
  }
  if (n === 2) {
    const a0 = _compileNode(rawArgs[0], S);
    const a1 = _compileNode(rawArgs[1], S);
    if (declared) {
      return function (env: any) { return opsModule.dispatch(op, [a0(env), a1(env)]); };
    }
    const f = _OPS[op];
    return function (env: any) { return f(a0(env), a1(env)); };
  }
  return _compileVariadic(op, rawArgs, S, declared);
}

// Arity 0 and 3-plus. Split from the arity-1/2 arms so the hot closures
// there hold a fixed-length argument literal and no loop.
function _compileVariadic(op: string, rawArgs: any[] | null | undefined,
                         S: Session, declared: boolean): any {
  const cs: any[] = [];
  const n = rawArgs == null ? 0 : rawArgs.length;
  for (let i = 0; i < n; i++) cs.push(_compileNode((rawArgs as any[])[i], S));
  if (declared) {
    return function (env: any) {
      const args: any[] = [];
      for (let i = 0; i < cs.length; i++) args.push(cs[i](env));
      return opsModule.dispatch(op, args);
    };
  }
  const f = _OPS[op];
  return function (env: any) {
    const args: any[] = [];
    for (let i = 0; i < cs.length; i++) args.push(cs[i](env));
    return f(...args);
  };
}

// Everything the compiler does not specialise runs through the
// interpreter's own `evaluateCall`, on the same node, so its behaviour
// and its error messages are unchanged. The children of an opaque node
// are NOT compiled — the interpreter walks them itself.
function _opaque(node: any): any {
  return function (env: any) { return _evaluateCall(node, env); };
}

// =====================================================================
// Public entry
// =====================================================================

// Compile `ir` for repeated single-point evaluation. Returns
//   { evalPoint(env), nextPoint(), stats() }
// `nextPoint()` invalidates the memo slots; call it before each point.
// Every body is compiled — there is deliberately NO small-body bail.
// A size threshold was written and measured: it cost 7 to 11 % on the
// small bodies it was meant to protect (0/13 pairs on three cases),
// because probing the node count is itself a walk and the bodies at
// stake evaluate in tens of nanoseconds. It made the one case that
// compiling had slowed (`minimal.flatppl` `f_sqrt`, -5.5 % on a 14 us
// sweep) worse, and it turned two 1.3x WINS into losses. Do not add it
// back without measuring the small cases too.
function compileProfileBody(ir: any): any {
  const S = newSession();
  _analyse(ir, S.An);
  const root = _compileNode(ir, S);
  // Bodies reached through the aggregate lowering compile lazily into
  // the SAME session, so their repeated subtrees share the outer body's
  // slots whenever they arrive with the same env object.
  const bodyEval = function (bodyIR: any, env: any): any {
    if (bodyIR === null || typeof bodyIR !== 'object') {
      return _evaluateExpr(bodyIR, env);
    }
    if (S.compiled === null) S.compiled = new WeakMap();
    let fn = S.compiled.get(bodyIR);
    if (fn === undefined) {
      _analyse(bodyIR, S.An);
      fn = _compileNode(bodyIR, S);
      S.compiled.set(bodyIR, fn);
    }
    return fn(env);
  };
  return {
    bodyEval,
    evalPoint: function (env: any) { return root(env); },
    nextPoint: function () { S.g++; },
    stats: function () {
      return { slots: S.slots, distinctSubtrees: S.An.next, compiled: true };
    },
  };
}

module.exports = {
  initProfileCompiler, compileProfileBody,
  // Exported for tests.
  newAnalysis, _idOf, _analyse, _cacheKind, _rewrap,
  _hashParts, _intern, _samePartsAs,
  _CACHE_NONE, _CACHE_PRIM, _CACHE_COMPLEX, _CACHE_VALUE,
};
