'use strict';

// =====================================================================
// sampler-eval-batched.ts — batched IR evaluator + ARITH_OPS_N table
// =====================================================================
//
// Extracted from sampler.ts as part of the §17.5 sampler split
// (engine-concepts §11). Holds the batched evaluator dispatcher
// (evaluateExprN / _evalN / _batchedApproximation / _perAtomFallback),
// the scalar-batched broadcast helpers (broadcast1/2/3, isBatch), the
// ARITH_OPS_N table, the shape-aware mul/add/sub/neg wrappers, and
// the complex-aware batched dispatchers (_cxXxx).
//
// CJS module (mirrors sampler-aggregate.ts) — uses require() to break
// the cycle with sampler.ts on `evaluateExpr`/`resolveConst`/`ARITH_OPS`,
// so the file must NOT use ES export syntax.
//
// Build sequence: sampler.ts defines ARITH_OPS, then calls
// `initARITHOPSN(ARITH_OPS)` (this module) to populate the ARITH_OPS_N
// table. The dispatcher core can be required eagerly — its functions
// only invoke ARITH_OPS_N entries at call time, so the order of
// require → init → first-call is safe.

const valueLib = require('./value.ts');
const valueOps = require('./value-ops.ts');
const { _isComplex } = require('./sampler-complex.ts');

// Lazy access to sampler.ts (cycle: this module evaluates IR via the
// single-point evaluator + reads runtime constants from sampler.ts).
// Same non-memoised pattern as sampler-aggregate.ts; Node caches the
// module object so re-`require` is cheap.
function _sampler(): any { return require('./sampler.ts'); }
function evaluateExpr(ir: any, env: any): any {
  return _sampler().evaluateExpr(ir, env);
}
function resolveConst(name: any): any {
  return _sampler()._internal.resolveConst(name);
}

// =====================================================================
// Batched scalar-arithmetic helpers — broadcast1/2/3, isBatch, etc.
// =====================================================================
//
// Polymorphic on Value inputs (engine-concepts §2.1). Each input may
// be a bare JS number/boolean, a bare Float64Array of length N, a
// Value with shape=[] (atom-indep scalar), or a Value with shape=[N]
// (atom-batched scalar).
//
// "Same kind as inputs" output semantics: when ANY input is a Value
// the result is wrapped as a Value (scalar(out) or batchedScalar(out));
// when all inputs are bare primitives the output stays bare. Preserves
// zero-allocation behaviour for single-point evaluation while letting
// Value-aware callers thread shape information through.

const isValueObj = valueLib.isValue;

function isBatch(v: any, N: any) {
  if (v == null) return false;
  // Bare Float64Array of length N (legacy path; also catches any
  // typed array with matching length — same as before).
  if (v.BYTES_PER_ELEMENT !== undefined
      && typeof v.length === 'number' && v.length === N) {
    return true;
  }
  // Value with shape=[N].
  if (isValueObj(v)
      && v.shape.length === 1 && v.shape[0] === N) {
    return true;
  }
  return false;
}

// Underlying Float64Array for a batched input (either bare or wrapped).
function _batchData(v: any) {
  return v instanceof Float64Array ? v : v.data;
}

// JS-number view of an atom-indep scalar (either bare or shape=[] Value).
// Bare arrays / non-batched arrays return unchanged — the scalar fn
// receives them as-is (legacy behaviour: garbage-in-garbage-out for
// shape mismatches at the scalar layer).
function _scalarVal(v: any) {
  if (isValueObj(v) && v.shape.length === 0) return v.data[0];
  return v;
}

// Any input is a Value? Triggers Value-wrapped output.
function _anyValue1(a: any)       { return isValueObj(a); }
function _anyValue2(a: any, b: any)    { return isValueObj(a) || isValueObj(b); }
function _anyValue3(a: any, b: any, c: any) { return isValueObj(a) || isValueObj(b) || isValueObj(c); }

function broadcast1(fn: any, a: any, N: any) {
  const wantValue = _anyValue1(a);
  if (!isBatch(a, N)) {
    const r = fn(_scalarVal(a));
    return wantValue ? valueLib.scalar(r) : r;
  }
  const ad = _batchData(a);
  const out = new Float64Array(N);
  for (let i = 0; i < N; i++) out[i] = +fn(ad[i]);
  return wantValue ? valueLib.batchedScalar(out) : out;
}

function broadcast2(fn: any, a: any, b: any, N: any) {
  const aB = isBatch(a, N), bB = isBatch(b, N);
  const wantValue = _anyValue2(a, b);
  if (!aB && !bB) {
    const r = fn(_scalarVal(a), _scalarVal(b));
    return wantValue ? valueLib.scalar(r) : r;
  }
  const out = new Float64Array(N);
  if (aB && bB) {
    const ad = _batchData(a), bd = _batchData(b);
    for (let i = 0; i < N; i++) out[i] = +fn(ad[i], bd[i]);
  } else if (aB) {
    const ad = _batchData(a), bs = _scalarVal(b);
    for (let i = 0; i < N; i++) out[i] = +fn(ad[i], bs);
  } else {
    const as = _scalarVal(a), bd = _batchData(b);
    for (let i = 0; i < N; i++) out[i] = +fn(as, bd[i]);
  }
  return wantValue ? valueLib.batchedScalar(out) : out;
}

function broadcast3(fn: any, a: any, b: any, c: any, N: any) {
  const aB = isBatch(a, N), bB = isBatch(b, N), cB = isBatch(c, N);
  const wantValue = _anyValue3(a, b, c);
  if (!aB && !bB && !cB) {
    const r = fn(_scalarVal(a), _scalarVal(b), _scalarVal(c));
    return wantValue ? valueLib.scalar(r) : r;
  }
  const ad = aB ? _batchData(a) : null;
  const bd = bB ? _batchData(b) : null;
  const cd = cB ? _batchData(c) : null;
  const as = aB ? null : _scalarVal(a);
  const bs = bB ? null : _scalarVal(b);
  const cs = cB ? null : _scalarVal(c);
  const out = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    out[i] = +fn(aB ? ad[i] : as, bB ? bd[i] : bs, cB ? cd[i] : cs);
  }
  return wantValue ? valueLib.batchedScalar(out) : out;
}

// =====================================================================
// Scalar-primitive arity table — drives ARITH_OPS_N construction
// =====================================================================
const _SCALAR_PRIM_ARITY = {
  // Arithmetic. `divide` is the spec §07 function-form of `/` (passed
  // to broadcast/reduce/scan); same scalar semantics as `div`, so it
  // belongs on the batched-broadcast path. Without an entry here it
  // fell to the slow per-atom fallback (real) and broke outright for
  // complex (the fallback can't pack {re,im}).
  add: 2, sub: 2, mul: 2, div: 2, divide: 2, mod: 2, neg: 1, pos: 1, pow: 2,
  // Elementary math
  abs: 1, abs2: 1, exp: 1, log: 1, log10: 1, log1p: 1, expm1: 1, sqrt: 1,
  sin: 1, cos: 1, tan: 1,
  asin: 1, acos: 1, atan: 1, atan2: 2,
  sinh: 1, cosh: 1, tanh: 1,
  asinh: 1, acosh: 1, atanh: 1,
  floor: 1, ceil: 1, round: 1,
  // Pairwise reductions
  min: 2, max: 2,
  // Special functions & link functions
  gamma: 1, loggamma: 1,
  logit: 1, invlogit: 1, probit: 1, invprobit: 1,
  // Comparison
  lt: 2, le: 2, gt: 2, ge: 2, equal: 2, unequal: 2,
  // Predicates
  isfinite: 1, isinf: 1, isnan: 1, iszero: 1,
  // Logic + conditional
  land: 2, lor: 2, lxor: 2, lnot: 1, ifelse: 3,
  // Scalar restrictors
  boolean: 1, integer: 1,
};

// =====================================================================
// ARITH_OPS_N table — populated by initARITHOPSN(ARITH_OPS)
// =====================================================================
//
// The table object is created here as an empty {} so consumers can
// hold a stable reference (the dispatcher closes over it). Entries are
// installed at sampler.ts module-load via `initARITHOPSN` once ARITH_OPS
// is fully defined.
const ARITH_OPS_N: any = {};

// Shape-aware mul / add / sub / neg take precedence over the
// broadcast{1,2}-based dispatch when any operand carries an intrinsic
// vector/matrix shape (Value with rank ≥ 1 whose leading dim isn't the
// atom count). Bare scalars and batched scalars (shape=[N]) stay on
// the scalar broadcast path.
function _shapeAwareCandidate(v: any, N: any) {
  if (!valueLib.isValue(v)) return false;
  const r = v.shape.length;
  if (r === 0) return false;
  if (r === 1 && v.shape[0] === N) return false;  // batched scalar
  return true;  // shape-rich OR atom-batched non-scalar
}

function _wrapShapeAwareBinopN(opName: any, opNameN: any) {
  const fallback = ARITH_OPS_N[opName];
  return (args: any, N: any) => {
    const a = args[0], b = args[1];
    if (_shapeAwareCandidate(a, N) || _shapeAwareCandidate(b, N)) {
      return valueOps[opNameN](valueLib.asValue(a), valueLib.asValue(b), N);
    }
    return fallback(args, N);
  };
}

// =====================================================================
// Complex batched broadcast (planar re/im, shape=[] or shape=[N])
// =====================================================================
//
// broadcast{1,2,3} coerce per-atom results with `+fn(...)`, which turns
// a complex {re, im} into NaN. Complex *shape-rich* values (shape=[N,k]
// per-atom vectors/matrices) already route through the shape-aware
// value-ops path (complex-aware since chunk 2). The remaining gap is:
//
//   - complex *batched scalars* (shape=[N] of one complex per atom), and
//   - the complex constructors / accessors (complex / real / imag /
//     conj / cis) which never had ARITH_OPS_N entries at all (atom-
//     dependent complex previously threw, per TODO §03).
//
// _cxBroadcast runs the EXISTING scalar ARITH_OPS primitive per atom
// (those already dispatch on {re, im}) and writes into two parallel
// Float64Arrays — the planar layout makes the batched form a pure
// re-pack of the scalar op, exactly as the design intends. Each arg is
// presented to the primitive in its natural scalar form (a JS number
// for real inputs, a {re, im} object for complex), identical to the
// atom-indep evaluateExpr path — so the primitives need no batched
// variant. Result type is detected from the primitive's output
// (number ⇒ real Value, {re, im} ⇒ complex Value).

// Does the complex path apply? True if any arg is complex (scalar
// {re, im} or a complex Value). Constructors (complex / cis) force the
// complex path explicitly at their registration site, since their
// inputs are real but the output is complex.
function _cxInPlay(args: any) {
  for (let i = 0; i < args.length; i++) {
    const v = args[i];
    if (_isComplex(v) || valueLib.isComplexValue(v)) return true;
  }
  return false;
}

// Per-arg accessor: returns a function i → (number | {re, im}) plus an
// `atomBatched` flag. Mirrors isBatch/_scalarVal/_batchData but is
// complex-aware and rejects shape-rich complex (handled elsewhere).
function _cxArgAccessor(v: any, N: any) {
  if (_isComplex(v)) return { batched: false, at: () => v };
  if (valueLib.isComplexValue(v)) {
    if (v.shape.length === 0) {
      const c = valueLib.readComplex(v);
      const z = { re: c.re[0], im: c.im[0] };
      return { batched: false, at: () => z };
    }
    if (v.shape.length === 1 && v.shape[0] === N) {
      const c = valueLib.readComplex(v);   // resolves conj once
      return { batched: true, at: (i: any) => ({ re: c.re[i], im: c.im[i] }) };
    }
    // Shape-rich complex is handled by _cxElementwise before any
    // accessor is built; reaching here would be a dispatch bug.
    throw new Error(
      '_cxArgAccessor: shape-rich complex Value (shape=[' +
      v.shape.join(',') + ']) must be routed through _cxElementwise');
  }
  if (valueLib.isValue(v)) {
    if (v.shape.length === 0) { const s = v.data[0]; return { batched: false, at: () => s }; }
    if (v.shape.length === 1 && v.shape[0] === N) {
      const d = v.data; return { batched: true, at: (i: any) => d[i] };
    }
    const s0 = v.data[0]; return { batched: false, at: () => s0 };
  }
  if (v instanceof Float64Array && v.length === N) {
    return { batched: true, at: (i: any) => v[i] };
  }
  const s = (typeof v === 'boolean') ? (v ? 1 : 0) : +v;
  return { batched: false, at: () => s };
}

// Shape-rich complex/real elementwise: applies the scalar primitive to
// every element of a per-atom vector/matrix (shape=[N, k…] or an
// atom-indep vector shape=[k>1]). Used for complex unary ops (abs2 /
// exp / log / sqrt / real / imag / conj / cis / pos) and the complex
// constructor over real vectors — the "free reshaping" ops that
// shouldn't be limited to scalars. Scalar args broadcast (length-1
// buffers); the governing shape-rich arg's swapped (transpose) bit is
// preserved (readComplex already folded conj, so the result is a pure
// transpose view, never a residual adjoint).
//
// Note: binary add/sub/mul over shape-rich complex never reach here —
// their wrapper routes shape-rich operands to the complex-aware
// value-ops path first. This covers the unary / constructor / scalar-
// op-over-vector cases that the value-ops shape dispatch doesn't.
function _cxRichArgGetter(v: any) {
  if (_isComplex(v)) return () => v;
  if (valueLib.isComplexValue(v)) {
    const c = valueLib.readComplex(v);
    if (c.re.length === 1) { const z = { re: c.re[0], im: c.im[0] }; return () => z; }
    return (i: any) => ({ re: c.re[i], im: c.im[i] });
  }
  if (valueLib.isValue(v)) {
    if (v.data.length === 1) { const s = v.data[0]; return () => s; }
    const d = v.data; return (i: any) => d[i];
  }
  if (v instanceof Float64Array) {
    if (v.length === 1) { const s = v[0]; return () => s; }
    return (i: any) => v[i];
  }
  const s = (typeof v === 'boolean') ? (v ? 1 : 0) : +v;
  return () => s;
}

function _cxElementwise(fn: any, args: any, shape: any, swapped: any) {
  let M = 1;
  for (let d = 0; d < shape.length; d++) M *= shape[d];
  const get = args.map(_cxRichArgGetter);
  const callArgs = new Array(args.length);
  const re = new Float64Array(M);
  let im: Float64Array | null = null;
  for (let i = 0; i < M; i++) {
    for (let k = 0; k < get.length; k++) callArgs[k] = get[k](i);
    const r = fn.apply(null, callArgs);
    if (_isComplex(r)) {
      if (im === null) im = new Float64Array(M);
      re[i] = r.re; im[i] = r.im;
    } else {
      re[i] = +r;
    }
  }
  let out;
  if (im === null) out = { shape: shape.slice(), data: re };
  else out = valueLib.complexValue(re, im, shape);
  if (swapped) out.t = 'T';
  return out;
}

// Is `v` a shape-rich Value for the complex path — a per-atom
// vector/matrix (rank ≥ 2) or an atom-indep vector (rank 1, length ≠ N
// and > 1)? Batched scalars (shape=[N]) and scalars are NOT rich.
function _cxIsRich(v: any, N: any) {
  if (!valueLib.isValue(v)) return false;
  const r = v.shape.length;
  if (r === 0) return false;
  if (r === 1) return v.shape[0] !== N && v.shape[0] > 1;
  return true;
}

function _cxBroadcast(fn: any, args: any, N: any) {
  // Shape-rich elementwise (per-atom vectors/matrices, atom-indep
  // vectors). The first rich arg governs the output shape + swapped
  // bit; remaining shape-rich args must match its element count.
  let richShape = null, richSwapped = false;
  for (let k = 0; k < args.length; k++) {
    if (_cxIsRich(args[k], N)) {
      richShape = args[k].shape;
      richSwapped = valueLib.isTransposeView(args[k]);
      break;
    }
  }
  if (richShape) return _cxElementwise(fn, args, richShape, richSwapped);

  const accs = new Array(args.length);
  let anyBatched = false;
  for (let k = 0; k < args.length; k++) {
    accs[k] = _cxArgAccessor(args[k], N);
    if (accs[k].batched) anyBatched = true;
  }
  if (!anyBatched) {
    const r = fn.apply(null, accs.map((a) => a.at(0)));
    return _isComplex(r)
      ? valueLib.complexValue([r.re], [r.im], [])
      : valueLib.scalar(+r);
  }
  const re = new Float64Array(N);
  let im: Float64Array | null = null;                       // allocated lazily on first complex
  const callArgs = new Array(args.length);
  for (let i = 0; i < N; i++) {
    for (let k = 0; k < accs.length; k++) callArgs[k] = accs[k].at(i);
    const r = fn.apply(null, callArgs);
    if (_isComplex(r)) {
      if (im === null) im = new Float64Array(N);
      re[i] = r.re; im[i] = r.im;
    } else {
      re[i] = +r;
    }
  }
  return im === null
    ? valueLib.batchedScalar(re)
    : valueLib.complexValue(re, im, [N]);
}

// Rewire complex-relevant arithmetic ops to take the complex path when
// complex is in play; otherwise keep the zero-alloc real Float64Array
// broadcast. add/sub/mul/neg already have a shape-aware wrapper (which
// handles complex shape-rich via value-ops); we only need to redirect
// their *scalar* fallback (non-shape-rich) to _cxBroadcast.
function _cxOrRealShapeAware(opName: any, ARITH_OPS: any) {
  const realEntry = ARITH_OPS_N[opName];
  const prim = ARITH_OPS[opName];
  return (args: any, N: any) => {
    for (let i = 0; i < args.length; i++) {
      if (_shapeAwareCandidate(args[i], N)) return realEntry(args, N);
    }
    return _cxInPlay(args) ? _cxBroadcast(prim, args, N) : realEntry(args, N);
  };
}

// div/divide/pos/pow/abs/abs2/exp/log/sqrt are scalar-only ops with no
// shape-rich value-ops path. When complex is in play they go through
// _cxBroadcast (which itself raises a clear guard for shape-rich
// complex — better than silently producing NaN). Otherwise the
// existing real broadcast entry is preserved unchanged.
function _cxOrReal(opName: any, ARITH_OPS: any) {
  const realEntry = ARITH_OPS_N[opName];
  const prim = ARITH_OPS[opName];
  return (args: any, N: any) => _cxInPlay(args)
    ? _cxBroadcast(prim, args, N)
    : realEntry(args, N);
}

// One-shot ARITH_OPS_N construction, invoked by sampler.ts once
// ARITH_OPS is fully defined. After this returns, every dispatch path
// (single ARITH_OPS_N[op](args, N)) is wired and frozen.
function initARITHOPSN(ARITH_OPS: any) {
  for (const op of Object.keys(_SCALAR_PRIM_ARITY)) {
    const arity = (_SCALAR_PRIM_ARITY as any)[op];
    const fn = (ARITH_OPS as any)[op];
    if (typeof fn !== 'function') {
      throw new Error(`ARITH_OPS_N: scalar primitive '${op}' has no ARITH_OPS entry`);
    }
    if (arity === 1) ARITH_OPS_N[op] = (args: any, N: any) => broadcast1(fn, args[0], N);
    else if (arity === 2) ARITH_OPS_N[op] = (args: any, N: any) => broadcast2(fn, args[0], args[1], N);
    else if (arity === 3) ARITH_OPS_N[op] = (args: any, N: any) => broadcast3(fn, args[0], args[1], args[2], N);
    else throw new Error(`ARITH_OPS_N: unsupported arity ${arity} for '${op}'`);
  }

  // Shape-aware mul / add / sub / neg.
  ARITH_OPS_N.mul = _wrapShapeAwareBinopN('mul', 'mulN');
  ARITH_OPS_N.add = _wrapShapeAwareBinopN('add', 'addN');
  ARITH_OPS_N.sub = _wrapShapeAwareBinopN('sub', 'subN');
  const _negBroadcast = ARITH_OPS_N.neg;
  ARITH_OPS_N.neg = (args: any, N: any) => {
    const a = args[0];
    if (_shapeAwareCandidate(a, N)) return valueOps.negN(valueLib.asValue(a), N);
    return _negBroadcast(args, N);
  };

  // Complex-aware wrappers.
  for (const op of ['add', 'sub', 'mul', 'neg']) {
    if (ARITH_OPS_N[op]) ARITH_OPS_N[op] = _cxOrRealShapeAware(op, ARITH_OPS);
  }
  for (const op of ['div', 'divide', 'pos', 'pow',
                     'abs', 'abs2', 'exp', 'log', 'sqrt']) {
    if (ARITH_OPS_N[op]) ARITH_OPS_N[op] = _cxOrReal(op, ARITH_OPS);
  }
  // complex / real / imag / conj / cis had no ARITH_OPS_N entry (atom-
  // dependent complex previously threw). Route them unconditionally
  // through _cxBroadcast: degrades to a real Value when inputs and
  // output are real (e.g. conj / real on real data), and produces a
  // complex Value for the constructors (complex / cis) and for complex
  // inputs. Arity is taken from the IR args length at call time.
  for (const op of ['complex', 'real', 'imag', 'conj', 'cis']) {
    const prim = (ARITH_OPS as any)[op];
    ARITH_OPS_N[op] = (args: any, N: any) => _cxBroadcast(prim, args, N);
  }
}

// =====================================================================
// Dispatcher core — evaluateExprN / _evalN / _batchedApproximation /
// _perAtomFallback. The batched IR walker.
// =====================================================================
//
//   refArrays: { name → Float64Array(count) } per-atom value overrides.
//              May be null/empty.
//   baseEnv:   atom-independent env (session env + lifted fixed-phase).
//   opts.overlay: { name → scalar } that wins over BOTH refArrays and
//                 baseEnv at every leaf — used by density.js for
//                 env-threading from consumed observation fields.
//
// Env-precedence at refs (highest first): overlay > refArrays > baseEnv.
//
// Scalar arith ops dispatch through ARITH_OPS_N (batched broadcast).
// Non-scalar ops fall back to per-atom dispatch via the single-point
// `evaluateExpr` — fine for ops where per-atom inputs are uncommon
// (vector/matrix/record builders typically take atom-indep inputs);
// future batched-non-scalar rewrites would eliminate the fallback.

function evaluateExprN(ir: any, refArrays: any, count: any, baseEnv: any, opts: any) {
  const N = count | 0;
  if (N <= 0) throw new Error('evaluateExprN: count must be positive');
  const overlay = (opts && opts.overlay) || null;
  return _evalN(ir, refArrays || null, N, baseEnv || {}, overlay);
}

function _evalN(ir: any, refArrays: any, N: any, baseEnv: any, overlay: any) {
  switch (ir.kind) {
    case 'lit':
      return ir.value;
    case 'const':
      return resolveConst(ir.name);
    case 'ref': {
      if (overlay && Object.prototype.hasOwnProperty.call(overlay, ir.name)) {
        return overlay[ir.name];
      }
      if (refArrays && Object.prototype.hasOwnProperty.call(refArrays, ir.name)) {
        return refArrays[ir.name];
      }
      if (baseEnv != null && ir.name in baseEnv) return baseEnv[ir.name];
      throw new Error(
        `evaluateExprN: unbound ${ir.ns} reference '${ir.name}' — env must ` +
        `provide values for all upstream-resolved names`
      );
    }
    case 'call': {
      const op = ir.op;
      // Scalar-batched dispatch.
      if (op in ARITH_OPS_N) {
        const args = (ir.args || []).map((a: any) => _evalN(a, refArrays, N, baseEnv, overlay));
        return ARITH_OPS_N[op](args, N);
      }
      // Batched approximation functions. Each has the shape
      // (atom-indep coefficients/edges/values, per-atom x). If x batches
      // (Float64Array(N) or Value shape=[N]), collapse the per-atom JS-
      // call overhead into a single tight inner loop. Otherwise fall
      // through to _perAtomFallback (atom-indep eval or per-atom dispatch).
      if (op === 'polynomial' || op === 'bernstein' || op === 'stepwise') {
        const r = _batchedApproximation(op, ir, refArrays, N, baseEnv, overlay);
        if (r !== _BATCH_FELL_THROUGH) return r;
      }
      // Atom-batched aggregate (engine-concepts §20.10.10): the
      // fusion (a) Step 2 rewrite emits `aggregate(R, [.atom], body)`
      // IRs where some body refs are atom-batched. The non-batched
      // evaluator would per-atom-loop here (~60 µs/atom); the
      // batched evaluator lifts the body tensor to shape
      // [N, ...outAxes, ...reduceAxes] in one pass and tail-reduces
      // (~5 µs/atom). Returns a Value shape=[N, ...outAxes].
      if (op === 'aggregate') {
        const agg = require('./sampler-aggregate.ts');
        return agg._evalAggregateBroadcastReduceN(ir, refArrays, N, baseEnv, overlay);
      }
      // Non-batched op: per-atom dispatch through the existing
      // single-point evaluateExpr. Atom-indep when no per-atom refs
      // touch this subtree (cheap one-shot); per-atom loop otherwise.
      return _perAtomFallback(ir, refArrays, N, baseEnv, overlay);
    }
    default:
      throw new Error(`evaluateExprN: unsupported IR node kind '${ir.kind}'`);
  }
}

// Batched approximation function dispatch (polynomial /
// bernstein / stepwise). Each takes atom-indep coefficient-class
// arguments and a per-atom x; if x batches, we run one tight Horner /
// Bernstein-basis / stepwise loop over the entire N-atom batch
// rather than N JS function calls through the per-atom fallback.
//
// The sentinel `_BATCH_FELL_THROUGH` distinguishes "I refused to
// handle this; please use the per-atom fallback" from a legitimate
// `null` / `undefined` result.
const _BATCH_FELL_THROUGH = Symbol('batch-fell-through');

function _batchedApproximation(op: any, ir: any, refArrays: any, N: any, baseEnv: any, overlay: any) {
  const kw = ir.kwargs || {};
  // Each op has the same coefficient-class + x shape; pull both.
  let coeffsIR, xIR, edgesIR;
  if (op === 'polynomial') {
    coeffsIR = kw.coefficients != null ? kw.coefficients : ir.args && ir.args[0];
    xIR      = kw.x            != null ? kw.x            : ir.args && ir.args[1];
  } else if (op === 'bernstein') {
    coeffsIR = kw.coefficients != null ? kw.coefficients : ir.args && ir.args[0];
    xIR      = kw.x            != null ? kw.x            : ir.args && ir.args[1];
  } else { // stepwise
    edgesIR  = kw.edges        != null ? kw.edges        : ir.args && ir.args[0];
    coeffsIR = kw.values       != null ? kw.values       : ir.args && ir.args[1];
    xIR      = kw.x            != null ? kw.x            : ir.args && ir.args[2];
  }
  if (!coeffsIR || !xIR) return _BATCH_FELL_THROUGH;

  // Evaluate the atom-indep operands once. For batching to apply,
  // coefficients/edges/values must not depend on per-atom refs — we
  // detect this by evaluating against baseEnv WITHOUT refArrays.
  // The cheap path: if the IR has no self-refs, evaluation against
  // baseEnv succeeds; otherwise it throws and we fall through.
  let coeffs, edges;
  try {
    coeffs = evaluateExpr(coeffsIR, baseEnv);
    if (op === 'stepwise') edges = evaluateExpr(edgesIR, baseEnv);
  } catch (_) {
    return _BATCH_FELL_THROUGH;
  }
  // Unwrap shape-explicit Value → typed-array data view.
  if (valueLib.isValue(coeffs)) coeffs = coeffs.data;
  if (op === 'stepwise' && valueLib.isValue(edges)) edges = edges.data;
  if (!Array.isArray(coeffs) && !(coeffs && coeffs.BYTES_PER_ELEMENT)) {
    return _BATCH_FELL_THROUGH;
  }
  if (op === 'stepwise'
      && !Array.isArray(edges) && !(edges && edges.BYTES_PER_ELEMENT)) {
    return _BATCH_FELL_THROUGH;
  }

  // Evaluate x. If it's per-atom (Float64Array(N) or Value shape=[N]),
  // batch. Otherwise fall through.
  const xVal = _evalN(xIR, refArrays, N, baseEnv, overlay);
  const xIsBatch = isBatch(xVal, N);
  if (!xIsBatch) {
    // x is atom-indep — the per-atom fallback already short-circuits
    // to a single evaluateExpr call (one-shot path). No reason to
    // run the batched loop here.
    return _BATCH_FELL_THROUGH;
  }
  const xData = xVal instanceof Float64Array ? xVal : xVal.data;

  // Tight inner loops. coeffs is treated as length-k indexed via [];
  // works for Array and Float64Array uniformly.
  const out = new Float64Array(N);
  if (op === 'polynomial') {
    const k = coeffs.length;
    if (k === 0) {
      // Zero polynomial → 0 for every atom; out is already zero-filled.
    } else {
      for (let i = 0; i < N; i++) {
        const x = xData[i];
        let acc = coeffs[k - 1];
        for (let j = k - 2; j >= 0; j--) acc = acc * x + coeffs[j];
        out[i] = acc;
      }
    }
  } else if (op === 'bernstein') {
    const n = coeffs.length - 1;
    if (n < 0) {
      // out already zeros
    } else {
      for (let i = 0; i < N; i++) {
        const x = xData[i];
        const omx = 1 - x;
        if (omx === 0) {
          out[i] = coeffs[n];
          continue;
        }
        let acc = 0, binom = 1, xk = 1, omxn = Math.pow(omx, n);
        for (let k = 0; k <= n; k++) {
          acc += coeffs[k] * binom * xk * omxn;
          xk *= x;
          omxn /= omx;
          binom = binom * (n - k) / (k + 1);
        }
        out[i] = acc;
      }
    }
  } else { // stepwise
    const nBin = coeffs.length;
    if (edges.length !== nBin + 1) {
      throw new Error('stepwise: edges length must equal values length + 1');
    }
    const eLast = edges[nBin];
    for (let i = 0; i < N; i++) {
      const x = xData[i];
      if (x < edges[0] || x > eLast) { out[i] = NaN; continue; }
      let v = NaN;
      for (let b = 0; b < nBin; b++) {
        if (x >= edges[b]
            && (x < edges[b + 1] || (b === nBin - 1 && x === edges[b + 1]))) {
          v = coeffs[b];
          break;
        }
      }
      out[i] = v;
    }
  }
  // Same "kind as inputs" semantics: if xVal was a
  // Value, return a Value; if it was a bare Float64Array, return one.
  if (valueLib.isValue(xVal)) return valueLib.batchedScalar(out);
  return out;
}

function _perAtomFallback(ir: any, refArrays: any, N: any, baseEnv: any, overlay: any) {
  const refNames = refArrays ? Object.keys(refArrays) : [];
  const overlayKeys = overlay ? Object.keys(overlay) : null;
  // Fast path: nothing varies per atom AND overlay is empty → one shot.
  if (refNames.length === 0 && !overlayKeys) {
    return evaluateExpr(ir, baseEnv);
  }
  // Per-atom loop. Build a callEnv that overlays per-atom refArrays
  // values + the overlay (overlay last so it wins over refArrays).
  const callEnv = Object.assign({}, baseEnv);
  if (overlayKeys) for (let j = 0; j < overlayKeys.length; j++) {
    callEnv[overlayKeys[j]] = overlay[overlayKeys[j]];
  }
  // Pre-compute per-ref accessors. refArrays uniformly carry Values
  // internally; bare Float64Array entries are accepted as a back-
  // compat path for tests that pass refArrays directly.
  // shape=[N] (scalar atoms) → `i => data[i]`; shape=[N,
  // ...rest] (vector / matrix atoms) → atom i is a length-prod(rest)
  // sub-Value (subarray view). Computing the access pattern once per
  // ref keeps the inner N-atom loop branch-free.
  const accessors = new Array(refNames.length);
  for (let j = 0; j < refNames.length; j++) {
    const k = refNames[j];
    const v = refArrays[k];
    if (valueLib.isValue(v)) {
      const shape = v.shape;
      const data = v.data;
      if (shape.length === 1) {
        accessors[j] = (i: any) => data[i];
      } else {
        const tailDims = shape.slice(1);
        const tailLen = tailDims.reduce((a: any, b: any) => a * b, 1);
        accessors[j] = (i: any) => ({
          shape: tailDims,
          data: data.subarray(i * tailLen, (i + 1) * tailLen),
        });
      }
    } else {
      // Back-compat: refArrays passed directly as a Float64Array (by
      // tests / external callers that haven't migrated to Values).
      const arr = v;
      accessors[j] = (i: any) => arr[i];
    }
  }
  const out = new Array(N);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < refNames.length; j++) {
      const k = refNames[j];
      // overlay wins over refArrays — skip the write when overlay has it.
      if (overlayKeys && Object.prototype.hasOwnProperty.call(overlay, k)) continue;
      callEnv[k] = accessors[j](i);
    }
    out[i] = evaluateExpr(ir, callEnv);
  }
  // Pack to Float64Array if every result is numeric/boolean (the
  // common case for non-scalar ops whose outputs happen to be scalar,
  // e.g. polynomial / get_field returning numbers).
  let allNumeric = true;
  for (let i = 0; i < N; i++) {
    const t = typeof out[i];
    if (t !== 'number' && t !== 'boolean') { allNumeric = false; break; }
  }
  if (allNumeric) {
    const arr = new Float64Array(N);
    for (let i = 0; i < N; i++) arr[i] = +out[i];
    return arr;
  }
  // Pack to a complex Value (shape=[N], dtype='complex') when every
  // result is a scalar complex `{re, im}` — the per-atom return of
  // user-defined functions that compute complex arithmetic (e.g. a
  // std-module function like `resonance_breitwigner` returning the
  // BW amplitude). Without this packing, downstream ARITH_OPS_N
  // entries (mul / add) see a JS array of complex objects and emit
  // NaN. The canonical complex Value shape (engine-concepts §2.1)
  // comes from `valueLib.complexValue`.
  let allComplex = true;
  for (let i = 0; i < N; i++) {
    const v = out[i];
    if (!v || typeof v !== 'object'
        || typeof v.re !== 'number' || typeof v.im !== 'number') {
      allComplex = false; break;
    }
  }
  if (allComplex) {
    const re = new Float64Array(N);
    const im = new Float64Array(N);
    for (let i = 0; i < N; i++) { re[i] = out[i].re; im[i] = out[i].im; }
    return valueLib.complexValue(re, im, [N]);
  }
  // Per-atom uniform-length numeric array outputs (e.g.
  // softmax / l1unit / l2unit / logsoftmax on per-atom inputs) pack
  // atom-major into a Value shape=[N, k]. Detect uniform shape across
  // atoms; non-uniform results stay as a JS array of per-atom values
  // (the caller chooses how to surface them).
  let allUniformArrays = true;
  const sample0 = out[0];
  let k0 = 0;
  if (Array.isArray(sample0) || (sample0 && sample0.BYTES_PER_ELEMENT !== undefined)) {
    k0 = sample0.length;
  } else if (valueLib.isValue(sample0) && sample0.shape.length === 1) {
    k0 = sample0.shape[0];
  } else {
    allUniformArrays = false;
  }
  for (let i = 0; allUniformArrays && i < N; i++) {
    const s = out[i];
    let len;
    if (Array.isArray(s) || (s && s.BYTES_PER_ELEMENT !== undefined)) {
      len = s.length;
    } else if (valueLib.isValue(s) && s.shape.length === 1) {
      len = s.shape[0];
    } else {
      allUniformArrays = false; break;
    }
    if (len !== k0) { allUniformArrays = false; break; }
    // Also require numeric entries throughout.
    for (let j = 0; allUniformArrays && j < k0; j++) {
      const e = valueLib.isValue(s) ? s.data[j] : s[j];
      if (typeof e !== 'number' && typeof e !== 'boolean') {
        allUniformArrays = false; break;
      }
    }
  }
  if (allUniformArrays && k0 > 0) {
    const data = new Float64Array(N * k0);
    for (let i = 0; i < N; i++) {
      const s = out[i];
      const src = valueLib.isValue(s) ? s.data : s;
      const base = i * k0;
      for (let j = 0; j < k0; j++) data[base + j] = +src[j];
    }
    return { shape: [N, k0], data };
  }
  return out;
}

module.exports = {
  evaluateExprN,
  _evalN,
  _batchedApproximation,
  _perAtomFallback,
  ARITH_OPS_N,
  isBatch,
  initARITHOPSN,
  _SCALAR_PRIM_ARITY,
  broadcast1,
  broadcast2,
  broadcast3,
};
