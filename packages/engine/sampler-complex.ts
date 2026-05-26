// =====================================================================
// sampler-complex.ts — scalar complex arithmetic primitives (spec §03/§07)
// =====================================================================
//
// Extracted from sampler.ts as part of the §17.5 sampler split
// (engine-concepts §11). Pure leaf module — no engine-internal deps.
//
// Scalar complex representation: a plain JS object `{re, im}` of two
// numbers. Operations auto-promote real → complex when meeting a
// complex operand. Storage is plain-object rather than a class so
// JSON round-trips work and structured clone preserves shape across
// postMessage.
//
// For atom-batched complex storage (a per-atom complex value), the
// design is: Value with `dtype: 'complex'`, `data: Float64Array(N)` for
// real parts, `im: Float64Array(N)` for imaginary parts (separate
// buffers, matching TF.js's complex-tensor representation). That
// extension lands when a concrete use case demands it; for now scalar
// complex values are sufficient for the spec §07 operator set.

export function _isComplex(v: any) {
  return v != null && typeof v === 'object'
    && typeof v.re === 'number' && typeof v.im === 'number';
}

export function _toComplex(v: any) {
  if (_isComplex(v)) return v;
  if (typeof v === 'number') return { re: v, im: 0 };
  if (typeof v === 'boolean') return { re: v ? 1 : 0, im: 0 };
  throw new Error('cannot promote ' + typeof v + ' to complex');
}

export function _cAdd(a: any, b: any) { return { re: a.re + b.re, im: a.im + b.im }; }
export function _cSub(a: any, b: any) { return { re: a.re - b.re, im: a.im - b.im }; }
export function _cNeg(a: any)    { return { re: -a.re,       im: -a.im }; }
export function _cMul(a: any, b: any) {
  return { re: a.re * b.re - a.im * b.im,
           im: a.re * b.im + a.im * b.re };
}
export function _cDiv(a: any, b: any) {
  const d = b.re * b.re + b.im * b.im;
  return { re: (a.re * b.re + a.im * b.im) / d,
           im: (a.im * b.re - a.re * b.im) / d };
}
export function _cConj(a: any)   { return { re: a.re, im: -a.im }; }
export function _cAbs(a: any)    { return Math.hypot(a.re, a.im); }
export function _cAbs2(a: any)   { return a.re * a.re + a.im * a.im; }
export function _cExp(z: any) {
  const r = Math.exp(z.re);
  return { re: r * Math.cos(z.im), im: r * Math.sin(z.im) };
}
export function _cLog(z: any) {
  // Principal branch: arg ∈ (-π, π] per spec §07.
  return { re: Math.log(Math.hypot(z.re, z.im)),
           im: Math.atan2(z.im, z.re) };
}
export function _cSqrt(z: any) {
  // Principal branch.
  if (z.im === 0 && z.re >= 0) return { re: Math.sqrt(z.re), im: 0 };
  const r = Math.hypot(z.re, z.im);
  // half-angle formulas avoiding catastrophic cancellation near the
  // negative real axis (preferred over re-deriving via _cExp(0.5·_cLog)).
  const reOut = Math.sqrt((r + z.re) / 2);
  const sign = z.im >= 0 ? 1 : -1;
  return { re: reOut, im: sign * Math.sqrt((r - z.re) / 2) };
}
export function _cPow(base: any, exp: any) {
  // z^w = exp(w · log(z)), principal branch (spec §07).
  return _cExp(_cMul(exp, _cLog(base)));
}
