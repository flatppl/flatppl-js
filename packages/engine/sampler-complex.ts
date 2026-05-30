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

// =====================================================================
// Complex trig + hyperbolic (closed-form via the real/imaginary
// decomposition; principal branches per spec §07)
// =====================================================================
//
//   sin(z) = sin(re)·cosh(im) + i·cos(re)·sinh(im)
//   cos(z) = cos(re)·cosh(im) − i·sin(re)·sinh(im)
//   tan(z) = sin(z) / cos(z)            (could overflow near poles;
//                                       use the safer formulation below)
//   sinh(z) = i^{-1} · sin(i · z)       (= -i · sin(iz))
//   cosh(z) =          cos(i · z)
//   tanh(z) = sinh(z) / cosh(z)
//
// Inverse trig functions (asin / acos / atan / asinh / acosh / atanh)
// derive from log via the standard identities; principal-branch
// convention.

export function _cSin(z: any) {
  return { re: Math.sin(z.re) * Math.cosh(z.im),
           im: Math.cos(z.re) * Math.sinh(z.im) };
}
export function _cCos(z: any) {
  return { re:  Math.cos(z.re) * Math.cosh(z.im),
           im: -Math.sin(z.re) * Math.sinh(z.im) };
}
export function _cTan(z: any) {
  // tan(z) = sin(z) / cos(z) via the cosh-stable formulation:
  //   tan(re + i·im) = (sin(2re) + i·sinh(2im)) / (cos(2re) + cosh(2im))
  // Numerically stable for large |im|.
  const r2 = 2 * z.re, i2 = 2 * z.im;
  const d = Math.cos(r2) + Math.cosh(i2);
  if (d === 0) return { re: NaN, im: NaN };
  return { re: Math.sin(r2) / d, im: Math.sinh(i2) / d };
}
export function _cSinh(z: any) {
  return { re: Math.sinh(z.re) * Math.cos(z.im),
           im: Math.cosh(z.re) * Math.sin(z.im) };
}
export function _cCosh(z: any) {
  return { re: Math.cosh(z.re) * Math.cos(z.im),
           im: Math.sinh(z.re) * Math.sin(z.im) };
}
export function _cTanh(z: any) {
  const r2 = 2 * z.re, i2 = 2 * z.im;
  const d = Math.cosh(r2) + Math.cos(i2);
  if (d === 0) return { re: NaN, im: NaN };
  return { re: Math.sinh(r2) / d, im: Math.sin(i2) / d };
}

// Inverse trig / hyperbolic — principal branch (spec §07).
// All derive from log:
//   asin(z)   = -i · log(i·z + sqrt(1 - z²))
//   acos(z)   = -i · log(z + i·sqrt(1 - z²))   = π/2 - asin(z)
//   atan(z)   =  i/2 · log((i + z) / (i - z))
//   asinh(z)  =  log(z + sqrt(z² + 1))
//   acosh(z)  =  log(z + sqrt(z² - 1))
//   atanh(z)  =  0.5 · log((1 + z) / (1 - z))
function _cReal(x: number) { return { re: x, im: 0 }; }

export function _cAsinh(z: any) {
  // log(z + sqrt(z² + 1))
  const z2 = _cMul(z, z);
  const s = _cSqrt(_cAdd(z2, _cReal(1)));
  return _cLog(_cAdd(z, s));
}
export function _cAcosh(z: any) {
  // log(z + sqrt(z² - 1)) — principal branch as spec §07.
  const z2 = _cMul(z, z);
  const s = _cSqrt(_cSub(z2, _cReal(1)));
  return _cLog(_cAdd(z, s));
}
export function _cAtanh(z: any) {
  // 0.5 · log((1 + z) / (1 - z))
  const num = _cAdd(_cReal(1), z);
  const den = _cSub(_cReal(1), z);
  return _cMul(_cReal(0.5), _cLog(_cDiv(num, den)));
}
export function _cAsin(z: any) {
  // asin(z) = -i · asinh(i·z)
  const iz = { re: -z.im, im: z.re };               // i·z = -im + i·re
  const r = _cAsinh(iz);
  return { re: r.im, im: -r.re };                    // -i · r
}
export function _cAcos(z: any) {
  // acos(z) = π/2 - asin(z)
  const a = _cAsin(z);
  return { re: Math.PI / 2 - a.re, im: -a.im };
}
export function _cAtan(z: any) {
  // atan(z) = -i · atanh(i·z)
  const iz = { re: -z.im, im: z.re };
  const r = _cAtanh(iz);
  return { re: r.im, im: -r.re };
}
