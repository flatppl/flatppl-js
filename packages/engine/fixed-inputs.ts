'use strict';

// Fixed-data acceptance for the §03 vector sets, shared by module loading
// and host-supplied externals. Runtime in/density membership deliberately
// has a different contract: it does not check the linear equalities.
const valueSets = require('./value-set.ts');

/**
 * Resolve a declared vector set in its own namespace, including named
 * sets and supplied dimensions. Other domains need no vector validation.
 * An unresolved dimension still permits checking the vector's constraints;
 * a known dimension also constrains its structural length.
 */
function resolveFixedVectorSet(setIR: any,
  resolveValue: (ir: any) => any, resolveSet: (name: string) => any): any {
  const declared = valueSets.setExprValueset(setIR, undefined, resolveSet);
  if (!['stdsimplex', 'ordered', 'zerosum'].includes(declared?.vs)) return null;
  const domain = valueSets.setExprValueset(setIR, (ir: any) => {
    const v = resolveValue(ir);
    return typeof v === 'number' ? v : v?.shape?.length === 0 ? v.data[0] : null;
  }, resolveSet);
  return domain;
}

/** Validate concrete fixed data against a resolved §03 vector set. */
function validateFixedVectorInput(name: string, value: any, domain: any): void {
  if (!containsFixedVector(value, domain)) {
    throw new Error(`Fixed input '${name}' requires a value in ${valueSets.toSexpr(domain)}`);
  }
}

/**
 * |sum(x)-c| <= 8*n*u*sum(abs(x)), u = eps(source dtype)/2. The factor
 * eight allows a short producing chain per entry. Scale before summing
 * so finite data near overflow follows the same rule. Inequalities stay
 * exact; source f32 values may have been widened exactly for storage.
 */
function containsFixedVector(value: any, domain: any): boolean {
  const raw = value?.data ?? value;
  if (!Array.isArray(raw) && !ArrayBuffer.isView(raw)) return false;
  const data = raw as ArrayLike<number>;
  const shape = value?.shape ?? [data?.length];
  if (typeof data.length !== 'number' || shape.length !== 1 || shape[0] !== data.length
      || value?.dtype === 'complex'
      || (typeof domain.n === 'number' && data.length !== domain.n)) return false;
  let scale = 0;
  for (let i = 0; i < data.length; i++) {
    const x = data[i];
    if (typeof x !== 'number' || Number.isNaN(x)) return false;
    if (domain.vs === 'ordered') {
      if (i > 0 && !(data[i - 1] < x)) return false;
    } else {
      if (!Number.isFinite(x) || (domain.vs === 'stdsimplex' && x < 0)) return false;
      scale = Math.max(scale, Math.abs(x));
    }
  }
  if (domain.vs === 'ordered') return true;
  const target = domain.vs === 'stdsimplex' ? 1 : 0;
  if (scale === 0) return target === 0;
  let sum = 0, absSum = 0;
  for (let i = 0; i < data.length; i++) {
    const x = data[i] / scale;
    sum += x;
    absSum += Math.abs(x);
  }
  const epsilon = data instanceof Float32Array || value?.dtype === 'f32'
    ? 2 ** -23 : Number.EPSILON;
  return Math.abs(sum - target / scale) <= 8 * data.length * (epsilon / 2) * absSum;
}

module.exports = { resolveFixedVectorSet, validateFixedVectorInput };
