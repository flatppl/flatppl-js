'use strict';

// Continuous reference supports (§03, §06). This resolver
// keeps support geometry separate from probability normalization and sampling.
// Numeric bounds may be fixed during planning or supplied by a density's env.
const { parseSetIR } = require('./ir-shared.ts');
const values = require('./value.ts');

type Support = { kind: string; [key: string]: any };

function supportArg(ir: any): any {
  return ir.kwargs?.support ?? ir.args?.[0];
}

/** Resolve the continuous subset of §03; null means unavailable geometry. */
function resolveSupport(
  ir: any, number: (ir: any) => number | null,
  reference: (ir: any) => any = () => null,
): Support | null {
  if (!ir) return null;
  if (ir.kind === 'ref') {
    const resolved = reference(ir);
    if (resolved && resolved !== ir) return resolveSupport(resolved, number, reference);
  }
  if (ir.kind === 'const' || ir.kind === 'ref') {
    const s = parseSetIR(ir, new Map());
    return s && ['reals', 'posreals', 'nonnegreals', 'interval'].includes(s.kind) ? s : null;
  }
  if (ir.kind !== 'call') return null;
  const args = ir.args || [];
  if (ir.op === 'interval' && args.length === 2) {
    const lo = number(args[0]), hi = number(args[1]);
    return lo == null || hi == null ? null : { kind: 'interval', lo, hi };
  }
  if (['stdsimplex', 'ordered', 'zerosum'].includes(ir.op)) {
    const n = number(args[0]);
    return n != null && Number.isInteger(n) && n >= 1 ? { kind: 'vectorset', set: ir.op, n } : null;
  }
  if (ir.op === 'cartpow') {
    const elem = resolveSupport(args[0], number, reference), n = number(args[1]);
    return elem && n != null && Number.isInteger(n) && n >= 0 ? { kind: 'power', elem, n } : null;
  }
  if (ir.op === 'cartprod') {
    const fields = ir.fields ?? Object.entries(ir.kwargs || {}).map(([name, value]) => ({ name, value }));
    if (fields.length) {
      const parts = fields.map((f: any) => ({ name: f.name, support: resolveSupport(f.value, number, reference) }));
      return parts.every((p: any) => p.support) ? { kind: 'record', fields: parts } : null;
    }
    const parts = args.map((a: any) => resolveSupport(a, number, reference));
    return parts.every((p: any) => p) ? { kind: 'product', parts } : null;
  }
  return null;
}

/** Log volume uses the free-coordinate reference, not embedded surface area. */
function logVolume(s: Support): number {
  if (s.kind === 'interval') return s.hi <= s.lo ? -Infinity : Math.log(s.hi - s.lo);
  if (s.kind === 'vectorset') {
    if (s.set === 'zerosum') return s.n === 1 ? 0 : Infinity;
    if (s.set === 'ordered') return Infinity;
    let logFactorial = 0;
    for (let k = 2; k < s.n; k++) logFactorial += Math.log(k);
    return -logFactorial;
  }
  if (s.kind === 'power') return s.n === 0 ? 0 : s.n * logVolume(s.elem);
  if (s.kind === 'product' || s.kind === 'record') {
    const terms = (s.parts ?? s.fields.map((f: any) => f.support)).map(logVolume);
    return terms.includes(-Infinity) ? -Infinity : terms.reduce((a: number, b: number) => a + b, 0);
  }
  return Infinity;
}

/** A product is flattened by cat (§03); a record retains its field structure. */
function width(s: Support): number {
  if (s.kind === 'vectorset') return s.n;
  if (s.kind === 'power') return s.n * width(s.elem);
  if (s.kind === 'product') return s.parts.reduce((n: number, p: Support) => n + width(p), 0);
  if (s.kind === 'record') throw new Error('reference support: a record support is not a positional vector');
  return 1;
}

/** Cartesian powers retain nesting; positional products concatenate entries. */
function axes(s: Support): number[] {
  if (s.kind === 'power') return [s.n, ...axes(s.elem)];
  if (s.kind === 'vectorset' || s.kind === 'product') return [width(s)];
  return [];
}

/** Density one on support. Affine vector sets reuse the engine's chart convention. */
function contains(s: Support, value: any): boolean {
  if (s.kind === 'record') return s.fields.every((f: any) => contains(f.support, value?.[f.name]));
  const data = values.isValue(value) ? value.data : Array.isArray(value) ? value.flat(Infinity) : value;
  if (s.kind === 'vectorset') {
    return data?.length === s.n && Array.from(data as ArrayLike<number>).every(Number.isFinite)
      && require('./sampler.ts').inVectorSet(data, s.set);
  }
  if (s.kind === 'power' || s.kind === 'product') {
    if (data?.length !== width(s)) return false;
    const parts = s.kind === 'power' ? Array(s.n).fill(s.elem) : s.parts;
    let offset = 0;
    return parts.every((p: Support) => {
      const n = width(p), part = data.slice(offset, offset + n);
      offset += n;
      return contains(p, ['vectorset', 'power', 'product'].includes(p.kind) ? part : part[0]);
    });
  }
  const x = typeof data === 'number' ? data : data?.length === 1 ? data[0] : NaN;
  if (!Number.isFinite(x)) return false;
  if (s.kind === 'posreals') return x > 0;
  if (s.kind === 'nonnegreals') return x >= 0;
  if (s.kind === 'interval') return x >= s.lo && x <= s.hi;
  return s.kind === 'reals';
}

/** Use the established scalar/vector charts; unsupported structured charts refuse. */
function transformSupport(s: Support): any {
  if (s.kind === 'reals') return { kind: 'real' };
  if (s.kind === 'posreals' || s.kind === 'nonnegreals') return { kind: 'positive' };
  if (s.kind === 'interval') {
    if (Number.isFinite(s.lo) && Number.isFinite(s.hi)) return { kind: 'interval', a: s.lo, b: s.hi };
    if (Number.isFinite(s.lo)) return { kind: 'greaterThan', lo: s.lo };
    if (Number.isFinite(s.hi)) return { kind: 'lessThan', hi: s.hi };
    return { kind: 'real' };
  }
  if (s.kind === 'vectorset') return { kind: s.set === 'stdsimplex' ? 'simplex' : s.set, n: s.n };
  if (s.kind === 'power' && width(s.elem) === 1 && s.elem.kind !== 'vectorset') {
    return { ...transformSupport(s.elem), n: s.n };
  }
  if (s.kind === 'product' && s.parts.every((p: Support) => width(p) === 1 && p.kind !== 'vectorset')) {
    return { kind: 'product', elements: s.parts.map(transformSupport), n: s.parts.length };
  }
  return { kind: 'unsupported', distOp: 'Lebesgue with structured support' };
}

module.exports = { supportArg, resolveSupport, logVolume, width, axes, contains, transformSupport };
