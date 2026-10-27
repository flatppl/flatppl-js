'use strict';
// Per-parameter finite-box integrals for §06 normalize. A fixed quadrature
// rule is safe only after its exactness is proved from the weight expression:
// even two agreeing rules can miss the same narrow smooth peak. Here tensor
// Gauss-Legendre integrates polynomials of degree at most nine per coordinate.
// Exponentials of affine expressions use their analytic integral instead.
// Unknown forms return null to the caller, which refuses the query.

const { GL2, GL3, GL5 } = require('./quadrature.ts');
const { mapIR } = require('./ir-walk.ts');
const lit = (value: number) => ({ kind: 'lit', value });
const call = (op: string, ...args: any[]) => ({ kind: 'call', op, args });

// Keep coefficient expressions small without cancelling terms whose values
// may be infinite or undefined. Only neutral literal operands disappear.
function binary(op: string, a: any, b: any): any {
  if (op === 'add' && a.kind === 'lit' && a.value === 0) return b;
  if ((op === 'add' || op === 'sub') && b.kind === 'lit' && b.value === 0) return a;
  if (op === 'mul' && a.kind === 'lit' && a.value === 1) return b;
  if ((op === 'mul' || op === 'divide') && b.kind === 'lit' && b.value === 1) return a;
  if (a.kind === 'lit' && b.kind === 'lit') {
    const value = op === 'add' ? a.value + b.value : op === 'sub' ? a.value - b.value
      : op === 'mul' ? a.value * b.value : a.value / b.value;
    if (Number.isFinite(value)) return lit(value);
  }
  return call(op, a, b);
}

/** Return log ∫box w and, for exponentials, a centered density expression. */
function boxLogMass(body: any, param: string, axes: Array<{ lo: number; hi: number }>,
  arrayVariate: boolean, nodeBudget: number): any | null {
  const dims = axes.length;
  const zero = () => Array(dims).fill(0);
  const mentions = (n: any): boolean => {
    if (!n || typeof n !== 'object') return false;
    if (Array.isArray(n)) return n.some(mentions);
    if (n.kind === 'ref' && n.name === param) return true;
    return Object.keys(n).some((k) => k !== 'loc' && k !== 'meta' && mentions(n[k]));
  };
  const coordinate = (n: any): number => {
    if (!arrayVariate) return n?.kind === 'ref' && n.name === param ? 0 : -1;
    if (n?.kind !== 'call' || n.op !== 'get' || n.args?.length !== 2
        || n.args[0]?.kind !== 'ref' || n.args[0].name !== param) return -1;
    const index = n.args[1];
    return index?.kind === 'lit' && Number.isInteger(index.value)
      && index.value >= 1 && index.value <= dims ? index.value - 1 : -1;
  };
  // Coefficients may depend on any parameters, but divisors and powers must
  // not introduce a non-polynomial dependence on the integration coordinates.
  const degree = (n: any): number[] | null => {
    if (!mentions(n)) return zero();
    const axis = coordinate(n);
    if (axis >= 0) { const d = zero(); d[axis] = 1; return d; }
    if (n.kind !== 'call' || !Array.isArray(n.args)) return null;
    const a = degree(n.args[0]);
    if (!a) return null;
    if (n.op === 'neg') return a;
    if (n.op === 'pow') {
      const p = n.args[1];
      return p?.kind === 'lit' && Number.isInteger(p.value) && p.value >= 0 && p.value <= 9
        ? a.map((d) => d * p.value) : null;
    }
    const b = degree(n.args[1]);
    if (!b) return null;
    if (n.op === 'add' || n.op === 'sub') return a.map((d, i) => Math.max(d, b[i]));
    if (n.op === 'mul') return a.map((d, i) => d + b[i]);
    if (n.op === 'divide' && b.every((d) => d === 0)) return a;
    return null;
  };
  // Structural affine coefficients avoid subtracting f(0) from f(e_i), which
  // can erase a small slope when the constant term is large.
  const affine = (n: any): any[] | null => {
    if (!mentions(n)) return [n, ...axes.map(() => lit(0))];
    const axis = coordinate(n);
    if (axis >= 0) return [lit(0), ...axes.map((_, i) => lit(i === axis ? 1 : 0))];
    if (n.kind !== 'call' || !Array.isArray(n.args)) return null;
    const a = affine(n.args[0]);
    if (!a) return null;
    if (n.op === 'neg') return a.map((c) => binary('sub', lit(0), c));
    if (n.op === 'divide' && !mentions(n.args[1])) {
      return a.map((c) => binary('divide', c, n.args[1]));
    }
    if (n.op === 'mul') {
      if (!mentions(n.args[1])) return a.map((c) => binary('mul', c, n.args[1]));
      if (!mentions(n.args[0])) {
        const b = affine(n.args[1]);
        return b && b.map((c) => binary('mul', n.args[0], c));
      }
    }
    if (n.op === 'add' || n.op === 'sub') {
      const b = affine(n.args[1]);
      return b && a.map((c, i) => binary(n.op, c, b[i]));
    }
    return null;
  };
  if (body.kind === 'call' && body.op === 'exp' && body.args?.length === 1) {
    const coeff = affine(body.args[0]);
    if (!coeff) return null;
    let logArea = lit(0);
    const corner: any[] = [];
    for (let i = 0; i < dims; i++) {
      const { lo, hi } = axes[i];
      const width = hi - lo;
      const slope = coeff[i + 1];
      const abs = call('abs', slope);
      const span = binary('mul', abs, lit(width));
      corner.push(call('ifelse', call('ge', slope, lit(0)), lit(hi), lit(lo)));
      // log ∫ exp(a*x) dx = max(a*lo,a*hi) + log(1-exp(-|a|*width)) - log|a|.
      // expm1 avoids cancellation near zero; the explicit limit also covers
      // an underflowed span. This stays finite when exp(peak) or span overflows.
      const area = call('ifelse', call('equal', span, lit(0)), lit(Math.log(width)),
        binary('sub', call('log', call('neg', call('expm1', call('neg', span)))), call('log', abs)));
      logArea = binary('add', logArea, area);
    }
    // Evaluate the original exponent at the maximizing corner. Expanding
    // t*(x-1) into -t+t*x would lose log(t) when t is large and the peak is 0.
    const point = arrayVariate ? call('vector', ...corner) : corner[0];
    const peak = mapIR(body.args[0], (n: any) => n?.kind === 'ref' && n.name === param ? point : n);
    // Subtract the peak before multiplying the slopes. Subtracting log Z from
    // a large uncentered exponent loses the small log-area correction, e.g.
    // exp(1e17*x) at x=1 must score log(1e17), not a rounded multiple of 16.
    let centered = lit(0);
    for (let i = 0; i < dims; i++) {
      const x = arrayVariate ? call('get', { kind: 'ref', ns: '%local', name: param }, lit(i + 1))
        : { kind: 'ref', ns: '%local', name: param };
      centered = binary('add', centered, binary('mul', coeff[i + 1], binary('sub', x, corner[i])));
    }
    return {
      logMass: binary('add', peak, logArea), logWeight: centered,
      // Cancel only a finite offset. Undefined source weights must still
      // reach the normalizer's nonfinite guard rather than acquire a density.
      logShift: call('ifelse', call('lt', call('abs', peak), lit(Infinity)), logArea, peak),
    };
  }
  const degrees = degree(body);
  if (!degrees || degrees.some((d) => d > 9)) return null;
  const rules = degrees.map((d) => d <= 1 ? [{ x: 0.5, w: 1 }] : d <= 3 ? GL2 : d <= 5 ? GL3 : GL5);
  // A narrow interval far from zero may have no representable interior nodes.
  // Then a polynomial-exact rule ceases to be exact: on [1e16,1e16+2] its
  // midpoint rounds to an endpoint. Check the mapped coordinate in unit-box
  // units, allowing only ordinary floating-point rounding (four ulps).
  if (rules.some((rule, i) => degrees[i] > 0 && rule.some((n: any) => {
    const { lo, hi } = axes[i];
    const x = lo + n.x * (hi - lo);
    return Math.abs((x - lo) / (hi - lo) - n.x) > 4 * Number.EPSILON;
  }))) return null;
  const { nodeCount, balancedAdd, bindWeightAt } = require('./crn-normalize.ts');
  const points = rules.reduce((n, r) => n * r.length, 1);
  if (nodeCount(body) * points > nodeBudget) return null;
  const terms: any[] = [];
  const integrate = (axis: number, point: number[], weight: number) => {
    if (axis === dims) {
      const bound = bindWeightAt([param], body, point, arrayVariate);
      terms.push(binary('mul', lit(weight), bound));
      return;
    }
    const { lo, hi } = axes[axis];
    for (const n of rules[axis]) integrate(axis + 1, point.concat(lo + n.x * (hi - lo)), weight * n.w);
  };
  integrate(0, [], 1);
  // Keep the box volume in log space so products of finite widths do not
  // overflow before the normalizer's logarithm is taken.
  const logVolume = axes.reduce((s, a) => s + Math.log(a.hi - a.lo), 0);
  return { logMass: binary('add', call('log', balancedAdd(terms)), lit(logVolume)) };
}

module.exports = { boxLogMass };
