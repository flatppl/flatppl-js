'use strict';

// Gauss–Kronrod 7/15. Splitting the initial domain limits oscillations per
// panel. The local error includes a roundoff floor; failure never passes as
// a converged value. Nodes and weights are the standard 15-point rule.
const X = [0.9914553711208126, 0.9491079123427585, 0.8648644233597691,
  0.7415311855993945, 0.5860872354676911, 0.4058451513773972, 0.2077849550078985, 0];
const W = [0.02293532201052922, 0.06309209262997855, 0.1047900103222502,
  0.1406532597155259, 0.1690047266392679, 0.1903505780647854, 0.2044329400752989, 0.2094821410847278];
const G = [0.1294849661688697, 0.2797053914892767, 0.3818300505051189, 0.4179591836734694];

function integral(f: (x: number) => number, a: number, b: number, panels = 1,
    rtol = 2e-13, abstol = 2e-15): number {
  function panel(lo: number, hi: number, depth: number, atol: number): number {
    const mid = (lo + hi) / 2, half = (hi - lo) / 2, center = f(mid);
    let fine = W[7] * center, coarse = G[3] * center, magnitude = W[7] * Math.abs(center);
    for (let k = 0; k < 7; k++) {
      const left = f(mid - half * X[k]), right = f(mid + half * X[k]);
      fine += W[k] * (left + right); magnitude += W[k] * (Math.abs(left) + Math.abs(right));
      if (k % 2 === 1) coarse += G[(k - 1) / 2] * (left + right);
    }
    fine *= half; coarse *= half; magnitude *= half;
    const tolerance = Math.max(atol, rtol * Math.abs(fine), 32 * Number.EPSILON * magnitude);
    if (Math.abs(fine - coarse) <= tolerance) return fine;
    if (depth === 0) throw new Error('Scalar quadrature did not converge');
    return panel(lo, mid, depth - 1, atol / 2) + panel(mid, hi, depth - 1, atol / 2);
  }
  let sum = 0;
  for (let k = 0; k < panels; k++) {
    sum += panel(a + (b - a) * k / panels, a + (b - a) * (k + 1) / panels, 20, abstol / panels);
  }
  return sum;
}

module.exports = { integral };
