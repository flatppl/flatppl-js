'use strict';

// =====================================================================
// optimizer/coords.ts — value-set ↔ optimiser coordinate transforms
// =====================================================================
//
// The optimiser searches a normalised coordinate `z` (origin at the pivot,
// |z|≈1 ≙ one plot-scale displacement); the objective is evaluated in the
// model's original parameter space `x`. `makeCoords` builds the x↔z maps
// plus a feasibility `project` from each free input's value-set domain:
//
//   reals               → affine  x = x0 + z·s            (no projection)
//   interval(lo, hi)     → affine + box clamp on z         (boundary maxima
//                          reachable — we prefer real bounds to a logit map)
//   posreals/nonnegreals → log map x = exp(log x0 + z·s/x0) (stays > 0,
//                          unbounded above; nonneg's x=0 edge is asymptotic)
//
// Per-axis scale `s` comes from the plot range; a degenerate scale falls
// back to max(1, |x0|). Pure CJS leaf (no engine imports). Simplex / multi-
// variate axes are deferred (see TODO) — function/likelihood plot inputs
// are overwhelmingly scalar real / interval / positive.

function isLog(kind: string): boolean {
  return kind === 'posreals' || kind === 'nonnegreals';
}

/**
 * @param spec `{ domains: Domain[], scales: number[], x0: number[] }`,
 *   `Domain = {kind:'real'} | {kind:'interval',lo,hi} | {kind:'posreals'} | {kind:'nonnegreals'}`.
 * @returns `{ dim, toX(z), toZ(x), project(z), scales }`.
 */
function makeCoords(spec: any): any {
  const domains = spec.domains;
  const x0 = spec.x0;
  const n = domains.length;
  const rawScales = spec.scales || [];

  const s: number[] = [];
  const logOrigin: number[] = new Array(n).fill(0);
  const logScale: number[] = new Array(n).fill(1);
  for (let i = 0; i < n; i++) {
    const sc = rawScales[i];
    s[i] = Number.isFinite(sc) && sc > 0 ? sc : Math.max(1, Math.abs(x0[i]));
    if (isLog(domains[i].kind)) {
      const x0i = x0[i] > 0 ? x0[i] : Math.max(1e-6, s[i]);
      logOrigin[i] = Math.log(x0i);
      // A displacement s near x0 is s/x0 in log space (d log x = dx/x).
      logScale[i] = Math.max(1e-12, s[i] / x0i);
    }
  }

  function toX(z: number[]): number[] {
    const x = new Array(n);
    for (let i = 0; i < n; i++) {
      const d = domains[i];
      if (isLog(d.kind)) {
        x[i] = Math.exp(logOrigin[i] + z[i] * logScale[i]);
      } else {
        let xi = x0[i] + z[i] * s[i];
        if (d.kind === 'interval') {
          if (xi < d.lo) xi = d.lo; else if (xi > d.hi) xi = d.hi;
        }
        x[i] = xi;
      }
    }
    return x;
  }

  function toZ(x: number[]): number[] {
    const z = new Array(n);
    for (let i = 0; i < n; i++) {
      const d = domains[i];
      if (isLog(d.kind)) {
        const xi = x[i] > 0 ? x[i] : Math.exp(logOrigin[i]);
        z[i] = (Math.log(xi) - logOrigin[i]) / logScale[i];
      } else {
        z[i] = (x[i] - x0[i]) / s[i];
      }
    }
    return z;
  }

  function project(z: number[]): number[] {
    const out = z.slice();
    for (let i = 0; i < n; i++) {
      const d = domains[i];
      if (d.kind === 'interval') {
        const za = (d.lo - x0[i]) / s[i];
        const zb = (d.hi - x0[i]) / s[i];
        const a = Math.min(za, zb), b = Math.max(za, zb);
        if (out[i] < a) out[i] = a; else if (out[i] > b) out[i] = b;
      }
    }
    return out;
  }

  return { dim: n, toX, toZ, project, scales: s };
}

module.exports = { makeCoords };
