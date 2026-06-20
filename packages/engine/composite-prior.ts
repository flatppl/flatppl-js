'use strict';
// composite-prior.ts — recognise a draw whose measure is a kernel-broadcast over
// a USER FUNCTION returning an iid block, e.g.
//
//   beta_row_K = (a_g, b_g) -> iid(Beta(a_g, b_g), N)
//   p ~ beta_row_K.(a, b)                       # a [G, N] matrix of Betas
//
// The forward sampler resolves this via the composite-body recogniser; the
// density side did not, so the per-draw prior scorer scored `p` as a single
// scalar Beta → −∞. This shared helper returns enough to (a) infer the
// elementwise support and total size, and (b) score the prior as a flat product
// of the inner built-in distribution with per-row params expanded across the
// inner iid axis. Returns null for any non-composite-iid draw.

const compositeBodies = require('./composite-body-recognizers.ts');
const matBroadcast    = require('./mat-broadcast.ts');

function recognizeCompositeIidDraw(name: string, ctx: any): any {
  const derivs = ctx.derivations || {};
  let cur = name; const seen = new Set<string>();
  while (cur && derivs[cur] && (derivs[cur].kind === 'alias' || derivs[cur].kind === 'normalize') && !seen.has(cur)) {
    seen.add(cur); cur = derivs[cur].from;
  }
  const d = derivs[cur];
  if (!d || d.kind !== 'kernelbroadcast') return null;
  let cb: any;
  try { cb = compositeBodies.tryRecognizeCompositeBody(d, ctx); } catch (_) { return null; }
  if (!cb || cb.kind !== 'iid') return null;

  // Surface kwargs: positional dot-call args (d.argIRs) map onto the kernel's
  // surface param names (cb.paramKwargs) in order; explicit kwargs win.
  const effBc: Record<string, any> = Object.assign({}, d.kwargIRs || {});
  const pk: string[] = cb.paramKwargs || [];
  if (Array.isArray(d.argIRs)) {
    for (let i = 0; i < d.argIRs.length && i < pk.length; i++) {
      if (!Object.prototype.hasOwnProperty.call(effBc, pk[i])) effBc[pk[i]] = d.argIRs[i];
    }
  }
  // Inner builtin's params with the kernel formals substituted to the outer
  // broadcast args — e.g. { alpha: <ref a>, beta: <ref b> }.
  const sub: Record<string, any> = {};
  for (const pn of cb.distParams) {
    if (cb.distKwargs[pn] !== undefined) {
      sub[pn] = matBroadcast.substituteKernelParams(cb.distKwargs[pn], cb.params, cb.paramKwargs, effBc);
    }
  }
  return { distOp: cb.distOp as string, D: cb.n as number, paramNames: cb.distParams.slice(), sub };
}

module.exports = { recognizeCompositeIidDraw };
