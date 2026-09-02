'use strict';
// leaf-params — bind a distribution leaf's call arguments to its DECLARED
// parameter names, whichever spelling the model used.
//
// WHY THIS EXISTS. §04's calling convention makes the keyword and positional
// spellings of one call the same call, and §08 fixes the binding order: "The
// names and order of the distribution parameters specified below define the
// names and positional order of the kernel arguments." A recogniser that reads
// `kwargs` alone therefore sees only half the language: it declines
// `Normal(0.0, 1.0)` while accepting `Normal(mu = 0.0, sigma = 1.0)`, so two
// spellings of ONE measure take different routes and disagree. Measured before
// this module, on four separate mass routes:
//
//   arm                              positional        keyword (= exact)
//   normalize(truncate(N(p,1), S))   refused           −0.78652606990187
//   §12 shared-variate product       refused           −1.05542925754757
//   normalize(weighted(w, N(0,1)))   −1.46354830996    −0.93893853320467
//   the same with a θ-dependent w    −1.69236065835    −1.04393852953800
//
// The two numeric rows are the dangerous ones: a wrong number, half a nat out,
// with no diagnostic.
//
// WHAT IT DOES NOT DO. No constant folding and no point evaluation — callers
// resolve the bound IR themselves, because they resolve against different
// things (a literal, a fixed point θ, a per-atom column). This module owns the
// BINDING rule alone, in one place, so the recognisers cannot drift on it.

// The binding precedence, mirroring `sampler-registry.resolveParams` /
// `resolveParamsN`: a keyword by declared name, then by alias, then the
// positional argument at the parameter's declared index.
//
// Returns `{ kernel, entry, paramIRs }` where `paramIRs[i]` is the IR bound to
// `entry.params[i]`, or null when nothing binds there. null for anything that
// is not a REGISTRY distribution call — the accepted set is the registry's own,
// which is also the key set `forward-cdf`, `inverse-cdf` and
// `density-prims.builtinLogdensityof` read.
function leafParamIRs(ir: any): { kernel: string; entry: any; paramIRs: any[] } | null {
  if (!ir || ir.kind !== 'call' || typeof ir.op !== 'string') return null;
  const samplerLib = require('./sampler.ts');
  if (!samplerLib.isKnownDistribution(ir.op)) return null;
  // `lookupDistribution` THROWS on a surplus keyword, which is §04's own static
  // error and belongs to the call site that RECOGNISES the distribution — not to
  // a recogniser whose contract is "a binding or null". Every consumer looks the
  // same IR up itself, so declining here changes only WHERE the error surfaces.
  let entry: any;
  try {
    entry = samplerLib.lookupDistribution(ir);
  } catch {
    return null;
  }
  const kwargs = ir.kwargs || {};
  const positional = Array.isArray(ir.args) ? ir.args : [];
  const paramIRs: any[] = [];
  for (let i = 0; i < entry.params.length; i++) {
    const name = entry.params[i];
    if (name in kwargs) { paramIRs.push(kwargs[name]); continue; }
    /* c8 ignore start -- no REGISTRY entry currently declares an alias, so this
       arm is unreachable today; it is kept because both registry resolvers
       consult aliases and this rule must not drift from theirs */
    const alias = entry.aliases && entry.aliases[name];
    if (alias && alias in kwargs) { paramIRs.push(kwargs[alias]); continue; }
    /* c8 ignore stop */
    paramIRs.push(i < positional.length ? positional[i] : null);
  }
  return { kernel: ir.op, entry, paramIRs };
}

module.exports = { leafParamIRs };
