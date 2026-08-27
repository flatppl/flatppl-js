'use strict';
// crn-normalize — a θ-DEPENDENT normalizer for
// `normalize(weighted(f, Lebesgue(support = S)))` built by reweighting ONE
// FIXED base sample (common random numbers).
//
// WHY THIS EXISTS. Spec §06 "Density reweighting" makes this measure canonical:
// "`normalize(weighted(f, Lebesgue(support = S)))` produces a probability
// distribution whose density w.r.t. Lebesgue on S is proportional to f". Its
// density needs Z = ∫_S f dx, and §06 "Density of composed measures" fixes the
// shift: "`normalize` (from M / Z): logdensityof(normalize(M), x) =
// logdensityof(M, x) − log Z, with Z = totalmass(M) finite and nonzero".
//
// When f references a LATENT, Z is a function of θ. Two exact routes already
// cover parts of that surface: `normalize-mass.totalMassExpr` (a superpose of
// weighted probability measures — Z is algebraic in θ) and mat-density's
// `weightedLebesgueQuadLogZ` (a θ-INDEPENDENT weight — Z is one deterministic
// quadrature). Everything else fell through to materialising the inner measure
// once and baking a CONSTANT −log Z, which is correct only when Z does not move
// with θ. For a non-bilinear parameter — a width, a mass, a slope inside an
// exponential — it is silently wrong at every θ but the materialised one.
//
// THE ESTIMATOR. Draw M points x_1..x_M from the box ONCE, then
//
//     Ẑ(θ) = (V / M) · Σ_m f(x_m; θ),      V = ∏ (hi_i − lo_i)
//
// emitted as an IR EXPRESSION in θ, so the density walker evaluates it at the
// scored point exactly as it evaluates `totalMassExpr`'s algebraic mass. Ẑ is
// an unbiased estimator of Z. One pass over the M points per θ, and no
// materialisation.
//
// WHY THE SAMPLE IS FIXED, AND WHERE IT LIVES. The point set is generated at
// REWRITE time and BAKED INTO THE IR as literals, from a Philox key seeded by
// the BASE MEASURE alone — the box axes and the point count, never the session
// seed, the sample count, the weight, or a clock. Four consequences, each of
// which is the reason for the choice rather than a side effect:
//
//   1. Ẑ(θ) is a DETERMINISTIC function of θ, so the engine scores an
//      exactly-defined surrogate measure rather than a noisy one. The
//      likelihood surface is smooth in θ (correlated estimates — the common
//      random numbers), and MH over that surrogate is ordinary MCMC for the
//      surrogate posterior, with no noisy-acceptance pathology.
//   2. The IS route (mat-density) and the MH route (mcmc-density) build the
//      SAME expression for the same model, so the two routes of one measure
//      cannot disagree — the failure class measure-algebra-audit.md exists for.
//   3. Two SPELLINGS of one measure agree bit-for-bit. Seeding off the weight
//      body instead broke exactly this: a lambda weight and the equivalent
//      `functionof` reification differ in parameter names, so they drew
//      different samples and scored the same measure differently.
//   4. Spec §06 "Reproducibility" is satisfied by construction: "An engine may
//      compute a density by any method the reductions above admit, stochastic
//      methods included, provided the value is reproducible with respect to
//      that engine: the same query, on the same implementation and hardware,
//      yields the same value."
//
// A θ-INDEPENDENT weight never reaches here — the CALLER gates on that, so the
// deterministic quadrature and the pre-existing N-D box estimator keep every
// number they already produce.
//
// THE CAVEAT, WHICH IS A BIAS AND NOT NOISE. Ẑ is unbiased for Z, but the
// density needs −log Ẑ, and E[−log Ẑ] > −log Z by Jensen. The plug-in density
// is therefore biased, and a posterior sampled against it is an APPROXIMATE
// posterior, not the exact one. Exact treatment of a normalizer that is
// intractable per θ is the pseudo-marginal / exchange family (Andrieu & Roberts
// 2009; Murray, Ghahramani & MacKay 2006) and is explicitly OUT OF SCOPE here:
// those algorithms need an unbiased estimate of the target itself, whereas Z
// sits in the DENOMINATOR, and they replace the inference kernel rather than
// the density. `crnNormalizeNote` states the caveat once per node.

const rng = require('./rng.ts');
const { mapIR } = require('./ir-walk.ts');

// Points in the fixed base sample. The estimator's error is O(M^-1/2) with the
// stratification below, and the emitted expression holds M copies of the weight
// body, so this trades accuracy against IR size.
const CRN_POINTS_DEFAULT = 128;

// Refuse rather than emit an expression bigger than this. A weight body large
// enough to blow the budget is a real feature request (a sum over the sample
// axis instead of M inlined copies), not something to silently shrink M for.
const CRN_NODE_BUDGET = 400000;

// =====================================================================
// Shape recognition
// =====================================================================

// Unwrap the reference-measure slot of a `weighted` to the "Lebesgue over a
// box" reading, returning its support set IR. Three spellings denote it, and
// all three have effective base density ≡ 1, so Z = ∫_S f dx with no extra
// factor:
//   • `Lebesgue(support = S)` — written directly.
//   • `logweighted(log(b−a), Uniform(interval(a,b)))` — the #307 shift that
//     scales Uniform's 1/(b−a) back to 1.
//   • `Uniform(interval(a, b))` with b − a = 1 — the lowering's unshifted form,
//     where Uniform's density is already 1.
// A NON-unit bare `Uniform` is declined: its density is 1/(b−a), so reading it
// as Lebesgue would drop that factor.
function lebesgueBoxSupport(refIR: any): any | null {
  let ref = refIR;
  let unwrappedShift = false;
  if (ref && ref.kind === 'call' && ref.op === 'logweighted'
      && Array.isArray(ref.args) && ref.args.length === 2) {
    ref = ref.args[1];
    unwrappedShift = true;
  }
  if (!ref || ref.kind !== 'call') return null;
  const supp = (ref.kwargs && ref.kwargs.support)
    || (Array.isArray(ref.args) ? ref.args[0] : null);
  if (!supp) return null;
  if (ref.op === 'Lebesgue') return supp;
  if (ref.op !== 'Uniform') return null;
  if (unwrappedShift) return supp;
  // Bare Uniform: only the unit interval reads as Lebesgue.
  const ax = _parseAxes(supp);
  if (!ax || ax.length !== 1 || ax[0].kind !== 'finite') return null;
  return Math.abs((ax[0].hi - ax[0].lo) - 1) < 1e-12 ? supp : null;
}

// Axis list for a support set IR, via mat-density's parseTruncationBox — one
// reader for the truncation-region and the CRN paths, so a shape one accepts
// cannot be a shape the other mis-parses. Required lazily: mat-density
// require()s this module.
function _parseAxes(setIR: any): any[] | null {
  const { parseTruncationBox } = require('./mat-density.ts');
  return parseTruncationBox(setIR);
}

type CrnShape = {
  weightParams: string[];
  weightBody: any;
  axes: Array<{ lo: number; hi: number; kind: string }>;
};

// Recognise `normalize(weighted(<functionof weight>, <Lebesgue over a
// finite box>))` and return its weight function plus the box axes. null for
// every other shape, which leaves the caller's existing fallback in place.
function crnRecognize(node: any): CrnShape | null {
  if (!node || node.kind !== 'call' || node.op !== 'normalize'
      || !Array.isArray(node.args) || node.args.length !== 1) return null;
  const inner = node.args[0];
  if (!inner || inner.kind !== 'call' || inner.op !== 'weighted'
      || !Array.isArray(inner.args) || inner.args.length !== 2) return null;
  const fn = inner.args[0];
  if (!fn || fn.kind !== 'call' || fn.op !== 'functionof'
      || !Array.isArray(fn.params) || fn.params.length < 1 || !fn.body) return null;
  const supp = lebesgueBoxSupport(inner.args[1]);
  if (!supp) return null;
  const axes = _parseAxes(supp);
  if (!axes || axes.length === 0) return null;
  // Every axis must be bounded: the points are drawn uniformly over the box.
  // An unbounded axis would need a change-of-variables whose Jacobian is
  // heavy-tailed, which is a high-variance estimator this does not ship.
  for (const a of axes) if (a.kind !== 'finite') return null;
  // §06 weight arity: "A one-parameter weight receives the variate whole. If
  // the variate is a k-element array with k >= 2, a weight of exactly k scalar
  // parameters instead receives one component per parameter, in order; any
  // other arity is an error."
  if (fn.params.length !== 1 && fn.params.length !== axes.length) return null;
  return { weightParams: fn.params.slice(), weightBody: fn.body, axes };
}

// Does the weight actually depend on a latent? Only then is a per-θ normalizer
// needed, and only then may this path take a node away from the deterministic
// quadrature or the pre-existing N-D box estimator.
//
// The weight's OWN parameters must come out first. mat-density's
// `weightedFixedWeightEnv` does not remove them, which is harmless where it is
// used (a lambda's parameter is `%local`, so `collectSelfRefs` cannot see it)
// and wrong here: a REIFIED weight — `functionof(body, m = m, c = c)` — carries
// its parameters as the BOUNDARY BINDING names in `self`, so every reified
// weight looked latent-dependent and pulled a θ-independent 2-D box onto this
// path, changing numbers that were already right.
function crnWeightIsThetaDependent(shape: CrnShape, ctx: any): boolean {
  const orchestrator = require('./orchestrator.ts');
  const { isFunctionLikeBinding } = require('./materialiser-shared.ts');
  const params = new Set(shape.weightParams);
  for (const n of orchestrator.collectSelfRefs(shape.weightBody)) {
    if (params.has(n)) continue;                                  // the variate
    const b = ctx && ctx.bindings && ctx.bindings.get(n);
    if (isFunctionLikeBinding(b)) continue;                        // a called fn
    if (ctx && ctx.fixedValues && ctx.fixedValues.has(n)) continue; // a constant
    return true;
  }
  return false;
}

// =====================================================================
// The fixed point set
// =====================================================================

// Seed string for the point set. It names the BASE MEASURE (the box axes) and
// the point count, and DELIBERATELY NOT the weight: the sample is a sample of
// the base, so the weight has no business choosing it. Two consequences the
// tests pin:
//   • Two spellings of one measure — a lambda weight and the equivalent
//     `functionof` reification, whose bodies differ in parameter names and
//     structure — draw the SAME points and therefore score BIT-FOR-BIT alike.
//   • The IS and MH routes seed identically, since neither the source span nor a
//     route-local annotation can reach this string.
function seedString(axes: Array<{ lo: number; hi: number }>, M: number): string {
  return 'crn|M=' + M + '|' + axes.map((a) => a.lo + ':' + a.hi).join('x');
}

// M points over the box, as a latin-hypercube sample with one jittered point
// per stratum: axis j takes stratum perm_j[m], and axis 0 keeps the identity
// permutation (LHS needs the permutations to be independent ACROSS axes, not
// randomized on every one). Unbiased for the integral, and strictly lower
// variance than an unstratified sample; in 1-D it reduces to plain
// stratification. Deterministic in `seed` alone.
function crnFixedPoints(
  axes: Array<{ lo: number; hi: number }>, M: number, seed: number,
): number[][] {
  const k = axes.length;
  const key = rng.keyFromSeed(seed >>> 0);
  const state = rng.stateFromKey(key[0], key[1]);
  // One stream for every uniform this function needs: M·k jitters first, then
  // the (k−1)·M swap draws for the permutations.
  const nJitter = M * k;
  const nSwap = (k - 1) * M;
  const { out } = rng.philoxNUniform(state, nJitter + nSwap);
  const perms: number[][] = [];
  for (let j = 0; j < k; j++) {
    const p = new Array(M);
    for (let m = 0; m < M; m++) p[m] = m;
    if (j > 0) {
      // Fisher-Yates over p, drawing from the swap segment of the stream.
      const base = nJitter + (j - 1) * M;
      for (let m = M - 1; m > 0; m--) {
        const r = Math.min(m, Math.floor(out[base + m] * (m + 1)));
        const t = p[m]; p[m] = p[r]; p[r] = t;
      }
    }
    perms.push(p);
  }
  const pts: number[][] = new Array(M);
  for (let m = 0; m < M; m++) {
    const row = new Array(k);
    for (let j = 0; j < k; j++) {
      const u = (perms[j][m] + out[m * k + j]) / M;
      row[j] = axes[j].lo + u * (axes[j].hi - axes[j].lo);
    }
    pts[m] = row;
  }
  return pts;
}

// =====================================================================
// The emitted expression
// =====================================================================

// Bind the weight function's parameters to one sample point, following the
// SAME convention density.ts's walkWeighted and mat-density's makeIntegrandND
// use — a k-parameter weight takes one coordinate per parameter in axis order,
// a one-parameter weight over a k-axis box takes the whole k-vector — so the
// normalizer integrates the function the density path scores.
function bindWeightAt(params: string[], body: any, pt: number[]): any {
  const bound: Record<string, any> = {};
  if (params.length === pt.length && pt.length > 1) {
    for (let i = 0; i < params.length; i++) bound[params[i]] = { kind: 'lit', value: pt[i] };
  } else if (pt.length > 1) {
    bound[params[0]] = {
      kind: 'call', op: 'vector',
      args: pt.map((c) => ({ kind: 'lit', value: c })),
    };
  } else {
    bound[params[0]] = { kind: 'lit', value: pt[0] };
  }
  return mapIR(body, (n: any) => (n && n.kind === 'ref' && n.name in bound ? bound[n.name] : n));
}

// Σ terms as a BALANCED binary tree of `add`, not a left-leaning chain. Depth
// log₂M rather than M: the engine's IR walkers (collectRefNames, mapIR, the
// worker's evaluator) all recurse, and a chain of M = 2048 terms overflowed the
// stack — which `likWith` then swallowed to −∞, so the model reported a constant
// chain rather than an error. Pairwise summation is also the more accurate
// reduction.
function balancedAdd(terms: any[]): any {
  let level = terms;
  while (level.length > 1) {
    const next: any[] = [];
    for (let i = 0; i < level.length; i += 2) {
      next.push(i + 1 < level.length
        ? { kind: 'call', op: 'add', args: [level[i], level[i + 1]] }
        : level[i]);
    }
    level = next;
  }
  return level[0];
}

function nodeCount(n: any): number {
  if (n == null || typeof n !== 'object') return 0;
  let c = 1;
  for (const k in n) { const v = n[k]; if (v && typeof v === 'object') c += nodeCount(v); }
  return c;
}

const _noted = new Set<string>();

// State the plug-in caveat once per distinct node. Not an error: the density is
// well defined and reproducible, it is the POSTERIOR it induces that is
// approximate, and the user is the one who can judge whether that matters.
function crnNormalizeNote(key: string, M: number, dims: number) {
  if (_noted.has(key)) return;
  // The dedup set is process-global; clear it rather than grow without bound in
  // a long-lived session that scores many models.
  if (_noted.size > 256) _noted.clear();
  _noted.add(key);
  // eslint-disable-next-line no-console
  console.warn('normalize(weighted(f, Lebesgue(box))): the weight depends on a latent, '
    + 'so Z(θ) is estimated by reweighting one fixed ' + M + '-point sample of the '
    + dims + '-D box (common random numbers). The estimate is deterministic in θ and '
    + 'reproducible, but −log Ẑ is biased for −log Z (Jensen), so a posterior scored '
    + 'against it is an APPROXIMATE posterior. Exact treatment (pseudo-marginal / '
    + 'exchange) is out of scope for this engine.');
}

// Build the IR expression for Ẑ(θ) of a `normalize` node, or null when the node
// is not the recognised shape (the caller keeps its existing fallback).
// `opts.points` overrides M.
function crnNormalizeMassExpr(node: any, opts?: any): any | null {
  const shape = crnRecognize(node);
  if (!shape) return null;
  const M = Math.max(1, Math.floor((opts && opts.points) || CRN_POINTS_DEFAULT));
  const key = seedString(shape.axes, M);
  const bodySize = nodeCount(shape.weightBody);
  if (bodySize * M > CRN_NODE_BUDGET) {
    // eslint-disable-next-line no-console
    console.warn('normalize(weighted(f, Lebesgue(box))): the weight body is '
      + bodySize + ' nodes, too large to inline ' + M + ' times for the fixed-sample '
      + 'normalizer (budget ' + CRN_NODE_BUDGET + '); falling back to a θ-independent '
      + 'constant Z, which is wrong wherever Z moves with θ');
    return null;
  }
  const seed = rng.xxhash32(key, 0) >>> 0;
  const pts = crnFixedPoints(shape.axes, M, seed);
  let volume = 1;
  for (const a of shape.axes) volume *= (a.hi - a.lo);
  const terms = pts.map((pt) => bindWeightAt(shape.weightParams, shape.weightBody, pt));
  const sum = balancedAdd(terms);
  crnNormalizeNote(key, M, shape.axes.length);
  return {
    kind: 'call', op: 'mul',
    args: [{ kind: 'lit', value: volume / M }, sum],
  };
}

module.exports = {
  crnNormalizeMassExpr,
  crnRecognize,
  crnWeightIsThetaDependent,
  crnFixedPoints,
  lebesgueBoxSupport,
  seedString,
  CRN_POINTS_DEFAULT,
  CRN_NODE_BUDGET,
  _internal: { bindWeightAt, nodeCount, balancedAdd, _noted },
};
