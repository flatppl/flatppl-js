'use strict';

const { isKnownName, MEASURE_PRODUCING, DISTRIBUTIONS } = require('./builtins.ts');
const AST = require('./ast.ts');
// Lazy require to avoid a circular load (disintegrate requires analyzer).
let _disintegratePlan: ((...a: any[]) => any) | null = null;
function disintegratePlan(...args: any[]) {
  if (!_disintegratePlan) _disintegratePlan = require('./disintegrate.ts').disintegratePlan;
  return _disintegratePlan!(...args);
}

/**
 * Determine whether an expression produces a measure (probability measure
 * or general measure) — as opposed to a value, kernel, or function.
 *
 * Measures come from: `lawof(...)`, distribution constructors (Normal, ...),
 * and measure-algebra ops that combine measures (iid, joint, chain, ...).
 *
 * NB: kernels (kernelof, functionof on a measure) are NOT measures — they
 * are functions returning measures.
 *
 * @param {object} node - AST expression
 * @param {Map} bindings - bindings map (for Identifier resolution)
 * @param {Set} [seen] - cycle guard
 */
function isMeasureExpr(node: any, bindings: any, seen?: Set<string>): boolean {
  if (!node) return false;
  if (!seen) seen = new Set();
  switch (node.type) {
    case 'Identifier': {
      const name = node.name;
      if (seen.has(name)) return false;
      seen.add(name);
      const b = bindings.get(name);
      if (!b) return false;
      if (b.type === 'lawof') return true;
      // 'call'-type bindings can be measure-typed (e.g., theta_dist = Normal(...)).
      if (b.type === 'call' && b.node && b.node.value) {
        return isMeasureExpr(b.node.value, bindings, seen);
      }
      return false;
    }
    case 'CallExpr': {
      if (!node.callee || node.callee.type !== 'Identifier') return false;
      const name = node.callee.name;
      if (name === 'lawof') return true;
      if (MEASURE_PRODUCING.has(name)) return true;
      // `broadcast` is dual (spec §04 "Stochastic broadcast"):
      // `broadcast(f, …)` over a value function is an array VALUE, but
      // `broadcast(K, …)` over a distribution/kernel is an array-valued
      // MEASURE — the independent product of the kernel applications.
      // It is therefore measure-producing iff the kernel arg (args[0])
      // is a distribution constructor or itself a measure-typed
      // expression (mirrors classifyKernelBroadcast). Without this an
      // `iid(Normal.(mu, sigma), L)` base fails resolveMeasureBaseName
      // and the whole iid binding goes unclassified.
      if (name === 'broadcast' && Array.isArray(node.args) && node.args.length >= 1) {
        const k = node.args[0];
        if (k && k.type === 'Identifier'
            && DISTRIBUTIONS.has(k.name) && !bindings.has(k.name)) {
          return true;
        }
        return isMeasureExpr(k, bindings, seen);
      }
      // `record(field = expr, ...)` with every field value resolving
      // to a draw / lawof / measure-typed call is a record-measure —
      // matches the engine's `classifyRecordOrJoint` derivation
      // (joint of the field measures). Without this, the complement-
      // route restrict's synthesised kernel body — `record(name =
      // variate_binding, ...)` — fails `resolveMeasureBaseName` and
      // the `logweighted(scalar, kernel(x))` wrapper produced by
      // `expandRestrictStatements` won't classify. The check
      // recurses through Identifier refs (chase a draw / lawof /
      // measure-typed call binding) and bottoms out on any non-
      // measure field value, so plain value records like
      // `default_pars = record(theta1 = 0.5, theta2 = 1.0)` still
      // return false.
      if (name === 'record' && Array.isArray(node.args) && node.args.length > 0) {
        for (const a of node.args) {
          if (!a || a.type !== 'KeywordArg' || !a.value) return false;
          const v = a.value;
          if (v.type === 'Identifier') {
            if (seen.has(v.name)) return false;
            const fb = bindings.get(v.name);
            if (!fb) return false;
            if (fb.type === 'draw' || fb.type === 'lawof') continue;
            if (!isMeasureExpr(v, bindings, seen)) return false;
            continue;
          }
          if (!isMeasureExpr(v, bindings, seen)) return false;
        }
        return true;
      }
      return false;
    }
    default:
      return false;
  }
}

/**
 * Classify a statement by examining the RHS expression.
 */
function classifyStatement(valueNode: any) {
  if (!valueNode) return 'call';

  if (valueNode.type === 'CallExpr' && valueNode.callee.type === 'Identifier') {
    const name = valueNode.callee.name;
    switch (name) {
      case 'draw': return 'draw';
      case 'elementof': return 'input';
      case 'external': return 'input';
      case 'lawof': return 'lawof';
      case 'functionof': return 'functionof';
      case 'kernelof': return 'kernelof';
      case 'fn': return 'fn';
      case 'bijection': return 'bijection';
      // fchain produces a function-valued binding (spec §04
      // "Function composition and annotation"; engine-concepts §19).
      // Distinct producer tag so the IR's op stays grep-able, but
      // consumers test via the function-like predicate, not by tag.
      case 'fchain': return 'fchain';
      case 'likelihoodof': return 'likelihood';
      case 'bayesupdate': return 'bayesupdate';
      case 'load_module': return 'module';
      case 'standard_module': return 'module';
      case 'load_data': return 'data';
    }
  }

  if (valueNode.type === 'ArrayLiteral' || valueNode.type === 'NumberLiteral'
      || valueNode.type === 'StringLiteral' || valueNode.type === 'TupleLiteral'
      || valueNode.type === 'BoolLiteral') {
    return 'literal';
  }

  return 'call';
}

/**
 * Validate argument structure of special operations.
 * Returns an array of diagnostics.
 */
function validateSpecialOperation(valueNode: any) {
  if (!valueNode || valueNode.type !== 'CallExpr') return [];
  if (!valueNode.callee || valueNode.callee.type !== 'Identifier') return [];

  const name = valueNode.callee.name;
  const args = valueNode.args;
  const diags: any[] = [];

  switch (name) {
    case 'functionof':
    case 'kernelof': {
      // First arg must be a positional expression, rest must be keyword args
      if (args.length === 0) {
        diags.push({ severity: 'error', message: `${name}() requires at least one argument`, loc: valueNode.loc });
        break;
      }
      if (args[0].type === 'KeywordArg') {
        diags.push({ severity: 'error', message: `First argument of ${name}() must be an expression, not a keyword argument`, loc: args[0].loc });
      }
      for (let i = 1; i < args.length; i++) {
        if (args[i].type !== 'KeywordArg') {
          diags.push({ severity: 'error', message: `Arguments after the first in ${name}() must be keyword boundary inputs (name = node)`, loc: args[i].loc });
        }
      }
      break;
    }
    case 'lawof': {
      // Unary: a single positional expression. Boundary keyword args were
      // moved to kernelof() — flag them with a migration hint.
      if (args.length === 0) {
        diags.push({ severity: 'error', message: `lawof() requires exactly one argument`, loc: valueNode.loc });
        break;
      }
      if (args[0].type === 'KeywordArg') {
        diags.push({ severity: 'error', message: `lawof() argument must be an expression, not a keyword argument`, loc: args[0].loc });
      }
      for (let i = 1; i < args.length; i++) {
        diags.push({
          severity: 'error',
          message: `lawof() takes a single argument; for a Markov kernel use kernelof(expr, ...keyword boundaries)`,
          loc: args[i].loc,
        });
      }
      break;
    }
    case 'fn': {
      // Single positional expression
      if (args.length !== 1) {
        diags.push({ severity: 'error', message: `fn() requires exactly one expression argument`, loc: valueNode.loc });
      } else if (args[0].type === 'KeywordArg') {
        diags.push({ severity: 'error', message: `fn() argument must be an expression, not a keyword argument`, loc: args[0].loc });
      }
      break;
    }
    case 'draw':
    case 'elementof':
    case 'external':
    case 'valueset': {
      // Single positional expression
      if (args.length !== 1) {
        diags.push({ severity: 'error', message: `${name}() requires exactly one argument`, loc: valueNode.loc });
      } else if (args[0].type === 'KeywordArg') {
        diags.push({ severity: 'error', message: `${name}() argument must be an expression, not a keyword argument`, loc: args[0].loc });
      }
      break;
    }
    case 'load_module': {
      // First arg must be a string, rest are optional keyword args
      if (args.length === 0) {
        diags.push({ severity: 'error', message: `load_module() requires a file path argument`, loc: valueNode.loc });
      } else if (args[0].type === 'KeywordArg') {
        diags.push({ severity: 'error', message: `First argument of load_module() must be a file path`, loc: args[0].loc });
      }
      for (let i = 1; i < args.length; i++) {
        if (args[i].type !== 'KeywordArg') {
          diags.push({ severity: 'error', message: `Arguments after the file path in load_module() must be keyword substitutions (name = value)`, loc: args[i].loc });
        }
      }
      break;
    }
    case 'standard_module': {
      // Two positional arguments: module name and version string
      if (args.length !== 2) {
        diags.push({ severity: 'error', message: `standard_module() requires exactly two arguments (name, version)`, loc: valueNode.loc });
      }
      for (const arg of args) {
        if (arg.type === 'KeywordArg') {
          diags.push({ severity: 'error', message: `standard_module() takes positional arguments only`, loc: arg.loc });
        }
      }
      break;
    }
    case 'load_data': {
      // source (positional or keyword) + valueset (keyword)
      if (args.length === 0) {
        diags.push({ severity: 'error', message: `load_data() requires source and valueset arguments`, loc: valueNode.loc });
      }
      break;
    }

    // ---- Measure-algebra binary ops: M and one other operand --------
    case 'weighted':
    case 'logweighted':
    case 'truncate':
    case 'pushfwd':
    case 'bayesupdate':
    case 'likelihoodof':
    case 'densityof':
    case 'logdensityof': {
      // Two positional args (weight/set/fn/likelihood/obs, measure).
      // Order varies per op but the shape (positional pair, no kwargs)
      // is uniform.
      if (args.length !== 2) {
        diags.push({ severity: 'error', message: `${name}() takes exactly two arguments`, loc: valueNode.loc });
      }
      for (const arg of args) {
        if (arg.type === 'KeywordArg') {
          diags.push({ severity: 'error', message: `${name}() takes positional arguments only`, loc: arg.loc });
        }
      }
      break;
    }

    case 'joint_likelihood': {
      // joint_likelihood(L1, L2, ...) — combine ≥2 likelihoods (spec §06).
      if (args.length < 2) {
        diags.push({ severity: 'error', message: `joint_likelihood() requires at least two arguments`, loc: valueNode.loc });
      }
      for (const arg of args) {
        if (arg.type === 'KeywordArg') {
          diags.push({ severity: 'error', message: `joint_likelihood() takes positional arguments only`, loc: arg.loc });
        }
      }
      break;
    }

    case 'totalmass': {
      // totalmass(M) — single positional measure.
      if (args.length !== 1) {
        diags.push({ severity: 'error', message: `totalmass() takes exactly one argument`, loc: valueNode.loc });
      } else if (args[0].type === 'KeywordArg') {
        diags.push({ severity: 'error', message: `totalmass() argument must be a positional expression`, loc: args[0].loc });
      }
      break;
    }

    // ---- Variadic-positional measure ops -----------------------------
    case 'superpose':
    case 'fchain': {
      // N components, all positional.
      if (args.length < 2) {
        diags.push({ severity: 'error', message: `${name}() requires at least two arguments`, loc: valueNode.loc });
      }
      for (const arg of args) {
        if (arg.type === 'KeywordArg') {
          diags.push({ severity: 'error', message: `${name}() takes positional arguments only`, loc: arg.loc });
        }
      }
      break;
    }

    // ---- Mixed positional/kwarg measure ops --------------------------
    case 'joint':
    case 'jointchain':
    case 'kchain': {
      // Two forms: all-positional or all-kwarg. Mixed is rejected so
      // the classifier sees one shape per call site.
      // `joint(x = M)` (1-arg kwarg form) is the canonical "relabel-as"
      // shape — accept it. jointchain/kchain need at least 2 components.
      const minArgs = (name === 'joint') ? 1 : 2;
      if (args.length < minArgs) {
        diags.push({ severity: 'error', message: `${name}() requires at least ${minArgs === 1 ? 'one argument' : 'two arguments'}`, loc: valueNode.loc });
        break;
      }
      const allKw  = args.every((a: any) => a.type === 'KeywordArg');
      const allPos = args.every((a: any) => a.type !== 'KeywordArg');
      if (!allKw && !allPos) {
        diags.push({
          severity: 'error',
          message: `${name}() arguments must be either all positional or all keyword (name = value), not mixed`,
          loc: valueNode.loc,
        });
      }
      break;
    }

    case 'iid': {
      // iid(M, n) or iid(M, dims...) — first arg is measure, rest are
      // integer dims. All positional.
      if (args.length < 2) {
        diags.push({ severity: 'error', message: `iid() requires at least two arguments (measure, dim)`, loc: valueNode.loc });
      }
      for (const arg of args) {
        if (arg.type === 'KeywordArg') {
          diags.push({ severity: 'error', message: `iid() takes positional arguments only`, loc: arg.loc });
        }
      }
      break;
    }

    // ---- Function-typed special ops ---------------------------------
    case 'bijection': {
      // bijection(f, f_inv, logvolume) — three positional args.
      if (args.length !== 3) {
        diags.push({
          severity: 'error',
          message: `bijection() takes exactly three arguments (f, f_inv, logvolume)`,
          loc: valueNode.loc,
        });
      }
      for (const arg of args) {
        if (arg.type === 'KeywordArg') {
          diags.push({ severity: 'error', message: `bijection() takes positional arguments only`, loc: arg.loc });
        }
      }
      break;
    }

    case 'broadcasted': {
      // broadcasted(f) — curried single-fn form (returns a callable).
      if (args.length !== 1) {
        diags.push({ severity: 'error', message: `broadcasted() takes exactly one argument`, loc: valueNode.loc });
      } else if (args[0].type === 'KeywordArg') {
        diags.push({ severity: 'error', message: `broadcasted() argument must be positional`, loc: args[0].loc });
      }
      break;
    }

    case 'broadcast': {
      // broadcast(f, args...) — first arg is the head (function /
      // distribution / measure op); collection args follow positionally
      // or by kwarg (matching the callee's signature). Minimum 2 args.
      if (args.length < 2) {
        diags.push({ severity: 'error', message: `broadcast() requires at least two arguments (head, collection)`, loc: valueNode.loc });
      } else if (args[0].type === 'KeywordArg') {
        diags.push({ severity: 'error', message: `broadcast() first argument must be a positional head expression`, loc: args[0].loc });
      }
      break;
    }

    case 'aggregate': {
      // aggregate(f_reduction, output_axes, expr) — spec §04 §sec:aggregate.
      // Three distinguished positional inputs, no kwargs.
      if (args.length !== 3) {
        diags.push({
          severity: 'error',
          message: `aggregate() takes exactly three arguments (f_reduction, output_axes, expr)`,
          loc: valueNode.loc,
        });
        break;
      }
      for (const arg of args) {
        if (arg.type === 'KeywordArg') {
          diags.push({
            severity: 'error',
            message: `aggregate() takes positional arguments only`,
            loc: arg.loc,
          });
        }
      }
      // First arg: one of the seven order-invariant reductions.
      const fArg = args[0];
      const ALLOWED_REDUCTIONS = new Set([
        'sum', 'prod', 'mean', 'var', 'std', 'maximum', 'minimum',
      ]);
      if (fArg.type !== 'Identifier' || !ALLOWED_REDUCTIONS.has(fArg.name)) {
        diags.push({
          severity: 'error',
          message: `aggregate()'s first argument must be one of: `
            + `sum, prod, mean, var, std, maximum, minimum`,
          loc: fArg.loc,
        });
      }
      // Second arg: array literal of distinct AxisRef. The list may
      // be empty (spec §04 §sec:aggregate: "The bracketed axis list
      // may be empty for full reduction to a scalar") — `s = aggregate
      // (sum, [], A[.i] * B[.i])` reduces over every axis in expr.
      const oaArg = args[1];
      if (oaArg.type !== 'ArrayLiteral') {
        diags.push({
          severity: 'error',
          message: `aggregate()'s second argument must be an array literal `
            + `of axis names (e.g. [.i, .k] — or [] for full reduction)`,
          loc: oaArg.loc,
        });
      } else {
        const seen = new Set<string>();
        const declared: string[] = [];
        for (const el of oaArg.elements) {
          if (el.type !== 'AxisRef') {
            diags.push({
              severity: 'error',
              message: `aggregate() output_axes entries must be axis names (.name)`,
              loc: el.loc,
            });
            continue;
          }
          if (seen.has(el.name)) {
            diags.push({
              severity: 'error',
              message: `aggregate() output_axes contains duplicate axis '.${el.name}'`,
              loc: el.loc,
            });
          }
          seen.add(el.name);
          declared.push(el.name);
        }
        // Third arg: every declared output axis must appear at least once
        // in expr (per spec). Walk expr collecting axis-name usage.
        const exprArg = args[2];
        if (exprArg) {
          const used = collectAxisRefs(exprArg);
          for (const name of declared) {
            if (!used.has(name)) {
              diags.push({
                severity: 'error',
                message: `aggregate(): output axis '.${name}' does not appear in expr`,
                loc: oaArg.loc,
              });
            }
          }
        }
      }
      break;
    }

    case 'metricsum': {
      // metricsum(metric, output_axes, expr) — spec §04 §sec:metricsum.
      // Three distinguished positional inputs, no kwargs. The surface
      // shorthand `metric: result[output_indices] := expr` desugars to
      // this form in the parser, so most well-formed metricsums arrive
      // here with the canonical shape; user-written direct calls land
      // here too and get the same checks.
      if (args.length !== 3) {
        diags.push({
          severity: 'error',
          message: `metricsum() takes exactly three arguments (metric, output_axes, expr)`,
          loc: valueNode.loc,
        });
        break;
      }
      for (const arg of args) {
        if (arg.type === 'KeywordArg') {
          diags.push({
            severity: 'error',
            message: `metricsum() takes positional arguments only`,
            loc: arg.loc,
          });
        }
      }
      // Second arg: array literal of variance-marked AxisRefs (or empty).
      // The empty case `[]` reduces to a scalar (same semantics as
      // aggregate's empty case); no metric raise is needed because there
      // are no output axes to raise to upper canonical storage.
      const oaArg = args[1];
      const declared: { name: string; variance?: 'upper' | 'lower' }[] = [];
      if (oaArg.type !== 'ArrayLiteral') {
        diags.push({
          severity: 'error',
          message: `metricsum()'s second argument must be an array literal `
            + `of variance-marked axis names (e.g. [.mu^, .nu_])`,
          loc: oaArg.loc,
        });
      } else {
        const seen = new Set<string>();
        for (const el of oaArg.elements) {
          if (el.type !== 'AxisRef') {
            diags.push({
              severity: 'error',
              message: `metricsum() output_axes entries must be axis names (.name^ or .name_)`,
              loc: el.loc,
            });
            continue;
          }
          if (seen.has(el.name)) {
            diags.push({
              severity: 'error',
              message: `metricsum() output_axes contains duplicate axis '.${el.name}'`,
              loc: el.loc,
            });
          }
          seen.add(el.name);
          declared.push({ name: el.name, variance: el.variance });
        }
      }
      // Body axis-name usage map: name → list of {variance, loc} entries
      // for every AxisRef occurrence in expr (under nested aggregates /
      // metricsums we'd cross scopes; collectAxisRefsWithVariance stops
      // descent at nested calls of either kind to mirror typeinfer's
      // scope rule).
      const exprArg = args[2];
      const bodyOccurrences = exprArg
        ? collectAxisRefsWithVariance(exprArg)
        : new Map<string, { variance?: string; loc: any }[]>();

      // Static check #1 (spec §sec:metricsum "Static checks"):
      // **bare neutral aggregate axes (`.i` without a variance marker) are
      // not allowed inside `metricsum`**. Each body occurrence must carry
      // a variance marker. Same goes for output axes (which are also
      // checked in the inner loop below, but doing it here too gives a
      // sharper "in expr" diagnostic).
      for (const [name, occs] of bodyOccurrences) {
        for (const occ of occs) {
          if (!occ.variance) {
            diags.push({
              severity: 'error',
              message: `metricsum(): bare-neutral axis '.${name}' is not allowed `
                + `inside metricsum() — use '.${name}^' (upper) or '.${name}_' (lower)`,
              loc: occ.loc,
            });
          }
        }
      }
      // Output axis must carry a variance marker — the shorthand parse
      // always produces marked axes, but a direct metricsum() call could
      // pass bare-neutral axes through.
      for (const d of declared) {
        if (!d.variance) {
          diags.push({
            severity: 'error',
            message: `metricsum(): output axis '.${d.name}' is missing a variance `
              + `marker — use '.${d.name}^' (upper) or '.${d.name}_' (lower)`,
            loc: oaArg.loc,
          });
        }
      }

      // Static check #2 (spec §sec:metricsum "Static checks"):
      // **every output index must occur in `expr` with the same variance
      // and may not also be contracted** — i.e. it must appear in the body
      // with the same variance, and the body must use it only as that
      // variance (no second occurrence under the opposite marker).
      for (const d of declared) {
        if (!d.variance) continue;  // already diagnosed above
        const occs = bodyOccurrences.get(d.name);
        if (!occs || occs.length === 0) {
          diags.push({
            severity: 'error',
            message: `metricsum(): output axis '.${d.name}${d.variance === 'upper' ? '^' : '_'}'`
              + ` does not appear in expr`,
            loc: oaArg.loc,
          });
          continue;
        }
        // Every body occurrence of an output-axis must match the output
        // variance — otherwise the user has implicitly asked to contract
        // an output index, which spec forbids.
        for (const occ of occs) {
          if (occ.variance && occ.variance !== d.variance) {
            diags.push({
              severity: 'error',
              message: `metricsum(): output axis '.${d.name}${d.variance === 'upper' ? '^' : '_'}'`
                + ` also appears in expr with opposite variance — output indices `
                + `may not also be contracted`,
              loc: occ.loc,
            });
          }
        }
      }

      // Static check #3 (spec §sec:metricsum "Static checks"):
      // **every repeated non-output index in `expr` must occur exactly
      // twice — once upper and once lower**. Walk body occurrences,
      // skipping any names that ARE output indices (they're checked above);
      // for each non-output name, count upper / lower occurrences and
      // require exactly one of each.
      const outNames = new Set(declared.map(d => d.name));
      for (const [name, occs] of bodyOccurrences) {
        if (outNames.has(name)) continue;
        let uppers = 0, lowers = 0;
        for (const occ of occs) {
          if (occ.variance === 'upper') uppers++;
          else if (occ.variance === 'lower') lowers++;
        }
        if (uppers !== 1 || lowers !== 1) {
          // Pick the first occurrence of this name for the diagnostic
          // location — pointing at any one occurrence is enough to find
          // the offending axis in the source.
          const loc = occs[0] && occs[0].loc;
          diags.push({
            severity: 'error',
            message: `metricsum(): contracted axis '.${name}' must appear exactly `
              + `twice in expr — once upper ('.${name}^') and once lower ('.${name}_'); `
              + `got ${uppers} upper, ${lowers} lower`,
            loc: loc || valueNode.loc,
          });
        }
      }
      break;
    }

    case 'reduce': {
      // reduce(f, xs) — two positional args (spec §07).
      if (args.length !== 2) {
        diags.push({ severity: 'error', message: `reduce() takes exactly two arguments (f, xs)`, loc: valueNode.loc });
      }
      for (const arg of args) {
        if (arg.type === 'KeywordArg') {
          diags.push({ severity: 'error', message: `reduce() takes positional arguments only`, loc: arg.loc });
        }
      }
      break;
    }

    case 'scan': {
      // scan(f, init, xs) — three positional args (spec §07).
      if (args.length !== 3) {
        diags.push({ severity: 'error', message: `scan() takes exactly three arguments (f, init, xs)`, loc: valueNode.loc });
      }
      for (const arg of args) {
        if (arg.type === 'KeywordArg') {
          diags.push({ severity: 'error', message: `scan() takes positional arguments only`, loc: arg.loc });
        }
      }
      break;
    }

    case 'checked': {
      // checked(value, condition=...) OR checked(value=..., condition=...).
      // Per spec §07 the canonical form uses keyword arguments
      // (`value = ..., condition = ...`); the positional-value variant
      // `checked(value_expr, condition = ...)` is also accepted.
      if (args.length === 0) {
        diags.push({ severity: 'error', message: `checked() requires value and condition arguments`, loc: valueNode.loc });
        break;
      }
      // After any positional arg, every following arg must be a kwarg.
      let sawKwarg = false;
      for (const arg of args) {
        if (arg.type === 'KeywordArg') {
          sawKwarg = true;
        } else if (sawKwarg) {
          diags.push({
            severity: 'error',
            message: `checked() positional arguments must come before keyword arguments`,
            loc: arg.loc,
          });
        }
      }
      break;
    }

    // ---- Composite constructors --------------------------------------
    case 'record': {
      // record(name = expr, ...) — all keyword.
      if (args.length === 0) {
        diags.push({ severity: 'error', message: `record() requires at least one field`, loc: valueNode.loc });
      }
      for (const arg of args) {
        if (arg.type !== 'KeywordArg') {
          diags.push({ severity: 'error', message: `record() takes keyword arguments only (name = value)`, loc: arg.loc });
        }
      }
      break;
    }

    case 'table': {
      // Two forms (spec §03):
      //   table(col1 = [...], col2 = [...])   — column kwargs (canonical)
      //   table(r)                            — promote a record-of-equal-
      //                                          length-vectors to a table
      if (args.length === 0) {
        diags.push({ severity: 'error', message: `table() requires at least one column (or a record argument)`, loc: valueNode.loc });
        break;
      }
      // Single positional arg → record-promotion form. OK.
      if (args.length === 1 && args[0].type !== 'KeywordArg') {
        break;
      }
      // Otherwise: all kwargs required.
      for (const arg of args) {
        if (arg.type !== 'KeywordArg') {
          diags.push({ severity: 'error', message: `table() takes either a single positional record or column keyword arguments (name = value)`, loc: arg.loc });
        }
      }
      break;
    }

    case 'tuple':
    case 'vector': {
      // tuple(a, b, ...) / vector(...) — all positional.
      for (const arg of args) {
        if (arg.type === 'KeywordArg') {
          diags.push({ severity: 'error', message: `${name}() takes positional arguments only`, loc: arg.loc });
        }
      }
      break;
    }

    // ---- Disintegration ----------------------------------------------
    case 'disintegrate': {
      // disintegrate(name, joint_measure) — first arg names the
      // disintegration target(s): either a single string literal
      // ("obs") or an array of string literals (["obs1", "obs2"]) for
      // multi-field disintegration. Second arg is the joint measure.
      if (args.length !== 2) {
        diags.push({ severity: 'error', message: `disintegrate() takes exactly two arguments (name, joint)`, loc: valueNode.loc });
      }
      for (const arg of args) {
        if (arg.type === 'KeywordArg') {
          diags.push({ severity: 'error', message: `disintegrate() takes positional arguments only`, loc: arg.loc });
        }
      }
      if (args.length > 0 && args[0].type !== 'KeywordArg') {
        const first = args[0];
        const okString = first.type === 'StringLiteral';
        const okArray  = first.type === 'ArrayLiteral'
          && Array.isArray(first.elements)
          && first.elements.every((e: any) => e && e.type === 'StringLiteral');
        if (!okString && !okArray) {
          diags.push({
            severity: 'error',
            message: `disintegrate() first argument must be a string literal or an array of string literals naming the disintegration target(s)`,
            loc: first.loc,
          });
        }
      }
      break;
    }

    // ---- Set constructors --------------------------------------------
    case 'interval': {
      // interval(lo, hi) — two positional.
      if (args.length !== 2) {
        diags.push({ severity: 'error', message: `interval() takes exactly two arguments (lo, hi)`, loc: valueNode.loc });
      }
      for (const arg of args) {
        if (arg.type === 'KeywordArg') {
          diags.push({ severity: 'error', message: `interval() takes positional arguments only`, loc: arg.loc });
        }
      }
      break;
    }

    case 'cartprod': {
      // Two forms (same as joint / jointchain / kchain):
      //   cartprod(S1, S2, ...)                  — positional
      //   cartprod(name1 = S1, name2 = S2, ...)  — named (labels each axis)
      // Mixed is rejected. cartprod is a FIELD_FORM in lower.ts so the
      // named form lowers with `fields: [{name, value}, ...]`.
      if (args.length < 2) {
        diags.push({ severity: 'error', message: `cartprod() requires at least two arguments`, loc: valueNode.loc });
        break;
      }
      const allKw  = args.every((a: any) => a.type === 'KeywordArg');
      const allPos = args.every((a: any) => a.type !== 'KeywordArg');
      if (!allKw && !allPos) {
        diags.push({
          severity: 'error',
          message: `cartprod() arguments must be either all positional or all keyword (name = value), not mixed`,
          loc: valueNode.loc,
        });
      }
      break;
    }

    case 'cartpow': {
      // cartpow(S, n) — two positional (set, integer).
      if (args.length !== 2) {
        diags.push({ severity: 'error', message: `cartpow() takes exactly two arguments (set, n)`, loc: valueNode.loc });
      }
      for (const arg of args) {
        if (arg.type === 'KeywordArg') {
          diags.push({ severity: 'error', message: `cartpow() takes positional arguments only`, loc: arg.loc });
        }
      }
      break;
    }

    case 'stdsimplex': {
      // stdsimplex(n) — single positional integer.
      if (args.length !== 1) {
        diags.push({ severity: 'error', message: `stdsimplex() takes exactly one argument (n)`, loc: valueNode.loc });
      } else if (args[0].type === 'KeywordArg') {
        diags.push({ severity: 'error', message: `stdsimplex() argument must be positional`, loc: args[0].loc });
      }
      break;
    }
  }

  return diags;
}

/**
 * Walk an expression tree and collect referenced identifiers.
 * Skips keyword argument names (the 'name' in KeywordArg is not a reference).
 *
 * For `functionof` / `kernelof` calls, refs are partitioned along the
 * spec's two-scope semantics (engine-concepts §8):
 *   - **body refs** are identifier refs inside the reification's body
 *     (the first positional arg) that name an outer-scope binding
 *     and are NOT shadowed by a boundary kwarg. These are the
 *     closure captures the body actually computes against.
 *   - **paramSource refs** are identifier refs in the kwarg RHS
 *     positions of the reification. These name the binding whose
 *     `valueset` declares the formal's domain (spec §sec:functionof:
 *     "functionof effectively substitutes each boundary node a with
 *     an input node elementof(valueset(a))"). They are real outer-
 *     scope refs but their phase / value does not flow through the
 *     function's result — the call site binds the formal to its own
 *     argument.
 *
 * The body-internal occurrences of boundary names are excluded from
 * deps entirely — they designate the callable's INPUTS (fed at
 * application, decoupled from the like-named module nodes per spec
 * §04), regardless of how they lower (placeholder formals → `%local`;
 * identifier-form cuts → plain `self` refs, spec §11 — the dep on the
 * cut node itself is the paramSourceDeps entry). Refs that fall in
 * neither bucket (everything else, including nested non-reification
 * calls) go into the default bucket and contribute to `deps`.
 *
 * Returns { deps, callDeps, bodyDeps, paramSourceDeps }. `deps` is the
 * union of bodyDeps + paramSourceDeps + other refs, preserving the
 * existing contract for callers that don't care about the split.
 */
function collectDeps(node: any, definedNames: Set<string>) {
  const deps = new Set<string>();
  const callDeps = new Set<string>();
  const bodyDeps = new Set<string>();
  const paramSourceDeps = new Set<string>();

  // localStack tracks the formal-parameter names of every enclosing
  // reification. An Identifier whose name is on the stack is a
  // body-internal reference to a formal / boundary input and is NOT a
  // dep at the outer-scope level (the cut-node dep rides
  // paramSourceDeps).
  const localStack: Set<string>[] = [];

  function isLocal(name: string): boolean {
    for (let i = localStack.length - 1; i >= 0; i--) {
      if (localStack[i].has(name)) return true;
    }
    return false;
  }

  // bucket controls which secondary set an Identifier add lands in.
  // For non-reification bindings everything is 'body' (closure captures
  // and direct refs are indistinguishable, and bodyDeps mirrors deps).
  // Reification walking flips to 'paramSource' for kwarg RHS expressions.
  function add(name: string, isCallee: boolean, bucket: 'body' | 'paramSource') {
    deps.add(name);
    if (isCallee) callDeps.add(name);
    if (bucket === 'body') bodyDeps.add(name);
    else paramSourceDeps.add(name);
  }

  function walk(node: any, isCallee: boolean, bucket: 'body' | 'paramSource') {
    if (!node) return;

    switch (node.type) {
      case 'Identifier':
        if (isLocal(node.name)) return;       // %local ref — body's own formal
        if (definedNames.has(node.name)) add(node.name, isCallee, bucket);
        break;
      case 'BinaryExpr':
        walk(node.left, false, bucket);
        walk(node.right, false, bucket);
        break;
      case 'UnaryExpr':
        walk(node.operand, false, bucket);
        break;
      case 'CallExpr': {
        const callee = node.callee;
        const isReif = callee && callee.type === 'Identifier'
          && (callee.name === 'functionof' || callee.name === 'kernelof');
        if (isReif) {
          // Two-scope walk. First collect the formal-parameter names
          // declared by this reification's kwargs (boundary names per
          // spec §sec:functionof). The body's body-internal refs to
          // these names are %local and excluded from deps.
          const formals = new Set<string>();
          for (let i = 1; i < node.args.length; i++) {
            const arg = node.args[i];
            if (arg && arg.type === 'KeywordArg' && arg.value) {
              if (arg.value.type === 'Identifier') formals.add(arg.value.name);
              else if (arg.value.type === 'Placeholder') formals.add('_' + arg.value.name + '_');
            }
          }
          // Walk the body (first positional arg) under the formal-name
          // local scope, tagging refs as 'body' so they land in bodyDeps.
          localStack.push(formals);
          const body = node.args[0];
          if (body && body.type !== 'KeywordArg') walk(body, false, 'body');
          localStack.pop();
          // Walk the kwarg RHS expressions without the local scope,
          // tagging them as 'paramSource' so they land in paramSourceDeps.
          for (let i = 1; i < node.args.length; i++) {
            const arg = node.args[i];
            if (arg && arg.type === 'KeywordArg') walk(arg.value, false, 'paramSource');
          }
          // The reification callee itself (`functionof` / `kernelof`)
          // is a built-in name; nothing to add. (Skipped because the
          // bare-symbol callee isn't a definedNames entry anyway.)
        } else {
          walk(node.callee, true, bucket);
          for (const arg of node.args) walk(arg, false, bucket);
        }
        break;
      }
      case 'IndexExpr':
        walk(node.object, false, bucket);
        for (const idx of node.indices) walk(idx, false, bucket);
        break;
      case 'FieldAccess':
        walk(node.object, false, bucket);
        break;
      case 'ArrayLiteral':
      case 'TupleLiteral':
        for (const el of node.elements) walk(el, false, bucket);
        break;
      case 'KeywordArg':
        // Only walk the value, not the keyword name
        walk(node.value, false, bucket);
        break;
      // Leaf nodes: NumberLiteral, StringLiteral, BoolLiteral,
      // ConstantRef, SetRef, Placeholder, Hole, SliceAll — no deps
    }
  }

  walk(node, false, 'body');
  return { deps, callDeps, bodyDeps, paramSourceDeps };
}

/**
 * For functionof/kernelof calls, extract boundary inputs from keyword args.
 * Returns Map<argName, varName> for args after the first positional arg.
 * Placeholders resolve to their inner name. (lawof is unary and has no
 * boundary kwargs.)
 */
function extractBoundaries(valueNode: any) {
  if (!valueNode || valueNode.type !== 'CallExpr') return null;
  const callee = valueNode.callee;
  if (!callee || callee.type !== 'Identifier') return null;
  if (callee.name !== 'functionof' && callee.name !== 'kernelof') return null;

  const boundaries = new Map<string, string>();
  const args = valueNode.args;

  // Skip first arg (the expression to reify), process keyword args
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg.type === 'KeywordArg') {
      const argName = arg.name;
      let varName: string | null = null;
      if (arg.value.type === 'Identifier') {
        varName = arg.value.name;
      } else if (arg.value.type === 'Placeholder') {
        varName = '_' + arg.value.name + '_'; // full placeholder form for matching
      }
      if (varName) boundaries.set(argName, varName);
    }
  }
  return boundaries.size > 0 ? boundaries : null;
}

/**
 * Extract the field map from a joint-measure expression, when it can be
 * statically resolved.
 *
 * Recognised forms:
 *  - Tier 1 (`lawof_record`): `lawof(record(name1 = node1, ...))`
 *    — each field maps to a module-level node name. lawof is unary now,
 *    so there are no inherited boundaries from this form.
 *  - Tier 2 (`joint`): `joint(name1 = M1, ...)` keyword form
 *    — each field maps to an inline measure expression. Components are
 *    independent (no cross-boundaries between fields).
 *
 * Returns:
 *   { kind: 'lawof_record' | 'joint' | 'jointchain',
 *     fields: Map<fieldName, AST-expression> }
 * or null if the structure cannot be statically resolved.
 *
 * For 'lawof_record', each field's expression is an Identifier referring to a
 * module-level binding.
 * For 'joint', each field's expression is an arbitrary measure expression
 * (typically a CallExpr like Normal(...) or a measure-algebra construction).
 */
function extractJointFields(valueNode: any) {
  if (!valueNode || valueNode.type !== 'CallExpr') return null;
  if (!valueNode.callee || valueNode.callee.type !== 'Identifier') return null;

  // ----- Tier 1: lawof(record(...)) (unary) -----
  // Disintegrate operates on joint *measures*, so only unary lawof of a
  // record qualifies as a Tier 1 target. `kernelof(record(...), kwargs...)`
  // produces a kernel, not a joint measure, and cannot be disintegrated.
  if (valueNode.callee.name === 'lawof') {
    if (valueNode.args.length !== 1) return null;
    const firstArg = valueNode.args[0];
    if (firstArg.type !== 'CallExpr' || !firstArg.callee
        || firstArg.callee.type !== 'Identifier'
        || firstArg.callee.name !== 'record') {
      return null;
    }

    const fields = new Map<string, any>();
    for (const arg of firstArg.args) {
      if (arg.type !== 'KeywordArg') return null; // not statically resolvable
      if (arg.value.type !== 'Identifier') return null; // can't trace back to a node name
      fields.set(arg.name, arg.value);
    }
    return { kind: 'lawof_record', fields };
  }

  // ----- Tier 2: joint(name1 = M1, ...) keyword form -----
  if (valueNode.callee.name === 'joint') {
    if (valueNode.args.length === 0) return null;
    const fields = new Map<string, any>();
    for (const arg of valueNode.args) {
      if (arg.type !== 'KeywordArg') return null; // positional joint not statically inspectable here
      fields.set(arg.name, arg.value); // arbitrary measure expression
    }
    return { kind: 'joint', fields };
  }

  // ----- Tier 2: jointchain(name1 = M1, name2 = K2, ...) keyword form -----
  // The chain order matters: later fields may depend on earlier fields' variates.
  // We don't try to model that fully here — for disintegration along trailing
  // selected fields, the kernel gets synthesized chain-earlier field labels as
  // boundary inputs (see dag.js).
  if (valueNode.callee.name === 'jointchain') {
    if (valueNode.args.length === 0) return null;
    const fields = new Map<string, any>();
    for (const arg of valueNode.args) {
      if (arg.type !== 'KeywordArg') return null;
      fields.set(arg.name, arg.value);
    }
    return { kind: 'jointchain', fields };
  }

  return null;
}

/**
 * Detect a disintegrate-decomposition statement and resolve its structure.
 *
 *   kernel_name, prior_name = disintegrate(selector, joint_ref)
 *
 * `selector` may be a string literal or an array of string literals.
 * `joint_ref` must be an Identifier referencing a binding whose RHS is a
 * statically-resolvable joint measure (currently: lawof(record(...))).
 *
 * Returns null if any part doesn't match.
 *
 * @param {object} stmt - AssignStatement
 * @param {Map} bindingMap - already-built bindings map (for joint lookup)
 * @returns {{ kernelName, priorName, selectorFields, jointName, jointKind, jointFields, selectorLoc } | null}
 */
function detectDisintegration(stmt: any, bindingMap: any) {
  if (stmt.type !== 'AssignStatement') return null;
  if (stmt.names.length !== 2) return null;
  if (stmt.value.type !== 'CallExpr') return null;
  if (!stmt.value.callee || stmt.value.callee.type !== 'Identifier') return null;
  if (stmt.value.callee.name !== 'disintegrate') return null;
  if (stmt.value.args.length !== 2) return null;

  const selectorArg = stmt.value.args[0];
  const jointArg = stmt.value.args[1];

  // Parse selector. Per spec §06 lines 550-551, the surface form
  // matters: `"b"` selects the **bare value** (kernel yields the
  // variate directly), `["b"]` selects a **single-field record**
  // (kernel yields `record(b = ...)`). We preserve the distinction
  // via `selectorBareString`; downstream `disintegratePlan` reads it
  // when synthesising the kernel shape.
  let selectorFields: string[] | null = null;
  let selectorBareString = false;
  if (selectorArg.type === 'StringLiteral') {
    selectorFields = [selectorArg.value];
    selectorBareString = true;
  } else if (selectorArg.type === 'ArrayLiteral') {
    selectorFields = [];
    for (const el of selectorArg.elements) {
      if (el.type !== 'StringLiteral') return null;
      selectorFields.push(el.value);
    }
  } else {
    return null;
  }

  if (jointArg.type !== 'Identifier') return null;
  const jointBinding = bindingMap.get(jointArg.name);
  if (!jointBinding) return null;

  // We no longer pre-screen with extractJointFields — the rewriter will
  // determine structurally whether this disintegration is supported, and
  // selector errors come back as Unsupported reasons. extractJointFields
  // is kept around for selector-error diagnostics on the cases it does
  // recognise (see pass 3 in analyze).
  const jointInfo = extractJointFields(jointBinding.node.value);

  return {
    kernelName: stmt.names[0].name,
    priorName: stmt.names[1].name,
    selectorFields,
    selectorBareString,
    jointName: jointArg.name,
    jointKind:   jointInfo ? jointInfo.kind   : null,
    jointFields: jointInfo ? jointInfo.fields : null,
    selectorLoc: selectorArg.loc,
  };
}

/**
 * Attach an "effective RHS" view to a binding so the DAG renderer can
 * treat it as if its source were `effectiveValue` instead of the
 * statement's literal RHS. Used by disintegration to render synthesized
 * kernel/prior expressions naturally.
 */
function attachEffectiveRhs(binding: any, effectiveValue: any, definedNames: Set<string>) {
  binding.effectiveValue = effectiveValue;
  const { deps, callDeps } = collectDeps(effectiveValue, definedNames);
  // Self-references are never deps for rendering purposes.
  for (const n of binding.names || [binding.name]) {
    deps.delete(n);
    callDeps.delete(n);
  }
  binding.effectiveDeps = [...deps];
  binding.effectiveCallDeps = [...callDeps];
}

/**
 * For a Plan.delegate disintegration result, mirror the delegate target's
 * RHS view onto this binding so it renders identically — same kernelof/
 * lawof structure, same boundaries, same ancestor trace. The binding
 * keeps its own identity (LHS name, source location) but shares the
 * target's effective semantics.
 */
function attachDelegate(binding: any, targetName: string, bindings: any) {
  const target = bindings.get(targetName);
  if (!target || !target.node || !target.node.value) return;
  binding.effectiveValue    = target.node.value;
  binding.effectiveDeps     = [...(target.deps || [])];
  binding.effectiveCallDeps = [...(target.callDeps || [])];
}

/**
 * Classify the surface keyword of a disintegration's kernel-side
 * result — 'functionof' when the underlying reified body is a
 * measure, 'kernelof' when it's a (typically stochastic) value.
 * Per FlatPPL spec §sec:kernelof, `kernelof(x, ...)` requires `x`
 * to not be a measure, so a measure-bodied parametric kernel is
 * canonically `functionof(<measure>, ...)`.
 *
 *   - delegate plan: mirror the delegate target's own type. If the
 *     user wrote `forward_kernel = functionof(obs_dist, theta=theta)`
 *     and `disintegrate(...)` recovers it, `forward_kernel2.type`
 *     should also read 'functionof' — it's the same kernel.
 *   - synthesized plan: ask isMeasureExpr of the synthesized kernel
 *     expression. disintegrate.js's wrapAsKernelOrFunctionOf already
 *     emits the spec-compliant keyword in the AST; we mirror it on
 *     the binding type so the renderer's tether label matches.
 */
function kernelTypeForPlan(plan: any, bindings: any) {
  if (plan.kind === 'delegate') {
    const target = bindings.get(plan.kernel.binding);
    if (target && (target.type === 'functionof' || target.type === 'kernelof')) {
      return target.type;
    }
    // Target isn't itself a reification (e.g. it's a measure binding
    // playing the role of a constant kernel). 'kernelof' stays the
    // safe default; the renderer's color override won't surprise the
    // user since the kind resolution downstream still reads the body.
    return 'kernelof';
  }
  // synthesized: the kernel expression is a CallExpr to either
  // 'functionof' or 'kernelof'. Honour whatever wrapAsKernelOrFunctionOf
  // emitted.
  const expr = plan.kernel;
  if (expr && expr.type === 'CallExpr' && expr.callee && expr.callee.type === 'Identifier') {
    const name = expr.callee.name;
    if (name === 'functionof' || name === 'kernelof') return name;
  }
  // Constant-kernel synthesis (no boundary inputs) emits the body
  // directly — classify it by whether it's a measure.
  return isMeasureExpr(expr, bindings) ? 'functionof' : 'kernelof';
}

/**
 * Compute the phase of every binding via ancestor analysis, per spec
 * (`docs/04-design.md#phases`).
 *
 *  - `draw(...)` self → stochastic
 *  - `elementof(...)` self → parameterized
 *  - `external(...)` self → fixed (despite being an "input")
 *  - any other binding → max of its dependencies' phases, where
 *    stochastic > parameterized > fixed
 *
 * @param {Map} bindings
 * @returns {Map<string, 'fixed' | 'parameterized' | 'stochastic'>}
 */
/**
 * Phase of a degenerate (zero-entropy) draw, or null if the draw
 * isn't degenerate. Implements two spec identities:
 *
 *   draw(Dirac(value = e))    ≡ e         → phase(e)
 *   draw(lawof(e))            ≡ e         → phase(e)   (when e is
 *                                            value-typed; lawof of
 *                                            value-typed e is
 *                                            Dirac(value=e), so the
 *                                            outer draw extracts e)
 *
 * Walks at most one binding hop: if the draw's argument is a
 * reference to a binding whose RHS is a Dirac/lawof call, that
 * binding's value-arg phase is returned. Doesn't chase longer alias
 * chains — keeps the logic local and predictable. Multi-hop cases
 * (rare in practice) fall back to the structural 'stochastic'.
 *
 * Phase of the value AST is computed by walking refs (an Identifier
 * resolves to its binding's phase via the supplied phaseOf, anything
 * else falls back to 'fixed' for literals or recurses for nested
 * calls). This stays inside the analyzer's existing phase machinery
 * — no separate "alias-chasing" infrastructure.
 */
function phaseOfDegenerateDraw(drawArg: any, bindings: any, phaseOf: any) {
  if (!drawArg) return null;
  // Inline form: draw(Dirac(...)) or draw(lawof(...)).
  const inlinePhase = degenerateMeasurePhase(drawArg, bindings, phaseOf);
  if (inlinePhase != null) return inlinePhase;
  // One-hop alias form: draw(<measure-binding-name>).
  if (drawArg.type === 'Identifier' && bindings.has(drawArg.name)) {
    const target = bindings.get(drawArg.name);
    if (target && target.node && target.node.value) {
      return degenerateMeasurePhase(target.node.value, bindings, phaseOf);
    }
  }
  return null;
}

// Returns the phase of the value wrapped by a Dirac / lawof
// expression, or null if the expression isn't a recognised
// degenerate measure form. Internal helper for phaseOfDegenerateDraw.
function degenerateMeasurePhase(ast: any, bindings: any, phaseOf: any) {
  if (!ast || ast.type !== 'CallExpr' || !ast.callee
      || ast.callee.type !== 'Identifier') return null;
  const op = ast.callee.name;
  if (op === 'Dirac') {
    let valueAst = null;
    if (Array.isArray(ast.args)) {
      for (const a of ast.args) {
        if (a && a.type === 'KeywordArg' && a.name === 'value') {
          valueAst = a.value; break;
        }
      }
      if (!valueAst) {
        for (const a of ast.args) {
          if (a && a.type !== 'KeywordArg') { valueAst = a; break; }
        }
      }
    }
    return valueAst ? phaseOfAstExpr(valueAst, bindings, phaseOf) : null;
  }
  if (op === 'lawof' && Array.isArray(ast.args) && ast.args.length === 1) {
    // lawof(e) ≡ Dirac(value=e) only when e is value-typed and
    // non-stochastic (for stochastic e the spec identity is
    // lawof(draw(m)) ≡ m, which is *not* a Dirac and shouldn't
    // be sharpened here). Compute e's phase; if it's stochastic,
    // signal "not a degenerate Dirac form" so the caller falls
    // back to the structural rule.
    const ePhase = phaseOfAstExpr(ast.args[0], bindings, phaseOf);
    return ePhase === 'stochastic' ? null : ePhase;
  }
  return null;
}

// Phase of an arbitrary AST expression. Identifiers resolve to
// their binding's phase via phaseOf; literals are fixed; calls
// take the max over their args. Used by the degenerate-draw
// sharpening — kept tiny and local since the analysis is best-
// effort (returning a too-conservative phase is safe).
function phaseOfAstExpr(ast: any, bindings: any, phaseOf: any): string {
  if (!ast) return 'fixed';
  if (ast.type === 'Identifier') {
    return bindings.has(ast.name) ? phaseOf(ast.name) : 'fixed';
  }
  if (ast.type === 'CallExpr' && Array.isArray(ast.args)) {
    let phase = 'fixed';
    for (const a of ast.args) {
      const argAst = (a && a.type === 'KeywordArg') ? a.value : a;
      const argPhase = phaseOfAstExpr(argAst, bindings, phaseOf);
      if (argPhase === 'stochastic') return 'stochastic';
      if (argPhase === 'parameterized') phase = 'parameterized';
    }
    return phase;
  }
  // Literals (NumberLit, BoolLit, StringLit, ArrayLit, …) → fixed.
  return 'fixed';
}

function computePhases(bindings: any) {
  const phases = new Map<string, string>();
  const visiting = new Set<string>();

  function calleeName(b: any) {
    const v = b && b.node && b.node.value;
    if (v && v.type === 'CallExpr' && v.callee && v.callee.type === 'Identifier') {
      return v.callee.name;
    }
    return null;
  }

  function maxPhase(a: string, b: string) {
    if (a === 'stochastic' || b === 'stochastic') return 'stochastic';
    if (a === 'parameterized' || b === 'parameterized') return 'parameterized';
    return 'fixed';
  }

  // Per spec §sec:lawof line 309-314 ("lawof absorbs stochasticity
  // into the reified law rather than propagating it outward"), some
  // ops *absorb* stochastic ancestors — the result is a deterministic
  // measure / function / kernel whose phase depends only on the
  // parameterized leaves of its ancestor closure, not on whether
  // those leaves are reached through a draw. Recursively walks
  // body-deps, collapsing 'stochastic' verdicts to 'fixed' along the
  // way and surfacing only the highest non-stochastic phase.
  //
  // The walker reads `bodyDeps` (engine-concepts §8) rather than
  // `deps`. For non-reification bindings the two sets coincide; for
  // `functionof` / `kernelof` they differ — the kwarg RHS refs
  // (paramSources) are real outer-scope refs but they declare formal
  // parameters' value-sets, not the function's result. Walking them
  // would propagate the formal's elementof phase through any
  // downstream `rand` / `lawof`, mis-classifying calls like
  // `a = f(par = beta1); rand(state, lawof(a))` as parameterized
  // even when beta1 (the call's argument) is fixed.
  const absorbedCache = new Map<string, string>();
  function absorbedPhaseOf(name: string): string {
    if (absorbedCache.has(name)) return absorbedCache.get(name)!;
    const b = bindings.get(name);
    if (!b)                          { absorbedCache.set(name, 'fixed');         return 'fixed'; }
    const cn = calleeName(b);
    if (cn === 'elementof')          { absorbedCache.set(name, 'parameterized'); return 'parameterized'; }
    if (cn === 'external')           { absorbedCache.set(name, 'fixed');         return 'fixed'; }
    // CALLABLE bindings contribute 'fixed' — do NOT descend into the body
    // closure. A callable's parameterization is decided at its APPLICATION
    // (spec §04: boundary substitution precedes the ancestor trace), and the
    // application node's args are ordinary deps this walker already visits
    // (`k(theta)` surfaces theta's phase through the arg). Descending into
    // the body instead surfaces elementof leaves BEHIND the declared cut —
    // `kernelof(zs, pars = pars)` reaches the module `pars` through body
    // refs — so an applied kernel with FIXED args (`rand(state,
    // k_model(glob_pars))`) would mis-phase as parameterized, knocking the
    // draw off the fixed pre-eval path (spec §07: rand propagates phases
    // normally). Type list mirrors ir-shared.isCallableLikeBindingType
    // (the canonical catalogue; inlined here because ir-shared imports
    // analyzer at module load — the one consumer below the root).
    if (b.type === 'fn' || b.type === 'functionof' || b.type === 'kernelof'
        || b.type === 'bijection' || b.type === 'fchain') {
      absorbedCache.set(name, 'fixed');
      return 'fixed';
    }
    const deps = (b.bodyDeps != null) ? b.bodyDeps : b.deps;
    let phase = 'fixed';
    for (const dep of deps) {
      phase = maxPhase(phase, absorbedPhaseOf(dep));
      if (phase === 'parameterized') break;
    }
    absorbedCache.set(name, phase);
    return phase;
  }

  function phaseOf(name: string): string {
    if (phases.has(name)) return phases.get(name)!;
    if (visiting.has(name)) return 'fixed'; // cycle (shouldn't occur in valid code)
    visiting.add(name);

    const b = bindings.get(name);
    if (!b) {
      visiting.delete(name);
      phases.set(name, 'fixed');
      return 'fixed';
    }

    const cn = calleeName(b);
    let phase: string;
    if (cn === 'draw') {
      // Spec identities: draw(Dirac(value=e)) ≡ e, and lawof of a
      // value-typed e is Dirac(value=e), so draw(lawof(e)) ≡ e too.
      // For a degenerate (zero-entropy) draw, phase inherits e's
      // phase rather than the structural-stochastic default.
      // Non-degenerate draws (Normal, Exp, …) fall back to stochastic.
      const drawArg = b.node.value && b.node.value.args
                      && b.node.value.args[0];
      const sharpened = phaseOfDegenerateDraw(drawArg, bindings, phaseOf);
      phase = sharpened != null ? sharpened : 'stochastic';
    } else if (cn === 'elementof') {
      phase = 'parameterized';
    } else if (cn === 'external') {
      phase = 'fixed';
    } else if (cn === 'lawof' || cn === 'rand') {
      // Absorbs stochasticity. `lawof` reifies the ancestor sub-DAG
      // as a measure (spec §sec:lawof); `rand(state, M)` collapses
      // a measure into one threaded draw, owning the state across
      // every stochastic ancestor of M. In both cases the result is
      // fixed UNLESS an elementof remains in the ancestor closure
      // — then it propagates as parameterized.
      phase = 'fixed';
      for (const dep of b.deps) {
        phase = maxPhase(phase, absorbedPhaseOf(dep));
        if (phase === 'parameterized') break;
      }
    } else if (cn === 'functionof' || cn === 'kernelof' || cn === 'fn') {
      // The function/kernel *value* itself is fixed — the definition
      // doesn't change with inputs (spec §sec:functionof example
      // line 492-495: "the function value itself is %fixed"). Its
      // body's phase, evaluated on per-call inputs, is a separate
      // computation handled by computePhasesForScope inside the
      // reification bubble.
      phase = 'fixed';
    } else if (rhsContainsInlineDraw(b && b.node && b.node.value)) {
      // Hidden draw inside arithmetic / call — e.g. `s = 2 * draw(m)`.
      // The top-level callee isn't 'draw' so the cn check above
      // misses it, but the binding's value is still stochastic. The
      // walker skips reification bodies (their draws live in a
      // different scope).
      phase = 'stochastic';
    } else {
      phase = 'fixed';
      for (const dep of b.deps) {
        phase = maxPhase(phase, phaseOf(dep));
        if (phase === 'stochastic') break;
      }
    }

    visiting.delete(name);
    phases.set(name, phase);
    return phase;
  }

  for (const name of bindings.keys()) phaseOf(name);
  return phases;
}

/**
 * True iff the AST contains a `draw(...)` call somewhere inside,
 * not counting reification bodies (which have their own scope).
 *
 * Used by computePhases / computePhasesForScope to catch hidden
 * draws — e.g. `s = 2 * draw(m)` should classify s as stochastic
 * even though the binding's top-level callee is `mul`, not `draw`.
 */
function rhsContainsInlineDraw(node: any): boolean {
  if (!node || typeof node !== 'object') return false;
  if (node.type === 'CallExpr' && node.callee
      && node.callee.type === 'Identifier') {
    if (node.callee.name === 'draw') return true;
    // Don't descend into reification bodies (different namespace).
    if (node.callee.name === 'functionof'
        || node.callee.name === 'kernelof'
        || node.callee.name === 'fn') return false;
    if (Array.isArray(node.args)) {
      for (const a of node.args) if (rhsContainsInlineDraw(a)) return true;
    }
    return false;
  }
  for (const k of ['args', 'value', 'left', 'right', 'operand', 'object', 'elements', 'indices']) {
    const c = node[k];
    if (c == null) continue;
    if (Array.isArray(c)) {
      for (const x of c) if (rhsContainsInlineDraw(x)) return true;
    } else if (typeof c === 'object' && rhsContainsInlineDraw(c)) {
      return true;
    }
  }
  return false;
}

/**
 * Phase computation parameterised by a set of "boundary" names that
 * are treated as `parameterized` leaves regardless of their global
 * phase. Used to compute scope-local phases inside reified subgraphs:
 * a kernelof / functionof / lawof bubble cuts the phase chain at its
 * kwargs, because those names are *parameters* of the reification —
 * a value gets passed in for them at call time, so within the body
 * they are by definition `parameterized` rather than `stochastic`.
 *
 * Example:
 *   theta1 = draw(theta1_dist)               # globally: stochastic
 *   beta1  = 2 * theta1                       # globally: stochastic
 *   k = functionof(beta1, theta1 = theta1)
 * Inside k's body, theta1 is the kernel parameter, so:
 *   computePhasesForScope(bindings, new Set(['theta1']))
 * gives:
 *   theta1 → 'parameterized'   (overridden by boundary)
 *   beta1  → 'parameterized'   (depends only on theta1 in this scope)
 *
 * Nested reifications: pass the union of all enclosing scopes'
 * boundary names — every "outer" parameter is also `parameterized`
 * from the inner scope's perspective (the outer hasn't been called
 * yet either).
 *
 * Implementation note: this is a near-copy of computePhases with a
 * boundaryNames-set check. Keeping them separate (rather than making
 * computePhases a special case of this with an empty set) lets the
 * fast common-case path stay tight, and the scope helper carries
 * comments explaining its reification-specific semantics.
 *
 * @param {Map} bindings
 * @param {Set<string>} boundaryNames
 * @returns {Map<string, 'fixed' | 'parameterized' | 'stochastic'>}
 */
function computePhasesForScope(bindings: any, boundaryNames: Set<string>) {
  if (!boundaryNames || boundaryNames.size === 0) return computePhases(bindings);
  const phases = new Map<string, string>();
  const visiting = new Set<string>();

  function calleeName(b: any) {
    const v = b && b.node && b.node.value;
    if (v && v.type === 'CallExpr' && v.callee && v.callee.type === 'Identifier') {
      return v.callee.name;
    }
    return null;
  }

  function maxPhase(a: string, b: string) {
    if (a === 'stochastic' || b === 'stochastic') return 'stochastic';
    if (a === 'parameterized' || b === 'parameterized') return 'parameterized';
    return 'fixed';
  }

  function phaseOf(name: string): string {
    if (phases.has(name)) return phases.get(name)!;
    // Boundary cut: a kwarg of the enclosing reification. The phase
    // walk stops here — the value gets supplied at call time, so by
    // construction the body sees it as a parameter.
    if (boundaryNames.has(name)) {
      phases.set(name, 'parameterized');
      return 'parameterized';
    }
    if (visiting.has(name)) return 'fixed';
    visiting.add(name);

    const b = bindings.get(name);
    if (!b) {
      visiting.delete(name);
      phases.set(name, 'fixed');
      return 'fixed';
    }

    const cn = calleeName(b);
    let phase: string;
    if (cn === 'draw') {
      // Same degenerate-draw sharpening as in computePhases — see
      // its comment for the rationale (draw(Dirac(value=e)) ≡ e).
      const drawArg = b.node.value && b.node.value.args
                      && b.node.value.args[0];
      const sharpened = phaseOfDegenerateDraw(drawArg, bindings, phaseOf);
      phase = sharpened != null ? sharpened : 'stochastic';
    } else if (cn === 'elementof') {
      phase = 'parameterized';
    } else if (cn === 'external') {
      phase = 'fixed';
    } else if (rhsContainsInlineDraw(b && b.node && b.node.value)) {
      // Hidden draw inside an expression — same reasoning as in
      // computePhases. The boundary cut at boundaryNames takes
      // precedence (caught at the top of phaseOf), so an inline
      // draw on a name shadowed by a boundary still resolves to
      // 'parameterized'.
      phase = 'stochastic';
    } else {
      phase = 'fixed';
      for (const dep of b.deps) {
        phase = maxPhase(phase, phaseOf(dep));
        if (phase === 'stochastic') break;
      }
    }

    visiting.delete(name);
    phases.set(name, phase);
    return phase;
  }

  for (const name of bindings.keys()) phaseOf(name);
  return phases;
}

/**
 * Validate that literal integer indices in `IndexExpr` nodes obey the
 * surface variant's indexing convention:
 *   - `get`  (FlatPPL/FlatPPJ, 1-based): reject `xs[0]` and `xs[-N]`.
 *   - `get0` (FlatPPY,         0-based): reject `xs[-N]` only.
 * Runtime expressions are not checked. The lowering target lives on
 * the IndexExpr node (set at parse time from variant.indexingLowersTo),
 * so this pass remains variant-agnostic.
 *
 * @param {object} node - root expression node
 * @param {Diagnostic[]} diagnostics - mutable, appended to
 */
function validateIndexing(node: any, diagnostics: any[]) {
  function checkIndex(idx: any, indexOp: string) {
    const minIdx = (indexOp === 'get0') ? 0 : 1;
    const baseLabel = (indexOp === 'get0')
      ? 'FlatPPY uses 0-based indexing (indices start at 0)'
      : 'FlatPPL uses 1-based indexing (indices start at 1)';
    // Direct integer literal: x[0], x[2.5] (non-integer is a type error elsewhere)
    if (idx.type === 'NumberLiteral'
        && Number.isInteger(idx.value) && idx.value < minIdx) {
      diagnostics.push({
        severity: 'error',
        message: `Invalid index ${idx.value}: ${baseLabel}`,
        loc: idx.loc,
      });
      return;
    }
    // Negated literal: x[-1] parses as UnaryExpr('-', NumberLiteral(1))
    if (idx.type === 'UnaryExpr' && idx.op === '-'
        && idx.operand.type === 'NumberLiteral'
        && Number.isInteger(idx.operand.value) && idx.operand.value > 0) {
      diagnostics.push({
        severity: 'error',
        message: `Invalid index -${idx.operand.value}: ${baseLabel}`,
        loc: idx.loc,
      });
    }
  }

  function walk(node: any) {
    if (!node) return;
    if (node.type === 'IndexExpr') {
      walk(node.object);
      const op = node.indexOp || 'get';
      for (const i of node.indices) {
        checkIndex(i, op);
        walk(i);
      }
      return;
    }
    if (node.type === 'CallExpr') { walk(node.callee); for (const a of node.args) walk(a); return; }
    if (node.type === 'BinaryExpr') { walk(node.left); walk(node.right); return; }
    if (node.type === 'UnaryExpr') { walk(node.operand); return; }
    if (node.type === 'ArrayLiteral' || node.type === 'TupleLiteral') {
      for (const e of node.elements) walk(e);
      return;
    }
    if (node.type === 'FieldAccess') { walk(node.object); return; }
    if (node.type === 'KeywordArg') { walk(node.value); return; }
    // Leaves: NumberLiteral, StringLiteral, BoolLiteral, ConstantRef, SetRef,
    // Identifier, Placeholder, Hole, SliceAll — no recursion.
  }
  walk(node);
}

/**
 * Validate hole (`_`) and placeholder (`_name_`) usage according to the spec:
 *  - `_` is only valid inside `fn(...)`.
 *  - `_name_` is only valid inside `functionof(...)` or `kernelof(...)`.
 *    (lawof is unary now and cannot bind placeholders.)
 *
 * Scope is determined by the nearest enclosing special operation.
 *
 * @param {object} node - root expression node
 * @param {Diagnostic[]} diagnostics - mutable, appended to
 */
function validateHolesAndPlaceholders(node: any, diagnostics: any[]) {
  // scope can be: 'normal', 'fn', 'reify' (functionof/kernelof), 'aggregate'
  function walk(node: any, scope: string) {
    if (!node) return;
    switch (node.type) {
      case 'Hole':
        if (scope !== 'fn') {
          diagnostics.push({
            severity: 'error',
            message: "Hole '_' may only appear inside fn(...)",
            loc: node.loc,
          });
        }
        return;
      case 'Placeholder':
        if (scope !== 'reify') {
          diagnostics.push({
            severity: 'error',
            message: `Placeholder '_${node.name}_' may only appear inside functionof(...) or kernelof(...)`,
            loc: node.loc,
          });
        }
        return;
      case 'AxisRef':
        // Per spec §05 Axis names: an axis label `.name` (or its
        // variance-marked form `.name^` / `.name_`) is legal only inside
        // an enclosing `aggregate(...)` or `metricsum(...)` — as an entry
        // of output_axes, as an `[...]` index in the body, or as a binder
        // on the LHS of `:=` (which the parser desugars to `aggregate(...)`
        // or `metricsum(...)`).
        if (scope !== 'aggregate' && scope !== 'metricsum') {
          diagnostics.push({
            severity: 'error',
            message: `Axis name '.${node.name}' may only appear inside `
              + `aggregate(...) or metricsum(...)`,
            loc: node.loc,
          });
        }
        return;
      case 'CallExpr': {
        let inner = scope;
        if (node.callee && node.callee.type === 'Identifier') {
          if (node.callee.name === 'fn') inner = 'fn';
          else if (node.callee.name === 'functionof' || node.callee.name === 'kernelof') inner = 'reify';
          else if (node.callee.name === 'aggregate') inner = 'aggregate';
          else if (node.callee.name === 'metricsum') inner = 'metricsum';
        }
        walk(node.callee, scope);
        for (const a of node.args) walk(a, inner);
        return;
      }
      case 'BinaryExpr':
        walk(node.left, scope);
        walk(node.right, scope);
        return;
      case 'UnaryExpr':
        walk(node.operand, scope);
        return;
      case 'ArrayLiteral':
      case 'TupleLiteral':
        for (const e of node.elements) walk(e, scope);
        return;
      case 'IndexExpr':
        walk(node.object, scope);
        for (const i of node.indices) walk(i, scope);
        return;
      case 'FieldAccess':
        walk(node.object, scope);
        return;
      case 'KeywordArg':
        walk(node.value, scope);
        return;
      // Identifier, NumberLiteral, StringLiteral, BoolLiteral, ConstantRef,
      // SetRef, SliceAll: nothing to do.
    }
  }
  walk(node, 'normal');
}

/**
 * Walk an AST collecting all distinct axis-ref names that appear in it.
 * Used by `validateSpecialOperation` for `aggregate` to verify each
 * declared output axis is actually referenced in the expr.
 */
function collectAxisRefs(node: any): Set<string> {
  const names = new Set<string>();
  function walk(n: any) {
    if (!n || typeof n !== 'object') return;
    if (Array.isArray(n)) { for (const c of n) walk(c); return; }
    if (n.type === 'AxisRef') { names.add(n.name); return; }
    for (const k of Object.keys(n)) {
      if (k === 'loc') continue;
      walk(n[k]);
    }
  }
  walk(node);
  return names;
}

/**
 * Walk an AST collecting every axis-ref occurrence with its variance
 * marker and location, grouped by axis name. Used by
 * `validateSpecialOperation` for `metricsum` to enforce the
 * paired-upper/lower-twice rule and the same-variance / not-contracted
 * rule (spec §04 §sec:metricsum "Static checks"). We stop descending
 * at nested `aggregate(...)` / `metricsum(...)` calls because those
 * introduce a fresh axis-name scope per spec §05.
 */
function collectAxisRefsWithVariance(node: any): Map<string, { variance?: string; loc: any }[]> {
  const m = new Map<string, { variance?: string; loc: any }[]>();
  function walk(n: any) {
    if (!n || typeof n !== 'object') return;
    if (Array.isArray(n)) { for (const c of n) walk(c); return; }
    // Stop descending into nested aggregate / metricsum — their axes
    // live in a separate lexical scope.
    if (n.type === 'CallExpr' && n.callee && n.callee.type === 'Identifier'
        && (n.callee.name === 'aggregate' || n.callee.name === 'metricsum')) {
      return;
    }
    if (n.type === 'AxisRef') {
      const occs = m.get(n.name);
      const entry = { variance: n.variance, loc: n.loc };
      if (occs) occs.push(entry); else m.set(n.name, [entry]);
      return;
    }
    for (const k of Object.keys(n)) {
      if (k === 'loc') continue;
      walk(n[k]);
    }
  }
  walk(node);
  return m;
}

/**
 * Reconstruct the RHS expression as source text from the original source.
 */
function sliceSource(source: string, loc: any) {
  const lines = source.split('\n');
  if (loc.start.line === loc.end.line) {
    return lines[loc.start.line].slice(loc.start.col, loc.end.col);
  }
  let result = lines[loc.start.line].slice(loc.start.col);
  for (let i = loc.start.line + 1; i < loc.end.line; i++) {
    result += '\n' + lines[i];
  }
  result += '\n' + lines[loc.end.line].slice(0, loc.end.col);
  return result;
}

/**
 * Find all identifier references in an expression and their locations.
 * Used by definition/hover providers to find what's under the cursor.
 */
function collectIdentRefs(node: any) {
  const refs: any[] = [];
  function walk(node: any) {
    if (!node) return;
    if (node.type === 'Identifier') { refs.push(node); return; }
    if (node.type === 'CallExpr') { walk(node.callee); for (const a of node.args) walk(a); }
    if (node.type === 'BinaryExpr') { walk(node.left); walk(node.right); }
    if (node.type === 'UnaryExpr') walk(node.operand);
    if (node.type === 'ArrayLiteral' || node.type === 'TupleLiteral') for (const e of node.elements) walk(e);
    if (node.type === 'IndexExpr') { walk(node.object); for (const i of node.indices) walk(i); }
    if (node.type === 'FieldAccess') walk(node.object);
    if (node.type === 'KeywordArg') walk(node.value);
  }
  walk(node);
  return refs;
}

// ---------------------------------------------------------------------
// `restrict(M, x)` expansion (spec §06 "Measure restriction").
//
// `restrict(M, x)` is shorthand for
//
//     kernel, marginal = disintegrate(<fields-of-x>, M)
//     restrictResult   = bayesupdate(likelihoodof(kernel, x), marginal)
//
// We expand each `restrict(...)` AssignStatement at the top of analyze
// so the rest of the pipeline never has to know `restrict` exists. The
// two ops introduced (disintegrate + bayesupdate + likelihoodof) are
// already first-class. The synthesised `kernel`/`marginal` anons use
// `__restrict_*` names so they're elidable per spec §04
// "Auto-generated names".
//
// Surface forms accepted:
//   nu = restrict(M, x)              # positional: x is the observed record
//   nu = restrict(M, a = .., b = ..) # kwarg form — bundle to a record anon
//
// The `M` argument must currently be an Identifier so disintegrate's
// joint-resolver (which requires an identifier joint ref) applies. If
// `x` is an inline `record(a=..., b=...)` literal we read field names
// from it directly. If `x` is an Identifier whose binding is a record
// literal, we likewise extract field names. Other shapes (kwarg form,
// or identifier whose type can be statically derived) cover the
// expected use cases for v0.1.

let _restrictCounter = 0;
function _freshRestrictAnon(role: string): string {
  return `__restrict_${role}_${_restrictCounter++}`;
}

/**
 * Statically enumerate the field-names of a joint measure binding by
 * walking the AST. Used by `expandRestrictStatements` (which runs
 * before the bindings map is populated) to compute the complement of
 * `x`'s fields for the complement-disintegration route per spec §06.
 *
 * Covers the common cases:
 *   M = lawof(record(name = ..., ...))    → record kwarg names
 *   M = joint(name = ..., ...)            → joint kwarg names
 *   M = jointchain(name = ..., ...)       → jointchain kwarg names
 *   M = some_other_M                      → recurse through the
 *                                            identifier (with cycle guard)
 *
 * Returns null when the shape isn't covered — the caller falls back
 * to the selector-disintegration route (spec equivalence). Pure
 * AST inspection; bindings map not required.
 */
function _jointMeasureFields(measureName: string, ast: any, seen?: Set<string>): Set<string> | null {
  if (!ast || !ast.body) return null;
  if (!seen) seen = new Set();
  if (seen.has(measureName)) return null;
  seen.add(measureName);
  for (const stmt of ast.body) {
    if (stmt.type !== 'AssignStatement' || !stmt.names
        || stmt.names.length !== 1) continue;
    if (stmt.names[0].name !== measureName) continue;
    return _extractFieldsFromMeasureRhs(stmt.value, ast, seen);
  }
  return null;
}

/**
 * Statically determine the dependency DAG among variates of a
 * `lawof(record(...))` joint measure, used by restrict-expand to
 * choose between the selector and complement disintegration routes
 * (spec §06). For each variate field, returns the set of OTHER
 * variates referenced (directly or transitively, only one binding-hop
 * deep) in its defining RHS.
 *
 * Restricted to the canonical shape `M = lawof(record(name1 = ident1,
 * name2 = ident2, ...))` where each `ident_i` is a binding (typically
 * a draw, e.g. `theta1 ~ Normal(0, 1)` or `obs ~ iid(Normal(mu = …,
 * sigma = …), n)`). Returns null when the shape isn't recognised —
 * the caller falls back to the selector route.
 *
 * Disintegration admissibility:
 *   - For a selector S to be structurally admissible, no unselected
 *     variate u may depend on any selected variate s ∈ S. (Spec §06:
 *     the kernel goes from unselected → selected; selected ones must
 *     be downstream.)
 *
 * For `lawof(record(theta1=theta1, theta2=theta2, obs=obs))` where
 * `obs ~ iid(Normal(mu=theta1, ...), n)`:
 *   - Variate `obs` depends on `theta1`, `theta2`.
 *   - Variates `theta1` / `theta2` depend on nothing.
 *   - selector = ["obs"]: unselected = {theta1, theta2}, no dep on
 *     selected → admissible. (Forward / standard posterior direction.)
 *   - selector = ["theta1", "theta2"]: unselected = {obs}, obs
 *     depends on selected theta1/theta2 → NOT admissible. (Posterior
 *     direction, needs model inversion.)
 */
function _jointVariateDeps(measureName: string, ast: any): Map<string, Set<string>> | null {
  const Mrhs = _findRhsOfBinding(measureName, ast, new Set());
  if (!Mrhs) return null;
  // Recognise `lawof(record(field = ident, ...))`.
  if (Mrhs.type !== 'CallExpr' || !Mrhs.callee
      || Mrhs.callee.type !== 'Identifier' || Mrhs.callee.name !== 'lawof'
      || !Array.isArray(Mrhs.args) || Mrhs.args.length !== 1) return null;
  const inner = Mrhs.args[0];
  if (!inner || inner.type !== 'CallExpr' || !inner.callee
      || inner.callee.type !== 'Identifier' || inner.callee.name !== 'record') return null;
  // Map field name → identifier name (the variate binding).
  const variateIdent = new Map<string, string>();
  for (const a of inner.args || []) {
    if (!a || a.type !== 'KeywordArg') return null;
    if (!a.value || a.value.type !== 'Identifier') return null;
    variateIdent.set(a.name, a.value.name);
  }
  if (variateIdent.size === 0) return null;
  // For each variate, walk its binding's RHS transitively (chasing
  // identifier refs through intermediate bindings) and collect any
  // identifier that resolves to another variate. Transitive chasing
  // is necessary because models commonly route through intermediate
  // deterministic bindings: `obs ~ iid(Normal(mu = a, sigma = b), n)`
  // with `a = theta1 + theta2`; obs's DIRECT refs include `a` (not
  // a variate), but obs transitively depends on theta1, theta2.
  const identToVar = new Map<string, string>();
  for (const [vName, iName] of variateIdent) identToVar.set(iName, vName);
  const deps = new Map<string, Set<string>>();
  for (const [vName, iName] of variateIdent) {
    const ds = new Set<string>();
    const seenChase = new Set<string>([iName]);
    const stack: any[] = [];
    const startRhs = _findRhsOfBinding(iName, ast, new Set());
    if (startRhs) stack.push(startRhs);
    while (stack.length > 0) {
      const node = stack.pop();
      _collectIdentifierRefs(node, (name: string) => {
        const otherVar = identToVar.get(name);
        if (otherVar && otherVar !== vName) {
          ds.add(otherVar);
          return;   // don't chase into another variate's binding;
                    // its own deps are tracked separately.
        }
        if (seenChase.has(name)) return;
        seenChase.add(name);
        const sub = _findRhsOfBinding(name, ast, new Set());
        if (sub) stack.push(sub);
      });
    }
    deps.set(vName, ds);
  }
  return deps;
}

function _findRhsOfBinding(name: string, ast: any, seen: Set<string>): any | null {
  if (!ast || !ast.body) return null;
  if (seen.has(name)) return null;
  seen.add(name);
  for (const stmt of ast.body) {
    if (stmt.type !== 'AssignStatement' || !stmt.names
        || stmt.names.length !== 1) continue;
    if (stmt.names[0].name !== name) continue;
    return stmt.value;
  }
  return null;
}

function _collectIdentifierRefs(node: any, visit: (n: string) => void): void {
  if (!node || typeof node !== 'object') return;
  if (Array.isArray(node)) { for (const x of node) _collectIdentifierRefs(x, visit); return; }
  if (node.type === 'Identifier' && typeof node.name === 'string') visit(node.name);
  for (const k of Object.keys(node)) {
    if (k === 'loc') continue;
    _collectIdentifierRefs(node[k], visit);
  }
}

/**
 * Is `disintegrate(selectorFields, M)` structurally admissible given
 * the AST-level variate dependency DAG? Per spec §06: no unselected
 * variate may depend on a selected variate.
 */
function _disintegrateAdmissible(deps: Map<string, Set<string>>, selector: string[]): boolean {
  const sel = new Set(selector);
  for (const [name, depsOf] of deps) {
    if (sel.has(name)) continue;   // selected
    for (const d of depsOf) {
      if (sel.has(d)) return false;
    }
  }
  return true;
}

function _extractFieldsFromMeasureRhs(node: any, ast: any, seen: Set<string>): Set<string> | null {
  if (!node) return null;
  // Identifier ref: chase through the AST.
  if (node.type === 'Identifier') {
    return _jointMeasureFields(node.name, ast, seen);
  }
  if (node.type !== 'CallExpr' || !node.callee
      || node.callee.type !== 'Identifier') return null;
  const op = node.callee.name;
  // lawof(record(name=..., ...)): fields are the record kwargs.
  if (op === 'lawof' && Array.isArray(node.args) && node.args.length === 1) {
    const inner = node.args[0];
    if (inner && inner.type === 'CallExpr' && inner.callee
        && inner.callee.type === 'Identifier'
        && inner.callee.name === 'record') {
      const fields = new Set<string>();
      for (const a of inner.args || []) {
        if (a && a.type === 'KeywordArg') fields.add(a.name);
      }
      return fields.size > 0 ? fields : null;
    }
  }
  // joint / jointchain in keyword form: fields are the call kwargs.
  if ((op === 'jointchain' || op === 'joint') && Array.isArray(node.args)
      && node.args.length > 0) {
    const fields = new Set<string>();
    let allKw = true;
    for (const a of node.args) {
      if (a && a.type === 'KeywordArg') fields.add(a.name);
      else { allKw = false; break; }
    }
    if (allKw && fields.size > 0) return fields;
  }
  return null;
}

function _recordFieldNames(node: any, ast?: any, seen?: Set<string>): string[] | null {
  if (!node) return null;
  if (node.type === 'CallExpr' && node.callee
      && node.callee.type === 'Identifier' && node.callee.name === 'record') {
    const names: string[] = [];
    for (const a of node.args || []) {
      if (!a || a.type !== 'KeywordArg') return null;
      names.push(a.name);
    }
    return names;
  }
  // Resolve `restrict(M, x)` where `x` is a binding ref to a record
  // construction. We look up `x`'s defining statement in the AST and
  // recurse into its RHS. Cycle guard prevents infinite recursion on
  // pathological inputs like `r = r`.
  if (node.type === 'Identifier' && ast && ast.body) {
    if (!seen) seen = new Set();
    if (seen.has(node.name)) return null;
    seen.add(node.name);
    for (const stmt of ast.body) {
      if (stmt.type !== 'AssignStatement' || !stmt.names) continue;
      // Match single-LHS bindings and the multi-LHS case where this
      // name appears (record-of-record etc.). The recursion only
      // looks at the binding's RHS, so multi-LHS gives the tuple
      // expression which probably isn't a record literal — skip.
      if (stmt.names.length === 1 && stmt.names[0].name === node.name) {
        return _recordFieldNames(stmt.value, ast, seen);
      }
    }
  }
  return null;
}

function expandRestrictStatements(ast: any, diagnostics: any[]) {
  if (!ast || !ast.body) return;
  const newBody: any[] = [];
  for (const stmt of ast.body) {
    // Two top-level patterns supported by this pass:
    //   name = restrict(M, x)                — spec-compliant non-normalized
    //                                          conditional, totalmass = density
    //                                          of marginal at x.
    //   name = normalize(restrict(M, x))     — the normalized variant. Per the
    //                                          peephole `normalize(weighted(c, M))
    //                                          = normalize(M)`, the scalar mass
    //                                          factor cancels under normalize,
    //                                          so the complement-route expansion
    //                                          drops the logweighted wrap (and
    //                                          the marginal-density dead-code
    //                                          computation that goes with it).
    //                                          `normalize(...)` itself is kept —
    //                                          the underlying kernel/bayesupdate
    //                                          may carry non-unit mass.
    // Other nestings of restrict (e.g. inside truncate, weighted, broadcast)
    // are not currently rewritten; users can lift the inner restrict to a
    // named binding by hand.
    let restrictCall: any = null;
    let wrapInNormalize = false;
    if (stmt.type === 'AssignStatement'
        && stmt.value && stmt.value.type === 'CallExpr'
        && stmt.value.callee && stmt.value.callee.type === 'Identifier') {
      if (stmt.value.callee.name === 'restrict') {
        restrictCall = stmt.value;
      } else if (stmt.value.callee.name === 'normalize'
                 && Array.isArray(stmt.value.args)
                 && stmt.value.args.length === 1
                 && stmt.value.args[0]
                 && stmt.value.args[0].type === 'CallExpr'
                 && stmt.value.args[0].callee
                 && stmt.value.args[0].callee.type === 'Identifier'
                 && stmt.value.args[0].callee.name === 'restrict') {
        restrictCall = stmt.value.args[0];
        wrapInNormalize = true;
      }
    }
    if (!restrictCall) {
      newBody.push(stmt);
      continue;
    }
    const call = restrictCall;
    const args = call.args || [];
    if (args.length < 2) {
      diagnostics.push({
        severity: 'error',
        message: `restrict() requires a measure and observations (got ${args.length} args)`,
        loc: call.loc,
      });
      newBody.push(stmt);
      continue;
    }
    if (args[0].type === 'KeywordArg') {
      diagnostics.push({
        severity: 'error',
        message: `restrict()'s first argument (the measure) must be positional`,
        loc: args[0].loc,
      });
      newBody.push(stmt);
      continue;
    }
    const measureArg = args[0];
    if (measureArg.type !== 'Identifier') {
      diagnostics.push({
        severity: 'error',
        message: `restrict()'s measure argument must be a binding reference (got ${measureArg.type})`,
        loc: measureArg.loc,
      });
      newBody.push(stmt);
      continue;
    }
    const rest = args.slice(1);
    // Two surface shapes: a single positional record-like x, OR
    // a list of kwargs (auto-splat to a record).
    const allKwargs = rest.every((a: any) => a && a.type === 'KeywordArg');
    const allPositional = rest.every((a: any) => a && a.type !== 'KeywordArg');

    let xExpr: any = null;
    let fieldNames: string[] | null = null;

    if (allKwargs && rest.length >= 1) {
      // restrict(M, a = .., b = ..) → x = record(a = .., b = ..)
      fieldNames = rest.map((a: any) => a.name);
      xExpr = AST.CallExpr(
        AST.Identifier('record', AST.synthLoc('restrict-expand')),
        rest.map((a: any) => AST.KeywordArg(a.name, a.value, a.loc)),
        AST.synthLoc('restrict-expand'));
    } else if (allPositional && rest.length === 1) {
      xExpr = rest[0];
      fieldNames = _recordFieldNames(xExpr, ast);
      if (!fieldNames) {
        diagnostics.push({
          severity: 'error',
          message: `restrict()'s observation argument must be an inline 'record(...)' literal or a record-typed binding; field names could not be determined statically`,
          loc: xExpr.loc,
        });
        newBody.push(stmt);
        continue;
      }
    } else {
      diagnostics.push({
        severity: 'error',
        message: `restrict() takes 'restrict(M, x)' or 'restrict(M, a = .., b = ..)' (cannot mix positional and keyword observations)`,
        loc: call.loc,
      });
      newBody.push(stmt);
      continue;
    }

    if (!fieldNames || fieldNames.length === 0) {
      diagnostics.push({
        severity: 'error',
        message: `restrict() requires at least one observed field`,
        loc: call.loc,
      });
      newBody.push(stmt);
      continue;
    }

    // Use the user's `restrict(...)` source range for all synthesized
    // statements/exprs — sliceSource and source-located diagnostics
    // then point back at the original line. The synth marker on the
    // loc object preserves provenance for renderer / IR consumers
    // that want to distinguish synthetic from user-written.
    const sloc = { ...call.loc, synthetic: true, source: 'restrict-expand' };

    // Try the COMPLEMENT-disintegration route first (spec §06
    // "Measure restriction"): if `M`'s field set is statically
    // resolvable and the complement of `x`'s fields admits a
    // structural disintegration, the kernel goes from `x`'s
    // variates to the complement (the forward model when `M` is a
    // generative `lawof(record(...))` / jointchain). The result is
    // `logweighted(logdensityof(marginal, x), kernel(x))` — a
    // single scalar reweighting that restores the marginal-
    // likelihood mass factor `mu(x, complement)` that selector-
    // route bayesupdate produces via per-point reweighting. This
    // is the tractable direction for the "fix parameters, observe
    // predictive" use case where `x` covers parameters and the
    // selector direction (which would yield the posterior kernel)
    // is intractable.
    //
    // Fall back to the SELECTOR-disintegration route (kernel goes
    // from complement to `x`'s variates; bayesupdate reweights the
    // marginal) when the complement direction isn't admissible —
    // either the field set can't be statically resolved, the
    // complement is empty (x covers every field), or the
    // complement disintegrate's admissibility check fails.
    // Choose route via AST-level disintegrate admissibility.
    //
    // Always try the SELECTOR route first (matches spec equivalence;
    // standard posterior pattern). Fall back to the COMPLEMENT route
    // only when the selector route's disintegrate is not structurally
    // admissible (i.e., some unselected variate depends on a selected
    // one — the posterior direction in a typical forward generative
    // model). Spec §06 "Measure restriction" gives both routes as
    // equivalent; engines pick whichever admits.
    //
    // The dependency DAG is read from the AST (`_jointVariateDeps`)
    // because the analyzer's bindings map isn't populated yet at this
    // pre-pass — only the canonical `lawof(record(name = ident, ...))`
    // shape is recognised, which covers the common restrict
    // use cases (posteriors, max-likelihood predictives).
    const Mfields = _jointMeasureFields(measureArg.name, ast);
    const variateDeps = _jointVariateDeps(measureArg.name, ast);
    const complementFields: string[] = Mfields
      ? [...Mfields].filter((f: string) => !fieldNames.includes(f))
      : [];
    let useComplementRoute = false;
    if (variateDeps && complementFields.length > 0) {
      const selectorAdmits = _disintegrateAdmissible(variateDeps, fieldNames);
      const complementAdmits = _disintegrateAdmissible(variateDeps, complementFields);
      // If selector admits, prefer it (matches spec equivalence and
      // existing posterior-construction patterns). If selector
      // doesn't admit but complement does, take the complement route
      // — this is the "max-likelihood predictive" case where x covers
      // upstream variates.
      if (!selectorAdmits && complementAdmits) useComplementRoute = true;
    }

    const kernelAnon   = _freshRestrictAnon('kernel');
    const marginalAnon = _freshRestrictAnon('marginal');

    if (useComplementRoute) {
      // Complement-disintegration expansion (spec §06):
      //   kernel, marginal = disintegrate([...complement...], M)
      //   nu = logweighted(logdensityof(marginal, x), kernel(x))
      //
      // `disintegrate(S, M)` returns `(kernel, base_measure)` where
      // the kernel produces the SELECTED variates given the
      // complement and the base_measure is the marginal on the
      // complement-of-selector. With selector set to the complement
      // of x.fields, the kernel goes from x's variates to the
      // unobserved (complement) variates — the forward model when M
      // is generative — and the marginal lives on x's variates.
      //
      // `kernel(x)` alone is the NORMALIZED conditional measure
      // p(unobserved | x) — total mass 1. `restrict(M, x)` is
      // defined as the NON-NORMALIZED conditional with total mass
      // equal to the marginal density of M at x — the evidence /
      // marginal-likelihood factor. So the kernel application must
      // be reweighted by the scalar `densityof(marginal, x)`. The
      // selector-route equivalent (bayesupdate) gets this factor
      // automatically via the per-point reweighting of the marginal
      // by the likelihood; the complement route's reweighting is a
      // single scalar by construction (x is fixed when the kernel
      // is applied). Emitted in log-space (`logweighted` +
      // `logdensityof`) for numerical robustness when the marginal
      // density is tiny.
      const compSelectorElems = complementFields.map(
        (n: string) => AST.StringLiteral(n, n, sloc));
      const compSelector = AST.ArrayLiteral(compSelectorElems, sloc);
      const disintCall = AST.CallExpr(
        AST.Identifier('disintegrate', sloc),
        [compSelector, measureArg],
        sloc);
      const disintStmt = AST.AssignStatement(
        [AST.Identifier(kernelAnon, sloc), AST.Identifier(marginalAnon, sloc)],
        disintCall,
        sloc);
      // Apply the kernel to x. The user-call substitution path
      // (lift's inlineUserCall, then the classifier) handles
      // `kernel_anon(x)` exactly as it would handle any
      // user-written kernel-application call site.
      const applyCall = AST.CallExpr(
        AST.Identifier(kernelAnon, sloc),
        [xExpr],
        sloc);
      // For `name = restrict(M, x)`: emit the spec-compliant form
      //     logweighted(logdensityof(marginal, x), kernel(x))
      // For `name = normalize(restrict(M, x))`: drop the scalar weight
      // layer — the peephole `normalize(weighted(scalar, M))` ≡
      // `normalize(M)` lets the constant `logdensityof(marginal, x)`
      // factor cancel under normalize, so computing it is dead code.
      // Emit `name = normalize(kernel(x))` directly. `normalize(...)`
      // is preserved because the underlying kernel application may
      // have non-unit total mass (depends on the disintegration
      // convention; FlatPPL spec §06 doesn't pin the kernel to
      // probability measures, only that jointchain(marginal, kernel) ≡
      // the original joint).
      let resultExpr: any;
      if (wrapInNormalize) {
        resultExpr = AST.CallExpr(
          AST.Identifier('normalize', sloc),
          [applyCall],
          sloc);
      } else {
        // Scalar marginal log-density `logdensityof(marginal, x)`.
        // Lifted through `logweighted(...)` it shifts the result
        // measure's log total mass by exactly this scalar without
        // changing per-atom samples (uniform scaling).
        const logDensityScalar = AST.CallExpr(
          AST.Identifier('logdensityof', sloc),
          [AST.Identifier(marginalAnon, sloc), xExpr],
          sloc);
        resultExpr = AST.CallExpr(
          AST.Identifier('logweighted', sloc),
          [logDensityScalar, applyCall],
          sloc);
      }
      const userStmt = AST.AssignStatement(stmt.names, resultExpr, stmt.loc);
      newBody.push(disintStmt);
      newBody.push(userStmt);
      continue;
    }

    // Selector-disintegration expansion (existing default route).
    //
    // Selector: array literal of string literals (works uniformly for
    // single- and multi-field selectors). The disintegrate detector
    // also accepts a bare StringLiteral for "bare value" selectors,
    // but `restrict` always operates on a record of observations, so
    // the array form is the right canonical shape.
    const selectorElems = fieldNames.map(
      (n: string) => AST.StringLiteral(n, n, sloc));
    const selector = AST.ArrayLiteral(selectorElems, sloc);

    // Statement 1: kernel_anon, marginal_anon = disintegrate(selector, M)
    const disintCall = AST.CallExpr(
      AST.Identifier('disintegrate', sloc),
      [selector, measureArg],
      sloc);
    const disintStmt = AST.AssignStatement(
      [AST.Identifier(kernelAnon, sloc), AST.Identifier(marginalAnon, sloc)],
      disintCall,
      sloc);

    // Statement 2: <user name> = bayesupdate(likelihoodof(kernel_anon, x), marginal_anon)
    // For the normalize-wrapped form, the bayesupdate is wrapped in
    // an outer normalize (no peephole shortcut for the selector
    // route — bayesupdate's per-atom reweighting doesn't collapse
    // under normalize the way the complement route's single scalar
    // weight does).
    const likelihoodCall = AST.CallExpr(
      AST.Identifier('likelihoodof', sloc),
      [AST.Identifier(kernelAnon, sloc), xExpr],
      sloc);
    const bayesCall = AST.CallExpr(
      AST.Identifier('bayesupdate', sloc),
      [likelihoodCall, AST.Identifier(marginalAnon, sloc)],
      sloc);
    const finalCall = wrapInNormalize
      ? AST.CallExpr(AST.Identifier('normalize', sloc), [bayesCall], sloc)
      : bayesCall;
    const userStmt = AST.AssignStatement(stmt.names, finalCall, stmt.loc);

    newBody.push(disintStmt);
    newBody.push(userStmt);
  }
  ast.body = newBody;
}

// `locscale(m, shift, scale)` expansion (spec §06 "locscale").
//
// `locscale(m, shift, scale)` is shorthand for the affine pushforward
// `pushfwd(x -> scale * x + shift, m)`. We expand it before analysis to
// a synthetic `bijection(fwd, inv, logvolume)` driving a `pushfwd`, so
// the existing pushforward density (density.walkPushfwd, AST bijection
// path) and sampler (matPushfwd) handle it with no new derivation kind —
// exactly as `restrict` reduces to disintegrate/bayesupdate. The
// synthesised `__locscale_*` names carry the conventional `__`-prefix for
// auto-generated bindings (spec §04); like `__restrict_*` they are NOT
// elided inside this engine — they appear in the exported `symbols` array
// the same way restrict's synthetics do (any LSP/outline elision happens in
// the downstream consumer, not here).
//
//   __locscale_fwd_N = scale * _x + shift        # functionof, arg x
//   __locscale_inv_N = (_y - shift) / scale       # functionof, arg y
//   __locscale_lv_N  = log(abs(scale))            # forward log|J|, 0-arg
//   __locscale_bij_N = bijection(fwd, inv, lv)
//   <original RHS with locscale(m, .., ..) → pushfwd(__locscale_bij_N, m)>
//
// Scope: SCALAR shift and SCALAR scale only. A non-scalar literal shift or
// scale is rejected with a diagnostic in _buildLocscalePushfwd; a matrix or
// vector affine map should use pushfwd directly (spec §06). (Multivariate
// locscale via type-inference-directed routing is tracked separately.)
let _locscaleCounter = 0;

const _LOCSCALE_LITERAL_TYPES = new Set([
  'NumberLiteral', 'StringLiteral', 'BoolLiteral', 'ArrayLiteral', 'TupleLiteral']);

function expandLocscaleStatements(ast: any, diagnostics: any[]) {
  if (!ast || !ast.body) return;
  const newBody: any[] = [];
  for (const stmt of ast.body) {
    if (stmt.type !== 'AssignStatement' || !stmt.value) {
      newBody.push(stmt);
      continue;
    }
    const synth: any[] = [];
    const rewritten = _rewriteLocscale(stmt.value, synth, diagnostics, ast.body);
    for (const s of synth) newBody.push(s);
    newBody.push(rewritten === stmt.value
      ? stmt
      : AST.AssignStatement(stmt.names, rewritten, stmt.loc));
  }
  ast.body = newBody;
}

// Recursively rewrite every `locscale(...)` call node within `node`,
// appending the synthetic bijection/functionof binding statements to
// `synth` (in dependency order). Returns the rewritten node, and crucially
// returns `node` *unchanged* (same reference, no allocation) when nothing
// was rewritten — so a model with no locscale pays no per-node clone cost.
//
// The generic structural recursion is REQUIRED, not speculative: `x ~ M`
// desugars to `x = draw(M)` at parse time (parser.ts), so a locscale written
// as `b ~ locscale(...)` arrives nested inside `draw(...)`; nested positions
// inside iid/superpose/etc. are reached the same way.
function _rewriteLocscale(node: any, synth: any[], diagnostics: any[], body: any[]): any {
  if (node == null || typeof node !== 'object') return node;
  if (Array.isArray(node)) {
    let out: any = node;
    for (let i = 0; i < node.length; i++) {
      const r = _rewriteLocscale(node[i], synth, diagnostics, body);
      if (r !== node[i]) {
        if (out === node) out = node.slice();
        out[i] = r;
      }
    }
    return out;
  }
  if (node.type === 'CallExpr' && node.callee
      && node.callee.type === 'Identifier'
      && node.callee.name === 'locscale') {
    return _buildLocscalePushfwd(node, synth, diagnostics, body);
  }
  // Lazy-clone structural recursion: only spread `node` once a child actually
  // changes, so the no-locscale common case returns the original reference.
  let out: any = node;
  for (const k of Object.keys(node)) {
    if (k === 'loc' || k === 'type') continue;
    const r = _rewriteLocscale(node[k], synth, diagnostics, body);
    if (r !== node[k]) {
      if (out === node) out = { ...node };
      out[k] = r;
    }
  }
  return out;
}

function _buildLocscalePushfwd(call: any, synth: any[], diagnostics: any[], body: any[]): any {
  // Reuse the call's real line/col (marked synthetic) so source-slicing
  // for symbols stays well-formed — mirrors restrict-expand's sloc.
  const sloc = { ...call.loc, synthetic: true, source: 'locscale-expand' };
  const args = call.args || [];
  if (args.some((a: any) => a && a.type === 'KeywordArg')) {
    diagnostics.push({
      severity: 'error',
      message: `locscale() does not accept keyword arguments; pass `
        + `(measure, shift, scale) positionally`,
      loc: call.loc,
    });
    return call;
  }
  if (args.length !== 3) {
    diagnostics.push({
      severity: 'error',
      message: `locscale() takes exactly three positional arguments `
        + `(measure, shift, scale); got ${args.length}`,
      loc: call.loc,
    });
    return call;
  }
  // The base measure must be a measure-producing expression (a distribution
  // call like Normal(...)/StudentT(...) or a bound measure name) — never a
  // literal. Mirrors restrict-expand's measure-arg guard (this combinator
  // wraps the base in pushfwd, so unlike restrict it allows a CallExpr base,
  // not only an Identifier).
  if (args[0] && _LOCSCALE_LITERAL_TYPES.has(args[0].type)) {
    diagnostics.push({
      severity: 'error',
      message: `locscale()'s first argument (the measure) must be a measure `
        + `expression, not a ${args[0].type}`,
      loc: args[0].loc,
    });
    return call;
  }
  // P3: a vector/matrix shift or scale routes through lift's affine-registry
  // lowering (lift.inlineLocscaleAffineLift), NOT the scalar expansion here.
  // Detect the non-scalar forms statically (the analyzer pre-pass runs before
  // type inference, so this mirrors lift's own conservative syntactic gate)
  // and leave the call UNEXPANDED so it survives `analyze()` to reach lift.
  const MATRIXY_OPS = new Set([
    'lower_cholesky', 'cholesky', 'inv', 'transpose', 'rowstack', 'colstack',
    'eye', 'diagm', 'diag']);
  const looksNonScalar = (e: any): boolean => {
    if (!e) return false;
    if (e.type === 'ArrayLiteral' || e.type === 'TupleLiteral') return true;
    if (e.type === 'CallExpr' && e.callee && e.callee.type === 'Identifier'
        && MATRIXY_OPS.has(e.callee.name)) return true;
    if (e.type === 'Identifier') {
      // 1-level binding lookup within this program body.
      for (const st of body || []) {
        if (st.type === 'AssignStatement' && st.names
            && st.names.some((n: any) => n.name === e.name)) {
          return looksNonScalar(st.value);
        }
      }
      /* c8 ignore start */
      // Defensive loop-completion: a bound identifier matches an
      // AssignStatement (or a `~`-statement carrying its name) and returns
      // above; only an unbound name runs the loop to completion.
    }
    /* c8 ignore stop */
    return false;
  };
  // The registry affine-density path that lift routes a non-scalar locscale
  // through requires the base to score as `iid(<scalar dist>, D)`. Detect an
  // iid base (an `iid(...)` call, or a 1-level ref to such) so we can tell a
  // routable locscale from a deferred one.
  const baseIsIid = (e: any): boolean => {
    if (!e) return false;
    if (e.type === 'CallExpr' && e.callee && e.callee.type === 'Identifier'
        && e.callee.name === 'iid') return true;
    if (e.type === 'Identifier') {
      for (const st of body || []) {
        if (st.type === 'AssignStatement' && st.names
            && st.names.some((n: any) => n.name === e.name)) {
          return baseIsIid(st.value);
        }
      }
      /* c8 ignore start */
      // Defensive loop-completion: only an unbound base name runs the loop to
      // completion; a bound name matches and returns above.
    }
    /* c8 ignore stop */
    return false;
  };
  if (looksNonScalar(args[1]) || looksNonScalar(args[2])) {
    // Only an iid base routes through lift's affine-registry lowering. For
    // any other base — including MvNormal, itself a pushfwd∘pushfwd whose
    // affine registry entries do not auto-compose — emit a clean diagnostic
    // HERE (the pre-pass owns the diagnostic channel; lift has none) rather
    // than letting an unhandled locscale IR node survive to materialisation
    // and fail with a cryptic registry shape error. This keeps the iid-base
    // detection and lift's gate in agreement: exactly the iid-base forms are
    // left to survive; everything else multivariate is diagnosed.
    if (!baseIsIid(args[0])) {
      diagnostics.push({
        severity: 'error',
        message: `locscale() with a vector/matrix scale requires an iid(<dist>, D) `
          + `base (it lowers to an affine-registry pushfwd); for other bases — `
          + `including MvNormal — compose with pushfwd directly (spec §06)`,
        loc: call.loc,
      });
      return call;
    }
    // Base IS iid → lift's affine-registry gate handles it. But lift's gate
    // only fires for a square [D,D] scale + matching [D] shift; an
    // unroutable shape would otherwise leave an unhandled locscale that
    // silently drops the binding at materialisation. The pre-pass can
    // syntactically validate the LITERAL forms here (it runs before type
    // inference, so ref/op shapes defer to lift). For a literal scale we
    // require a square [D,D] matrix and a literal shift (if literal) of
    // length D — anything else is diagnosed now, keeping the invariant
    // "no locscale survives unhandled to materialisation" for literal forms.
    const resolveLit = (e: any): any => {
      if (!e) return null;
      if (e.type === 'ArrayLiteral') return e;
      if (e.type === 'Identifier') {
        for (const st of body || []) {
          if (st.type === 'AssignStatement' && st.names
              && st.names.some((n: any) => n.name === e.name)) {
            return resolveLit(st.value);
          }
        }
      /* c8 ignore start */
      // Defensive loop-completion: only an unbound scale|shift name runs the
      // loop to completion; a bound name matches and returns above. (The tail
      // `return null` below IS reached — e.g. a named lower_cholesky ref.)
      }
      /* c8 ignore stop */
      return null;
    };
    const litRows = (e: any): any[] | null => {
      const lit = resolveLit(e);
      return lit ? lit.elements : null;
    };
    // Static iid base count K: base is `iid(<dist>, K)` with a NumberLiteral
    // 2nd positional arg, or a 1-level ref to such. Returns null when K is
    // not a statically-known integer (then we defer the cross-check to lift).
    const baseIidCount = (e: any): number | null => {
      if (!e) return null;
      if (e.type === 'CallExpr' && e.callee && e.callee.type === 'Identifier'
          && e.callee.name === 'iid' && Array.isArray(e.args)) {
        const positional = e.args.filter((a: any) => !(a && a.type === 'KeywordArg'));
        const sizeArg = positional[1];
        if (sizeArg && sizeArg.type === 'NumberLiteral'
            && Number.isInteger(sizeArg.value)) {
          return sizeArg.value;
        }
        return null;
      }
      if (e.type === 'Identifier') {
        for (const st of body || []) {
          if (st.type === 'AssignStatement' && st.names
              && st.names.some((n: any) => n.name === e.name)) {
            return baseIidCount(st.value);
          }
        }
      /* c8 ignore start */
      // Defensive tail: baseIidCount is only called after baseIsIid(args[0])
      // returned true, which for an Identifier base means the same 1-level loop
      // found an AssignStatement binding to iid above; an inline iid base hits
      // the CallExpr branch. The fall-through is unreachable for a routed
      // locscale, so the loop-completing brace + tail return aren't exercised.
      }
      return null;
      /* c8 ignore stop */
    };
    const scaleRows = litRows(args[2]);
    if (scaleRows) {
      const D = scaleRows.length;
      const squareMatrix = D >= 1 && scaleRows.every((r: any) =>
        r && r.type === 'ArrayLiteral' && r.elements.length === D);
      const shiftRows = litRows(args[1]);
      const shiftOk = shiftRows == null
        || (shiftRows.length === D
            && shiftRows.every((s: any) => !(s && s.type === 'ArrayLiteral')));
      if (!squareMatrix || !shiftOk) {
        diagnostics.push({
          severity: 'error',
          message: `locscale() with a matrix scale over an iid base requires a `
            + `square [D, D] scale and a length-D vector shift (the affine-registry `
            + `pushfwd contract); for other shapes use pushfwd directly (spec §06)`,
          loc: call.loc,
        });
        return call;
      }
      // Reconcile the iid base dimension against the scale dimension: the
      // affine map scale@x + shift needs a length-D base. Only enforce when
      // BOTH are statically known (dynamic K is left to lift's gate / the
      // buildDerivations safety net).
      const K = baseIidCount(args[0]);
      if (K != null && K !== D) {
        diagnostics.push({
          severity: 'error',
          message: `locscale(): iid base dimension (${K}) does not match the `
            + `scale matrix dimension (${D}); the affine map scale@x+shift requires `
            + `a length-${D} base`,
          loc: call.loc,
        });
        return call;
      }
    }
    return call;  // iid base, routable shape: survive to lift's lowering
  }
  // The base measure may itself contain a nested locscale.
  const mExpr = _rewriteLocscale(args[0], synth, diagnostics, body);
  const shiftExpr = args[1];
  const scaleExpr = args[2];
  const clone = (n: any) => structuredClone(n);

  const n = _locscaleCounter++;
  const fwdName = `__locscale_fwd_${n}`;
  const invName = `__locscale_inv_${n}`;
  const lvName  = `__locscale_lv_${n}`;
  const bijName = `__locscale_bij_${n}`;

  // fwd: functionof(scale * _x + shift, x = _x) — mirrors the parser's
  // single-arg lambda desugaring (functionof body + Placeholder kwarg).
  const fwdBody = AST.BinaryExpr('+',
    AST.BinaryExpr('*', clone(scaleExpr), AST.Placeholder('x', sloc), sloc),
    clone(shiftExpr), sloc);
  const fwd = AST.CallExpr(AST.Identifier('functionof', sloc),
    [fwdBody, AST.KeywordArg('x', AST.Placeholder('x', sloc), sloc)], sloc);

  // inv: functionof((_y - shift) / scale, y = _y)
  const invBody = AST.BinaryExpr('/',
    AST.BinaryExpr('-', AST.Placeholder('y', sloc), clone(shiftExpr), sloc),
    clone(scaleExpr), sloc);
  const inv = AST.CallExpr(AST.Identifier('functionof', sloc),
    [invBody, AST.KeywordArg('y', AST.Placeholder('y', sloc), sloc)], sloc);

  // logvolume: the forward Jacobian of x -> scale*x + shift is constant
  // |scale|, so log|J| = log(abs(scale)). When `scale` is a numeric
  // literal we fold it to a scalar NumberLiteral (the cheapest density
  // path); otherwise a 0-arg `functionof(log(abs(scale)))` constant.
  let lvArg: any;
  if (scaleExpr.type === 'NumberLiteral') {
    const sv = +scaleExpr.value;
    if (sv === 0 || !Number.isFinite(sv)) {
      diagnostics.push({
        severity: 'error',
        message: `locscale() scale must be a nonzero finite value `
          + `(got ${scaleExpr.value}); a zero or non-finite scale is a `
          + `degenerate, non-invertible affine map`,
        loc: scaleExpr.loc,
      });
      return call;
    }
    const v = Math.log(Math.abs(sv));
    lvArg = AST.NumberLiteral(v, String(v), sloc);
  } else {
    const lvBody = AST.CallExpr(AST.Identifier('log', sloc),
      [AST.CallExpr(AST.Identifier('abs', sloc), [clone(scaleExpr)], sloc)], sloc);
    const lv = AST.CallExpr(AST.Identifier('functionof', sloc), [lvBody], sloc);
    synth.push(AST.AssignStatement([AST.Identifier(lvName, sloc)], lv, sloc));
    lvArg = AST.Identifier(lvName, sloc);
  }

  const bij = AST.CallExpr(AST.Identifier('bijection', sloc),
    [AST.Identifier(fwdName, sloc), AST.Identifier(invName, sloc), lvArg], sloc);

  synth.push(AST.AssignStatement([AST.Identifier(fwdName, sloc)], fwd, sloc));
  synth.push(AST.AssignStatement([AST.Identifier(invName, sloc)], inv, sloc));
  synth.push(AST.AssignStatement([AST.Identifier(bijName, sloc)], bij, sloc));

  return AST.CallExpr(AST.Identifier('pushfwd', sloc),
    [AST.Identifier(bijName, sloc), mExpr], sloc);
}

/**
 * Analyze a parsed AST.
 * Returns { bindings, diagnostics, symbols }.
 *
 * @param {object} ast - Program AST node
 * @param {string} source - original source text (for expression slicing)
 */
function analyze(ast: any, source: string, opts?: any) {
  // `opts` (optional) carries the multi-file compilation context (spec
  // §04 Module composition): `opts.modulePath` is this module's own
  // resolved path (importer path for its `load_module` deps); `opts.modules`
  // is the registry of already-compiled sibling modules (resolved-path →
  // compiled module) that cross-module type inference reads. Both are
  // absent for a standalone single-file compile.
  const moduleCtx = opts || {};
  const diagnostics: any[] = [];
  const bindings = new Map<string, any>();
  const symbols: any[] = [];
  const definedNames = new Set<string>();

  // Pre-pass: expand `restrict(M, x)` into the equivalent
  // disintegrate + likelihoodof + bayesupdate chain per spec §06
  // "Measure restriction". The rewrite is purely structural — every
  // op it introduces (disintegrate, likelihoodof, bayesupdate) is
  // already classified and materialised, so restrict needs no
  // derivation kind of its own.
  expandRestrictStatements(ast, diagnostics);

  // Pre-pass: expand `locscale(m, shift, scale)` into a synthetic affine
  // `bijection` + `pushfwd` per spec §06 "locscale". Like restrict, the
  // ops it introduces (bijection, functionof, pushfwd) are already
  // first-class, so locscale needs no derivation kind of its own.
  expandLocscaleStatements(ast, diagnostics);

  // First pass: collect all defined names
  for (const stmt of ast.body) {
    if (stmt.type !== 'AssignStatement') continue;
    for (const nameNode of stmt.names) {
      if (definedNames.has(nameNode.name)) {
        diagnostics.push({
          severity: 'error',
          message: `Duplicate variable name '${nameNode.name}'`,
          loc: nameNode.loc,
        });
      }
      definedNames.add(nameNode.name);
    }
  }

  // Second pass: classify, extract deps, build bindings
  for (const stmt of ast.body) {
    if (stmt.type !== 'AssignStatement') continue;

    const stmtType = classifyStatement(stmt.value);
    diagnostics.push(...validateSpecialOperation(stmt.value));
    validateHolesAndPlaceholders(stmt.value, diagnostics);
    validateIndexing(stmt.value, diagnostics);
    const { deps, callDeps, bodyDeps, paramSourceDeps } = collectDeps(stmt.value, definedNames);
    const rhs = sliceSource(source, stmt.value.loc);

    // Remove self-references
    for (const nameNode of stmt.names) {
      deps.delete(nameNode.name);
      callDeps.delete(nameNode.name);
      bodyDeps.delete(nameNode.name);
      paramSourceDeps.delete(nameNode.name);
    }

    // Check for undefined references
    const refs = collectIdentRefs(stmt.value);
    for (const ref of refs) {
      if (!definedNames.has(ref.name) && !isKnownName(ref.name)) {
        diagnostics.push({
          severity: 'warning',
          message: `Undefined variable '${ref.name}'`,
          loc: ref.loc,
        });
      }
    }

    // Build binding info for each name
    for (const nameNode of stmt.names) {
      const info = {
        name: nameNode.name,
        names: stmt.names.map((n: any) => n.name),
        line: stmt.loc.start.line,
        rhs,
        type: stmtType,
        deps: [...deps],
        callDeps: [...callDeps],
        // For functionof / kernelof bindings these split out the body's
        // closure captures from the kwarg-RHS value-set declarations
        // (engine-concepts §8). For non-reification bindings the body
        // bucket carries everything and paramSource is empty. Walkers
        // that need the spec's two-scope semantics (absorbedPhaseOf,
        // future inlineOnce cleanup) consume `bodyDeps`.
        bodyDeps: [...bodyDeps],
        paramSourceDeps: [...paramSourceDeps],
        node: stmt,
        nameLoc: nameNode.loc,
      };
      bindings.set(nameNode.name, info);

      // Build symbol for outline
      const kindMap: Record<string, string> = {
        draw: 'Variable', input: 'Variable', call: 'Variable',
        lawof: 'Function', functionof: 'Function', kernelof: 'Function', fn: 'Function',
        likelihood: 'Variable', bayesupdate: 'Variable',
        literal: 'Constant', module: 'Module', table: 'Variable',
      };
      symbols.push({
        name: nameNode.name,
        kind: kindMap[stmtType] || 'Variable',
        type: stmtType,
        loc: stmt.loc,
        nameLoc: nameNode.loc,
      });
    }
  }

  // Third pass: detect disintegrate-decompositions, tag results, validate selectors.
  for (const stmt of ast.body) {
    if (stmt.type !== 'AssignStatement') continue;
    const info = detectDisintegration(stmt, bindings);
    if (!info) continue;

    // Validate selector fields exist in the joint's record (when the joint
    // is in a form whose fields we can statically enumerate). For positional
    // jointchain or other forms, the rewriter's Unsupported reason carries
    // the equivalent diagnostic — emitted below.
    if (info.jointFields) {
      for (const field of info.selectorFields) {
        if (!info.jointFields.has(field)) {
          diagnostics.push({
            severity: 'error',
            message: `disintegrate: selector field '${field}' not found in joint measure '${info.jointName}'`,
            loc: info.selectorLoc,
          });
        }
      }
    }

    // Compute the structural-disintegration Plan first; downstream tagging
    // depends on whether the rewriter could resolve the joint structurally.
    const jointBinding = bindings.get(info.jointName);
    let plan: any = null;
    if (jointBinding && jointBinding.node && jointBinding.node.value) {
      plan = disintegratePlan(
        jointBinding.node.value, info.selectorFields, bindings,
        { seen: new Set(), source: info.jointName,
          selectorBareString: info.selectorBareString });
    }

    const kernel = bindings.get(info.kernelName);
    const prior  = bindings.get(info.priorName);

    // Only tag the result bindings as kernel/prior of a structural
    // disintegration when the rewriter actually resolved one. Unsupported
    // plans fall back to the plain dep trace via the literal RHS.
    const resolved = plan && (plan.kind === 'synthesized' || plan.kind === 'delegate');
    if (resolved) {
      // The kernel-side result classifies as 'functionof' when its
      // underlying body is a measure (per FlatPPL spec §sec:kernelof:
      // `kernelof(x, ...)` requires `x` to NOT be a measure; reifying a
      // measure-typed body uses `functionof`). For a delegate plan we
      // mirror the delegate target's surface type so that
      // forward_kernel2 = disintegrate("obs", joint_model)
      // displays the same keyword as the user-written forward_kernel
      // it recovers. For a synthesized plan we read the synthesized
      // expression's keyword directly — disintegrate.js already picks
      // it via isMeasureExpr.
      if (kernel) {
        kernel.type = kernelTypeForPlan(plan, bindings);
        kernel.disintegrateRole = { kind: 'kernel', ...info };
        kernel.disintegratePlan = plan;
      }
      if (prior) {
        prior.type = 'lawof';
        prior.disintegrateRole = { kind: 'prior', ...info };
        prior.disintegratePlan = plan;
      }

      if (plan.kind === 'synthesized') {
        if (kernel) attachEffectiveRhs(kernel, plan.kernel, definedNames);
        if (prior)  attachEffectiveRhs(prior,  plan.prior,  definedNames);
      } else /* delegate */ {
        // Render the result identically to the delegate target — this is
        // the "the disintegration recovered an existing binding" case.
        if (kernel) attachDelegate(kernel, plan.kernel.binding, bindings);
        if (prior)  attachDelegate(prior,  plan.prior.binding,  bindings);
      }
    } else if (plan) {
      // Keep the Plan around (even Unsupported) so the renderer or a
      // future diagnostic surface can read its reason. Don't change the
      // binding's type — fall back to plain dep trace.
      if (kernel) kernel.disintegratePlan = plan;
      if (prior)  prior.disintegratePlan  = plan;
    }
  }

  // Multi-LHS pass: rewrite `a, b, ... = <call returning tuple>` so each
  // named binding's effective RHS is `tuple_get(<shared-call>, <slot>)`,
  // backed by one synthetic shared binding that holds the original
  // call. Per spec §sec:random's `rand(rstate, m)` returns a tuple
  // `(value, new_rstate)`; this generalises to any tuple-returning
  // RHS so multi-LHS is a uniform feature.
  //
  // We skip groups already claimed by the disintegrate pass (those
  // bindings carry a `disintegrateRole` and a per-name effectiveValue
  // synthesised from the disintegration plan — overwriting that would
  // lose the kernelof / lawof structure that downstream rendering and
  // sampling depend on). All other multi-LHS groups go through this
  // rewrite; the resulting tuple_get IR is recognised as evaluable by
  // sampler.evaluateExpr and the orchestrator's chain classifier.
  for (const stmt of ast.body) {
    if (stmt.type !== 'AssignStatement') continue;
    if (!stmt.names || stmt.names.length < 2) continue;
    const groupBindings = stmt.names.map((n: any) => bindings.get(n.name)).filter(Boolean);
    if (groupBindings.length === 0) continue;
    // Skip groups owned by another pass. Two markers cover the cases:
    //   - effectiveValue set: a prior rewrite (resolved disintegrate)
    //     has already attached the per-name view we'd be clobbering.
    //   - disintegratePlan set: the disintegrate pass detected the
    //     call shape and recorded a Plan (resolved or Unsupported);
    //     either way the binding's RHS is a `disintegrate(...)` call
    //     that isn't tuple-typed, so a tuple_get rewrite would just
    //     produce a type error.
    if (groupBindings.some((b: any) => b.effectiveValue || b.disintegratePlan)) continue;

    const sloc = stmt.loc && stmt.loc.start
      ? `${stmt.loc.start.line + 1}:${stmt.loc.start.col + 1}`
      : 'anon';
    const synName = `%mlhs:${sloc}`;
    const synLoc = AST.synthLoc('multi-LHS');

    // Insert the synthetic shared binding holding the original RHS.
    // It needs to look enough like a regular analyzer binding that
    // phase computation, lowering, and the orchestrator pick it up.
    // Type is read from the original RHS via classifyStatement so
    // e.g. a `rand(...)` call still classifies as 'call'.
    const synStmtType = classifyStatement(stmt.value);
    const synStmt = {
      type:  'AssignStatement',
      names: [AST.Identifier(synName, synLoc)],
      value: stmt.value,
      loc:   stmt.loc,
    };
    const synDeps = collectDeps(stmt.value, definedNames);
    const synBinding = {
      name:     synName,
      names:    [synName],
      line:     stmt.loc.start.line,
      type:     synStmtType,
      deps:     [...synDeps.deps],
      callDeps: [...synDeps.callDeps],
      node:     synStmt,
      nameLoc:  synLoc,
      synthetic: true,
    };
    bindings.set(synName, synBinding);
    definedNames.add(synName);

    // Per spec §05 the decomposition is by position (or by record-
    // field-order). Lowering depends on the RHS's shape:
    //   - array literal / vector-producing call → `get(shared, i)`
    //     (1-based)
    //   - record literal → `get_field(shared, name_i)`
    //   - tuple-producing call (rand, anything returning a tuple) →
    //     `tuple_get(shared, i)` (0-based; engine-internal)
    // The shape recogniser inspects the RHS AST conservatively;
    // unrecognised forms fall back to `tuple_get`. That covers the
    // pre-existing rand() case AND any future tuple-returning ops,
    // while routing array/record decomposition through the standard
    // value-access ops.
    const decomp = _multiLHSAccessor(stmt.value, synName, synLoc, bindings);
    for (let i = 0; i < stmt.names.length; i++) {
      const nameNode = stmt.names[i];
      const b = bindings.get(nameNode.name);
      if (!b) continue;
      const accessorCall = decomp.makeCall(i, nameNode.name);
      attachEffectiveRhs(b, accessorCall, definedNames);
    }
  }

  // Fourth pass: compute phases (stochastic | parameterized | fixed) by
  // ancestor analysis, per spec.
  const phases = computePhases(bindings);
  for (const [name, phase] of phases) {
    const b = bindings.get(name);
    if (b) b.phase = phase;
  }

  // Phase check: per spec §04 sec:functionof, "Boundary inputs themselves
  // may be of parametric or stochastic phase, but not fixed phase."
  // The rule says a closed-form constant can't be lifted into a
  // function/kernel input — it's already a value, there's nothing to
  // parameterise over. Without this diagnostic, the engine silently
  // produces wrong substitutions: `f = functionof(d, theta = c)` with
  // `c = 2.0` and `d = c + 1` leaves d's body unsubstituted (no 'c' in
  // it after lift), so `f(5)` evaluates to `c + 1 = 3` and ignores the
  // user's 5 entirely.
  //
  // We only check Identifier-form boundary values (the common surface
  // syntax). Placeholders are inherently parametric per spec; complex
  // expression boundaries are rejected at lower-time.
  for (const [, b] of bindings) {
    if (b.type !== 'functionof' && b.type !== 'kernelof') continue;
    const callExpr = b.node && b.node.value;
    if (!callExpr || callExpr.type !== 'CallExpr') continue;
    const args = callExpr.args || [];
    for (let i = 1; i < args.length; i++) {
      const arg = args[i];
      if (!arg || arg.type !== 'KeywordArg') continue;
      if (!arg.value || arg.value.type !== 'Identifier') continue;
      const refName = arg.value.name;
      const target = bindings.get(refName);
      if (!target) continue;  // unbound identifier is caught elsewhere
      if (target.phase === 'fixed') {
        diagnostics.push({
          severity: 'error',
          message: `Boundary input '${arg.name}' of ${b.type} references '${refName}', `
            + `which has fixed phase. Spec §04 (sec:functionof) requires boundary `
            + `inputs to be of parametric or stochastic phase — fixed values are `
            + `closed over by the reified callable, not lifted into its signature.`,
          loc: arg.loc || arg.value.loc,
        });
      }
    }
  }

  // Phase check: per spec §07, `checked(value, condition)` requires
  // `condition` to be a fixed-phase boolean — it is "evaluated at
  // load/inference time" and a false result is a STATIC error. A
  // parametric / stochastic condition can't be settled at load time, so
  // the assertion semantics don't apply. This is the phase half of the
  // contract (validated here, the phase authority, alongside the
  // functionof-boundary check above); arg shape is validated earlier in
  // `validateSpecialOperation`, and the condition-is-boolean type check
  // lives in typeinfer. Walk every binding's RHS (a `checked` can nest,
  // e.g. `y = 2 * checked(x, condition = ...)`).
  {
    const condPhaseOf = (name: string) => phases.get(name) || 'fixed';
    const checkCheckedCondition = (ast: any) => {
      if (!ast || typeof ast !== 'object') return;
      if (ast.type === 'CallExpr' && ast.callee
          && ast.callee.type === 'Identifier' && ast.callee.name === 'checked') {
        const args = ast.args || [];
        let condAst: any = null;
        let condLoc: any = ast.loc;
        for (const a of args) {
          if (a && a.type === 'KeywordArg' && a.name === 'condition') {
            condAst = a.value;
            condLoc = a.loc || (a.value && a.value.loc) || ast.loc;
          }
        }
        if (!condAst) {
          // Positional condition: the second positional (non-kwarg) arg.
          const positional = args.filter((a: any) => !(a && a.type === 'KeywordArg'));
          if (positional.length >= 2) { condAst = positional[1]; condLoc = condAst.loc; }
        }
        if (condAst) {
          const ph = phaseOfAstExpr(condAst, bindings, condPhaseOf);
          if (ph !== 'fixed') {
            diagnostics.push({
              severity: 'error',
              message: `checked(): condition must be fixed-phase — spec §07 evaluates it `
                + `at load/inference time — but it has ${ph} phase.`,
              loc: condLoc,
            });
          }
        }
      }
      for (const k in ast) {
        if (k === 'loc') continue;
        const v = ast[k];
        if (Array.isArray(v)) v.forEach(checkCheckedCondition);
        else if (v && typeof v === 'object') checkCheckedCondition(v);
      }
    };
    for (const [, b] of bindings) {
      if (b.node && b.node.value) checkCheckedCondition(b.node.value);
    }
  }

  // Lower to FlatPIR-aligned in-memory module. The LoweredModule is
  // the single source of truth for the program's executable form;
  // all subsequent passes (type inference now, derivation building
  // later) operate on it. The original AST stays in `bindings.node`
  // for source-level concerns (DAG display, source-located
  // diagnostics).
  const pir = require('./pir.ts');
  const loweredModule = pir.lowerToModule(bindings, { modulePath: moduleCtx.modulePath });

  // Alias resolution (engine-concepts §19 / spec §04 "Aliasing is
  // just assignment"). Bindings whose RHS is a single ref node — pure
  // aliases like `breit_wigner = hepphys.X` or `theta = some_param`
  // — get canonicalised here: every `(%ref self <alias>)` elsewhere
  // in the module rewrites to the alias's canonical target. Aliases
  // survive in the binding map (tagged `isAlias: true`) for DAG /
  // viewer affordances, but downstream consumers (typeinfer,
  // derivations, materialiser, sampler) see one canonical ref per
  // object. Matches Rust HIR's `use`-path resolution + LLVM @alias /
  // MLIR symbol resolution — one canonical-IR pass instead of an
  // alias-chain walk in every consumer.
  require('./alias-resolution.ts').resolveAliases(loweredModule);

  // Structural type inference (FlatPIR §sec:flatpir). Mutates each
  // lowered binding to set `inferredType` and writes per-call
  // `meta.type` annotations. We mirror inferredType back onto the
  // analyzer-level bindings for consumers that haven't migrated yet.
  //
  // Const-eval shim (engine-concepts §17.1 "resolve, don't rewrite"):
  // a fixed-eval resolver is handed to typeinfer so shape positions
  // can fold computed integers (`iid(M, length(data))` → array([N],
  // elem)) without typeinfer importing the value-mode evaluator
  // directly. The resolver wraps `sampler.evaluateExpr` in a
  // try/catch; if a const expression can't be evaluated (refs to
  // unresolved bindings, stochastic deps), the type stays %dynamic —
  // same fall-through as before this shim landed.
  const fixedEval = require('./fixed-eval.ts');
  const resolveFixed = fixedEval.makeResolver({ loweredModule });
  const typeDiagnostics = require('./typeinfer.ts')
    .inferTypes(loweredModule, { resolveFixed, modules: moduleCtx.modules });
  for (const [name, lb] of loweredModule.bindings) {
    const b = bindings.get(name);
    if (b) b.inferredType = lb.inferredType;
  }
  for (const d of typeDiagnostics) diagnostics.push(d);

  // Validate load-time `load_module(…, input = value)` substitutions
  // against the loaded module's inputs (spec §04 "Load-time
  // substitution"): the LHS must name an input (elementof / external) of
  // the loaded module, and the RHS phase must be compatible — external
  // inputs take fixed values, elementof inputs take parameterized (or
  // fixed) values, and a stochastic value is never substitutable
  // (referential transparency). Runs after type inference so the
  // dependency modules are fully compiled.
  _validateModuleSubstitutions(loweredModule, moduleCtx.modules, diagnostics);

  // Bin diagnostics back onto their bindings so downstream consumers
  // (DAG view, plot pane) can answer "does this binding have an
  // error?" without re-walking the global diagnostic list. First match
  // wins — diagnostics carry a single source location and bindings
  // don't overlap by construction.
  for (const d of diagnostics) {
    if (!d.loc) continue;
    for (const [, b] of bindings) {
      const nl = b.node && b.node.loc;
      if (!nl) continue;
      if (d.loc.start.line >= nl.start.line && d.loc.start.line <= nl.end.line) {
        if (!b.diagnostics) b.diagnostics = [];
        b.diagnostics.push(d);
        break;
      }
    }
  }

  return { bindings, loweredModule, diagnostics, symbols };
}

// Phase ordering helper: fixed < parameterized < stochastic.
function _maxPhase(a: string, b: any): string {
  const rank: Record<string, number> = { fixed: 0, parameterized: 1, stochastic: 2 };
  const ra = rank[a] != null ? rank[a] : 0;
  const rb = rank[b] != null ? rank[b] : 0;
  return ra >= rb ? a : (b as string);
}

// The phase of a substitution VALUE expression (spec §04 phases). Direct
// phase-bearing ops short-circuit; otherwise the phase is the dominant
// phase of the loading-module bindings the expression references.
function _irPhase(ir: any, loweredModule: any): string {
  if (ir && ir.kind === 'call') {
    if (ir.op === 'draw')      return 'stochastic';
    if (ir.op === 'elementof') return 'parameterized';
    if (ir.op === 'external')  return 'fixed';
  }
  const irShared = require('./ir-shared.ts');
  let ph = 'fixed';
  for (const refName of irShared.collectSelfRefs(ir)) {
    const b = loweredModule.bindings.get(refName);
    ph = _maxPhase(ph, b && b.phase);
  }
  return ph;
}

// Validate `load_module("…", input = value)` substitutions against the
// loaded module's declared inputs (spec §04). Emits diagnostics; mutates
// nothing else.
function _validateModuleSubstitutions(loweredModule: any, modules: any, diagnostics: any[]) {
  if (!modules || modules.size === 0) return;
  const reg = loweredModule.moduleRegistry || {};
  for (const [name, lb] of loweredModule.bindings) {
    const rhs = lb.rhs;
    if (!rhs || rhs.kind !== 'call' || rhs.op !== 'load_module') continue;
    if (!Array.isArray(rhs.assigns) || rhs.assigns.length === 0) continue;
    const path = reg[name] && reg[name].path;
    if (!path) continue;            // unresolved / non-literal path — already diagnosed
    const dep = modules.get(path);
    if (!dep) continue;             // missing source — already diagnosed
    for (const a of rhs.assigns) {
      const inputBinding = dep.loweredModule.bindings.get(a.name);
      const inputRhs = inputBinding && inputBinding.rhs;
      const inputOp = (inputRhs && inputRhs.kind === 'call') ? inputRhs.op : null;
      const loc = (a.value && a.value.loc) || rhs.loc;
      if (inputOp !== 'elementof' && inputOp !== 'external') {
        diagnostics.push({ severity: 'error',
          message: "'" + a.name + "' is not an input (elementof / external) of module '"
            + name + "' (loaded from '" + path + "'); only module inputs may be "
            + 'substituted at load time (spec §04)',
          loc });
        continue;
      }
      const vphase = _irPhase(a.value, loweredModule);
      if (vphase === 'stochastic') {
        diagnostics.push({ severity: 'error',
          message: "cannot bind a stochastic value to module input '" + a.name
            + "' of '" + name + "' — load-time substitution values must be fixed or "
            + 'parameterized (spec §04, referential transparency)',
          loc });
        continue;
      }
      if (inputOp === 'external' && vphase !== 'fixed') {
        diagnostics.push({ severity: 'error',
          message: "external input '" + a.name + "' of module '" + name
            + "' requires a fixed-phase value, got " + vphase
            + ' (spec §04: external ← fixed)',
          loc });
      }
    }
  }
}

/**
 * Test whether a string is a valid public/private/auto-generated binding name.
 *
 * Rejects: reserved names (self, base), bare `_`, placeholder pattern `_x_`,
 * and any name that doesn't match one of the canonical regular-expression
 * patterns from the spec (`docs/04-design.md#sec:binding-names`).
 */
function isValidBindingName(name: any) {
  if (typeof name !== 'string' || name.length === 0) return false;
  if (name === 'self' || name === 'base') return false;
  if (name === '_') return false; // discard, not a renameable target
  // Public:        ^[A-Za-z][A-Za-z0-9_]*$
  // Private:       ^_[A-Za-z]([A-Za-z0-9_]*[A-Za-z0-9])?$
  // Auto-gen:      ^__[A-Za-z0-9][A-Za-z0-9_]*$
  return /^[A-Za-z][A-Za-z0-9_]*$/.test(name)
      || /^_[A-Za-z]([A-Za-z0-9_]*[A-Za-z0-9])?$/.test(name)
      || /^__[A-Za-z0-9][A-Za-z0-9_]*$/.test(name);
}

/**
 * Test whether a string is a valid placeholder source token (with surrounding
 * underscores, e.g. `_par_`).
 */
function isValidPlaceholderText(text: any) {
  return typeof text === 'string'
      && /^_[A-Za-z]([A-Za-z0-9_]*[A-Za-z0-9])?_$/.test(text);
}

/**
 * Plan a rename action at a given cursor position.
 *
 * Walks the AST, identifies what's under the cursor, and returns enough info
 * for a rename provider to act on. Returns null when the position isn't a
 * renameable target (e.g. on a literal, a comment, or a bare-`_` LHS).
 *
 * @param {object} ast - parsed Program AST
 * @param {Map} bindings - analyzer bindings map
 * @param {number} line - 0-based cursor line
 * @param {number} col - 0-based cursor column
 * @returns {{ kind: 'binding', oldName: string, targetLoc, locs: Loc[] }
 *         | { kind: 'placeholder', oldName: string, targetLoc, locs: Loc[] }
 *         | null}
 *
 * For 'binding': `locs` includes the binding's defining nameLoc plus every
 *   Identifier reference site in any other statement's RHS.
 * For 'placeholder': `oldName` is the placeholder *inner* name (without the
 *   surrounding underscores). `locs` includes every Placeholder node that
 *   shares the same nearest enclosing `functionof`/`lawof` scope.
 *   Each loc covers the full `_name_` source span.
 */
// Determine the per-slot accessor for a multi-LHS decomposition
// based on the RHS's AST shape (spec §05 — "Decomposition is by
// position. For records, the field order determines which value
// each name receives; for arrays and tuples, positional index does.
// This is syntactic sugar: it lowers to an assignment followed by
// indexed or field-access bindings.").
//
// Returns `{ makeCall(i, lhsName): CallExpr }`. The default
// (anything we don't structurally recognise as array / record /
// vector) is the engine-internal `tuple_get(shared, i)` — the
// pre-existing convention for tuple-returning calls like `rand()`.
function _multiLHSAccessor(rhsAST: any, synName: string, synLoc: any, bindings: any) {
  // Resolve through identifier indirections (with a cycle guard).
  // `a, b, c = arr` where `arr = [1, 2, 3]` lets us follow `arr`'s
  // binding to its ArrayLiteral RHS and pick the right accessor.
  function resolveAST(node: any, seen: Set<string>): any {
    if (!node) return node;
    if (node.type === 'Identifier' && bindings && bindings.has(node.name)) {
      if (seen.has(node.name)) return node;
      seen.add(node.name);
      const b = bindings.get(node.name);
      const inner = b && b.node && b.node.value;
      if (inner) return resolveAST(inner, seen);
    }
    return node;
  }
  const resolved = resolveAST(rhsAST, new Set());

  // Record-shaped RHS: emit `get_field(shared, name_i)` per name.
  if (resolved && resolved.type === 'CallExpr' && resolved.callee
      && resolved.callee.name === 'record') {
    const fieldNames: string[] = [];
    for (const a of resolved.args || []) {
      if (a && a.type === 'KeywordArg') fieldNames.push(a.name);
    }
    return {
      makeCall: (i: number, lhsName: string) => AST.CallExpr(
        AST.Identifier('get_field', synLoc),
        [
          AST.Identifier(synName, synLoc),
          AST.StringLiteral(fieldNames[i] || lhsName, fieldNames[i] || lhsName, synLoc),
        ],
        synLoc,
      ),
    };
  }
  // Vector-shaped RHS: array literal, vector(...) call, or a call
  // to a vector-producing distribution / measure-algebra op (often
  // wrapped in `draw(...)` from a tilde-binding). Spec §05 says
  // decomposition is by position; the lowered access is `get`
  // (1-based per FlatPPL canonical indexing).
  function vectorShaped(node: any): boolean {
    if (!node) return false;
    if (node.type === 'ArrayLiteral') return true;
    if (node.type !== 'CallExpr' || !node.callee) return false;
    const name = node.callee.name;
    if (name === 'vector') return true;
    if (name === 'cat') return true;
    if (name === 'iid') return true;
    if (name === 'draw' && node.args && node.args[0]) {
      return vectorShaped(node.args[0]);
    }
    const VEC_DISTS = new Set([
      'MvNormal', 'Dirichlet', 'Multinomial',
      'Wishart', 'InverseWishart', 'LKJ', 'LKJCholesky',
    ]);
    return VEC_DISTS.has(name);
  }
  if (vectorShaped(resolved)) {
    return {
      makeCall: (i: number, _lhsName: string) => AST.CallExpr(
        AST.Identifier('get', synLoc),
        [
          AST.Identifier(synName, synLoc),
          AST.NumberLiteral(i + 1, String(i + 1), synLoc),
        ],
        synLoc,
      ),
    };
  }
  // Default: tuple_get (rand and other tuple-returning calls).
  return {
    makeCall: (i: number, _lhsName: string) => AST.CallExpr(
      AST.Identifier('tuple_get', synLoc),
      [
        AST.Identifier(synName, synLoc),
        AST.NumberLiteral(i, String(i), synLoc),
      ],
      synLoc,
    ),
  };
}

function planRename(ast: any, bindings: any, line: number, col: number) {
  function inLoc(loc: any) {
    return loc && loc.start.line <= line && line <= loc.end.line
        && (loc.start.line < line || col >= loc.start.col)
        && (line < loc.end.line || col <= loc.end.col);
  }

  for (const stmt of ast.body) {
    if (stmt.type !== 'AssignStatement') continue;

    // LHS names — direct binding references.
    for (const nameNode of stmt.names) {
      if (!inLoc(nameNode.loc)) continue;
      const name = nameNode.name;
      // Per spec §04 the parser renames each bare `_` LHS to a
      // distinct `__discard_N` synthetic — discards aren't
      // renamable (the user can't even type the name).
      if (name === '_' || /^__discard_/.test(name)) return null;
      if (!bindings.has(name)) return null;
      return planBindingRename(ast, bindings, name);
    }

    // RHS expression — could be an identifier reference or a placeholder.
    const target = findCursorTargetInExpr(stmt.value, inLoc);
    if (target) {
      if (target.kind === 'identifier') {
        if (!bindings.has(target.name)) return null;
        return planBindingRename(ast, bindings, target.name);
      }
      if (target.kind === 'placeholder' && target.scope) {
        return planPlaceholderRename(target.scope, target.name, target.loc);
      }
    }
  }
  return null;
}

function planBindingRename(ast: any, bindings: any, name: string) {
  const binding = bindings.get(name);
  if (!binding) return null;

  const locs = [binding.nameLoc];
  for (const stmt of ast.body) {
    if (stmt.type !== 'AssignStatement') continue;
    const refs = collectIdentRefs(stmt.value);
    for (const ref of refs) {
      if (ref.name === name) locs.push(ref.loc);
    }
  }
  return { kind: 'binding', oldName: name, targetLoc: binding.nameLoc, locs };
}

function planPlaceholderRename(scopeCallExpr: any, name: string, targetLoc: any) {
  const locs: any[] = [];
  function walk(node: any) {
    if (!node) return;
    if (node.type === 'Placeholder') {
      if (node.name === name) locs.push(node.loc);
      return;
    }
    if (node.type === 'CallExpr') {
      // Stop at NESTED functionof/kernelof — those are different placeholder scopes.
      if (node !== scopeCallExpr
          && node.callee && node.callee.type === 'Identifier'
          && (node.callee.name === 'functionof' || node.callee.name === 'kernelof')) {
        return;
      }
      walk(node.callee);
      for (const a of node.args) walk(a);
      return;
    }
    if (node.type === 'BinaryExpr') { walk(node.left); walk(node.right); return; }
    if (node.type === 'UnaryExpr') { walk(node.operand); return; }
    if (node.type === 'ArrayLiteral' || node.type === 'TupleLiteral') {
      for (const e of node.elements) walk(e);
      return;
    }
    if (node.type === 'IndexExpr') {
      walk(node.object);
      for (const i of node.indices) walk(i);
      return;
    }
    if (node.type === 'FieldAccess') { walk(node.object); return; }
    if (node.type === 'KeywordArg') { walk(node.value); return; }
  }
  for (const a of scopeCallExpr.args) walk(a);
  return { kind: 'placeholder', oldName: name, targetLoc, locs };
}

/**
 * Find a renameable AST node at the cursor position within an expression.
 * Tracks the nearest enclosing functionof/kernelof CallExpr as the
 * placeholder scope.
 */
function findCursorTargetInExpr(root: any, inLoc: any) {
  let result: any = null;
  function walk(node: any, scope: any) {
    if (!node || result) return;
    if (node.type === 'Identifier' && inLoc(node.loc)) {
      result = { kind: 'identifier', name: node.name, loc: node.loc };
      return;
    }
    if (node.type === 'Placeholder' && inLoc(node.loc)) {
      result = { kind: 'placeholder', name: node.name, loc: node.loc, scope };
      return;
    }
    if (node.type === 'CallExpr') {
      let inner = scope;
      if (node.callee && node.callee.type === 'Identifier'
          && (node.callee.name === 'functionof' || node.callee.name === 'kernelof')) {
        inner = node;
      }
      walk(node.callee, scope);
      for (const a of node.args) walk(a, inner);
      return;
    }
    if (node.type === 'BinaryExpr') { walk(node.left, scope); walk(node.right, scope); return; }
    if (node.type === 'UnaryExpr') { walk(node.operand, scope); return; }
    if (node.type === 'ArrayLiteral' || node.type === 'TupleLiteral') {
      for (const e of node.elements) walk(e, scope);
      return;
    }
    if (node.type === 'IndexExpr') {
      walk(node.object, scope);
      for (const i of node.indices) walk(i, scope);
      return;
    }
    if (node.type === 'FieldAccess') { walk(node.object, scope); return; }
    if (node.type === 'KeywordArg') { walk(node.value, scope); return; }
  }
  walk(root, null);
  return result;
}

/**
 * Find the chain of enclosing AST node ranges at a given cursor position,
 * ordered from innermost to outermost.
 *
 * Used by SelectionRangeProvider to power "Expand Selection" (Shift+Alt+→).
 *
 * @param {object} ast - parsed Program AST
 * @param {number} line - 0-based cursor line
 * @param {number} col - 0-based cursor column
 * @returns {Array<Loc>} innermost first
 */
function findEnclosingRanges(ast: any, line: number, col: number) {
  function inLoc(loc: any) {
    return loc && loc.start.line <= line && line <= loc.end.line
        && (loc.start.line < line || col >= loc.start.col)
        && (line < loc.end.line || col <= loc.end.col);
  }

  const ranges: any[] = []; // outermost first; we'll reverse at the end

  function walk(node: any) {
    if (!node) return;
    // Program has no .loc — descend into body without recording a range.
    if (node.type === 'Program') {
      for (const s of node.body) walk(s);
      return;
    }
    if (!node.loc || !inLoc(node.loc)) return;
    ranges.push(node.loc);
    // Recurse into children — the deepest matching node is appended last.
    switch (node.type) {
      case 'AssignStatement':
        for (const n of node.names) walk(n);
        walk(node.value);
        break;
      case 'CallExpr':
        walk(node.callee);
        for (const a of node.args) walk(a);
        break;
      case 'BinaryExpr':
        walk(node.left);
        walk(node.right);
        break;
      case 'UnaryExpr':
        walk(node.operand);
        break;
      case 'ArrayLiteral':
      case 'TupleLiteral':
        for (const e of node.elements) walk(e);
        break;
      case 'IndexExpr':
        walk(node.object);
        for (const i of node.indices) walk(i);
        break;
      case 'FieldAccess':
        walk(node.object);
        break;
      case 'KeywordArg':
        walk(node.value);
        break;
      // Leaf nodes have no children to walk.
    }
  }

  walk(ast);
  return ranges.reverse(); // innermost first
}

module.exports = {
  analyze, classifyStatement, collectDeps,
  extractBoundaries, extractJointFields, detectDisintegration,
  computePhases, computePhasesForScope, isMeasureExpr,
  collectIdentRefs, sliceSource,
  planRename, isValidBindingName, isValidPlaceholderText,
  findEnclosingRanges,
};
