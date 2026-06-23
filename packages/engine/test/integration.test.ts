'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { processSource, computeSubDAG, moduleDeps, moduleResolve } = require('../index.ts');

// Integration tests run against bundled flatppl source files copied from
// the flatppl-examples and statsmodel-rosetta-stone sibling repos. Update
// the copies in `fixtures/` when the originals change.

const FIXTURES_DIR = path.join(__dirname, 'fixtures');

// Read a fixture plus any transitive `load_module` dependencies it pulls
// in (spec §04), mirroring a host resolver: walk `moduleDeps` +
// `resolveModulePath`, read each sibling from the fixtures dir. Returns
// the args for `processSource` — `{ source, opts }` with the assembled
// bundle (empty for a single-file fixture).
function loadFixture(name: string) {
  const source = fs.readFileSync(path.join(FIXTURES_DIR, name), 'utf8');
  const sources: Record<string, string> = {};
  (function walk(importer: string, text: string) {
    for (const rel of moduleDeps(text)) {
      const resolved = moduleResolve.resolveModulePath(importer, rel);
      if (sources[resolved]) continue;
      sources[resolved] = fs.readFileSync(path.join(FIXTURES_DIR, resolved), 'utf8');
      walk(resolved, sources[resolved]);
    }
  })(name, source);
  const hasDeps = Object.keys(sources).length > 0;
  return { source, opts: hasDeps ? { path: name, bundle: { sources } } : { path: name } };
}

const PARSE_FIXTURES = [
  'bayesian_inference_1.flatppl',
  'bayesian_inference_2.flatppl',
  // 3 (disintegrate) + 4 (restrict) are multi-file: they load_module a
  // shared `bayesian_inference_common.flatppl` (loadFixture assembles the
  // bundle).
  'bayesian_inference_3.flatppl',
  'bayesian_inference_4.flatppl',
  'flatppl-uncorrelated_background-ma-auxm.flatppl',
  'flatppl-uncorrelated_background-ma-priors.flatppl',
  'flatppl-uncorrelated_background-draws-auxm.flatppl',
  'flatppl-uncorrelated_background-draws-priors.flatppl',
  'minimal.flatppl',
  // Generative transport-kernel variants (the k_model / k_model_n
  // motivating model): `1` reifies via a `pars` RECORD boundary, `2`
  // via scalar `a`/`b`/`mu` boundaries. `1` also has deeper coverage
  // in kernel-sample-applied / mc-likelihood-e2e / generative-kernel-
  // broadcast; the parse-smoke here is the shared tripwire for the pair.
  'simple-transport1.flatppl',
  'simple-transport2.flatppl',
];

for (const name of PARSE_FIXTURES) {
  test(`integration: ${name} parses without errors`, () => {
    const f = loadFixture(name);
    const { diagnostics } = processSource(f.source, f.opts);
    const errors = diagnostics.filter((d: any) => d.severity === 'error');
    if (errors.length > 0) {
      for (const e of errors) {
        console.error(`  line ${e.loc.start.line + 1}: ${e.message}`);
      }
    }
    assert.equal(errors.length, 0, `${errors.length} errors in ${name}`);
  });
}

test('integration: bayesian_inference_1 forward_kernel DAG includes correct ancestors', () => {
  const src = fs.readFileSync(path.join(FIXTURES_DIR, 'bayesian_inference_1.flatppl'), 'utf8');
  const { bindings } = processSource(src);
  const dag = computeSubDAG(bindings, 'forward_kernel');
  const ids = new Set(dag.nodes.map((n: any) => n.id));
  // Must include kernel target (obs), boundaries (theta1, theta2),
  // and the deterministic chain (a, b, f_a, f_b)
  for (const id of ['forward_kernel', 'obs', 'a', 'b', 'f_a', 'f_b', 'theta1', 'theta2']) {
    assert.ok(ids.has(id), `missing ${id}`);
  }
  assert.equal(dag.nodes.find((n: any) => n.id === 'theta1').isBoundary, true);
  assert.equal(dag.nodes.find((n: any) => n.id === 'theta2').isBoundary, true);
});

// --- disintegrate_dual_kernel: structural disintegration via positional
//     jointchain → Delegate plan. This fixture (a functionof forward_kernel
//     PLUS a forward_kernel2/prior2 = disintegrate(...) in one module) is a
//     test-only superset — it has no public example; the renamed example
//     bayesian_inference_3 is the plain single-disintegrate form. -----------

test('integration: disintegrate_dual_kernel disintegration produces Delegate plans', () => {
  const src = fs.readFileSync(path.join(FIXTURES_DIR, 'disintegrate_dual_kernel.flatppl'), 'utf8');
  const { bindings, diagnostics } = processSource(src);
  assert.equal(diagnostics.filter((d: any) => d.severity === 'error').length, 0);

  // The disintegrate(["obs"], joint_model) statement should resolve via
  // the jointchain-positional rule into a Delegate identifying the kernel
  // side with `forward_kernel` and the prior side with `prior`.
  const fk2 = bindings.get('forward_kernel2');
  const pr2 = bindings.get('prior2');
  assert.equal(fk2.disintegratePlan.kind, 'delegate');
  assert.equal(pr2.disintegratePlan.kind, 'delegate');
  assert.equal(fk2.disintegratePlan.kernel.binding, 'forward_kernel');
  assert.equal(pr2.disintegratePlan.prior.binding,  'prior');
  // The delegate target's surface type wins so forward_kernel2 reads
  // the same keyword the user wrote on forward_kernel:
  //   forward_kernel = functionof(obs_dist, theta1=theta1, theta2=theta2)
  // — `functionof` because obs_dist is a measure (spec §sec:kernelof).
  assert.equal(fk2.type, 'functionof');
  assert.equal(pr2.type, 'lawof');
});

test('integration: disintegrate_dual_kernel forward_kernel2 sub-DAG mirrors forward_kernel', () => {
  // Delegate semantics: forward_kernel2 should render with the same
  // structural ancestors and boundaries as forward_kernel.
  const src = fs.readFileSync(path.join(FIXTURES_DIR, 'disintegrate_dual_kernel.flatppl'), 'utf8');
  const { bindings } = processSource(src);

  const fkDag  = computeSubDAG(bindings, 'forward_kernel');
  const fk2Dag = computeSubDAG(bindings, 'forward_kernel2');

  // Strip the root-id difference and the synthetic ":target" anon node
  // (different bubble owners → different synthetic ids), then compare
  // the set of "real" binding ancestors reached.
  const realIds = (dag: any, root: any) => new Set(
    dag.nodes
      .filter((n: any) => n.id !== root && !n.id.includes(':'))
      .map((n: any) => n.id)
  );
  assert.deepEqual(
    [...realIds(fk2Dag, 'forward_kernel2')].sort(),
    [...realIds(fkDag,  'forward_kernel') ].sort(),
    'forward_kernel2 should reach the same module-level ancestors as forward_kernel');

  // Boundaries are the same (theta1, theta2).
  const fk2Boundaries  = fk2Dag.nodes.filter((n: any) => n.isBoundary).map((n: any) => n.id).sort();
  const fkBoundaries   = fkDag.nodes.filter((n: any) => n.isBoundary).map((n: any) => n.id).sort();
  assert.deepEqual(fk2Boundaries, fkBoundaries);
});

test('integration: bayesian_inference_3 posterior derivation cascade', () => {
  // Regression guard: bayesupdate(L, prior2) with L = likelihoodof(K, obs)
  // and K = functionof(body, …) must classify into a 'bayesupdate'
  // derivation. The functionof IR shape is { op: 'functionof',
  // params, paramKwargs, body } — NOT { op, args } — so the
  // classifier must read the body via Kir.body, not Kir.args[0]. The
  // A1 IR-migration originally read .args[0], silently nulling the
  // posterior derivation and breaking the visualization.
  // bayesian_inference_3 is now multi-file (loads the shared common
  // module), so derivations must be built from the LINKED graph.
  const { buildDerivations } = require('../orchestrator.ts');
  const f = loadFixture('bayesian_inference_3.flatppl');
  const r = processSource(f.source, f.opts);
  assert.equal(r.diagnostics.filter((d: any) => d.severity === 'error').length, 0,
    'bayesian_inference_3 (multi-file) resolves without errors');
  const { derivations } = buildDerivations(r.linkedBindings);

  assert.ok(derivations.posterior, 'posterior should classify');
  assert.equal(derivations.posterior.kind, 'bayesupdate');
  assert.ok(derivations.posterior.from, 'posterior.from set');
  assert.ok(derivations.posterior.obsIR, 'posterior.obsIR set');
});

test('integration: disintegrate_dual_kernel lp_obs / d_obs classify end-to-end', () => {
  // logdensityof binding broadcasts log p(observed_data | theta1, theta2)
  // over prior atoms (the same primitive as bayesupdate's reweight,
  // exposed as a scalar binding). densityof rewrites to
  // exp(logdensityof(...)) at AST time. Both must reach a derivation
  // when their inputs are derivable.
  const { buildDerivations, resolveIRToValue } = require('../orchestrator.ts');
  const src = fs.readFileSync(path.join(FIXTURES_DIR, 'disintegrate_dual_kernel.flatppl'), 'utf8');
  const { bindings } = processSource(src);
  const ds = buildDerivations(bindings);
  const { derivations } = ds;

  assert.ok(derivations.lp_obs, 'lp_obs should classify');
  assert.equal(derivations.lp_obs.kind, 'logdensityof');
  // Measure should resolve to obs_dist (or one of its lifted aliases).
  assert.ok(derivations.lp_obs.measureName, 'lp_obs.measureName set');
  assert.ok(derivations.lp_obs.obsIR, 'lp_obs.obsIR set');
  // Resolution lives at the materialiser layer — same code path the
  // viewer takes. The literal observation matches the fixture.
  assert.deepEqual(
    resolveIRToValue(derivations.lp_obs.obsIR, ds.bindings, ds.fixedValues),
    { obs: [1.2, 3.4, 5.1, 2.8, 4.0, 3.7, 5.5, 2.1, 4.3, 3.9] });

  // d_obs lowers to exp(logdensityof(...)). The exp wrapping makes
  // the *outer* binding evaluate-kind (logdensityof was lifted to a
  // synthetic anon during liftValue, which itself is kind=logdensityof).
  assert.ok(derivations.d_obs, 'd_obs should classify');
  assert.equal(derivations.d_obs.kind, 'evaluate');
});

// =====================================================================
// minimal.flatppl — small fixture that exercises a pile of engine
// features in one ~12-line model:
//
//   - elementof leaves (a, mu)
//   - functionof with NO boundary kwargs → auto-promote of `a` via
//     canonicalizeImplicitBoundaries (f_sqrt = functionof(b))
//   - function call site with implicit boundary substitution
//     (sigma = f_sqrt(sigma2))
//   - tilde-binding lowering (sigma2 ~ Exp; x ~ iid(...))
//   - kernelof with explicit boundary (kernel = kernelof(x, mu = mu))
//   - record auto-splat at the kernel call site (dist = kernel(kernel_input))
// =====================================================================

function loadMinimal(file: any) {
  const src = fs.readFileSync(path.join(FIXTURES_DIR, file), 'utf8');
  return processSource(src, { path: file });
}

function userBindings(parsed: any) {
  // Skip anonymous lift-introduced names; we care about user bindings.
  const out: any = {};
  for (const [n, b] of parsed.bindings) {
    if (n.startsWith('__anon')) continue;
    out[n] = { type: b.type, phase: b.phase,
               inferredType: b.inferredType && b.inferredType.kind };
  }
  return out;
}

test('integration: minimal.flatppl parses without errors', () => {
  const { diagnostics } = loadMinimal('minimal.flatppl');
  const errors = diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errors.length, 0,
    `minimal.flatppl: ${errors.length} errors (${errors.map((e: any) => e.message).join('; ')})`);
});

test('integration: minimal — f_sqrt auto-promotes `a` as its single input', () => {
  // `functionof(b)` with no boundary kwargs invokes
  // canonicalizeImplicitBoundaries to add `a = a` from b's transitive
  // ancestor walk. signatureOf then reads ir.params verbatim.
  const { signatureOf, liftInlineSubexpressions } = require('../orchestrator.ts');
  const { bindings } = loadMinimal('minimal.flatppl');
  const lifted = liftInlineSubexpressions(bindings);
  const sig = signatureOf('f_sqrt', lifted);
  assert.ok(sig);
  assert.equal(sig.kind, 'function');
  assert.equal(sig.inputs.length, 1);
  assert.equal(sig.inputs[0].paramName, 'a');
});

test('integration: minimal — sigma is computed via implicit boundary substitution', () => {
  // `sigma = f_sqrt(sigma2)` exercises inlineOnce's positional-arg
  // path against the implicit boundary added by canonicalize. After
  // lift, sigma should compute pow(sigma2, 0.5) — NOT contain an
  // unbound ref to the elementof leaf `a`.
  const { buildDerivations } = require('../orchestrator.ts');
  const { bindings } = loadMinimal('minimal.flatppl');
  const ds = buildDerivations(bindings);
  // Walk sigma's transitive closure for any lingering 'a' refs.
  function walk(name: any, seen: any) {
    if (seen.has(name)) return;
    seen.add(name);
    const b = ds.bindings.get(name);
    if (!b || !b.ir) return;
    const refs = new Set();
    (function collect(ir) {
      if (!ir || typeof ir !== 'object') return;
      if (ir.kind === 'ref' && ir.ns === 'self') refs.add(ir.name);
      if (ir.args) for (const a of ir.args) collect(a);
      if (ir.kwargs) for (const k in ir.kwargs) collect(ir.kwargs[k]);
    })(b.ir);
    for (const r of refs) walk(r, seen);
  }
  const seen = new Set();
  walk('sigma', seen);
  assert.ok(!seen.has('a'),
    "sigma's closure must not reference elementof leaf 'a' — implicit substitution regressed");
  // sigma2 should be reached (the substituted call arg).
  assert.ok(seen.has('sigma2'),
    "sigma's closure should reach sigma2 via the call substitution");
});

test('integration: minimal — kernel applied to kernel_input record auto-splats', () => {
  // `dist = kernel(kernel_input)` with `kernel_input = record(mu = 1.5)`
  // and `kernel = kernelof(x, mu = mu)` exercises the splat path in
  // inlineOnce: single positional record arg with field names covering
  // the kernel's surface kwargs.
  const { buildDerivations } = require('../orchestrator.ts');
  const { bindings } = loadMinimal('minimal.flatppl');
  const ds = buildDerivations(bindings);
  // dist should classify successfully (no derivation gap from a
  // failed splat).
  assert.ok(ds.bindings.has('dist'));
  // After splat + inline, dist's transitive closure should reference
  // the literal 1.5 somewhere (the substituted mu value).
  let sawLit = false;
  const seen = new Set();
  const stack = ['dist'];
  while (stack.length > 0) {
    const n = stack.pop();
    if (seen.has(n)) continue;
    seen.add(n);
    const b = ds.bindings.get(n);
    if (!b || !b.ir) continue;
    (function walk(ir) {
      if (!ir || typeof ir !== 'object') return;
      if (ir.kind === 'lit' && ir.value === 1.5) { sawLit = true; return; }
      if (ir.kind === 'ref' && ir.ns === 'self') { stack.push(ir.name); return; }
      if (ir.args) for (const a of ir.args) walk(a);
      if (ir.kwargs) for (const k in ir.kwargs) walk(ir.kwargs[k]);
      // record / joint IRs carry their kids under `fields: [{name, value}]`;
      // 1.5 lives inside kernel_input's record() field, so we walk this too.
      if (Array.isArray(ir.fields)) for (const f of ir.fields) walk(f && f.value);
    })(b.ir);
  }
  assert.ok(sawLit,
    "dist's closure should contain the literal 1.5 after record-splat substitution");
});

test('integration: minimal — phases line up with what the model semantics imply', () => {
  // Lock the phase classification in as a regression guard: this
  // exact set is what every downstream pass (canonicalize,
  // signatureOf, the viewer's implicit-kernel/-function dispatch)
  // relies on for this fixture. If a future analyzer change shifts
  // any of these, the rest of the engine cascade breaks silently.
  const { bindings } = loadMinimal('minimal.flatppl');
  const expected = {
    a:            'parameterized',
    b:            'parameterized',
    f_sqrt:       'fixed',
    mu:           'parameterized',
    sigma2:       'stochastic',
    sigma:        'stochastic',
    x:            'stochastic',
    kernel:       'fixed',
    kernel_input: 'fixed',
    dist:         'fixed',
  };
  const actual: any = {};
  for (const k in expected) actual[k] = bindings.get(k) && bindings.get(k).phase;
  assert.deepEqual(actual, expected);
});

test('integration: indicesof0-driven polyeval (scalar form)', () => {
  // End-to-end exercise of the motivating use case for the new
  // `indicesof0` builtin (spec §07): polynomial evaluation by
  // broadcasting `x .^ k` over `k = [0, 1, ..., N-1]`. Covers:
  //   - parser + analyzer + typeinfer accept `indicesof0(coeffs)`
  //     in a deterministic-broadcast context;
  //   - `x .^ indicesof0(coeffs)` evaluates as `[x^0, x^1, x^2]`
  //     (scalar-vector dotted exponent — broadcasting an
  //     element-wise pow);
  //   - elementwise mul + sum reduces to the polynomial value.
  // Reference: polyeval([2.3, 1.5, 0.7], x) = 2.3 + 1.5 x + 0.7 x²
  //   at x = 3.3 → 2.3 + 4.95 + 7.623 = 14.873.
  //
  // The broadcast form `polyeval.([C], X)` (Ref(C)-style hold-
  // constant idiom) is pinned in test/nested-broadcast.test.ts —
  // see the "Ref-wrap motivating case" test there. This test keeps
  // the simpler scalar form as the primary indicesof0 demonstration.
  const src = `
polyeval = (coeffs, x) -> sum(coeffs .* x .^ indicesof0(coeffs))
C = [2.3, 1.5, 0.7]
y = polyeval(C, 3.3)
`;
  const r = processSource(src);
  const errs = r.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0,
    'polyeval source should parse + analyze without errors; got ['
    + errs.map((e: any) => e.message).join(', ') + ']');

  const { buildDerivations } = require('../orchestrator.ts');
  const ds = buildDerivations(r.bindings);
  const yVal = ds.fixedValues && ds.fixedValues.get('y');
  assert.ok(yVal != null,
    'y should land in fixedValues (deterministic over fixed-phase inputs)');
  const expectedY = 2.3 + 1.5 * 3.3 + 0.7 * 3.3 * 3.3;   // = 14.873
  assert.ok(Math.abs(yVal - expectedY) < 1e-9,
    'polyeval scalar result should match analytical (got ' + yVal
    + ', expected ' + expectedY + ')');
});

test('integration: polyeval-iid-broadcast fixture — Ref-wrap broadcast over iid', () => {
  // The canonical nested-broadcast test case: `polyeval.([C], X)`
  // with X ~ iid(Normal(0,1), 10). Verifies the full pipeline:
  //   - lift hoists the inline `[C]` ArrayLiteral to an anon binding;
  //   - the anon classifies as kind:'tuple' (a tuple of refs);
  //   - cascade-prune sweep skips the callable-head ref to polyeval;
  //   - Y classifies as 'evaluate' (not 'broadcast');
  //   - materialiser's matEvaluate inlines polyeval's functionof body
  //     before worker hand-off so the broadcast dispatcher can
  //     resolve the head without `env.__resolveFnBody`.
  const src = fs.readFileSync(
    path.join(FIXTURES_DIR, 'polyeval-iid-broadcast.flatppl'), 'utf8');
  const r = processSource(src);
  const errs = r.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0,
    'polyeval-iid-broadcast fixture should parse + analyze without errors; got ['
    + errs.map((e: any) => e.message).join(', ') + ']');

  const { buildDerivations } = require('../orchestrator.ts');
  const ds = buildDerivations(r.bindings);

  // Y should have a derivation. Phase 1 was: classifier produced
  // null (vector(C) inline blocked isEvaluable). Phase 2 was:
  // refs-valid sweep pruned because polyeval (a functionof) wasn't
  // in derivations/fixedValues. Both fixes verified. Phase 3
  // (fusion (a) Step 2, 2026-05-29): the broadcast(polyeval, [C], X)
  // form rewrites to an aggregate IR before classification — the
  // assertions below pin the FUSED structure.
  assert.ok(ds.derivations.Y,
    'Y should classify (post lift + cascade-prune exemption)');
  assert.equal(ds.derivations.Y.kind, 'evaluate');

  // Post-fusion Y.ir: aggregate(sum, [.atom], <body with get(C, .j)
  // and get(X, .atom) and get(literal_vector, .j)>). The fusion (a)
  // Step 2 rewrite collapses the nested broadcasts and lifts the
  // shape-folded indicesof0(C) to a literal vector.
  const yBinding = ds.bindings.get('Y');
  assert.equal(yBinding.ir.op, 'aggregate',
    'polyeval-shape broadcast now fuses to an aggregate IR');
  assert.equal(yBinding.ir.args[0].name, 'sum',
    'reducer is sum');
  assert.equal(yBinding.ir.args[1].op, 'vector',
    'output_axes is a vector of axis refs');
  assert.equal(yBinding.ir.args[1].args[0].kind, 'axis',
    'output axis is .atom (the broadcast iteration axis)');

  // Y's inferred type — array(rank=1, shape=[10], real). Inferred
  // type comes from typeinfer (runs BEFORE the dissolver), so the
  // type carries the pre-fusion shape regardless of what the IR
  // looks like post-dissolution.
  assert.equal(yBinding.inferredType.kind, 'array');
  assert.equal(yBinding.inferredType.rank, 1);
  assert.deepEqual(yBinding.inferredType.shape, [10]);
});

test('integration: disintegrate_dual_kernel posterior view uses the literal source structure', () => {
  // When viewing posterior (transitively reaches prior2 / forward_kernel2),
  // the trace should follow the user's source — not the rewriter's
  // delegate view. So joint_model must be reachable, and prior2 /
  // forward_kernel2 must NOT receive their own reification bubbles.
  const src = fs.readFileSync(path.join(FIXTURES_DIR, 'disintegrate_dual_kernel.flatppl'), 'utf8');
  const { bindings } = processSource(src);
  const dag = computeSubDAG(bindings, 'posterior');
  const ids = new Set(dag.nodes.map((n: any) => n.id));
  for (const id of ['posterior', 'L', 'forward_kernel2', 'prior2', 'joint_model', 'prior', 'forward_kernel']) {
    assert.ok(ids.has(id), `missing ${id} from posterior trace`);
  }
  // Disintegration results don't get bubbles in this view.
  const reifNames = new Set(dag.reifications.map((r: any) => r.name));
  assert.ok(!reifNames.has('forward_kernel2'), 'forward_kernel2 should not get a bubble here');
  assert.ok(!reifNames.has('prior2'),          'prior2 should not get a bubble here');
  // The user-written prior and forward_kernel still get bubbles.
  assert.ok(reifNames.has('prior'));
  assert.ok(reifNames.has('forward_kernel'));
});
