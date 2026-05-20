'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { processSource, computeSubDAG } = require('../index.ts');

// Integration tests run against bundled flatppl source files copied from
// the flatppl-examples and statsmodel-rosetta-stone sibling repos. Update
// the copies in `fixtures/` when the originals change.

const FIXTURES_DIR = path.join(__dirname, 'fixtures');

const PARSE_FIXTURES = [
  'bayesian_inference_1.flatppl',
  'bayesian_inference_2.flatppl',
  'bayesian_inference_3.flatppl',
  'flatppl-uncorrelated_background-ma-auxm.flatppl',
  'flatppl-uncorrelated_background-ma-priors.flatppl',
  'flatppl-uncorrelated_background-draws-auxm.flatppl',
  'flatppl-uncorrelated_background-draws-priors.flatppl',
  'minimal.flatppl',
];

for (const name of PARSE_FIXTURES) {
  test(`integration: ${name} parses without errors`, () => {
    const src = fs.readFileSync(path.join(FIXTURES_DIR, name), 'utf8');
    const { diagnostics } = processSource(src, { path: name });
    const errors = diagnostics.filter(d => d.severity === 'error');
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
  const ids = new Set(dag.nodes.map(n => n.id));
  // Must include kernel target (obs), boundaries (theta1, theta2),
  // and the deterministic chain (a, b, f_a, f_b)
  for (const id of ['forward_kernel', 'obs', 'a', 'b', 'f_a', 'f_b', 'theta1', 'theta2']) {
    assert.ok(ids.has(id), `missing ${id}`);
  }
  assert.equal(dag.nodes.find(n => n.id === 'theta1').isBoundary, true);
  assert.equal(dag.nodes.find(n => n.id === 'theta2').isBoundary, true);
});

// --- bayesian_inference_3: structural disintegration via positional
//     jointchain → Delegate plan ----------------------------------------

test('integration: bayesian_inference_3 disintegration produces Delegate plans', () => {
  const src = fs.readFileSync(path.join(FIXTURES_DIR, 'bayesian_inference_3.flatppl'), 'utf8');
  const { bindings, diagnostics } = processSource(src);
  assert.equal(diagnostics.filter(d => d.severity === 'error').length, 0);

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

test('integration: bayesian_inference_3 forward_kernel2 sub-DAG mirrors forward_kernel', () => {
  // Delegate semantics: forward_kernel2 should render with the same
  // structural ancestors and boundaries as forward_kernel.
  const src = fs.readFileSync(path.join(FIXTURES_DIR, 'bayesian_inference_3.flatppl'), 'utf8');
  const { bindings } = processSource(src);

  const fkDag  = computeSubDAG(bindings, 'forward_kernel');
  const fk2Dag = computeSubDAG(bindings, 'forward_kernel2');

  // Strip the root-id difference and the synthetic ":target" anon node
  // (different bubble owners → different synthetic ids), then compare
  // the set of "real" binding ancestors reached.
  const realIds = (dag, root) => new Set(
    dag.nodes
      .filter(n => n.id !== root && !n.id.includes(':'))
      .map(n => n.id)
  );
  assert.deepEqual(
    [...realIds(fk2Dag, 'forward_kernel2')].sort(),
    [...realIds(fkDag,  'forward_kernel') ].sort(),
    'forward_kernel2 should reach the same module-level ancestors as forward_kernel');

  // Boundaries are the same (theta1, theta2).
  const fk2Boundaries  = fk2Dag.nodes.filter(n => n.isBoundary).map(n => n.id).sort();
  const fkBoundaries   = fkDag.nodes.filter(n => n.isBoundary).map(n => n.id).sort();
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
  const { buildDerivations } = require('../orchestrator.ts');
  const src = fs.readFileSync(path.join(FIXTURES_DIR, 'bayesian_inference_3.flatppl'), 'utf8');
  const { bindings } = processSource(src);
  const { derivations } = buildDerivations(bindings);

  assert.ok(derivations.posterior, 'posterior should classify');
  assert.equal(derivations.posterior.kind, 'bayesupdate');
  assert.ok(derivations.posterior.from, 'posterior.from set');
  assert.ok(derivations.posterior.obsIR, 'posterior.obsIR set');
});

test('integration: bayesian_inference_3 lp_obs / d_obs classify end-to-end', () => {
  // logdensityof binding broadcasts log p(observed_data | theta1, theta2)
  // over prior atoms (the same primitive as bayesupdate's reweight,
  // exposed as a scalar binding). densityof rewrites to
  // exp(logdensityof(...)) at AST time. Both must reach a derivation
  // when their inputs are derivable.
  const { buildDerivations, resolveIRToValue } = require('../orchestrator.ts');
  const src = fs.readFileSync(path.join(FIXTURES_DIR, 'bayesian_inference_3.flatppl'), 'utf8');
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

function loadMinimal(file) {
  const src = fs.readFileSync(path.join(FIXTURES_DIR, file), 'utf8');
  return processSource(src, { path: file });
}

function userBindings(parsed) {
  // Skip anonymous lift-introduced names; we care about user bindings.
  const out = {};
  for (const [n, b] of parsed.bindings) {
    if (n.startsWith('__anon')) continue;
    out[n] = { type: b.type, phase: b.phase,
               inferredType: b.inferredType && b.inferredType.kind };
  }
  return out;
}

test('integration: minimal.flatppl parses without errors', () => {
  const { diagnostics } = loadMinimal('minimal.flatppl');
  const errors = diagnostics.filter((d) => d.severity === 'error');
  assert.equal(errors.length, 0,
    `minimal.flatppl: ${errors.length} errors (${errors.map((e) => e.message).join('; ')})`);
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
  function walk(name, seen) {
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
  const actual = {};
  for (const k in expected) actual[k] = bindings.get(k) && bindings.get(k).phase;
  assert.deepEqual(actual, expected);
});

test('integration: bayesian_inference_3 posterior view uses the literal source structure', () => {
  // When viewing posterior (transitively reaches prior2 / forward_kernel2),
  // the trace should follow the user's source — not the rewriter's
  // delegate view. So joint_model must be reachable, and prior2 /
  // forward_kernel2 must NOT receive their own reification bubbles.
  const src = fs.readFileSync(path.join(FIXTURES_DIR, 'bayesian_inference_3.flatppl'), 'utf8');
  const { bindings } = processSource(src);
  const dag = computeSubDAG(bindings, 'posterior');
  const ids = new Set(dag.nodes.map(n => n.id));
  for (const id of ['posterior', 'L', 'forward_kernel2', 'prior2', 'joint_model', 'prior', 'forward_kernel']) {
    assert.ok(ids.has(id), `missing ${id} from posterior trace`);
  }
  // Disintegration results don't get bubbles in this view.
  const reifNames = new Set(dag.reifications.map(r => r.name));
  assert.ok(!reifNames.has('forward_kernel2'), 'forward_kernel2 should not get a bubble here');
  assert.ok(!reifNames.has('prior2'),          'prior2 should not get a bubble here');
  // The user-written prior and forward_kernel still get bubbles.
  assert.ok(reifNames.has('prior'));
  assert.ok(reifNames.has('forward_kernel'));
});
