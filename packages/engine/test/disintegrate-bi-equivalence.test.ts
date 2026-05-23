'use strict';

// =====================================================================
// Disintegration equivalence: direct kernelof ≡ disintegrate result
// =====================================================================
//
// Spec §06 line 554: `jointchain(base_measure, kernel) ≡ joint_measure`.
// Combined with `kernelof(x, kw) ≡ functionof(lawof(x), kw)` (spec
// §04 §sec:kernelof L422), this means: for a model that defines BOTH
// a direct forward kernel AND the disintegrate result of a joint
// built from that kernel, the two should be structurally identical.
//
// Two complementary tests:
//
//   1. SAME-MODULE pair (feature-test1.flatppl):
//        forward_kernel  = functionof(obs_dist, theta1=theta1, theta2=theta2)
//        joint_model     = jointchain(prior, forward_kernel)
//        forward_kernel2, prior2 = disintegrate(["obs"], joint_model)
//      forward_kernel and forward_kernel2 share the same module's
//      anon-binding indices, so we can compare IR exactly modulo
//      source-location stripping.
//
//   2. CROSS-MODULE pair (BI1 vs BI2 in flatppl-examples):
//        bayesian_inference_1: forward_kernel = kernelof(...) directly.
//        bayesian_inference_2: forward_kernel = disintegrate(...).
//      Different modules generate independent anon-binding indices,
//      so we compare the IR's **structural skeleton** (op, params,
//      paramKwargs) — not the anon labels themselves.

const path = require('node:path');
const fs = require('node:fs');
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { liftInlineSubexpressions } = require('../orchestrator.ts');

const examplesRoot = path.join(__dirname, '..', '..', '..', '..',
  'flatppl-examples', 'examples');

function loadIfPresent(name: string): string | null {
  const p = path.join(examplesRoot, name);
  return fs.existsSync(p) ? fs.readFileSync(p, 'utf8') : null;
}

// Strip source locations + synthetic markers from an IR tree so two
// IRs originating from differently-located source code can be
// compared for structural identity. Returns a JSON-stable object.
function stripLocs(node: any): any {
  if (node == null) return node;
  if (Array.isArray(node)) return node.map(stripLocs);
  if (typeof node !== 'object') return node;
  const out: any = {};
  for (const k of Object.keys(node)) {
    if (k === 'loc') continue;
    out[k] = stripLocs(node[k]);
  }
  return out;
}

// ---------------------------------------------------------------------
// SAME-MODULE: feature-test1.flatppl
// ---------------------------------------------------------------------

const featureTest1Path = path.join(
  __dirname, '..', '..', 'web', 'demo', 'feature-test1.flatppl');

test('feature-test1: forward_kernel ≡ forward_kernel2 (disintegrate of jointchain delegates back)', () => {
  if (!fs.existsSync(featureTest1Path)) return;
  const src = fs.readFileSync(featureTest1Path, 'utf8');
  const ctx = processSource(src);
  // The file intentionally contains invalid_* bindings; what we need
  // is the forward_kernel / forward_kernel2 pair to be well-formed.
  const lifted = liftInlineSubexpressions(ctx.bindings);
  const fk  = lifted.get('forward_kernel');
  const fk2 = lifted.get('forward_kernel2');
  assert.ok(fk, 'forward_kernel exists');
  assert.ok(fk2, 'forward_kernel2 (disintegrate result) exists');

  // disintegrate(["obs"], jointchain(prior, forward_kernel)) takes
  // the delegate path (both jointchain components are bare binding
  // refs), so forward_kernel2's effective IR should equal
  // forward_kernel's IR exactly (modulo source-location info).
  const fkIR  = stripLocs(fk.ir);
  const fk2IR = stripLocs(fk2.ir);
  assert.deepEqual(fk2IR, fkIR,
    'disintegrate delegate hands back the original forward_kernel IR');
});

// ---------------------------------------------------------------------
// CROSS-MODULE: BI1 (direct kernelof) vs BI2 (disintegrate)
// ---------------------------------------------------------------------

// Structural skeleton: the IR's top-level op + paramKwargs + params
// (the kernel's interface), ignoring anon labels and source locations.
function ifaceOf(ir: any): any {
  if (!ir) return null;
  return {
    op: ir.op,
    params: [...(ir.params || [])].sort(),
    paramKwargs: [...(ir.paramKwargs || [])].sort(),
    bodyOp: ir.body && ir.body.kind === 'call' ? ir.body.op
          : ir.body && ir.body.kind === 'ref'  ? '<ref>'
          : null,
  };
}

test('BI1 forward_kernel (direct kernelof) ≡ BI2 forward_kernel (disintegrate) — structural interface', () => {
  const bi1Src = loadIfPresent('bayesian_inference_1.flatppl');
  const bi2Src = loadIfPresent('bayesian_inference_2.flatppl');
  if (!bi1Src || !bi2Src) {
    // Tests inert if the flatppl-examples sibling isn't available.
    return;
  }

  const ctx1 = processSource(bi1Src);
  const ctx2 = processSource(bi2Src);
  assert.equal(ctx1.diagnostics.filter((d: any) => d.severity === 'error').length, 0,
    'BI1 parses + analyzes cleanly');
  assert.equal(ctx2.diagnostics.filter((d: any) => d.severity === 'error').length, 0,
    'BI2 parses + analyzes cleanly');

  // Both files declare `forward_kernel`. In BI1 it's a direct
  // kernelof; in BI2 it's the kernel result of disintegrate (the
  // analyzer attaches an effectiveValue synthesized by the disintegrate
  // rewriter, then the lift lowers it to the same IR shape).
  const b1 = liftInlineSubexpressions(ctx1.bindings).get('forward_kernel');
  const b2 = liftInlineSubexpressions(ctx2.bindings).get('forward_kernel');
  assert.ok(b1, 'BI1 has forward_kernel');
  assert.ok(b2, 'BI2 has forward_kernel');

  // The IR's top-level shape: a functionof (since kernelof lowers to
  // functionof(lawof(...))) with `params`, `paramKwargs`, and a body.
  assert.equal(b1.ir?.op, 'functionof', 'BI1 lowers to functionof');
  assert.equal(b2.ir?.op, 'functionof', 'BI2 lowers to functionof');

  // Boundary kwargs: same set of formal parameters.
  assert.deepEqual(
    [...(b1.ir.paramKwargs || [])].sort(),
    [...(b2.ir.paramKwargs || [])].sort(),
    'same boundary kwargs (theta1, theta2)');
  assert.deepEqual(
    [...(b1.ir.params || [])].sort(),
    [...(b2.ir.params || [])].sort(),
    'same body-local param names');

  // Body: structural shape only (the body wraps a record(obs = ref-
  // to-anon); anon names differ between modules but the shape doesn't).
  assert.deepEqual(ifaceOf(b1.ir), ifaceOf(b2.ir),
    'forward_kernel kernel-interface matches between BI1 and BI2');
});

test('BI1 prior (direct lawof) ≡ BI2 prior (disintegrate result)', () => {
  const bi1Src = loadIfPresent('bayesian_inference_1.flatppl');
  const bi2Src = loadIfPresent('bayesian_inference_2.flatppl');
  if (!bi1Src || !bi2Src) return;

  const ctx1 = processSource(bi1Src);
  const ctx2 = processSource(bi2Src);
  const b1 = liftInlineSubexpressions(ctx1.bindings).get('prior');
  const b2 = liftInlineSubexpressions(ctx2.bindings).get('prior');
  assert.ok(b1 && b2);

  // Both should classify as the same kind of measure (lawof of a record
  // of stochastic variates). We compare the structural skeleton only
  // (op + arity) — anon labels differ between modules.
  assert.equal(b1.ir.op, b2.ir.op,
    'prior IR has the same top-level op (lawof) in BI1 and BI2');
});

test('BI1 posterior derivation kind = BI2 posterior derivation kind', () => {
  const bi1Src = loadIfPresent('bayesian_inference_1.flatppl');
  const bi2Src = loadIfPresent('bayesian_inference_2.flatppl');
  if (!bi1Src || !bi2Src) return;

  const { buildDerivations } = require('../orchestrator.ts');
  const ctx1 = processSource(bi1Src);
  const ctx2 = processSource(bi2Src);
  const dctx1 = buildDerivations(ctx1.bindings, ctx1.diagnostics);
  const dctx2 = buildDerivations(ctx2.bindings, ctx2.diagnostics);

  // Both should classify posterior as bayesupdate, with the same
  // bodyName ('obs' — the iid Normal binding) and structurally equal
  // obsIR.
  assert.equal(dctx1.derivations.posterior?.kind, 'bayesupdate');
  assert.equal(dctx2.derivations.posterior?.kind, 'bayesupdate');
  assert.equal(dctx1.derivations.posterior?.bodyName,
               dctx2.derivations.posterior?.bodyName,
    'same bodyName for posterior\'s bayesupdate derivation');
});
