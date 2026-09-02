'use strict';

// =====================================================================
// Every binding of every coverage-corpus model reaches a plot.
// =====================================================================
//
// The user report this closes: "all likelihoods should plot and they do not".
// The corpus dirs in flatppl-testsuite are full worked models, each written to
// score a language axis no other row reaches — so they are the sharpest
// available probe of the viewer's per-binding plot path, and five of them had
// bindings that stopped somewhere in the pipeline (fixed-phase evaluation,
// typeinfer, the classifier, the materialiser).
//
// This asserts the ENGINE half of the viewer's contract per binding: a
// materialisable binding materialises to a shape the renderer reads, and the
// bindings that legitimately have no measure (a module namespace, a free
// `elementof` boundary, a callable) are named rather than silently absent. The
// viewer's own routing half is `viewer/src/plot-plan*.test.ts`.
//
// The corpus is a SIBLING checkout, so the whole file skips when it is not
// present. The per-rule tests that must hold with no sibling repo live in
// `weighted-variate-arity.test.ts`, `fixed-eval-higher-order.test.ts` and
// `bayesupdate-prune-diagnostic.test.ts`.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const { processSource, orchestrator, materialiser } = require('../index.ts');
const { createWorkerHandler } = require('../worker.ts');

// Walk up for the sibling checkout rather than counting `..` segments: this
// repo is also checked out as a git worktree, which sits one level deeper.
function findCorpus(): string | null {
  let dir = __dirname;
  for (let i = 0; i < 8; i++) {
    const cand = path.join(dir, 'flatppl-testsuite', 'corpora', 'coverage');
    if (fs.existsSync(cand)) return cand;
    const up = path.dirname(dir);
    if (up === dir) break;
    dir = up;
  }
  return null;
}

const CORPUS = findCorpus() || '';
const HAVE_CORPUS = CORPUS !== '';

// Bindings expected to MATERIALISE, per dir. A name absent from this list is
// not asserted either way — the list grows as engine gaps close, and the
// entries below are the ones a user plots: every likelihood, every posterior,
// and the measures they are built from.
//
// The dirs with a remaining engine gap carry it in EXPECTED_GAPS instead, with
// the diagnostic the binding must still produce — recorded, never silent.
const EXPECTED_MATERIALISES: Record<string, string[]> = {
  allele_freq:         ['counts_data', 'p', 'prior', 'posterior'],
  ar1_drift:           ['x_data', 'sigma_step', 'prior', 'x', 'posterior'],
  dose_surface:        ['square', 'm1', 'm2', 'lp1', 'lp2', 'lp'],
  mv_mixture:          ['mus', 'cA', 'cB', 'covs', 'w', 'mix'],
  out_of_window:       ['obs_w', 'n', 'prior', 'posterior'],
  paired_assay:        ['mu_a', 'mu_b', 'run', 'prior', 'posterior'],
  sensor_calibration:  ['X', 'y_obs', 'beta', 'mu_pred', 'y', 'prior', 'posterior'],
  spectral_lines:      ['centers', 'y_obs', 'w', 'sigma', 'line', 'y', 'prior', 'posterior'],
  two_instruments:     ['obs_a', 'obs_b', 'prior'],
  censored_lifetimes:  ['t_obs', 'prior'],
  stdmod_interp_poly6: ['reference_point'],
  beam_bunch:          ['s', 'b', 'sig_shape', 'bkg_shape', 'intensity', 'prior'],
  kscan_walk:          ['dts', 'traj_obs', 'prior'],
  mv_mixture_sample:   ['mus', 'cA', 'cB', 'covs', 'w', 'mix', 'y', 'y1', 'y2'],
  b_mass_peak:         ['m_obs', 'N', 'f', 'prior'],
};

// Bindings that still stop at an engine gap, with the substring their error
// must contain. Each is recorded with its root cause in
// flatppl-dev/TODO-flatppl-js.md. A gap that CLOSES fails here loudly (the
// binding materialises and the assertion below says so), which is the point:
// the list cannot rot into a silencer.
const EXPECTED_GAPS: Record<string, Record<string, string>> = {
  allele_freq:        { y: '' },
  beam_bunch:         { events: 'PoissonProcess', posterior: 'PoissonProcess' },
  kscan_walk:         { posterior: "measure op 'kscan'" },
  paired_assay:       { tab: 'iid over a record measure' },
  censored_lifetimes: { posterior: 'truncate set could not be resolved' },
  mv_mixture_sample:  { draws: '' },
  b_mass_peak:        { posterior: '' },
};

function ctxFor(src: string) {
  const lifted = processSource(src);
  const built = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 0xC0FFEEEE });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations,
    bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    sampleCount: 64,
    rootSeed: 0xC0FFEEEE,
    getMeasure(n: string) {
      if (cache.has(n)) return cache.get(n);
      const p = materialiser.materialiseMeasure(n, ctx);
      cache.set(n, p);
      return p;
    },
    sendWorker(m: any) {
      const reply = worker.handle(m);
      if (reply && reply.type === 'error') return Promise.reject(new Error(reply.message));
      return Promise.resolve(reply);
    },
  };
  return { ctx, lifted, built };
}

// A shape the viewer's renderers read: samples, a shaped value, record fields,
// or tuple elements.
function hasRenderableShape(m: any): boolean {
  if (!m) return false;
  return m.samples != null || m.value != null || m.fields != null
    || (Array.isArray(m.elems) && m.elems.length > 0);
}

if (!HAVE_CORPUS) {
  test('coverage corpus sweep', { skip: 'flatppl-testsuite sibling checkout absent' }, () => {});
}

for (const dir of HAVE_CORPUS ? Object.keys(EXPECTED_MATERIALISES).sort() : []) {
  const modelPath = path.join(CORPUS, dir, 'model.flatppl');
  if (!fs.existsSync(modelPath)) continue;

  test(`coverage/${dir}: every expected binding materialises`, async () => {
    const src = fs.readFileSync(modelPath, 'utf8');
    const { ctx, built } = ctxFor(src);
    for (const name of EXPECTED_MATERIALISES[dir]) {
      assert.ok(built.bindings.has(name),
        `${dir}: no binding '${name}' — the corpus model changed, update this list`);
      let m: any;
      try {
        m = await ctx.getMeasure(name);
      } catch (e: any) {
        assert.fail(`${dir}.${name} did not materialise: ${e.message}`);
      }
      assert.ok(hasRenderableShape(m),
        `${dir}.${name} materialised with no renderable shape `
        + `(keys: ${Object.keys(m || {}).join(', ')})`);
    }
  });

  const gaps = EXPECTED_GAPS[dir];
  if (!gaps) continue;
  test(`coverage/${dir}: recorded engine gaps still report loudly`, async () => {
    const src = fs.readFileSync(modelPath, 'utf8');
    const { ctx } = ctxFor(src);
    for (const name of Object.keys(gaps)) {
      let err: any = null;
      try { await ctx.getMeasure(name); } catch (e: any) { err = e; }
      assert.ok(err, `${dir}.${name} now materialises — a gap closed. `
        + 'Move it to EXPECTED_MATERIALISES and update TODO-flatppl-js.md.');
      if (gaps[name]) {
        assert.ok(String(err.message).includes(gaps[name]),
          `${dir}.${name} failed for a DIFFERENT reason than recorded: ${err.message}`);
      }
    }
  });
}

// The two dirs whose numbers this lane changed, checked against their own
// independent oracle rather than against the other engine.
if (HAVE_CORPUS) {
  test('coverage/dose_surface: both weight-arity spellings score the closed form', async () => {
    const src = fs.readFileSync(path.join(CORPUS, 'dose_surface', 'model.flatppl'), 'utf8');
    const { ctx } = ctxFor(src);
    // The base `iid(Uniform(interval(0, 1)), 2)` has density 1 on the unit
    // square, so each spelling's log-density at (0.5, 0.8) is log(0.5 * 0.8^2).
    const want = Math.log(0.5 * 0.8 ** 2);
    for (const name of ['lp1', 'lp2']) {
      const m = await ctx.getMeasure(name);
      const got = m.value ? m.value.data[0] : m.samples[0];
      assert.ok(Math.abs(got - want) < 1e-12, `${name} = ${got}, closed form ${want}`);
    }
    const total = await ctx.getMeasure('lp');
    const gotTotal = total.value ? total.value.data[0] : total.samples[0];
    assert.ok(Math.abs(gotTotal - 2 * want) < 1e-12,
      `lp = ${gotTotal}, closed form ${2 * want}`);
  });

  test('coverage/two_instruments: the joint-likelihood posterior scores the closed form',
    async () => {
      const base = fs.readFileSync(
        path.join(CORPUS, 'two_instruments', 'model.flatppl'), 'utf8');
      const L2P = Math.log(2 * Math.PI);
      // Three closed-form Normal terms, written out: instrument A reads mu with
      // unit noise at 1.5, instrument B reads 2*mu with sigma 0.5 at 3.2, and
      // the prior is Normal(0, 2) at mu.
      const closed = (mu: number) =>
        (-0.5 * L2P - 0.5 * (1.5 - mu) ** 2)
        + (-0.5 * L2P - Math.log(0.5) - 0.5 * ((3.2 - 2 * mu) / 0.5) ** 2)
        + (-0.5 * L2P - Math.log(2) - 0.5 * (mu / 2) ** 2);
      for (const mu of [1.6, 0.0, 1.0, -0.8]) {
        const { ctx } = ctxFor(
          base + `\nlp_probe = logdensityof(posterior, record(mu = ${mu.toFixed(6)}))\n`);
        const m = await ctx.getMeasure('lp_probe');
        const got = m.value ? m.value.data[0] : m.samples[0];
        assert.ok(Math.abs(got - closed(mu)) < 1e-9,
          `mu=${mu}: got ${got}, closed form ${closed(mu)}`);
      }
    });

  test('coverage/out_of_window: the empty window folds to size 0 with no diagnostic',
    async () => {
      const src = fs.readFileSync(path.join(CORPUS, 'out_of_window', 'model.flatppl'), 'utf8');
      const lifted = processSource(src);
      const errs = (lifted.diagnostics || []).filter((d: any) => d.severity === 'error');
      assert.deepEqual(errs.map((d: any) => d.message), [],
        'no fixed-phase evaluation diagnostic (the filtered size folds)');
      // §06 iid + the zero-size-arrays ruling: every observation (8.2, 9.4,
      // 12.1) misses interval(-3, 3), so the derived size is exactly 0. The
      // TYPE stays %dynamic — filter's output length is not a static property
      // of the type — while the VALUE folds on demand, which is what the shape
      // position needed.
      const { ctx } = ctxFor(src);
      const n = await ctx.getMeasure('n');
      assert.equal(n.value ? n.value.data[0] : n.samples[0], 0,
        'the derived iid size is 0');
      const obsW = await ctx.getMeasure('obs_w');
      assert.equal(obsW.value.data.length, 0, 'the filtered data is empty');
    });

  test('every coverage model analyses with no error diagnostics', () => {
    const dirs = fs.readdirSync(CORPUS)
      .filter((f: string) => fs.statSync(path.join(CORPUS, f)).isDirectory());
    // b_mass_peak reaches §09 standard-module DISTRIBUTION members, which have
    // no surface-language measure lowering in this engine (only the FlatPDL
    // `builtin_logdensityof(CrystalBall, …)` the determiniser emits). Recorded
    // in TODO-flatppl-js.md; pinned WITH its message so it cannot acquire a new
    // error quietly. paired_assay's table-in-record refusal is recorded there
    // too.
    const KNOWN: Record<string, RegExp> = {
      b_mass_peak: /weighted: arg 2 expects measure, got deferred/,
      paired_assay: /a table may not appear inside a record/,
    };
    for (const dir of dirs.sort()) {
      const modelPath = path.join(CORPUS, dir, 'model.flatppl');
      if (!fs.existsSync(modelPath)) continue;
      const errs = (processSource(fs.readFileSync(modelPath, 'utf8')).diagnostics || [])
        .filter((d: any) => d.severity === 'error');
      if (KNOWN[dir]) {
        for (const e of errs) {
          assert.match(e.message, KNOWN[dir],
            `${dir} acquired a NEW error diagnostic: ${e.message}`);
        }
        continue;
      }
      assert.deepEqual(errs.map((d: any) => d.message), [], `${dir} has error diagnostics`);
    }
  });
}
