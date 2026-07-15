'use strict';

// =====================================================================
// lowering-failure-diagnostic.test.ts — Buffy #73(5)
// =====================================================================
//
// A binding whose RHS fails to LOWER (a lowering-time error that is NOT a
// parse/analyze error — e.g. a FieldAccess in a reification boundary
// position) was caught in pir.lowerToModule, stashed as `lowerError` on a
// null-lit placeholder, and NEVER surfaced. The comment "the analyzer will
// already have flagged it" is false for a lowering-time failure, so the
// model silently produced a null binding. Now: the analyzer surfaces every
// stashed lowerError as an error diagnostic.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const eng = require('..');

// A FieldAccess in a kernelof boundary-kwarg position parses and analyzes
// clean but throws in lowering ("kernelof parameter must be bound to an
// identifier or placeholder, not FieldAccess"). Pre-fix: 0 diagnostics
// (silent). Post-fix: an error diagnostic.
const LOWER_FAIL_SRC = `rec = record(a = 1.0)
mu ~ Normal(0.0, 1.0)
k = kernelof(record(y = mu), theta = rec.a)
`;

test('#73(5) a lowering-time failure surfaces as an error diagnostic (not silently swallowed)', () => {
  const proc = eng.processSource(LOWER_FAIL_SRC);
  const diags = proc.diagnostics || proc.diags || [];
  const errs = diags.filter((d: any) => d.severity === 'error');
  assert.ok(errs.length >= 1,
    `a binding that fails to lower must produce an error diagnostic; got ${JSON.stringify(diags)}`);
  assert.ok(errs.some((d: any) => /lower|kernelof|FieldAccess/i.test(d.message)),
    `the diagnostic should describe the lowering failure; got ${errs.map((d: any) => d.message).join(' | ')}`);
});
