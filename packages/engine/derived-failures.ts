'use strict';

// Derived-binding failures. A scorer or prior transform evaluates each derived
// binding into an env and leaves one that throws out of it: it may be a
// function or a measure, which is not a value. A reader that then reads it as
// a value hits an unbound ref, and must report the binding's own cause
// instead of a missing name or -Infinity.

const DERIVED_FAILURES = Symbol('derivedFailures');

// The message is built on read: an env is rebuilt at every MCMC step, and most
// recorded failures (function and measure bindings) are never read.
function recordDerivedFailure(env: any, name: string, err: any): void {
  let failures: Map<string, { name: string; err: any }> = env[DERIVED_FAILURES];
  if (!failures) {
    failures = new Map();
    Object.defineProperty(env, DERIVED_FAILURES, { value: failures, enumerable: false });
  }
  // A failure that is an unbound read of an earlier failure keeps the root.
  const root = err?.flatpplRefName != null ? failures.get(err.flatpplRefName) : undefined;
  failures.set(name, root ?? { name, err });
}

// The failure behind an unbound read of a derived binding, or `null`.
function derivedFailureOf(err: any, env: any): string | null {
  if (!err || err.flatpplRefName == null || !env) return null;
  const f = env[DERIVED_FAILURES]?.get(err.flatpplRefName);
  return f ? `binding '${f.name}' failed to evaluate: ${f.err?.message ?? String(f.err)}` : null;
}

// Why a posterior field read from an env has no value.
function derivedFieldFailure(env: any, name: string): string {
  const f = env?.[DERIVED_FAILURES]?.get(name);
  return f ? `binding '${f.name}' failed to evaluate: ${f.err?.message ?? String(f.err)}`
    : `binding '${name}' has no value at the posterior draws`;
}

module.exports = { recordDerivedFailure, derivedFailureOf, derivedFieldFailure };
