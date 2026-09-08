'use strict';
// Engine-limitation errors.
//
// The engine refuses two very different things, and a caller must be able to
// tell them apart WITHOUT reading English prose:
//
//   * a MODEL error — the model is invalid under `flatppl-design/docs/`, so no
//     engine will ever score it. These stay plain `Error`s, optionally with
//     their own structured code (`CLM_SINGULAR_JOINT`, …).
//   * an ENGINE LIMITATION — the model is well-formed and the spec gives it a
//     value, but THIS engine does not implement the route that reaches it yet.
//     A later release scores the same source unchanged.
//
// Conflating them makes a conformance harness freeze a wrong verdict: a
// limitation looks like a rejected fixture, and a genuinely invalid model looks
// like a missing feature. So every limitation carries ONE structured marker,
// `code === ENGINE_LIMITATION`, in the same `err.code` style the engine already
// uses for `CLM_SUBSET_VIOLATION` and `JOINT_FANOUT_INLINE_FN_COMPONENT`.
//
// Match on the CODE, not on the message. `flatppl-testsuite`'s `score_js.cjs`
// prints only `e.message`, so the message also states plainly that the
// construct is not implemented rather than invalid — but that text is for
// humans and may be reworded; the code is the contract.

const ENGINE_LIMITATION = 'ENGINE_LIMITATION';

/**
 * Build the engine-limitation error for an unimplemented construct.
 *
 * @param construct what the model asked for, in spec vocabulary — the reader
 *   must be able to find it in `flatppl-design/docs/` (for example
 *   `normalize over a parameter-dependent total mass`).
 * @param route which inference path hit the gap (`density`, `MCMC`, …). The
 *   same construct can be supported on one route and unimplemented on another,
 *   so naming the route tells the user whether a different query still works.
 * @param detail optional specifics — the offending names, the spec section.
 *
 * The error carries `code` for tools and `limitation: {construct, route}` so a
 * host can group gaps without re-parsing the sentence.
 */
function engineLimitation(construct: string, route: string, detail?: string): Error {
  const err: any = new Error(
    'engine limitation: ' + construct + ' is not implemented on the '
    + route + ' route' + (detail ? ' (' + detail + ')' : '')
    + '. The model is not invalid — this engine cannot score it yet.');
  err.code = ENGINE_LIMITATION;
  err.limitation = { construct, route };
  return err;
}

/** True for an error raised by `engineLimitation`, whatever its wording. */
function isEngineLimitation(err: any): boolean {
  return !!err && err.code === ENGINE_LIMITATION;
}

module.exports = { ENGINE_LIMITATION, engineLimitation, isEngineLimitation };
