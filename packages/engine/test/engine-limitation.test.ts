'use strict';

// The limitation marker is a CONTRACT with hosts and with
// `flatppl-testsuite`: a conformance harness must be able to tell "this engine
// does not implement the route yet" from "this model is invalid" without
// reading English. These tests pin the code and the shape, so a reworded
// message cannot break a downstream matcher silently.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const {
  ENGINE_LIMITATION, engineLimitation, isEngineLimitation,
} = require('../limitations.ts');
const engine = require('../index.ts');

test('a limitation carries the structured code and its construct/route', () => {
  const err: any = engineLimitation('normalize over a moving base', 'MCMC', 'unresolved: theta');
  assert.equal(err.code, ENGINE_LIMITATION);
  assert.ok(isEngineLimitation(err));
  assert.deepEqual(err.limitation,
    { construct: 'normalize over a moving base', route: 'MCMC' });
  // score_js.cjs prints only the message, so it must say which construct,
  // which route, and that the source is not at fault.
  assert.match(err.message, /normalize over a moving base/);
  assert.match(err.message, /MCMC route/);
  assert.match(err.message, /unresolved: theta/);
  assert.match(err.message, /not implemented/);
  assert.match(err.message, /not invalid/);
});

test('the detail is optional and leaves no empty parenthesis', () => {
  const err: any = engineLimitation('iid over a tuple variate', 'sampling');
  assert.equal(err.code, ENGINE_LIMITATION);
  assert.equal(err.message.includes('()'), false, err.message);
  assert.match(err.message, /sampling route\. The model is not invalid/);
});

test('a model error and a non-error are not limitations', () => {
  const modelError: any = new Error('joint: singular');
  modelError.code = 'CLM_SINGULAR_JOINT';
  assert.equal(isEngineLimitation(modelError), false);
  assert.equal(isEngineLimitation(new Error('plain')), false);
  assert.equal(isEngineLimitation(null), false);
  assert.equal(isEngineLimitation(undefined), false);
});

test('the marker is reachable from the engine entry point', () => {
  // A host outside this package (viewer, testsuite) sees only `index.ts`.
  assert.equal(engine.limitations.ENGINE_LIMITATION, ENGINE_LIMITATION);
  assert.equal(typeof engine.limitations.engineLimitation, 'function');
  assert.equal(typeof engine.limitations.isEngineLimitation, 'function');
});
