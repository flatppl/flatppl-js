'use strict';

// Spec-pattern smoke tests — drawn directly from flatppl-design
// §02 (overview), §03 (value types), §04 (design), §06 (measure
// algebra), §07 (functions), §08 (distributions), §10 (worked
// examples). Each test takes a near-spec source pattern, runs it
// through `processSource`, and asserts the engine parses + analyses
// without errors. A separate group runs the materialiser on
// fixed-phase patterns to catch end-to-end evaluation regressions.
//
// The point isn't deep semantic verification — the per-feature test
// files (aggregate, only-axis, density, etc.) cover that. This file
// guards against the *class* of bug where a simple spec pattern
// trips an unhandled code path (the immediate motivator being the
// `iid(Normal, [3, 3])` regression: every static analysis was fine,
// but a runtime walker silently coerced the vector size).
//
// Patterns are grouped by spec section so failures point at a
// concrete idiom in the spec.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');

function errors(src: string) {
  return processSource(src).diagnostics.filter(
    (d: any) => d.severity === 'error');
}

function parsesCleanly(src: string, label?: string) {
  const errs = errors(src);
  assert.equal(errs.length, 0,
    `${label || ''}: expected clean parse + analyse, got: `
    + errs.map((d: any) => `[L${(d.loc && d.loc.start && d.loc.start.line) ?? '?'}] ${d.message}`).join('; '));
}

// ---------------------------------------------------------------------
// §02 Overview — first-example HEP model
// ---------------------------------------------------------------------

test('§02 first example: HEP signal+background+resolution model parses', () => {
  parsesCleanly(`
n_sig = elementof(reals)
n_bkg = elementof(reals)
raw_syst ~ Normal(mu = 0.0, sigma = 1.0)
resolution = 2.5 + 0.3 * raw_syst
signal_shape = Normal(mu = 125.0, sigma = resolution)
background_shape = Exponential(rate = 0.05)
observed_data = [120.1, 124.8, 125.3, 130.2, 135.7, 142.0]
intensity = superpose(
    weighted(n_sig, signal_shape),
    weighted(n_bkg, background_shape)
)
events ~ PoissonProcess(intensity = intensity)
L = likelihoodof(kernelof(events), observed_data)
`);
});

// ---------------------------------------------------------------------
// §02 Core concepts + Tour of FlatPPL
// ---------------------------------------------------------------------

test('§02 tour: scalars + collections + indexing + records', () => {
  parsesCleanly(`
x = 3.14
n = 42
b = true
v = [1.0, 2.0, 3.0]
nested = [[1, 2], [3, 4]]
M = rowstack([[1, 2, 3], [4, 5, 6]])
r = record(mu = 3.0, sigma = 1.0)
y = v[2]
z = M[1, 2]
w = r.mu
col_j = M[:, 2]
`);
});

test('§02 tour: arithmetic + comparisons + function calls', () => {
  parsesCleanly(`
mu_sig = elementof(reals)
background = elementof(reals)
efficiency = 0.5
x = elementof(reals)
rate = efficiency * mu_sig + background
is_positive = x > 0
y = exp(x)
z = ifelse(is_positive, mu_sig, background)
`);
});

test('§02 tour: tilde-decomposition into named scalars', () => {
  parsesCleanly(`
mean = [1.0, 2.0, 3.0]
cov_matrix = rowstack([[1.0, 0.1, 0.1], [0.1, 1.0, 0.1], [0.1, 0.1, 1.0]])
a, b, c ~ MvNormal(mu = mean, cov = cov_matrix)
`);
});

test('§02 tour: complex arithmetic', () => {
  parsesCleanly(`
z1 = complex(3.0, 2.0)
z2 = 3.0 + 2.0 * im
phase = cis(3 * pi / 4)
A_sig = z1
A_bkg = z2
coupling = z1
A_total = A_sig * coupling + A_bkg
intensity = abs2(A_total)
x = real(z1)
y = imag(z1)
z_bar = conj(z1)
`);
});

test('§02 tour: draw / lawof / functionof / kernelof', () => {
  parsesCleanly(`
mu = elementof(reals)
sigma = elementof(interval(0.0, inf))
a ~ Normal(mu = mu, sigma = sigma)
M = lawof(a)
b = 2 * a + 1
f = functionof(b, x = a)
K = kernelof(b, x = a)
`);
});

test('§02 tour: broadcast over arrays (deterministic)', () => {
  parsesCleanly(`
A = [1.0, 2.0, 3.0]
f = x -> x * 2
C = broadcast(f, x = A)
`);
});

test('§02 tour: kernel broadcast (stochastic)', () => {
  parsesCleanly(`
A = [1.0, 2.0, 3.0]
K = fn(Normal(mu = _, sigma = 0.1))
D ~ broadcast(K, A)
`);
});

test('§02 tour: relabel + get-subset + pushfwd projection', () => {
  parsesCleanly(`
mean = [1.0, 2.0, 3.0]
cov = rowstack([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
some_array = [1.0, 2.0, 3.0]
some_record = record(a = 1.0, b = 2.0, c = 3.0)
field_a = get(some_record, "a")
sub = get(some_record, ["a", "c"])
named = relabel(some_array, ["a", "b", "c"])
mvmodel = relabel(MvNormal(mu = mean, cov = cov), ["a", "b", "c"])
marginal_ac = pushfwd(fn(get(_, ["a", "c"])), mvmodel)
`);
});

test('§02 tour: iid + superpose + normalize + joint + jointchain + kchain', () => {
  parsesCleanly(`
n_sig = elementof(reals)
sig = Normal(mu = 0, sigma = 1)
bkg = Exponential(rate = 1)
M1 = Normal(mu = 0, sigma = 1)
M2 = Normal(mu = 1, sigma = 1)
xs ~ iid(Normal(mu = 0, sigma = 1), 100)
sp = superpose(weighted(n_sig, sig), bkg)
mix = normalize(superpose(weighted(0.7, M1), weighted(0.3, M2)))
j = joint(M1, M2)
prior = Normal(mu = 0, sigma = 1)
forward_kernel = fn(Normal(mu = _, sigma = 0.1))
pp = kchain(prior, forward_kernel)
`);
});

test('§02 tour: hierarchical joint via pushfwd-relabel', () => {
  parsesCleanly(`
M1 = Normal(mu = 0, sigma = 1)
K_b = fn(Normal(mu = _, sigma = 0.1))
hj = jointchain(
    pushfwd(fn(relabel(_, ["a"])), M1),
    pushfwd(fn(relabel(_, ["b"])), K_b))
`);
});

test('§02 tour: truncate + Lebesgue + density-defined distribution', () => {
  parsesCleanly(`
c0 = elementof(reals)
c1 = elementof(reals)
c2 = elementof(reals)
lo = elementof(reals)
hi = elementof(reals)
positive_normal = truncate(Normal(mu = 0, sigma = 1), interval(0, inf))
leb = Lebesgue(support = reals)
bern = fn(bernstein(coefficients = [c0, c1, c2], x = _))
smooth_shape = normalize(weighted(bern, Lebesgue(support = interval(lo, hi))))
`);
});

test('§02 tour: anonymous functions (lambda + fn)', () => {
  parsesCleanly(`
a0 = elementof(reals)
a1 = elementof(reals)
a2 = elementof(reals)
poly = x -> polynomial(coefficients = [a0, a1, a2], x = x)
squared = x -> x^2
ratio_sq = (a, b) -> (a / b)^2
neg = fn(0 - _)
poly_fn = fn(polynomial(coefficients = [a0, a1, a2], x = _))
g = fn(_ * _)
`);
});

// ---------------------------------------------------------------------
// §03 Value types — presets, predefined sets
// ---------------------------------------------------------------------

test('§03 presets: preset point with fixed', () => {
  parsesCleanly(`
L_init = record(a = 2.0, b = [4, 5, 6], c = fixed(8.0))
`);
});

test('§03 presets: preset domain with cartprod', () => {
  parsesCleanly(`
L_domain = cartprod(a = interval(0, 5),
                    b = cartpow(interval(-10, 10), 3),
                    c = interval(0, 20))
`);
});

test('§03 sets: every predefined set parses in a measure context', () => {
  parsesCleanly(`
m1 = Lebesgue(support = reals)
m2 = Lebesgue(support = posreals)
m3 = Lebesgue(support = nonnegreals)
m4 = Lebesgue(support = unitinterval)
m5 = Counting(support = integers)
m6 = Counting(support = posintegers)
m7 = Counting(support = nonnegintegers)
m8 = Counting(support = booleans)
`);
});

test('§03 sets: cartpow and stdsimplex in elementof', () => {
  parsesCleanly(`
n_dim = elementof(reals)
x_3d = elementof(cartpow(reals, 3))
x_3x3 = elementof(cartpow(reals, [3, 3]))
x_simplex = elementof(stdsimplex(5))
`);
});

// ---------------------------------------------------------------------
// §04 Design — phases, reification, lambdas
// ---------------------------------------------------------------------

test('§04 phases: fixed / parameterized / stochastic ancestors', () => {
  parsesCleanly(`
n_dims = external(posintegers)
mu = elementof(reals)
sigma = elementof(interval(0.0, inf))
dist = iid(Normal(mu = mu, sigma = sigma), n_dims)
x ~ dist
y = 2 * x
`);
});

test('§04 functionof: boundary inputs + Identity law', () => {
  parsesCleanly(`
a = elementof(reals)
b = elementof(reals)
c = a^2
d = max(b, 1.5)
e = c * d
f = functionof(e)
g = functionof(e, p = a, q = d)
h = functionof(e, a = a, d = d)
`);
});

test('§04 kernelof: spec equivalence example', () => {
  parsesCleanly(`
theta1 ~ Normal(mu = 0.0, sigma = 1.0)
theta2 ~ Exponential(rate = 1.0)
a = 5.0 * theta1
b = abs(theta1) * theta2
obs ~ iid(Normal(mu = a, sigma = b), 10)
joint_model = lawof(record(theta1 = theta1, theta2 = theta2, obs = obs))
prior_predictive = lawof(record(obs = obs))
prior = lawof(record(theta1 = theta1, theta2 = theta2))
forward_kernel = kernelof(record(obs = obs), theta1 = theta1, theta2 = theta2)
`);
});

test('§04 kernelof: pure-measure-algebra equivalent', () => {
  parsesCleanly(`
theta1 = elementof(reals)
theta2 = elementof(posreals)
a = 5.0 * theta1
b = abs(theta1) * theta2
obs_dist = iid(Normal(mu = a, sigma = b), 10)
prior = joint(theta1 = Normal(mu = 0.0, sigma = 1.0),
              theta2 = Exponential(rate = 1.0))
forward_kernel = functionof(obs_dist)
joint_model = jointchain(prior, forward_kernel)
prior_predictive = kchain(prior, forward_kernel)
`);
});

test('§04 placeholders: nested functionof scoping', () => {
  parsesCleanly(`
b = elementof(reals)
some_value = elementof(reals)
f = functionof(functionof(_a_ * b, a = _a_)(some_value) + _a_, a = _a_)
`);
});

test('§04 bijection: annotated function', () => {
  parsesCleanly(`
exp_bij = bijection(exp, log, identity)
`);
});

test('§04 fchain: deterministic composition', () => {
  parsesCleanly(`
calc_kinematics = identity
apply_cuts = identity
pipeline = fchain(calc_kinematics, apply_cuts)
`);
});

test('§04 broadcast: NumPy- and Julia-style alignment via addaxes', () => {
  parsesCleanly(`
A = rowstack([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = [10.0, 20.0]
f = (x, y) -> x + y
C_numpy = broadcast(f, A, addaxes(b, 1, 0))
`);
});

// ---------------------------------------------------------------------
// §06 Measure algebra — combinator coverage
// ---------------------------------------------------------------------

test('§06 weighted / logweighted', () => {
  parsesCleanly(`
M = Normal(mu = 0, sigma = 1)
x = elementof(reals)
w_lin = weighted(2.5, M)
w_log = logweighted(0.5, M)
w_fn  = weighted(fn(abs(_) + 0.1), M)
`);
});

test('§06 normalize + totalmass', () => {
  parsesCleanly(`
M1 = weighted(0.3, Normal(0, 1))
M2 = weighted(0.7, Normal(1, 1))
mix = normalize(superpose(M1, M2))
Z = totalmass(superpose(M1, M2))
`);
});

test('§06 joint: positional + keyword + IID forms', () => {
  parsesCleanly(`
M1 = Normal(mu = 0, sigma = 1)
M2 = Exponential(rate = 1.0)
vj = joint(M1, M2)
rj = joint(name1 = M1, name2 = M2)
xs ~ iid(Normal(mu = 0, sigma = 1), 100)
mat ~ iid(Normal(mu = 0, sigma = 1), [3, 3])
`);
});

test('§06 kchain + jointchain with kernels', () => {
  parsesCleanly(`
prior = joint(theta1 = Normal(0, 1), theta2 = Exponential(1))
kernel = fn(Normal(mu = 0, sigma = 1))
marginal = kchain(prior, kernel)
joint_dep = jointchain(prior, kernel)
`);
});

test('§06 truncate + pushfwd', () => {
  parsesCleanly(`
M = Normal(mu = 0, sigma = 1)
half = normalize(truncate(M, interval(0, inf)))
ln = pushfwd(exp, Normal(0, 1))
proj = pushfwd(fn(get(_, ["a", "c"])),
               relabel(iid(Normal(0, 1), 3), ["a", "b", "c"]))
`);
});

test('§06 likelihoodof + joint_likelihood + bayesupdate', () => {
  parsesCleanly(`
mu = elementof(reals)
model1 = functionof(Normal(mu = mu, sigma = 1.0))
model2 = functionof(Normal(mu = 2.0 * mu, sigma = 0.5))
L1 = likelihoodof(model1, 1.5)
L2 = likelihoodof(model2, 3.2)
L = joint_likelihood(L1, L2)
prior = joint(mu = Normal(0, 2.0))
posterior = bayesupdate(L, prior)
`);
});

test('§06 region-restricted likelihood (Poisson process)', () => {
  parsesCleanly(`
lambda_bar = elementof(posreals)
intensity = weighted(lambda_bar, Lebesgue(support = reals))
obs_events = [1.2, 3.4, 5.1, 2.8]
R = interval(-3.0, 3.0)
obs_R = filter(fn(_ in R), obs_events)
model_R = PoissonProcess(intensity = truncate(intensity, R))
L_R = likelihoodof(functionof(model_R), obs_R)
`);
});

test('§06 disintegrate + restrict for posterior', () => {
  parsesCleanly(`
sigma = 1.0
a ~ Normal(mu = 0.0, sigma = 2.0)
b ~ Normal(mu = a, sigma = sigma)
joint_model = lawof(record(a = a, b = b))
forward_kernel, prior = disintegrate(["b"], joint_model)
obs = record(b = 2.1)
L = likelihoodof(forward_kernel, obs)
posterior = bayesupdate(L, prior)
`);
});

test('§06 restrict for prior parameter pinning', () => {
  parsesCleanly(`
prior = joint(mu = Normal(0, 1), sigma = Exponential(1))
restricted = restrict(prior, sigma = 0.8)
`);
});

// ---------------------------------------------------------------------
// §07 Functions — value-level ops
// ---------------------------------------------------------------------

test('§07 array generation: fill / zeros / ones / eye / onehot / linspace', () => {
  parsesCleanly(`
v1 = fill(0, 10)
v2 = fill(0, [2, 3])
z = zeros([4, 4])
o = ones([2, 2, 2])
I = eye(3)
e2 = onehot(2, 5)
ls = linspace(0.0, 10.0, 5)
xs = extlinspace(0.0, 10.0, 5)
`);
});

test('§07 indexing: scalar / array / record / slicing', () => {
  parsesCleanly(`
v = [10.0, 20.0, 30.0, 40.0]
M = rowstack([[1, 2, 3], [4, 5, 6]])
r = record(a = 1, b = 2, c = 3)
e1 = v[2]
e2 = M[1, 2]
row = M[1, :]
col = M[:, 2]
fa = r.a
fb = get(r, "b")
sub = get(r, ["a", "c"])
mixed = get(M, [1, 2], 2)
`);
});

test('§07 array reshaping ops', () => {
  parsesCleanly(`
v1 = [1, 2, 3]
v2 = [4, 5, 6]
M = rowstack([[1, 2], [3, 4]])
joined = cat(v1, v2)
M_rs = rowstack([v1, v2])
M_cs = colstack([v1, v2])
parts = partition([1, 2, 3, 4, 5, 6], [2, 3, 1])
rev = reverse(v1)
expanded = addaxes(v1, 1, 1)
`);
});

test('§07 linear algebra: transpose / det / inv / linsolve / lower_cholesky', () => {
  parsesCleanly(`
A = rowstack([[2.0, 0.0], [0.0, 3.0]])
b = [4.0, 6.0]
At = transpose(A)
d  = det(A)
ld = logabsdet(A)
Ai = inv(A)
x  = linsolve(A, b)
L  = lower_cholesky(rowstack([[4.0, 2.0], [2.0, 3.0]]))
`);
});

test('§07 reductions: sum / mean / var / std / max / min / lengthof / sizeof', () => {
  parsesCleanly(`
v = [1.0, 2.0, 3.0, 4.0, 5.0]
M = rowstack([[1.0, 2.0], [3.0, 4.0]])
s = sum(v)
m = mean(v)
vr = var(v)
sd = std(v)
mx = maximum(v)
mn = minimum(v)
lv = lengthof(v)
sM = sizeof(M)
`);
});

test('§07 norms + softmax family', () => {
  parsesCleanly(`
v = [1.0, 2.0, 3.0]
n1 = l1norm(v)
n2 = l2norm(v)
u1 = l1unit(v)
u2 = l2unit(v)
lse = logsumexp(v)
sm = softmax(v)
lsm = logsoftmax(v)
`);
});

test('§07 logic and conditionals', () => {
  parsesCleanly(`
a = true
b = false
c = land(a, b)
d = lor(a, b)
e = lnot(a)
f = lxor(a, b)
x = elementof(reals)
g = ifelse(x > 0, 1.0, -1.0)
`);
});

test('§07 predicates', () => {
  parsesCleanly(`
x = elementof(reals)
yf = isfinite(x)
yi = isinf(x)
yn = isnan(x)
yz = iszero(x)
`);
});

test('§07 filter + selectbins + bincounts', () => {
  parsesCleanly(`
data = [1.2, 3.4, 5.1, 2.8, 4.0]
edges = [0.0, 2.5, 5.0, 7.5, 10.0]
counts = [10, 12, 15, 8]
sub  = filter(fn(_ in interval(2.0, 8.0)), data)
sel  = selectbins(edges, interval(2.0, 8.0), counts)
binned = bincounts(edges, data)
`);
});

test('§07 random: rnginit + rand + scalar', () => {
  parsesCleanly(`
rngseed = [0xb2, 0x51, 0xa4, 0x93, 0x49, 0xd8, 0x68, 0x88]
rstate = rnginit(rngseed)
x, rstate2 = rand(rstate, Normal(0, 1))
xs, rstate3 = rand(rstate2, iid(Normal(0, 1), 10))
`);
});

test('§07 random: rand with multi-axis iid (the regression motivator)', () => {
  parsesCleanly(`
rstate = rnginit([0xb2, 0x51, 0xa4, 0x93])
A, _ = rand(rstate, iid(Normal(0, 1), [3, 3]))
B, _ = rand(rstate, iid(Normal(0, 1), [2, 3, 4]))
`);
});

test('§07 polynomial / bernstein / stepwise', () => {
  parsesCleanly(`
coeffs = [1.0, 2.0, 3.0]
edges = [0.0, 1.0, 2.0, 3.0]
values = [10.0, 20.0, 30.0]
x = elementof(reals)
p1 = polynomial(coefficients = coeffs, x = x)
p2 = bernstein(coefficients = [0.5, 0.3, 0.2], x = x)
p3 = stepwise(edges = edges, values = values, x = x)
`);
});

// ---------------------------------------------------------------------
// §08 Distributions — at least one binding per family
// ---------------------------------------------------------------------

test('§08 univariate continuous distributions parse', () => {
  parsesCleanly(`
d1  = Uniform(support = interval(0, 1))
d2  = Normal(mu = 0, sigma = 1)
d3  = GeneralizedNormal(mean = 0, alpha = 1, beta = 2)
d4  = Cauchy(location = 0, scale = 1)
d5  = StudentT(nu = 3)
d6  = Logistic(mu = 0, s = 1)
d7  = LogNormal(mu = 0, sigma = 1)
d8  = Exponential(rate = 1)
d9  = Gamma(shape = 2, rate = 1)
d10 = Weibull(shape = 2, scale = 1)
d11 = InverseGamma(shape = 2, scale = 1)
d12 = Beta(alpha = 2, beta = 3)
d13 = ChiSquared(k = 3)
d14 = VonMises(mu = 0, kappa = 1)
d15 = Laplace(location = 0, scale = 1)
`);
});

test('§08 univariate discrete distributions parse', () => {
  parsesCleanly(`
d1 = Bernoulli(p = 0.5)
d2 = Categorical(p = [0.3, 0.5, 0.2])
d3 = Categorical0(p = [0.3, 0.5, 0.2])
d4 = Binomial(n = 10, p = 0.3)
d5 = Geometric(p = 0.4)
d6 = NegativeBinomial(alpha = 2, beta = 3)
d7 = NegativeBinomial2(mu = 5, psi = 2)
d8 = Poisson(rate = 3.2)
`);
});

test('§08 multivariate distributions parse', () => {
  parsesCleanly(`
mu = [0.0, 0.0, 0.0]
cov = rowstack([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
d1 = MvNormal(mu = mu, cov = cov)
d2 = Wishart(nu = 5, scale = cov)
d3 = InverseWishart(nu = 5, scale = cov)
d4 = LKJ(n = 3, eta = 1)
d5 = LKJCholesky(n = 3, eta = 1)
d6 = Dirichlet(alpha = [1.0, 1.0, 1.0])
d7 = Multinomial(n = 10, p = [0.3, 0.5, 0.2])
`);
});

test('§08 composite distributions: PoissonProcess + BinnedPoissonProcess', () => {
  parsesCleanly(`
edges = linspace(0.0, 10.0, 5)
intensity = weighted(5.0, Normal(mu = 5, sigma = 2))
pp = PoissonProcess(intensity = intensity)
bpp = BinnedPoissonProcess(bins = edges, intensity = intensity)
`);
});

// ---------------------------------------------------------------------
// §10 Worked examples — full HEP analysis (spec §10 §sec:example)
// ---------------------------------------------------------------------

test('§10 worked HEP example parses end-to-end', () => {
  parsesCleanly(`
lo = 0.0
hi = 10.0
bin_edges = linspace(lo, hi, 5)
signal_bins = [1.0, 2.0, 3.0, 1.0]
bkg_bins = [0.5, 0.4, 0.3, 0.2]
mu_sig = elementof(reals)
raw_eff_syst ~ Normal(mu = 0.0, sigma = 1.0)
efficiency = 0.9 + 0.05 * raw_eff_syst
sig_shape = fn(stepwise(edges = bin_edges, values = signal_bins, x = _))
bkg_shape = fn(stepwise(edges = bin_edges, values = bkg_bins, x = _))
signal_template = normalize(weighted(sig_shape, Lebesgue(support = interval(lo, hi))))
bkg_template = normalize(weighted(bkg_shape, Lebesgue(support = interval(lo, hi))))
rate = superpose(
    weighted(mu_sig * efficiency, signal_template),
    bkg_template
)
events ~ PoissonProcess(intensity = rate)
L_obs = likelihoodof(
    kernelof(events, raw_eff_syst = raw_eff_syst),
    [3.1, 5.7, 2.4, 8.9, 4.2])
aux_eff ~ Normal(mu = raw_eff_syst, sigma = 1.0)
L_constr = likelihoodof(kernelof(aux_eff, raw_eff_syst = raw_eff_syst), 0.0)
L = joint_likelihood(L_obs, L_constr)
`);
});

test('§10 worked example: Bayesian posterior add-on', () => {
  parsesCleanly(`
mu_sig_prior ~ Uniform(support = interval(0, 20))
raw_eff_syst_prior ~ Normal(mu = 0, sigma = 1)
prior = lawof(record(mu_sig = mu_sig_prior, raw_eff_syst = raw_eff_syst_prior))
L_obs = likelihoodof(functionof(Normal(mu = mu_sig_prior, sigma = 1)), 1.0)
posterior = bayesupdate(L_obs, prior)
`);
});

test('§10 additional patterns: hypothesis-testing IID models', () => {
  parsesCleanly(`
mass_data = [90.1, 91.8, 124.5, 125.2]
model_H0 = iid(Normal(mu = 91.2, sigma = 2.5), 4)
model_H1 = iid(Normal(mu = 125.0, sigma = 3.0), 4)
L_H0 = likelihoodof(model_H0, mass_data)
L_H1 = likelihoodof(model_H1, mass_data)
`);
});

test('§10 additional: density-defined distribution + Bernstein', () => {
  parsesCleanly(`
c0 = elementof(reals)
c1 = elementof(reals)
c2 = elementof(reals)
c3 = elementof(reals)
lo = elementof(reals)
hi = elementof(reals)
bern = fn(bernstein(coefficients = [c0, c1, c2, c3], x = _))
smooth_bkg = normalize(weighted(bern, Lebesgue(support = interval(lo, hi))))
`);
});

test('§10 additional: variate naming with pushfwd-relabel + likelihood', () => {
  parsesCleanly(`
mean = [0.0, 0.0, 0.0]
cov = rowstack([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
mvmodel = pushfwd(fn(relabel(_, ["a", "b", "c"])),
                  MvNormal(mu = mean, cov = cov))
L_mv = likelihoodof(functionof(mvmodel), record(a = 1.1, b = 2.1, c = 3.1))
`);
});

test('§10 additional: kernel broadcast over a parameter array', () => {
  parsesCleanly(`
a = elementof(reals)
noisy ~ Normal(mu = a, sigma = 0.1)
K = kernelof(noisy, a = a)
A = [1.0, 2.0, 3.0, 4.0]
noisy_array ~ broadcast(K, a = A)
`);
});

// ---------------------------------------------------------------------
// Mixed multi-axis idioms — multi-axis iid, cartpow, etc.
// ---------------------------------------------------------------------

test('multi-axis idioms: matrix-valued elementof / iid / draw', () => {
  parsesCleanly(`
mu = elementof(reals)
some_cov = elementof(cartpow(reals, [3, 3]))
mat ~ iid(Normal(mu = mu, sigma = 1), [3, 3])
cube ~ iid(Normal(mu = mu, sigma = 1), [2, 3, 4])
`);
});

test('multi-axis idioms: dimension referenced by binding', () => {
  parsesCleanly(`
n = external(posintegers)
mat ~ iid(Normal(mu = 0, sigma = 1), [n, n])
`);
});

// ---------------------------------------------------------------------
// Decomposition LHS forms
// ---------------------------------------------------------------------

test('decomposition: array LHS', () => {
  parsesCleanly(`
arr = [1.0, 2.0, 3.0]
a, b, c = arr
`);
});

test('decomposition: record LHS', () => {
  parsesCleanly(`
r = record(x = 1, y = 2)
x, y = r
`);
});

test('decomposition: tuple LHS', () => {
  parsesCleanly(`
t = (1, 2, 3)
a, b, c = t
`);
});

test('decomposition: drop with bare `_`', () => {
  parsesCleanly(`
rstate = rnginit([1, 2, 3, 4])
value, _ = rand(rstate, Normal(0, 1))
`);
});

// ---------------------------------------------------------------------
// Higher-order: reduce / scan / broadcasted
// ---------------------------------------------------------------------

test('higher-order: reduce + scan', () => {
  parsesCleanly(`
xs = [1.0, 2.0, 3.0, 4.0]
add_op = (a, b) -> a + b
total = reduce(add_op, xs)
cums  = scan(add_op, 0.0, xs)
`);
});

test('higher-order: broadcasted curried form', () => {
  parsesCleanly(`
bcadd = broadcasted(add)
A = [1.0, 2.0, 3.0]
B = [10.0, 20.0, 30.0]
out = bcadd(A, B)
`);
});

// ---------------------------------------------------------------------
// Dot-notation broadcasting (spec §05 / §04)
// ---------------------------------------------------------------------

test('dot-notation: f.(args) and a .op b', () => {
  parsesCleanly(`
A = [1.0, 2.0, 3.0]
B = [10.0, 20.0, 30.0]
squared = (x -> x*x).(A)
sum_v = A .+ B
neg_v = .- A
`);
});

// ---------------------------------------------------------------------
// Aggregate (spec §04 §sec:aggregate)
// ---------------------------------------------------------------------

test('aggregate spec examples: all four §04 forms parse', () => {
  parsesCleanly(`
A = rowstack([[1.0, 2.0], [3.0, 4.0]])
B = rowstack([[5.0, 6.0], [7.0, 8.0]])
W = [0.5, 0.5]
M = rowstack([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
C = aggregate(sum, [.i, .k], A[.i, .j] * B[.j, .k])
D = aggregate(sum, [.i, .k], (A[.i, .j] - B[.j, .k])^2 * W[.j])
V = aggregate(var, [.j], M[.i, .j])
S = aggregate(sum, [.i], M[.i, 1])
`);
});

test('aggregate := shorthand for matmul', () => {
  parsesCleanly(`
A = rowstack([[1.0, 2.0], [3.0, 4.0]])
B = rowstack([[5.0, 6.0], [7.0, 8.0]])
C[.i, .k] := A[.i, .j] * B[.j, .k]
`);
});

// ---------------------------------------------------------------------
// Final smoke tests — known fixtures parse
// ---------------------------------------------------------------------

test('fixture: minimal.flatppl parses', () => {
  const fs = require('node:fs');
  const path = require('node:path');
  const file = path.join(__dirname, 'fixtures', 'minimal.flatppl');
  if (!fs.existsSync(file)) return;
  parsesCleanly(fs.readFileSync(file, 'utf8'), 'minimal');
});

test('fixture: bayesian_inference_3.flatppl parses', () => {
  const fs = require('node:fs');
  const path = require('node:path');
  const file = path.join(__dirname, 'fixtures', 'bayesian_inference_3.flatppl');
  if (!fs.existsSync(file)) return;
  parsesCleanly(fs.readFileSync(file, 'utf8'), 'bi3');
});
