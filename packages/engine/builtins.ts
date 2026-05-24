'use strict';

// Catalogs of known FlatPPL names: constants, sets, functions, distributions,
// special operations, and measure algebra operations.

// Predefined constants (scalar values). `true`/`false` are handled by
// the parser as BoolLiteral AST nodes (variant-aware: `true`/`false`
// in FlatPPL/FlatPPJ; `True`/`False` in FlatPPY), so they don't sit
// in CONSTANTS — keeping them out also means a FlatPPY source that
// types bare `true` gets the right "Undefined variable" hint rather
// than being silently classified as a ConstantRef.
const CONSTANTS = new Set([
  'pi', 'inf', 'im',
]);

// Spellings of boolean literals across all variants — useful as a
// "would this name spell a boolean somewhere" check. The parser
// decides which spelling is active for a given variant.
const BOOL_LITERALS = new Set(['true', 'false', 'True', 'False']);

// Predefined sets
const SETS = new Set([
  'reals', 'posreals', 'nonnegreals', 'unitinterval',
  'posintegers', 'nonnegintegers', 'integers',
  'booleans', 'complexes',
  'rngstates', 'anything',
  // Axis selector keywords (spec §05 reserved words; surface forms
  // `:` and `!` lower to these). `all` selects an entire axis; `only`
  // selects the unique element of a length-1 axis.
  'all', 'only',
]);

// Set constructors (callables that build sets)
const SET_CONSTRUCTORS = new Set([
  'interval', 'cartprod', 'cartpow', 'stdsimplex',
]);

// Reserved names (not bindable as ordinary names)
const RESERVED_NAMES = new Set([
  'self', 'base',
]);

// Reserved binding names with special meaning at the module level
const SPECIAL_BINDINGS = new Set([
  'flatppl_compat',
]);

// Special operations — not ordinary function calls, have custom syntax rules
const SPECIAL_OPERATIONS = new Set([
  // Variates and reification
  'draw', 'lawof', 'functionof', 'kernelof', 'fn',
  // Inputs
  'elementof', 'external', 'valueset',
  // Module operations
  'load_module', 'standard_module', 'load_data',
  // Higher-order
  'broadcast', 'broadcasted', 'reduce', 'scan',
  // Multi-axis aggregation (spec §04 §sec:aggregate): three distinguished
  // inputs (reduction function, output_axes array, expr). Not an ordinary
  // function — the second arg contains symbolic axis labels and the third
  // contains expressions that bind those labels lexically. Its own
  // derivation kind in derivations.ts; materialiser dispatches via a
  // pattern table with a general nested-loop interpreter as fallback.
  'aggregate',
  // Function composition / annotation
  'fchain', 'bijection',
  // Assertions
  'checked',
  // Tuple/record/table constructors with structural meaning
  'record', 'table', 'tuple', 'vector', 'fixed',
  // Engine-internal: tuple_get(tuple_expr, slot_lit) — projects the
  // i-th element of a tuple-typed expression. Emitted by the analyzer's
  // multi-LHS rewriter (`a, b = rand(s, m)` → each name becomes a
  // tuple_get call) and never appears in user source. Listed here so
  // the lowerer treats it as a built-in op rather than routing through
  // the user-defined-function path.
  'tuple_get',
]);

// Built-in functions (with defined argument order — positional calling allowed)
const BUILTIN_FUNCTIONS = new Set([
  // Identity
  'identity',
  // Array/table generation
  'array', 'fill', 'zeros', 'ones', 'eye', 'onehot',
  'linspace', 'extlinspace',
  // Access and reshaping. `get` is FlatPPL/FlatPPJ's 1-based
  // indexing op; `get0` is FlatPPY's 0-based variant emitted by
  // the parser when xs[i] is lowered under the FlatPPY surface.
  // Semantically `get0(xs, i)` ≡ `get(xs, i + 1)`.
  'get', 'get0', 'cat', 'rowstack', 'colstack', 'partition', 'reverse', 'addaxes', 'relabel',
  // Reshaping additions (spec §07)
  'tile', 'splitblocks', 'joinblocks',
  // Scalar restrictions/constructors
  'boolean', 'integer', 'real', 'complex', 'string', 'imag',
  // Elementary math
  'exp', 'log', 'log10', 'log1p', 'expm1', 'pow', 'sqrt', 'abs', 'abs2',
  'sin', 'cos', 'tan',
  'asin', 'acos', 'atan', 'atan2',
  'sinh', 'cosh', 'tanh',
  'asinh', 'acosh', 'atanh',
  'min', 'max', 'floor', 'ceil', 'round',
  'div', 'mod',
  'conj', 'cis',
  'gamma', 'loggamma',
  'logit', 'invlogit', 'probit', 'invprobit',
  // Operator-equivalent functions
  'add', 'sub', 'mul', 'divide', 'neg',
  'equal', 'unequal', 'lt', 'le', 'gt', 'ge',
  // Predicates
  'isfinite', 'isinf', 'isnan', 'iszero',
  // Linear algebra
  'transpose', 'adjoint', 'det', 'logabsdet', 'inv', 'trace',
  'linsolve', 'lower_cholesky',
  'row_gram', 'col_gram', 'self_outer', 'diagmat',
  // Diagonal extract / block-matrix constructors (spec §07)
  'diag', 'blockdiagmat', 'bandedmat',
  // Reductions
  'sum', 'mean', 'var', 'std', 'prod', 'maximum', 'minimum', 'lengthof', 'sizeof',
  'cumsum', 'cumprod',
  // Norms and normalization
  'l1norm', 'l2norm', 'l1unit', 'l2unit',
  'logsumexp', 'softmax', 'logsoftmax',
  // Logic and conditionals
  'land', 'lor', 'lnot', 'lxor', 'ifelse',
  // Membership and filtering
  'filter', 'selectbins',
  // Binning
  'bincounts',
  // Approximation functions
  'polynomial', 'bernstein', 'stepwise',
  // Random value generation
  'rngstate', 'rnginit', 'rand',
]);

// Built-in distribution constructors (kernels)
const DISTRIBUTIONS = new Set([
  // Continuous
  'Uniform', 'Normal', 'GeneralizedNormal', 'Cauchy', 'StudentT',
  'Logistic', 'LogNormal', 'Exponential', 'Gamma', 'Weibull',
  'InverseGamma', 'Beta', 'ChiSquared', 'VonMises', 'Laplace',
  // Discrete
  'Bernoulli', 'Categorical', 'Categorical0', 'Binomial', 'Poisson',
  'Geometric', 'NegativeBinomial', 'NegativeBinomial2',
  // Multivariate
  'MvNormal', 'Wishart', 'InverseWishart',
  'LKJ', 'LKJCholesky', 'Dirichlet', 'Multinomial',
  // Composite
  'PoissonProcess', 'BinnedPoissonProcess',
  // Fundamental measures
  'Dirac', 'Lebesgue', 'Counting',
]);

// Measure algebra operations
const MEASURE_OPS = new Set([
  // Reweighting
  'weighted', 'logweighted', 'bayesupdate',
  // Normalization and mass
  'normalize', 'totalmass',
  // Composition
  'superpose', 'joint', 'iid', 'kchain', 'jointchain',
  // Restriction and transformation
  'truncate', 'pushfwd',
  // Likelihoods
  'likelihoodof', 'joint_likelihood',
  'densityof', 'logdensityof',
  // Disintegration
  'disintegrate',
  // Measure restriction (spec §06; lowers in the analyzer to
  // disintegrate + bayesupdate + likelihoodof, so no derivation kind
  // of its own).
  'restrict',
]);

// Built-in callables that produce a measure-typed value. Used to decide
// whether `functionof(expr, ...)` reifies a function (value expr) or a
// kernel (measure expr). NB: `totalmass`, `densityof`, `logdensityof`,
// `likelihoodof`, `joint_likelihood`, and `disintegrate` are excluded —
// they return scalars, density functions, or tuples, not measures.
const MEASURE_PRODUCING = new Set([
  ...DISTRIBUTIONS,
  'weighted', 'logweighted', 'bayesupdate', 'normalize',
  'superpose', 'joint', 'iid', 'kchain', 'jointchain',
  'truncate', 'pushfwd',
]);

// All known names (union of everything that is a built-in callable, set, or constant)
const ALL_KNOWN = new Set([
  ...CONSTANTS, ...SETS, ...SET_CONSTRUCTORS,
  ...SPECIAL_OPERATIONS, ...BUILTIN_FUNCTIONS, ...DISTRIBUTIONS, ...MEASURE_OPS,
  ...RESERVED_NAMES, ...SPECIAL_BINDINGS,
]);

function isKnownName(name: string) {
  return ALL_KNOWN.has(name);
}

function isConstant(name: string) {
  return CONSTANTS.has(name);
}

function isBoolLiteral(name: string) {
  return BOOL_LITERALS.has(name);
}

function isSet(name: string) {
  return SETS.has(name);
}

function isSpecialOperation(name: string) {
  return SPECIAL_OPERATIONS.has(name);
}

function isReserved(name: string) {
  return RESERVED_NAMES.has(name);
}

module.exports = {
  CONSTANTS, BOOL_LITERALS, SETS, SET_CONSTRUCTORS,
  RESERVED_NAMES, SPECIAL_BINDINGS, SPECIAL_OPERATIONS,
  BUILTIN_FUNCTIONS, DISTRIBUTIONS, MEASURE_OPS, MEASURE_PRODUCING, ALL_KNOWN,
  isKnownName, isConstant, isBoolLiteral, isSet, isSpecialOperation, isReserved,
};
