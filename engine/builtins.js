'use strict';

// Catalogs of known FlatPPL names: constants, sets, functions, distributions,
// special operations, and measure algebra operations.

// Predefined constants (scalar values)
const CONSTANTS = new Set([
  'true', 'false', 'pi', 'inf', 'im',
]);

const BOOL_LITERALS = new Set(['true', 'false']);

// Predefined sets
const SETS = new Set([
  'reals', 'posreals', 'nonnegreals', 'unitinterval',
  'posintegers', 'nonnegintegers', 'integers',
  'booleans', 'complexes',
  'rngstates', 'anything',
  'all', // axis selector
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
  'draw', 'lawof', 'functionof', 'fn',
  // Inputs
  'elementof', 'external', 'valueset',
  // Module operations
  'load_module', 'standard_module', 'load_data',
  // Higher-order
  'broadcast', 'broadcasted', 'reduce', 'scan',
  // Function composition / annotation
  'fchain', 'bijection',
  // Assertions
  'checked',
  // Tuple/record/table constructors with structural meaning
  'record', 'table', 'tuple', 'vector', 'preset', 'fixed',
]);

// Built-in functions (with defined argument order — positional calling allowed)
const BUILTIN_FUNCTIONS = new Set([
  // Identity
  'identity',
  // Array/table generation
  'array', 'fill', 'zeros', 'ones', 'eye', 'onehot',
  'linspace', 'extlinspace',
  // Access and reshaping
  'get', 'cat', 'rowstack', 'colstack', 'partition', 'reverse', 'relabel',
  // Scalar restrictions/constructors
  'boolean', 'integer', 'real', 'complex', 'string', 'imag',
  // Elementary math
  'exp', 'log', 'log10', 'pow', 'sqrt', 'abs', 'abs2',
  'sin', 'cos', 'tan',
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
  // Reductions
  'sum', 'mean', 'var', 'prod', 'maximum', 'minimum', 'length',
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
  'InverseGamma', 'Beta',
  // Discrete
  'Bernoulli', 'Categorical', 'Binomial', 'Poisson',
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
  'superpose', 'joint', 'iid', 'chain', 'jointchain',
  // Restriction and transformation
  'truncate', 'pushfwd',
  // Likelihoods
  'likelihoodof', 'joint_likelihood',
  'densityof', 'logdensityof',
  // Disintegration
  'disintegrate',
]);

// All known names (union of everything that is a built-in callable, set, or constant)
const ALL_KNOWN = new Set([
  ...CONSTANTS, ...SETS, ...SET_CONSTRUCTORS,
  ...SPECIAL_OPERATIONS, ...BUILTIN_FUNCTIONS, ...DISTRIBUTIONS, ...MEASURE_OPS,
  ...RESERVED_NAMES, ...SPECIAL_BINDINGS,
]);

function isKnownName(name) {
  return ALL_KNOWN.has(name);
}

function isConstant(name) {
  return CONSTANTS.has(name);
}

function isBoolLiteral(name) {
  return BOOL_LITERALS.has(name);
}

function isSet(name) {
  return SETS.has(name);
}

function isSpecialOperation(name) {
  return SPECIAL_OPERATIONS.has(name);
}

function isReserved(name) {
  return RESERVED_NAMES.has(name);
}

module.exports = {
  CONSTANTS, BOOL_LITERALS, SETS, SET_CONSTRUCTORS,
  RESERVED_NAMES, SPECIAL_BINDINGS, SPECIAL_OPERATIONS,
  BUILTIN_FUNCTIONS, DISTRIBUTIONS, MEASURE_OPS, ALL_KNOWN,
  isKnownName, isConstant, isBoolLiteral, isSet, isSpecialOperation, isReserved,
};
