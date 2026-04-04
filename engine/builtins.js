'use strict';

// Catalogs of known FlatPPL names: constants, sets, functions, distributions,
// special forms, and measure algebra operations.

const CONSTANTS = new Set([
  'true', 'false', 'pi', 'inf', 'im',
]);

const BOOL_LITERALS = new Set(['true', 'false']);

const SETS = new Set([
  'reals', 'posreals', 'nonnegreals', 'integers', 'complexes', 'anything', 'all',
]);

// Special forms — not ordinary function calls, have custom syntax rules
const SPECIAL_FORMS = new Set([
  'draw', 'elementof', 'valueset',
  'lawof', 'functionof', 'fn',
  'load_module', 'load_table',
]);

// Built-in functions (with defined argument order — positional calling allowed)
const BUILTIN_FUNCTIONS = new Set([
  // Elementary
  'exp', 'log', 'log10', 'pow', 'sqrt', 'abs', 'abs2',
  'sin', 'cos', 'tan',
  'min', 'max', 'floor', 'ceil',
  // Complex
  'complex', 'real', 'imag', 'conj', 'cis',
  // Array generation
  'fill', 'eye', 'onehot', 'linspace', 'extlinspace',
  // Access and reshaping
  'get', 'cat', 'rowstack', 'colstack', 'record', 'table', 'relabel',
  // Reductions
  'sum', 'product', 'length',
  // Logic and conditionals
  'land', 'lor', 'lnot', 'lxor', 'ifelse',
  // Linear algebra
  'transpose', 'adjoint', 'det', 'logabsdet', 'inv', 'trace',
  'linsolve', 'lower_cholesky', 'row_gram', 'col_gram',
  'self_outer', 'diagmat',
  // Interpolation
  'interp_pwlin', 'interp_pwexp',
  'interp_poly2_lin', 'interp_poly6_lin', 'interp_poly6_exp',
  // Shape functions
  'polynomial', 'bernstein', 'stepwise',
  // Filtering and binning
  'filter', 'selectbins', 'bincounts',
  // Set construction
  'interval',
  // Higher-order
  'broadcast', 'fchain', 'bijection',
  // Density
  'densityof', 'logdensityof',
]);

// Built-in distribution constructors (keyword-only calling)
const DISTRIBUTIONS = new Set([
  'Normal', 'Exponential', 'LogNormal', 'Gamma', 'Beta',
  'Uniform', 'Poisson', 'ContinuedPoisson',
  'Bernoulli', 'Binomial', 'MvNormal',
  'PoissonProcess',
  // HEP-specific
  'CrystalBall', 'DoubleSidedCrystalBall', 'Argus',
  'BreitWigner', 'RelativisticBreitWigner', 'Voigtian',
  'BifurcatedGaussian', 'GeneralizedNormal',
  // Fundamental measures
  'Dirac', 'Lebesgue', 'Counting',
]);

// Measure algebra operations
const MEASURE_OPS = new Set([
  'weighted', 'logweighted', 'bayesupdate',
  'normalize', 'totalmass',
  'superpose',
  'joint', 'iid',
  'chain', 'jointchain',
  'truncate',
  'pushfwd',
  'likelihoodof', 'joint_likelihood',
]);

// All known names (union of everything above)
const ALL_KNOWN = new Set([
  ...CONSTANTS, ...SETS, ...SPECIAL_FORMS,
  ...BUILTIN_FUNCTIONS, ...DISTRIBUTIONS, ...MEASURE_OPS,
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

function isSpecialForm(name) {
  return SPECIAL_FORMS.has(name);
}

module.exports = {
  CONSTANTS, BOOL_LITERALS, SETS, SPECIAL_FORMS,
  BUILTIN_FUNCTIONS, DISTRIBUTIONS, MEASURE_OPS, ALL_KNOWN,
  isKnownName, isConstant, isBoolLiteral, isSet, isSpecialForm,
};
