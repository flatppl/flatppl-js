// @flatppl/viewer — pure utility/formatter module (Phase 4b).
//
// Functions in this module have no per-mount state dependency: they
// take only their declared parameters (plus a few that consult the
// runtime global `FlatPPLEngine.rng` and `FlatPPLEngine.empirical` /
// `FlatPPLEngine.orchestrator` namespaces — same pattern the rest of
// the viewer uses for the engine bundle). Safe to test in isolation,
// safe to import from any other viewer module.
export function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

export function truncateExpr(expr) {
  if (!expr) return '';
  // Truncate array literals: [1, 2, 3, ..., 8, 9, 10]
  var arrMatch = expr.match(/^\\[(.+)\\]$/);
  if (arrMatch) {
    var items = arrMatch[1].split(/\\s*,\\s*/);
    if (items.length > 6) {
      var head = items.slice(0, 3).join(', ');
      var tail = items.slice(-3).join(', ');
      return '[' + head + ', \u2026, ' + tail + ']';
    }
  }
  // Truncate other long expressions in the middle
  if (expr.length > 80) {
    return expr.slice(0, 38) + ' \u2026 ' + expr.slice(-38);
  }
  return expr;
}

/**
 * True when every sample equals the first one. Catches:
 *   - literal bindings (c = 5.0)
 *   - derived constants (d = c + 1) — same value at every i
 *   - aliases to constants
 *   - degenerate distributions (e.g. Normal(0, 0))
 * Plotting a histogram for these is just a single tall bar; render
 * the scalar value as text instead. The check is O(n) but n=5000
 * floats is sub-millisecond.
 */
export function samplesAreConstant(samples) {
  if (!samples || samples.length === 0) return false;
  var v = samples[0];
  for (var i = 1; i < samples.length; i++) if (samples[i] !== v) return false;
  return true;
}

export function formatScalar(v) {
  if (!Number.isFinite(v)) return String(v);
  if (Number.isInteger(v)) return String(v);
  return String(parseFloat(v.toPrecision(4)));
}

export function formatComplexScalar(re, im) {
  var imv = im === 0 ? 0 : im;            // normalise -0 → 0
  var sign = imv < 0 ? ' - ' : ' + ';
  return formatScalar(re) + sign + formatScalar(Math.abs(imv)) + ' i';
}

export function complexReBadge() {
  var b = document.createElement('span');
  b.textContent = 'complex — showing Re(z)';
  b.title = 'This binding is complex-valued; the histogram is its '
    + 'real part. Modulus / Im / Argand views are planned.';
  b.style.padding = '0.15em 0.6em';
  b.style.borderRadius = '3px';
  b.style.fontSize = '0.92em';
  b.style.background = 'var(--vscode-badge-background, #4d4d4d)';
  b.style.color = 'var(--vscode-badge-foreground, #fff)';
  return b;
}

export function formatArrayParts(parts, fullLength, maxShown) {
  var max = maxShown == null ? 8 : maxShown;
  var n = parts.length;
  if (fullLength == null) fullLength = n;
  if (n <= max) return '[' + parts.join(', ') + ']';
  var headN = 3, tailN = 1;
  var head = parts.slice(0, headN);
  var tail = parts.slice(n - tailN, n);
  // No "(length N)" suffix — see formatValue array branch for
  // rationale. Keeping both array-formatters in sync.
  return '[' + head.join(', ') + ', …, ' + tail.join(', ') + ']';
}

export function formatArrayWithEllipsis(values, maxShown) {
  var parts = new Array(values.length);
  for (var i = 0; i < values.length; i++) parts[i] = formatScalar(values[i]);
  return formatArrayParts(parts, values.length, maxShown);
}

/**
 * Pretty-print a FlatPIR IR node as canonical FlatPPL surface
 * syntax. Used by the fixed-Dirac viewer path to render the
 * value argument of `Dirac(value = ...)` without evaluating it
 * (since the value may be non-scalar — record, array — that
 * the engine's main-thread evaluator can't materialise without
 * running the worker).
 *
 * Handles the IR shapes the value-position of a fixed binding
 * can plausibly take: literals, named constants (pi, inf),
 * binding refs, unary neg of literals, vector / record literals.
 * Anything more exotic (calls into transcendental ops, etc.)
 * gets a placeholder "<op>(…)" so the surface form stays
 * legible without claiming false precision.
 */
export function formatIRValue(ir) {
  if (!ir) return '?';
  if (ir.kind === 'lit')   return formatScalar(ir.value);
  if (ir.kind === 'const') return ir.name; // pi / e / inf / true / false
  if (ir.kind === 'ref') {
    return (ir.ns && ir.ns !== 'self' ? ir.ns + '.' : '') + ir.name;
  }
  if (ir.kind === 'call' && ir.op === 'neg' && ir.args && ir.args.length === 1) {
    return '-' + formatIRValue(ir.args[0]);
  }
  if (ir.kind === 'call' && ir.op === 'vector' && Array.isArray(ir.args)) {
    return '[' + ir.args.map(formatIRValue).join(', ') + ']';
  }
  if (ir.kind === 'call' && ir.op === 'record') {
    var entries = [];
    var kwargs = ir.kwargs || {};
    for (var k in kwargs) {
      entries.push(k + ' = ' + formatIRValue(kwargs[k]));
    }
    return 'record(' + entries.join(', ') + ')';
  }
  if (ir.kind === 'call' && ir.op === 'tuple' && Array.isArray(ir.args)) {
    return '(' + ir.args.map(formatIRValue).join(', ') + ')';
  }
  if (ir.kind === 'call' && ir.op) {
    // Generic call: op(arg1, arg2, k=v) — useful for
    // fchain-style nestings without committing to a precise
    // pretty-print of unknown ops.
    var parts = [];
    if (Array.isArray(ir.args)) {
      for (var i = 0; i < ir.args.length; i++) parts.push(formatIRValue(ir.args[i]));
    }
    var kw = ir.kwargs || {};
    for (var k2 in kw) parts.push(k2 + ' = ' + formatIRValue(kw[k2]));
    return ir.op + '(' + parts.join(', ') + ')';
  }
  return '?';
}

export function formatValue(v, opts) {
  if (typeof v === 'number')  return formatScalar(v);
  if (typeof v === 'boolean') return String(v);
  if (typeof v === 'string')  return JSON.stringify(v);
  if (v == null)              return 'null';
  var isArrayLike = Array.isArray(v) || ArrayBuffer.isView(v);
  if (isArrayLike) {
    var len = v.length;
    var max = (opts && opts.maxArray) || 8;
    if (len <= max) {
      var parts = new Array(len);
      for (var i = 0; i < len; i++) parts[i] = formatValue(v[i], opts);
      return '[' + parts.join(', ') + ']';
    }
    var headN = 3, tailN = 1;
    var head = new Array(headN);
    for (var hi = 0; hi < headN; hi++) head[hi] = formatValue(v[hi], opts);
    var tail = new Array(tailN);
    for (var ti = 0; ti < tailN; ti++) {
      tail[ti] = formatValue(v[len - tailN + ti], opts);
    }
    // Ellipsis form drops the explicit "(length N)" suffix —
    // panes are often narrow and the elided "…" already signals
    // the array continues. Callers that need the count can
    // surface it separately (the corner-plot axis labels and
    // the info panel both already do).
    return '[' + head.join(', ') + ', …, ' + tail.join(', ') + ']';
  }
  if (typeof v === 'object') {
    var keys = Object.keys(v);
    var entries = new Array(keys.length);
    for (var k = 0; k < keys.length; k++) {
      entries[k] = keys[k] + ' = ' + formatValue(v[keys[k]], opts);
    }
    return 'record(' + entries.join(', ') + ')';
  }
  return String(v);
}

export function formatLogTotalmass(logTotalmass) {
  if (!Number.isFinite(logTotalmass)) {
    return logTotalmass === -Infinity ? '0 (zero mass)' : null;
  }
  if (Math.abs(logTotalmass) < 1e-9) return null;   // ≈ normalized
  // Float64 representable: ~ log(Number.MAX_VALUE) ≈ 709.78. Use a
  // tighter range so the linear form stays human-readable; outside
  // that, render in exp(...) form which stays meaningful at any
  // scale.
  if (Math.abs(logTotalmass) <= 12) {
    var linear = Math.exp(logTotalmass);
    if (linear >= 0.01 && linear < 10000) return linear.toPrecision(4);
    return linear.toExponential(3);
  }
  return 'exp(' + (logTotalmass >= 0 ? '+' : '') + logTotalmass.toPrecision(5) + ')';
}

/**
 * Tooltip text for the quality-readout span. Spells out the
 * diagnostic ingredients so a hover gives the full picture.
 */
export function qualityTooltip(q) {
  var parts = [
    'Importance-sampling quality: ' + q.label,
    '',
    'Kish ESS: ' + Math.round(q.ess).toLocaleString('en-US')
      + ' / ' + q.N.toLocaleString('en-US')
      + ' (' + (q.ratio * 100).toFixed(1) + '%)',
  ];
  if (Number.isFinite(q.kHat)) {
    parts.push('PSIS k̂: ' + q.kHat.toFixed(3)
      + '  (≤0.5 finite variance · ≤0.7 usable · >1 untrustworthy)');
  } else {
    parts.push('PSIS k̂: not applicable (unweighted measure)');
  }
  parts.push('Max single-atom weight: ' + (q.wmax * 100).toFixed(2) + '%');
  parts.push('Effective DOF (estimate): ' + q.dof);
  return parts.join('\n');
}

export function measureAtomCount(measure) {
  // Record / tuple measures have no top-level .samples; pull
  // length from any sub-measure's samples — all components share
  // the same atom count by construction.
  if (measure.fields) {
    var anyKey = Object.keys(measure.fields)[0];
    return anyKey ? measureAtomCount(measure.fields[anyKey]) : 0;
  }
  if (Array.isArray(measure.elems) && measure.elems.length > 0) {
    return measureAtomCount(measure.elems[0]);
  }
  // Array-shape measure: samples is a flat atom-major buffer of
  // length N × stride (e.g. iid(Normal, 10) at N=100k → buffer
  // length 1M, dims=[10]). Divide out the stride so N reads as
  // 100,000, not 1,000,000. Scalar measures have no dims (or
  // dims=[]) and pass through unchanged.
  if (measure.samples) {
    if (measure.dims && measure.dims.length > 0) {
      var stride = measure.dims.reduce(function(p, n) { return p * n; }, 1);
      return stride > 0 ? measure.samples.length / stride : 0;
    }
    return measure.samples.length;
  }
  return 0;
}

export function formatCount(n) {
  // Integer-formatted count with thousands separators.
  return Math.round(n).toLocaleString('en-US');
}

export function formatSampleCount(n) {
  if (n > 0 && Math.floor(n) === n) {
    var lg = Math.log10(n);
    if (lg >= 2 && Number.isInteger(lg)) {
      var sup = '';
      var s = String(lg);
      for (var i = 0; i < s.length; i++) {
        sup += '⁰¹²³⁴⁵⁶⁷⁸⁹'[+s[i]];
      }
      return '10' + sup;
    }
  }
  return formatCount(n);
}

export function hexToRgba(hex, alpha) {
  var m = /^#([0-9a-f]{6})$/i.exec(hex);
  if (!m) return hex;
  var n = parseInt(m[1], 16);
  var r = (n >> 16) & 0xff, g = (n >> 8) & 0xff, b = n & 0xff;
  return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
}

/**
 * Wrap a Philox state in a closure that returns U(0,1) uniforms,
 * matching the "() => number" callback shape that
 * empirical.systematicResample / multinomialResample expect.
 * Used for deterministic main-thread resampling under a
 * per-binding seed (currently: superpose).
 *
 * Same-engine RNG as the worker's sampleN, but instantiated locally
 * so empirical.js stays dep-free of rng.js — visualPanel does
 * the wiring at the call site.
 */
export function makeMainThreadPrng(seed) {
  var state = FlatPPLEngine.rng.stateFromKey(seed);
  return function() {
    var pair = FlatPPLEngine.rng.nextUniform(state);
    state = pair[1];
    return pair[0];
  };
}

/**
 * Common echarts zoom config — mouse-wheel + drag zoom on x via
 * the inside-type dataZoom, plus a top-left toolbox button for
 * rectangle-select zoom and a reset button. y-axis stays fixed:
 * zooming probability values alone is rarely useful and the
 * rectangle-select would otherwise need a second click to reset
 * each axis.
 *
 * filterMode 'none' keeps out-of-window data rendered (just
 * clipped) so panning is smooth.
 *
 * Returned fresh each call so the caller can pass to setOption.
 */
export function plotZoomOptions(fg) {
  return {
    dataZoom: [
      { type: 'inside', xAxisIndex: 0, filterMode: 'none' },
    ],
    toolbox: {
      show: true,
      left: 12, top: 4,
      itemSize: 14,
      iconStyle: { borderColor: fg, opacity: 0.55 },
      emphasis: { iconStyle: { borderColor: fg, opacity: 1 } },
      feature: {
        dataZoom: {
          yAxisIndex: 'none',
          title: { zoom: 'Zoom rectangle', back: 'Reset zoom' },
        },
        restore: { title: 'Reset' },
      },
    },
  };
}

/**
 * Enumerate the plottable scalar leaves of a multivariate
 * EmpiricalMeasure, with display labels and synthetic per-axis
 * sample arrays.
 *
 *   - record fields with scalar samples → axis with the field name.
 *   - record fields with array samples → one axis per array slot,
 *     labelled "field[i]" (1-indexed per spec §03 line 148);
 *     the per-axis samples array is a strided extract from the
 *     atom-major buffer.
 *   - top-level array measures → one axis per slot, labelled
 *     "[i]" (the binding name itself shows up in the bubble).
 *
 * Strided extracts allocate fresh Float64Arrays — small (~N
 * elements) so cheap; gives downstream code the same scalar-shape
 * Float64Array regardless of whether the source was a flat record
 * field or a column of an iid array.
 */
export function listScalarAxes(measure) {
  var out = [];
  function walk(m, prefix) {
    if (m.fields) {
      var ks = Object.keys(m.fields);
      for (var i = 0; i < ks.length; i++) {
        walk(m.fields[ks[i]], prefix ? (prefix + '.' + ks[i]) : ks[i]);
      }
      return;
    }
    if (m.shape === 'tuple' && Array.isArray(m.elems)) {
      // Positional analogue of record. 1-indexed component
      // labels per FlatPPL convention (xs[1], xs[2], ...).
      for (var ti = 0; ti < m.elems.length; ti++) {
        var label = (prefix ? prefix : '') + '[' + (ti + 1) + ']';
        walk(m.elems[ti], label);
      }
      return;
    }
    if (m.shape === 'array' && m.samples instanceof Float64Array && m.dims) {
      // Total inner stride per atom = prod(dims).
      var k = m.dims.reduce(function(p, n) { return p * n; }, 1);
      var N = m.samples.length / k;
      for (var slot = 0; slot < k; slot++) {
        // Stride-extract slot j across atoms.
        var col = new Float64Array(N);
        for (var i = 0; i < N; i++) col[i] = m.samples[i * k + slot];
        // 1-indexed labels per FlatPPL convention.
        var label = (prefix ? prefix : '') + '[' + (slot + 1) + ']';
        out.push({ key: label, label: label, samples: col });
      }
      return;
    }
    if (m.samples instanceof Float64Array) {
      // Plain scalar leaf.
      out.push({ key: prefix, label: prefix, samples: m.samples });
    }
  }
  walk(measure, '');
  return out;
}

/**
 * Walk the derivation chain for a measure binding to find the
 * value-typed binding it's mathematically equivalent to (if any).
 * Two equivalence forms after engine canonicalisation:
 *
 *   m = lawof(observed_data)   → derivation 'alias' to observed_data
 *   m = Dirac(observed_data)   → derivation 'sample' on Dirac IR
 *                                whose kwargs.value is a ref to
 *                                observed_data
 *
 * Both routes resolve to 'observed_data' here. Composes through
 * multiple alias hops; cycle-guarded.
 *
 * Returns null when the chain doesn't bottom out on a single
 * named value-typed source binding (e.g. Dirac(value = literal),
 * Dirac(value = inline-call), or a non-degenerate sample step).
 * The caller then falls through to the existing dispatch.
 */
export function resolveMeasureAlias(name, derivations, bindings) {
  if (!derivations || !bindings) return null;
  var seen = new Set();
  var cur = name;
  while (cur && !seen.has(cur)) {
    seen.add(cur);
    var d = derivations[cur];
    if (!d) return null;
    if (d.kind === 'alias') { cur = d.from; continue; }
    // Engine-canonicalised Dirac with a ref to a known binding
    // → follow the ref. Anything else (literal value, inline
    // expression, non-Dirac sample) terminates the resolution.
    if (d.kind === 'sample' && d.distIR
        && d.distIR.kind === 'call' && d.distIR.op === 'Dirac'
        && d.distIR.kwargs && d.distIR.kwargs.value) {
      var v = d.distIR.kwargs.value;
      if (v.kind === 'ref' && v.ns === 'self' && bindings.has(v.name)) {
        cur = v.name;
        continue;
      }
    }
    break;
  }
  return cur === name ? null : cur;
}

export function domainBoundsText(kwargOrder, ranges, setNames) {
  var parts = [];
  for (var i = 0; i < kwargOrder.length; i++) {
    var k = kwargOrder[i];
    var r = ranges && ranges[k];
    if (r) {
      parts.push(k + ' ∈ [' + formatScalar(r.lo) + ', ' + formatScalar(r.hi) + ']');
    } else if (setNames && setNames[k]) {
      parts.push(k + ' ∈ ' + setNames[k]);
    }
  }
  return parts.length ? parts.join(', ') : '(no range)';
}

export function isPersistableSetField(v) {
  if (!v) return false;
  if (v.type === 'CallExpr' && v.callee && v.callee.name === 'interval'
      && Array.isArray(v.args) && v.args.length === 2
      && v.args[0].type === 'NumberLiteral'
      && v.args[1].type === 'NumberLiteral') return true;
  if (v.type === 'Identifier' && KNOWN_NAMED_SETS[v.name]) return true;
  return false;
}

// Recognised cartprod named-set identifiers (the bareword form a
// FlatPPL domain may use, e.g. `cartprod(x = reals, y = posreals)`).
// Used by isPersistableSetField above. Defined as a module-private
// const — not exported because no other module reads it.
const KNOWN_NAMED_SETS = {
  reals: 1, posreals: 1, nonnegreals: 1, unitinterval: 1,
  integers: 1, posintegers: 1, nonnegintegers: 1, booleans: 1,
};

export function presetValuesText(values) {
  var text = formatValue(values);
  if (text.indexOf('record(') === 0 && text.charAt(text.length - 1) === ')') {
    return text.slice('record('.length, -1);
  }
  return text;
}

export function defaultValueForLeafType(leafType) {
  if (!leafType) return 0;
  if (leafType.kind === 'scalar') {
    if (leafType.prim === 'integer') return 0;
    if (leafType.prim === 'boolean') return false;
    return 0;
  }
  return 0;
}

export function defaultRangeForLeafType(leafType) {
  if (leafType && leafType.kind === 'scalar' && leafType.prim === 'integer') {
    return [-10, 10];
  }
  return [-5, 5];
}

export function rangeFromSetDescriptor(descriptor) {
  if (!descriptor) return null;
  switch (descriptor.kind) {
    case 'interval':       return [descriptor.lo, descriptor.hi];
    case 'reals':          return [-5, 5];
    case 'posreals':       return [0.01, 5];
    case 'nonnegreals':    return [0, 5];
    case 'unitinterval':   return [0, 1];
    case 'integers':       return [-10, 10];
    case 'posintegers':    return [1, 20];
    case 'nonnegintegers': return [0, 20];
    case 'booleans':       return [0, 1];
    default:               return null;
  }
}

export function filterOverrideToAxes(override, axisKwargs, key) {
  if (!override) return null;
  var src = override[key] || {};
  var dst = {};
  var kept = false;
  for (var k in src) {
    if (!Object.prototype.hasOwnProperty.call(src, k)) continue;
    if (!axisKwargs.has(k)) continue;
    dst[k] = src[k];
    kept = true;
  }
  if (!kept) return null;
  var out = Object.assign({}, override);
  out[key] = dst;
  return out;
}

export function bubbleMemberIds(r, allReifications) {
  var ids = {};
  for (var i = 0; i < r.kernel.length; i++) ids[r.kernel[i]] = true;
  for (var j = 0; j < allReifications.length; j++) {
    var r2 = allReifications[j];
    if (r2 === r || !ids[r2.name]) continue;
    for (var k = 0; k < r2.kernel.length; k++) ids[r2.kernel[k]] = true;
  }
  return ids;
}
