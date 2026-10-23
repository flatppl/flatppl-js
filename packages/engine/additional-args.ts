'use strict';

// Spec §04 "Additional arguments": an operation that supplies the leading arguments of a
// function or kernel binds the call's remaining arguments to the callable's remaining inputs,
// positionally or by name. A sole additional record or table passes whole if the remaining
// inputs accept one positional argument; otherwise its fields or columns bind the remaining
// inputs by name. No input may be left unbound or bound twice.
//
// This is the ONE statement of that rule. Every layer that applies it (type inference, the
// lift's call construction, the evaluator, the trajectory descriptors) supplies its own kind of
// element (a type, an AST, a value, an IR) and asks which element binds which input, so the rule
// cannot drift between layers.

interface AdditionalArguments<T> {
  /** The callable's input names, in order, or null when they are not known. */
  declared: string[] | null;
  /** How many leading inputs the operation supplies itself. */
  leading: number;
  /** Additional arguments given positionally. */
  positional: T[];
  /** Additional arguments given by name. */
  keywords: Record<string, T>;
  /** The fields (record) or columns (table) of the argument, when it is a record or table. */
  soleFields?: Record<string, T> | null;
}

/**
 * The elements bound to the callable's remaining inputs, in the callable's own order, or the
 * reason the arguments do not bind. With positional arguments the elements are returned as
 * given: how many inputs the callable declares is the caller's arity check.
 */
function bindAdditionalArguments<T>(args: AdditionalArguments<T>):
  { rest: T[] } | { error: string } {
  const { declared, leading, positional, keywords } = args;
  const names = Object.keys(keywords);
  if (names.length && positional.length) {
    return { error: 'additional arguments are all positional or all named, not both' };
  }
  let named: Record<string, T> | null = names.length ? keywords : null;
  if (!named && positional.length === 1 && args.soleFields && declared
      && declared.length - leading >= 2) {
    named = args.soleFields;
  }
  if (!named) return { rest: positional };
  if (!declared) {
    return { error: 'the callable\'s input names are not known, so additional arguments cannot bind by name' };
  }
  const rest = declared.slice(leading);
  for (const name of Object.keys(named)) {
    if (!rest.includes(name)) return { error: 'the callable has no remaining input named "' + name + '"' };
  }
  for (const name of rest) {
    if (!(name in named)) return { error: 'the remaining input "' + name + '" is not bound' };
  }
  return { rest: rest.map(name => named![name]) };
}

module.exports = { bindAdditionalArguments };
