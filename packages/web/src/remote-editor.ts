// @flatppl/web — the source editor while it lives in another window.
//
// When the user pops the source pane out, the playground keeps calling its
// editor exactly as before (setSource, getSource, setReadOnly, setLanguage,
// revealLine, replaceRange). This proxy stands in for the CodeMirror editor:
// it keeps a mirror of the text, and forwards each call to the source window
// through `send`. The source window is the only place the user types; its
// edits come back through `receiveEdit`, which updates the mirror before the
// playground's normal change handler runs. So drafts, the dirty marker,
// Save, bundle saves, editSource and revealLine keep one code path.
//
// The mirror lags the source window by the edits still in flight, so the
// two sides track revisions. Each setSource here starts a new epoch: the
// owner's text wins, and the window drops what it typed against an older
// epoch. Within an epoch the window numbers the edits it sends, and the
// mirror records the last one it received. replaceRange carries that
// revision, and the window maps the range through the changes it made since
// (pane-app.ts) before it applies it.
//
// Calls within one task coalesce into one `state.source` message, because
// the playground sets the text and then the language and read-only mode
// right after each other.
//
// Loaded as a plain <script> installing window.FlatPPLWebRemoteEditor.

'use strict';

(function (globalScope: any) {
  interface Initial { text: string; readOnly: boolean; language: string | null }
  interface Revision { epoch: string; textRev: number }

  // Epochs must differ across owner reloads, as the window outlives them.
  const EPOCH_PREFIX = Math.random().toString(36).slice(2);
  let epochCount = 0;
  function newEpoch(): string { return EPOCH_PREFIX + '.' + (++epochCount); }

  /**
   * send(type, body) — deliver a message to the source window.
   * path()           — the model path the router shows.
   */
  function create(opts: { send: (type: string, body: any) => void; path: () => string | null; initial: Initial }) {
    let text = opts.initial.text;
    let readOnly = opts.initial.readOnly;
    let language = opts.initial.language;
    let epoch = newEpoch();
    let textRev = 0;
    let pending = false;

    function state() {
      return { text, readOnly, language, path: opts.path(), epoch, textRev };
    }

    function flush(): void {
      if (!pending) return;
      pending = false;
      opts.send('state.source', state());
    }

    function schedule(): void {
      if (pending) return;
      pending = true;
      globalScope.queueMicrotask(flush);
    }

    return {
      setSource(next: any): void {
        text = next == null ? '' : String(next);
        epoch = newEpoch();
        textRev = 0;
        schedule();
      },
      getSource(): string { return text; },
      setReadOnly(ro: any): void { readOnly = !!ro; schedule(); },
      setLanguage(type: any): void { language = type == null ? null : String(type); schedule(); },
      revealLine(line: any, revealOpts?: any): void {
        flush();
        opts.send('state.sourceReveal', { line: line | 0, silent: !!(revealOpts && revealOpts.silent) });
      },
      replaceRange(from: any, to: any, insert: any): void {
        flush();
        // The source window applies it; its edit comes back through
        // receiveEdit like a keystroke, so the mirror follows from there.
        opts.send('state.sourceReplace', { from: from | 0, to: to | 0, text: String(insert), epoch, textRev });
      },
      /** The state to hand to the source window when it says hello. */
      state,
      /** An edit typed in the source window, tagged with its revision.
       *  Returns whether the mirror's text changed. */
      receiveEdit(next: string, at: Revision | null | undefined): boolean {
        // Typed against text this side has since replaced.
        if (!at || at.epoch !== epoch || !(at.textRev > textRev)) return false;
        textRev = at.textRev;
        // The window forgets the changes this revision covers.
        opts.send('state.sourceAck', { epoch, textRev });
        if (next === text) return false;
        text = next;
        return true;
      },
      destroy(): void { pending = false; },
    };
  }

  globalScope.FlatPPLWebRemoteEditor = { create };
})(typeof window !== 'undefined' ? window : globalThis);
