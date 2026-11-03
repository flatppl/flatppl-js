// @flatppl/web — the message session between the playground window and the
// windows its panels were moved into.
//
// One playground window (the owner) holds the model and the router. Each
// detached panel runs in its own window (a replica) and talks to the owner
// through one channel per session. Every message is an envelope:
//
//   { p: 'flatppl-pane', v: 1, sid, from, epoch, seq, to?, type, body }
//
// - `p` and `sid` keep foreign traffic and other playground tabs out.
// - `v` is the protocol version. After a deploy, two windows can run
//   different builds; a receiver drops an unknown version and reports it.
// - `epoch` is random per page load of the sender, so a reload is visible.
// - `seq` counts per (from, epoch); a receiver drops anything not newer,
//   which removes duplicates on the window.postMessage fallback.
// - Owner state messages carry `body.rev`, a counter within the owner's
//   epoch. RevisionTracker applies only newer revisions and reports a new
//   owner epoch, after which the replica asks for a full snapshot.
//
// Transport: BroadcastChannel when the browser has it (it survives a reload
// of either side without window references), else window.postMessage with
// an origin check. The owner reaches replicas through their window handles;
// a replica reaches the owner through window.opener. Replicas only ever talk
// to the owner, and ignore each other, so the fallback needs no relay.
//
// Liveness: each side pings every interval and treats any message as a
// sign of life. A peer counts as lost only after the timeout while both
// documents are visible, because browsers throttle timers in hidden pages.
//
// Loaded as a plain <script> installing window.FlatPPLPaneSession.

'use strict';

(function (globalScope: any) {
  const NAMESPACE = 'flatppl-pane';
  const VERSION = 1;

  interface Envelope {
    p: string; v: number; sid: string; from: string; epoch: string;
    seq: number; to?: string; type: string; body: any;
  }

  interface Transport {
    post(envelope: Envelope, to?: string): void;
    listen(handler: (envelope: any, source: any) => void): void;
    close(): void;
  }

  interface Clock {
    now(): number;
    setInterval(fn: () => void, ms: number): any;
    clearInterval(id: any): void;
  }

  /** 128 random bits as hex: session ids and epochs. */
  function randomId(): string {
    const bytes = new Uint8Array(16);
    globalScope.crypto.getRandomValues(bytes);
    return Array.from(bytes, (b: number) => b.toString(16).padStart(2, '0')).join('');
  }

  function channelName(sid: string): string {
    return NAMESPACE + ':' + sid;
  }

  /** The primary transport: one BroadcastChannel per session. */
  function broadcastTransport(sid: string, Channel: any = globalScope.BroadcastChannel): Transport {
    const channel = new Channel(channelName(sid));
    return {
      post(envelope) { channel.postMessage(envelope); },
      listen(handler) { channel.onmessage = (event: any) => handler(event.data, null); },
      close() { channel.close(); },
    };
  }

  /**
   * The fallback transport: window.postMessage, same origin only.
   * `targets()` returns the windows to post to: the owner's replica handles,
   * or a replica's opener. `self` receives.
   */
  function windowTransport(opts: { self: any; origin: string; targets: () => any[] }): Transport {
    let listener: any = null;
    return {
      post(envelope) {
        for (const target of opts.targets()) {
          try { target.postMessage(envelope, opts.origin); } catch (_) { /* closed window */ }
        }
      },
      listen(handler) {
        listener = (event: any) => {
          if (event.origin !== opts.origin) return;
          handler(event.data, event.source);
        };
        opts.self.addEventListener('message', listener);
      },
      close() { if (listener) opts.self.removeEventListener('message', listener); },
    };
  }

  /** The transport this browser supports for a session. */
  function defaultTransport(sid: string, targets: () => any[]): Transport {
    if (typeof globalScope.BroadcastChannel === 'function') return broadcastTransport(sid);
    return windowTransport({ self: globalScope, origin: globalScope.location.origin, targets });
  }

  /**
   * One endpoint of a session. `id` is 'owner' or 'pane:<panel>'.
   * Handlers:
   *   onMessage(envelope, source) — a valid, new envelope addressed here;
   *   onVersion(v)                — a peer speaks another protocol version;
   *   onPeerEpoch(from, epoch, previous) — a peer page loaded again;
   *   onLost(from)                — a watched peer stopped answering.
   */
  function createSession(opts: {
    sid: string; id: string; transport: Transport; epoch?: string;
    clock?: Clock; isVisible?: () => boolean;
    onMessage: (envelope: Envelope, source: any) => void;
    onVersion?: (v: unknown) => void;
    onPeerEpoch?: (from: string, epoch: string, previous: string | undefined) => void;
    onLost?: (from: string) => void;
  }) {
    const epoch = opts.epoch || randomId();
    const clock: Clock = opts.clock || {
      now: () => Date.now(),
      setInterval: (fn, ms) => globalScope.setInterval(fn, ms),
      clearInterval: (id) => globalScope.clearInterval(id),
    };
    const visible = opts.isVisible || (() => !globalScope.document || globalScope.document.visibilityState !== 'hidden');
    let seq = 0;
    let closed = false;
    const lastSeq = new Map<string, { epoch: string; seq: number }>();
    const watched = new Map<string, { heard: number; peerVisible: boolean }>();
    let pinger: any = null;

    function send(type: string, body: any = {}, to?: string): Envelope {
      if (closed) throw new Error('pane session is closed');
      const envelope: Envelope = { p: NAMESPACE, v: VERSION, sid: opts.sid, from: opts.id, epoch, seq: ++seq, type, body };
      if (to) envelope.to = to;
      opts.transport.post(envelope, to);
      return envelope;
    }

    function receive(raw: any, source: any): void {
      if (closed || !raw || typeof raw !== 'object' || raw.p !== NAMESPACE) return;
      if (raw.sid !== opts.sid || raw.from === opts.id) return;
      // Replicas talk only to the owner, never to each other.
      if (opts.id !== 'owner' && raw.from !== 'owner') return;
      if (raw.v !== VERSION) { if (opts.onVersion) opts.onVersion(raw.v); return; }
      if (typeof raw.from !== 'string' || typeof raw.epoch !== 'string' || !Number.isInteger(raw.seq)
          || typeof raw.type !== 'string') return;
      if (raw.to && raw.to !== opts.id) return;
      const last = lastSeq.get(raw.from);
      if (last && last.epoch === raw.epoch && raw.seq <= last.seq) return;
      if (last && last.epoch !== raw.epoch && opts.onPeerEpoch) opts.onPeerEpoch(raw.from, raw.epoch, last.epoch);
      if (!last && opts.onPeerEpoch) opts.onPeerEpoch(raw.from, raw.epoch, undefined);
      lastSeq.set(raw.from, { epoch: raw.epoch, seq: raw.seq });
      const peer = watched.get(raw.from);
      if (peer) {
        peer.heard = clock.now();
        if (raw.body && typeof raw.body.visible === 'boolean') peer.peerVisible = raw.body.visible;
      }
      if (raw.type === 'ping') { send('pong', { visible: visible() }, raw.from); return; }
      if (raw.type === 'pong') return;
      opts.onMessage(raw as Envelope, source);
    }

    opts.transport.listen(receive);

    /** Start counting silence from `from`; it is lost after `timeout` ms
     *  of silence while both sides are visible. */
    function watch(from: string, interval: number, timeout: number): void {
      watched.set(from, { heard: clock.now(), peerVisible: true });
      if (pinger) return;
      pinger = clock.setInterval(function () {
        if (closed) return;
        const now = clock.now();
        const here = visible();
        for (const [peer, state] of watched) {
          // A hidden side runs throttled timers, so its silence proves nothing.
          if (!here || !state.peerVisible) { state.heard = now; continue; }
          if (now - state.heard > timeout) {
            watched.delete(peer);
            if (opts.onLost) opts.onLost(peer);
          }
        }
        if (watched.size) send('ping', { visible: here });
      }, interval);
    }

    function unwatch(from: string): void {
      watched.delete(from);
      if (!watched.size && pinger) { clock.clearInterval(pinger); pinger = null; }
    }

    return {
      epoch, send, watch, unwatch,
      close(): void {
        if (closed) return;
        closed = true;
        if (pinger) clock.clearInterval(pinger);
        opts.transport.close();
      },
    };
  }

  /**
   * Owner state ordering on a replica. accept(envelope) returns:
   *   'apply'     — newer revision of the current owner epoch;
   *   'stale'     — not newer, drop it;
   *   'new-epoch' — first message of another owner page load: the replica
   *                 asks for a snapshot; the tracker now follows that epoch.
   */
  function createRevisionTracker() {
    let epoch: string | null = null;
    let rev = 0;
    return {
      accept(envelope: Envelope): 'apply' | 'stale' | 'new-epoch' {
        const r = envelope.body && envelope.body.rev;
        if (!Number.isInteger(r)) return 'stale';
        if (envelope.epoch !== epoch) {
          const first = epoch === null;
          epoch = envelope.epoch;
          rev = r;
          return first ? 'apply' : 'new-epoch';
        }
        if (r <= rev) return 'stale';
        rev = r;
        return 'apply';
      },
      reset(): void { epoch = null; rev = 0; },
      position(): { epoch: string | null; rev: number } { return { epoch, rev }; },
    };
  }

  globalScope.FlatPPLPaneSession = {
    NAMESPACE, VERSION,
    randomId, channelName,
    broadcastTransport, windowTransport, defaultTransport,
    createSession, createRevisionTracker,
  };
})(typeof window !== 'undefined' ? window : globalThis);
