// Worker error boundary shared by the handler and its hosts. Error objects do
// not retain custom fields under structured clone; carry only the documented,
// clone-safe diagnostics (limitations.ts), never arbitrary thrown properties.

function copyMetadata(source: any, target: any) {
  if (typeof source?.stack === 'string') target.stack = source.stack;
  if (typeof source?.code === 'string') target.code = source.code;
  const limitation = source?.limitation;
  if (typeof limitation?.construct === 'string' && typeof limitation?.route === 'string') {
    target.limitation = { construct: limitation.construct, route: limitation.route };
  }
  return target;
}

/** Encode both synchronous and asynchronous failures as ordinary message data. */
export function errorReply(error: any, id: any) {
  return copyMetadata(error, {
    type: 'error', id,
    message: typeof error?.message === 'string' ? error.message : String(error),
  });
}

/** Reconstruct an Error without erasing the host-visible classification. */
export function errorFromReply(reply: any): Error {
  return copyMetadata(reply, new Error(reply.message || 'worker error'));
}
