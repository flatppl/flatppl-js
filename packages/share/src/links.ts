/**
 * Share links on the page side: bundle members, URLs and QR codes. This
 * module holds no codec, so pages load the dictionary only in the worker.
 */
import type { QRCode } from 'qrcode';
import { MAX_SHARE_URL_LENGTH, SHARE_ARCHIVE_LIMITS, type ShareMember } from './format.ts';

export const MAX_SHARE_QR_VERSION = 40;

export interface ShareTokens {
  link: string;
  /** Null when the payload exceeds the largest QR code. */
  qr: string | null;
}

export interface ShareLinks {
  url: string;
  /** The same share with a decimal token, or null when no QR code can hold it. */
  qrUrl: string | null;
  tooLong: boolean;
}

export interface ArchiveEngine {
  readArchive(bytes: Uint8Array, options: { id: string }): Promise<any>;
  writeArchive(files: Record<string, Uint8Array>, options: object): Promise<Uint8Array>;
}

/** Members relative to the bundle root, as the archive reader sees them. */
export async function archiveMembers(zip: Uint8Array, engine: ArchiveEngine): Promise<ShareMember[]> {
  const archive = await engine.readArchive(zip, { id: 'share.flatppl.zip' });
  try {
    const members: ShareMember[] = [];
    for (const [name, entry] of Object.entries(archive.entries) as [string, any][]) {
      members.push(entry.directory ? { name, kind: 'directory', bytes: new Uint8Array() }
        : entry.symlink ? { name, kind: 'symlink', bytes: new TextEncoder().encode(await archive.readLink(name)) }
        : { name, kind: 'file', bytes: await archive.read(name) });
    }
    return members;
  } finally { await archive.close(); }
}

/** A STORE ZIP of the members, as an opened share rebuilds it. */
export function membersArchive(members: ShareMember[], engine: ArchiveEngine): Promise<Uint8Array> {
  const files: Record<string, Uint8Array> = Object.create(null);
  const symlinks: string[] = [];
  for (const member of members) {
    files[member.kind === 'directory' ? member.name + '/' : member.name] = member.bytes;
    if (member.kind === 'symlink') symlinks.push(member.name);
  }
  return engine.writeArchive(files, { compression: 'store', symlinks, limits: SHARE_ARCHIVE_LIMITS });
}

/**
 * The token holds the bundle and the view, so the fragment holds nothing else.
 * A QR token starts with its marker and needs no `share=`.
 */
export function makeShareLinks(tokens: ShareTokens, baseUrl: string): ShareLinks {
  const base = new URL(baseUrl);
  base.search = '';
  base.hash = '';
  if (!['http:', 'https:'].includes(base.protocol)) throw new Error('Share links require an HTTP or HTTPS playground URL.');
  const url = base.href + '#share=' + tokens.link;
  return {
    url, qrUrl: tokens.qr === null ? null : base.href + '#' + tokens.qr,
    tooLong: url.length > MAX_SHARE_URL_LENGTH,
  };
}

/**
 * The QR library splits the text into optimal segments: numeric for the
 * token. Scheme and host ignore case, so upper case fits them into the
 * alphanumeric mode. Use medium error correction and allow every standard QR
 * size. The QR library and its renderers load only when this runs.
 */
export async function createShareQr(links: ShareLinks) {
  if (links.tooLong || links.qrUrl === null) return null;
  const imported = await import('qrcode');
  const qr = imported.default ?? imported;
  const { origin } = new URL(links.qrUrl);
  const text = origin.toUpperCase() + links.qrUrl.slice(origin.length);
  let matrix: QRCode;
  try { matrix = qr.create(text, { errorCorrectionLevel: 'M' }); }
  catch (error) {
    if (error instanceof Error && error.message.includes('amount of data is too big')) return null;
    throw error;
  }
  if (matrix.version > MAX_SHARE_QR_VERSION) return null;
  const options = {
    errorCorrectionLevel: 'M' as const, version: matrix.version,
    maskPattern: matrix.maskPattern, margin: 4, scale: 8,
  };
  const [svg, png] = await Promise.all([
    qr.toString(text, { ...options, type: 'svg' }),
    qr.toDataURL(text, { ...options, type: 'image/png' }),
  ]);
  return { url: text, svg, png, size: matrix.modules.size, version: matrix.version };
}
