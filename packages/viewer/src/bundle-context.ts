// Bundle fields shared by viewer updates, workers, and Math (§04 Bundles).
// Bytes stay separate from module text. Import edges retain archive identities.

export type BundleFiles = Record<string, Uint8Array>;
export type BundleResolutions = Record<string, {
  modules: Record<string, string>;
  data: Record<string, string>;
}>;

export interface BundleContext {
  path?: string | null;
  bundleSources?: Record<string, string> | null;
  bundleFiles?: BundleFiles | null;
  bundleResolutions?: BundleResolutions | null;
}

/** Parsing needs source text and reached data, not copies of the ZIP containers. */
function dataFiles(context: BundleContext): BundleFiles {
  const files = context.bundleFiles || {};
  if (!context.bundleResolutions) return files;
  const paths = new Set(Object.values(context.bundleResolutions).flatMap(edges => Object.values(edges.data)));
  return Object.fromEntries([...paths].filter(path => files[path]).map(path => [path, files[path]]));
}

/** The same compilation input must reach both main-thread and worker engines. */
export function engineBundle(context: BundleContext) {
  if (!context.bundleSources && !context.bundleFiles && !context.bundleResolutions) return undefined;
  return {
    sources: context.bundleSources || {},
    files: dataFiles(context),
    resolutions: context.bundleResolutions || {},
  };
}

/** VS Code transports bytes as JSON arrays. Convert once when adopting a message. */
export function bundleFilesFromMessage(files: Record<string, Uint8Array | number[]> | null | undefined): BundleFiles | null {
  if (!files) return null;
  if (Object.values(files).every(bytes => bytes instanceof Uint8Array)) return files as BundleFiles;
  return Object.fromEntries(Object.entries(files).map(([path, bytes]) => [path, Uint8Array.from(bytes)]));
}

/** Math's JSON boundary uses arrays, while the engine and workers use typed bytes. */
export function mathBundleFields(context: BundleContext) {
  return {
    ...(context.bundleFiles ? { files: Object.fromEntries(Object.entries(dataFiles(context))
      .map(([path, bytes]) => [path, Array.from(bytes)])) } : {}),
    ...(context.bundleResolutions ? { resolutions: context.bundleResolutions } : {}),
  };
}
