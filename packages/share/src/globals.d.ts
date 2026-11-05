// Share-link dictionaries, embedded by esbuild's binary loader.
declare module '*.zdict' {
  const bytes: Uint8Array;
  export default bytes;
}
