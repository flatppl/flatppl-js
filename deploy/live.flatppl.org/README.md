# Public deploy: live.flatppl.org

Deployment-specific content for the public FlatPPL live demo, kept out
of the `@flatppl/web` package because it belongs to this deployment,
not to the gallery shell:

- `site/` — the site overlay: the legal notice page and, derived from
  it, the gallery's footer links. Rendered by the web package's build
  when `FLATPPL_SITE_DIR` points here, which the Pages workflow
  (`.github/workflows/pages.yml`) does. Format: the header comment of
  `packages/web/build-site.mjs`. Local review of this deploy, from
  `packages/web/`:

  ```sh
  FLATPPL_SITE_DIR=../../deploy/live.flatppl.org/site npm run dev
  ```
