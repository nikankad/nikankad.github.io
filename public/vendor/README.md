# Vendored assets

## dot-globe.js

Framework-free build of [dot-globe](https://www.npmjs.com/package/dot-globe)'s
embed API, loaded lazily by `src/components/HeroGlobe.astro` (exposes
`window.DotGlobe.create(options)`). Vendored so the site ships no React runtime.

To regenerate after a dot-globe upgrade:

```bash
npm i -D dot-globe@<version>
# take node_modules/dot-globe/dist/embed.global.js, remove the trailing
# auto-init block (the `const init = () => { ... }` + readyState listener that
# calls createDotGlobe(parseUrlParams()) on load), then minify:
npx esbuild embed.global.trimmed.js --minify --outfile=public/vendor/dot-globe.js
npm rm dot-globe
```

Then bump the `?v=` query in `HeroGlobe.astro` to bust caches.
