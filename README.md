# nikankad.github.io

Personal site at [nikankad.github.io](https://nikankad.github.io).

## Stack

Astro 5, deployed via Cloudflare Pages (`wrangler`).

## Design

Visual system inspired by [dither-kit](https://www.tripwire.sh/dither-kit) — ordered-dither textures, Geist Pixel display type, Tripwire dark palette (`#0d0d0f` / `#34a6ff`).

## Develop

```bash
pnpm install
pnpm dev
```

## Build

```bash
pnpm build      # astro build → dist/
pnpm preview    # build + serve locally via wrangler
pnpm deploy     # build + push to Cloudflare
```

## Content

All editable content in `src/content/`:

| Directory | Purpose |
|-----------|---------|
| `blog/` | Blog posts (markdown) |
| `projects/` | Project entries |
| `experience/` | Work history |
| `site/config.json` | Name, title, social links |

## Config

- **Theme**: dither-kit dark tokens in `src/styles/theme.css`
- **Fonts**: Geist, Geist Mono, Geist Pixel
- **Math**: KaTeX via remark-math/rehype-katex
- **Favicon**: `public/images/favicon.ico`
