# nikankad.github.io

Personal site at [nikankad.github.io](https://nikankad.github.io).

## Stack

Astro 5, deployed via Cloudflare Pages (`wrangler`).

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
| `bookmarks/` | Saved links |
| `notes/` | Personal notes |
| `site/config.json` | Name, title, social links |

## Config

- **Theme**: dark, hardcoded in `BaseLayout.astro`
- **Fonts**: Inter, Roboto Mono, Source Sans Pro (via Fontsource)
- **Math**: KaTeX via remark-math/rehype-katex
- **Favicon**: `public/images/favicon.ico`
