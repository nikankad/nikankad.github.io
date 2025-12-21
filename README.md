# Personal Portfolio

My personal portfolio website built with Astro and deployed on Cloudflare Pages.

## Quick Start

```bash
# Install dependencies
pnpm install

# Start dev server
pnpm dev

# Build for production
pnpm build

# Preview production build locally
pnpm preview

# Deploy to Cloudflare Pages
pnpm deploy
```

## Tech Stack

- **Astro** - Static site framework
- **Cloudflare Pages** - Hosting & deployment
- **TypeScript** - Type safety
- **pnpm** - Package manager

## Project Structure

```
src/
├── components/     # Reusable UI components
├── content/        # Content collections (blog, projects, experience, etc.)
├── pages/          # Page routes
├── styles/         # Global styles
├── utils/          # Helper functions
└── types/          # TypeScript type definitions
```

## Content

Edit content in `src/content/`:
- `blog/` - Blog posts
- `projects/` - Project showcase
- `experience/` - Work experience
- `bookmarks/` - Saved bookmarks
- `notes/` - Personal notes
- `site/config.json` - Site configuration

## Notes to Self

- Favicon is at `/public/images/favicon.ico`
- Site config lives in `src/content/site/config.json`
- Dark theme is hardcoded (check BaseLayout.astro)
- Using Inter, Roboto Mono, and Source Sans Pro fonts

## Deployment

Automatically deploys to Cloudflare Pages on push. Manual deploy with `pnpm deploy`.
