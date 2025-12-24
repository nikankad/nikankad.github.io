import { defineConfig } from 'astro/config';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export default defineConfig({
  output: "static",
  prefetch: true,
  compressHTML: true,
  site: 'https://nikankad.github.io',
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
  vite: {
    ssr: {
      external: ['rehype-katex']
    }
  }
});