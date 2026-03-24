# Dmitry Grinko's Blog

A personal blog documenting a learning journey from traditional software development into machine learning engineering and computational biology.

Built with Angular 19, deployed to GitHub Pages.

## Content

Posts are organized as a structured study path:

1. Mathematics — Linear Algebra, Probability & Statistics
2. Machine Learning — Fundamentals, Deep Learning, Optimization, Neural Architectures
3. Sequence Modeling — Transformers, Attention, Embeddings
4. Molecular Biology — Genetics, DNA Structure, Mutations & Variants
5. Genomics — Gene Regulation, Gene Expression
6. Bioinformatics — DNA Sequencing, Sequence Alignment, Variant Calling, Pipelines
7. DNA Machine Learning — Sequence Analysis, Variant Effect Prediction, Regulatory Modeling

## Tech Stack

- Angular 19 (standalone components, lazy-loaded routes)
- Markdown rendering via `marked`
- Syntax highlighting via `highlight.js`
- SCSS styling with dark/light theme support
- Deployed to GitHub Pages (`docs/` output)

## Adding a New Post

1. Create a markdown file at `src/data/posts/{category-slug}/{subcategory-slug}/{post-slug}.md`

   Example: `src/data/posts/mathematics/linear-algebra/my-new-post.md`

2. Register the post in `src/app/services/blog.service.ts` by adding an entry to the relevant subcategory's `posts` array inside `predefinedCategories`:

   ```ts
   {
     title: 'My New Post',
     slug: 'my-new-post',
     excerpt: 'A short description shown in post listings.'
   }
   ```

   The `slug` must match the filename (without `.md`) and the folder path must match the category and subcategory slugs.

3. Run `npm start` to verify it renders correctly, then `npm run deploy` to publish.

## Development

```bash
npm start        # dev server at http://localhost:4200
npm run build    # production build → docs/
npm run deploy   # build + push to GitHub Pages
npm test         # run unit tests
```
