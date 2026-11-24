---
name: hale-publish
description: Update and rebuild this repository's public-facing material — the technical report (technical_report/), the GitHub Pages project page (project-page/), the docs (docs/), the Mermaid architecture diagrams and the blog (docs/blog/). Use after changing the architecture or results, or when asked to write up, publish, or refresh the technical report, page, docs or blog.
---

# Publishing: technical report, project page, docs, blog

| Material | Location | Source of truth |
|---|---|---|
| technical report | `technical_report/main.tex`, `technical_report/references.bib` | code + recorded experiments |
| Project page | `project-page/index.html` | mirrors the technical report abstract, diagrams and BibTeX |
| Docs | `docs/*.md` | code |
| Architecture diagrams | `docs/architecture.md` (Mermaid) | code |
| Blog | `docs/blog/*.md` | narrative version of the technical report |

## Rules

- **Never invent numbers.** Tables in the technical report hold only results you actually ran and
  can reproduce with a command in the repo. Unrun experiments stay marked `TBD` with a
  `% TODO` comment.
- When the architecture changes, update in this order: code → `docs/architecture.md`
  → technical report method section and figure → project page diagram → blog if the story changed.
- Keep the Mermaid source identical between `docs/architecture.md` and
  `project-page/index.html` (the page renders the same diagrams with mermaid@11).
- Mermaid labels containing brackets, parentheses or `<tokens>` must be quoted:
  `A["&lt;image&gt; tokens (196)"]`.

## Build the technical report

```bash
cd technical_report && latexmk -pdf main.tex      # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

For arXiv submission upload `main.tex`, `main.bbl` and `references.bib` (arXiv does not
run BibTeX reliably, so include the `.bbl`). `latexmk -c` cleans aux files.

## Project page

`project-page/` is static HTML (no build step). Preview with
`python -m http.server -d project-page 8000`.

`.github/workflows/pages.yml` compiles the technical report and publishes `project-page/` plus
`technical_report.pdf` to GitHub Pages on every push to `masked_diffusion_llm` that touches `technical_report/` or
`project-page/`. One-time setup: *Settings → Pages → Source: GitHub Actions*.
The page is then served at `https://basaanithanaveenkumar.github.io/Hale-LLM/`.

## Blog

One Markdown file per post in `docs/blog/`, named `YYYY-MM-DD-slug.md`, with a title,
date and a one-paragraph summary at the top. Link new posts from `docs/blog/README.md`.

For local preview of the GIF: `mkdir -p project-page/static && cp assets/block_diffusion_step4900.gif project-page/static/` (git-ignored; the Pages workflow copies it on deploy). The blog index here is `docs/blog.md`.
