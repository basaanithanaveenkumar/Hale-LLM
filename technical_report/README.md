# HALE-LLM technical report

LaTeX source of the technical report. `main.tex` is the document, `references.bib` the
bibliography, `figures/` the images it includes.

## Compile

You need a TeX distribution with `latexmk` (TeX Live, MacTeX or MiKTeX).

```bash
# macOS (smaller install than full MacTeX)
brew install --cask basictex && sudo tlmgr install latexmk collection-latexrecommended

make -C technical_report          # builds technical_report/main.pdf
make -C technical_report clean    # removes build artefacts
```

Without `make`:

```bash
cd technical_report
latexmk -pdf -interaction=nonstopmode main.tex
```

## View

```bash
make -C technical_report view     # opens main.pdf (open on macOS, xdg-open on Linux)
latexmk -pdf -pvc main.tex        # rebuild and refresh the viewer on every save
```

The same source is built in CI by `.github/workflows/pages.yml` and published as
`technical_report.pdf` next to the project page.

## Figures

- `figures/block_diffusion_mid.png` is a frame from the block-diffusion visualiser:

  ```bash
  uv run hale-llm-viz --config configs/block_diffusion/small.yaml
  ```

- Mermaid diagrams live in `docs/architecture.md` and render on GitHub.

## Submit to arXiv

```bash
make -C technical_report arxiv    # arxiv.tar.gz with main.tex, main.bbl, references.bib
```
