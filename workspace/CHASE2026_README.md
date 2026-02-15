# IEEE/ACM CHASE 2026 paper (IEEE format)

CHASE 2026 requires **IEEE conference paper format (double column) and double-blind reviewing** for regular + short papers. The official CHASE 2026 submission page states page limits of **12 pages (regular)** and **5 pages (short)**, **including references and appendices**.

This folder contains a ready-to-compile IEEE conference LaTeX paper:
- `CHASE2026_anonymous.tex` (double-blind by default)
- `CHASE2026_refs.bib`

## How to compile

From the repo root, run:

```bash
cd workspace
pdflatex CHASE2026_anonymous.tex
bibtex CHASE2026_anonymous
pdflatex CHASE2026_anonymous.tex
pdflatex CHASE2026_anonymous.tex
```

If you compile inside Overleaf, upload both files and ensure the main file is `CHASE2026_anonymous.tex`.

## Double-blind vs camera-ready

In `CHASE2026_anonymous.tex`, there is a toggle near the top:

```tex
\newif\ifreview
\reviewtrue % set \reviewfalse for camera-ready
```

- For **review submission**, keep `\reviewtrue` and do **not** add identifying information (names, affiliations, links to your repo, acknowledgments, etc.).
- For **camera-ready**, set `\reviewfalse` and fill the author block (and acknowledgments if desired).

## Optional: include existing result figures

Your repo already contains figures you can include, e.g.:
- `workspace/orkspace/visualizations/test_confusion_matrix.png`
- `workspace/orkspace/visualizations/test_per_class_f1.png`
- `workspace/orkspace/visualizations/training_curves.png`

If you want them embedded in the PDF, the simplest approach is to copy them into the same folder as the `.tex` file (or update `\includegraphics{...}` paths accordingly).

