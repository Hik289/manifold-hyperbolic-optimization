"""Export the native LaTeX comparison table from the compiled manuscript."""
from pathlib import Path
import fitz
root=Path(__file__).resolve().parents[1]
doc=fitz.open(root/'paper/.latex-build/main/pdflatex/main.pdf')
for page in doc:
    hits=page.search_for('Mechanisms in related work.')
    if not hits:continue
    begin=hits[0].y0-5
    ends=page.search_for('The map being compared.')
    if not ends:raise RuntimeError('Table boundary not found; inspect the PDF before exporting')
    end=ends[0].y0-8
    textblocks=[b for b in page.get_text('blocks') if b[1]>=begin-3 and b[3]<=end+3]
    left=min(b[0] for b in textblocks)-4;right=max(b[2] for b in textblocks)+4
    clip=fitz.Rect(left,begin,right,end)
    page.get_pixmap(matrix=fitz.Matrix(4.5,4.5),clip=clip).save(root/'paper/figures/related_work_comparison.png')
    print('Exported native related-work table from page',page.number+1)
    break
else:raise RuntimeError('Comparison table not found')
