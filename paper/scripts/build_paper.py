"""Compile the complete local paper with a standard TeX Live/latexmk installation."""
from pathlib import Path
import shutil
import subprocess


def main():
    paper = Path(__file__).resolve().parents[1]
    if not shutil.which('latexmk'):
        raise SystemExit('Install TeX Live with latexmk, or upload the paper folder to Overleaf.')
    subprocess.run(['latexmk', '-pdf', '-bibtex', '-interaction=nonstopmode',
                    '-halt-on-error', '-outdir=.', '-auxdir=.', 'main.tex'],
                   cwd=paper, check=True)
    if not (paper/'main.pdf').exists() or not (paper/'main.bbl').exists():
        raise RuntimeError('Compilation did not produce both the PDF and bibliography.')
    print('Compiled paper/main.pdf')


if __name__ == '__main__':
    main()
