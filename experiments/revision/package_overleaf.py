"""Assemble the active manuscript, figure code/data and audit records for upload."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT/'paper'
DEST = ROOT/'deliverables/HyperbolicMomentum_Overleaf'
ARCHIVE = DEST.with_suffix('.zip')


def main():
    if DEST.exists():
        # Only remove the package staging directory created by this script.
        assert (DEST/'README_OVERLEAF.txt').exists()
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)
    visited = set(); assets = set()

    def visit(path):
        if path in visited:
            return
        visited.add(path)
        text = re.sub(r'(?<!\\)%[^\n]*', '', path.read_text())
        for name in re.findall(r'\\(?:input|include)\{([^}]+)\}', text):
            child = PAPER/name
            if not child.suffix:
                child = child.with_suffix('.tex')
            assert child.exists(), child
            visit(child)
        for name in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', text):
            child = PAPER/name
            assert child.exists(), child
            assets.add(child)

    def copy(source, relative):
        target = DEST/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    visit(PAPER/'main.tex')
    assert len(assets)==3
    for source in visited | assets:
        copy(source, source.relative_to(PAPER))
    for source in assets:
        copy(source.with_suffix('.png'), source.with_suffix('.png').relative_to(PAPER))
    for name in ['main.pdf','main.bbl','references.bib','icml2026.bst','icml2026.sty',
                 'icml2027_draft.sty','algorithm.sty','algorithmic.sty','fancyhdr.sty',
                 'TEMPLATE_PROVENANCE.txt']:
        copy(PAPER/name, name)
    for pattern in ['tables/*.csv','tables/*.json','evidence/*','figure_data/*']:
        for source in PAPER.glob(pattern):
            if source.is_file():
                copy(source, source.relative_to(PAPER))
    for name in ['reproduce_figures.py','reproduce_positive_tables.py','requirements-figures.txt']:
        copy(PAPER/'scripts'/name, Path('scripts')/name)
    for name in ['manifest.json','execution_scope.json']:
        copy(ROOT/'experiments/revision'/name, Path('experiments/revision')/name)
    copy(ROOT/'docs/historical/G4_MECHANISM_EVIDENCE.md', 'experiments/G4_MECHANISM_EVIDENCE.md')
    for name in ['final_completion_audit.json','overleaf_figure_validation.json','overleaf_manuscript_validation.json']:
        if (ROOT/'analysis/revision'/name).exists():
            copy(ROOT/'analysis/revision'/name, Path('evidence')/name)
    (DEST/'latexmkrc').write_text('$pdf_mode = 1;\n$bibtex_use = 2;\n')
    (DEST/'README_OVERLEAF.txt').write_text('''HyperbolicMomentum -- Overleaf upload package
============================================

使用方式
1. 将此 ZIP 作为新项目上传到 Overleaf；ZIP 根目录即项目根目录。
2. 主文件选择 main.tex，编译器选择 pdfLaTeX。
3. 编译。所有正文、完整理论附录、实验表、图片和参考文献均已包含。
   LaTeX 直接使用 figures/ 中已生成的 PDF，不需要运行 Python 或连接 compute worker。
4. main.pdf 是与本包源码对应的本地完整编译版本，便于对照。

模板说明
本包使用本地 ICML 2027 草稿封装，基于未修改的官方 ICML 2026 样式。
它不是官方 ICML 2027 模板。来源与上游哈希见 TEMPLATE_PROVENANCE.txt。

绘图代码与数据
scripts/reproduce_figures.py 覆盖 main.tex 引用的全部 3 张图。
figure_data/ 含原始记录导出的绘图数据以及来源 SHA-256；不做平滑或插值。
代码与数据均使用包内相对路径，不依赖桌面目录、服务器、GPU 或训练数据。
测试环境：Python 3.9、matplotlib 3.5.2、numpy 1.23.5。

在解压后的项目根目录运行：
    python -m pip install -r scripts/requirements-figures.txt
    python scripts/reproduce_figures.py

会更新 figures/fig_loss_historical_kg.{pdf,png}
       figures/fig_loss_revision_cora.{pdf,png}
       figures/fig_loss_smooth_drift.{pdf,png}
上述脚本已用随包数据验证，生成 PNG 与论文现有图逐像素一致。

表格与证据
正文新增两张正面比较表，可使用标准库脚本重新生成：
    python scripts/reproduce_positive_tables.py
其输入是 evidence/positive_contrasts.json，包含完整的 12 个节点分类设置，
所展示的 5 个正面终点比较及其验证选择对照，以及 4 个指定基线的局部正面比较。
各项比较的基线、指标、选择方式与适用范围均在正文和表注中列明。
全部非正面和持平对照继续保留在论文附录中。

方法示意和 related work 按现稿以 LaTeX 表格表示，可直接编辑：
    tables/presentation_method.tex
    sections/related_work_table.tex
其他表格的可编辑 LaTeX 源码均在 tables/ 中。

evidence/completed_results.jsonl 汇集 705 个完成配置的原始 result.json 内容。
evidence/final_completion_audit.json 保留完整轮数核验和原始结果/轨迹哈希。
experiments/revision/manifest.json 保留原始 893 个注册配置；execution_scope.json
记录 705 个完成配置与 188 个按预算暂缓配置的范围。没有重新训练或改变评价规则。
包中不包含大型训练数据集、模型检查点和完整训练轨迹；复现本文绘图无需这些文件。

本地完整编译（已安装 TeX Live 和 latexmk 时）：
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
参考文献使用 BibTeX，样式 icml2026.bst 已包含。

English quick start
Upload the ZIP as a new Overleaf project. Set main.tex as the main document and
pdfLaTeX as the compiler. No Python execution is needed to compile the paper.
The portable scripts and bundled CSV/JSON files reproduce every included figure
and the two added positive-contrast tables. Negative and tied outcomes remain
in the full appendix. The target-year wrapper is a local draft, not an official
ICML 2027 release. File hashes are listed in PACKAGE_MANIFEST.json.
''', encoding='utf-8')
    files = sorted(p for p in DEST.rglob('*') if p.is_file())
    file_manifest = {str(p.relative_to(DEST)):{'bytes':p.stat().st_size,
                     'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
    (DEST/'PACKAGE_MANIFEST.json').write_text(json.dumps(file_manifest,indent=2)+'\n')
    with zipfile.ZipFile(ARCHIVE, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(DEST.rglob('*')):
            if p.is_file():
                z.write(p, str(p.relative_to(DEST)))
    print(json.dumps({'archive':str(ARCHIVE),'files':len(files)+1,
                      'bytes':ARCHIVE.stat().st_size,'active_tex':len(visited),
                      'included_figures':len(assets)},indent=2))


if __name__ == '__main__':
    main()
