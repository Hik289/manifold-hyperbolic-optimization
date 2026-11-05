"""WWW reference table styling; highlighting never changes a recorded value."""
import re
from pathlib import Path

SCORES={'E','B3','B35','B5','A5','Shared','B4','Best fixed','Best A5','Matched A5'}


def number(cell):
    try:return float(cell.strip().replace(r'\%',''))
    except ValueError:return None


def style_text(text,lower=False):
    if '% WWW_REFERENCE_STYLE' in text:return text
    def block(match):
        part=match.group(0);lines=part.splitlines();header=None
        for line in lines:
            if '&' in line:
                cells=[x.strip() for x in (line[:-2] if line.endswith(r'\\') else line).split('&')]
                if 'B5' in cells and not any(number(x) is not None for x in cells):header=cells;break
        indices=[i for i,c in enumerate(header or []) if c in SCORES]
        is_cora='tab:revision-cora' in part
        cora_rows=[]
        if is_cora:
            for line in lines:
                cells=[x.strip() for x in (line[:-2] if line.endswith(r'\\') else line).split('&')]
                if len(cells)==5 and cells[0] in ['E','B2','B3','B4','B5','A5','B35','shared']:
                    cora_rows.append(cells)
        maxima={i:max(number(r[i]) for r in cora_rows) for i in [1,2,3]} if cora_rows else {}
        out=[]
        for line in lines:
            if '&' in line and line.rstrip().endswith(r'\\'):
                cells=[x.strip() for x in line.rstrip()[:-2].split('&')]
                if header and len(cells)==len(header) and any(number(cells[i]) is not None for i in indices):
                    numeric={i:number(cells[i]) for i in indices if number(cells[i]) is not None}
                    best=(min if lower else max)(numeric.values())
                    for i,v in numeric.items():
                        if v==best:cells[i]=r'\textbf{'+cells[i]+'}'
                        if header[i]=='B5':cells[i]=r'\cellcolor{oursbg}'+cells[i]
                    line=' & '.join(cells)+r' \\'
                elif is_cora and len(cells)==5 and cells[0] in ['E','B2','B3','B4','B5','A5','B35','shared']:
                    for i,best in maxima.items():
                        if number(cells[i])==best:cells[i]=r'\textbf{'+cells[i]+'}'
                    line=(r'\rowcolor{oursbg}'+'\n' if cells[0]=='B5' else '')+' & '.join(cells)+r' \\'
            out.append(line)
        part='\n'.join(out)
        # Wider comparison tables span both ICML columns, keeping text readable.
        if header and len(header)>5:part=part.replace(r'\begin{table}',r'\begin{table*}').replace(r'\end{table}',r'\end{table*}')
        part=part.replace(r'\centering\small',r'\centering').replace(r'\centering\scriptsize',r'\centering')
        part=part.replace(r'\centering',r'\centering'+'\n'+r'\footnotesize\setlength{\tabcolsep}{4.2pt}\renewcommand{\arraystretch}{1.12}',1)
        part=part.replace(r'\begin{tabular}',r'\begin{adjustbox}{max width=\linewidth}'+'\n'+r'\begin{tabular}')
        part=part.replace(r'\end{tabular}',r'\end{tabular}'+'\n'+r'\end{adjustbox}')
        return part
    text=re.sub(r'\\begin\{table\*?\}.*?\\end\{table\*?\}',block,text,flags=re.S)
    # The full run ledger is paginated; it must not be put in an unbreakable box.
    if r'\begin{longtable}' in text:
        text=text.replace(r'\scriptsize',r'\scriptsize\setlength{\tabcolsep}{4.2pt}\renewcommand{\arraystretch}{1.12}',1)
        rows=[]
        for line in text.splitlines():
            cells=[c.strip() for c in line.split('&')]
            if len(cells)==7 and cells[2]=='B5':line=r'\rowcolor{oursbg}'+'\n'+line
            rows.append(line)
        text='\n'.join(rows)+'\n'
    return '% WWW_REFERENCE_STYLE\n'+text


def style_directory(folder):
    for p in Path(folder).glob('revision_*.tex'):
        p.write_text(style_text(p.read_text(),lower='mechanism' in p.name))
