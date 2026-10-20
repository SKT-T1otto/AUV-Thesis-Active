"""Document-only algebra, source recording and PDF layout QA. No simulator calls."""
from pathlib import Path
import sys,json,hashlib,re,importlib.util
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from pypdf import PdfReader
import pypdfium2 as pdfium

sys.stdout.reconfigure(encoding='utf-8')
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
TMP=ROOT/'tmp/pdfs/ch3_rl_124_20261018'
PDF=HERE/'第三章强化学习方案124_原理实现与实验设计_20261018.pdf'
reader=PdfReader(str(PDF))
texts=[p.extract_text() for p in reader.pages]
assert len(texts)==28
assert all(len(t)>300 for t in texts)
assert not any('\ufffd' in t for t in texts)
assert 'Model-based Action-Gradient-Estimator' in texts[26]
eq_ids=re.findall(r'\((\d+)\)', '\n'.join(texts))
assert set(map(str,range(1,36))).issubset(eq_ids)
all_text='\n\n'.join(f'PAGE {i+1}\n{t}' for i,t in enumerate(texts))
(TMP/'extracted_text.txt').write_text(all_text,encoding='utf-8')

g=.95
rs=np.array([0.,0.,1.,0.,0.,2.])
full=sum(g**k*r for k,r in enumerate(rs))
tail=sum(g**k*r for k,r in enumerate(rs[3:]))
stopped=sum(g**k*r for k,r in enumerate(rs[:3]))+g**3*tail
c=np.array([1.,-1.,-1.,1.])
pair_metric=np.eye(2)+np.outer([1,-1],[1,-1])
four_metric=np.eye(4)+np.outer(c,c)
clip=lambda x:float(np.clip(x,-10,10))
checks={
 'scope':'Algebraic and document QA only; no training or environment experiment',
 'return_full':full,'return_event':stopped,'return_abs_error':abs(full-stopped),
 'pair_eigenvalues':np.linalg.eigvalsh(pair_metric).tolist(),
 'interaction_eigenvalues':np.linalg.eigvalsh(four_metric).tolist(),
 'interactions':{'additive':float(c@np.array([0,1,1,2])), 'AND':float(c@np.array([0,0,0,1])), 'OR':float(c@np.array([0,1,1,1]))},
 'unequal_horizon_common_error':g**2-g**10,
 'clip_recursive':clip(-2+g*clip(3+g*10)),
 'clip_once':clip(-2+g*3+g*g*10),
 'tail_weights':{str(h):g**h for h in [5,10,20]},
 'A1_max_branch_steps':32*3*2*2*10,'A2_max_branch_steps':24*2*4*10,
}
def integrate(actions):
    x=v=0.
    first=None
    for k,u in enumerate(actions,1):
        v += .2*float(u)*.2
        x += v*.2
        if x>=1 and first is None: first=k
    return {'final_x':x,'first_threshold_step':first}
checks['one_step_intervention']=integrate([1]+[0]*19)
checks['repeated_intervention']=integrate([1]*20)
assert np.isclose(checks['one_step_intervention']['final_x'],.16)
assert np.isclose(checks['repeated_intervention']['final_x'],1.68)
assert checks['repeated_intervention']['first_threshold_step']==16
assert abs(full-stopped)<1e-12
assert np.allclose(checks['interaction_eigenvalues'],[1,1,1,5])
assert np.isclose(checks['clip_recursive'],7.5) and np.isclose(checks['clip_once'],9.875)
files=[
 'AGENTS.md','core/env/uav_env.py','core/env/mission_env.py','core/algorithms/maddpg.py',
 'core/runtime/engine.py','chapter3_bser/experiments/baselines/common/model.py',
 'chapter3_bser/experiments/baselines/common/runtime.py','chapter3_bser/experiments/hgr/runtime.py',
 '文献/第三章文献/RL创新候选交叉核验_20261018/manifest.json',
 '文献/第三章文献/RL创新候选交叉核验_20261018/候选方案与交叉核验.md',
 '文献/第三章文献/RL创新候选交叉核验_20261018/选定124后的研究推进.md',
]
checks['read_source_hashes']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}
checks['pdf_sha256']=hashlib.sha256(PDF.read_bytes()).hexdigest()
checks['pdf_pages']=len(reader.pages)
checks['characters_per_page']=[len(t) for t in texts]
checks['links_per_page']=[len(p.get('/Annots',[])) for p in reader.pages]
checks['visual_review']='Pending manual inspection of rendered pages'
(HERE/'制作与公式核查.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')

doc=pdfium.PdfDocument(str(PDF))
for i in range(len(doc)):
    pg=doc[i]
    img=pg.render(scale=1.5).to_pil().convert('RGB')
    img.save(TMP/f'page_{i+1:02d}.png')
    pg.close()
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
for start in range(0,len(doc),4):
    thumb_w=595;thumb_h=842
    sheet=Image.new('RGB',(2*thumb_w+36,2*(thumb_h+35)+24),'#dfe5eb')
    draw=ImageDraw.Draw(sheet)
    for local,idx in enumerate(range(start,min(start+4,len(doc)))):
        img=Image.open(TMP/f'page_{idx+1:02d}.png')
        img.thumbnail((thumb_w,thumb_h))
        x=12+(local%2)*(thumb_w+12);y=12+(local//2)*(thumb_h+35)
        sheet.paste(img,(x,y));draw.text((x+5,y+thumb_h+3),f'Page {idx+1}',fill='#14324a',font=font)
    sheet.save(TMP/f'contact_{start//4+1:02d}.png')
print(json.dumps({'pages':len(doc),'characters':sum(len(t) for t in texts),'formula_checks':'passed','render_directory':str(TMP),'pdf_sha256':checks['pdf_sha256']},ensure_ascii=False))
