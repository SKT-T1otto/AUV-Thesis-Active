"""Synthetic report wiring only: mocked I/O, no performance results or envs."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from tools import r123_tracking_experiment as tool

source=tool.identity()
base=Path('synthetic-unit-data')
plan=base/'plan'
results=base/'results'
ids=[f'unit_scene_{i}' for i in range(6)]
data={}
rows={}
for arm in ('baseline',*tool.NEW_ARMS):
    rows[arm]=[dict(scenario_id=i,**{key:0 for key in tool.PAIRED}) for i in ids]
for arm,indices in [('path_only',[0]),('feedforward_only',[1]),('both',[0,1,2])]:
    for i in indices:rows[arm][i]['success']=1
data[str(plan/'baseline.json')]=dict(rows=rows['baseline'])
data[str(plan/'development.json')]=dict(scenarios=[dict(scenario_id=i) for i in ids])
data[str(plan/'config.json')]={}
data[str(plan/'options.json')]={}
for arm in tool.NEW_ARMS:
    data[str(results/arm/'completed.json')]=dict(source=source,variant=arm,seed=982001,
        switches=dict(zip(('path','feedforward'),tool.ARMS[arm])),config_hash=tool.digest({}),
        options_hash=tool.digest({}),scenario_ids=ids,
        episodes_sha256='synthetic_hash',metrics_sha256='synthetic_hash',summary_sha256='synthetic_hash')
captured=[]


def read_text(path,*a,**kw):
    arm=path.parent.name
    if path.name!='metrics.jsonl' or arm not in tool.NEW_ARMS:raise AssertionError('unexpected read: '+str(path))
    return '\n'.join(json.dumps(r) for r in rows[arm])


args=NS(root=str(plan),results=str(results),output=str(base/'output'))
with patch.object(tool,'frozen',return_value=plan),patch.object(tool,'identity',return_value=source),\
     patch.object(tool,'read',side_effect=lambda p:data[str(p)]),\
     patch.object(tool,'sha',return_value='synthetic_hash'),\
     patch.object(tool,'new_output',side_effect=lambda p:Path(p)),\
     patch.object(tool,'write',side_effect=lambda p,v:captured.append(v)),\
     patch.object(Path,'read_text',read_text),patch.object(Path,'mkdir'):
    result=tool.report(args)
    assert result['complete_jobs']==3 and result['baseline_reused']
    assert result['paired_differences']['path_only']['success']['per_scene']==[1,0,0,0,0,0]
    assert result['factorial_interaction_per_scene']['success']==[0,0,1,0,0,0]
    data[str(results/'both'/'completed.json')]['switches']['path']=False
    try:tool.report(args)
    except ValueError as e:assert 'switches' in str(e)
    else:raise AssertionError('wrong arm switches accepted')
    assert len(captured)==1

receipt=dict(synthetic_report_passed=True,factorial_interaction_checked=True,
             incorrect_switches_rejected=True,filesystem_output_mocked=True,
             environment_initializations=0,training_updates=0,performance_claim=False)
(Path(__file__).parent/'report_wiring.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt))
