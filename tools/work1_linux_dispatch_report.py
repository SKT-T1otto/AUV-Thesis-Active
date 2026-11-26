"""Score precommitted assignments after their paired native physics logs are closed."""
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np

from tools.work1_linux_dispatch_maps import MODELS, H


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def facts(folder):
    with gzip.open(folder/'physics.jsonl.gz','rt',encoding='utf8') as file:
        rows=[json.loads(line) for line in file]
    if [r['step'] for r in rows]!=list(range(1,len(rows)+1)) or len(rows)!=read(folder/'complete.json')['steps']:
        raise ValueError(f'Incomplete physical logs: {folder}')
    return rows


def ending(rows):
    return dict(steps=len(rows),first_discovery=next((r['step'] for r in rows if r['found'] and not r['found_before']),None),
                termination=rows[-1]['reason'] if rows[-1]['terminal'] else 'external_cap')


def window(rows, ids, expected, h):
    valid=[]
    for r in rows[:h]:
        if any(r['commands'][i]!=expected[i] for i in ids):
            return valid, 'new_or_unissued_command'
        if not r['checked']:
            return valid, 'frozen' if r['found_before'] else r['reason']
        valid.append(r)
        if len(valid)==h:
            return valid,'complete'
        if r['found'] or r['terminal']:
            return valid,'discovery' if r['found'] else r['reason']
    return valid,'external_cap'


def observe(rows, decision, plan_id, h):
    plan=decision['plans'][plan_id]
    offers=decision['offers']
    expected=[None]*3
    details=[]
    for token in plan:
        item=offers[token]
        i=item['boat']
        expected[i]=[token,item['goal']]
        valid,stop=window(rows,[i],expected,h)
        first=next((r['step'] for r in valid if item['cell'] in r['hits'][i]),None)
        issued=rows[0]['commands'][i]==expected[i]
        details.append(dict(boat=i,cell=item['cell'],first=first,valid=len(valid),stop=stop,
                            issued=issued,accepted=issued and rows[0]['receipts'][i]['accepted'],
                            hit=True if first is not None else False if len(valid)==h else None))
    valid,stop=window(rows,range(3),expected,h)
    b=np.array(decision['belief'])
    mask=np.zeros((3,len(b)),bool)
    for r in valid:
        for i,cells in enumerate(r['hits']):
            mask[i,cells]=True
    individual=mask@b
    union=float(b[mask.any(axis=0)].sum())
    return dict(h=h,valid=len(valid),complete=len(valid)==h,stop=stop,boats=details,
                observed_boat_mass=individual.tolist(),observed_union=union,
                observed_overlap=float(individual.sum()-union),unique_cells=int(mask.any(axis=0).sum()))


def sign(lo,hi):
    if lo>1e-12:
        return 'positive'
    if hi< -1e-12:
        return 'negative'
    return 'zero' if max(abs(lo),abs(hi))<=1e-12 else 'unknown'


def paired_metrics(d,j,base,other):
    delta=None
    if base['complete'] and other['complete']:
        delta=dict(union=other['observed_union']-base['observed_union'],
                   boats=(np.array(other['observed_boat_mass'])-base['observed_boat_mass']).tolist(),
                   overlap=other['observed_overlap']-base['observed_overlap'])
        assert abs(sum(delta['boats'])-delta['overlap']-delta['union'])<1e-10
    predicted={}
    for m in MODELS:
        a,b=(d['scores'][m][k] for k in (0,j))
        if a is None or b is None:
            continue
        lo=b['coverage'][0]-a['coverage'][1]
        hi=b['coverage'][1]-a['coverage'][0]
        direction=sign(lo,hi)
        exact=np.allclose(a['boats'][0],a['boats'][1],rtol=0,atol=1e-12) and np.allclose(
            b['boats'][0],b['boats'][1],rtol=0,atol=1e-12) and hi-lo<1e-12
        boats=(np.array(b['boats'][0])-a['boats'][0]).tolist() if exact else None
        predicted[m]=dict(interval=[lo,hi],direction=direction,boats=boats,
            overlap=sum(boats)-lo if exact else None,
            correct=direction==sign(delta['union'],delta['union']) if direction!='unknown' and delta else None,
            coverage_policy_chose_this=d['choices'][m]['coverage']==j,
            region_policy_chose_this=d['choices'][m]['region']==j)
    exchanges=[]
    for cell in d['eligibility']:
        a={r['boat']:r for r in base['boats'] if r['cell']==int(cell)}
        b={r['boat']:r for r in other['boats'] if r['cell']==int(cell)}
        for i in sorted(a.keys()-b.keys()):
            for k in sorted(b.keys()-a.keys()):
                left,right=a[i],b[k]
                accepted=left['accepted'] and right['accepted']
                if not accepted:
                    status='admission_unknown'
                elif left['hit'] and right['hit']:
                    status='both_swept'
                elif left['hit'] is False or right['hit'] is False:
                    status='at_least_one_observed_miss'
                else:
                    status='censored'
                exchanges.append(dict(cell=int(cell),boats=[i,k],first=[left['first'],right['first']],status=status))
    return dict(observed_delta=delta,predicted=predicted,exchanges=exchanges)


def audit_family(folder):
    space=folder/'spatial'
    d=read(space/'prediction.json')
    ref=facts(space)
    windows=(4,8,16,24,32)
    base={h:observe(ref,d,0,h) for h in windows}
    exposures=list(d['eligibility'].values())
    info=dict(family=folder.name,gate=d['gate'],tasks=d['eligibility'],plans=d['plans'],endings=[ending(ref)],
              choices=d['choices'],reason=d['branch_reason'],branch=d['branch'],spatial=base,episodes=1,
              layers={k:sum(bool(e[k]) for e in exposures) for k in ('native_pairs','possible_pairs','comparable_pairs')})
    info['layers']['menu']=sum(len(e['menu'])>=2 for e in exposures)
    if d['branch'] is None:
        return info
    other=folder/'alternative'
    second=facts(other)
    u,v=read(space/'complete.json'),read(other/'complete.json')
    for k in ('seed','layout','decision_sha256'):
        if u[k]!=v[k]:
            raise ValueError(f'Paired {k} mismatch')
    if v['plan']!=d['branch'] or u['plan']!=0:
        raise ValueError('Different assignment executed than registered')
    for k in ('start','velocity_start'):
        if ref[0][k]!=second[0][k]:
            raise ValueError(f'Physical initial {k} differs')
    if ref[0]['commands'][3]!=second[0]['commands'][3]:
        raise ValueError('Initial executor instruction differs')
    for a,b in zip(ref,second):
        if not a['checked'] or not b['checked']:
            break
        if a['target_motion_digest']!=b['target_motion_digest']:
            raise ValueError('Exogenous target realization differs before termination')
    obs={h:observe(second,d,d['branch'],h) for h in windows}
    info['endings'].append(ending(second))
    info.update(episodes=2,alternative=obs,paired=paired_metrics(d,d['branch'],base[H],obs[H]))
    return info


def summarize(out):
    families=[audit_family(folder) for folder in sorted(out.glob('family_*'))]
    if len(families)!=4 or sum(f['episodes'] for f in families)>8:
        raise ValueError('Incomplete roster or excess physical budget')
    paired=[f for f in families if 'paired' in f]
    measured=[f for f in paired if f['paired']['observed_delta'] is not None]
    proven=[f for f in paired if any(x['status']=='both_swept' for x in f['paired']['exchanges'])]
    totals={k:dict(families=sum(f['layers'][k]>0 for f in families),regions=sum(f['layers'][k] for f in families))
            for k in ('menu','native_pairs','possible_pairs','comparable_pairs')}
    tested_regions={(f['family'],x['cell']) for f in paired for x in f['paired']['exchanges']}
    verified={(f['family'],x['cell']) for f in proven for x in f['paired']['exchanges'] if x['status']=='both_swept'}
    direction={}
    for m in MODELS:
        rows=[f['paired']['predicted'][m] for f in measured if m in f['paired']['predicted']]
        direction[m]=dict(scorable=sum(r['correct'] is not None for r in rows),
                          correct=sum(r['correct'] is True for r in rows),
                          policy_pairs=sum(r['coverage_policy_chose_this'] for r in rows),
                          policy_positive_actual=sum(f['paired']['predicted'].get(m,{}).get('coverage_policy_chose_this',False)
                              and f['paired']['observed_delta']['union']>1e-12 for f in measured))
    gates=dict(Counter(f['gate'] for f in families))
    result=dict(families=families,layers=totals,gate_counts=gates,physical_pairs=len(paired),complete_pairs=len(measured),
                tested_regions=len(tested_regions),confirmed_regions=len(verified),confirmed_families=len(proven),
                direction=direction)
    with (out/'results.json').open('x',encoding='utf8') as file:
        json.dump(result,file,ensure_ascii=False,allow_nan=False,indent=2)
    lines=['# 16 步派艇配对实验','',
        f'四个预登记家族中，{totals["menu"]["families"]} 个在菜单中为同一区域提供至少两艘艇，'
        f'{totals["native_pairs"]["families"]} 个在保留其余任务约束时允许换艇，'
        f'{totals["possible_pairs"]["families"]} 个有两艇通过物理时间下界，'
        f'{totals["comparable_pairs"]["families"]} 个同时通过共享路径及分工核验。',
        f'实际配对交换涉及 {len(tested_regions)} 个家族内区域，其中 {len(verified)} 个在两侧均由不同的已接令艇按时扫到。'
        f'完成 {len(paired)} 对执行，其中 {len(measured)} 对有完整的 16 步团队覆盖窗口。','',
        '| 冻结模型 | 预测增益方向正确／方向可判定的完整配对 | 自己推荐换艇的完整配对 | 换后实际覆盖增加 |',
        '|---|---:|---:|---:|']
    for m,title in zip(MODELS,('历史联合误差','运动状态分箱')):
        v=direction[m]
        lines.append(f'| {title} | {v["correct"]}/{v["scorable"]} | {v["policy_pairs"]} | {v["policy_positive_actual"]} |')
    lines+=['','| 家族 | 共享路径 | 分支依据 | 历史预测增益（百分点） | 分箱预测增益（百分点） | 实际增益（百分点） |',
            '|---|---|---|---:|---:|---:|']
    for f in families:
        pair=f.get('paired',{})
        def interval(m):
            pred=pair.get('predicted',{}).get(m)
            if not pred:
                return '—'
            lo,hi=pred['interval']
            return f'{100*lo:+.4f}' if abs(lo-hi)<1e-12 else f'[{100*lo:+.4f}, {100*hi:+.4f}]'
        delta=pair.get('observed_delta')
        actual=f'{100*delta["union"]:+.4f}' if delta is not None else '未知'
        lines.append(f'| {f["family"]} | {f["gate"]} | {f["reason"]} | {interval("history")} | {interval("bins")} | {actual} |')
    lines+=['','物理下界是必要条件。完整窗口中的零增益按持平记账；拒令、换令、提前发现或碰撞保留原始事件与可观察前缀。'
            '诊断换艇与模型主动换艇分别计数。样本单位为家族，本轮仅用于判断改动方向。','',
            'results.json 保存逐艇、重复覆盖、团队覆盖增量与每区首次命中。各回合目录保存发令前预测和逐步执行轨迹。']
    with (out/'REPORT.md').open('x',encoding='utf8') as file:
        file.write('\n'.join(lines)+'\n')
