"""Offline, hash-checked empirical learning reports. Never construct an environment."""
import csv
import html
import json
from pathlib import Path
import numpy as np
from chapter3_bser.reach_avoid.learning_diagnostics import read_journal, settings, assess, sha, create_json


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def member(root,name):
    path=(root/name).resolve()
    if not path.is_relative_to(root):raise ValueError('report input escapes run directory')
    return path


def inputs(root):
    root=Path(root).resolve();base=root/'monitoring';protocol=read(base/'protocol.json');run=read(root/'run.json')
    if protocol['source_identity']!=run['source_identity']:raise ValueError('run/monitor source mismatch')
    journals={};receipts={}
    for name in ('updates','training_episodes','validation','snapshots'):
        journals[name],receipts[name]=read_journal(base/(name+'.jsonl'))
    curve=journals['validation'];snapshots=journals['snapshots']
    if [r['episode'] for r in curve]!=protocol['evaluation_points'][:len(curve)]:raise ValueError('evaluation schedule drift')
    for row in curve:
        if row['development_sha256']!=protocol['development_sha256'] or row['episodes']!=protocol['development_cases']:
            raise ValueError('fixed development population drift')
        matching=[s for s in snapshots if s['episode']==row['episode']]
        if len(matching)!=1:raise ValueError('snapshot missing or duplicated')
        snapshot=matching[0];path=member(root,snapshot['path']);meta=read(str(path)+'.json')
        if (sha(path)!=snapshot['policy_sha256'] or sha(str(path)+'.json')!=snapshot['metadata_sha256'] or
            meta['source_identity']!=run['source_identity'] or meta['state_sha256']!=row['policy_state_sha256'] or
            meta['sha256']!=row['policy_sha256'] or meta['training_episode']!=row['episode'] or
            meta['updates']!=row['updates'] or meta['resume_supported'] is not False):
            raise ValueError('policy snapshot identity changed')
        path=member(root,row['metrics_path'])
        if sha(path)!=row['metrics_sha256']:raise ValueError('development metrics changed')
        metrics=read(path)
        if sha(path.parent/'episodes.jsonl')!=metrics['raw_episodes_sha256']:raise ValueError('raw development episodes changed')
        if any(row[k]!=v for k,v in metrics.items() if k!='episode_metrics'):
            raise ValueError('validation journal differs from development metrics')
    train=journals['training_episodes'];updates=journals['updates']
    if [r['episode'] for r in train]!=list(range(1,len(train)+1)):raise ValueError('training episode sequence changed')
    if any(a['complete_updates']>=b['complete_updates'] for a,b in zip(updates,updates[1:])):
        raise ValueError('optimizer update sequence changed')
    completed=(base/'completed.json').exists() and (root/'completed.json').exists()
    if completed:
        receipt=read(base/'completed.json');final=read(root/'completed.json')
        if (receipt['source_identity']!=run['source_identity'] or final['source_identity']!=run['source_identity'] or
            not final['completed'] or final['episodes']!=protocol['online_training_episodes'] or
            len(train)!=protocol['online_training_episodes'] or
            [r['episode'] for r in curve]!=protocol['evaluation_points']):raise ValueError('incomplete final report')
        for name,h in receipt['journals'].items():
            if sha(base/(name+'.jsonl'))!=h:raise ValueError('completed journal changed: '+name)
    return root,protocol,journals,receipts,completed


def table(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def plots(output,journals):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:return dict(available=False,reason=str(exc),fallback='JSON/CSV/HTML remain available')
    curve=journals['validation'];train=journals['training_episodes'];updates=journals['updates']
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for ax,key in zip(axes[0],('success','collision','timeout')):
        x=[r['episode'] for r in curve];y=[r[key+'_rate'] for r in curve]
        ci=np.asarray([r[key+'_ci95'] for r in curve])
        if len(x):
            ax.plot(x,y,'o-');ax.fill_between(x,ci[:,0],ci[:,1],alpha=.15,label='Wilson 95% interval')
            ax.legend(fontsize=8)
        ax.set(title=key+' rate (fixed development)',ylim=(-.03,1.03),xlabel='Online training episode')
    ax=axes[1,0]
    if curve:
        ax.plot([r['episode'] for r in curve],[r['team_discounted_return'] for r in curve],'o-',label='Development')
    if train:
        vals=[r['team_discounted_return'] for r in train];window=min(25,len(vals))
        ax.plot([r['episode'] for r in train],vals,alpha=.15,label='Training (exploration)')
        ax.plot(np.arange(window,len(vals)+1),np.convolve(vals,np.ones(window)/window,mode='valid'),label=f'Training mean({window})')
    ax.set(title='Original discounted task return',xlabel='Online episode');ax.legend(fontsize=8)
    ax=axes[1,1]
    for key in ('event_loss_and_task_loss','actor_loss'):
        if updates:ax.plot([r['complete_updates'] for r in updates],[r[key] for r in updates],label=key)
    ax.set(title='Optimization diagnostics (not convergence proof)',xlabel='Actual optimizer updates',yscale='symlog');ax.legend(fontsize=8)
    ax=axes[1,2]
    for key in ('stagnation_rate','uncertified_fraction'):
        if curve:ax.plot([r['episode'] for r in curve],[r[key] for r in curve],'o-',label=key)
    ax.set(title='Failure to make progress / uncertified actions',xlabel='Online episode',ylim=(-.03,1.03));ax.legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle('Single-seed empirical development monitoring; no formal convergence or safety claim')
    for suffix in ('png','svg'):fig.savefig(output/('learning_curves.'+suffix),dpi=150)
    plt.close(fig)
    return dict(available=True,files=['learning_curves.png','learning_curves.svg'])


def render(root,output):
    root,protocol,journals,receipts,completed=inputs(root)
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=False)
    result=assess(journals['validation'],settings(protocol['settings']),protocol['max_episode_steps'])
    result['provisional']=not completed
    if not completed:
        result['provisional_convergence_candidate']=result['convergence_candidate'];result['convergence_candidate']=False
    report=dict(schema='auv.reach_avoid.empirical_report.v1',source_identity=protocol['source_identity'],
        training_completed=completed,planned_training_episodes=protocol['online_training_episodes'],
        completed_training_episodes=len(journals['training_episodes']),development_cases=protocol['development_cases'],
        evaluation_points=[r['episode'] for r in journals['validation']],assessment=result,
        input_journal_prefixes=receipts,run=str(root),final_test_used=False,
        episode0=protocol['episode0'],single_seed_limit='12 development cases do not establish generalization or RL convergence')
    updates=journals['updates'];train=journals['training_episodes']
    report['optimization_evidence']=dict(recorded_complete_updates=len(updates),
        pretrain_updates=sum(r['phase']=='pretrain' for r in updates),online_updates=sum(r['phase']=='online' for r in updates),
        episodes_with_actor_parameter_change=sum(r['actor_parameter_delta_l2']>0 for r in train),
        episodes_with_critic_parameter_change=sum(r['critic_parameter_delta_l2']>0 for r in train),
        loss_flatness_is_not_convergence=True)
    for name,rows in journals.items():table(output/(name+'.csv'),rows)
    report['plots']=plots(output,journals);create_json(output/'report.json',report)
    columns=('episode','updates','success_rate','collision_rate','timeout_rate','stagnation_rate','team_discounted_return','penalized_completion_steps')
    cells=lambda row:''.join('<td>'+html.escape(str(row.get(k,'')))+'</td>' for k in columns)
    body=''.join('<tr>'+cells(r)+'</tr>' for r in journals['validation'])
    with (output/'report.html').open('x',encoding='utf-8') as f:
        f.write('<!doctype html><meta charset="utf-8"><title>Reach-avoid learning report</title><style>body{font:16px system-ui;margin:2em}td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}img{max-width:100%}</style>'
            '<h1>Empirical learning report</h1><p>'+html.escape(result['status'])+'</p><p>Training completed: '+str(completed)+
            '. Development monitoring only; one training seed; no convergence theorem or final-test acceptance.</p>'+
            ('<img src="learning_curves.png" alt="Training and fixed development curves">' if report['plots']['available'] else '<p>Plot dependency missing; CSV and JSON are complete.</p>')+
            '<table><tr>'+''.join('<th>'+k+'</th>' for k in columns)+'</tr>'+body+'</table><pre>'+html.escape(json.dumps(result,indent=2))+'</pre>')
    return report
