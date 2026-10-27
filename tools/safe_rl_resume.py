"""Validate a completed five-case prefix; no checkpoint or environment restore.

Only the published b57d8e5 transport-timeout prefix is accepted. Borrowed
results keep their original source identity and are hashed as parent evidence.
"""
import hashlib,json,platform,sys
from pathlib import Path
import numpy as np
import torch
from project_paths import ROOT
from tools.safe_rl_validation import require_same_host_repeat,numerical_frames
from tools.safe_rl_acceptance import require_recoverable_case

PREFIX=(('baseline_0002','baseline'),('baseline_repeat_0002','repeat'),('composition_0002','collection'),('rule_0002','rule'),('rule_0006','rule'))
PARENT_COMMIT='b57d8e5c9f776c912138a71dc613e3bfa4f29fa4'
PARENT_REVIEW='088a5dc07f206d3855aa548c5da0d64aca5d8317efdfdd5dffcf2cc1329c0c41'


def claim_prefix(receipt,output,*,root=ROOT):
    parent=Path(root)/'work/safe_rl_prefix_claims';parent.mkdir(parents=True,exist_ok=True)
    key=hashlib.sha256(receipt['directory'].encode()).hexdigest();path=parent/(key+'.json')
    payload=dict(prefix_receipt=receipt,continuation_output=str(Path(output).resolve()),reserved_additional_initializations=2)
    try:
        with path.open('x',encoding='utf-8') as stream:json.dump(payload,stream,indent=2)
    except FileExistsError:raise ValueError('prefix continuation already claimed; inspect prior output rather than resetting its budget')
    return str(path)


def require_prefix_unchanged(receipt):
    for name,expected in receipt['files'].items():
        if hashlib.sha256((Path(receipt['directory'])/name).read_bytes()).hexdigest()!=expected:raise ValueError('borrowed prefix changed during verification')


def load_prefix(directory,*,root=ROOT):
    root=Path(root).resolve();directory=Path(directory).resolve()
    if not directory.is_relative_to(root/'work'):raise ValueError('prefix must be this checkout work directory')
    hashes={}
    def read(name):
        p=directory/name;raw=p.read_bytes();hashes[name]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
    v=read('validation.json');source=v.get('source',{})
    if (v.get('passed') is not False or 'TimeoutExpired' not in v.get('traceback','')
        or source.get('review_sha256')!=PARENT_REVIEW or source.get('profile')!='active_safe_rl_recovery_v2'
        or v.get('git',{}).get('head')!=PARENT_COMMIT or v.get('git',{}).get('clean') is not True
        or v.get('package_before',{}).get('passed') is not True):raise ValueError('unsupported timeout prefix identity')
    expected_identity=json.loads((root/'provenance/baselines/safe_rl_recovery_20261027_v2/baseline.json').read_text(encoding='utf-8'))['source_identity']
    if source!=expected_identity:raise ValueError('prefix source identity differs from frozen parent')
    if (v.get('platform')!=platform.platform() or v.get('python')!=sys.version
        or v.get('linux_execution')!=(platform.system()=='Linux')
        or v.get('versions')!={'numpy':np.__version__,'torch':torch.__version__}):raise ValueError('prefix host/runtime differs')
    structural=read('checks/structural.json');regression=read('checks/regression.json')
    if structural.get('passed') is not True or structural.get('source',{}).get('manifest_sha256')!=PARENT_REVIEW:raise ValueError('prefix structural proof missing')
    if (regression.get('passed') is not True or regression.get('tests_run')!=363 or regression.get('expected')!=363
        or any(regression.get(k) for k in ('failures','errors','skipped'))):raise ValueError('prefix regression proof incomplete')
    native=directory/'checks/safe_rl'
    expected={f'initialization_{i:02d}.json' for i in range(1,6)}
    if {p.name for p in native.glob('initialization_*.json')}!=expected:raise ValueError('prefix must contain exactly five initialization reservations')
    if (native/'fault_all_four').exists() or (native/'buffer_exit_all_four').exists():raise ValueError('suffix already attempted; do not reset its budget')
    records=[];all_frames={}
    for i,(name,mode) in enumerate(PREFIX,1):
        reservation=read(f'checks/safe_rl/initialization_{i:02d}.json')
        if reservation!=dict(case=name,max_steps=32,source=source):raise ValueError('prefix reservation changed')
        episode=read(f'checks/safe_rl/{name}/episode.json');frames=read(f'checks/safe_rl/{name}/physical_frames.json')
        if (episode.get('actual_length')!=32 or episode.get('termination_reason')!='timeout'
            or episode.get('first_collision_agent_ids') or episode.get('collision_records')):raise ValueError('prefix episode incomplete or collided')
        if not numerical_frames(frames,frames)['complete'] or len(frames)!=32:raise ValueError('prefix physical frames incomplete')
        all_frames[name]=frames
        if mode=='rule':
            trace=episode['safety_rl']['trace'];initial=[next(r['physical_position'] for r in trace if r['step']==0 and r['agent_id']==a) for a in range(4)]
            report=require_recoverable_case(episode,frames,initial)
            report_name=f'checks/safe_rl/{name}/acceptance.json'
            if (directory/report_name).exists() and read(report_name)!=report:raise ValueError('prefix acceptance changed')
        records.append(dict(case=name,mode=mode,physical_steps=32,termination_reason='timeout',success=episode['success'],
            collision_agents=[],safety_counts=episode.get('safety_rl',{}).get('counts',{}),
            evidence_origin='immutable_parent_prefix',source_identity=source))
    first=read('checks/safe_rl/baseline_0002/signatures.json');second=read('checks/safe_rl/baseline_repeat_0002/signatures.json')
    parity=require_same_host_repeat(first,second,physical_steps=32)
    if read('checks/safe_rl/baseline_repeat_0002/same_host_parity.json')!=parity:raise ValueError('prefix parity report changed')
    composition=numerical_frames(all_frames['baseline_0002'],all_frames['composition_0002'])
    if not composition['numerically_close_at_5e_5']:raise ValueError('prefix composition parity failed')
    saved=read('checks/safe_rl/composition_0002/composition_parity.json')
    if saved.get('passed') is not True or saved.get('max_errors')!=composition['max_absolute_errors']:raise ValueError('prefix composition report changed')
    archive=read('checks/safe_rl/baseline_0002/archived_reference_comparison.json')
    receipt=dict(directory=str(directory),source_identity=source,files=hashes,
        completed_cases=5,environment_initializations=5,physical_steps=160,
        results_relabelled=False,checkpoint_restore=False,scope='Parent five-case results; current suffix is recorded separately')
    return dict(receipt=receipt,records=records,same_host_parity=parity,archived_reference_comparison=archive)
