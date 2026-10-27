"""Hash-bound factual trajectories. No hypothetical action receives a safe label."""
import hashlib,json
from pathlib import Path
import numpy as np
from chapter3_bser.safe_rl.options import SCHEMA

def save(path,transitions):
    path=Path(path)
    arrays=dict(features=np.asarray([r['safety_features'] for r in transitions],np.float32),
        actions=np.asarray([r['normalized_executed_acceleration'] for r in transitions],np.float32),
        costs=np.asarray([r['collision_cost'] for r in transitions],np.float32),
        modes=np.asarray([[x['mode'] for x in r['safety_rows']] for r in transitions]),
        checked=np.asarray([[x['model_checked'] for x in r['safety_rows']] for r in transitions],bool))
    with path.open('xb') as stream:np.savez_compressed(stream,**arrays)
    return dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),physical_steps=len(transitions))

def load_collection(root,source,options):
    root=Path(root);read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    run=read(root/'run.json');receipt=read(root/'dataset.json')
    if run.get('schema')!=SCHEMA or run.get('source_identity')!=source or run.get('command')!='collect' or run.get('options')!=options:
        raise ValueError('factual safety dataset identity mismatch')
    if receipt.get('complete') is not True or receipt.get('source_identity')!=source:raise ValueError('incomplete safety dataset')
    completed=read(root/'completed.json')
    if completed.get('completed') is not True or completed.get('source_identity')!=source:raise ValueError('incomplete factual run')
    if (receipt.get('run_sha256')!=hashlib.sha256((root/'run.json').read_bytes()).hexdigest()
        or receipt.get('scenarios_sha256')!=hashlib.sha256((root/'scenarios.json').read_bytes()).hexdigest()):raise ValueError('factual dataset metadata changed')
    scenes=read(root/'scenarios.json')['scenarios']
    if any(s.get('scenario_role')!='train' or s.get('scenario_split')!='train' for s in scenes):raise ValueError('evaluation data cannot train safety models')
    episodes=[];positives=0
    for record in receipt['files']:
        p=(root/record['file']).resolve()
        if p.parent!=root.resolve() or p.suffix!='.npz' or hashlib.sha256(p.read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('safety dataset file/hash mismatch')
        with np.load(p,allow_pickle=False) as archive:arrays={k:archive[k].copy() for k in archive.files}
        n=record['physical_steps']
        if (arrays['features'].shape!=(n,4,192) or arrays['actions'].shape!=(n,4,3)
            or arrays['costs'].shape!=(n,4) or arrays['checked'].shape!=(n,4) or arrays['modes'].shape!=(n,4)):
            raise ValueError('safety dataset tensor contract mismatch')
        if any(not np.isfinite(arrays[k]).all() for k in ('features','actions','costs')):raise ValueError('nonfinite factual safety dataset')
        if not np.isin(arrays['costs'],[0.,1.]).all() or np.any(np.abs(arrays['actions'])>1.+1e-5):raise ValueError('invalid factual actions/costs')
        if not np.all(arrays['modes']=='NOMINAL_COLLECTION') or arrays['checked'].any():raise ValueError('only original nominal factual data permitted here')
        rows=[]
        for t in range(n):rows.append(dict(safety_features=arrays['features'][t],normalized_executed_acceleration=arrays['actions'][t],
            collision_cost=arrays['costs'][t],safety_rows=[dict(mode=str(arrays['modes'][t,i]),model_checked=bool(arrays['checked'][t,i])) for i in range(4)]))
        positives+=int(arrays['costs'].sum());episodes.append(rows)
    if not episodes or positives<1:raise ValueError('risk pretraining requires observed collision positives; safe-only data is insufficient')
    return episodes,scenes
