"""Hash-bound factual transition data with public snapshots and projected actions."""
import hashlib,json
from pathlib import Path
import numpy as np
from chapter3_bser.reach_avoid.features import Packet
from chapter3_bser.reach_avoid.options import SCHEMA, FRAME_DIM

FIELDS=('features','positions','velocities','flow','phases','intents','target_velocity','clearances',
        'goals','goal_mask','actions','action_mask','checked')


def save(path, records):
    if not records:raise ValueError('empty reach-avoid dataset episode')
    packets=[records[0]['packet']]+[r['next_packet'] for r in records]
    arrays={name:np.asarray([getattr(p,name) for p in packets]) for name in FIELDS}
    for name in ('origin','spacing','shape','bounds'):
        arrays[name]=np.asarray(getattr(packets[0],name))
        if any(not np.array_equal(getattr(p,name),arrays[name]) for p in packets):raise ValueError('public grid changed inside episode')
    arrays.update(steps=np.array([p.step for p in packets]),
        free=np.packbits(np.array([p.free for p in packets]),axis=1),
        occupied=np.packbits(np.array([p.occupied for p in packets]),axis=1),
        selected_goals=np.array([r['goals'] for r in records]),actual_actions=np.array([r['actual_actions'] for r in records]),
        rewards=np.array([r['reward'] for r in records]),done=np.array([r['done'] for r in records]),
        collisions=np.array([r['team_collision'] for r in records]),
        intervened=np.array([[x['projection_changed'] for x in r['rows']] for r in records]))
    path=Path(path)
    with path.open('xb') as f:np.savez_compressed(f,**arrays)
    return dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),physical_steps=len(records))


def load_file(path, record, options):
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('reach-avoid dataset hash changed')
    with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    n=record['physical_steps'];k=options['candidates'];m=options['action_bank_size']
    if type(n) is not int or n<1:raise ValueError('positive factual physical length required')
    expected=dict(features=(n+1,4,FRAME_DIM*options['history']),positions=(n+1,4,3),velocities=(n+1,4,3),
        flow=(n+1,4,2),phases=(n+1,4),intents=(n+1,4,3),target_velocity=(n+1,3),clearances=(n+1,4),
        goals=(n+1,4,k,13),goal_mask=(n+1,4,k),actions=(n+1,4,m,3),action_mask=(n+1,4,m),checked=(n+1,4,m),
        selected_goals=(n,4,13),actual_actions=(n,4,3),rewards=(n,),done=(n,),collisions=(n,),intervened=(n,4),steps=(n+1,))
    for name,shape in expected.items():
        if name not in a or a[name].shape!=shape or not np.isfinite(a[name]).all():raise ValueError('invalid reach-avoid array: '+name)
    for name in ('goal_mask','action_mask','checked','done','collisions','intervened'):
        if a[name].dtype!=np.bool_:raise ValueError('boolean factual mask required: '+name)
    if np.any(a['checked'] & ~a['action_mask']):raise ValueError('checked action absent from executable library')
    if not np.isin(a['phases'],[0,1,2,3]).all():raise ValueError('invalid factual task phase')
    if a['collisions'][:-1].any() or np.any(a['collisions'] & ~a['done']):raise ValueError('collision must terminate the factual team episode')
    if not np.array_equal(a['steps'],np.arange(n+1)) or not a['done'][-1] or a['done'][:-1].any():raise ValueError('incomplete episode clock/termination')
    if (np.any(np.abs(a['actual_actions'])>1.+1e-5) or np.any(np.abs(a['actions'])>1.+1e-5)
        or not a['action_mask'].any(axis=-1).all()):raise ValueError('invalid physical action library')
    for name in ('origin','spacing','shape','bounds'):
        if name not in a or a[name].shape!=(3,) or not np.isfinite(a[name]).all():raise ValueError('invalid public grid: '+name)
    if np.any(a['spacing']<=0) or np.any(a['bounds']<=0) or np.any(a['shape']<=0) or not np.equal(a['shape'],a['shape'].astype(int)).all():raise ValueError('invalid public grid dimensions')
    shape=a['shape'].astype(int);cells=int(np.prod(shape));packets=[]
    if a['free'].shape!=(n+1,(cells+7)//8) or a['occupied'].shape!=a['free'].shape:raise ValueError('invalid public occupancy packets')
    for t in range(n+1):
        p=Packet(int(a['steps'][t]),*[a[name][t] for name in ('features','positions','velocities','flow','phases','intents','target_velocity','clearances')],
            a['origin'],a['spacing'],shape,np.unpackbits(a['free'][t])[:cells].astype(bool),
            np.unpackbits(a['occupied'][t])[:cells].astype(bool),a['bounds'])
        for name in ('goals','goal_mask','actions','action_mask','checked'):setattr(p,name,a[name][t])
        packets.append(p)
    return [dict(packet=packets[t],next_packet=packets[t+1],goals=a['selected_goals'][t],actual_actions=a['actual_actions'][t],
        reward=float(a['rewards'][t]),done=bool(a['done'][t]),team_collision=bool(a['collisions'][t]),
        rows=[dict(projection_changed=bool(x)) for x in a['intervened'][t]]) for t in range(n)]


def load_collection(root, source, options):
    root=Path(root).resolve();read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    receipt=read(root/'dataset.json');run=read(root/'run.json');complete=read(root/'completed.json')
    if (run.get('schema')!=SCHEMA or run.get('source_identity')!=source or run.get('command')!='collect'
        or run.get('options')!=options or receipt.get('complete') is not True or receipt.get('source_identity')!=source
        or complete.get('completed') is not True or complete.get('source_identity')!=source):raise ValueError('independent reach-avoid collection identity mismatch')
    for name,key in (('run.json','run_sha256'),('scenarios.json','scenarios_sha256')):
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=receipt.get(key):raise ValueError('reach-avoid collection metadata changed')
    scenes=read(root/'scenarios.json')['scenarios']
    if any(s.get('scenario_role')!='train' or s.get('scenario_split')!='train' for s in scenes):raise ValueError('only independent train/train collections may train reach-avoid')
    if len(receipt['files'])!=complete['episodes']:raise ValueError('incomplete collection episode list')
    if len({r['file'] for r in receipt['files']})!=len(receipt['files']):raise ValueError('duplicate factual episode file')
    episodes=[]
    for record in receipt['files']:
        path=(root/record['file']).resolve()
        if path.parent!=root or path.suffix!='.npz':raise ValueError('unsafe reach-avoid dataset path')
        episodes.append(load_file(path,record,options))
    if not episodes:raise ValueError('empty factual reach-avoid collection')
    return episodes,scenes
