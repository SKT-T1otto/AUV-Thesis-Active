"""Pure comparisons: exact same-host repeat, informational cross-host archive."""
import json,platform,re,sys,zipfile
from pathlib import Path
import numpy as np
import torch

FIELDS={'positions':(4,3),'velocities':(4,3),'rewards':(4,),'tracking':(4,3)}


def environment():
    return dict(platform=platform.platform(),python=sys.version,numpy=np.__version__,torch=torch.__version__,
        device='cpu',torch_threads=torch.get_num_threads())


def signature_comparison(current,reference):
    if not isinstance(current,list) or not isinstance(reference,list):raise ValueError('signature lists required')
    if any(not isinstance(x,str) or re.fullmatch('[0-9a-f]{64}',x) is None for x in current+reference):
        raise ValueError('invalid complete state signature')
    mismatches=[i for i,(a,b) in enumerate(zip(current,reference)) if a!=b]
    length_equal=len(current)==len(reference)
    if not length_equal:mismatches.extend(range(min(len(current),len(reference)),max(len(current),len(reference))))
    return dict(exact=len(current)>0 and length_equal and not mismatches,
        current_count=len(current),reference_count=len(reference),mismatch_count=len(mismatches),
        first_mismatch_index=mismatches[0] if mismatches else None)


def require_same_host_repeat(first,second,*,physical_steps):
    if type(physical_steps) is not int or not 1<=physical_steps<=32:raise ValueError('invalid bounded repeat length')
    report=signature_comparison(first,second)
    if len(first)!=physical_steps+1 or len(second)!=physical_steps+1 or not report['exact']:
        raise ValueError('same-host baseline repeat differs: '+json.dumps(report,sort_keys=True))
    return dict(passed=True,scope='same host, same source, same inputs and seeds; full state/RNG/map/guidance signatures',
        physical_steps_per_run=physical_steps,comparison=report,parent_cross_host_identity_claimed=False)


def numerical_frames(current,reference):
    errors={k:0. for k in FIELDS};shape_errors=[]
    for index,(a,b) in enumerate(zip(current,reference)):
        if a.get('t')!=index or b.get('t')!=index:raise ValueError('archive frame clock/order invalid')
        for key,shape in FIELDS.items():
            x,y=np.asarray(a[key],float),np.asarray(b[key],float)
            if x.shape!=shape or y.shape!=shape:shape_errors.append(dict(frame=index,field=key));continue
            if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('nonfinite archive comparison')
            errors[key]=max(errors[key],float(np.max(np.abs(x-y))))
    complete=len(current)==len(reference) and len(current)>0 and not shape_errors
    return dict(current_frames=len(current),reference_frames=len(reference),complete=complete,shape_errors=shape_errors,
        max_absolute_errors=errors,numerically_close_at_5e_5=complete and max(errors.values())<=5e-5,
        tolerance_scope='diagnostic comparison only; does not change same-host exact parity or composition admission')


def archived_report(signatures,frames,archive,reference_environment):
    with zipfile.ZipFile(Path(archive)) as z:
        old_signatures=json.loads(z.read('baseline_0002/signatures.json'))
        old_frames=json.loads(z.read('baseline_0002/physical_frames.json'))
    return dict(scope='cross-host published archive diagnostic; not a baseline admission gate',
        admission_gate=False,current_environment=environment(),reference_environment=reference_environment,
        signatures=signature_comparison(signatures,old_signatures),physical=numerical_frames(frames,old_frames),
        difference_cause='not inferred from aggregate hash or frame errors',
        cross_platform_bitwise_reproducibility_established=False)
