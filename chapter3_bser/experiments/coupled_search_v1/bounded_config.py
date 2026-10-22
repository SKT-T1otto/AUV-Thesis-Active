"""Explicit v3 approximation: persistent voxels and finite posterior replay."""
import copy
import math
from .core_config import CORE_DEFAULT

SCHEMA='ch3.coupled_search.core.v3'
PLANNER_ID='coupled_search_bounded_v3'
PRIOR_ID='persistent_voxel_conditional_aabb_v3'
BOUNDED_DEFAULT=copy.deepcopy(CORE_DEFAULT)
BOUNDED_DEFAULT.update(schema=SCHEMA,history_steps=64)
BOUNDED_DEFAULT['geometry']['prior_id']=PRIOR_ID
BOUNDED_DEFAULT['bounded']=dict(grid_shape=[20,20,10],max_cells=4096,max_boxes=192,
    scan_sample_spacing=.4,free_logodds=-.85,occupied_logodds=1.7,logodds_clip=6.,
    free_threshold=.3,occupied_threshold=.7,map_memory='persistent',boundary_mode='posterior',
    local_recovery=True,position_jitter=.35,velocity_jitter=.08,resample_ess_fraction=.5,
    design_variant='full')


def validate_bounded_config(value):
    from .core_config import validate_core_config
    c=copy.deepcopy(value)
    if not isinstance(c,dict) or set(c)!=set(BOUNDED_DEFAULT) or c.get('schema')!=SCHEMA:
        raise ValueError('complete bounded v3 configuration required')
    b=c.pop('bounded')
    if not isinstance(b,dict) or set(b)!=set(BOUNDED_DEFAULT['bounded']):
        raise ValueError('complete bounded-state parameters required')
    if c['geometry']['prior_id']!=PRIOR_ID:raise ValueError('bounded geometry prior mismatch')
    c['schema']=CORE_DEFAULT['schema'];c['geometry']['prior_id']=CORE_DEFAULT['geometry']['prior_id']
    c=validate_core_config(c)
    if c['history_steps'] not in (16,32,64,128):raise ValueError('bounded replay window must be 16, 32, 64 or 128')
    shape=b['grid_shape']
    if not isinstance(shape,list) or len(shape)!=3 or any(type(n) is not int or not 2<=n<=40 for n in shape):
        raise ValueError('invalid persistent grid shape')
    for key,lo,hi in (('max_cells',8,16384),('max_boxes',4,256)):
        if type(b[key]) is not int or not lo<=b[key]<=hi:raise ValueError('invalid bounded capacity: '+key)
    if math.prod(shape)>b['max_cells']:raise ValueError('persistent grid exceeds cell budget')
    for key in ('scan_sample_spacing','free_logodds','occupied_logodds','logodds_clip','free_threshold',
                'occupied_threshold','position_jitter','velocity_jitter','resample_ess_fraction'):
        v=b[key]
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v):raise ValueError('invalid bounded numeric: '+key)
    if (not .1<=b['scan_sample_spacing']<=1. or not b['free_logodds']<0<b['occupied_logodds']
        or not 1<=b['logodds_clip']<=12 or not 0<b['free_threshold']<.5<b['occupied_threshold']<1
        or not 0<=b['position_jitter']<=1. or not 0<=b['velocity_jitter']<=.3
        or not 0<=b['resample_ess_fraction']<=1 or type(b['local_recovery']) is not bool):
        raise ValueError('invalid bounded inference parameters')
    expected={'full':('persistent','posterior',True),'no_boundary':('persistent','uniform_ablation',True),
              'no_local_recovery':('persistent','posterior',False),'window_map':('window_ablation','posterior',True)}
    if b['design_variant'] not in expected or (b['map_memory'],b['boundary_mode'],b['local_recovery'])!=expected[b['design_variant']]:
        raise ValueError('ablation label does not match bounded algorithm switches')
    c.update(schema=SCHEMA,bounded=b);c['geometry']['prior_id']=PRIOR_ID
    return c
