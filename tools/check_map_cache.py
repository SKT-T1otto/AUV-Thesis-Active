"""Bounded pure-data cache parity and timings; no simulator or model creation."""
import argparse
import copy
from dataclasses import replace
import itertools
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
from project_paths import ROOT
from core.env.public_events import ScanObservation
from chapter3_bser.experiments.coupled_search_v1.core_config import CORE_DEFAULT
from chapter3_bser.experiments.coupled_search_v1.geometry_hypotheses import GeometryHypothesis, GeometryHypothesisBank, compatible
from chapter3_bser.experiments.coupled_search_v1.geometry_cache import ObservationCache
from chapter3_bser.experiments.coupled_search_v1.numerics import Budget, BudgetExceeded
from checks.map_cache_reference import ReferenceBank, compatible as reference_compatible

BOUNDS=np.array([20.,20.,10.])
CONTRACT=dict(bounds=BOUNDS,obstacle_sensor_range=4.5)


def scalar_distance(origin,direction,center,size,limit):
    low=np.asarray(center)-np.asarray(size)/2;high=np.asarray(center)+np.asarray(size)/2
    near=0.;far=np.inf
    for axis in range(3):
        if abs(direction[axis])<=1e-12:
            if origin[axis]<low[axis] or origin[axis]>high[axis]:return limit
            continue
        left=(low[axis]-origin[axis])/direction[axis];right=(high[axis]-origin[axis])/direction[axis]
        near=max(near,min(left,right));far=min(far,max(left,right))
    return near if near<=far and far>=0 and near<limit else limit


def synthetic_history(count):
    directions=np.array([v for v in itertools.product((-1.,0.,1.),repeat=3) if any(v)])
    directions/=np.linalg.norm(directions,axis=1)[:,None]
    scans=[]
    for step in range(count):
        origins=np.array([(6.1+.15*np.sin(step*.03+i),4.1+(step*.037+i*2.6)%11.5,2.1+i*1.6) for i in range(4)])
        distances=[];hits=[]
        for origin in origins:
            row=[];flags=[]
            for direction in directions:
                limit=4.5
                for axis in range(3):
                    if abs(direction[axis])>1e-12:
                        limit=min(limit,max(0.,((BOUNDS[axis] if direction[axis]>0 else 0.)-origin[axis])/direction[axis]))
                distance=scalar_distance(origin,direction,(10.,10.,5.),(1.,12.,8.),limit)
                row.append(float(distance));flags.append(bool(distance<limit-1e-10))
            distances.append(tuple(row));hits.append(tuple(flags))
        scans.append(ScanObservation(tuple(map(tuple,origins)),tuple(map(tuple,directions)),tuple(distances),tuple(hits),(0,1,2,3)))
    return scans


def timed_proposal(bank,scans):
    started=time.perf_counter()
    try:
        maps=bank.propose(scans,Budget(2.))
        result=dict(outcome='completed',map_ids=[geometry.identity for geometry in maps],
                    rng_state=copy.deepcopy(bank.rng.bit_generator.state))
    except BudgetExceeded as error:
        result=dict(outcome=str(error))
    result['seconds']=time.perf_counter()-started
    result['kernel_calls']=bank.kernel_calls if isinstance(bank,ReferenceBank) else bank.last_entry_stats.get('kernel_calls',0)
    if isinstance(bank,GeometryHypothesisBank):
        result['entry_cache']=dict(bank.last_entry_stats)
        result['observation_cache']=bank.scan_cache.summary()
    return result


def run(output):
    from chapter3_bser.learning.provenance import verify_reviewed_changes,require_review
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT/'work'):
        raise ValueError('fresh output under Active/work is required')
    output.mkdir(parents=True,exist_ok=False)
    source=verify_reviewed_changes();started=time.perf_counter()
    scans=synthetic_history(256);rows=[];complete=0
    for count in (32,128,256):
        left=ReferenceBank(copy.deepcopy(CORE_DEFAULT),CONTRACT,np.random.default_rng(17))
        right=GeometryHypothesisBank(copy.deepcopy(CORE_DEFAULT),CONTRACT,np.random.default_rng(17))
        reference=timed_proposal(left,scans[:count]);cached=timed_proposal(right,scans[:count])
        row=dict(scans=count,rays=count*104,reference=reference,cached=cached,per_proposal_budget_seconds=2.)
        if reference['outcome']==cached['outcome']=='completed':
            if reference['map_ids']!=cached['map_ids'] or reference['rng_state']!=cached['rng_state']:
                raise AssertionError('proposal or RNG parity failure')
            complete+=1
            row.update(map_and_rng_equal=True,speedup=reference['seconds']/cached['seconds'],
                       kernel_call_reduction=1-cached['kernel_calls']/reference['kernel_calls'])
        else:
            row['comparison_censored_by_compute_budget']=True
        rows.append(row)
    if complete<1:raise AssertionError('no complete geometry comparison')
    # Mix unchanged observations, new content and changed map/parameters. No
    # history likelihood is assumed true from a successful geometry proposal.
    cache=ObservationCache(513);geometry=GeometryHypothesis((((10.,10.,5.),(1.,12.,8.)),))
    expected=[];start=time.perf_counter()
    for scan in scans:
        expected.append(reference_compatible(geometry,scan,BOUNDS,4.5,.25))
    reference_time=time.perf_counter()-start
    start=time.perf_counter()
    cold=[cache.compatible(geometry,scan,BOUNDS,4.5,.25,compatible) for scan in scans]
    cold_time=time.perf_counter()-start
    start=time.perf_counter()
    warm=[cache.compatible(geometry,scan,BOUNDS,4.5,.25,compatible) for scan in scans]
    warm_time=time.perf_counter()-start
    if expected!=cold or expected!=warm:raise AssertionError('compatibility parity failure')
    false_cases=0
    for scan in scans[:16]:
        changed=replace(scan,distances=tuple(tuple(0. for _ in row) for row in scan.distances))
        for candidate,tolerance in ((geometry,.25),(GeometryHypothesis(()),.25),(geometry,0.)):
            value=reference_compatible(candidate,changed,BOUNDS,4.5,tolerance)
            if value!=cache.compatible(candidate,changed,BOUNDS,4.5,tolerance,compatible):
                raise AssertionError('changed-input invalidation failure')
            false_cases+=int(not value)
    require_review(source)
    result=dict(passed=True,source_identity=source,python=sys.version,numpy=np.__version__,platform=platform.platform(),
        physical_episodes=0,physical_steps=0,model_updates=0,checkpoint_restores=0,formal_training_started=False,
        original_kernel_unchanged=True,full_history_unchanged=True,proposal_comparisons=rows,
        compatibility=dict(comparisons=2*len(scans)+48,reference_seconds=reference_time,cold_seconds=cold_time,
            warm_seconds=warm_time,warm_speedup=reference_time/warm_time,false_cases=false_cases,
            results_equal=True,cache=cache.summary()),elapsed_seconds=time.perf_counter()-started,
        timing_scope='Windows pure-data samples; 2-second proposal budget; censored comparisons are not speedups',
        performance_passed=False)
    with (output/'cache_checks.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(executed=False,physical_episodes=0,output=args.output)));return
    print(json.dumps(run(args.output),indent=2,ensure_ascii=False))


if __name__=='__main__':main()
