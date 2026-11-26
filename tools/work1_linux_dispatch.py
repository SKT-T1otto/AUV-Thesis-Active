"""Bounded, paired Linux experiment on four preregistered, independent R123 launches."""
import argparse
from contextlib import ExitStack
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import torch
from chapter3_bser.d2 import allocator as native
from chapter3_bser.learning.config import validate
from chapter3_bser.learning.reference import ZeroResidualPolicy
from chapter3_bser.objective import evaluate_objective
from chapter3_bser.r123_final_leg import RuleRuntime
from chapter3_bser.sweep_events import swept_cells
from project_paths import ROOT
from tools.work1_linux_dispatch_maps import snapshot

CONFIG = ROOT / 'configs/work1_dispatch_linux_v1.json'
RULE = ROOT / 'configs/r123_expansion_v1/config.json'
OPTIONS = ROOT / 'configs/r123_expansion_v1/options.json'
FORECAST = ROOT / 'configs/coupled_search_core_v1.json'
PROTOCOL = ROOT / 'docs/WORK1_DISPATCH_LINUX.md'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def convert(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def encode(value):
    return json.dumps(value, default=convert, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def write(path, value):
    with path.open('x', encoding='utf8') as file:
        file.write(encode(value))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def episode(folder, scene, ordinal, model, baseline=None, cap=32):
    """Select once at t=0. Save predictions before physics; record actual receiver and scan facts."""
    if not 1 <= cap <= 32:
        raise ValueError('Physical episode cap must be 1..32')
    folder.mkdir()
    entry = {}
    options = read(OPTIONS)
    config = validate(read(RULE))
    forecast = read(FORECAST)
    original = native.EffectAllocator._proposal
    solve = native.solve

    def proposal(allocator, state, candidates, frozen, standbys, executor, trigger):
        if state.step != 0 or entry:
            return original(allocator, state, candidates, frozen, standbys, executor, trigger)
        def launch(arm, menu, ys, context, *, frozen):
            if arm != 'D2' or frozen or executor is not None:
                raise RuntimeError('Unexpected initial allocation contract')
            space = solve(arm, menu, ys, context, frozen=frozen)
            decision = snapshot(allocator, state, menu, space, context, model, forecast,
                                options['arrival_radius'], ordinal)
            decision['rng_digest'] = sha(encode([random.getstate(), np.random.get_state(),
                                                 torch.random.get_rng_state().tolist()]).encode())
            serialized = encode(decision)
            if baseline is not None and sha(serialized.encode()) != baseline['decision_sha256']:
                raise RuntimeError('Paired launch differs in public state, RNG, menus or frozen predictions')
            plan_id = 0 if baseline is None else decision['branch']
            if plan_id is None:
                raise RuntimeError('Attempt to execute an unscheduled branch')
            write(folder/'prediction.json', decision)
            entry.update(decision=decision, decision_sha256=sha(serialized.encode()), plan=plan_id)
            if plan_id == 0:
                return space
            by_id = {c.candidate_id:c for c in menu}
            selected = tuple(by_id[token] for token in decision['plans'][plan_id])
            return replace(space, selected=selected, objective=evaluate_objective(selected, space.standby,
                                                                                 context, search_only=True))
        with patch.object(native, 'solve', launch):
            return original(allocator, state, candidates, frozen, standbys, executor, trigger)

    with ExitStack() as stack:
        stack.enter_context(patch.object(native.EffectAllocator, '_proposal', proposal))
        runtime = RuleRuntime(config, scene, seed=int(scene['scenario_seed']), episode_id=0, options=options,
                              arm='rule', tracking_arm='both', task_sync=True, direct_final_leg=True)
        stack.callback(runtime.close)
        decision = entry['decision']
        env = runtime.env
        before_step = env.step
        started = []
        def observe(*args, **kwargs):
            commands, receipts = [], []
            for i in range(4):
                cmd = runtime.assignment(i)
                task = runtime.lifecycle.tasks[i]
                accepted = bool(cmd['assigned'] and cmd['token'] == runtime._task_tokens.get(i)
                                and np.allclose(cmd['point'], task.point, atol=1e-4, rtol=0) and not task.released)
                commands.append(cmd['token'])
                receipts.append(dict(accepted=accepted, accepted_token=runtime._task_tokens.get(i),
                                     route_id=runtime.manager.current[i].route.route_id))
            started.append((commands, receipts))
            return before_step(*args, **kwargs)
        stack.enter_context(patch.object(env, 'step', observe))
        stream = stack.enter_context(gzip.open(folder/'physics.jsonl.gz','xt',encoding='utf8'))
        cursor = max((e.sequence for e in env.read_public_events(0)), default=0)
        policy = ZeroResidualPolicy()
        for _ in range(cap):
            if runtime.terminal:
                break
            started.clear()
            # The exogenous target coordinates are hashed only on the offline evidence side.
            target_start = env.unwrapped.target_state.position.copy()
            runtime.advance(policy, explore=False)
            (commands, receipts), = started
            batch = env.read_public_events(cursor)
            cursor = batch[-1].sequence
            event, = [e for e in batch if e.kind=='step']
            obs = event.payload
            if event.physical_step != runtime.step:
                raise RuntimeError('Physical clock mismatch')
            cells = [swept_cells(decision['centers'], obs.agent_start[i], obs.agent_end[i], r,
                                 obs.discovery_check_completed) for i,r in enumerate(decision['radii'])]
            target_digest = sha(encode([target_start, env.unwrapped.target_state.position]).encode())
            fact = dict(step=event.physical_step, commands=commands, receipts=receipts, hits=cells,
                        start=obs.agent_start, end=obs.agent_end, velocity_start=obs.velocity_start,
                        checked=obs.discovery_check_completed, found_before=obs.found_before,
                        found=obs.found_after, terminal=obs.episode_terminated, reason=obs.terminal_reason,
                        target_motion_digest=target_digest)
            stream.write(encode(fact)+'\n')
    result = dict(decision_sha256=entry['decision_sha256'], plan=entry['plan'], steps=runtime.step,
                  seed=scene['scenario_seed'], layout=scene['obstacle_layout_sha256'], terminal=runtime.terminal)
    write(folder/'complete.json', result)
    return result


def run(out):
    from tools.work1_linux_dispatch_report import summarize
    torch.set_num_threads(1)
    cfg = read(CONFIG)
    scenes = cfg['scenes']
    if len(scenes)!=4 or len({s['obstacle_layout_sha256'] for s in scenes})!=4:
        raise ValueError('Expected four registered physical families')
    weights = ROOT/cfg['calibration']
    if sha(weights.read_bytes()) != cfg['model_sha256']:
        raise ValueError('Frozen calibration identity mismatch')
    model = read(weights)
    out.mkdir(parents=True, exist_ok=False)
    sources = [CONFIG, RULE, OPTIONS, FORECAST, PROTOCOL, weights, Path(__file__),
               ROOT/'tools/work1_linux_dispatch_maps.py', ROOT/'tools/work1_linux_dispatch_report.py',
               *(ROOT/'chapter3_bser'/name for name in ('predispatch_evidence.py','predispatch_sweep.py',
                  'execution_calibration.py','sweep_events.py','r123_final_leg.py',
                  'experiments/coupled_search_v1/execution_forecast.py'))]
    rev = subprocess.run(['git','rev-parse','HEAD'], cwd=ROOT, capture_output=True, text=True)
    write(out/'registration.json', dict(protocol=PROTOCOL.read_text(encoding='utf8'),
          roster=[dict(ordinal=i,seed=s['scenario_seed'],layout=s['obstacle_layout_sha256']) for i,s in enumerate(scenes)],
          preregistered_inputs={p.relative_to(ROOT).as_posix():sha(p.read_bytes()) for p in sources},
          revision=rev.stdout.strip(), platform=platform.platform(), python=sys.version,
          torch=str(torch.__version__), numpy=np.__version__, max_episodes=8, cap=32))
    for i,scene in enumerate(scenes):
        family = out/f'family_{i+1:02d}'
        family.mkdir()
        print(f'[{i+1}/4] seed={scene["scenario_seed"]} spatial',flush=True)
        anchor = episode(family/'spatial', scene, i, model)
        decision = read(family/'spatial/prediction.json')
        if decision['branch'] is not None:
            print(f'[{i+1}/4] paired assignment {decision["branch"]}: {decision["branch_reason"]}',flush=True)
            episode(family/'alternative', scene, i, model, anchor)
    summarize(out)
    print(f'Report: {out / "REPORT.md"}',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path,help='new directory for the registered batch')
    p.add_argument('--execute',action='store_true',required=True)
    run(p.parse_args().output.resolve())
