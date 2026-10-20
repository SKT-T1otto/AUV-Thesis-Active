import hashlib
import time
from chapter3_bser.learning.checkpoint import ROOT
from core.scenarios.ch3_generator_impl import build_scenario_manifests
import copy
import json
import random
import traceback
from pathlib import Path
import numpy as np
import torch
from chapter3_bser.d2.logging import JsonlJournal
from chapter3_bser.learning.checkpoint import digest
from chapter3_bser.learning.checkpoint import fresh_source_identity
from chapter3_bser.learning.checkpoint import require_source_match
from chapter3_bser.learning.checkpoint import state_digest
from chapter3_bser.learning.checkpoint import validated_output
from chapter3_bser.learning.checkpoint import write_json
from chapter3_bser.learning.checkpoint import _finite_state
from chapter3_bser.learning.config import VERSION
from chapter3_bser.learning.config import validate
from chapter3_bser.learning.config import scenario_fingerprint
from chapter3_bser.learning.model import build_model
from chapter3_bser.learning.replay import ExecutionReplay
from chapter3_bser.learning.runtime import RepairedMissionRuntime
from chapter3_bser.learning.provenance import training_admission
from chapter3_bser.learning.provenance import verify_reviewed_changes
from chapter3_bser.learning.provenance import require_review
from chapter3_bser.learning import map_contract

class Trainer:

    def __init__(self, config, output, *, mechanism=False):
        self.config = validate(config)
        self.mechanism_only = bool(mechanism)
        self.review = training_admission(mechanism=mechanism, config=self.config)
        self.output = validated_output(self.config, output)
        self.config['output_dir'] = str(self.output)
        self.source = fresh_source_identity()
        self.scenarios = self._load_scenarios()
        random.seed(self.config['seed'])
        np.random.seed(self.config['seed'] % 2 ** 32)
        torch.manual_seed(self.config['seed'])
        self.model = build_model(self.config)
        self.initial_model_state_sha256=state_digest(self.model.training_state_dict())
        self.replay = ExecutionReplay(self.config['rl']['replay_size'])
        self.global_step = self.completed_main = self.optimizer_updates = 0
        self.training_scenario_ids = set()
        self.training_scenario_seeds = set()
        self.training_scenario_fingerprints = set()
        self.output.mkdir(parents=True, exist_ok=True)
        self.episodes = JsonlJournal(self.output / 'episodes.jsonl')
        self.metrics = JsonlJournal(self.output / 'training_metrics.jsonl')
        write_json(self.output / 'config.json', self.config)

    def _update(self):
        require_review(self.review)
        require_source_match(self.source, context='v2 learner update')
        rl = self.config['rl']
        eligible = len(self.replay) >= rl['batch_size'] and self.global_step >= rl['warmup_steps'] and (self.global_step % rl['update_frequency'] == 0)
        if self.mechanism_only and eligible and (self.optimizer_updates + rl['updates_per_train'] > 4):
            raise RuntimeError('mechanism model update limit is four complete rounds')
        return self._perform_update()

    def run_episode(self):
        (scenario, seed) = self._scenario(self.completed_main + 1)
        self.training_scenario_ids.add(scenario['scenario_id'])
        self.training_scenario_seeds.add(int(scenario['scenario_seed']))
        self.training_scenario_fingerprints.add(scenario_fingerprint(scenario))
        runtime = RepairedMissionRuntime(self.config, scenario, seed=seed, episode_id=self.completed_main + 1,
                                        map_research=map_contract.research_config(self.config))
        try:
            self.model.prep_rollouts(device='cpu')
            self.model.reset_noise()
            while not runtime.terminal:
                self.replay.push_transition(map_contract.advance(runtime,self.model,explore=True))
                self.global_step += 1
                self._update()
            row = runtime.summary()
            row.update(map_contract.metrics(runtime))
        except Exception as exc:
            state = self.model.training_state_dict()
            failure = dict(error=repr(exc), traceback=traceback.format_exc(), environment_step=self.global_step, model_state_sha256=state_digest(state), successful=False, diagnostic_state=None)
            try:
                verify_reviewed_changes()
                require_source_match(self.source, context='diagnostic failure snapshot')
            except ValueError as source_error:
                failure['diagnostic_state_blocked_by_source'] = str(source_error)
            else:
                path = self.output / 'diagnostic_failure_state.pt'
                torch.save(dict(diagnostic_failure=True, formal_use_allowed=False, state=state), path)
                failure['diagnostic_state'] = str(path)
            write_json(self.output / 'failure.json', failure)
            raise
        finally:
            runtime.close()
        self.completed_main += 1
        row.update(episode=self.completed_main, global_step=self.global_step, method=self.config['method'], trajectory_complete=True, environment_innovation_seed=seed, max_steps=self.config['max_steps'])
        self.episodes.append(row)
        return row

    def save(self, label):
        require_review(self.review)
        if training_admission(mechanism=self.mechanism_only, config=self.config) != self.review:
            raise ValueError('checkpoint source identity changed')
        require_source_match(self.source, context='repair v2 checkpoint save')
        if self.completed_main < 1:
            raise ValueError('v2 checkpoint requires a completed mechanism or training episode')
        state = self.model.training_state_dict()
        payload = dict(schema=VERSION, artifact_schema='auv.active.checkpoint.rebuild.v1', config=copy.deepcopy(self.config), config_hash=digest(self.config), source_identity=self.source, source_review=self.review, model_state=state, model_state_sha256=state_digest(state), global_step=self.global_step, episodes=self.completed_main, optimizer_updates=self.optimizer_updates, actor_optimizer_steps=list(self.model.actor_updates), critic_optimizer_steps=8 * self.optimizer_updates, training_scenario_ids=sorted(self.training_scenario_ids), training_scenario_seeds=sorted(self.training_scenario_seeds), training_scenario_fingerprints=sorted(self.training_scenario_fingerprints), resume_supported=False, mechanism_only=self.mechanism_only, execution_scope='mechanism' if self.mechanism_only else 'manual_research', evaluation_use_allowed=not self.mechanism_only, formal_use_allowed=False, performance_passed=False)
        payload.update(map_contract.metadata(self.config),initial_model_state_sha256=self.initial_model_state_sha256)
        path = self.output / f'checkpoint_{label}.pt'
        if path.exists():
            raise FileExistsError(path)
        tmp = path.with_suffix('.tmp')
        torch.save(payload, tmp)
        tmp.replace(path)
        return path

    def run(self):
        training_admission(mechanism=self.mechanism_only, config=self.config)
        result = self._run_loop()
        require_review(self.review)
        result.update(schema=VERSION, method=self.config['method'], source_review=self.review, source_identity=self.source, mechanism_only=self.mechanism_only, execution_scope='mechanism' if self.mechanism_only else 'manual_research', performance_passed=False)
        result.update(map_contract.metadata(self.config),initial_model_state_sha256=self.initial_model_state_sha256)
        write_json(self.output / 'summary.json', result)
        return result

    def _perform_update(self):
        rl = self.config['rl']
        if len(self.replay) < rl['batch_size'] or self.global_step < rl['warmup_steps'] or self.global_step % rl['update_frequency']:
            return
        self.model.prep_training(device='cpu')
        for _ in range(rl['updates_per_train']):
            sample = self.replay.sample(rl['batch_size'])
            updates = []
            for i in range(4):
                (critic, actor, _) = self.model.update(sample, i)
                updates.append(dict(agent=i, actor_updated=self.model.last_actor_updated, critic_loss=critic, actor_loss=actor, **self.model.last_target_diagnostics))
            self.model.update_all_targets()
            self.optimizer_updates += 1
            self.metrics.append(dict(global_step=self.global_step, update=self.optimizer_updates, agents=updates))
        self.model.prep_rollouts(device='cpu')

    def _run_loop(self):
        threads = torch.get_num_threads()
        started = time.perf_counter()
        try:
            torch.set_num_threads(1)
            budget = self.config.get('max_total_environment_steps')
            while self.completed_main < self.config['total_main_trajectories']:
                if budget is not None and self.global_step >= budget:
                    break
                self.run_episode()
                print(f"[{self.config['method']}] episode={self.completed_main} steps={self.global_step}", flush=True)
                if self.completed_main % self.config['checkpoint_interval'] == 0:
                    self.save(f'episode_{self.completed_main:06d}')
            checkpoint = self.save('final') if self.completed_main else None
            self.episodes.materialize()
            self.metrics.materialize()
            result = dict(episodes=self.completed_main, environment_steps=self.global_step, optimizer_updates=self.optimizer_updates, actor_optimizer_steps=self.model.actor_updates, critic_optimizer_steps=8 * self.optimizer_updates, budget_policy='finish_started_episode', latest_checkpoint=None if checkpoint is None else str(checkpoint), wall_seconds=time.perf_counter() - started, performance_passed=False, formal_experiments_completed=False)
            write_json(self.output / 'summary.json', result)
            return result
        finally:
            torch.set_num_threads(threads)

    def _load_scenarios(self):
        if not self.config.get('scenario_manifest'):
            return None
        path = Path(self.config['scenario_manifest'])
        path = path if path.is_absolute() else ROOT / path
        if self.config.get('scenario_manifest_sha256') != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('training scenario manifest identity mismatch')
        manifest = json.loads(path.read_text(encoding='utf-8'))
        scenarios = manifest.get('scenarios', [])
        if not scenarios or manifest.get('scenario_profile', self.config['profile']) != self.config['profile'] or manifest.get('scenario_split', 'train') not in ('train', 'smoke_train') or (manifest.get('scenario_role', 'train') not in ('train', 'smoke_train')):
            raise ValueError('training requires a nonempty matching training manifest')
        identifiers = []
        for scenario in scenarios:
            identifiers.append(scenario.get('scenario_id'))
            if not scenario.get('scenario_id') or type(scenario.get('scenario_seed')) is not int or scenario.get('scenario_profile') != self.config['profile'] or (scenario.get('scenario_split') not in ('train', 'smoke_train')) or (scenario.get('scenario_role') != scenario.get('scenario_split')) or (scenario.get('max_steps') != self.config['max_steps']) or (scenario.get('protocol') not in ('ch3_unknown_map_v1', 'CH3_UNKNOWN_MAP_V1')):
                raise ValueError('training scenario identity/profile/role/horizon/protocol mismatch')
        if len(set(identifiers)) != len(identifiers) or manifest.get('scenario_count', len(scenarios)) != len(scenarios):
            raise ValueError('training manifest has duplicate scenario IDs or a count mismatch')
        if 'scenario_schedule' in self.config:
            from chapter3_bser.learning.scenarios import validate_manifest
            validate_manifest(manifest,self.config,evaluation=False)
            if any(i>=len(scenarios) for i in self.config['scenario_schedule']):raise ValueError('episode schedule index outside training manifest')
        return scenarios

    def _scenario(self, episode):
        seed = self.config['seed'] + episode
        if self.scenarios is None:
            manifests = build_scenario_manifests(count=1, generator_seed=seed, split='train', profiles=[self.config['profile']])
            scenario = manifests[self.config['profile']]['scenarios'][0]
            scenario['max_steps'] = self.config['max_steps']
        else:
            index = (self.config['scenario_schedule'][episode-1] if 'scenario_schedule' in self.config
                     else int(np.random.default_rng(seed).integers(len(self.scenarios))))
            scenario = self.scenarios[index]
        return (copy.deepcopy(scenario), seed)

def load_for_evaluation(path, *, allow_mechanism=False):
    review = verify_reviewed_changes()
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('schema') != VERSION or payload.get('artifact_schema') != 'auv.active.checkpoint.rebuild.v1' or payload.get('config_hash') != digest(payload['config']) or payload.get('source_review') != review:
        raise ValueError('not a source-verified v2 checkpoint')
    config = validate(payload['config'])
    map_contract.validate_checkpoint_map(config,payload)
    if type(payload.get('episodes')) is not int or not 1 <= payload['episodes'] <= config['total_main_trajectories'] or type(payload.get('global_step')) is not int or (not payload['episodes'] <= payload['global_step'] <= payload['episodes'] * config['max_steps']) or (type(payload.get('optimizer_updates')) is not int) or (payload['optimizer_updates'] < 0):
        raise ValueError('v2 checkpoint environment/episode counters disagree')
    mechanism = payload.get('mechanism_only')
    if (type(mechanism) is not bool or payload.get('formal_use_allowed') is not False
            or payload.get('evaluation_use_allowed') is not (not mechanism)
            or payload.get('execution_scope') != ('mechanism' if mechanism else 'manual_research')):
        raise ValueError('invalid checkpoint execution scope')
    if mechanism and (not allow_mechanism):
        raise ValueError('mechanism checkpoint is not admitted for formal evaluation')
    training_admission(mechanism=mechanism, config=config)
    require_source_match(payload['source_identity'], context='repair v2 evaluation')
    state = payload['model_state']
    if payload.get('model_state_sha256') != state_digest(state) or not _finite_state(state):
        raise ValueError('invalid or nonfinite v2 checkpoint state')
    with torch.random.fork_rng():
        model = build_model(config)
        model.load_training_state_dict(state)
    if model.niter != payload['optimizer_updates'] or model.actor_updates != payload['actor_optimizer_steps'] or payload['critic_optimizer_steps'] != 8 * model.niter:
        raise ValueError('checkpoint optimizer counters disagree')
    for (i, agent) in enumerate(model.agents):
        for (opt, count) in ((agent.policy_optimizer, model.actor_updates[i]), (agent.critic1_optimizer, model.niter), (agent.critic2_optimizer, model.niter)):
            params = [p for group in opt.param_groups for p in group['params']]
            if set(opt.state) != (set(params) if count else set()):
                raise ValueError('optimizer state inventory disagrees')
            for (p, item) in opt.state.items():
                if int(item['step']) != count or any((item[k].shape != p.shape for k in ('exp_avg', 'exp_avg_sq'))):
                    raise ValueError('optimizer steps/moment shapes disagree')
    model.prep_rollouts(device='cpu')
    require_review(review)
    return (model, config, payload)

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--mechanism-check', action='store_true')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.seed is not None:
        config['seed'] = args.seed
    config = validate(config)
    if not args.execute:
        try:
            review = training_admission(mechanism=args.mechanism_check, config=config)
            admitted = True
            reason = None
        except ValueError as exc:
            admitted = False
            reason = str(exc)
            review = None
        print(json.dumps(dict(executed=False, method=config['method'], admitted=admitted, reason=reason, source_review=review)))
        return
    print(json.dumps(Trainer(config, args.output, mechanism=args.mechanism_check).run(), indent=2))
