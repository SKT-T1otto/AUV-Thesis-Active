"""Compare compute reuse with hash-pinned pre-change code, never a simulator."""
import copy
import hashlib
import json
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from chapter3_bser.path_option.learner import Learner
from chapter3_bser.path_option.networks import tensor_batch
from chapter3_bser.path_option.options import DEFAULTS, VARIANTS, INTERVENTIONS
from checks.test_path_option import fixture
from project_paths import ROOT

BASELINE = ROOT/'verification/path_option_compute_v1/20261112_v1'
PINNED = {'chapter3_bser/path_option/networks.py': 'b06794bcf1fff5ef65bc6652af6fbd9c61f473fa1cdafe3276a2495099882647',
          'chapter3_bser/path_option/learner.py': 'e56f0d6300412cf928c0bdcaf8c3b7722dc2d346b2d3ac1155b509c1d825b12e'}
ERRORS = {}


def reference():
    record = json.loads((BASELINE/'before.json').read_text(encoding='utf-8'))
    modules = []
    for name in ('networks', 'learner'):
        relative = f'chapter3_bser/path_option/{name}.py'
        raw = (BASELINE/'before'/relative).read_bytes()
        if (hashlib.sha256(raw).hexdigest() != PINNED[relative]
                or record['snapshots'][relative] != PINNED[relative]):
            raise AssertionError('frozen reference changed: '+relative)
        module = types.ModuleType('frozen_compute_reference_'+name)
        source = raw.decode('utf-8')
        if name == 'learner':
            source = source.replace('from chapter3_bser.path_option.networks import Policy, Critic, tensor_batch, masked_distribution', '')
            module.__dict__.update({k: getattr(modules[0], k) for k in
                                   ('Policy', 'Critic', 'tensor_batch', 'masked_distribution')})
        exec(compile(source, str(BASELINE/'before'/relative), 'exec'), module.__dict__)
        modules.append(module)
    return modules[-1].Learner


def close(a, b, category, path=''):
    """Compare every leaf, including absent gradients and Adam counters."""
    if isinstance(a, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            raise AssertionError((category, path, 'shape/dtype'))
        error = float((a-b).abs().max()) if a.numel() else 0.
        ERRORS[category] = max(ERRORS.get(category, 0.), error)
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7, msg=f'{category}:{path}')
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise AssertionError((category, path, 'keys'))
        for k in a:
            close(a[k], b[k], category, path+'/'+str(k))
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError((category, path, 'length'))
        for i, (x, y) in enumerate(zip(a, b)):
            close(x, y, category, path+'/'+str(i))
    elif isinstance(a, float):
        ERRORS[category] = max(ERRORS.get(category, 0.), abs(a-b))
        if not np.isclose(a, b, rtol=1e-5, atol=1e-7):
            raise AssertionError((category, path, a, b))
    elif a != b:
        raise AssertionError((category, path, a, b))


def pair(variant='A'):
    return reference()(dict(DEFAULTS), variant), Learner(dict(DEFAULTS), variant)


def rows(count=8, length=None):
    rng = np.random.default_rng(7001)
    result = []
    for i in range(count):
        r = fixture()
        r['history'] = rng.normal(size=(16, 90)).astype(np.float32)
        r['goals'] = rng.normal(size=(8, 19)).astype(np.float32)
        r['paths'] = [rng.normal(size=((length if length else 1+(i+j)*3), 13)).astype(np.float32) for j in range(8)]
        r['forced'] = 8 if i % 4 == 1 else -1
        r['beta_allowed'] = i % 4 != 2
        r['forced_terminate'] = i % 2 == 0
        r['rule_terminate'] = i % 2 == 1
        r['periodic_terminate'] = i % 3 == 0
        r['phase'] = 3 if i % 4 == 3 else i % 3
        result.append(r)
    return result


def fill(model, states):
    for k in range(128):
        r = copy.deepcopy(states[k % len(states)])
        nxt = copy.deepcopy(states[(k+1) % len(states)])
        model.add(dict(state=r, next=nxt, goal_index=k % 7,
            actual_action=r['actions'][k % 12].copy(), reward=float((k % 7)-3)/10,
            done=k % 11 == 0, reached=k % 13 == 0, collision=k % 17 == 0,
            event_deadline=k % 5 == 0, event_target=np.array([2., 3., 4.]),
            policy_executed=k % 3 != 0, decision=k % 2 == 0))


def captured_update(model):
    gradients = []
    clip = torch.nn.utils.clip_grad_norm_
    def capture(parameters, *args, **kwargs):
        parameters = list(parameters)
        before = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
        norm = clip(parameters, *args, **kwargs)
        after = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
        gradients.append(dict(before=before, after=after, norm=norm))
        return norm
    with patch('torch.nn.utils.clip_grad_norm_', side_effect=capture):
        loss = model.update()
    return loss, gradients


def model_state(model):
    return dict(policy=model.policy.state_dict(), critics=model.critics.state_dict(),
        target_policy=model.target_policy.state_dict(), target_critics=model.target_critics.state_dict(),
        actor_optimizer=model.actor_optimizer.state_dict(), critic_optimizer=model.critic_optimizer.state_dict(),
        rng=model.rng.bit_generator.state, updates=model.updates, options=model.o)


class ComputeEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_outputs_all_variants_interventions_modes(self):
        for variant in VARIANTS:
            old, new = pair(variant)
            close(model_state(old), model_state(new), 'initial_state')
            for training in (True, False):
                for model in (old, new):
                    model.policy.train(training); model.critics.train(training)
                    model.target_policy.train(training); model.target_critics.train(training)
                for intervention in INTERVENTIONS:
                    rr = rows()
                    for r in rr:
                        r['intervention'] = intervention
                    batch = tensor_batch(rr, intervention)
                    for target in (False, True):
                        with torch.no_grad():
                            a = old.distributions(batch, rr, target=target)
                            b = new.distributions(batch, rr, target=target)
                        close(a, b, 'outputs')

    def test_four_updates_gradients_parameters_adam_and_rng(self):
        for variant in VARIANTS:
            old, new = pair(variant)
            for model in (old, new):
                fill(model, rows())
            for step in range(4):
                rng = torch.get_rng_state().clone()
                a, ag = captured_update(old)
                old_rng = torch.get_rng_state().clone()
                torch.set_rng_state(rng)
                b, bg = captured_update(new)
                close(old_rng, torch.get_rng_state(), 'global_rng')
                close(a, b, 'losses')
                close(ag, bg, 'gradients')
                close(model_state(old), model_state(new), 'updated_state')
                self.assertEqual(new.updates, step+1)

    def test_sampled_decisions_and_rng_order(self):
        for variant in VARIANTS:
            old, new = pair(variant)
            for intervention in INTERVENTIONS:
                for r in rows():
                    r['intervention'] = intervention
                    a = old.decide(r); b = new.decide(r)
                    close(a, b, 'sampled_decisions')
                    close(old.rng.bit_generator.state, new.rng.bit_generator.state, 'sampling_rng')

    def test_encoder_call_counts(self):
        counts = []
        for model in pair():
            count = [0]
            def increment(*args):
                count[0] += 1
            handles = [m.encoder.register_forward_hook(increment) for m in
                       [model.policy, *model.critics, model.target_policy, *model.target_critics]]
            try:
                rr = rows()
                with torch.no_grad():
                    model.distributions(tensor_batch(rr), rr)
                inference = count[0]; count[0] = 0
                fill(model, rr); model.update()
                counts.append((inference, count[0]))
            finally:
                for handle in handles:
                    handle.remove()
        self.assertEqual(counts, [(4, 16), (3, 9)])


if __name__ == '__main__':
    unittest.main()
