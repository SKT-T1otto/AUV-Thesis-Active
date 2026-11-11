"""Synthetic-only contracts. No environment factory, reset, rollout or restore."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from project_paths import ROOT
from chapter3_bser.path_option.options import DEFAULTS, validate
from chapter3_bser.path_option.path_context import Route, GoalManager, PATH_DIM
from chapter3_bser.path_option.networks import tensor_batch, masked_distribution
from chapter3_bser.path_option.learner import Learner, option_return
from chapter3_bser.reach_avoid.learner import first_event_cdf, bellman_events
from chapter3_bser.path_option.metrics import learning_assessment, event_diagnostics


def fixture():
    rng = np.random.default_rng(1)
    return dict(history=rng.normal(size=(16, 90)).astype(np.float32),
        paths=[rng.normal(size=(j+1, PATH_DIM)).astype(np.float32) for j in range(8)],
        goals=rng.normal(size=(8, 19)).astype(np.float32),
        actions=rng.uniform(-1, 1, (16, 3)).astype(np.float32),
        action_mask=np.array([True]*12+[False]*4), goal_mask=np.array([True]*7+[False]),
        event_relative=rng.normal(size=(8, 3)).astype(np.float32),
        position=np.ones(3), bounds=np.array([20., 20., 8.]), forced=-1,
        rule_goal=1, rule_terminate=False, periodic_terminate=False,
        forced_terminate=False, beta_allowed=True, intervention='none')


class PathOptionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_option_target_discount_once(self):
        y = option_return(torch.tensor([1., 1.]), torch.tensor([0., 1.]),
            torch.tensor([2., 2.]), torch.tensor([4., 4.]), torch.tensor([.25, .25]))
        self.assertTrue(torch.allclose(y, torch.tensor([3.375, 1.])))

    def test_competing_events_collision_wins(self):
        target = bellman_events(torch.ones(2, 32, 2)*.3, reached=torch.tensor([True, False]),
            collision=torch.tensor([True, False]), deadline=torch.tensor([False, True]))
        self.assertTrue(torch.equal(target[0], torch.tensor([[0., 1.]]).expand(32, -1)))
        self.assertEqual(float(target[1].sum()), 0.)

    def test_hazards_monotone_and_bounded(self):
        v = first_event_cdf(torch.randn(2, 96))
        self.assertTrue(bool((v.sum(-1) <= 1.000001).all()))
        self.assertTrue(bool((v[:, 1:]-v[:, :-1] >= -1e-6).all()))

    def test_mask(self):
        p = masked_distribution(torch.tensor([[10., 1.]]), torch.tensor([[False, True]]))
        self.assertEqual(p.tolist(), [[0., 1.]])
        with self.assertRaises(ValueError):
            masked_distribution(torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.bool))

    def test_padding_keeps_all_vertices(self):
        r = fixture(); r['paths'][0] = np.zeros((91, PATH_DIM), np.float32)
        b = tensor_batch([r])
        self.assertEqual(int(b['path_mask'][0, 0].sum()), 91)
        self.assertEqual(int(b['path_mask'][0, 7].sum()), 8)

    def test_route_progress_stable(self):
        r = Route([[0, 0, 0], [1, 0, 0], [1, 1, 0]], 1, 0, True)
        r.advance(np.zeros(3), .2)
        route_id = r.route_id; r.revision = 100
        r.advance(np.array([.1, 0, 0]), .2)
        self.assertEqual(r.index, 1); self.assertEqual(r.route_id, route_id)

    def test_cross_track_is_not_waypoint_distance(self):
        route = Route([[0., 0., 0.], [4., 0., 0.]], 0, 0, True)
        self.assertAlmostEqual(route.cross_track(np.array([2., 1., 0.])), 1.)

    def test_invalid_goal_not_hold(self):
        m = GoalManager(DEFAULTS); r = Route([[1, 1, 1]], 0, 0, False)
        m.commit(3, [1]*3, [1]*3, [1]*3, 0, 1, r, reason='INITIAL')
        self.assertEqual(m.status(3, np.ones(3), np.zeros(3), 1, 1, False), 'KNOWN_INVALID')

    def test_handoff_phase(self):
        m = GoalManager(DEFAULTS); r = Route([[0]*3], 0, 0, True)
        m.commit(3, [2]*3, [2]*3, [2]*3, 0, 1, r, reason='INITIAL')
        self.assertEqual(m.status(3, np.zeros(3), np.zeros(3), 2, 10, True), 'PHASE_CHANGED')

    def test_rules_hold_and_periodic(self):
        m = GoalManager(DEFAULTS)
        self.assertFalse(m.rule_terminate('HOLD', 500, periodic=True))
        self.assertTrue(m.rule_terminate('CONTINUE', 100, periodic=True))
        self.assertFalse(m.rule_terminate('CONTINUE', 100))

    def test_no_feasible_option_does_not_switch_every_step(self):
        m = GoalManager(DEFAULTS)
        self.assertEqual(m.termination_flags('NO_ROUTE', 300, 0, False), (False, False))
        self.assertEqual(m.termination_flags('NO_ROUTE', 300, 0, True), (True, False))

    def test_query_connector_does_not_change_physical_positions(self):
        from dataclasses import dataclass
        from unittest.mock import patch
        from chapter3_bser.path_option.path_context import public_route
        @dataclass
        class Public:
            positions: tuple
            navigation_targets: tuple
        original = Public(((1., 1., 1.),)*4, ((3., 3., 3.),)*4)
        env = SimpleNamespace(get_agent_state=lambda: original)
        def extract(view):
            p = view.get_agent_state().positions[0]
            return SimpleNamespace(agents=[SimpleNamespace(agent_id=0, position=p)])
        with patch('chapter3_bser.path_option.path_context.extract_planning_state', side_effect=extract), patch('chapter3_bser.path_option.path_context.TravelCostService') as service:
            public_route(env, 0, [4.]*3, query_start=[2.]*3)
            self.assertEqual(service.return_value.query.call_args.args[0], (2., 2., 2.))
        self.assertEqual(env.get_agent_state().positions[0], (1., 1., 1.))

    def test_causal_prefix(self):
        model = Learner(); x = torch.randn(1, 16, 90)
        with torch.no_grad():
            before = model.policy.encoder.history_encoding(x)
            x[:, 9:] += 100
            after = model.policy.encoder.history_encoding(x)
        self.assertTrue(torch.allclose(before[:, :9], after[:, :9], atol=1e-6))

    def test_action_permutation_equivariance(self):
        model = Learner(); r = fixture(); permutation = np.arange(16)[::-1].copy()
        q = copy.deepcopy(r); q['actions'] = q['actions'][permutation].copy(); q['action_mask'] = q['action_mask'][permutation].copy()
        with torch.no_grad():
            a = model.policy(tensor_batch([r]), torch.zeros(1, 8, 2))[0]
            b = model.policy(tensor_batch([q]), torch.zeros(1, 8, 2))[0]
        self.assertTrue(torch.allclose(a[:, :, permutation], b, atol=1e-6))

    def test_synthetic_update_all_primary_variants(self):
        # ONE complete update per temporary model (well below the four-update cap).
        for variant in ('A', 'B', 'C', 'D'):
            o = dict(DEFAULTS, batch_size=2, warmup=2)
            model = Learner(o, variant)
            for k in range(2):
                r = fixture()
                model.add(dict(state=r, next=copy.deepcopy(r), goal_index=k,
                    actual_action=r['actions'][k], reward=.1, done=False, reached=False,
                    collision=False, event_deadline=False, event_target=np.array([2., 2., 2.]),
                    policy_executed=True, decision=True))
            before = [p.detach().clone() for p in model.policy.parameters()]
            result = model.update()
            self.assertEqual(result['updates'], 1)
            self.assertTrue(np.isfinite(result['actor_loss']))
            self.assertTrue(any(not torch.equal(p, q) for p, q in zip(before, model.policy.parameters())))

    def test_forced_action_actor_loss_mask(self):
        model = Learner(); r = fixture(); r['forced'] = 8
        low, _, _, _, _, _, _ = model.distributions(tensor_batch([r]), [r])
        self.assertTrue(torch.equal(low[0, :, 8], torch.ones(8)))
        self.assertEqual(float(low[0, :, :8].sum().detach()), 0.)

    def test_rule_ablation_controls_beta(self):
        model = Learner(variant='C'); r = fixture(); r['rule_terminate'] = True
        result = model.distributions(tensor_batch([r]), [r])
        self.assertEqual(float(result[2][0, 0].detach()), 1.)

    def test_no_ra_zeroes_decision_events(self):
        model = Learner(variant='D'); r = fixture()
        self.assertEqual(float(model.distributions(tensor_batch([r]), [r])[-1].abs().sum().detach()), 0.)

    def test_no_path_and_no_history_are_actual_interventions(self):
        r = fixture(); a = tensor_batch([r], 'no_path'); b = tensor_batch([r], 'no_history')
        self.assertEqual(float(a['paths'].abs().sum()), 0.)
        self.assertTrue(torch.equal(b['history'][:, 0], b['history'][:, -1]))

    def test_config_identity(self):
        with self.assertRaises(ValueError):
            validate(dict(DEFAULTS, gamma=.99))

    def test_no_execute_no_files_or_runtime(self):
        from tools.path_option_experiment import main
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as d:
            path = Path(d)/'not_created'
            result = main(['prepare', '--output', str(path)])
            self.assertFalse(result['executed']); self.assertFalse(path.exists())

    def test_report_not_convergence_from_loss(self):
        curve = [dict(success=.2, collision=0., timeout=.8, penalized_steps=400.) for _ in range(4)]
        self.assertEqual(learning_assessment(curve)['status'], 'stable_but_ineffective')

    def test_old_or_missing_metadata_cannot_restore(self):
        from chapter3_bser.path_option.workflow import load_policy
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as d:
            p = Path(d)/'old.pt'
            p.write_bytes(b'not weights')
            with self.assertRaises(FileNotFoundError):
                load_policy(p, [], {}, config_hash='x')

    def test_duplicate_physical_splits_rejected(self):
        from unittest.mock import patch
        from chapter3_bser.path_option.workflow import check_splits
        with patch('chapter3_bser.path_option.workflow.scenario_fingerprint', return_value='same'):
            with self.assertRaises(ValueError):
                check_splits([dict(scenario_id='a')], [dict(scenario_id='b')], [])

    def test_install_preserves_upstream_and_full_path(self):
        from dataclasses import dataclass
        from unittest.mock import patch
        from chapter3_bser.path_option.runtime import PathOptionRuntime
        @dataclass
        class Assignment:
            agent_id: int
            final_waypoint: tuple = (3., 3., 3.)
            planned_path: tuple = ()
            tracking_waypoint: tuple = (1., 1., 1.)
            hold_position: tuple = (3., 3., 3.)
            hold_state: bool = False
            reachable: bool = True
            target_region: tuple = (3., 3., 3.)
        @dataclass
        class Guidance:
            agent_assignments: tuple
            executor_assignment: Assignment
        runtime = PathOptionRuntime.__new__(PathOptionRuntime)
        runtime.o = dict(DEFAULTS); runtime.manager = GoalManager(DEFAULTS)
        installed = []; runtime.env = SimpleNamespace(install_guidance=installed.append,
            get_task_state=lambda: SimpleNamespace(step=0))
        runtime.guard = SimpleNamespace(previous_v=None)
        runtime.guidance = Guidance(tuple(Assignment(i) for i in range(4)), Assignment(3))
        upstream = runtime.guidance; states = []
        runtime._candidates = {}
        for i in range(4):
            route = Route([[1., 1., 1.], [1., 3., 1.], [3., 3., 3.]], 1, 0, True)
            runtime.manager.commit(i, [3.]*3, [3.]*3, [3.]*3, 0, 0, route, reason='INITIAL')
            s = fixture(); s.update(position=np.ones(3), velocity=np.zeros(3), phase=0,
                original=np.full(3, 3.), task=np.full(3, 3.), reason='CONTINUE')
            states.append(s); runtime._candidates[i] = [(np.full(3, 3.), route)]*8
        runtime.observe = lambda: states
        runtime.learner = None
        with patch('chapter3_bser.learning.runtime.RepairedMissionRuntime._actions_for_step', return_value=torch.zeros(4, 3)):
            commands = runtime._actions_for_step(torch.zeros(4, 3))
        self.assertEqual(tuple(commands.shape), (4, 3))
        self.assertIs(runtime.guidance, upstream)
        self.assertEqual(len(installed[0].agent_assignments[0].planned_path), 3)

    def test_forced_samples_do_not_train_candidate_action_head(self):
        o = dict(DEFAULTS, batch_size=2, warmup=2)
        model = Learner(o)
        before = [p.detach().clone() for p in model.policy.action.parameters()]
        for _ in range(2):
            r = fixture(); r['forced'] = 8; r['beta_allowed'] = False
            model.add(dict(state=r, next=copy.deepcopy(r), goal_index=0,
                actual_action=r['actions'][8], reward=.1, done=False, reached=False,
                collision=False, event_deadline=False, event_target=np.array([2., 2., 2.]),
                policy_executed=False, decision=False))
        model.update()  # ONE update for this temporary model.
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, model.policy.action.parameters())))

    def test_termination_gradient_sign(self):
        # If continuing is better, gradient descent must lower beta.
        logit = torch.tensor(0., requires_grad=True)
        (torch.sigmoid(logit)*torch.tensor(2.)).backward()
        self.assertGreater(float(logit.grad), 0.)

    def test_unknown_endpoint_not_falsely_occupied(self):
        from chapter3_bser.path_option.runtime import observed_occupied
        state = SimpleNamespace(grid=SimpleNamespace(origin=np.zeros(3), spacing=np.ones(3), shape=(2, 2, 2)),
            occupancy=SimpleNamespace(occupied_mask=np.zeros(8, bool)))
        self.assertFalse(observed_occupied(state, [.5]*3))
        self.assertTrue(observed_occupied(state, [-1.]*3))

    def test_empty_reach_diagnostics_not_calibration(self):
        r = event_diagnostics([])
        self.assertIsNone(r['brier']); self.assertFalse(r['calibrated_probability_claimed'])

    def test_frozen_plan_rejects_tampered_inputs(self):
        from unittest.mock import patch
        from tools.path_option_experiment import frozen
        from chapter3_bser.path_option.source import sha
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as d:
            root = Path(d); (root/'config.json').write_text('{}', encoding='utf-8')
            plan = dict(source={'x': 1}, inputs={'config.json': sha(root/'config.json')})
            (root/'plan.json').write_text(json.dumps(plan), encoding='utf-8')
            with patch('tools.path_option_experiment.identity', return_value={'x': 1}):
                frozen(root)
                (root/'config.json').write_text('{"changed": true}', encoding='utf-8')
                with self.assertRaises(ValueError):
                    frozen(root)

    def test_all_supplemental_networks_and_rule_selection(self):
        for variant in ('fixed_query', 'periodic', 'risk_only', 'rule_recovery'):
            m = Learner(variant=variant); r = fixture(); r['periodic_terminate'] = True
            low, mu, beta, *_ = m.distributions(tensor_batch([r]), [r])
            self.assertTrue(bool(torch.isfinite(low).all()))
            if variant == 'periodic':
                self.assertEqual(float(beta[0, 0].detach()), 1.)
            if variant == 'rule_recovery':
                self.assertEqual(float(mu[0, 1].detach()), 1.)

    def test_public_sensor_to_policy_integration_without_environment(self):
        from unittest.mock import patch
        from checks.reach_avoid_fixtures import public_controller
        from chapter3_bser.path_option.runtime import PathOptionRuntime
        from chapter3_bser.reach_avoid.features import Encoder
        from chapter3_bser.reach_avoid.scan_geometry import RaySubcells
        controller, _, state, contract = public_controller()
        state.occupancy.occupancy_probability = np.ones(800)*.1
        runtime = PathOptionRuntime.__new__(PathOptionRuntime)
        runtime.o = dict(DEFAULTS); runtime.manager = GoalManager(DEFAULTS)
        runtime.guard = controller.guard; runtime.subcells = RaySubcells(contract['bounds'])
        runtime.encoder = Encoder(dict(history=16), 400)
        runtime.faults = []; runtime._cached = None; runtime._alternatives = {}; runtime._last_candidates = {}
        runtime._candidate_goal_ids = {}; runtime._requested_goal_ids = set(); runtime._candidates = {}
        runtime._request_refresh = False; runtime.intervention = 'none'; runtime.observations = np.zeros((4, 28))
        runtime.guidance = SimpleNamespace(agent_assignments=tuple(SimpleNamespace(agent_id=i, final_waypoint=(9., 3., 4.)) for i in range(4)))
        runtime.env = SimpleNamespace(get_task_state=lambda: SimpleNamespace(step=0, executor_knows_target=False),
            read_public_events=lambda cursor: (), get_public_delivered_target=lambda: None)
        def query(i, point, s):
            p = np.array(s.agents[i].position)
            return Route([p, p+[0., 1., 0.], point], 1, 0, True,
                distance=float(np.linalg.norm(point-p)), estimated_seconds=5.)
        runtime._query = query
        with patch('chapter3_bser.path_option.runtime.extract_planning_state', return_value=state):
            rows = runtime.observe()
            self.assertIs(rows, runtime.observe())
        self.assertEqual(len(runtime.encoder.history[0]), 16)  # Initial past is explicitly left-padded.
        model = Learner()
        for row in rows:
            self.assertEqual(row['history'].shape, (16, 90))
            choice = model.decide(row)
            self.assertTrue(row['action_mask'][choice['action']])

    def test_inactive_transitions_keep_team_value_but_not_actor_labels(self):
        from unittest.mock import patch
        from chapter3_bser.path_option.runtime import PathOptionRuntime
        runtime = PathOptionRuntime.__new__(PathOptionRuntime)
        runtime.o = dict(DEFAULTS); runtime._request_refresh = False
        runtime.guard = SimpleNamespace(previous=None)
        runtime.transitions = []; runtime.trace = []
        runtime.env = SimpleNamespace(unwrapped=SimpleNamespace(_agent_acc=torch.zeros(4, 3), last_collision_records=[]))
        runtime._pending = []; nexts = []
        for _ in range(4):
            s = fixture()
            s.update(phase=3, velocity=np.zeros(3), event_targets=np.ones((8, 3)),
                target_velocity=np.zeros(3), free_pose=True, clearance=2.,
                original=np.ones(3), committed_original=np.ones(3), endpoint=np.ones(3),
                age=0, map_revision=1, reason='CONTINUE', cross_track=0., estimated_seconds=0., emergency=False)
            s['forced'] = 8
            d = dict(goal=0, action=8, terminate=False, executed_commitment=dict(task=[1.]*3, endpoint=[1.]*3,
                goal_id=1, route_id='route', mode='TASK', route_index=0, route_revision=1, cross_track=0.))
            runtime._pending.append((s, d)); nexts.append(s)
        runtime.observe = lambda: nexts
        with patch('chapter3_bser.learning.runtime.RepairedMissionRuntime.advance', return_value=dict(t=0, team_reward=.5)):
            runtime.advance(None)
        self.assertEqual(len(runtime.transitions), 4)
        self.assertTrue(all(t['reward'] == .5 for t in runtime.transitions))
        self.assertTrue(all(not t['policy_executed'] and t['event_deadline'] and not t['reached'] for t in runtime.transitions))

    def test_training_and_evaluation_execute_same_categorical_policy(self):
        a = Learner(); b = Learner(); r = fixture(); r['beta_allowed'] = False
        training = [a.decide(r, explore=True)['action'] for _ in range(20)]
        evaluation = [b.decide(r, explore=False)['action'] for _ in range(20)]
        self.assertEqual(training, evaluation)
        self.assertGreater(len(set(evaluation)), 1)

    def test_stable_good_prior_is_not_learning_gain(self):
        curve = [dict(episodes=12, success=.9, collision=0., timeout=.1, penalized_steps=200.,
            team_discounted_return=.2, updates=0) for _ in range(4)]
        result = learning_assessment(curve)
        self.assertEqual(result['status'], 'stable_good_policy_without_clear_learning_gain')
        self.assertTrue(result['return_stable']); self.assertFalse(result['online_task_gain'])

    def test_incomplete_evaluation_remains_in_report(self):
        from tools.path_option_report import build_report
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as d:
            root = Path(d); incomplete = root/'eval_interrupted'; incomplete.mkdir()
            (incomplete/'evaluation_request.json').write_text(json.dumps(dict(expected_scenes=100)), encoding='utf-8')
            (incomplete/'metrics.jsonl').write_text(json.dumps(dict(scenario_id='one', success=0))+'\n', encoding='utf-8')
            build_report(root, root/'report', dict(primary_arms=[], seeds=[]))
            r = json.loads((root/'report/report.json').read_text(encoding='utf-8'))
            self.assertEqual(r['incomplete_evaluations'][0]['committed_episodes'], 1)
            self.assertFalse(r['performance_accepted'])

    def test_inactive_team_return_does_not_train_event_heads(self):
        model = Learner(dict(DEFAULTS, batch_size=2, warmup=2))
        before = [p.detach().clone() for p in model.critics[0].events.parameters()]
        for _ in range(2):
            r = fixture(); r.update(phase=3, forced=8, beta_allowed=False)
            model.add(dict(state=r, next=copy.deepcopy(r), goal_index=0,
                actual_action=r['actions'][8], reward=.5, done=False, reached=False,
                collision=True, event_deadline=True, event_target=np.array([2., 2., 2.]),
                policy_executed=False, decision=False))
        model.update()  # ONE complete update on this temporary model.
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, model.critics[0].events.parameters())))


if __name__ == '__main__':
    unittest.main()
