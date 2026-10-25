"""Observer causality, bounded storage and numerical transparency; no episodes."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
from chapter3_bser.experiments.coupled_search_v1.diagnostics import StageAudit,ActivityAudit,observe_planning
from chapter3_bser.experiments.coupled_search_v1.bounded_belief import BoundedCoupledBelief
from chapter3_bser.experiments.coupled_search_v1.numerics import BudgetExceeded
from chapter3_bser.learning.map_diagnostics import tables
from checks.test_bounded_map import analytic_belief,evidence


def allocation():
    item=NS(agent_id=0,candidate_id='candidate',path=((1.,2.,3.),),waypoint=(1.,2.,3.))
    return NS(search_assignments=(item,))


def guidance(step=0,*,modified=False,hold=False):
    item=NS(agent_id=0,assignment_id='candidate',planned_path=((2.,2.,3.),) if modified else ((1.,2.,3.),),
            final_waypoint=(1.,2.,3.),hold_state=hold,reachable=not hold)
    return NS(step=step,agent_assignments=(item,))


def observation(before=False,after=False):
    return NS(found_before=before,found_after=after,agent_start=np.zeros((4,3)),agent_end=np.ones((4,3)))


class MapDiagnostics(unittest.TestCase):
    def test_failed_attempt_keeps_initial_cause_and_final_failure(self):
        a=StageAudit(clock=lambda:1.)
        a.begin(17,'inference');a.stage('target_update');a.note_failure('zero_mass')
        a.stage('window_replay');a.note_failure('wall_clock_budget');a.finish(error='wall_clock_budget')
        r=a.trace[-1]
        self.assertFalse(r['completed']);self.assertEqual(r['first_exception']['reason'],'zero_mass')
        self.assertEqual(r['failure_phase'],'window_replay');self.assertEqual(a.counts['failed'],1)

    def test_recovered_attempt_is_completed_without_hiding_trigger(self):
        a=StageAudit();a.begin(1,'inference');a.note_failure('zero_mass');a.finish()
        self.assertTrue(a.trace[-1]['completed']);self.assertEqual(a.trace[-1]['first_exception']['reason'],'zero_mass')
        self.assertFalse(a.failure_reasons)

    def test_attempt_trace_is_bounded_but_aggregate_count_is_complete(self):
        a=StageAudit(capacity=2)
        for i in range(5):a.begin(i,'inference');a.finish(error='failure' if i%2 else None)
        r=a.summary();self.assertEqual(len(r['trace']),2);self.assertEqual(r['counts']['attempts'],5)
        self.assertTrue(r['trace_truncated']);self.assertEqual(r['counts']['failed'],2)

    def test_stage_totals_measure_failed_work(self):
        ticks=iter([0.,1.,3.,4.]);a=StageAudit(clock=lambda:next(ticks))
        a.begin(1,'proposal');a.stage('route_generation');a.finish(error='wall_clock_budget')
        self.assertEqual(a.phase_seconds['route_generation'],2.)
        self.assertEqual(a.trace[-1]['seconds'],4.)

    def test_planning_error_is_propagated_and_recorded(self):
        error=BudgetExceeded('wall_clock_budget')
        class Planner:
            planning_diagnostics=StageAudit()
            @observe_planning('proposal')
            def run(self,state):raise error
        p=Planner()
        with self.assertRaises(BudgetExceeded) as caught:p.run(NS(target_found=False,step=4))
        self.assertIs(caught.exception,error);self.assertEqual(p.planning_diagnostics.counts['failed'],1)

    def test_rejected_proposal_is_not_counted_as_installed_core_guidance(self):
        a=ActivityAudit();a.proposed(allocation(),'coupled',1);a.install(guidance(1))
        a.advance(observation(),2)
        self.assertEqual(a.counts['unknown_agent_steps'],3)
        self.assertEqual(a.counts['accepted_coupled_proposals'],0)
        self.assertEqual(a.summary()['coupled_exact_agent_step_fraction'],0.)

    def test_initial_guidance_execution_uses_next_transition(self):
        a=ActivityAudit();a.proposed(allocation(),'coupled',0);a.install(guidance(),allocation())
        self.assertEqual(a.counts['search_physical_steps'],0)
        self.assertEqual(a.counts['accepted_coupled_proposals'],1)
        a.advance(observation(after=True),1);a.advance(observation(before=True,after=True),2)
        self.assertEqual(a.counts['search_physical_steps'],1);self.assertEqual(a.counts['execution_physical_steps'],1)
        self.assertEqual(a.counts['coupled_exact_agent_steps'],1)
        self.assertEqual(a.summary()['coupled_exact_agent_step_fraction'],1/3)

    def test_safety_modified_path_is_not_reported_as_an_exact_core_path(self):
        a=ActivityAudit();a.proposed(allocation(),'coupled',1)
        a.decision(1,NS(replanned=True,allocation=allocation(),decision_reason='accepted'))
        self.assertEqual(a.counts['accepted_coupled_proposals'],1)
        a.install(guidance(1,modified=True));a.advance(observation(),2)
        self.assertEqual(a.counts['coupled_modified_agent_steps'],1)
        self.assertEqual(a.counts['coupled_exact_agent_steps'],0)

    def test_holds_are_separate_even_if_candidate_identity_matches(self):
        a=ActivityAudit();a.proposed(allocation(),'coupled',0);a.install(guidance(hold=True),allocation())
        a.advance(observation(),1);self.assertEqual(a.counts['hold_agent_steps'],1)

    def test_d2_guidance_is_distinguishable_from_core_guidance(self):
        a=ActivityAudit();a.proposed(allocation(),'d2',0);a.install(guidance(),allocation())
        a.advance(observation(),1);self.assertEqual(a.counts['d2_exact_agent_steps'],1)

    def test_repeated_fallback_symptom_does_not_replace_first_cause(self):
        a=ActivityAudit();a.fallback(3,'wall_clock_budget');a.fallback(3,'core_failed_this_step')
        a.fallback(4,'missing_public_safe_route');r=a.summary()
        self.assertEqual(r['fallback_search_steps'],2)
        self.assertEqual(r['first_fallback_reasons'],{'wall_clock_budget':1,'missing_public_safe_route':1})

    def test_duplicate_transition_does_not_double_execution_counts(self):
        a=ActivityAudit(capacity=2)
        for i in range(1,6):a.advance(observation(),i);a.advance(observation(),i)
        self.assertEqual(a.counts['search_agent_steps'],15);self.assertEqual(len(a.trace),2)
        self.assertTrue(a.summary()['trace_truncated'])

    def test_observed_filter_preserves_arrays_and_rng_through_recovery(self):
        left=analytic_belief();right=analytic_belief()
        raw=BoundedCoupledBelief.consume.__wrapped__
        for step in range(1,41):
            negative=step in (1,32)
            if step==32:left.positions[:]=[2.,1.,1.];right.positions[:]=[2.,1.,1.]
            p=evidence(step,negative=negative);left.consume(p);raw(right,p)
            self.assertEqual(left.valid,right.valid)
            for name in ('positions','velocities','alpha','omega'):
                np.testing.assert_array_equal(getattr(left,name),getattr(right,name))
            self.assertEqual(left.rng.bit_generator.state,right.rng.bit_generator.state)
        self.assertGreater(left.counts['local_recoveries'],0)

    def test_timeout_after_zero_mass_keeps_both_diagnostic_causes(self):
        b=analytic_belief();b.positions[:]=[2.,1.,1.]
        with patch.object(b,'_recover',side_effect=BudgetExceeded('wall_clock_budget')):
            b.consume(evidence(1,negative=True))
        r=b.inference_diagnostics.trace[-1]
        self.assertFalse(b.valid);self.assertEqual(r['error'],'wall_clock_budget')
        self.assertEqual(r['first_exception']['reason'],'bounded_target_zero_mass')

    def test_historical_missing_diagnostics_are_unavailable_not_zero(self):
        row=dict(scenario_id='x',actual_length=400,found_step=13,success=False,
            termination_reason='timeout',pre_found_collision=False,
            map_metrics=dict(search_batches=13,core_scored_steps=12,core_proposals=2,d2_fallbacks=1))
        r=tables([('NEW_B2',2729,[row])])['participation'][0]
        self.assertEqual((r['search_physical_steps'],r['execution_physical_steps']),(13,387))
        self.assertIsNone(r['coupled_exact_fraction']);self.assertFalse(r['diagnostics_available'])

    def test_probe_dry_run_creates_no_output(self):
        from tools.check_map_diagnostics import main
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            path=Path(directory)/'absent';r=main(['--output',str(path)])
            self.assertFalse(r['executed']);self.assertFalse(path.exists())

    def test_probe_variants_preserve_kernel_sizes(self):
        import json
        from project_paths import ROOT
        from tools.check_map_diagnostics import cases
        matrix=json.loads((ROOT/'configs/bounded_design_experiment_v1.json').read_text(encoding='utf-8'))
        result=cases(matrix);self.assertEqual(len(result),12)
        for _,_,c in result:self.assertEqual((c['maps'],c['particles_per_map'],c['horizon']),(3,32,16))

    def test_selected_zero_report_needs_no_training_or_unselected_results(self):
        from checks.test_map_suite import fixture
        from checks.test_map_convergence import evaluation_rows
        from chapter3_bser.learning import map_report,map_suite
        with tempfile.TemporaryDirectory() as directory:
            root,plan=fixture(directory)
            for job in plan['jobs']:
                if job['seed'] is None:
                    evaluation_rows(root,plan,job,map_suite.read(root/'inputs/evaluation.json'),root/job['evaluation'],'zero',0)
            with patch.object(map_report,'training_receipt',side_effect=AssertionError('no trained policy required')):
                r=map_report.summarize(root,root/'zero_report',groups=['OLD_ZERO','NEW_ZERO'])
            self.assertTrue(r['complete']);self.assertFalse(r['all_scheduled_groups_reported'])
            self.assertEqual(len(r['groups']),2);self.assertTrue((root/'zero_report/participation.csv').exists())

    def test_unknown_or_duplicate_report_groups_are_rejected(self):
        from chapter3_bser.learning import map_report
        with patch.object(map_report,'load_plan',return_value={'spec':{'groups':['OLD_ZERO']}}):
            for groups in ([],['NEW_ZERO'],['OLD_ZERO','OLD_ZERO']):
                with self.assertRaisesRegex(ValueError,'report groups'):
                    map_report.summarize('unused','unused_output',groups=groups)

    def test_selected_group_summary_dry_run_does_not_touch_outputs(self):
        from chapter3_bser.learning.map_suite import main
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            path=Path(directory)/'absent'
            r=main(['summarize','--root','absent_plan','--output',str(path),'--groups','OLD_ZERO','NEW_ZERO'])
            self.assertFalse(r['executed']);self.assertFalse(path.exists())


if __name__=='__main__':unittest.main()
