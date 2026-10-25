"""Offline phase and participation reports from verified native episode rows."""
import csv
import json


def tables(sets):
    participation=[];attempts=[];fallbacks=[]
    for group,seed,rows in sets:
        for row in rows:
            health=row['map_metrics'];diagnostic=health.get('diagnostics') or {}
            activity=diagnostic.get('activity');counts={} if activity is None else activity['counts']
            found=row['found_step'];length=row.get('actual_length')
            search=None if length is None else length if found is None else min(length,found)
            base=dict(group=group,training_seed=seed,scenario_id=row['scenario_id'])
            batches=health.get('search_batches');eligible=None if batches is None else max(0,batches-1)
            participation.append(dict(base,actual_length=length,found_step=found,success=row['success'],
                termination_reason=row['termination_reason'],pre_found_collision=row['pre_found_collision'],
                search_physical_steps=search,execution_physical_steps=None if length is None else length-search,
                diagnostics_available=activity is not None,
                search_batches=batches,core_scored_steps=health.get('core_scored_steps'),
                eligible_scored_steps=eligible,
                scored_coverage=None if not eligible or health.get('core_scored_steps') is None else health['core_scored_steps']/eligible,
                core_proposals=health.get('core_proposals'),fallback_records=health['d2_fallbacks'],
                accepted_coupled_proposals=counts.get('accepted_coupled_proposals',0) if activity else None,
                proposal_acceptance_fraction=None if not activity or not health.get('core_proposals') else
                    counts.get('accepted_coupled_proposals',0)/health['core_proposals'],
                fallback_decision_steps=None if activity is None else activity['fallback_search_steps'],
                fallback_decision_fraction=None if activity is None or not batches else
                    activity['fallback_search_steps']/batches,
                search_agent_steps=counts.get('search_agent_steps'),
                coupled_exact_agent_steps=counts.get('coupled_exact_agent_steps',0) if activity else None,
                coupled_modified_agent_steps=counts.get('coupled_modified_agent_steps',0) if activity else None,
                d2_agent_steps=None if activity is None else sum(counts.get(k,0) for k in ('d2_exact_agent_steps','d2_modified_agent_steps')),
                hold_agent_steps=counts.get('hold_agent_steps',0) if activity else None,
                unknown_agent_steps=counts.get('unknown_agent_steps',0) if activity else None,
                coupled_exact_fraction=None if activity is None else activity['coupled_exact_agent_step_fraction']))
            for component in ('inference','planning'):
                audit=diagnostic.get(component)
                if audit is None:continue
                for item in audit['trace']:
                    attempts.append(dict(base,component=component,step=item['step'],operation=item['operation'],
                        completed=item['completed'],seconds=item['seconds'],error=item['error'],
                        failure_phase=item['failure_phase'],first_exception=json.dumps(item['first_exception']),
                        phases_json=json.dumps(item['phases'],sort_keys=True),
                        trace_truncated=audit['trace_truncated']))
            if activity:
                for reason,count in activity['fallback_reasons'].items():
                    fallbacks.append(dict(base,reason=reason,records=count,
                        first_cause_steps=activity['first_fallback_reasons'].get(reason,0)))
    return dict(participation=participation,compute_attempts=attempts,fallback_reasons=fallbacks)


def write_tables(output,sets):
    data=tables(sets)
    for name,rows in data.items():
        if not rows:continue
        with (output/(name+'.csv')).open('x',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Map diagnostics','',
        'Search ends at discovery; execution thereafter is reported separately.',
        'Scoring, proposing and installed guidance execution are separate measurements.',
        'Exact coupled guidance is a conservative path match; safety-modified paths, holds and unknown provenance are separate.',
        'Execution counts use public physical transitions, including the terminal transition. They do not prove causal action attribution.',
        'Compute attempts include failed attempts. Rescore fallback time is a separate d2_fallback phase.',
        'Trace percentiles cover retained attempts; truncation is explicit. Aggregate counts cover the complete episode.',
        'OLD groups and historical outputs may lack diagnostics; blanks mean unavailable, not zero.',
        'CSV files: participation.csv, compute_attempts.csv and fallback_reasons.csv (when applicable).','']
    (output/'diagnostics.md').write_text('\n'.join(lines),encoding='utf-8')
    return data
