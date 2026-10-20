"""Manual bounded zero-residual audit: observe-only D2 or coupled-search core.

This entrypoint never trains or establishes map-performance acceptance.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path


def evaluate(config, extension, manifest_path, output):
    from chapter3_bser.learning.provenance import training_admission, require_review
    from chapter3_bser.learning.config import validate
    from chapter3_bser.learning.scenarios import validate_manifest
    from chapter3_bser.learning.checkpoint import validated_output
    from chapter3_bser.learning.reference import ZeroResidualPolicy
    from chapter3_bser.learning.runtime import RepairedMissionRuntime
    from .config import validate_research_config
    config, extension = validate(config), validate_research_config(extension)
    if not extension['enabled']:
        raise ValueError('explicitly enable a research configuration')
    identity = training_admission(mechanism=True, config=config)
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    scenes = validate_manifest(manifest, config, evaluation=True, mechanism=True)
    if len(scenes)*config['max_steps'] > 256:
        raise ValueError('interface audit exceeds the total physical-step budget')
    output = validated_output(config, output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'resolved_config.json').write_text(json.dumps(dict(base=config, interface=extension),indent=2),encoding='utf-8')
    rows = []
    for index, scene in enumerate(scenes):
        runtime = RepairedMissionRuntime(config, scene, seed=int(scene['scenario_seed']),
                                         episode_id=index, map_research=extension)
        try:
            cursor = 0
            with (output/f'public_events_{index}.jsonl').open('x',encoding='utf-8') as stream:
                while True:
                    events = runtime.env.read_public_events(cursor)
                    for event in events:
                        stream.write(json.dumps(asdict(event),allow_nan=False)+'\n')
                        cursor = event.sequence
                    if not runtime.map_research_session.valid:
                        raise ValueError('invalid interface evidence: '+str(runtime.map_research_session.error))
                    if runtime.terminal:
                        break
                    runtime.advance(ZeroResidualPolicy(),explore=False)
            (output/f'route_audit_{index}.json').write_text(json.dumps(list(runtime.map_research_session.records),indent=2),encoding='utf-8')
            rows.append(runtime.summary())
        finally:
            runtime.close()
    require_review(identity)
    core=extension['mode']=='coupled_search'
    result = dict(schema='ch3.coupled_search.interface_audit.v1', source_identity=identity,
                  mode=extension['mode'], search_planner_id='coupled_search_v1' if core else 'd2_v1', episodes=rows,
                  completed=True, algorithm_changed=core, coupled_search_implemented=core,
                  performance_passed=False, model_restored=False, optimizer_updates=0)
    (output/'interface_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--interface-config',required=True)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args(argv)
    from .config import validate_research_config
    extension = validate_research_config(json.loads(Path(args.interface_config).read_text(encoding='utf-8')))
    if not args.execute:
        result = dict(executed=False, mode=extension['mode'], enabled=extension['enabled'],
                      search_planner_id='coupled_search_v1' if extension['mode']=='coupled_search' else 'd2_v1',
                      coupled_search_implemented=extension['mode']=='coupled_search',
                      manifest=args.manifest, output=args.output)
    else:
        result = evaluate(json.loads(Path(args.config).read_text(encoding='utf-8')),extension,args.manifest,args.output)
    print(json.dumps(result,indent=2))
    return result


if __name__ == '__main__':
    main()
