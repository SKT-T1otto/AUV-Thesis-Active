"""Manual D2/B2/B3 workflow. Every execution stage requires --execute."""
import argparse
import importlib
import json
import sys

COMMANDS = {
    'train': 'chapter3_bser.learning.train',
    'evaluate': 'chapter3_bser.learning.evaluate',
    'ordering': 'chapter3_bser.learning.evaluate',
    'reference': 'chapter3_bser.learning.reference',
    'scenarios': 'chapter3_bser.learning.scenarios',
    'summarize': 'chapter3_bser.learning.summarize',
    'suite': 'chapter3_bser.learning.suite',
    'map-suite': 'chapter3_bser.learning.map_suite',
    'bounded-experiment': 'chapter3_bser.learning.bounded_experiment',
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('command', choices=('verify', *COMMANDS))
        parser.print_help()
        return
    command, rest = sys.argv[1], sys.argv[2:]
    if command == 'verify':
        from chapter3_bser.learning.provenance import verify_reviewed_changes, research_readiness
        identity = verify_reviewed_changes()
        try:
            receipt = research_readiness(identity)
            readiness = dict(manual_research_execution_allowed=True, readiness_run=receipt['run_id'])
        except ValueError as exc:
            readiness = dict(manual_research_execution_allowed=False, reason=str(exc))
        print(json.dumps(dict(source=identity, **readiness, performance_passed=False), indent=2))
        return
    if command not in COMMANDS:
        raise SystemExit('unknown command: ' + command)
    entry = importlib.import_module(COMMANDS[command]).main
    sys.argv = [sys.argv[0], *([command] if command in ('evaluate', 'ordering') else []), *rest]
    return entry()


if __name__ == '__main__':
    main()
