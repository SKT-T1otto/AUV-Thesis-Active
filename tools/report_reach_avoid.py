"""Read-only report generation from completed or live journals; zero environment steps."""
import argparse
import json
from pathlib import Path
from chapter3_bser.reach_avoid.monitor_reports import render


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--execute',action='store_true');args=p.parse_args(argv)
    root=Path(args.run).resolve()
    if not (root/'monitoring').exists() and (root/'train/monitoring').exists():root=root/'train'
    result=render(root,args.output) if args.execute else dict(executed=False,run=str(root),output=args.output)
    print(json.dumps(result,ensure_ascii=False,indent=2));return result


if __name__=='__main__':main()
