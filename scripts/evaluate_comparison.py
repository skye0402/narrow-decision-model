"""Score a designated role using both fixed complete comparison paths."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prism_model.comparison_evaluation import score

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--role', required=True, choices=('development', 'regression', 'prototype_evaluation'))
    p.add_argument('--procedure')
    p.add_argument('--regression-file')
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    result = score(a.bundle, a.dataset, a.output, role=a.role, procedure=a.procedure,
                   regression_file=a.regression_file, threads=a.threads)
    print(json.dumps({'role': result['role'], 'cases': result['cases'], 'primary_families': result['primary_families'],
                      'eligible': {k: v['eligible'] for k, v in result['summary']['pipelines'].items()},
                      'qualifies_cpu_performance': False}))
