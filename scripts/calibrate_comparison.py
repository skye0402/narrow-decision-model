"""Fit designated calibration roles for an eligible complete comparison model."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prism_model.comparison_calibration import fit_bundle

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--model', required=True)
p.add_argument('--data', required=True)
p.add_argument('--output', required=True)
p.add_argument('--threads', type=int, default=2)
a = p.parse_args()
result = fit_bundle(a.model, a.data, a.output, threads=a.threads)
print(json.dumps({'model_identity': result['model_identity'],
                  'eligible': {k: v['eligible'] for k, v in result['pipelines'].items()},
                  'reserved_evaluation_used': False}))
