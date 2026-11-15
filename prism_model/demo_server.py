"""Presentation for the pinned round-two candidate; inference stays unchanged."""
import copy
import json
from pathlib import Path

from .provenance import file_sha256
from .task_server import ReviewServer

ROOT = Path(__file__).resolve().parents[1]
LEGACY_BUNDLE = ROOT / '.cache/natural-transfer-r2-20261111/calibration-profile/model'
LEGACY_ARTIFACT = '90e3527df5c3325b5f15d7b02bd91a3d01c9be58fdc9538f8f46dd450688eb8f'
LEGACY_CALIBRATION = 'c6eced29a1bd5bf21d37f8e659220a17f3c23b801912eca70c9095b159ce2f34'
BUNDLE = ROOT / '.cache/category-gate-20261115/demo-update/model'
ARTIFACT = '82bd496705c18bd8355baf4555dc61d08e4010f94519bb95c1dbadcaa368f98a'
CALIBRATION = '4df4339744a865859f4cf2d71741d3713866aced0537378ee174d82d5b851de0'
CANDIDATES = {ARTIFACT: CALIBRATION, LEGACY_ARTIFACT: LEGACY_CALIBRATION}
RESULTS = ROOT / 'research/2026-11-15-category-gate-results.json'

# Public, illustrative walkthroughs, not evaluation data or expected answers.
EXAMPLES = [
    {'id': 'motor-incomplete', 'label': 'The request that started it', 'category': 'Motor',
     'description': 'Motor type 230 for conveyor actor replacement', 'reference_id': ''},
    {'id': 'motor-complete', 'label': 'Add the missing specifications', 'category': 'Motor',
     'description': 'Supply a replacement motor rated 5.5 kW, 400 V, three-phase.', 'reference_id': ''},
    {'id': 'motor-reference', 'label': 'Use a like-for-like reference', 'category': 'Motor',
     'description': 'Replace the motor with the same model, like for like.', 'reference_id': 'REF-MOTOR-400'},
    {'id': 'motor-conflict', 'label': 'Introduce a conflicting rating', 'category': 'Motor',
     'description': 'Replace the motor like for like, but the replacement must be 230 V instead of 400 V.',
     'reference_id': 'REF-MOTOR-400'},
    {'id': 'roller', 'label': 'Conveyor roller', 'category': 'Roller',
     'description': 'Supply a replacement conveyor roller, diameter 50 mm and length 500 mm.', 'reference_id': ''},
    {'id': 'gearbox', 'label': 'Gearbox', 'category': 'Gearbox',
     'description': 'Supply a replacement gearbox with a gear ratio of 5:1 and a shaft diameter of 20 mm.', 'reference_id': ''},
    {'id': 'belt', 'label': 'Drive belt', 'category': 'Belt',
     'description': 'Supply a replacement drive belt, width 20 mm, length 500 mm, profile SPA.', 'reference_id': ''},
    {'id': 'bearing', 'label': 'Bearing', 'category': 'Bearing',
     'description': 'Supply a replacement ball bearing, bore 20 mm, outside diameter 62 mm, width 20 mm.', 'reference_id': ''},
    {'id': 'sensor', 'label': 'Proximity sensor', 'category': 'Sensor',
     'description': 'Supply an inductive proximity sensor, sensing distance 5 mm, supply 24 V DC, PNP output.', 'reference_id': ''},
    {'id': 'outside', 'label': 'Something outside the catalog', 'category': 'Other',
     'description': 'Supply a replacement hydraulic pump.', 'reference_id': ''},
]


class DemoRuntime:
    """Add public presentation metadata without changing requests or predictions."""
    def __init__(self, runtime):
        if runtime.mode != 'hybrid' or not runtime.ready_enabled:
            raise ValueError('The demo requires the fitted hybrid candidate')
        if runtime.artifact_identity not in CANDIDATES or CANDIDATES[runtime.artifact_identity] != runtime.calibration_identity:
            raise ValueError('The demo results belong to a different artifact/calibration')
        self.runtime = runtime

    @classmethod
    def from_directory(cls, directory=BUNDLE, *, threads=2):
        from .comparison_runtime import ComparisonRuntime
        directory = Path(directory)
        if file_sha256(directory / 'bundle-manifest.json') not in CANDIDATES:
            raise ValueError('Use a pinned round-two bundle; the demo does not rebind or recalibrate it')
        return cls(ComparisonRuntime.from_directory(directory, mode='hybrid', threads=threads))

    def evaluate(self, request):
        return self.runtime.evaluate(request)

    def metadata(self):
        config = self.runtime.config
        updated = self.runtime.artifact_identity == ARTIFACT
        snapshot = {
                    'date': '2026-11-11', 'blind': False, 'human_reviewed': False,
                    'exposed100': {'correct_ready': 23, 'truly_ready': 34, 'false_ready': 0,
                                   'identification_f1': .6572, 'correct_status': 89, 'cases': 100},
                    'threshold_selection': {'predicted_ready': 693, 'false_ready': 1, 'safety_false_ready': 0},
                    'identification_target': .85, 'constrained_cpu_requalified': False,
                    'source': 'research/2026-11-11-improvement-results.json'}
        if updated:
            result = json.loads(RESULTS.read_text())
            if (result['artifact_identity'] != ARTIFACT or result['calibration_identity'] != CALIBRATION
                    or result['regressions_passed'] is not True):
                raise ValueError('Demo update has no matching regression evidence')
            snapshot = copy.deepcopy(result['evidence_snapshot'])
        return {**self.runtime.metadata(), 'device': 'cpu',
                'demo_name': 'PRISM Decision Lab', 'candidate_label': 'Round 02 · gate update' if updated else 'Round 02',
                'category_gates': copy.deepcopy(getattr(self.runtime, 'gates', {}).get('group', {})),
                'catalog': copy.deepcopy(config['catalog']),
                'references': copy.deepcopy(config['references']),
                'plants': list(config['plants']), 'evaluation_date': config['evaluation_date'],
                'requirements': {group: list(config['policies'][policy]['required'])
                                 for group, policy in config['default_policies'].items()},
                'examples': copy.deepcopy(EXAMPLES),
                'evidence_snapshot': snapshot}


class DemoServer(ReviewServer):
    def __init__(self, address, runtime):
        super().__init__(address, runtime)
        self.page = Path(__file__).with_name('demo.html').read_bytes()


def main():
    import argparse
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=BUNDLE)
    parser.add_argument('--port', type=int, default=8769)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_interop_threads(1)
    runtime = DemoRuntime.from_directory(args.bundle, threads=args.threads)
    runtime.metadata()  # Refuse to start a candidate without its review evidence.
    with DemoServer(('127.0.0.1', args.port), runtime) as server:
        print(json.dumps({'event': 'demo_ready', 'url': f'http://127.0.0.1:{server.server_port}',
                          'artifact_identity': runtime.runtime.artifact_identity, 'mode': 'hybrid', 'device': 'cpu'}), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
