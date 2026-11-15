import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from prism_model.demo_server import ARTIFACT, CALIBRATION, LEGACY_ARTIFACT, LEGACY_CALIBRATION, DemoRuntime


class CandidateFixture:
    mode = 'hybrid'
    ready_enabled = True
    artifact_identity = LEGACY_ARTIFACT
    calibration_identity = LEGACY_CALIBRATION

    def __init__(self):
        self.config = {'catalog': {'MG_MOTORS': 'Motors'}, 'references': {'REF': [{'power_kw': 5.5}]},
                       'plants': ['1100'], 'evaluation_date': '2026-11-01',
                       'default_policies': {'MG_MOTORS': 'P'}, 'policies': {'P': {'required': ['power_kw']}}}
        self.prediction = {'http_status': 200, 'status': 'clarification_needed',
                           'evidence': {'attributes': {'power_kw': {'choice': 'unclear'}}}}

    def metadata(self):
        return {'artifact_identity': self.artifact_identity, 'calibration_identity': self.calibration_identity}

    def evaluate(self, request):
        self.request = request
        return self.prediction


class DemoBoundaryTests(unittest.TestCase):
    def test_presentation_cannot_change_request_or_model_decision(self):
        candidate = CandidateFixture()
        request = {'item_id': 'DEMO', 'description': 'motor'}
        original = copy.deepcopy(request)
        result = DemoRuntime(candidate).evaluate(request)
        self.assertIs(candidate.request, request)
        self.assertIs(result, candidate.prediction)
        self.assertEqual(request, original)

    def test_metadata_does_not_expose_mutable_policy_or_reference_state(self):
        candidate = CandidateFixture()
        runtime = DemoRuntime(candidate)
        metadata = runtime.metadata()
        metadata['references']['REF'][0]['power_kw'] = 999
        metadata['requirements']['MG_MOTORS'].clear()
        metadata['examples'][0]['description'] = 'changed'
        second = runtime.metadata()
        self.assertEqual(second['references']['REF'][0]['power_kw'], 5.5)
        self.assertEqual(second['requirements']['MG_MOTORS'], ['power_kw'])
        self.assertNotEqual(second['examples'][0]['description'], 'changed')
        self.assertEqual(second['evidence_snapshot']['threshold_selection']['false_ready'], 1)
        self.assertFalse(second['evidence_snapshot']['blind'])

    def test_other_candidate_and_ineligible_path_cannot_inherit_demo_claims(self):
        for field, value in [('mode', 'learned'), ('ready_enabled', False),
                             ('artifact_identity', 'another-model'), ('calibration_identity', 'other-gates')]:
            candidate = CandidateFixture()
            setattr(candidate, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                DemoRuntime(candidate)
        candidate=CandidateFixture();candidate.artifact_identity='unknown';candidate.calibration_identity=None
        with self.assertRaises(ValueError):DemoRuntime(candidate)

    def test_wrong_bundle_is_rejected_before_neural_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'bundle-manifest.json').write_text('{}')
            with patch('prism_model.comparison_runtime.ComparisonRuntime.from_directory') as load:
                with self.assertRaisesRegex(ValueError, 'pinned round-two'):
                    DemoRuntime.from_directory(directory)
                load.assert_not_called()

    def test_current_demo_cannot_start_without_matching_passed_regressions(self):
        import json
        candidate=CandidateFixture()
        candidate.artifact_identity=ARTIFACT;candidate.calibration_identity=CALIBRATION
        with tempfile.TemporaryDirectory() as directory:
            receipt=Path(directory)/'results.json'
            for artifact,passed in [(ARTIFACT,False),('other',True)]:
                receipt.write_text(json.dumps({'artifact_identity':artifact,'calibration_identity':CALIBRATION,'regressions_passed':passed}))
                with patch('prism_model.demo_server.RESULTS',receipt),self.assertRaises(ValueError):
                    DemoRuntime(candidate).metadata()
