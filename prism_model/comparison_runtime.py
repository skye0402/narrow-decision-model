"""Complete learned and hybrid paths over the same neural model and policy."""
import copy
import hashlib
import json
import re
from time import perf_counter

import torch

from .evidence_data import ATTRIBUTES, attribute_row, compose
from .evidence_format import canonical_row
from .evidence_orchestrator import (CompositeJudgment, EvidenceOrchestrator, EvidencePart,
                                    InputLimitError, summarize_parts)
from .hybrid_evidence import compare_attribute, unaccounted_quantities
from .orchestrator import Judgment, Orchestrator, resolve_reference
from .ownership_inference import bind_extraction, decode_attribute, probabilities, reference_use
from .provenance import file_sha256


def inference_hashes():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    names = ('comparison_runtime', 'ownership_inference', 'ownership_model', 'ownership_format',
             'span_model', 'span_format', 'hybrid_evidence', 'neural', 'renderer', 'orchestrator',
             'evidence_orchestrator', 'evidence_data', 'evidence_format', 'relation_probe', 'evidence_span_data',
             'task_model', 'task_data', 'provenance')
    return {**{'prism_model/' + name + '.py': file_sha256(root / 'prism_model' / (name + '.py')) for name in names},
            'uv.lock': file_sha256(root / 'uv.lock')}


def config_identity(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def defaults(config):
    questions = ('material_group', 'replacement_identification', *('attribute:' + key for key in ATTRIBUTES))
    temperatures = {q: 1. for q in (*questions, 'bio', 'owner', 'state', 'reference', 'permission')}
    gates = {'group': {key: 0. for key in config['catalog']},
             **{q: {key: 0. for key in config['identification_options']} for q in questions if q != 'material_group'},
             'evidence': 0., 'reference': 0., 'coverage': 0.}
    return temperatures, gates


class ComparisonJudge:
    def __init__(self, runtime, config, records=None):
        self.runtime, self.config, self.records = runtime, config, records
        self.probabilistic = runtime.mode == 'learned'
        self.trace, self.evidence, self.reference_details = [], {}, None
        self.use, self.input_limit = None, None

    def predict(self, rows):
        started = perf_counter()
        try:
            self.runtime.model.eval()
            with torch.inference_mode():
                output = self.runtime.model.forward_batch(rows)
        except InputLimitError as error:
            self.input_limit = {'question_id': error.question_id, 'tokens': error.tokens, 'max_tokens': error.limit}
            raise
        except ValueError as error:
            found = re.fullmatch(r'Input has (\d+) tokens; limit is (\d+); no truncation', str(error))
            if found:
                self.input_limit = {'question_id': rows[0]['question_id'], 'tokens': int(found[1]), 'max_tokens': int(found[2])}
                raise InputLimitError(rows[0]['question_id'], int(found[1]), int(found[2])) from error
            raise
        self.trace.extend({'question_id': row['question_id'], 'input_sha256': row['text_sha256'],
                           'batch_rows': len(rows), 'batch_elapsed_ms': None if getattr(self.runtime.model, 'is_quality_replay', False)
                           else (perf_counter() - started) * 1000} for row in rows)
        return output

    def judgment(self, row, output, index):
        p = probabilities(output['decisions'][index, :len(row['option_ids'])], self.runtime.temperatures[row['question_id']])
        choice = int(p.argmax())
        value = Judgment(row['option_ids'][choice], dict(zip(row['option_ids'], p.tolist())))
        if self.records is not None:
            self.records.append({'row': row, 'choice': value.choice, 'probabilities': value.probabilities,
                                 'logits': output['decisions'][index, :len(row['option_ids'])].detach().float().cpu().tolist()})
        return value

    def judge_group(self, row, context):
        row = canonical_row(row)
        output = self.predict([row])
        result = self.judgment(row, output, 0)
        if self.runtime.mode == 'hybrid':
            self.use, self.reference_details = reference_use(output, 0, temperatures=self.runtime.temperatures,
                                                            threshold=0.)
        return result

    def judge_requirement(self, row, context, policy):
        attrs = [canonical_row(attribute_row(context, self.config, policy, key)) for key in policy['required']]
        if self.runtime.mode == 'learned':
            rows = [canonical_row(row), *attrs]
            output = self.predict(rows)
            parts = tuple(EvidencePart(r, self.judgment(r, output, i),
                                       self.runtime.gates[r['question_id']][self.judgment_choice(r, output, i)])
                          for i, r in enumerate(rows))
            choice, _ = summarize_parts(parts)
            return CompositeJudgment(choice, None, parts)
        if self.use is None:
            raise ValueError('Neural reference-use prediction is absent')
        output = self.predict(attrs)
        all_spans, uncertainties, details = [], [], {}
        for i, key in enumerate(policy['required']):
            spans, unknown, diagnostics = decode_attribute(output, i, key, temperatures=self.runtime.temperatures,
                                                           coverage_gate=self.runtime.gates['coverage'])
            all_spans.extend(spans)
            uncertainties.extend(unknown)
            details[key] = diagnostics
        extraction = bind_extraction(context['request']['description'], all_spans, self.use, uncertainties,
                                     self.runtime.model_identity)
        outcomes = {key: compare_attribute(context['request']['description'], key, extraction,
                                           context['reference']['components'], required_kind=policy['kind'],
                                           minimum_confidence=0.) for key in policy['required']}
        choice = compose([v['choice'] for v in outcomes.values()])
        # Fitted evidence/reference gates guard Ready only (11 November 2026).
        # A low-confidence span or reference decision cannot make a request
        # Ready, but it does not rewrite a missing or conflicting outcome.
        reference = min(v['reference_use']['confidence'] for v in outcomes.values())
        evidence = min([reference, *(s['confidence'] for v in outcomes.values() for s in v['evidence_spans'])])
        ready_gate = {'evidence': evidence, 'reference': reference,
                      'passed': evidence >= self.runtime.gates['evidence'] and reference >= self.runtime.gates['reference']}
        unaccounted = unaccounted_quantities(context['request']['description'], policy['required'], extraction.spans)
        self.evidence = {'attributes': outcomes, 'span_predictions': details,
                         'reference_predictions': self.reference_details,
                         'unaccounted_quantities': unaccounted,
                         'unaccounted_quantity_abstention': self.runtime.abstain_on_unaccounted,
                         'ready_gate': ready_gate}
        if choice == 'satisfied' and ((unaccounted and self.runtime.abstain_on_unaccounted) or not ready_gate['passed']):
            choice = 'unclear'
        return Judgment(choice, None)

    def judgment_choice(self, row, output, index):
        # Positive scalar temperature cannot change argmax. No gold/policy label
        # is used to pick the threshold of a learned judgment.
        return row['option_ids'][int(output['decisions'][index, :len(row['option_ids'])].argmax())]


class ComparisonRuntime:
    @classmethod
    def from_directory(cls, directory, *, mode, diagnostic=False, threads=2):
        from pathlib import Path
        from .ownership_bundle import load_model
        directory = Path(directory)
        model, config, manifest = load_model(directory, threads=threads)
        path = directory / ('calibration-' + mode + '.json')
        calibration = json.loads(path.read_text()) if path.exists() else None
        runtime = cls(model, config, manifest['model_identity'], mode=mode,
                      calibration=calibration, diagnostic=diagnostic)
        runtime.artifact_identity = file_sha256(directory / 'bundle-manifest.json')
        runtime.calibration_identity = file_sha256(path) if path.exists() else None
        return runtime

    def __init__(self, model, config, model_identity, *, mode, calibration=None, diagnostic=False):
        if mode not in ('learned', 'hybrid'):
            raise ValueError('Choose one complete comparison path')
        if calibration is None and not diagnostic:
            raise ValueError('Fit and freeze pipeline calibration before serving')
        self.model, self.config, self.model_identity, self.mode = model, config, model_identity, mode
        self.artifact_identity, self.calibration_identity = None, None
        self.temperatures, self.gates = defaults(config)
        self.calibration = calibration
        self.ready_enabled = True
        # Abstain-only coverage check; a fitted profile must declare it.
        self.abstain_on_unaccounted = True
        if calibration is not None:
            if (calibration['version'] != 'ownership-calibration-1' or calibration['model_identity'] != model_identity
                    or calibration['mode'] != mode or calibration['config_identity'] != config_identity(config)
                    or calibration['inference_source_sha256'] != inference_hashes()):
                raise ValueError('Calibration model/pipeline/source changed')
            if set(calibration['temperatures']) != set(self.temperatures) or set(calibration['gates']) != set(self.gates):
                raise ValueError('Incomplete pipeline calibration')
            for t in calibration['temperatures'].values():
                if type(t) not in (int, float) or not .25 <= t <= 10:
                    raise ValueError('Invalid temperature')
            for key, value in calibration['gates'].items():
                expected = self.gates[key]
                if isinstance(expected, dict):
                    if not isinstance(value, dict) or set(value) != set(expected):
                        raise ValueError('Invalid choice gates')
                    values = value.values()
                else:
                    values = [value]
                if any(v is not None and (type(v) not in (int, float) or not 0 <= v <= 1) for v in values):
                    raise ValueError('Invalid probability/evidence gate')
                if not isinstance(expected, dict) and value is None:
                    raise ValueError('Evidence gates must be explicit numbers')
            self.temperatures, self.gates = calibration['temperatures'], calibration['gates']
            if type(calibration.get('ready_enabled')) is not bool:
                raise ValueError('Declare whether fitted pipeline coverage permits Ready')
            self.ready_enabled = calibration['ready_enabled']
            if type(calibration.get('unaccounted_quantity_abstention')) is not bool:
                raise ValueError('Declare whether unaccounted quantities block Ready')
            self.abstain_on_unaccounted = calibration['unaccounted_quantity_abstention']

    def evaluate(self, request, *, config=None, lookup_failure=False, policy_id=None, records=None):
        started = perf_counter()
        local = copy.deepcopy(config or self.config)
        ref = request.get('reference_id') if isinstance(request, dict) else None
        if ref in local['references']:
            local['references'][ref] = sorted(local['references'][ref], key=lambda c: c['component_id'])
        judge = ComparisonJudge(self, local, records)
        cls = EvidenceOrchestrator if self.mode == 'learned' else Orchestrator
        result = cls(local, judge, self.gates).evaluate(request, lookup_failure=lookup_failure, policy_id=policy_id)
        if judge.input_limit is not None:
            result.update(error='input_too_long', input_limit=judge.input_limit)
        if not self.ready_enabled and result.get('status') == 'ready_for_buyer_review':
            result.update(status='clarification_needed', identification_accepted=False)
            result['clarification_ids'].append('confirm_identification')
        if result['http_status'] == 200:
            result['resolved_reference'] = resolve_reference(request, local, lookup_failure)
        if judge.evidence:
            result['evidence'] = judge.evidence
        result['model'] = {'name': 'PRISM ownership comparison', 'model_identity': self.model_identity,
                           'artifact_identity': self.artifact_identity, 'calibration_identity': self.calibration_identity,
                           'mode': self.mode, 'scope': 'synthetic_prototype',
                           'calibrated': self.calibration is not None, 'deployment_qualified': False}
        result['diagnostics'] = {'question_count': len(judge.trace), 'questions': judge.trace,
                                 'quality_replay': bool(getattr(self.model, 'is_quality_replay', False)),
                                 'elapsed_ms': None if getattr(self.model, 'is_quality_replay', False) else (perf_counter() - started) * 1000}
        return result

    def metadata(self):
        return {'model_name': 'PRISM ownership comparison', 'model_identity': self.model_identity,
                'artifact_identity': self.artifact_identity, 'calibration_identity': self.calibration_identity,
                'mode': self.mode, 'max_tokens': getattr(self.model, 'max_tokens', None),
                'calibrated': self.calibration is not None, 'scope': 'synthetic_prototype',
                'independently_reviewed': False, 'deployment_qualified': False}
