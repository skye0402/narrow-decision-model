"""Freeze and score complete comparison paths without fitting on evaluation roles."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform

import torch

from . import (comparison_calibration, comparison_quality, metrics, provided_evaluation,
               task_iteration, ownership_bundle, ownership_data)
from .comparison_quality import OutputCache, oracle_choice, primary_cases, summarize
from .comparison_runtime import ComparisonRuntime, inference_hashes
from . import comparison_scope
from .evidence_data import load_partition
from .ownership_bundle import load_model, verify_bundle
from .ownership_data import examples
from .ownership_inference import decode_attribute
from .provenance import file_sha256
from .task_data import case_config

VERSION = 'complete-comparison-evaluation-2'
REGRESSION_SHA256 = '4805aa9a62fc72f46b176351c7b4bada18c49a43fdd5f55a2947fc2db73c4409'
ROLE_EXECUTION = {'development': 'cached_fp32_rows', 'prototype_evaluation': 'cached_fp32_rows',
                  'regression': 'actual_ordered_runtime_batches'}


def sources():
    modules = (comparison_calibration, comparison_quality, metrics, provided_evaluation, task_iteration,
               ownership_bundle, ownership_data, comparison_scope)
    root = Path(__file__).resolve().parents[1]
    return {**inference_hashes(), 'prism_model/comparison_evaluation.py': file_sha256(__file__),
            **{'prism_model/' + m.__name__.split('.')[-1] + '.py': file_sha256(m.__file__) for m in modules},
            **{n: file_sha256(root / n) for n in ('scripts/evaluate_comparison.py', 'scripts/freeze_comparison.py')}}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def quality_gate(summary, ownership):
    """Recompute the fixed gate; do not trust a saved eligibility boolean."""
    routing = summary['routing']
    count = routing['predicted_ready']
    interval = metrics.wilson(count - routing['false_ready'], count)
    return (summary['supported_group_accuracy'] >= .90
            and summary['actual_identification_macro_f1'] >= .85
            and summary['oracle_identification_macro_f1'] >= .85
            and count >= 150 and (interval['estimate'] or 0.) >= .95
            and (interval['lower'] or 0.) >= .90 and (routing['ready_recall'] or 0.) >= .60
            and summary['safety_false_ready'] == 0
            and ownership['other_item']['recall'] >= .90 and ownership['other_item']['precision'] >= .90)


def identity(bundle, dataset):
    bundle, dataset = map(Path, (bundle, dataset))
    manifest = verify_bundle(bundle)
    config = read(bundle / 'pilot_config.json')
    profiles = {mode: read(bundle / ('calibration-' + mode + '.json')) for mode in ('learned', 'hybrid')}
    for mode, fitted in profiles.items():
        ComparisonRuntime(None, config, manifest['model_identity'], mode=mode, calibration=fitted)
        if set(fitted['fit_roles']) != {'calibration', 'threshold_selection'}:
            raise ValueError('Evaluation roles must never fit calibration')
    training = read(bundle / 'training-report.json')
    dataset_identity = file_sha256(dataset / 'dataset-manifest.json')
    if training['dataset_manifest']['full_cohort_manifest_sha256'] != dataset_identity:
        raise ValueError('Evaluation cohort differs from training provenance')
    if training['training']['max_batches'] is not None:
        raise ValueError('A mechanism run cannot enter final quality evaluation')
    return {'version': VERSION, 'artifact_identity': file_sha256(bundle / 'bundle-manifest.json'),
            'model_identity': manifest['model_identity'], 'dataset_manifest_sha256': dataset_identity,
            'calibration_sha256': {mode: file_sha256(bundle / ('calibration-' + mode + '.json')) for mode in profiles},
            'sources_sha256': sources(), 'precision': 'fp32',
            'runtime_dependencies': {'torch': torch.__version__.split('+')[0],
                'transformers': importlib.metadata.version('transformers'),
                'tokenizers': importlib.metadata.version('tokenizers'), 'threads': 2, 'device': 'cpu'}}, config, profiles


def freeze(bundle, dataset, calibration_report, development_report, output):
    """No reserved case is loaded here; freeze identities after development passes."""
    output = Path(output)
    if output.exists():
        raise ValueError('The procedure is already frozen; preserve it')
    signature, config, profiles = identity(bundle, dataset)
    development, fitted = read(development_report), read(calibration_report)
    if (development['role'] != 'development' or development['signature'] != signature
            or fitted['calibrated_artifact_identity'] != signature['artifact_identity']
            or fitted['model_identity'] != signature['model_identity']
            or fitted['data_manifest_sha256'] != signature['dataset_manifest_sha256']
            or fitted['source_sha256'] != file_sha256(comparison_calibration.__file__)
            or fitted['reserved_evaluation_used'] or fitted['author100_used']):
        raise ValueError('Development or fitting evidence changed')
    _, native, _ = load_partition(dataset, 'development', allowed_splits={'development'})
    by_id = {c['case_id']: c for c in native}
    verified = {}
    for mode in ('learned', 'hybrid'):
        records_path = Path(development_report).parent / ('records-' + mode + '.json')
        if file_sha256(records_path) != development['record_sha256'][mode]:
            raise ValueError('Development per-case evidence changed')
        rows = read(records_path)
        if len(rows) != len(native) or {r['case_id'] for r in rows} != set(by_id):
            raise ValueError('Development membership changed')
        if any(any(r[key] != by_id[r['case_id']][key] for key in ('expected', 'family_id', 'primary')) for r in rows):
            raise ValueError('Development targets or primary variants changed')
        primary = [r for r in rows if r['primary']]
        verified[mode] = comparison_scope.annotate(
            summarize(primary, [r['oracle_identification'] for r in primary], config), primary, config)
        if any(json.loads(json.dumps(verified[mode][key])) != development['summary']['pipelines'][mode][key] for key in verified[mode]):
            raise ValueError('Development aggregates disagree with per-case evidence')
    summary = development['summary']
    qualified = [mode for mode in ('learned', 'hybrid') if profiles[mode]['ready_enabled']
                 and quality_gate(verified[mode], summary['exact_source_ownership'])
                 and verified[mode]['supported_category_scope_covered']]
    if not qualified:
        raise ValueError('Neither path passes the unchanged complete development/ownership gates with full supported category scope')
    cohort = read(Path(dataset) / 'dataset-manifest.json')
    result = {'version': VERSION, 'frozen_at': datetime.now(timezone.utc).isoformat(), 'signature': signature,
              'calibration_report_sha256': file_sha256(calibration_report),
              'development_report_sha256': file_sha256(development_report),
              'evaluation_role': 'prototype_evaluation', 'evaluation_file_sha256': cohort['files_sha256']['prototype_evaluation.jsonl'],
              'regression_file_sha256': REGRESSION_SHA256,
              'primary_unit': 'one_preselected_variant_per_scenario_family',
              'all_variants': 'descriptive_correlated_results_only',
              'baseline_development_eligible': 'learned' in qualified,
              'hybrid_development_eligible': 'hybrid' in qualified, 'qualified_development_modes': qualified,
              'supported_scope_rule': 'Each declared supported category has positive primary families and at least one correct Ready; category recalls remain reported separately',
              'quality_execution_by_role': ROLE_EXECUTION,
              'independently_human_reviewed': False,
              'qualifies_cpu_performance': False, 'deployment_qualified': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def source_ownership(cache, cases, config, temperatures):
    correct, expected, predicted = Counter(), Counter(), Counter()
    for e in examples(primary_cases(cases), config):
        if e['annotation'] is None:
            continue
        attribute = e['row']['question_id'].removeprefix('attribute:')
        output = cache.forward_batch([e['row']])
        spans, _, _ = decode_attribute(output, 0, attribute, temperatures=temperatures, coverage_gate=0.)
        gold = {(s['start'], s['end'], s['role'], 'other_item' if s['role'] == 'background' else 'requested_item')
                for s in e['annotation']['spans'] if s['attribute'] == attribute or s['role'] == 'type'}
        actual = {(s.start, s.end, s.role, s.owner) for s in spans}
        for owner in ('requested_item', 'other_item'):
            a, b = {s for s in gold if s[3] == owner}, {s for s in actual if s[3] == owner}
            correct[owner] += len(a & b)
            expected[owner] += len(a)
            predicted[owner] += len(b)
    return {k: {'correct': correct[k], 'expected': expected[k], 'predicted': predicted[k],
                 'recall': correct[k] / expected[k] if expected[k] else 0.,
                 'precision': correct[k] / predicted[k] if predicted[k] else 0.}
            for k in ('requested_item', 'other_item')}


class ProgressCache(OutputCache):
    def preload(self, rows, batch_size=8):
        pending = list({r['text_sha256']: r for r in rows if r['text_sha256'] not in self.storage}.values())
        for start in range(0, len(pending), 128):
            super().preload(pending[start:start + 128], batch_size=batch_size)
            if len(pending) > 128:
                print(json.dumps({'event': 'comparison_quality_outputs', 'rows': min(start + 128, len(pending)),
                                  'total': len(pending), 'cpu_performance_evidence': False}), flush=True)


class ActualBatchCache(OutputCache):
    """Always execute the requested batch; retain outputs only for diagnostics."""
    is_quality_replay = False

    def forward_batch(self, rows):
        if not rows or len(rows) > 8 or len({r['text_sha256'] for r in rows}) != len(rows):
            raise ValueError('Actual regression batches must be nonempty, distinct and at most eight rows')
        for row in rows:
            self.storage.pop(row['text_sha256'], None)
        return super().forward_batch(rows)


def score(bundle, dataset, output, *, role, procedure=None, regression_file=None, threads=2):
    """Explicit role guard runs before reading any reserved request or model output."""
    if role not in ('development', 'prototype_evaluation', 'regression'):
        raise ValueError('Choose an explicit evaluation role')
    if threads != 2:
        raise ValueError('The frozen comparison uses two CPU threads')
    signature, config, profiles = identity(bundle, dataset)
    output = Path(output)
    if (output / 'evaluation-report.json').exists():
        raise ValueError('Evaluation already completed; read the saved results')
    if role != 'development':
        if procedure is None:
            raise ValueError('Freeze the complete comparison before opening reserved or regression cases')
        frozen = read(procedure)
        if (frozen['version'] != VERSION or frozen['signature'] != signature
                or frozen['evaluation_role'] != 'prototype_evaluation'
                or frozen.get('quality_execution_by_role') != ROLE_EXECUTION):
            raise ValueError('The frozen model, procedure or dataset changed')
        signature = {**signature, 'procedure_sha256': file_sha256(procedure)}
    lock = output / 'evaluation-started.json'
    run = {'signature': signature, 'role': role}
    if lock.exists() and read(lock) != run:
        raise ValueError('A different evaluation already started in this directory')
    write(lock, run)
    if role == 'regression':
        if regression_file is None or file_sha256(regression_file) != REGRESSION_SHA256:
            raise ValueError('Only the exposed author100 can be labeled regression')
        cases = [json.loads(line) for line in Path(regression_file).read_text().splitlines()]
        if len(cases) != 100:
            raise ValueError('The exposed regression inventory changed')
    else:
        if role == 'prototype_evaluation' and read(Path(dataset) / 'dataset-manifest.json')['files_sha256']['prototype_evaluation.jsonl'] != frozen['evaluation_file_sha256']:
            raise ValueError('Reserved case identity changed')
        local, cases, _ = load_partition(dataset, role, allowed_splits={role})
        if local != config:
            raise ValueError('Evaluation policy differs from the trained model')
    primary_cases(cases)
    model, _, _ = load_model(bundle, threads=threads)
    actual_batches = role == 'regression'
    cache = ActualBatchCache(model) if actual_batches else ProgressCache(model)
    if not actual_batches:
        cache.preload([e['row'] for e in examples(cases, config)])
    summaries, records = {}, {}
    for mode in ('learned', 'hybrid'):
        runtime = ComparisonRuntime(cache, config, signature['model_identity'], mode=mode, calibration=profiles[mode])
        runtime.artifact_identity, runtime.calibration_identity = signature['artifact_identity'], signature['calibration_sha256'][mode]
        rows, oracle = [], []
        for index, case in enumerate(cases):
            prediction = runtime.evaluate(case['request'], config=case_config(case, config),
                                          policy_id=case.get('policy_id'), lookup_failure=case.get('lookup_failure', False))
            rows.append({'case_id': case['case_id'], 'family_id': case['family_id'], 'primary': case['primary'],
                         'blueprint': case.get('intended_facts', {}).get('blueprint', 'unspecified'),
                         'expected': case['expected'], 'prediction': prediction,
                         'differences': {k: {'expected': value, 'predicted': prediction.get(k)} for k, value in case['expected'].items()
                                         if (set(value) != set(prediction.get(k, [])) if k in ('clarification_ids', 'notice_ids') else value != prediction.get(k))}})
            oracle.append(oracle_choice(runtime, case, config))
            rows[-1]['oracle_identification'] = oracle[-1]
            if (index + 1) % (10 if actual_batches else 256) == 0:
                print(json.dumps({'event': 'comparison_quality_routed', 'mode': mode, 'cases': index + 1, 'total': len(cases)}), flush=True)
        primary = [r for r in rows if r['primary']]
        primary_oracle = [o for r, o in zip(rows, oracle) if r['primary']]
        result = comparison_scope.annotate(summarize(primary, primary_oracle, config), primary, config)
        variants = summarize(rows, oracle, config)
        variants.update(eligible=None, base_eligible=None, selection_rank=None, intervals_are_independent=False)
        for key in ('ready_precision', 'status_only_ready_precision'):
            variants['routing'][key].update(lower=None, upper=None)
        slices = defaultdict(list)
        for r in primary:
            slices[r['blueprint']].append(r)
        result['descriptive_slices'] = {name: {'families': len(items),
            'complete_matches': sum(not r['differences'] for r in items),
            'false_ready': sum(r['prediction']['status'] == 'ready_for_buyer_review' and not comparison_calibration.correct_ready(r) for r in items)}
            for name, items in slices.items()}
        summaries[mode], records[mode] = result, rows
        write(output / ('records-' + mode + '.json'), rows)
        write(output / ('all-variants-' + mode + '.json'), variants)
    ownership = source_ownership(cache, cases, config, profiles['hybrid']['temperatures'])
    samples = comparison_calibration.observations(cache, primary_cases(cases), config)
    neural_calibration = {mode: {key: metrics.calibration_metrics(comparison_calibration.probability_records(
        rows, fitted['temperatures'][key])) for key, rows in samples.items()} for mode, fitted in profiles.items()}
    result = {'version': VERSION, 'signature': run['signature'], 'role': role, 'cases': len(cases),
              'primary_families': len(primary_cases(cases)),
              'summary': {'pipelines': summaries, 'exact_source_ownership': ownership},
              'oracle_policy_neural_calibration_diagnostics': neural_calibration,
              'neural_diagnostics_are_joint_ready_probability': False,
              'record_sha256': {mode: file_sha256(output / ('records-' + mode + '.json')) for mode in records},
              'reserved_evaluation_used': role == 'prototype_evaluation', 'regression_is_blind': False,
              'independently_human_reviewed': False, 'quality_uses_cached_real_outputs': not actual_batches,
              'quality_execution': ROLE_EXECUTION[role],
              'runtime': {'platform': platform.platform(), 'architecture': platform.machine(), 'precision': 'fp32',
                          'torch': torch.__version__, 'transformers': importlib.metadata.version('transformers'),
                          'tokenizers': importlib.metadata.version('tokenizers'), 'threads': torch.get_num_threads()},
              'qualifies_cpu_performance': False, 'deployment_qualified': False}
    write(output / 'evaluation-report.json', result)
    return result
