"""Role-isolated calibration and verified gates for the complete comparison."""
from collections import defaultdict
import copy
import itertools
import json
import math
from pathlib import Path
import shutil

import torch

from .calibration import select_outcome_threshold
from .comparison_quality import OutputCache, oracle_choice, primary_cases, summarize
from .comparison_runtime import ComparisonRuntime, config_identity, defaults, inference_hashes
from .evidence_data import load_partition
from .metrics import calibration_metrics, wilson
from .ownership_bundle import load_model, verify_bundle, write_manifest
from .ownership_data import examples
from .ownership_format import factor_targets
from .ownership_inference import probabilities
from .provenance import file_sha256

VERSION = 'ownership-calibration-1'


def fit_temperature(samples):
    """Bounded scalar NLL fit, batched by vocabulary size; never use padding."""
    if not samples:
        return 1.
    grouped = defaultdict(list)
    for s in samples:
        logits = torch.as_tensor(s['logits'], dtype=torch.float64)
        if (logits.ndim != 1 or len(logits) < 2 or not bool(torch.isfinite(logits).all())
                or type(s['target']) is not int or not 0 <= s['target'] < len(logits)):
            raise ValueError('Invalid calibration observation')
        grouped[len(logits)].append((logits, s['target']))
    packed = [(torch.stack([x for x, _ in rows]), torch.tensor([y for _, y in rows])) for rows in grouped.values()]
    def nll(log_t):
        t = math.exp(log_t)
        return sum(float(torch.nn.functional.cross_entropy(x / t, y, reduction='sum')) for x, y in packed) / len(samples)
    a, b = math.log(.25), math.log(10.)
    grid = [a + (b - a) * i / 80 for i in range(81)]
    index = min(range(len(grid)), key=lambda i: nll(grid[i]))
    a, b = grid[max(0, index - 1)], grid[min(80, index + 1)]
    ratio = (math.sqrt(5) - 1) / 2
    for _ in range(36):
        c, d = b - ratio * (b - a), a + ratio * (b - a)
        if nll(c) < nll(d):
            b = d
        else:
            a = c
    return min(10., max(.25, math.exp(min((grid[index], (a + b) / 2), key=nll))))


def correct_ready(row):
    return (row['expected']['status'] == 'ready_for_buyer_review'
            and row['prediction']['group'] == row['expected']['group']
            and row['prediction']['identification'] == row['expected']['identification'])


def safety_case(row):
    return (row['expected']['group'] in ('mixed_scope', 'insufficient_evidence', 'no_suitable_group')
            or row['expected']['identification'] in ('conflicting', 'unclear')
            or row['expected']['status'] == 'evaluation_unavailable')


def ready_metrics(kept, rows):
    correct = sum(correct_ready(r) for r in kept)
    truly = sum(r['expected']['status'] == 'ready_for_buyer_review' for r in rows)
    interval = wilson(correct, len(kept))
    recall = correct / truly if truly else 0.
    false_safety = sum(safety_case(r) for r in kept if not correct_ready(r))
    eligible = (len(kept) >= 150 and (interval['estimate'] or 0.) >= .95
                and (interval['lower'] or 0.) >= .90 and recall >= .60 and false_safety == 0)
    return {'correct_ready': correct, 'false_ready': len(kept) - correct, 'support': len(kept),
            'precision': interval, 'recall': recall, 'safety_false_ready': false_safety, 'eligible': eligible}


def axis_candidates(scored, axis):
    values = sorted({r['scores'][axis] for r in scored})
    if not values:
        return [0.]
    gaps = [(a + b) / 2 for a, b in zip(values, values[1:])]
    bad = {r['scores'][axis] for r in scored if not correct_ready(r)}
    bad_gaps = [g for v, g in zip(values, gaps) if v in bad]
    fixed = {0., .5, .7, .85, .95, .99, 1.}
    pool = sorted(set(bad_gaps or gaps))
    if len(pool) > 13:
        pool = [pool[round(i * (len(pool) - 1) / 12)] for i in range(13)]
    return sorted(fixed | set(pool))


def candidate_gates(rows, mode):
    raw = [r for r in rows if r['prediction']['status'] == 'ready_for_buyer_review']
    axes = ('satisfied',) if mode == 'learned' else ('evidence', 'reference', 'coverage')
    candidates = []
    for values in itertools.product(*(axis_candidates(raw, axis) for axis in axes)):
        gate = dict(zip(axes, values))
        kept = [r for r in raw if all(r['scores'][axis] >= cutoff for axis, cutoff in gate.items())]
        metrics = ready_metrics(kept, rows)
        if metrics['eligible']:
            candidates.append({'gates': gate, 'metrics': metrics,
                               'ready_case_ids': sorted(r['case_id'] for r in kept)})
    candidates.sort(key=lambda c: (-c['metrics']['correct_ready'], c['metrics']['false_ready'],
                                    sum(c['gates'].values()), tuple(c['gates'].values())))
    # Multiple gate rectangles can retain exactly the same requests. Verify at
    # most twelve distinct retained sets through actual complete inference.
    unique, seen = [], set()
    for candidate in candidates:
        key = tuple(candidate['ready_case_ids'])
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
        if len(unique) == 12:
            break
    return unique


def profile(config, model_identity, mode, temperatures):
    _, gates = defaults(config)
    return {'version': VERSION, 'model_identity': model_identity, 'mode': mode,
            'config_identity': config_identity(config), 'inference_source_sha256': inference_hashes(),
            'temperatures': copy.deepcopy(temperatures), 'gates': gates, 'ready_enabled': True,
            'unaccounted_quantity_abstention': mode == 'hybrid',
            'fit_roles': ['calibration', 'threshold_selection'], 'independently_human_reviewed': False,
            'deployment_qualified': False}


def replay(cache, cases, config, model_identity, fitted, *, scores=False):
    runtime = ComparisonRuntime(cache, config, model_identity, mode=fitted['mode'], calibration=fitted)
    rows, oracle = [], []
    from .task_data import case_config
    for case in primary_cases(cases):
        records = []
        result = runtime.evaluate(case['request'], config=case_config(case, config), policy_id=case.get('policy_id'),
                                  lookup_failure=case.get('lookup_failure', False), records=records)
        row = {'case_id': case['case_id'], 'family_id': case['family_id'], 'primary': True,
               'expected': case['expected'], 'prediction': result,
               'differences': {k: {'expected': v, 'predicted': result.get(k)} for k, v in case['expected'].items()
                               if (set(v) != set(result.get(k, [])) if k in ('clarification_ids', 'notice_ids') else v != result.get(k))}}
        if scores and result['status'] == 'ready_for_buyer_review':
            if fitted['mode'] == 'learned':
                row['scores'] = {'satisfied': min(r['probabilities']['satisfied'] for r in records if r['row']['question_id'] != 'material_group')}
            else:
                attrs = list(result['evidence']['attributes'].values())
                reference = min(v['reference_use']['confidence'] for v in attrs)
                evidence = min([reference, *(s['confidence'] for v in attrs for s in v['evidence_spans'])])
                coverage = 1.
                for q in result['diagnostics']['questions']:
                    if q['question_id'].startswith('attribute:'):
                        item = cache.storage[q['input_sha256']]
                        p = probabilities(item['bio'], fitted['temperatures']['bio'])
                        selected = item['mapping']['description_mask'] & (p.argmax(-1) == 0)
                        if bool(selected.any()):
                            coverage = min(coverage, float(p[selected, 0].min()))
                row['scores'] = {'evidence': evidence, 'reference': reference, 'coverage': coverage}
        rows.append(row)
        oracle.append(oracle_choice(runtime, case, config))
    return summarize(rows, oracle, config), rows


def observations(cache, cases, config):
    """Teacher labels are calibration targets only, never runtime features."""
    result = defaultdict(list)
    supervised = examples(cases, config)
    cache.preload([e['row'] for e in supervised])
    for e in supervised:
        row = e['row']
        item = cache.storage[row['text_sha256']]
        common = {k: e[k] for k in ('case_id', 'family_id', 'primary')}
        common.update(question_id=row['question_id'], option_ids=row['option_ids'], label=e['label'])
        result[row['question_id']].append({**common, 'logits': item['decisions'].tolist(), 'target': e['target']})
        if e['annotation'] is not None:
            targets = factor_targets(item['mapping'], e['annotation'], row['question_id'].removeprefix('attribute:'))
            for name, target in targets.items():
                valid = target >= 0
                for logits, truth in zip(item[name][valid].tolist(), target[valid].tolist()):
                    result[name].append({'logits': logits, 'target': truth})
        target = e['reference_target']
        ids = item['component_ids']
        if target is not None and ids:
            index = ids.index(target['component_id']) if target['mode'] == 'selected' else len(ids) + (target['mode'] == 'unresolved')
            result['permission'].append({**common, 'logits': item['permission'].tolist(), 'target': target['permission']})
            result['reference'].append({**common, 'logits': item['reference'].tolist(), 'target': int(index)})
    return dict(result)


def probability_records(samples, temperature):
    return [{**s, 'probabilities': probabilities(torch.tensor(s['logits']), temperature).tolist()} for s in samples]


def temperature_diagnostics(samples, temperature):
    before, after = probability_records(samples, 1.), probability_records(samples, temperature)
    slices = {}
    for name, key in [('target', 'target'), ('candidate_count', None)]:
        grouped = defaultdict(list)
        for s in after:
            grouped[str(s[key] if key else len(s['logits']))].append(s)
        slices[name] = {k: calibration_metrics(v) for k, v in grouped.items()}
    return {'temperature': temperature, 'before': calibration_metrics(before), 'after': calibration_metrics(after),
            'slices': slices, 'independent_family_intervals': False}


def fit_group_gates(samples, temperature, config):
    records = []
    for s in probability_records(samples, temperature):
        index = max(range(len(s['probabilities'])), key=s['probabilities'].__getitem__)
        records.append({**s, 'choice': s['option_ids'][index], 'confidence': s['probabilities'][index]})
    diagnostics = {key: select_outcome_threshold(records, key) for key in config['catalog']}
    return {key: value['threshold'] for key, value in diagnostics.items()}, diagnostics


def apply_candidate(base, candidate):
    fitted = copy.deepcopy(base)
    if fitted['mode'] == 'learned':
        for name, gates in fitted['gates'].items():
            if isinstance(gates, dict) and name != 'group':
                gates['satisfied'] = candidate['gates']['satisfied']
    else:
        fitted['gates'].update(candidate['gates'])
    return fitted


def fit_pipeline(cache, cases, config, model_identity, base):
    raw_summary, raw_rows = replay(cache, cases, config, model_identity, base, scores=True)
    verified = []
    for candidate in candidate_gates(raw_rows, base['mode']):
        fitted = apply_candidate(base, candidate)
        summary, rows = replay(cache, cases, config, model_identity, fitted)
        actual = sorted(r['case_id'] for r in rows if r['prediction']['status'] == 'ready_for_buyer_review')
        if actual != candidate['ready_case_ids']:
            raise ValueError('Gate search disagrees with actual complete inference')
        verified.append({'gates': candidate['gates'], 'summary': summary})
        if summary['eligible']:
            return fitted, {'eligible': True, 'raw': raw_summary, 'verified_candidates': verified}, rows
    rejected = copy.deepcopy(base)
    rejected['ready_enabled'] = False
    summary, rows = replay(cache, cases, config, model_identity, rejected)
    return rejected, {'eligible': False, 'reason': 'no_complete_pipeline_gate_meets_unchanged_targets',
                      'raw': raw_summary, 'verified_candidates': verified, 'disabled_ready': summary}, rows


def fit_bundle(model_directory, data_directory, output_directory, *, threads=2):
    """Fit only the two designated roles; keep the raw training bundle intact."""
    model_directory, data_directory, output_directory = map(Path, (model_directory, data_directory, output_directory))
    if output_directory.exists():
        raise ValueError('Preserve an existing calibration result')
    model, config, manifest = load_model(model_directory, threads=threads)
    report = json.loads((model_directory / 'training-report.json').read_text())
    if report['kind'] not in ('full_ownership_comparison_training', 'isolated_attribution_comparison_training') or report['training']['max_batches'] is not None:
        raise ValueError('Do not fit a truncated mechanism run as a quality candidate')
    if not report['selected_development']['pipelines']['hybrid']['base_eligible']:
        raise ValueError('Candidate fails the predeclared development semantic/recall floor')
    dataset_identity = file_sha256(data_directory / 'dataset-manifest.json')
    if report['dataset_manifest']['full_cohort_manifest_sha256'] != dataset_identity:
        raise ValueError('Calibration dataset differs from the pinned training cohort')
    roles, samples = {}, {}
    for role in ('calibration', 'threshold_selection'):
        local, cases, _ = load_partition(data_directory, role, allowed_splits={role})
        if local != config:
            raise ValueError('Calibration policy changed')
        roles[role] = cases
    if {c['family_id'] for c in roles['calibration']} & {c['family_id'] for c in roles['threshold_selection']}:
        raise ValueError('Calibration roles overlap')
    output_directory.mkdir(parents=True)
    cache = OutputCache(model)
    for role, cases in roles.items():
        samples[role] = observations(cache, cases, config)
        print(json.dumps({'event': 'comparison_calibration_collected', 'role': role,
                          'cases': len(cases), 'encoder_rows': cache.encoder_rows}), flush=True)
    temperatures, _ = defaults(config)
    diagnostics = {}
    for question in temperatures:
        rows = samples['calibration'].get(question, [])
        temperatures[question] = fit_temperature(rows)
        diagnostics[question] = temperature_diagnostics(rows, temperatures[question])
        print(json.dumps({'event': 'comparison_temperature_fit', 'question': question,
                          'observations': len(rows), 'temperature': temperatures[question]}), flush=True)
    groups, group_diagnostics = fit_group_gates(samples['threshold_selection']['material_group'],
                                               temperatures['material_group'], config)
    profiles, gate_diagnostics, records = {}, {}, {}
    for mode in ('learned', 'hybrid'):
        base = profile(config, manifest['model_identity'], mode, temperatures)
        base['gates']['group'] = groups
        profiles[mode], gate_diagnostics[mode], records[mode] = fit_pipeline(
            cache, roles['threshold_selection'], config, manifest['model_identity'], base)
    target = output_directory / 'model'
    shutil.copytree(model_directory, target)
    for mode, fitted in profiles.items():
        (target / ('calibration-' + mode + '.json')).write_text(json.dumps(fitted, indent=2, allow_nan=False) + '\n')
    fitted_manifest = write_manifest(target)
    if fitted_manifest['model_identity'] != manifest['model_identity']:
        raise ValueError('Calibration changed model weights or policy')
    verify_bundle(target)
    for mode in profiles:
        ComparisonRuntime(model, config, manifest['model_identity'], mode=mode, calibration=profiles[mode])
    result = {'version': VERSION, 'raw_artifact_identity': file_sha256(model_directory / 'bundle-manifest.json'),
              'model_identity': manifest['model_identity'], 'data_manifest_sha256': dataset_identity,
              'source_sha256': file_sha256(__file__), 'inference_source_sha256': inference_hashes(),
              'roles': {k: {'cases': len(v), 'primary_families': len(primary_cases(v))} for k, v in roles.items()},
              'temperature_diagnostics': diagnostics, 'group_gates': group_diagnostics,
              'pipelines': gate_diagnostics, 'reserved_evaluation_used': False, 'author100_used': False,
              'deployment_qualified': False, 'calibrated_artifact_identity': file_sha256(target / 'bundle-manifest.json')}
    (output_directory / 'calibration-report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (output_directory / 'threshold-results.json').write_text(json.dumps(records, allow_nan=False) + '\n')
    torch.save(cache.storage, output_directory / 'fitting-cache.pt')
    return result
