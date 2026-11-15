"""Batched real outputs replayed through both complete, predicted-group paths.

Caching accelerates quality evaluation only. CPU qualification must call the
actual neural model, never this replay wrapper.
"""
import copy
from collections import Counter

import torch

from .comparison_runtime import ComparisonJudge, ComparisonRuntime
from .evidence_data import ATTRIBUTES
from .evidence_orchestrator import group_row, requirement_row
from .metrics import classification
from .orchestrator import Judgment, resolve_reference
from .ownership_data import examples
from .ownership_inference import decode_attribute
from .provided_evaluation import decision_routing_metrics
from .task_data import case_config
from .task_iteration import predicted_group_identification


class OutputCache:
    is_quality_replay = True
    def __init__(self, model):
        self.model, self.storage, self.encoder_rows = model, {}, 0

    def eval(self):
        self.model.eval()
        return self

    def preload(self, rows, batch_size=8):
        missing = {r['text_sha256']: r for r in rows if r['text_sha256'] not in self.storage}
        pending = list(missing.values())
        for start in range(0, len(pending), batch_size):
            chosen = pending[start:start + batch_size]
            with torch.inference_mode():
                output = self.model.forward_batch(chosen)
            self.encoder_rows += len(chosen)
            for i, row in enumerate(chosen):
                mapping = output['mappings'][i]
                count = len(mapping['offsets'])
                item = {key: output[key][i, :count].detach().float().cpu().clone() for key in ('bio', 'owner', 'state')}
                item.update(decisions=output['decisions'][i, :len(row['option_ids'])].detach().float().cpu().clone(),
                            permission=output['permission'][i].detach().float().cpu().clone(),
                            reference=output['reference'][i].detach().float().cpu().clone(),
                            mapping=copy.deepcopy(mapping), component_ids=output['component_ids'][i],
                            option_ids=row['option_ids'])
                if not all(bool(torch.isfinite(item[k]).all()) for k in ('bio', 'owner', 'state', 'decisions', 'permission', 'reference')):
                    raise ValueError('Nonfinite cached neural output')
                self.storage[row['text_sha256']] = item

    def forward_batch(self, rows):
        self.preload(rows)
        items = [self.storage[r['text_sha256']] for r in rows]
        if any(r['option_ids'] != item['option_ids'] for r, item in zip(rows, items)):
            raise ValueError('Cached option vocabulary changed')
        width = max(len(x['mapping']['offsets']) for x in items)
        choices = max(len(x['decisions']) for x in items)
        result = {}
        for key in ('bio', 'owner', 'state'):
            result[key] = torch.stack([torch.nn.functional.pad(x[key], (0, 0, 0, width - len(x[key]))) for x in items])
        result.update(decisions=torch.stack([torch.nn.functional.pad(x['decisions'], (0, choices - len(x['decisions'])),
                                                                      value=torch.finfo(torch.float32).min) for x in items]),
                      permission=torch.stack([x['permission'] for x in items]), reference=[x['reference'] for x in items],
                      mappings=[x['mapping'] for x in items], component_ids=[x['component_ids'] for x in items])
        return result


def primary_cases(cases):
    result = [c for c in cases if c['primary']]
    if len({c['family_id'] for c in result}) != len(result):
        raise ValueError('Repeated primary scenario family')
    return result


def oracle_choice(runtime, case, config):
    """Correct policy only; evidence, ownership and reference use remain predicted."""
    group = case['semantic_labels']['group']
    if group not in config['default_policies'] or case['semantic_labels']['identification'] is None:
        return None
    local = case_config(case, config)
    ref = case['request'].get('reference_id')
    if ref in local['references']:
        local['references'][ref] = sorted(local['references'][ref], key=lambda c: c['component_id'])
    policy_id = case.get('policy_id') or local['default_policies'][group]
    policy = local['policies'][policy_id]
    context = {'request': case['request'], 'reference': resolve_reference(case['request'], local, case.get('lookup_failure', False)),
               'policy_id': policy_id}
    if context['reference']['status'] == 'unavailable' or context['reference']['integrity_issues']:
        return None
    refs = context['reference']['components']
    if len(refs) == 1 and refs[0]['kind'] != policy['kind']:
        return 'conflicting'
    judge = ComparisonJudge(runtime, local)
    judge.judge_group(group_row(context, local), context)
    return judge.judge_requirement(requirement_row(context, local, policy), context, policy).choice


def summarize(rows, oracle, config):
    truth_groups = [r['expected']['group'] for r in rows]
    predicted_groups = [r['prediction']['group'] for r in rows]
    groups = classification(truth_groups, predicted_groups,
                            [*config['catalog'], *([None] if None in truth_groups or None in predicted_groups else [])])
    supported = [r for r in rows if r['expected']['group'] in config['default_policies']]
    group_accuracy = sum(r['expected']['group'] == r['prediction']['group'] for r in supported) / len(supported) if supported else 0.
    routing = decision_routing_metrics(rows)
    actual = predicted_group_identification(rows, config['identification_options'])
    oracle_rows = [(r['expected']['identification'], c) for r, c in zip(rows, oracle) if r['expected']['identification'] is not None]
    oracle_metric = classification([a for a, _ in oracle_rows], [b for _, b in oracle_rows],
                                   [*config['identification_options'], None])
    oracle_f1 = sum(oracle_metric['per_label'][k]['f1'] for k in config['identification_options']) / len(config['identification_options'])
    actual_f1 = actual['macro_f1_over_true_identification_outcomes']
    safety = [r for r in rows if r['expected']['group'] in ('mixed_scope', 'insufficient_evidence', 'no_suitable_group')
              or r['expected']['identification'] in ('conflicting', 'unclear') or r['expected']['status'] == 'evaluation_unavailable']
    false = sum(r['prediction']['status'] == 'ready_for_buyer_review' for r in safety)
    base = group_accuracy >= .90 and oracle_f1 >= .85 and actual_f1 >= .85 and (routing['ready_recall'] or 0.) >= .60
    precision = routing['ready_precision']
    eligible = (base and (precision['estimate'] or 0.) >= .95 and (precision['lower'] or 0.) >= .90
                and precision['support'] >= 150 and false == 0)
    return {'primary_families': len(rows), 'group': groups, 'supported_group_accuracy': group_accuracy,
            'routing': routing, 'actual_identification': actual, 'actual_identification_macro_f1': actual_f1,
            'oracle_identification': oracle_metric, 'oracle_identification_macro_f1': oracle_f1,
            'safety_cases': len(safety), 'safety_false_ready': false, 'base_eligible': base, 'eligible': eligible,
            'selection_rank': [base, -false, -routing['false_ready'], actual_f1, oracle_f1]}


def evaluate(model, cases, config, *, model_identity='unexported-development-checkpoint', calibrations=None):
    cases = primary_cases(cases)
    cache = OutputCache(model)
    supervised = examples(cases, config)
    cache.preload([e['row'] for e in supervised])
    summaries, records = {}, {}
    for mode in ('learned', 'hybrid'):
        runtime = ComparisonRuntime(cache, config, model_identity, mode=mode,
                                    calibration=(calibrations or {}).get(mode), diagnostic=calibrations is None)
        rows, oracle = [], []
        for case in cases:
            result = runtime.evaluate(case['request'], config=case_config(case, config),
                                      policy_id=case.get('policy_id'), lookup_failure=case.get('lookup_failure', False))
            rows.append({'case_id': case['case_id'], 'family_id': case['family_id'], 'primary': True,
                         'expected': case['expected'], 'prediction': result,
                         'differences': {key: {'expected': value, 'predicted': result.get(key)} for key, value in case['expected'].items()
                                         if (set(value) != set(result.get(key, [])) if key in ('clarification_ids', 'notice_ids')
                                             else value != result.get(key))}})
            oracle.append(oracle_choice(runtime, case, config))
        summaries[mode] = summarize(rows, oracle, config)
        records[mode] = rows
    # Exact source-span ownership/state metrics use actual decoded spans, not
    # owner predictions read at teacher-supplied boundaries.
    correct, expected, predicted = Counter(), Counter(), Counter()
    for e in supervised:
        if e['annotation'] is None:
            continue
        attribute = e['row']['question_id'].removeprefix('attribute:')
        output = cache.forward_batch([e['row']])
        spans, _, _ = decode_attribute(output, 0, attribute, temperatures={'bio': 1., 'owner': 1., 'state': 1.}, coverage_gate=0.)
        gold = set()
        for s in e['annotation']['spans']:
            if s['attribute'] == attribute or s['role'] == 'type':
                owner = 'other_item' if s['role'] == 'background' else 'requested_item'
                gold.add((s['start'], s['end'], s['role'], owner))
        actual = {(s.start, s.end, s.role, s.owner) for s in spans}
        for owner in ('requested_item', 'other_item'):
            a, b = {s for s in gold if s[3] == owner}, {s for s in actual if s[3] == owner}
            correct[owner] += len(a & b)
            expected[owner] += len(a)
            predicted[owner] += len(b)
    ownership = {k: {'correct': correct[k], 'expected': expected[k], 'predicted': predicted[k],
                     'recall': correct[k] / expected[k] if expected[k] else 0.,
                     'precision': correct[k] / predicted[k] if predicted[k] else 0.}
                 for k in ('requested_item', 'other_item')}
    return {'pipelines': summaries, 'exact_source_ownership': ownership, 'encoder_rows': cache.encoder_rows,
            'quality_uses_cached_real_outputs': True, 'qualifies_cpu_performance': False}, records, cache
