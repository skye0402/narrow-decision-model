"""Crossed evidence scenarios, attribute supervision, and a family-level 80/20 split.

Labels derive from scenario state, never a model or the rules comparator. The
twenty-percent evaluation cohort has different inner wording, not just wrappers.
This is still assistant-authored synthesis, not independent human validation.
"""
from collections import Counter
import copy
import json
from pathlib import Path
import random

from .data_v2 import PRODUCTS, META, values, attribute_text
from .orchestrator import resolve_reference
from .provenance import file_sha256, verify_files
from .renderer import requirement_row
from .task_data import (case_config, expected_route, opaque, question_examples as
                        holistic_examples, stable_json)

VERSION = 'evidence-data-1'
SPLITS = ('train', 'development', 'calibration', 'threshold_selection', 'prototype_evaluation')
COUNTS = dict(zip(SPLITS, (3900, 600, 600, 900, 1500)))
MODES = ('complete', 'omit', 'bare_type', 'reference_conflict', 'reference_partial_conflict',
         'reference_only', 'reference_without_same', 'reference_ambiguous',
         'reference_selected', 'text_conflict', 'obsolete', 'obsolete_one',
         'unrelated', 'unrelated_one', 'negated', 'ambiguous_value', 'hp',
         'roller_dimensions', 'policy_counterfactual')
ATTRIBUTES = {
    'power_kw': 'rated power in kW', 'voltage_v': 'voltage in V', 'phases': 'phase count',
    'diameter_mm': 'roller diameter in mm', 'length_mm': 'length in mm',
    'gear_ratio': 'gear ratio', 'shaft_diameter_mm': 'shaft diameter in mm',
    'width_mm': 'width in mm', 'belt_profile': 'belt profile', 'bore_mm': 'bearing bore in mm',
    'outer_diameter_mm': 'bearing outer diameter in mm',
    'sensing_distance_mm': 'sensing distance in mm', 'supply_voltage_v': 'supply voltage in V',
    'output_type': 'PNP/NPN output type',
}


def alternative_value(key, current, rng):
    # Agreeing and disagreeing requests must draw from the same plausible pool;
    # unusual "+7" values would give the model a shortcut for finding conflicts.
    candidate=values(rng)[key]
    while candidate==current:
        candidate=values(rng)[key]
    return candidate


def compose(labels):
    """Semantic composition, separate from confidence gating and overall selection."""
    if not labels or any(x not in ('conflicting', 'unclear', 'missing', 'satisfied') for x in labels):
        raise ValueError('Complete typed evidence judgments required')
    for outcome in ('conflicting', 'unclear', 'missing', 'satisfied'):
        if outcome in labels:
            return outcome


def attribute_row(context, config, policy, attribute):
    if attribute not in policy['required']:
        raise ValueError('Attribute is not required by supplied policy')
    question = {
        'id': 'attribute:' + attribute,
        'instructions': (f'Assess only {ATTRIBUTES[attribute]}. Resolve the requested component from raw evidence. '
                         'Ignore unrelated, obsolete, and negated specifications.'),
        'options': list(config['identification_options']),
    }
    return requirement_row(context, config, {**policy, 'required': [attribute]}, question)


def question_examples(cases, config, *, attributes=True):
    from .evidence_format import canonical_row
    output = holistic_examples(cases, config)
    if not attributes:
        return [{**e, 'row': (row := canonical_row(e['row'])), 'target': row['option_ids'].index(e['label'])} for e in output]
    for case in cases:
        labels = case['semantic_labels'].get('attributes', {})
        if not labels:
            continue
        local = case_config(case, config)
        group = case['semantic_labels']['group']
        policy_id = case.get('policy_id') or local['default_policies'][group]
        policy = local['policies'][policy_id]
        context = {'request': case['request'], 'reference': resolve_reference(case['request'], local),
                   'policy_id': policy_id}
        for key in policy['required']:
            row = attribute_row(context, local, policy, key)
            label = labels[key]
            output.append({k: case[k] for k in ('case_id', 'family_id', 'primary')} | {
                'row': row, 'label': label, 'target': row['option_ids'].index(label)})
    return [{**e, 'row': (row := canonical_row(e['row'])), 'target': row['option_ids'].index(e['label'])} for e in output]


def specification(key, value, rng, heldout):
    if not heldout:
        return attribute_text(key, value, rng)
    # Different attribute constructions as well as different sentence frames.
    return {
        'power_kw': f'rated output is {value} kW', 'voltage_v': f'operates at {value} V',
        'phases': f'{value}-phase electrical connection',
        'diameter_mm': f'{value} mm roller diameter', 'length_mm': f'{value} mm long',
        'gear_ratio': f'reduction of {value}:1', 'shaft_diameter_mm': f'{value} mm shaft diameter',
        'width_mm': f'{value} mm wide', 'belt_profile': f'cross-section profile {value}',
        'bore_mm': f'{value} mm bore', 'outer_diameter_mm': f'{value} mm outside diameter',
        'sensing_distance_mm': f'detects at {value} mm', 'supply_voltage_v': f'powered by {value} V',
        'output_type': f'output configuration {value}',
    }[key]


def scenario_family(config, group, mode, number, seed, split):
    rng = random.Random(f'evidence:{seed}:{number}')
    heldout = split == 'prototype_evaluation'
    token = opaque(f'evidence:{seed}:{number}')
    if group in META:
        a, b = rng.sample(list(PRODUCTS), 2)
        first, second = rng.choice(PRODUCTS[a][2]), rng.choice(PRODUCTS[b][2])
        phrases = {
            'insufficient_evidence': (f'Either {first} or {second}; the choice is unresolved.',
                                      'Replacement component needed; we cannot identify which part yet.'),
            'mixed_scope': (f'{first} and {second} are both requested.',
                            f'Purchase {first} together with installation services.'),
            'no_suitable_group': ('A hydraulic pump is the only purchase.',
                                  'Inspection labor for the motor only; no replacement goods.'),
        }
        result = []
        for variant in (0, 1):
            text = phrases[group][variant]
            lead = {'train': 'Purchase request', 'development': 'Requisition content',
                    'calibration': 'Requested procurement scope', 'threshold_selection': 'Buying note',
                    'prototype_evaluation': 'Actual order line'}[split]
            text = f'{lead}: {text}'
            case = make_case(config, token, variant, split, text, group, None, {}, [], None, mode)
            result.append(case)
        return result
    kind, required, names = PRODUCTS[group]
    facts = {k: values(rng)[k] for k in required}
    candidates = [(key, rows[0]) for key, rows in config['references'].items()
                  if len(rows) == 1 and rows[0]['kind'] == kind
                  and not config.get('reference_integrity', {}).get(key)]
    reference_id = 'REF-E4-' + token
    component = {'component_id': 'COMP-E4-' + token, 'kind': kind, **facts}
    if candidates and rng.random() < .5:
        reference_id, component = copy.deepcopy(rng.choice(candidates))
        facts = {k: component[k] for k in required}
    changed = rng.choice(required)
    alternate = alternative_value(changed, facts[changed], rng)
    if mode == 'hp' and kind != 'motor': mode = 'omit'
    if mode == 'roller_dimensions' and kind != 'roller': mode = 'ambiguous_value'
    if mode == 'policy_counterfactual' and kind != 'motor': mode = 'omit'
    # Randomize which preselected primary variant is the complete counterpart.
    bad_primary = rng.choice((False, True))
    name = rng.choice(names)
    type_code = rng.choice(('230', '400', 'AX-' + str(rng.randrange(100, 10000))))
    result = []
    for variant in (0, 1):
        bad = bool(variant) != bad_primary
        local = random.Random(f'evidence-writing:{seed}:{number}:{variant}')
        supplied = dict(facts)
        labels = {k: 'satisfied' for k in required}
        references = []; ref = None; prefix = ''; suffix = ''; policy_id = None
        pieces = None; overall = None
        ref_mode = mode.startswith('reference_')
        if ref_mode:
            references = [copy.deepcopy(component)]; ref = reference_id
            prefix = (local.choice(('Like-for-like replacement. ', 'Replace with an identical unit. ',
                                    'Use the same model as the existing part. ')) if not heldout
                      else 'Match the existing unit exactly. ')
        if mode in ('reference_ambiguous', 'reference_selected'):
            references.append({**component, 'component_id': component['component_id'] + '-B', changed: alternate})
            local.shuffle(references)
        if bad:
            if mode == 'omit':
                supplied.pop(changed); labels[changed] = 'missing'
            elif mode == 'bare_type':
                supplied.clear(); labels = {k: 'missing' for k in required}
            elif mode in ('reference_conflict', 'reference_partial_conflict'):
                supplied[changed] = alternate; labels[changed] = 'conflicting'
                if mode == 'reference_partial_conflict':
                    supplied = {changed: alternate}
            elif mode == 'reference_only':
                supplied.clear()
            elif mode == 'reference_without_same':
                prefix = ''; supplied.clear(); labels = {k: 'missing' for k in required}
            elif mode == 'reference_ambiguous':
                supplied.clear(); labels[changed] = 'unclear'; overall = 'unclear'
            elif mode == 'reference_selected':
                supplied.clear()
                suffix = (' Do not choose the ' + specification(changed, alternate, local, heldout) + ' version.'
                          if local.random() < .5 else ' Choose the ' + specification(changed, facts[changed], local, heldout) + ' version.')
            elif mode == 'text_conflict':
                suffix = ' Also required: ' + specification(changed, alternate, local, heldout) + '.'
                labels[changed] = 'conflicting'
            elif mode in ('obsolete', 'unrelated'):
                supplied.clear(); labels = {k: 'missing' for k in required}
            elif mode in ('obsolete_one', 'unrelated_one'):
                supplied.pop(changed); labels[changed] = 'missing'
            elif mode == 'ambiguous_value':
                supplied.pop(changed); labels[changed] = 'unclear'
                suffix = (' The choice between ' + specification(changed, facts[changed], local, heldout)
                          + ' and ' + specification(changed, alternate, local, heldout) + ' has not been made.')
            elif mode == 'hp':
                supplied.pop('power_kw'); labels['power_kw'] = 'missing'; suffix = ' Power is 7.5 hp.'
            elif mode == 'roller_dimensions':
                pieces = f"{facts['diameter_mm']} x {facts['length_mm']} mm"
                supplied.clear()
                labels = {k: 'unclear' for k in required}
            elif mode == 'policy_counterfactual':
                supplied.pop('phases'); labels['phases'] = 'missing'
        if mode == 'policy_counterfactual':
            supplied.pop('phases', None)
            if not bad:
                policy_id = 'PILOT-MOTORS-NO-PHASE-EXPERIMENT'
                labels.pop('phases')
            else:
                labels['phases'] = 'missing'
        if mode == 'roller_dimensions' and not bad:
            pieces = f"Ø{facts['diameter_mm']} × {facts['length_mm']} mm"
        if pieces is None:
            ordered = list(supplied); local.shuffle(ordered)
            pieces = ', '.join(specification(k, supplied[k], local, heldout) for k in ordered)
        if mode.startswith(('obsolete', 'unrelated')):
            old = ', '.join(specification(k, facts[k], local, heldout) for k in required)
            prefix = (f'The old unit had {old}. Those specifications are obsolete. ' if mode.startswith('obsolete')
                      else f'A different machine has {old}; these facts do not apply to this purchase. ')
            prefix += 'Current requested specifications: '
        if mode == 'negated':
            suffix = ' Do not supply the ' + specification(changed, alternate, local, heldout) + ' alternative.'
        item = name + (' type ' + type_code if mode == 'bare_type' or local.random() < .35 else '')
        if heldout:
            frame = local.choice(('For procurement we need {item}. {prefix}Required details: {spec}.{suffix}',
                                  'Please get {item}: {prefix}{spec}.{suffix}',
                                  '{prefix}Order line / {item} / {spec}.{suffix}',
                                  'The part to buy is {item}. {prefix}{spec}.{suffix}'))
        else:
            frames = {
                'train': ('{prefix}Supply a {item}. {spec}.{suffix}', '{prefix}{item}, {spec}.{suffix}',
                          'Need {item}. {prefix}{spec}.{suffix}', '{prefix}Purchase: {item}; {spec}.{suffix}'),
                'development': ('Requisition details / {item}. {prefix}{spec}.{suffix}',
                                'For this request, get {item}. {prefix}{spec}.{suffix}'),
                'calibration': ('Technical order entry: {item}. {prefix}{spec}.{suffix}',
                                'Spare-part requirement is {item}: {prefix}{spec}.{suffix}'),
                'threshold_selection': ('Please process an order for {item}. {prefix}{spec}.{suffix}',
                                       'Buying note for {item}; {prefix}{spec}.{suffix}'),
            }
            frame = local.choice(frames[split])
        text = frame.format(item=item, prefix=prefix, spec=pieces, suffix=suffix).replace('..', '.')
        if local.random() < .2: text += ' For the conveyor actuator replacement.'
        overall = overall or compose(list(labels.values()))
        case = make_case(config, token, variant, split, text, group, overall, labels, references, ref, mode)
        if policy_id: case['policy_id'] = policy_id
        case['intended_facts'] |= {'current_specifications': supplied, 'reference_components': references,
            'base_specifications': facts, 'target_attribute': changed,
            'contradictory_current_values': {changed: [facts[changed], alternate]} if mode=='text_conflict' and bad else {},
            'ambiguous_current_values': ({changed: [facts[changed], alternate]} if mode=='ambiguous_value' and bad else
                {k: [facts['diameter_mm'],facts['length_mm']] for k in required} if mode=='roller_dimensions' and bad else {}),
            'reference_substitution_allowed': bool(ref_mode and not (mode=='reference_without_same' and bad)),
            'selected_reference_component': (None if mode=='reference_ambiguous' and bad else component['component_id']) if ref else None,
            'ignored_specifications': facts if mode.startswith(('obsolete','unrelated')) else {},
            'unsupported_units': {'power_hp': 7.5} if mode=='hp' and bad else {}}
        case['expected'] = expected_route(case, config)
        result.append(case)
    return result


def intended_attribute_labels(case, config):
    """Audit normalized scenario state independently of the text-rendering branches.

    This is a gold-data check, never an inference-time extraction/rules fallback.
    """
    group=case['semantic_labels']['group']
    if group not in PRODUCTS:return {}
    policy=config['policies'][case.get('policy_id') or config['default_policies'][group]]
    state=case['intended_facts'];current=state['current_specifications'];references=state['reference_components']
    selected=next((r for r in references if r['component_id']==state['selected_reference_component']),None)
    labels={}
    for key in policy['required']:
        if key in state['contradictory_current_values']:
            labels[key]='conflicting'
        elif key in state['ambiguous_current_values']:
            labels[key]='unclear'
        elif key in current:
            labels[key]='conflicting' if selected is not None and current[key]!=selected[key] else 'satisfied'
        elif state['reference_substitution_allowed']:
            sources=[selected] if selected is not None else references
            possible={r[key] for r in sources if key in r}
            labels[key]='missing' if not possible else 'satisfied' if len(possible)==1 else 'unclear'
        else:labels[key]='missing'
    return labels


def make_case(config, token, variant, split, text, group, identification, attributes, references, ref, mode):
    request = {'item_id': f'E4-{token}-{variant}', 'description': text, 'quantity': 2, 'unit': 'EA',
               'plant': '1100', 'required_date': '2026-12-15'}
    if ref: request['reference_id'] = ref
    case = {'case_id': request['item_id'], 'family_id': 'E4-F-' + token, 'primary': variant == 0,
            'variant': variant, 'split': split, 'contract_version': config['contract_version'],
            'author': 'crossed_facts_generator', 'review_status': 'not_independently_reviewed',
            'reference_fixture': references, 'request': request,
            'writing_template_id': 'evidence-inner-heldout' if split == 'prototype_evaluation' else 'evidence-inner-fit',
            'semantic_labels': {'group': group, 'identification': identification, 'attributes': attributes},
            'intended_facts': {'blueprint': mode}, 'tags': ['synthetic', 'crossed_evidence', 'slice:' + mode]}
    case['expected'] = expected_route(case, config)
    return case


def exposed_development(config):
    root = Path(__file__).resolve().parents[1]
    output = []
    # Labels already exposed during the review remain development evidence.
    for row in json.loads((root/'research/task-v3-review/cases.json').read_text()):
        request = copy.deepcopy(row['request']); key = 'REVIEW-' + row['case_id']
        case = make_case(config, key, 0, 'development', request['description'], row['expected']['group'],
                         row['expected']['identification'], {},
                         copy.deepcopy(config['references'].get(request.get('reference_id'), [])),
                         request.get('reference_id'), 'exposed_review')
        case.update(case_id=key, family_id='EXPOSED-REVIEW-' + row['case_id'], request=request)
        if row.get('policy_id'): case['policy_id'] = row['policy_id']
        case['expected'] = expected_route(case, config); output.append(case)
    previous = root/'.cache/task-v3/data/development.jsonl'
    if previous.is_file():
        for row in map(json.loads, previous.read_text().splitlines()):
            if row['case_id'].startswith(('EXPOSED-', 'CRITICAL-')):
                output.append({**copy.deepcopy(row), 'split': 'development'})
    return output


def generate(directory, *, counts=None, seed=120101, include_exposed=True):
    directory = Path(directory)
    if directory.exists(): raise ValueError('Use a new evidence dataset directory')
    counts = dict(COUNTS if counts is None else counts)
    if set(counts) != set(SPLITS) or any(type(n) is not int or n < 9 for n in counts.values()):
        raise ValueError('Declare each role with at least nine families')
    if sum(counts.values()) != 5 * counts['prototype_evaluation']:
        raise ValueError('Exactly 20 percent of prospective families must be evaluation')
    config = json.loads((Path(__file__).resolve().parents[1]/'data/pilot_config_v0_4.json').read_text())
    directory.mkdir(parents=True)
    (directory/'pilot_config.json').write_text(json.dumps(config, indent=2)+'\n')
    total = sum(counts.values()); family_ids = list(range(total))
    random.Random(seed).shuffle(family_ids); cursor = 0; groups = (*PRODUCTS, *META)
    manifest = {'version': VERSION, 'scope': 'synthetic_prototype', 'seed': seed,
                'prospective_families': total, 'model_development_fraction': .8, 'evaluation_fraction': .2,
                'split_unit': 'scenario_family', 'generator_sha256': file_sha256(__file__),
                'independent_human_challenge_present': False, 'gold_labels_reviewed': False,
                'evaluation_inner_writing_separate': True, 'partitions': {}, 'files_sha256': {}}
    known=exposed_development(config) if include_exposed else []
    exposed_texts={' '.join(c['request']['description'].casefold().split()) for c in known}
    for split, count in counts.items():
        cases = [];replaced=0
        for index, number in enumerate(family_ids[cursor:cursor+count]):
            group = groups[index % len(groups)]; mode = MODES[(index // len(groups)) % len(MODES)]
            attempt=0
            while True:
                generated=scenario_family(config,group,mode,number+attempt*total,seed,split)
                if not any(' '.join(c['request']['description'].casefold().split()) in exposed_texts for c in generated):break
                attempt+=1
                if attempt>100:raise ValueError('Unable to separate exposed development wording')
            replaced+=bool(attempt);cases.extend(generated)
        cursor += count
        exposed = known if split == 'development' else []
        cases.extend(exposed)
        random.Random(f'{seed}:shuffle:{split}').shuffle(cases)
        path = directory/(split+'.jsonl'); path.write_text(''.join(stable_json(c)+'\n' for c in cases))
        manifest['partitions'][split] = {'file': path.name, 'cases': len(cases), 'prospective_families': count,
            'exposed_cases': len(exposed), 'families': len({c['family_id'] for c in cases}),
            'families_replaced_for_exposed_description_collision':replaced,
            'group_counts': dict(Counter(c['semantic_labels']['group'] for c in cases)),
            'identification_counts': dict(Counter(c['semantic_labels']['identification'] for c in cases)),
            'question_rows': len(question_examples(cases, config))}
    manifest['files_sha256'] = {p.name: file_sha256(p) for p in directory.iterdir() if p.is_file()}
    (directory/'dataset-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def load_partition(directory, split, *, allowed_splits):
    directory = Path(directory)
    manifest = json.loads((directory/'dataset-manifest.json').read_text())
    if manifest['version'] != VERSION or split not in allowed_splits: raise ValueError('Wrong dataset or role')
    verify_files(directory, manifest['files_sha256'], exact=True, exclude=('dataset-manifest.json',))
    rows = [json.loads(line) for line in (directory/manifest['partitions'][split]['file']).read_text().splitlines()]
    if any(c['split'] != split for c in rows): raise ValueError('Case role differs from partition')
    return json.loads((directory/'pilot_config.json').read_text()), rows, manifest
