"""Second prospective cohort: varied scope, fact ownership, and policy changes.

The first cohort remains intact and its reserved predictions remain unopened.
This writer consumes normalized scenario facts; no predictions or external
challenge requests are inputs. The external AI package remains outside training.
"""
from collections import Counter
import copy
import json
from pathlib import Path
import random

from . import evidence_data as base
from .data_v2 import PRODUCTS, META
from .provenance import file_sha256
from .task_data import expected_route, opaque, stable_json

COUNTS = base.COUNTS
SPLITS = base.SPLITS
MODES = (*base.MODES, 'reference_conflict', 'reference_partial_conflict',
         'reference_selected', 'obsolete_one', 'unrelated_one')


def specification(key, value, rng, heldout):
    """Explicit units and meanings; no implicit hp conversion or type decoding."""
    fit = {
        'power_kw': (f'{value} kW', f'power {value} kW', f'output {value} kilowatts'),
        'voltage_v': (f'{value} V', f'{value} volts', f'voltage: {value} V'),
        'phases': (( 'single phase' if value == 1 else 'three phase'), f'{value} ph', f'{value}-phase'),
        'diameter_mm': (f'diameter {value} mm', f'roller dia {value} mm', f'roller OD {value} mm'),
        'length_mm': (f'length {value} mm', f'{value} mm long', f'L={value} mm'),
        'gear_ratio': (f'ratio {value}:1', f'{value}:1 reduction', f'reduction ratio {value}:1'),
        'shaft_diameter_mm': (f'shaft diameter {value} mm', f'{value} mm shaft', f'shaft dia {value} mm'),
        'width_mm': (f'width {value} mm', f'{value} mm wide', f'W={value} mm'),
        'belt_profile': (f'profile {value}', f'{value} profile', f'belt section {value}'),
        'bore_mm': (f'bore {value} mm', f'{value} mm bore', f'inner diameter {value} mm'),
        'outer_diameter_mm': (f'outer diameter {value} mm', f'OD {value} mm', f'outside dia {value} mm'),
        'sensing_distance_mm': (f'sensing distance {value} mm', f'{value} mm sensing range', f'detection distance {value} mm'),
        'supply_voltage_v': (f'supply {value} V', f'{value} V supply', f'supply voltage {value} volts'),
        'output_type': (f'{value} output', f'output: {value}', f'{value} switching output'),
    }
    test = {
        'power_kw': (f'rated output = {value} kW', f'power rating of {value} kW'),
        'voltage_v': (f'operating voltage = {value} V', f'electrical rating {value} volts'),
        'phases': (f'phase count = {value}', f'electrical connection: {value} phase'),
        'diameter_mm': (f'roller diameter = {value} millimetres', f'{value} mm across the roller'),
        'length_mm': (f'overall length = {value} mm', f'length of {value} millimetres'),
        'gear_ratio': (f'reduction of {value}:1', f'gear reduction = {value}:1'),
        'shaft_diameter_mm': (f'output shaft diameter = {value} mm', f'shaft of {value} mm diameter'),
        'width_mm': (f'overall width = {value} mm', f'width of {value} millimetres'),
        'belt_profile': (f'cross-section profile {value}', f'belt profile designation = {value}'),
        'bore_mm': (f'internal bore = {value} mm', f'bore diameter of {value} millimetres'),
        'outer_diameter_mm': (f'external diameter = {value} mm', f'outside diameter of {value} millimetres'),
        'sensing_distance_mm': (f'detection reach = {value} mm', f'detects at {value} millimetres'),
        'supply_voltage_v': (f'powered by {value} V', f'power supply rating = {value} volts'),
        'output_type': (f'output configuration {value}', f'switching arrangement = {value}'),
    }
    return rng.choice((test if heldout else fit)[key])


def scope_text(group, rng, heldout):
    a, b = rng.sample(list(PRODUCTS), 2)
    first, second = rng.choice(PRODUCTS[a][2]), rng.choice(PRODUCTS[b][2])
    fit = {
        'insufficient_evidence': (
            f'Either a {first} or a {second}; choice unresolved.',
            f'It might be a {first}, perhaps a {second}. The item has not been identified.',
            f'Do we need a {first} or a {second}? We have not decided.',
            f'Need a spare for the conveyor. The broken component is not identified.',
            'Replacement part type 230; the kind of part is unknown.',
            f'One replacement unit, either {first} / {second}; selection pending.',
            f'The technician has not confirmed whether to buy a {first} or a {second}.',
            'Please source the failed part; no information about its kind yet.'),
        'mixed_scope': (
            f'Supply a {first} and a {second}.', f'Buy a {first} plus a {second}.',
            f'Requesting two items: {first}; {second}.',
            f'A {first} with installation services is requested.',
            f'Purchase a {first} together with a {second}.',
            f'Please order one {first} and one {second}, both required.',
            f'This line includes a {first} and commissioning labor.',
            f'The repair needs a {first}, and we also need a {second}.'),
        'no_suitable_group': (
            'Inspect the motor only; no goods.', 'Motor inspection service only, no replacement goods.',
            'Conveyor maintenance labor only; do not purchase any parts.',
            'Buy a hydraulic pump.', 'Control valve for the conveyor hydraulics.',
            'Office chair for the maintenance desk.', 'Order printer paper.',
            'Laptop required for the workshop.',
            f'Diagnose the {first}; service work only, no replacement component.',
            'Inspection of the drive belt, labor only.'),
    }
    test = {
        'insufficient_evidence': (
            f'The choice of spare is still open: a {first} versus a {second}.',
            'An unclassified replacement component is needed after the breakdown.',
            f'Part identity awaiting confirmation; could be {first} or {second}.',
            f'One of these will be ordered: {first}, {second}. No selection yet.'),
        'mixed_scope': (
            f'Please procure the following together: {first}, {second}.',
            f'Two kinds of replacement goods are needed, a {first} along with a {second}.',
            f'The order covers both the {first} itself and the fitting service.',
            f'Goods and services on one line: {first}; installation labor.'),
        'no_suitable_group': (
            'Only a hydraulic pump is to be purchased for the conveyor.',
            'Arrange an inspection of the motor. No physical goods are ordered.',
            'The purchase is a replacement control valve.',
            f'Pay for inspection of the {first}, without ordering a spare.',
            'Workshop desk chair to be procured.'),
    }
    return rng.choice((test if heldout else fit)[group])


def write_product(case, rng, heldout):
    group = case['semantic_labels']['group']
    state = case['intended_facts']; mode = state['blueprint']
    kind, required, names = PRODUCTS[group]
    item = rng.choice(names)
    if mode == 'bare_type' or rng.random() < .3:
        item += ' type ' + rng.choice(('230', '400', 'ZX-' + str(rng.randrange(100, 9900))))
    current = state['current_specifications']; keys = list(current); rng.shuffle(keys)
    parts = [specification(k, current[k], rng, heldout) for k in keys]
    if mode == 'roller_dimensions':
        facts = state['base_specifications']
        if state['ambiguous_current_values']:
            parts = [f"{facts['diameter_mm']} x {facts['length_mm']} mm"]
        else:
            parts = [f"Ø{facts['diameter_mm']} × {facts['length_mm']} mm"]
    for key, vals in state['contradictory_current_values'].items():
        parts.append('Also required: ' + specification(key, vals[1], rng, heldout))
    if mode == 'ambiguous_value':
        for key, vals in state['ambiguous_current_values'].items():
            parts.append('The choice between ' + specification(key, vals[0], rng, heldout)
                         + ' and ' + specification(key, vals[1], rng, heldout) + ' is unresolved')
    if state['unsupported_units']:
        parts.append(f"{state['unsupported_units']['power_hp']} hp")
    reference = ''
    if state['reference_substitution_allowed']:
        reference = rng.choice(('Like-for-like replacement. ', 'Same model as the existing unit. ',
                                'Replace with an identical unit. ') if not heldout else
                               ('Match the existing component exactly. ', 'An identical spare is required. '))
    if mode == 'reference_selected' and not current:
        key = state['target_attribute']
        selected = next(c for c in state['reference_components'] if c['component_id'] == state['selected_reference_component'])
        other = next(c for c in state['reference_components'] if c['component_id'] != selected['component_id'])
        parts.append(('Choose the ' + specification(key, selected[key], rng, heldout) + ' version') if rng.random() < .5
                     else 'Do not choose the ' + specification(key, other[key], rng, heldout) + ' version')
    if mode == 'negated':
        key = state['target_attribute']; value = base.alternative_value(key, state['base_specifications'][key], rng)
        parts.append('Do not supply the ' + specification(key, value, rng, heldout) + ' alternative')
    main = reference + 'Requested: ' + item + '. ' + '; '.join(parts) + '.'
    ignored = state['ignored_specifications']
    if ignored:
        background = ', '.join(specification(k, ignored[k], rng, heldout) for k in required)
        if mode.startswith('obsolete'):
            note = rng.choice((f'The old unit had {background}; those specifications are obsolete.',
                               f'Obsolete specification, do not use: {background}.',
                               f'Previous drawing: {background}. This drawing is superseded.'))
        else:
            note = rng.choice((f'Unrelated equipment elsewhere has {background}; not this purchase.',
                               f'A different machine uses {background}. These facts do not apply to the requested item.',
                               f'For another part only: {background}; ignore these values for this order.'))
        placement = rng.choice(('before', 'after', 'between'))
        if placement == 'before': main = note + ' ' + main
        elif placement == 'after': main += ' ' + note
        else: main = reference + 'Requested: ' + item + '. ' + note + ' Current requirements: ' + '; '.join(parts) + '.'
        state['background_position'] = placement
    # Equipment context must not turn a single requested part into a mixed order.
    if rng.random() < .25:
        other = rng.choice([g for g in PRODUCTS if g != group]); other_name = rng.choice(PRODUCTS[other][2])
        main += rng.choice((f' For the machine driven by an existing {other_name}; only the {kind} is ordered.',
                            f' The {other_name} is not being replaced.'))
    return main.replace('..', '.')


def family(config, group, mode, number, seed, split):
    cases = base.scenario_family(config, group, mode, number, seed, split)
    heldout = split == 'prototype_evaluation'
    # Policy sensitivity is crossed with facts, not announced in the user text.
    if group == 'MG_MOTORS' and random.Random(f'policy:{seed}:{number}').random() < .5:
        facts = cases[0]['intended_facts']['base_specifications']
        for relaxed in (False, True):
            case = copy.deepcopy(cases[0]); case['variant'] = len(cases); case['primary'] = False
            case['case_id'] += '-policy-' + str(int(relaxed)); case['request']['item_id'] = case['case_id']
            case['request'].pop('reference_id', None); case['reference_fixture'] = []
            case.pop('policy_id', None)
            if relaxed: case['policy_id'] = 'PILOT-MOTORS-NO-PHASE-EXPERIMENT'
            case['intended_facts'] = {'blueprint': 'policy_counterfactual', 'current_specifications': {k:v for k,v in facts.items() if k != 'phases'},
                'reference_components': [], 'base_specifications': facts, 'target_attribute': 'phases',
                'contradictory_current_values': {}, 'ambiguous_current_values': {}, 'reference_substitution_allowed': False,
                'selected_reference_component': None, 'ignored_specifications': {}, 'unsupported_units': {}}
            case['semantic_labels']['attributes'] = base.intended_attribute_labels(case, config)
            case['semantic_labels']['identification'] = base.compose(list(case['semantic_labels']['attributes'].values()))
            cases.append(case)
    for case in cases:
        # Identical writing for each strict/relaxed policy pair.
        variant = 2 if case['variant'] >= 2 else case['variant']
        rng = random.Random(f'ownership-writer:{seed}:{number}:{variant}')
        text = scope_text(group, rng, heldout) if group in META else write_product(case, rng, heldout)
        prefixes = {
            'train': ('', 'Order: ', 'Need this for maintenance: ', 'Purchase request: '),
            'development': ('Requisition content: ', 'Spare request / '),
            'calibration': ('Technical order entry: ', 'Procurement note: '),
            'threshold_selection': ('Buying request: ', 'Please process: '),
            'prototype_evaluation': ('Workshop purchase entry: ', 'For the next repair: ')}
        text = rng.choice(prefixes[split]) + text
        case['request'].update(description=text, quantity=rng.choice((1,2,3,4,6)), plant=rng.choice(config['plants']))
        case['author'] = 'facts_first_ownership_writer_v2'
        case['writing_template_id'] = 'ownership-inner-heldout' if heldout else 'ownership-inner-fit'
        case['tags'] += ['ownership_writer_v2']
        case['expected'] = expected_route(case, config)
    return cases


def generate(directory, *, counts=None, seed=130201, include_exposed=True):
    directory = Path(directory); counts = dict(COUNTS if counts is None else counts)
    if directory.exists(): raise ValueError('Use a new dataset directory')
    if set(counts) != set(SPLITS) or any(type(n) is not int or n < 15 for n in counts.values()):
        raise ValueError('Declare every role with at least fifteen families')
    if sum(counts.values()) != 5 * counts['prototype_evaluation']:
        raise ValueError('Exactly 20 percent of prospective families must be evaluation')
    config = json.loads((Path(__file__).resolve().parents[1]/'data/pilot_config_v0_4.json').read_text())
    known = base.exposed_development(config) if include_exposed else []
    seen = {' '.join(c['request']['description'].casefold().split()) for c in known}
    total = sum(counts.values()); numbers = list(range(total)); random.Random(seed).shuffle(numbers)
    directory.mkdir(parents=True); (directory/'pilot_config.json').write_text(json.dumps(config,indent=2)+'\n')
    manifest = {'version': base.VERSION, 'scope':'synthetic_prototype', 'seed':seed, 'prospective_families':total,
        'model_development_fraction':.8, 'evaluation_fraction':.2, 'split_unit':'scenario_family',
        'generator_path':'prism_model/evidence_data_v2.py', 'generator_sha256':file_sha256(__file__),
        'base_generator_sha256':file_sha256(base.__file__), 'independent_human_challenge_present':False,
        'gold_labels_reviewed':False, 'evaluation_inner_writing_separate':True,
        'external_ai_requests_used_for_training_or_selection':False, 'partitions':{}, 'files_sha256':{}}
    cursor=0; groups=(*PRODUCTS,*PRODUCTS,*META)
    for split,count in counts.items():
        cases=[]; replaced=0; split_texts=set()
        for index,number in enumerate(numbers[cursor:cursor+count]):
            group=groups[index%len(groups)]; mode=MODES[(index//len(groups))%len(MODES)]
            for attempt in range(200):
                new=family(config,group,mode,number+attempt*total,seed,split)
                texts={' '.join(c['request']['description'].casefold().split()) for c in new}
                if not texts & seen: break
            else: raise ValueError('Cannot generate disjoint scenario writing')
            replaced+=bool(attempt); split_texts.update(texts); cases.extend(new)
        seen.update(split_texts)
        cursor+=count; exposed=known if split=='development' else []; cases.extend(exposed)
        random.Random(f'{seed}:shuffle:{split}').shuffle(cases)
        path=directory/(split+'.jsonl');path.write_text(''.join(stable_json(c)+'\n' for c in cases))
        manifest['partitions'][split]={'file':path.name,'cases':len(cases),'prospective_families':count,
            'exposed_cases':len(exposed),'families':len({c['family_id'] for c in cases}),
            'families_replaced_for_description_collision':replaced,
            'group_counts':dict(Counter(c['semantic_labels']['group'] for c in cases)),
            'identification_counts':dict(Counter(c['semantic_labels']['identification'] for c in cases)),
            'question_rows':len(base.question_examples(cases,config))}
    manifest['files_sha256']={p.name:file_sha256(p) for p in directory.iterdir() if p.is_file()}
    (directory/'dataset-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
