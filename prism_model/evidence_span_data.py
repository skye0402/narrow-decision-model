"""Writer-recorded evidence ownership spans; no request parser supplies labels."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random

from .data_v2 import PRODUCTS, values
from .evidence_data import SPLITS, alternative_value, make_case
from .provenance import file_sha256
from .reference_curriculum import labels
from .task_data import expected_route, opaque, stable_json, words

VERSION = 'evidence-ownership-spans-1'
ROLES = ('current', 'background', 'obsolete', 'negated', 'ambiguous', 'type')
MODES = ('rating', 'ownership', 'obsolete', 'negated', 'missing_type', 'active_conflict', 'ambiguous')
COUNTS = dict(zip(SPLITS, (1568, 336, 112, 224, 560)))
ATTRIBUTES = [(group, key) for group, (_, keys, _) in PRODUCTS.items() for key in keys]

FRAMES = {
    'train': ('Order {item}. Match the installed part. ', 'Buy {item} as an identical spare. ',
              '{item} needed like for like. ', 'Replacement order: {item}; keep the recorded specification. '),
    'development': ('Please source {item} with the same ratings as the existing component. ',
                    'We need an exact equivalent of the recorded {item}. ',
                    'Purchase line for {item}, matching the installed unit. '),
    'calibration': ('An identical {item} should be obtained. ', 'For this repair, procure {item} to the existing specification. '),
    'threshold_selection': ('Supply {item} without changing its specifications. ', 'Obtain a like-for-like {item} for maintenance. '),
    'prototype_evaluation': ('Purchasing should get {item} equivalent to the installed item. ',
                             'The requested goods are {item}; the existing ratings must be retained. ',
                             'Source an unchanged replacement {item}. '),
}
CLAUSES = {
    'train': {
        'current': ('For the requested part: ', 'Required specifications: ', 'The new part must have '),
        'background': ('For a different component, not being bought: ', 'The neighboring unit, which stays: '),
        'obsolete': ('Old specifications, superseded and not requested: ', 'The retired part had '),
        'negated': ('Do not supply the version with ', 'The rejected specification was '),
        'ambiguous': ('We have not chosen between ', 'The requested rating is undecided: '),
        'type': ('Type identifier ', 'Model code '),
    },
    'development': {
        'current': ('The unit to purchase is specified as ', 'For our actual replacement we require '),
        'background': ('A separate retained machine is specified as ', 'These figures concern another machine only: '),
        'obsolete': ('Previous figures have been replaced; they were ', 'Before revision the ratings were '),
        'negated': ('The buyer must exclude ', 'The option we ruled out had '),
        'ambiguous': ('Confirmation is pending for the alternatives ', 'Either rating is possible; nobody has decided: '),
        'type': ('Catalog type label ', 'Equipment model number '),
    },
    'calibration': {
        'current': ('Our intended purchase has ', 'Actual ordered specifications: '),
        'background': ('The other retained item has ',), 'obsolete': ('Outdated ratings, now withdrawn: ',),
        'negated': ('We specifically refuse ',), 'ambiguous': ('The undecided alternatives are ',),
        'type': ('Product type designation ',),
    },
    'threshold_selection': {
        'current': ('The part being sourced needs ', 'For delivery, specify '),
        'background': ('For comparison, a different existing unit uses ',),
        'obsolete': ('Discard the previous requirement of ',), 'negated': ('We are not ordering ',),
        'ambiguous': ('A choice remains open between ',), 'type': ('The type code reads ',),
    },
    'prototype_evaluation': {
        'current': ('Use these specifications for the new purchase: ', 'The replacement we actually want is '),
        'background': ('A nearby installation is staying in service and has ',),
        'obsolete': ('The figures below belong to an earlier, canceled revision: ',),
        'negated': ('The following configuration is expressly unwanted: ',),
        'ambiguous': ('The requester still cannot choose from ',), 'type': ('Its identifying model label is ',),
    },
}


class Writer:
    def __init__(self):
        self.text = ''
        self.spans = []

    def append(self, text):
        self.text += text

    def marked(self, text, *, attribute, role, value):
        if not text or role not in ROLES:
            raise ValueError('Nonempty known-role span required')
        start = len(self.text)
        self.text += text
        self.spans.append({'start': start, 'end': len(self.text), 'surface': text,
                           'attribute': attribute, 'role': role, 'value': value})


def fact(writer, key, value, role, rng):
    """Each value surface is chosen before insertion; no substring search."""
    number = f'{value:g}' if isinstance(value, (int, float)) else str(value)
    if isinstance(value, (int, float)) and rng.random() < .15:
        number = words(value)
    prefixes = {
        'power_kw': ('rated power ', 'output '), 'voltage_v': ('voltage ', 'electrical rating '),
        'phases': ('phase count ', 'connection '), 'diameter_mm': ('roller diameter ', 'roller dia '),
        'length_mm': ('length ', 'L='), 'gear_ratio': ('reduction ', 'ratio '),
        'shaft_diameter_mm': ('shaft diameter ', 'shaft dia '), 'width_mm': ('width ', 'W='),
        'belt_profile': ('belt profile ', 'cross section '), 'bore_mm': ('bore ', 'ID '),
        'outer_diameter_mm': ('outer diameter ', 'OD '), 'sensing_distance_mm': ('sensing distance ', 'range '),
        'supply_voltage_v': ('supply voltage ', 'power supply '), 'output_type': ('output type ', 'switching output '),
    }
    suffix = ' kW' if key == 'power_kw' else ' V' if key in ('voltage_v', 'supply_voltage_v') else \
             '-phase' if key == 'phases' else ':1' if key == 'gear_ratio' else \
             ' mm' if key.endswith('_mm') else ''
    if rng.random() < .4:
        suffix = suffix.replace(' ', '')
    writer.append(rng.choice(prefixes[key]))
    writer.marked(number + suffix, attribute=key, role=role, value=value)


def family(config, group, attribute, mode, number, seed, role):
    if (group, attribute) not in ATTRIBUTES or mode not in MODES or role not in SPLITS:
        raise ValueError('Unknown ownership family specification')
    rng = random.Random(f'span-facts:{seed}:{number}:{group}:{attribute}')
    token = opaque(f'span-facts:{seed}:{number}:{group}:{attribute}')
    kind, required, names = PRODUCTS[group]
    base = {key: values(rng)[key] for key in required}
    a, b = base[attribute], alternative_value(attribute, base[attribute], rng)
    primary = rng.randrange(4)
    cases = []
    for request_variant in range(2):
        for reference_variant in range(2):
            variant = 2 * request_variant + reference_variant
            writing = random.Random(f'span-writing:{seed}:{number}')
            writer = Writer()
            writer.append(writing.choice(FRAMES[role]).format(item=writing.choice(names)))
            writer.append(writing.choice(CLAUSES[role]['type']))
            # Deliberate repeated value: the identifier cannot supply a rating.
            identifier = f'{a:g}' if isinstance(a, (int, float)) else str(a)
            writer.marked(identifier, attribute=None, role='type', value=identifier)
            writer.append('. ')
            current = {key: value for key, value in base.items() if key != attribute}
            refs = [{'component_id': 'A1', 'kind': kind, **base}]
            refs[0][attribute] = (a, b)[reference_variant]
            contradictions, ambiguity = {}, {}
            chunks = [('current', list(current.items()))]
            if mode == 'missing_type':
                refs[0][attribute] = a
                if reference_variant:
                    refs[0].pop(attribute)
                if not request_variant:
                    current[attribute] = a
                    chunks.append(('current', [(attribute, a)]))
            elif mode in ('rating', 'ownership', 'obsolete', 'negated'):
                current[attribute] = (a, b)[request_variant]
                if mode == 'rating':
                    chunks.append(('current', [(attribute, current[attribute])]))
                else:
                    other_role = 'background' if mode == 'ownership' else mode
                    chunks.extend([('current' if request_variant == 0 else other_role, [(attribute, a)]),
                                   (other_role if request_variant == 0 else 'current', [(attribute, b)])])
            elif mode == 'active_conflict':
                current[attribute] = a
                chunks.extend([('current', [(attribute, a)]),
                               ('current', [(attribute, b if request_variant else a)])])
                if request_variant:
                    contradictions[attribute] = [a, b]
            else:
                if request_variant:
                    ambiguity[attribute] = [a, b]
                    chunks.append(('ambiguous', [(attribute, a), (attribute, b)]))
                else:
                    current[attribute] = a
                    chunks.append(('current', [(attribute, a)]))
            writing.shuffle(chunks)
            for evidence_role, items in chunks:
                writer.append(writing.choice(CLAUSES[role][evidence_role]))
                writing.shuffle(items)
                for i, (key, value) in enumerate(items):
                    if i:
                        writer.append(' or ' if evidence_role == 'ambiguous' else ', ')
                    fact(writer, key, value, evidence_role, writing)
                writer.append('. ')
            state = {'blueprint': 'span_' + mode, 'target_attribute': attribute,
                     'base_specifications': base, 'current_specifications': current,
                     'reference_components': refs, 'selected_reference_component': 'A1',
                     'reference_substitution_allowed': True, 'contradictory_current_values': contradictions,
                     'ambiguous_current_values': ambiguity, 'ignored_specifications': {}, 'unsupported_units': {}}
            case = make_case(config, token, variant, role, writer.text.rstrip(), group, None, {},
                             copy.deepcopy(refs), 'REF-SPAN-' + token, 'span_' + mode)
            case.update(case_id='SPAN-' + token + '-' + str(variant), family_id='SPAN-F-' + token,
                        primary=variant == primary, author='assistant_writer_recorded_spans',
                        writing_template_id='span-narrative:' + role, intended_facts=state,
                        evidence_annotations={'version': VERSION, 'description_sha256': hashlib.sha256(case['request']['description'].encode()).hexdigest(),
                                              'spans': writer.spans})
            case['request']['item_id'] = case['case_id']
            attrs, overall = labels(case, config)
            case['semantic_labels'].update(attributes=attrs, identification=overall)
            case['expected'] = expected_route(case, config)
            validate_annotations(case)
            cases.append(case)
    for left, right in ((0, 1), (2, 3)):
        assert cases[left]['request']['description'] == cases[right]['request']['description']
    return cases


def validate_annotations(case):
    annotation = case['evidence_annotations']
    text = case['request']['description']
    if annotation['version'] != VERSION or hashlib.sha256(text.encode()).hexdigest() != annotation['description_sha256']:
        raise ValueError('Annotation description identity changed')
    end = 0
    for span in annotation['spans']:
        a, b = span['start'], span['end']
        if type(a) is not int or type(b) is not int or not end <= a < b <= len(text):
            raise ValueError('Overlapping or invalid ownership spans')
        if text[a:b] != span['surface'] or span['role'] not in ROLES:
            raise ValueError('Ownership annotation differs from written text')
        if span['role'] == 'type' and span['attribute'] is not None:
            raise ValueError('A type identifier cannot supply an attribute')
        if span['role'] != 'type' and span['attribute'] not in PRODUCTS[case['semantic_labels']['group']][1]:
            raise ValueError('Unknown annotated attribute')
        end = b
    return annotation


def generate(directory, *, seed=261106, counts=None):
    directory = Path(directory)
    counts = dict(COUNTS if counts is None else counts)
    if set(counts) != set(SPLITS) or any(type(n) is not int or n < 1 for n in counts.values()):
        raise ValueError('Declare all five family roles')
    if sum(counts.values()) != 5 * counts['prototype_evaluation']:
        raise ValueError('Exactly 20% of new families must remain reserved')
    if directory.exists():
        raise ValueError('Preserve existing data')
    directory.mkdir(parents=True)
    config = json.loads((Path(__file__).resolve().parents[1] / 'data/pilot_config_v0_4.json').read_text())
    (directory / 'pilot_config.json').write_text(json.dumps(config, indent=2) + '\n')
    numbers = list(range(sum(counts.values())))
    random.Random(seed).shuffle(numbers)
    partitions = {}
    cursor = 0
    signatures = set()
    for role, count in counts.items():
        cases = []
        for i, number in enumerate(numbers[cursor:cursor + count]):
            group, attribute = ATTRIBUTES[i % len(ATTRIBUTES)]
            mode = MODES[(i // len(ATTRIBUTES)) % len(MODES)]
            for attempt in range(1000):
                rows = family(config, group, attribute, mode, f'{number}:{attempt}', seed, role)
                signature = stable_json({'group': group, 'attribute': attribute, 'mode': mode,
                                         'states': [c['intended_facts'] for c in rows]})
                if signature not in signatures:
                    signatures.add(signature)
                    cases.extend(rows)
                    break
            else:
                raise ValueError('Insufficient distinct constructed states; reduce declared cohort')
        cursor += count
        random.Random(f'span-order:{seed}:{role}').shuffle(cases)
        path = directory / (role + '.jsonl')
        path.write_text(''.join(stable_json(case) + '\n' for case in cases))
        partitions[role] = {'file': path.name, 'cases': len(cases), 'families': count,
                            'group_counts': dict(Counter(c['semantic_labels']['group'] for c in cases)),
                            'identification_counts': dict(Counter(c['semantic_labels']['identification'] for c in cases)),
                            'blueprint_counts': dict(Counter(c['intended_facts']['blueprint'] for c in cases))}
    manifest = {'version': 'evidence-data-1', 'generator': VERSION, 'scope': 'synthetic_prototype',
                'seed': seed, 'total_families': sum(counts.values()), 'split_unit': 'scenario_family',
                'model_development_fraction': .8, 'evaluation_fraction': .2,
                'source_generator_sha256': file_sha256(__file__), 'partitions': partitions,
                'writer_recorded_spans': True, 'independent_human_authorship': False,
                'new_cohort_constructed_state_signatures_unique': True,
                'exposed_author100_included': False, 'previous_evaluation_included': False,
                'files_sha256': {p.name: file_sha256(p) for p in directory.iterdir() if p.is_file()}}
    (directory / 'dataset-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest
