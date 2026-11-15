"""Factor ownership/state labels and merge existing, immutable family roles."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random

from .evidence_data import load_partition, question_examples, SPLITS
from .evidence_span_data import CLAUSES
from .provenance import file_sha256
from .task_data import stable_json

VERSION = 'ownership-comparison-data-1'
ROOT = Path(__file__).resolve().parents[1]
TRAIN_CLAUSES = {
    'current': ('For the item actually being purchased: ', 'Specify the ordered unit as ',
                'Our replacement must meet ', 'These ratings belong to the requested item: '),
    'background': ('Another installation, remaining in service, uses ', 'Not the item being ordered: another device has ',
                   'Elsewhere on the machine, a separate retained component has ',
                   'For comparison only, the adjacent equipment uses ', 'Do not order that neighboring unit; it has '),
    'obsolete': ('That earlier requirement is canceled; it specified ', 'Before the requirements were revised, they called for ',
                 'The superseded version of this part had ', 'Withdrawn figures for this item: '),
    'negated': ('For this purchase, specifically exclude ', 'This configuration must not be ordered: ',
                'The alternative we rejected is ', 'An unacceptable rating for this order is '),
    'ambiguous': ('Neither alternative has been confirmed: ', 'A decision is outstanding between ',
                  'The requester has left these options open: ', 'We still need to choose one of '),
    'type': ('The product identification code is ', 'The manufacturer type identifier is ',
              'Identifying label ', 'This number identifies the model: '),
}


def augment_training(case, epoch, seed):
    """Transform known writer clauses, carrying exact span offsets along with it.

    Only new annotated training cases are eligible. This is data augmentation,
    not an inference parser; it never locates ratings by matching numbers.
    """
    if case['split'] != 'train' or 'evidence_annotations' not in case:
        return case
    rng = random.Random(f'ownership-wording:{seed}:{epoch}:{case["family_id"]}')
    if rng.random() < .25:
        return case
    text = case['request']['description']
    replacements = []
    for role, phrases in CLAUSES['train'].items():
        replacement = rng.choice(TRAIN_CLAUSES[role])
        for phrase in phrases:
            cursor = 0
            while (start := text.find(phrase, cursor)) >= 0:
                replacements.append((start, start + len(phrase), replacement))
                cursor = start + len(phrase)
    replacements.sort()
    if any(a[1] > b[0] for a, b in zip(replacements, replacements[1:])):
        raise ValueError('Overlapping writer-clause substitutions')
    result = copy.deepcopy(case)
    pieces, cursor = [], 0
    for a, b, replacement in replacements:
        pieces.extend((text[cursor:a], replacement))
        cursor = b
    pieces.append(text[cursor:])
    transformed = ''.join(pieces)
    for span in result['evidence_annotations']['spans']:
        if any(max(span['start'], a) < min(span['end'], b) for a, b, _ in replacements):
            raise ValueError('A source clause unexpectedly overlaps an annotated value')
        delta = sum(len(value) - (b - a) for a, b, value in replacements if b <= span['start'])
        span['start'] += delta
        span['end'] += delta
        if transformed[span['start']:span['end']] != span['surface']:
            raise ValueError('Augmentation changed a recorded source value')
    result['request']['description'] = transformed
    result['evidence_annotations']['description_sha256'] = hashlib.sha256(transformed.encode()).hexdigest()
    return result


def sorted_case(case):
    result = {**case, 'reference_fixture': sorted(case['reference_fixture'], key=lambda c: c['component_id'])}
    return result


def examples(cases, config, *, epoch=None, seed=79):
    cases = [sorted_case(augment_training(c, epoch, seed) if epoch is not None else c) for c in cases]
    result = question_examples(cases, config)
    lookup = {c['case_id']: c for c in cases}
    for e in result:
        c = lookup[e['case_id']]
        e['annotation'] = c.get('evidence_annotations') if e['row']['question_id'].startswith('attribute:') else None
        state = c.get('intended_facts', {})
        e['reference_target'] = None
        if e['row']['question_id'] == 'material_group' and 'reference_substitution_allowed' in state:
            refs = c['reference_fixture']
            selected = state.get('selected_reference_component')
            permitted = bool(state['reference_substitution_allowed'])
            if selected is not None:
                if selected not in {r['component_id'] for r in refs}:
                    raise ValueError('Constructed selected reference is absent')
                mode = 'selected'
            elif permitted and len(refs) == 1:
                mode, selected = 'selected', refs[0]['component_id']
            else:
                mode = 'unresolved' if permitted and len(refs) > 1 else 'not_used'
            e['reference_target'] = {'permission': int(permitted), 'mode': mode, 'component_id': selected}
    return result


def merge(directory, first, second):
    directory, first, second = map(Path, (directory, first, second))
    if directory.exists():
        raise ValueError('Preserve existing combined data')
    directory.mkdir(parents=True)
    config = None
    parts, seen_ids, seen_families = {}, set(), {}
    for role in SPLITS:
        rows = []
        for parent in (first, second):
            current, cases, _ = load_partition(parent, role, allowed_splits={role})
            if config is not None and current != config:
                raise ValueError('Different purchasing contracts')
            config = current
            rows.extend(cases)
        for c in rows:
            if c['case_id'] in seen_ids or (c['family_id'] in seen_families and seen_families[c['family_id']] != role):
                raise ValueError('Combined case/family role overlap')
            seen_ids.add(c['case_id'])
            seen_families[c['family_id']] = role
        path = directory / (role + '.jsonl')
        path.write_text(''.join(stable_json(c) + '\n' for c in rows))
        parts[role] = {'file': path.name, 'cases': len(rows), 'families': len({c['family_id'] for c in rows}),
                       'annotated_cases': sum('evidence_annotations' in c for c in rows),
                       'group_counts': dict(Counter(c['semantic_labels']['group'] for c in rows))}
    if len(seen_families) != 5 * parts['prototype_evaluation']['families']:
        raise ValueError('Combined inventory must retain an exact 80/20 family split')
    (directory / 'pilot_config.json').write_text(json.dumps(config, indent=2) + '\n')
    manifest = {'version': 'evidence-data-1', 'generator': VERSION, 'scope': 'synthetic_prototype',
                'source_generator_sha256': file_sha256(__file__),
                'parent_manifest_sha256': [file_sha256(p / 'dataset-manifest.json') for p in (first, second)],
                'partitions': parts, 'total_families': len(seen_families), 'evaluation_fraction': .2,
                'split_unit': 'scenario_family', 'original_400_retained': True,
                'exposed_author100_included': False, 'previous_evaluation_included': False,
                'independently_human_reviewed': False,
                'files_sha256': {p.name: file_sha256(p) for p in directory.iterdir()}}
    (directory / 'dataset-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest
