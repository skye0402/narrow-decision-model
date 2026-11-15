"""Separate span boundaries, item ownership, and evidence-state supervision."""
import json
import torch

from .span_format import BIO_LABELS as JOINT_LABELS, auxiliary_targets
from .relation_probe import json_field_span

BIO_LABELS = ('O', 'B-value', 'I-value', 'B-type', 'I-type')
OWNERS = ('requested_item', 'other_item')
STATES = ('current', 'obsolete', 'negated', 'ambiguous')


def factor_targets(mapping, annotation, attribute):
    joint = auxiliary_targets(mapping, annotation, attribute)
    bio = torch.full_like(joint, -100)
    owner, state = torch.full_like(joint, -100), torch.full_like(joint, -100)
    for i, value in enumerate(joint):
        if int(value) < 0:
            continue
        name = JOINT_LABELS[int(value)]
        if name == 'O':
            bio[i] = 0
            continue
        prefix, role = name.split('-', 1)
        bio[i] = BIO_LABELS.index(prefix + ('-type' if role == 'type' else '-value'))
        owner[i] = OWNERS.index('other_item' if role == 'background' else 'requested_item')
        if role != 'type':
            state[i] = STATES.index('current' if role == 'background' else role)
    return {'bio': bio, 'owner': owner, 'state': state}


def component_layout(row, mapping, width):
    text = row['text']
    start = text.index('\nEVIDENCE ') + len('\nEVIDENCE ')
    end = text.index('\nPOLICY ', start)
    evidence = json.loads(text[start:end])
    components = evidence['resolved_reference']['components']
    masks = []
    for i, component in enumerate(components):
        a, b = json_field_span(evidence, ['resolved_reference', 'components', i], start)
        mask = torch.zeros(width, dtype=torch.bool)
        for j, ((x, y), special) in enumerate(zip(mapping['offsets'], mapping['special'])):
            mask[j] = not special and y > x and x < b and y > a
        if not bool(mask.any()):
            raise ValueError('Trusted component tokens are absent')
        masks.append(mask)
    ids = [c['component_id'] for c in components]
    if len(ids) != len(set(ids)):
        raise ValueError('Ambiguous trusted component IDs')
    return ids, masks
