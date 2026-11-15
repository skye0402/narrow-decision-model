"""Known input structure and auxiliary labels kept on separate code paths."""
import hashlib
import json

import torch

from .evidence_span_data import ROLES, VERSION
from .neural import tokenize_row
from .relation_probe import json_field_span
from .renderer import compact

BIO_LABELS = ('O',) + tuple(prefix + '-' + role for role in ROLES for prefix in ('B', 'I'))


def layout(tokenizer, row, attribute, max_tokens=512):
    """Inference inputs depend only on rendered evidence and the asked attribute."""
    inputs, options = tokenize_row(tokenizer, row, max_tokens)
    text = row['text']
    start = text.index('\nEVIDENCE ') + len('\nEVIDENCE ')
    end = text.index('\nPOLICY ', start)
    evidence = json.loads(text[start:end])
    if compact(evidence) != text[start:end]:
        raise ValueError('Unexpected evidence serialization')
    offsets = tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True,
                        truncation=False)
    if offsets['input_ids'] != inputs['input_ids'][0].tolist():
        raise ValueError('Offset/model tokenization mismatch')
    positions = offsets['offset_mapping']
    special = offsets['special_tokens_mask']

    def mask(a, b):
        selected = [not sp and y > x and x < b and y > a for (x, y), sp in zip(positions, special)]
        covered = {i for active, (x, y) in zip(selected, positions) if active
                   for i in range(max(x, a), min(y, b))}
        if a < b and (not any(selected) or any(i not in covered and not text[i].isspace() for i in range(a, b))):
            raise ValueError('Incomplete input span coverage')
        return torch.tensor(selected, dtype=torch.bool)

    key_start, _ = json_field_span(evidence, ['description'], start)
    content_start = key_start + len(compact('description')) + 2  # colon, opening quote
    description = evidence['description']
    # Each original character can expand when JSON-escaped. The map is structural
    # and remains correct for quotes, backslashes, newlines and repeated numbers.
    boundaries = [content_start]
    for ch in description:
        boundaries.append(boundaries[-1] + len(compact(ch)) - 2)
    if text[content_start:boundaries[-1]] != compact(description)[1:-1]:
        raise ValueError('Description escape mapping mismatch')
    description_mask = mask(boundaries[0], boundaries[-1])
    reference_mask = torch.zeros_like(description_mask)
    if attribute is not None:
        for i, component in enumerate(evidence['resolved_reference']['components']):
            if attribute in component:
                a, b = json_field_span(evidence, ['resolved_reference', 'components', i, attribute], start)
                reference_mask |= mask(a, b)
    if bool((reference_mask & description_mask).any()):
        raise ValueError('Trusted fields overlap request text')
    return {'inputs': inputs, 'option_mask': options, 'description_mask': description_mask,
            'reference_mask': reference_mask, 'description': description,
            'description_boundaries': boundaries, 'offsets': positions, 'special': special}


def auxiliary_targets(mapping, annotation, attribute):
    """Training-only BIO targets; never passed to encoder or decision features."""
    if (annotation['version'] != VERSION
            or hashlib.sha256(mapping['description'].encode()).hexdigest() != annotation['description_sha256']):
        raise ValueError('Span annotations belong to another description')
    targets = torch.full((len(mapping['offsets']),), -100, dtype=torch.long)
    targets[mapping['description_mask']] = 0
    for span in annotation['spans']:
        a, b = span['start'], span['end']
        if not 0 <= a < b <= len(mapping['description']) or mapping['description'][a:b] != span['surface']:
            raise ValueError('Annotation does not cover its written value')
        if span['role'] != 'type' and span['attribute'] != attribute:
            continue
        a, b = mapping['description_boundaries'][a], mapping['description_boundaries'][b]
        selected = [i for i, ((x, y), special) in enumerate(zip(mapping['offsets'], mapping['special']))
                    if not special and y > x and x < b and y > a]
        if not selected or any(int(targets[i]) != 0 for i in selected):
            raise ValueError('Missing or overlapping BIO ownership target')
        for j, i in enumerate(selected):
            targets[i] = BIO_LABELS.index(('I-' if j else 'B-') + span['role'])
    return targets
