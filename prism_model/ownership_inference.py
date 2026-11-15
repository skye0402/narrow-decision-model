"""Decode real neural outputs to source-bound evidence; no gold inputs."""
from bisect import bisect_left, bisect_right
import hashlib
import math

import torch

from .hybrid_evidence import EvidenceSpan, Extraction, ReferenceUse
from .ownership_format import BIO_LABELS, OWNERS, STATES


def probabilities(logits, temperature):
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or not .25 <= temperature <= 10:
        raise ValueError('Invalid neural temperature')
    if logits.ndim < 1 or not bool(torch.isfinite(logits).all()):
        raise ValueError('Nonfinite model output')
    return (logits.detach().float().cpu() / temperature).softmax(-1)


def source_interval(mapping, first, last):
    """Invert structural JSON escaping, then trim only surrounding whitespace."""
    a, b = mapping['offsets'][first][0], mapping['offsets'][last][1]
    boundaries = mapping['description_boundaries']
    start = max(0, min(len(boundaries) - 1, bisect_right(boundaries, a) - 1))
    end = max(start, min(len(boundaries) - 1, bisect_left(boundaries, b)))
    text = mapping['description']
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start == end:
        raise ValueError('Empty neural source span')
    # A span never ends or starts inside a word: "P" inside "PNP" or "imetres"
    # inside "millimetres" become the whole token. This decodes the model's own
    # choice of location; it never adds a value the model did not point at.
    while start > 0 and text[start - 1].isalnum() and text[start].isalnum():
        start -= 1
    while start > 1 and text[start] == '.' and text[start - 1].isdigit():
        start -= 1
        while start > 0 and text[start - 1].isdigit():
            start -= 1
    while end < len(text) and text[end - 1].isalnum() and text[end].isalnum():
        end += 1
    return start, end, text[start:end]


def merge_fragments(spans, description):
    """Join adjacent same-attribute pieces split only by spaces, such as HTD + 5M."""
    merged = []
    for span in sorted(spans, key=lambda s: s.start):
        last = merged[-1] if merged else None
        if (last is not None and span.attribute is not None and last.attribute == span.attribute
                and last.role == span.role and last.owner == span.owner and not last.uncertainties
                and not span.uncertainties and last.end < span.start
                and description[last.end:span.start].strip() == ''):
            merged[-1] = EvidenceSpan(last.attribute, last.start, span.end, description[last.start:span.end],
                                      last.role, last.owner, min(last.confidence, span.confidence), ())
        else:
            merged.append(span)
    return tuple(merged)


def decode_attribute(output, index, attribute, *, temperatures, coverage_gate):
    mapping = output['mappings'][index]
    count = len(mapping['offsets'])
    bio = probabilities(output['bio'][index, :count], temperatures['bio'])
    owner = probabilities(output['owner'][index, :count], temperatures['owner'])
    state = probabilities(output['state'][index, :count], temperatures['state'])
    visible = mapping['description_mask'].nonzero().flatten().tolist()
    chosen = bio.argmax(-1).tolist()
    spans, uncertainties, diagnostic = [], [], []
    active, kind, issues = [], None, []

    def close():
        nonlocal active, kind, issues
        if not active:
            return
        start, end, surface = source_interval(mapping, active[0], active[-1])
        owners = [int(owner[i].argmax()) for i in active]
        owner_choice = owners[0]
        local_issues = list(issues)
        if len(set(owners)) != 1:
            local_issues.append('inconsistent_span_ownership')
        owner_name = OWNERS[owner_choice] if len(set(owners)) == 1 else 'unresolved'
        states = [int(state[i].argmax()) for i in active]
        state_choice = states[0]
        if kind == 'type':
            role, key = 'type', None
        else:
            if len(set(states)) != 1:
                local_issues.append('inconsistent_span_state')
            role, key = STATES[state_choice], attribute
            if role == 'current' and owner_name == 'other_item':
                role = 'background'
        signals = {'boundary': min(float(bio[i, chosen[i]]) for i in active),
                   'ownership': min(float(owner[i, owner_choice]) for i in active)}
        if kind != 'type':
            signals['state'] = min(float(state[i, state_choice]) for i in active)
        # Minimum signal is a gate statistic, not a claimed joint probability.
        confidence = min(signals.values())
        spans.append(EvidenceSpan(key, start, end, surface, role, owner_name, confidence, tuple(local_issues)))
        diagnostic.append({'start': start, 'end': end, 'surface': surface, 'attribute': key,
                           'source_state': None if kind == 'type' else STATES[state_choice],
                           'owner': owner_name, 'confidence_signals': signals, 'uncertainties': local_issues})
        active, kind, issues = [], None, []

    for i in visible:
        label = BIO_LABELS[chosen[i]]
        if label == 'O':
            close()
            if float(bio[i, 0]) < coverage_gate:
                a, b, _ = source_interval(mapping, i, i)
                uncertainties.append(f'uncertain_evidence_coverage:{a}:{b}')
            continue
        prefix, next_kind = label.split('-', 1)
        if prefix == 'B' or kind != next_kind:
            close()
            if prefix != 'B':
                issues.append('bio_span_missing_begin')
            kind = next_kind
        active.append(i)
    close()
    # Snapping can make neighbouring token runs cover the same word; identical
    # snapped spans are one span, and space-separated pieces are one value.
    unique = []
    for span in spans:
        if span not in unique:
            unique.append(span)
    return merge_fragments(unique, mapping['description']), tuple(uncertainties), diagnostic


def reference_use(output, index, *, temperatures, threshold):
    ids = output['component_ids'][index]
    pointer = probabilities(output['reference'][index], temperatures['reference'])
    permission = probabilities(output['permission'][index], temperatures['permission'])
    if pointer.shape != (len(ids) + 2,) or permission.shape != (2,):
        raise ValueError('Reference output vocabulary differs from trusted candidates')
    choice, allowed = int(pointer.argmax()), int(permission.argmax())
    confidence = min(float(pointer[choice]), float(permission[allowed]))
    details = {'component_choices': [*ids, 'not_used', 'unresolved'], 'component_probabilities': pointer.tolist(),
               'permission_probabilities': permission.tolist()}
    if not ids:
        # Record availability is a known source fact: no absent source can fill
        # missing request facts, regardless of linguistic permission to use it.
        return ReferenceUse('not_used', None, 'not_permitted', 1.), details
    if confidence < threshold or choice == len(ids) + 1 or (choice == len(ids) and allowed):
        return ReferenceUse('unresolved', None, 'unresolved', confidence,
                            ('uncertain_or_inconsistent_reference_use',)), details
    if choice == len(ids):
        return ReferenceUse('not_used', None, 'not_permitted', confidence), details
    return ReferenceUse('selected', ids[choice], 'permitted' if allowed else 'not_permitted', confidence), details


def bind_extraction(description, spans, use, uncertainties, model_identity):
    # Queries can each return the same generic type span. Deduplicate identical
    # spans conservatively, retaining the lowest confidence, never a vote.
    unique = {}
    for span in spans:
        key = (span.attribute, span.start, span.end, span.surface, span.role, span.owner, span.uncertainties)
        if key not in unique or span.confidence < unique[key].confidence:
            unique[key] = span
    return Extraction(hashlib.sha256(description.encode()).hexdigest(), model_identity,
                      tuple(unique.values()), use, tuple(sorted(set(uncertainties))))
