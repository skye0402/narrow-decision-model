"""Gold span views for the supplied 400 natural training requests.

The supplied requests were composed from intended facts by a known writer
(data/ai_training_complement_2026_11_05/source/build.py), but they carry no
source spans. Training therefore used the teacher's extraction for them, and
the teacher reads a second live value ("The second current job sheet instead
specifies 12 V ... Both entries remain active") as another item's value.

This module re-derives spans from each case's intended facts: it renders the
writer's own phrasings for every intended value, locates them in the text and
keeps the smallest sub-span that normalizes back to that exact value. A case
is skipped unless every intended fact is found exactly once without overlap.
The originals stay byte-identical; annotated copies are training views of the
same family. Labels are unchanged and nothing is read from a model.
"""
import copy
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
import re

from .evidence_span_data import VERSION as SPAN_VERSION
from .hybrid_evidence import UnresolvedValue, normalize

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = 'ai-cedar-gold-span-view'
ROLES = {'withdrawn': 'obsolete', 'obsolete_only': 'obsolete', 'negated_value': 'negated',
         'explicitly unrelated': 'background', 'unrelated_only': 'background'}


def writer():
    spec = importlib.util.spec_from_file_location('supplied_writer', ROOT / 'data/ai_training_complement_2026_11_05/source/build.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def intended(case):
    """(attribute, value, role) facts the writer placed in the text."""
    facts = case['intended_facts']
    result = []
    for key, value in facts['current_specifications'].items():
        result.append((key, value, 'current'))
    for key, values in facts['contradictory_current_values'].items():
        result.extend((key, v, 'current') for v in values[1:])
    for key, values in facts['ambiguous_current_values'].items():
        if isinstance(values, dict):
            return None  # Unitless note: the bare number has no writer phrase to anchor.
        result.extend((key, v, 'ambiguous') for v in values)
    for item in facts['excluded_evidence']:
        role = ROLES[item['reason']]
        result.extend((key, value, role) for key, value in item['facts'].items())
    return result


def equal(attribute, surface, value):
    try:
        found = normalize(attribute, surface)
    except UnresolvedValue:
        return False
    expected = normalize(attribute, str(value)) if isinstance(value, str) else Decimal(str(value))
    return found == expected


def core(attribute, text, start, end, value):
    """Smallest sub-span of a located writer phrase that normalizes to the value."""
    phrase = text[start:end]
    best = None
    separators = r'[^\s;,=]+' if attribute == 'gear_ratio' else r'[^\s;,=:]+'
    tokens = [(m.start(), m.end()) for m in re.finditer(separators, phrase)]
    for i in range(len(tokens)):
        for j in range(i, len(tokens)):
            a, b = tokens[i][0], tokens[j][1]
            surface = phrase[a:b].rstrip('.')
            if surface and equal(attribute, surface, value) and (best is None or len(surface) < best[1] - best[0]):
                best = (start + a, start + a + len(surface))
    return best


def view(case, build):
    if case.get('author') != 'ai-cedar' or case['semantic_labels']['group'] not in (
            'MG_MOTORS', 'MG_CONVEYOR', 'MG_GEARBOXES', 'MG_BELTS', 'MG_BEARINGS', 'MG_SENSORS'):
        return None
    facts = intended(case)
    if facts is None or (case['intended_facts']['blueprint'] == 'hp_or_missing'
                         and case['semantic_labels']['group'] == 'MG_MOTORS'):
        return None
    text = case['request']['description']
    spans, used = [], []
    if case['intended_facts']['blueprint'] == 'model_only':
        # The writer's only identifier form is "model XX-nnn"; it is a type code, never a rating.
        codes = list(re.finditer(r'(?<=\bmodel )[A-Z]{2}-\d{3}\b', text))
        if len(codes) != 1:
            return None
        spans.append({'attribute': None, 'start': codes[0].start(), 'end': codes[0].end(), 'role': 'type',
                      'surface': codes[0][0], 'value': codes[0][0]})
        used.append((codes[0].start(), codes[0].end()))
    for attribute, value, role in facts:
        phrases = {build.fact(attribute, value, style) for style in range(5)}
        found = [(m.start(), m.end()) for p in phrases for m in re.finditer(re.escape(p), text)]
        found = [f for f in found if not any(max(f[0], u[0]) < min(f[1], u[1]) for u in used)]
        # Keep the longest match at each position; one location per intended fact.
        found = sorted(set(found), key=lambda f: (f[0], -(f[1] - f[0])))
        locations = []
        for f in found:
            if not locations or f[0] >= locations[-1][1]:
                locations.append(f)
        if len(locations) != 1:
            return None
        start, end = locations[0]
        interval = core(attribute, text, start, end, value)
        if interval is None:
            return None
        used.append((start, end))
        spans.append({'attribute': attribute, 'start': interval[0], 'end': interval[1], 'role': role,
                      'surface': text[interval[0]:interval[1]], 'value': value})
    # A request whose facts all come from a reference has no value spans: every
    # description token is gold "outside", which is itself useful supervision.
    if not spans and facts:
        return None
    result = copy.deepcopy(case)
    result.update(case_id=case['case_id'] + '-gold', primary=False, variant=1, author=AUTHOR,
                  writing_template_id=case['writing_template_id'] + ':gold-span-view',
                  tags=[*case['tags'], 'gold_span_view'])
    result['request']['item_id'] = result['case_id']
    result['evidence_annotations'] = {'version': SPAN_VERSION, 'description_sha256': hashlib.sha256(text.encode()).hexdigest(),
                                      'spans': sorted(spans, key=lambda s: s['start'])}
    return result


def views(cases):
    build = writer()
    result = [v for v in (view(c, build) for c in cases) if v is not None]
    from collections import Counter
    return result, {'supplied_cases': sum(c.get('author') == 'ai-cedar' for c in cases), 'annotated_views': len(result),
                    'by_blueprint': dict(Counter(v['intended_facts']['blueprint'] for v in result))}
