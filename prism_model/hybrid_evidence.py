"""Exact comparison of model-attributed spans, never raw-text fact extraction.

This is a comparison component, not a complete inference pipeline. A neural
adapter must supply ownership, confidence and reference-use decisions, with
calibration/provenance verified by the enclosing runtime.
"""
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import math
import re

from .evidence_data import ATTRIBUTES

ROLES = {'current', 'background', 'obsolete', 'negated', 'ambiguous', 'type'}
OWNERS = {'requested_item', 'other_item', 'unresolved'}


@dataclass(frozen=True)
class EvidenceSpan:
    attribute: str | None
    start: int
    end: int
    surface: str
    role: str
    owner: str
    confidence: float
    uncertainties: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReferenceUse:
    mode: str  # not_used, selected, unresolved
    component_id: str | None
    substitution: str  # permitted, not_permitted, unresolved
    confidence: float
    uncertainties: tuple[str, ...] = ()


@dataclass(frozen=True)
class Extraction:
    description_sha256: str
    model_artifact_identity: str
    spans: tuple[EvidenceSpan, ...]
    reference_use: ReferenceUse
    uncertainties: tuple[str, ...] = ()


class UnresolvedValue(ValueError):
    def __init__(self, reason, choice='unclear'):
        super().__init__(reason)
        self.reason, self.choice = reason, choice


SMALL = dict(zip(('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty').split(), range(21)))
TENS = dict(zip(('twenty thirty forty fifty sixty seventy eighty ninety').split(), range(20, 100, 10)))
FRACTIONS = {('a', 'half'): Decimal('.5'), ('one', 'half'): Decimal('.5'), ('a', 'quarter'): Decimal('.25'),
             ('one', 'quarter'): Decimal('.25'), ('three', 'quarters'): Decimal('.75')}


def number(text, *, allow_zero=False):
    text = text.strip().casefold()
    # Contract pilot-0.4 clarification, 11 November 2026: a comma followed by
    # exactly three digits is a thousands separator. Any other comma, including
    # a decimal comma such as "1,5", remains ambiguous.
    if re.fullmatch(r'\d{1,3}(?:,\d{3})+(?:\.\d+)?', text):
        text = text.replace(',', '')
    if ',' in text:
        raise UnresolvedValue('ambiguous_number_format')
    if re.fullmatch(r'\d+(?:\.\d+)?(?:e[+-]?\d+)?', text):
        result = Decimal(text)
    else:
        tokens = text.replace('-', ' ').split()
        if not tokens:
            raise UnresolvedValue('unrecognized_number')
        if tuple(tokens[-2:]) in FRACTIONS and (len(tokens) == 2 or tokens[-3] == 'and'):
            # "three quarters", "one and a half"; the whole part is optional.
            whole = number(' '.join(tokens[:-3]), allow_zero=True) if len(tokens) > 2 else Decimal(0)
            result = whole + FRACTIONS[tuple(tokens[-2:])]
        elif 'thousand' in tokens:
            index = tokens.index('thousand')
            if tokens.count('thousand') != 1 or index != 1 or tokens[0] not in SMALL or not 1 <= SMALL[tokens[0]] <= 9:
                raise UnresolvedValue('unrecognized_number')
            tail = tokens[2:]
            if tail and tail[0] == 'and':
                tail = tail[1:]
            result = Decimal(1000 * SMALL[tokens[0]]) + (number(' '.join(tail), allow_zero=True) if tail else Decimal(0))
        elif 'point' in tokens:
            if tokens.count('point') != 1:
                raise UnresolvedValue('unrecognized_number')
            index = tokens.index('point')
            if not tokens[index + 1:] or any(t not in SMALL or SMALL[t] > 9 for t in tokens[index + 1:]):
                raise UnresolvedValue('unrecognized_number')
            whole = number(' '.join(tokens[:index]), allow_zero=True)
            fraction = ''.join(str(SMALL[t]) for t in tokens[index + 1:])
            result = whole + Decimal('0.' + fraction)
        elif len(tokens) == 1 and tokens[0] in SMALL:
            result = Decimal(SMALL[tokens[0]])
        elif len(tokens) == 1 and tokens[0] in TENS:
            result = Decimal(TENS[tokens[0]])
        elif len(tokens) == 2 and tokens[0] in TENS and tokens[1] in SMALL and 1 <= SMALL[tokens[1]] <= 9:
            result = Decimal(TENS[tokens[0]] + SMALL[tokens[1]])
        elif tokens.count('hundred') == 1 and tokens.index('hundred') == 1 and tokens[0] in SMALL and 1 <= SMALL[tokens[0]] <= 9:
            tail = tokens[2:]
            if tail and tail[0] == 'and':
                tail = tail[1:]
            result = Decimal(100 * SMALL[tokens[0]]) + (number(' '.join(tail), allow_zero=True) if tail else Decimal(0))
        else:
            raise UnresolvedValue('unrecognized_number')
    if not result.is_finite() or result < 0 or (not allow_zero and result == 0):
        raise UnresolvedValue('nonpositive_or_nonfinite_rating')
    return result


# Closed lists of words that may surround a value inside ONE model-selected
# span. They carry no value; removing them never supplies a fact. Anything else
# left beside the value keeps the span unresolved.
_DIMENSION_WORDS = ('overall', 'nominal', 'roller', 'shaft', 'bearing', 'belt', 'loop', 'finished', 'rated',
                    'end-to-end', 'end to end', 'outer', 'outside', 'inner', 'inside', 'sensing', 'detection',
                    'diameter', 'dia', 'width', 'wide', 'length', 'long', 'bore', 'distance', 'range',
                    'od', 'id', 'is', 'of')
QUALIFIERS = {
    'power_kw': ('rated', 'nominal', 'motor', 'power', 'output', 'of a', 'of', 'is'),
    'voltage_v': ('rated', 'nominal', 'supply', 'operating', 'voltage', 'dc', 'ac', 'is'),
    'supply_voltage_v': ('rated', 'nominal', 'supply', 'operating', 'power', 'voltage', 'dc', 'ac', 'is'),
    'gear_ratio': ('gear', 'reduction', 'ratio', 'transmission', 'is'),
    'phases': ('supply', 'connection', 'is'),
    'belt_profile': ('cross section', 'belt', 'profile', 'section', 'tooth', 'timing', 'type', 'is'),
    'output_type': ('switching', 'transistor', 'output', 'out', 'type', 'is'),
}
UNIT = {'power_kw': r'kW|kilowatts?', 'voltage_v': r'V(?:DC|AC)?|volts?', 'supply_voltage_v': r'V(?:DC|AC)?|volts?'}
MM = r'mm|millimet(?:er|re)s?'


def _strip_qualifiers(attribute, surface):
    words = QUALIFIERS.get(attribute, _DIMENSION_WORDS)
    text = surface.strip().strip('()[]"\'').rstrip('.,;)').strip()
    # Label markers such as "L=", "W=", "i=" and a leading diameter sign.
    text = re.sub(r'^(?:[LWDi]|OD|ID)\s*=\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^[Øø⌀]\s*', '', text)
    # "=" and a label colon only separate a label from its value; a ratio keeps
    # its colon.
    text = text.replace('=', ' ')
    if attribute != 'gear_ratio':
        text = text.replace(':', ' ')
    phrases = sorted(words, key=len, reverse=True)
    pattern = r'(?<![\w-])(?:' + '|'.join(re.escape(w).replace(r'\ ', r'\s+') for w in phrases) + r')(?![\w-])'
    text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', text).strip()


def normalize(attribute, surface):
    """Consume exactly one model-selected value/unit span; never search prose.

    The span may contain closed-list qualifier words ("400 mm wide", "24 V DC",
    "15:1 reduction"). A value without its required unit stays unresolved.
    """
    surface = surface.strip()
    if attribute not in ATTRIBUTES or not surface:
        raise UnresolvedValue('unknown_attribute_or_empty_value')
    text = _strip_qualifiers(attribute, surface)
    if not text:
        raise UnresolvedValue('missing_or_unrecognized_unit')
    if attribute == 'belt_profile':
        if text.casefold() == 'flat':
            return 'FLAT'
        if not re.fullmatch(r'[A-Za-z]{1,4}(?:[ -]?\d+(?:\.\d+)?[A-Za-z]{0,2})?', text):
            raise UnresolvedValue('unrecognized_profile')
        return re.sub(r'[ -]', '', text).upper()
    if attribute == 'output_type':
        if text.upper() not in {'PNP', 'NPN'}:
            raise UnresolvedValue('unresolved_output_type')
        return text.upper()
    if attribute == 'phases':
        if re.fullmatch(r'(?:single|mono)\s*-?\s*phase', text, re.IGNORECASE):
            return Decimal(1)
        match = re.fullmatch(r'(.+?)(?:\s*-?\s*phases?|\s*ph|\s*~)', text, re.IGNORECASE)
    elif attribute == 'gear_ratio':
        text = re.sub(r'\s+', ' ', text.replace('=', ' ')).strip()
        match = (re.fullmatch(r'(.+?)\s*:\s*1(?:\.0+)?', text) or re.fullmatch(r'(.+?)\s+to\s+(?:1|one)', text, re.IGNORECASE))
    else:
        unit = UNIT.get(attribute, MM)
        match = re.fullmatch(r'(.+?)\s*(?:' + unit + r')', text, re.IGNORECASE)
    if not match:
        if attribute == 'power_kw' and re.fullmatch(r'.+?\s*(?:hp|horsepower|ps)', text, re.IGNORECASE):
            raise UnresolvedValue('required_kw_not_provided_no_hp_conversion', 'missing')
        raise UnresolvedValue('missing_or_unrecognized_unit')
    value = number(match[1])
    if attribute == 'phases' and value != value.to_integral_value():
        raise UnresolvedValue('noninteger_phase_count')
    return value


QUANTITY_PATTERNS = {
    'mm': r'(?<![\w.,])\d[\d.,]*\s*(?:mm|millimet(?:er|re)s?)(?![A-Za-z])',
    'V': r'(?<![\w.,])\d[\d.,]*\s*(?:V(?:DC|AC)?|volts?)(?![A-Za-z])',
    'kW': r'(?<![\w.,])\d[\d.,]*\s*(?:kW|kilowatts?)(?![A-Za-z])',
    'ratio': r'(?<![\w.,:])\d+(?:\.\d+)?\s*:\s*1(?![\d.])',
    # Coded values (12 November 2026 amendment). Case-sensitive so prose such as
    # "at 50" is not a profile; single-letter profiles (A, B, L) are not searched.
    'profile': r'(?-i:(?<![\w.-])(?:HTD ?\d+ ?M|GT\d|AT\d+|T\d+(?:\.\d+)?|S\d+M|MXL|XL|XH|SP[ZABC]|XP[ZABC])(?![\w-]))',
    'output': r'(?<![\w-])(?:PNP|NPN)(?![\w-])',
}


def unit_class(attribute):
    return ('mm' if attribute.endswith('_mm') else 'V' if attribute in ('voltage_v', 'supply_voltage_v')
            else 'kW' if attribute == 'power_kw' else 'ratio' if attribute == 'gear_ratio'
            else 'profile' if attribute == 'belt_profile' else 'output' if attribute == 'output_type' else None)


def unaccounted_quantities(description, required, spans):
    """Abstention signal only (protocol amendment, 11 November 2026).

    Return digit quantities in a unit the policy needs, and belt-profile or
    output codes when the policy needs those, that no model span touches,
    whatever its role. The result can only block Ready; no value from
    this search is ever compared or used as evidence.
    """
    classes = sorted({unit_class(a) for a in required} - {None})
    covered = [(s.start, s.end) for s in spans]
    return [{'start': m.start(), 'end': m.end(), 'surface': m[0], 'unit_class': c}
            for c in classes for m in re.finditer(QUANTITY_PATTERNS[c], description, re.IGNORECASE)
            if not any(max(m.start(), a) < min(m.end(), b) for a, b in covered)]


def normalize_source(attribute, value):
    if attribute in ('belt_profile', 'output_type'):
        if not isinstance(value, str):
            raise UnresolvedValue('invalid_trusted_field_type')
        return normalize(attribute, value)
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise UnresolvedValue('invalid_trusted_field_type')
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        raise UnresolvedValue('invalid_trusted_field_type') from None
    if not result.is_finite() or result <= 0 or (attribute == 'phases' and result != result.to_integral_value()):
        raise UnresolvedValue('invalid_trusted_rating')
    return result


def _confidence(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def compare_attribute(description, attribute, extraction, components, *, required_kind, minimum_confidence):
    """Return typed evidence outcome and a trace; unresolved facts never satisfy."""
    if attribute not in ATTRIBUTES or not _confidence(minimum_confidence):
        raise ValueError('Invalid policy attribute or evidence gate')
    if (not isinstance(extraction, Extraction) or not extraction.model_artifact_identity
            or extraction.description_sha256 != hashlib.sha256(description.encode()).hexdigest()):
        raise ValueError('Extraction is not bound to this request/model')
    use = extraction.reference_use
    if (use.mode not in {'not_used', 'selected', 'unresolved'} or use.substitution not in {'permitted', 'not_permitted', 'unresolved'}
            or not _confidence(use.confidence) or (use.mode == 'selected') != (use.component_id is not None)):
        raise ValueError('Invalid neural reference-use output')
    if use.mode == 'not_used' and use.substitution != 'not_permitted':
        raise ValueError('Unused reference cannot permit fact substitution')
    ids = [r.get('component_id') for r in components]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Invalid trusted component identities')
    if use.mode == 'selected' and use.component_id not in ids:
        raise ValueError('Model selected an unoffered reference component')
    all_spans = list(extraction.spans)
    for s in all_spans:
        if ((s.attribute not in ATTRIBUTES and not (s.attribute is None and s.role == 'type'))
                or s.role not in ROLES or s.owner not in OWNERS
                or type(s.start) is not int or type(s.end) is not int
                or not 0 <= s.start < s.end <= len(description) or description[s.start:s.end] != s.surface
                or not _confidence(s.confidence)):
            raise ValueError('Invalid model source span')
    spans = [s for s in all_spans if s.attribute == attribute or s.attribute is None]
    unresolved = list(extraction.uncertainties)
    normalized = []
    missing_values = []
    for s in spans:
        if s.uncertainties or s.confidence < minimum_confidence or s.owner == 'unresolved':
            unresolved.append('uncertain_evidence_at_' + str(s.start))
            continue
        if any(s is not other and max(s.start, other.start) < min(s.end, other.end) for other in all_spans):
            unresolved.append('overlapping_evidence_at_' + str(s.start))
            continue
        if s.role == 'ambiguous' and s.owner == 'requested_item':
            unresolved.append('unresolved_alternative_at_' + str(s.start))
        elif s.role == 'current':
            if s.owner != 'requested_item':
                unresolved.append('inconsistent_item_ownership_at_' + str(s.start))
                continue
            try:
                normalized.append((s, normalize(attribute, s.surface)))
            except UnresolvedValue as error:
                (missing_values if error.choice == 'missing' else unresolved).append(error.reason)
        elif s.role == 'background' and s.owner != 'other_item':
            unresolved.append('inconsistent_background_ownership_at_' + str(s.start))
        # Confident non-current and type spans are retained in the trace, ignored
        # as positive evidence; no value is guessed from an identifier.
    trusted = None
    def result(choice, reasons):
        return {'choice': choice, 'reasons': sorted(set(reasons)),
                'model_artifact_identity': extraction.model_artifact_identity,
                'description_sha256': extraction.description_sha256,
                'evidence_spans': [asdict(s) for s in spans],
                'normalized_current_values': [{'start': s.start, 'end': s.end, 'value': str(v)} for s, v in normalized],
                'reference_use': asdict(use), 'trusted_reference_field': trusted}
    current = {v for _, v in normalized}
    if len(current) > 1:
        return result('conflicting', ['multiple_affirmative_current_ratings'])
    if use.uncertainties or use.confidence < minimum_confidence or use.mode == 'unresolved':
        unresolved.append('unresolved_reference_use')
    elif use.mode == 'selected':
        reference = components[ids.index(use.component_id)]
        if reference.get('kind') != required_kind:
            return result('conflicting', ['reference_kind_mismatch'])
        if attribute in reference:
            try:
                value = normalize_source(attribute, reference[attribute])
                trusted = {'component_id': use.component_id, 'field': attribute, 'value': str(value)}
                if current and next(iter(current)) != value:
                    return result('conflicting', ['current_rating_differs_from_active_reference'])
            except UnresolvedValue as error:
                unresolved.append(error.reason)
        if use.substitution == 'unresolved':
            unresolved.append('reference_substitution_unresolved')
    if unresolved:
        return result('unclear', unresolved)
    if missing_values:
        return result('missing', missing_values)
    if current:
        return result('satisfied', ['explicit_current_evidence'])
    if use.mode == 'selected' and use.substitution == 'unresolved':
        return result('unclear', ['reference_substitution_unresolved'])
    if use.mode == 'selected' and use.substitution == 'permitted' and trusted is not None:
        return result('satisfied', ['permitted_trusted_reference_substitution'])
    return result('missing', ['no_supported_current_fact_or_permitted_reference_fact'])
