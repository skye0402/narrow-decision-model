"""Natural-wording evidence corpus with writer-recorded gold spans.

Written for the 11 November 2026 review. The existing generated corpora use a
narrow vocabulary, and the natural 400 have no gold spans. This writer composes
requests from intended facts and records every source span while writing, so
labels never come from a parser, a model or the rules comparator.

Writing styles are partitioned: every phrase bank is split into disjoint
groups for train, development, calibration/threshold selection and the
reserved evaluation role. A family's role decides which phrasings it may use,
so development and calibration measure transfer to unseen wording, not only
to unseen values. This is still one assistant's synthetic writing, not
independent human authorship.
"""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random

from .evidence_data import SPLITS, VERSION as DATA_VERSION
from .evidence_span_data import VERSION as SPAN_VERSION
from .provenance import file_sha256
from .task_data import expected_route, stable_json

VERSION = 'natural-evidence-1'
AUTHOR = 'assistant_natural_wording'
SEED = 20261111
FAMILIES = dict(zip(SPLITS, (4000, 600, 300, 500, 1350)))
STYLE_GROUP = {'train': 'train', 'development': 'development', 'calibration': 'fit',
               'threshold_selection': 'fit', 'prototype_evaluation': 'evaluation'}
CATEGORIES = {
    'MG_MOTORS': ('motor', ('power_kw', 'voltage_v', 'phases')),
    'MG_CONVEYOR': ('roller', ('diameter_mm', 'length_mm')),
    'MG_GEARBOXES': ('gearbox', ('gear_ratio', 'shaft_diameter_mm')),
    'MG_BELTS': ('belt', ('width_mm', 'length_mm', 'belt_profile')),
    'MG_BEARINGS': ('bearing', ('bore_mm', 'outer_diameter_mm', 'width_mm')),
    'MG_SENSORS': ('sensor', ('sensing_distance_mm', 'supply_voltage_v', 'output_type')),
}


def split_bank(items):
    """Deterministic disjoint style groups; each group gets at least one item."""
    if len(items) < 5:
        raise ValueError('A phrase bank needs at least five entries to hold out styles')
    groups = {'train': [], 'development': [], 'fit': [], 'evaluation': []}
    pattern = ('train', 'train', 'development', 'train', 'evaluation', 'train', 'fit', 'train', 'evaluation', 'train')
    for i, item in enumerate(items):
        groups[pattern[i % len(pattern)] if i >= 5 else ('train', 'development', 'fit', 'evaluation', 'train')[i]].append(item)
    return groups


# Request nouns per category. "{n}" is replaced by a noun.
NOUNS = {
    'MG_MOTORS': ['electric motor', 'motor', 'replacement motor', 'drive motor', 'AC motor', 'spare motor',
                  'induction motor', 'gear-less drive motor', 'three-phase motor unit', 'fan motor', 'pump motor'],
    'MG_CONVEYOR': ['conveyor roller', 'idler', 'idler roller', 'carrying roller', 'return roller', 'gravity roller',
                    'conveyor idler', 'roller for the conveyor', 'replacement roller', 'spare roller', 'belt conveyor roller'],
    'MG_GEARBOXES': ['gearbox', 'speed reducer', 'gear reducer', 'reduction gearbox', 'worm gearbox', 'reducer',
                     'helical gearbox', 'right-angle gearbox', 'replacement gearbox', 'spare reducer', 'inline gearbox'],
    'MG_BELTS': ['drive belt', 'timing belt', 'conveyor belt', 'replacement belt', 'belt', 'transmission belt',
                 'toothed belt', 'spare belt', 'machine belt', 'endless belt', 'power transmission belt'],
    'MG_BEARINGS': ['ball bearing', 'bearing', 'deep groove ball bearing', 'roller bearing', 'replacement bearing',
                    'spare bearing', 'radial bearing', 'shielded ball bearing', 'sealed bearing', 'bearing for the shaft',
                    'support bearing'],
    'MG_SENSORS': ['inductive proximity sensor', 'inductive sensor', 'proximity switch', 'inductive proximity switch',
                   'prox sensor (inductive)', 'inductive prox', 'metal-detecting proximity sensor (inductive)',
                   'inductive limit sensor', 'inductive switch', 'inductive position sensor', 'replacement inductive sensor'],
}
CONTEXTS = split_bank([
    'the bottle capper', 'packing line 3', 'the palletiser', 'the dough mixer', 'the shrink tunnel', 'the label applicator',
    'the cardboard folder', 'the glass inspection table', 'the paint booth extractor', 'the potato peeler',
    'the parcel sorter', 'the laser cutter bed', 'the bread slicer', 'the cooling tower fan', 'the boiler feed pump',
    'the cheese press', 'the textile winder', 'the sawmill infeed', 'the tablet press', 'the cable drum stand',
    'the egg grader', 'the tyre buffing station', 'the compost turner', 'the CNC chip conveyor', 'the bag filler',
    'the conveyor at goods-in', 'the paper slitter', 'the meat tumbler', 'the injection moulding robot', 'the screen printer',
    'the vacuum former', 'the bulk silo discharge', 'the grain auger', 'the tile press', 'the depalletiser',
    'the cold store door', 'the fruit sorter', 'the plating line hoist', 'the woodchip feeder', 'the dishwasher conveyor',
    'the lens polisher', 'the rubber extruder', 'the stretch wrapper', 'the can seamer', 'the tobacco dryer',
    'the press brake', 'the coil slitter', 'the flour sifter', 'the yoghurt filler', 'the bindery trimmer'])
INTROS = split_bank([
    'Please order {a} {n} for {c}.', '{C} needs {a} {n}.', 'We need {a} {n} for {c}.',
    'Requesting {a} {n} - it is for {c}.', 'Hi, can you buy {a} {n}? It goes on {c}.',
    'For {c}: {n} wanted.', 'Spares request for {c}. Item: {n}.', 'The {n} on {c} has failed and needs replacing.',
    '{A} {n} is needed on {c}.', 'Could purchasing source {a} {n} for {c}?',
    'Breakdown on {c}; please get {a} {n}.', '{C} - replacement {n}.', 'Order {a} {n}, please, for {c}.',
    'Maintenance asks for {a} {n} (for {c}).', 'Job card for {c}: supply {a} {n}.',
    'Our {n} on {c} is worn out, we need a new one.', 'Purchase request: {n}, location {c}.',
    'Could we get {a} {n} in for {c}?', 'Line stop on {c}. Need {a} {n}.', 'New {n} required at {c}.'])
JOINERS = split_bank([('; ', ''), (', ', ''), (' / ', ''), ('. ', '.'), (', ', '.'), (' - ', ''), ('; ', '.'), (' | ', ''),
                      (', and ', ''), ('. ', ''), (' + ', '')])
SPEC_LEADS = split_bank(['', 'Specs: ', 'Details: ', 'It needs to be ', 'Data from the plate: ', 'Spec - ',
                         'Measured: ', 'Requirements: ', 'From the label: ', 'What I have: ', 'Rating: ',
                         'Size: ', 'Info: '])

# Value expressions. "{v}" is the recorded span. Each entry is a phrase bank.
FORMS = {
    'power_kw': ['{v}', 'rated {v}', 'power {v}', '{v} rated output', 'output {v}', 'rated power {v}',
                 '{v} motor rating', 'rating {v}', 'power rating of {v}', 'a {v} rating', '{v} output power', 'nominal {v}'],
    'voltage_v': ['{v}', 'voltage {v}', '{v} supply', 'supply {v}', 'rated voltage {v}', '{v} mains',
                  'operating voltage {v}', 'running on {v}', '{v} rated', 'voltage of {v}', 'for {v}', 'wired for {v}'],
    'phases': ['{v}', '{v} supply', '{v} connection', 'wired {v}', '{v} motor', 'connection {v}',
               'supply {v}', '{v} type', 'runs on {v}', 'a {v} unit', '{v} only'],
    'diameter_mm': ['diameter {v}', '{v} diameter', 'dia {v}', 'Ø{v}', 'OD {v}', 'tube diameter {v}',
                    '{v} dia', 'roller diameter {v}', 'diameter of {v}', '{v} across', 'outside diameter {v}', 'Ø {v}'],
    'length_mm': ['length {v}', '{v} long', 'L={v}', '{v} length', 'overall length {v}', 'length of {v}',
                  'L {v}', '{v} overall', 'face length {v}', '{v} in length', 'total length {v}', 'length: {v}'],
    'gear_ratio': ['ratio {v}', '{v}', '{v} reduction', 'reduction {v}', 'i={v}', 'a {v} ratio',
                   'gear ratio {v}', '{v} ratio', 'ratio of {v}', 'reduction ratio {v}', 'reducing {v}', 'i = {v}'],
    'shaft_diameter_mm': ['shaft {v}', '{v} shaft', 'shaft diameter {v}', 'output shaft {v}', 'shaft dia {v}',
                          '{v} output shaft', 'shaft Ø{v}', 'a {v} shaft', 'shaft size {v}', '{v} solid shaft',
                          'output shaft diameter {v}', 'shaft of {v}'],
    'width_mm': ['width {v}', '{v} wide', 'W={v}', '{v} width', 'width of {v}', 'W {v}',
                 '{v} in width', 'overall width {v}', 'width: {v}', 'measuring {v} wide', 'face width {v}', '{v} across'],
    'belt_profile': ['profile {v}', '{v} profile', '{v} tooth profile', 'section {v}', '{v} section',
                     'type {v}', 'an {v} belt', '{v} pitch', 'tooth form {v}', '{v} type', 'profile: {v}', 'in {v}'],
    'bore_mm': ['bore {v}', '{v} bore', 'ID {v}', 'inner diameter {v}', 'bore of {v}', '{v} ID',
                'inside diameter {v}', 'bore diameter {v}', 'shaft bore {v}', 'bore: {v}', 'inner {v}', 'ID={v}'],
    'outer_diameter_mm': ['OD {v}', '{v} OD', 'outside diameter {v}', 'outer diameter {v}', 'outer dia {v}',
                          '{v} outside', 'OD={v}', 'outside dia {v}', '{v} outer diameter', 'housing OD {v}',
                          'external diameter {v}', 'OD: {v}'],
    'sensing_distance_mm': ['sensing distance {v}', '{v} sensing distance', 'range {v}', '{v} range',
                            'sensing range {v}', 'switching distance {v}', 'detects at {v}', '{v} sensing',
                            'rated distance {v}', 'distance {v}', 'nominal range {v}', 'reach {v}'],
    'supply_voltage_v': ['{v}', 'supply {v}', '{v} supply', 'powered at {v}', 'operating voltage {v}',
                         '{v} power', 'supply voltage {v}', 'runs on {v}', 'voltage {v}', '{v} operating',
                         'fed with {v}', 'supply: {v}'],
    'output_type': ['{v}', '{v} output', 'output {v}', '{v} switching', '{v} type', 'switching {v}',
                    'output type {v}', '{v} (3-wire)', 'wired {v}', '{v} transistor output', '{v} out', 'with {v} output'],
}
FORMS = {k: split_bank(v) for k, v in FORMS.items()}
POOLS = {
    'power_kw': [0.09, 0.12, 0.18, 0.25, 0.37, 0.55, 0.75, 1.1, 1.5, 2.2, 3, 4, 5.5, 7.5, 11],
    'voltage_v': [110, 230, 400, 415, 460, 690], 'phases': [1, 3],
    'diameter_mm': [20, 25, 32, 38, 40, 48, 50, 60, 63, 76, 89, 102, 108, 133],
    'length_mm': [150, 200, 240, 300, 380, 450, 500, 600, 750, 900, 1000, 1150, 1200, 1400, 1600],
    'gear_ratio': [3, 5, 7.5, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100],
    'shaft_diameter_mm': [12, 14, 16, 19, 20, 24, 25, 28, 30, 35, 40, 45, 50],
    'width_mm': [6, 9, 10, 12, 15, 16, 20, 25, 30, 40, 50, 100, 150, 400, 600],
    'belt_profile': ['T2.5', 'T5', 'T10', 'AT5', 'AT10', 'HTD 3M', 'HTD 5M', 'HTD 8M', 'GT2', 'GT3', 'XL', 'L',
                     'SPA', 'SPB', 'SPZ', 'XPZ', 'A', 'B', 'flat'],
    'bore_mm': [6, 8, 10, 12, 15, 17, 20, 25, 30, 35, 40, 45, 50],
    'outer_diameter_mm': [19, 22, 26, 30, 32, 35, 40, 42, 47, 52, 62, 72, 80, 85, 90],
    'sensing_distance_mm': [0.8, 1, 1.5, 2, 2.5, 4, 5, 8, 10, 12, 15, 20],
    'supply_voltage_v': [12, 24, 48], 'output_type': ['PNP', 'NPN'],
}
BEARINGS = [(6, 19, 6), (8, 22, 7), (10, 26, 8), (10, 30, 9), (12, 32, 10), (15, 35, 11), (17, 40, 12),
            (20, 42, 12), (20, 47, 14), (25, 52, 15), (30, 62, 16), (35, 72, 17), (40, 80, 18), (45, 85, 19), (50, 90, 20)]
MOTOR_SUPPLY = [(110, 1), (230, 1), (230, 3), (400, 3), (415, 3), (460, 3), (690, 3)]
TERMS = {'power_kw': 'power rating', 'voltage_v': 'voltage', 'phases': 'phase count', 'diameter_mm': 'diameter',
         'length_mm': 'length', 'gear_ratio': 'ratio', 'shaft_diameter_mm': 'shaft diameter', 'width_mm': 'width',
         'belt_profile': 'profile', 'bore_mm': 'bore', 'outer_diameter_mm': 'outside diameter',
         'sensing_distance_mm': 'sensing distance', 'supply_voltage_v': 'supply voltage', 'output_type': 'output type'}

SMALL = ('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen '
         'seventeen eighteen nineteen').split()
TENS = dict(zip(range(20, 100, 10), 'twenty thirty forty fifty sixty seventy eighty ninety'.split()))


def spoken(value):
    """Words for a value the normalizer accepts; None when no safe wording exists."""
    if value == 0.75:
        return 'three quarters'
    if value == 1.5:
        return 'one and a half'
    if value == 2.5:
        return 'two and a half'
    whole, _, fraction = f'{value:g}'.partition('.')
    n = int(whole)
    def integer(n):
        if n < 20:
            return SMALL[n]
        if n < 100:
            return TENS[n // 10 * 10] + ('-' + SMALL[n % 10] if n % 10 else '')
        if n < 1000:
            return SMALL[n // 100] + ' hundred' + (' and ' + integer(n % 100) if n % 100 else '')
        if n < 10000 and n % 1000 == 0:
            return SMALL[n // 1000] + ' thousand'
        if n < 10000:
            return SMALL[n // 1000] + ' thousand ' + integer(n % 1000)
        return None
    words = integer(n)
    if words is None:
        return None
    return words + (' point ' + ' '.join(SMALL[int(d)] for d in fraction) if fraction else '')


def core(attribute, value, rng, *, allow_words=True):
    """The exact span text for one value and the optional restatement in digits."""
    n = f'{value:g}' if isinstance(value, (int, float)) else value
    if isinstance(value, (int, float)) and value >= 1000 and rng.random() < .5:
        n = f'{int(value):,}'
    if attribute == 'phases':
        return rng.choice(['three-phase', '3-phase', '3 ph', '3~', '3ph', 'three phase', '3 phase'] if value == 3
                          else ['single-phase', 'single phase', '1-phase', '1 ph', '1~', '1 phase']), None
    if attribute == 'gear_ratio':
        words = spoken(value) if allow_words and value == int(value) else None
        if words and rng.random() < .2:
            return rng.choice([f'{words} to one', f'{words}:1']), (f'{n}:1' if rng.random() < .5 else None)
        return rng.choice([f'{n}:1', f'{n}:1', f'{n} to 1', f'{n} : 1']), None
    if attribute in ('belt_profile', 'output_type'):
        return str(value), None
    unit = {'power_kw': ('kW', 'kW', 'kilowatts', 'kilowatt', 'KW'),
            'voltage_v': ('V', 'V', 'volts', 'V AC', 'VAC'),
            'supply_voltage_v': ('V', 'V DC', 'VDC', 'volts DC', 'V'),
            }.get(attribute, ('mm', 'mm', 'mm', 'millimetres', 'millimeters'))
    chosen = rng.choice(unit)
    words = spoken(value) if allow_words else None
    if words and rng.random() < .18:
        long_unit = {'kW': 'kilowatts', 'V': 'volts', 'mm': 'millimetres'}.get(unit[0])
        if value == 0.75 and attribute == 'power_kw':
            text = 'three quarters of a kilowatt'
        else:
            text = f'{words} {long_unit}'
        digits = f'{n} {unit[0]}' if rng.random() < .6 else None
        return text, digits
    space = '' if rng.random() < .3 and chosen in ('kW', 'V', 'mm', 'KW', 'VDC', 'VAC') else ' '
    return f'{n}{space}{chosen}', None


class Writer:
    def __init__(self):
        self.text, self.spans = '', []

    def add(self, text):
        self.text += text

    def mark(self, text, attribute, role, value):
        start = len(self.text)
        self.text += text
        self.spans.append({'attribute': attribute, 'start': start, 'end': len(self.text), 'role': role,
                           'surface': text, 'value': value})

    def phrase(self, template, attribute, role, value, rng, *, allow_words=True):
        """Write "{v}" as a recorded span; an optional digit restatement is a second span."""
        before, after = template.split('{v}')
        main, digits = core(attribute, value, rng, allow_words=allow_words)
        self.add(before)
        self.mark(main, attribute, role, value)
        if digits:
            self.add(' (')
            self.mark(digits, attribute, role, value)
            self.add(')')
        self.add(after)


def article(noun):
    return 'an' if noun[0].lower() in 'aeiou' else 'a'


def values(group, rng):
    keys = CATEGORIES[group][1]
    result = {k: rng.choice(POOLS[k]) for k in keys}
    if group == 'MG_MOTORS':
        result['voltage_v'], result['phases'] = rng.choice(MOTOR_SUPPLY)
    if group == 'MG_BEARINGS':
        result['bore_mm'], result['outer_diameter_mm'], result['width_mm'] = rng.choice(BEARINGS)
    if group == 'MG_BELTS':
        result['length_mm'] = rng.choice([280, 400, 450, 600, 750, 850, 1000, 1200, 1250, 1500, 2000, 2400, 3200])
        if result['belt_profile'] == 'flat':
            result['width_mm'] = rng.choice([50, 100, 150, 400, 600])
    return result


def alternative(key, value, rng, specs):
    options = [v for v in POOLS[key] if v != value]
    if key == 'bore_mm':
        options = [v for v in options if v < specs.get('outer_diameter_mm', 999) - 4]
    if key == 'outer_diameter_mm':
        options = [v for v in options if v > specs.get('bore_mm', 0) + 4]
    if key == 'phases':
        options = [1 if value == 3 else 3]
    return rng.choice(options)


# Phrase banks for each scenario. "{x}" marks where facts are written.
CONFLICT = split_bank([
    'The second job sheet says {x} for this same part, and both sheets are still active.',
    'But the electrician wrote {x} for this replacement, and nobody has withdrawn either value.',
    'The current drawing gives {x} instead; both documents are meant for the part we are ordering.',
    'My notes also say {x} for the same part. Both entries are current.',
    'The supplier quote lists {x} for this exact part, and that has not been reconciled.',
    'The maintenance log however records {x} for the same position; neither has been corrected.',
    'Another live work order for this part asks for {x}. Both are valid at the moment.',
    'The commissioning sheet states {x} for this unit too - still unresolved.',
    'On the latest measurement sheet it is {x}, for this same item. Both stand.',
    'A colleague measured {x} on the same part today; we have not agreed which is right.',
    'Engineering also sent {x} for the identical replacement, and both instructions are open.',
    'The second label on the part shows {x}. Neither label has been ruled out.'])
CORRECTED = split_bank([
    'The old note with {x} is crossed out and no longer applies.',
    'Ignore the earlier value of {x}; it was superseded.',
    'The previous spec ({x}) was withdrawn after the upgrade.',
    'We used to fit one with {x}, but that is not what we need now.',
    'The former {x} unit was removed last year and is not the requirement.',
    'An older order listed {x} - that is out of date.',
    'Please disregard {x} on the old sheet; it was a typo that has been corrected.',
    'The obsolete drawing still shows {x}, which has been replaced.',
    'Before the rebuild it had {x}; that no longer applies.',
    'The crossed-out {x} on the job card is wrong.',
    '{x} was the value before the modification and is cancelled.',
    'Earlier we were told {x}, but that has been revised.'])
OBSOLETE_ONLY = split_bank([
    'The old one was {x}, but that was before the redesign and we have no current figure for the {t}.',
    'The scrapped unit had {x}; the new {t} has not been confirmed.',
    'The previous version used {x}, which no longer applies. I still need the current {t}.',
    'Before the upgrade it was {x}; that value is obsolete and the new {t} is unknown.',
    'The old drawing said {x}, but it is out of date, and nobody has the new {t}.',
    'Historic record: {x}. That is superseded; the current {t} still has to be measured.',
    'We removed the {x} version. The {t} for the new one is still open.',
    'The former part was {x}. That no longer applies and the {t} is not yet known.',
    'Ignore the old {x}; we do not have the replacement {t} yet.',
    'It used to be {x} but that changed, and nobody has told me the new {t}.',
    'The withdrawn spec gave {x}. The {t} for this order is still to be checked.'])
NEGATED = split_bank([
    'It is not the {x} version. The actual {t} still needs checking.',
    'Definitely not {x}; I have asked for the real {t} to be confirmed.',
    'Do not order the {x} one. We still need to find out the {t}.',
    'It is NOT {x}. Maintenance will confirm the {t}.',
    'Not the {x} type - the {t} has to be checked on site.',
    'We know it is not {x}, but the correct {t} is still unknown.',
    'Please do not supply {x}. The {t} will follow.',
    'Not {x}, as previously assumed; the {t} has not been measured.',
    'It must not be {x}. Still waiting for the {t}.',
    'Rule out {x}. The right {t} is not known yet.',
    'The {x} variant is wrong for us; {t} to be confirmed.'])
NEGATED_PLUS = split_bank([
    'Not the {x} one.', 'It is not {x}.', 'Do not send the {x} version.', 'Not {x} like the old stock.',
    'Make sure it is not {x}.', 'NOT {x}.', 'The {x} variant is wrong.', 'Avoid {x}.',
    'Please not {x} this time.', 'Not to be confused with the {x} version.', 'It should not be {x}.'])
AMBIGUOUS = split_bank([
    'The note says either {x} or {y}; nobody has decided which.',
    'It could be {x} or {y} - I could not tell from the label.',
    'We are not sure whether it is {x} or {y}.',
    'Either {x} or {y}; I will confirm later.',
    'Maintenance says {x}, or maybe {y}, they are not certain.',
    'Possibly {x}, possibly {y}. Not confirmed.',
    'Somewhere between {x} and {y}, I cannot be sure which one it is.',
    'The plate is worn: it is {x} or {y}.',
    'One of {x} or {y}, still to be chosen.',
    'I think {x}, but it might be {y}.',
    'Either {x} or {y} would be possible; the choice is still open.'])
UNRELATED = split_bank([
    'The neighbouring machine uses {x}, but that one is staying.',
    'For reference, the other line has {x}; not part of this order.',
    'The spare on the shelf next to it is {x}, which is for a different machine.',
    'Do not confuse it with the {x} unit on the adjacent conveyor.',
    'The twin machine has {x}; ignore that, it is not being replaced.',
    'Another part in the same cabinet is {x}, but we are not buying that.',
    'The {x} item listed in the old manual belongs to a different assembly.',
    'Our other site runs {x} - irrelevant for this request.',
    'The adjacent drive is {x} and stays in service.',
    'There is also {x} on the second station, not affected.',
    'Next to it there is one with {x}; that one is fine.'])
MISSING = split_bank([
    'I still need to check the {t}.', 'The {t} was not recorded.', 'No {t} on the label.',
    'The {t} is unknown at the moment.', 'I could not read the {t}.', 'We have not measured the {t} yet.',
    'The {t} is missing from the job sheet.', 'Nobody wrote down the {t}.', 'The {t} still has to be confirmed.',
    'Do not know the {t}.', 'The {t} will follow once the guard is off.', 'I have no {t} yet.', ''])
MODEL_ONLY = split_bank([
    'The only marking is model {m}; no rating plate or datasheet.',
    'All I have is the type code {m}.', 'It says {m} on the casing, nothing else.',
    'Part number {m} - no other details.', 'Model {m}. I do not have the specifications.',
    'The sticker reads {m}; please do not treat that as a size.', 'Type {m} is all we copied down.',
    'Only the code {m} is visible.', 'Reference on the housing: {m}. No data sheet.',
    'Manufacturer code {m}, no spec sheet available.', 'It is a {m}, that is all I know.'])
MODEL_PLUS = split_bank([
    'Model {m}.', 'Type {m}.', 'Code on the unit: {m}.', 'Part no. {m}.', 'It is marked {m}.',
    'Label: {m}.', '({m})', 'Ref {m}.', 'Maker code {m}.', 'Housing says {m}.', 'Type plate: {m}.'])
UNITLESS = split_bank([
    'My note just says {x} for the {t}, no unit written.', 'The {t} is {x} but I forgot to note the unit.',
    '{t} {x} - not sure if that is mm or inches.', 'I wrote {t} {x} without units.',
    'For the {t} I only have the number {x}.', 'The sketch says {x} for the {t}, no unit.',
    '{t}: {x} (unit not recorded).', 'The {t} figure is {x}, units unknown.',
    'Someone wrote {x} next to {t}, no units.', 'The {t} reads {x} on my note; unit unclear.',
    '{t} {x}, I am not sure of the units.'])
HP = split_bank([
    'The plate only says {x}.', 'Rated {x} according to the old card.', 'It is a {x} motor.',
    'Power is {x}; no kW figure.', 'Output {x}.', '{x} is all the power information we have.',
    'Nameplate: {x}.', 'Power: {x} (no kW value).', 'It was sold as {x}.', 'Shown as {x} on the label.',
    'Listed as {x} in the manual.'])
FIELD_ERRORS = {
    'quantity': (0, split_bank(['The quantity got left at zero.', 'Quantity box is still 0, sorry.',
                                'I forgot to fill in the quantity.', 'Quantity field needs fixing.',
                                'Qty was not entered.', 'The quantity is wrong on the form.'])),
    'unit': ('M', split_bank(['The unit field says M by mistake, these are pieces.', 'Unit should be each, not metres.',
                              'I picked metres in the unit box by accident.', 'Wrong unit selected.',
                              'Unit field is wrong, should be EA.', 'Ignore the M in the unit field.'])),
    'plant': ('1300', split_bank(['I put the old plant code in.', 'The plant field still has the old site.',
                                  'Plant code may be wrong, it is the old one.', 'Not sure the plant is right.',
                                  'Plant number copied from an old order.', 'Wrong plant entered.'])),
    'required_date': ('2026-10-30', split_bank(['The date was copied from an old request.', 'Required date needs updating.',
                                                'The date in the form is in the past.', 'Date is wrong, sorry.',
                                                'Old required date still there.', 'Ignore the October date.'])),
}
REF_FULL = split_bank([
    'Replace it like for like with {r}.', 'Exact replacement for {r}, please.', 'Same as {r}.',
    'Please copy {r} exactly.', 'Like-for-like swap of {r}; that is the one fitted.',
    'Identical spare to {r}.', 'Use the {r} card - same part again.', 'Order another {r}, exactly the same.',
    'It is the part recorded as {r}; same again please.', 'Replace with an identical one per {r}.',
    'Duplicate of {r}.', 'Match {r} exactly.'])
REF_PARTIAL = split_bank([
    'Same as {r}, but the card has no {t}; I measured it: {x}.',
    'Like for like with {r}. The {t} is missing on the card, it is {x}.',
    'Copy {r}. Its card omits the {t} - checked today: {x}.',
    'Exact replacement of {r}; I added the {t} myself: {x}.',
    'Use {r}. The card lacks {t}, so here it is: {x}.',
    'Identical to {r}. {t} not on file, measured {x}.',
    'Replace per {r}; missing {t} on the record is {x}.',
    'Same as {r} - card is blank for {t}, actual is {x}.',
    'Like-for-like with {r}; the {t} on the part is {x}.',
    'Copy {r}, and the {t} (not on the card) is {x}.',
    'Duplicate {r}. For the {t}: {x}, measured on the old part.'])
REF_INCOMPLETE = split_bank([
    'Copy {r} exactly. The card has no {t} and I cannot measure it.',
    'Same as {r}; the {t} is not on the record.', 'Like for like with {r}, though its {t} is blank.',
    'Replace per {r}. Note: {t} missing from the card.', 'Use {r}; nobody recorded the {t}.',
    'Identical spare to {r}. {t} unknown.', 'Exact copy of {r} - the {t} field is empty.',
    'Order the same as {r}. I do not have its {t}.', 'Match {r}; card does not show the {t}.',
    'Same again as {r}, {t} not listed.', 'Like-for-like for {r}; {t} still missing.'])
REF_CONFLICT = split_bank([
    'Replace like for like with {r}, with {x}.', 'Same as {r}, {x}.', 'Exact replacement of {r}: {x}.',
    'Copy {r}, it should be {x}.', 'Identical to {r}; spec {x}.', 'Like-for-like {r} - {x}.',
    'Use {r}. Required: {x}.', 'Match {r}, {x} please.', 'Same part as {r}, {x}.',
    'Order another {r} ({x}).', 'Per {r}, with {x}.'])
REF_DISCLAIMED = split_bank([
    '{r} is attached for background only; this is a new arrangement, not a copy of the old part.',
    'I attached {r} just for history. Please do not copy it.',
    '{r} is the removed part - ignore it for this order.',
    'The card {r} is from the old machine and does not apply.',
    'Do not use {r}; it is from a different setup.',
    '{r} is only included for reference, not to be matched.',
    '{r} describes the old design and is not the requirement.',
    'We are not replacing {r} like for like; it is attached for information.',
    '{r} is the scrapped unit. Not what we need.',
    'Please disregard {r}, that was the previous part.',
    '{r} is background from the old line, not this purchase.'])
REF_TWO = split_bank([
    'Like for like, but both {r} and {s} are listed and I cannot tell which is fitted.',
    'Replace exactly; it is either {r} or {s}, the label is gone.',
    'Same as the fitted part: {r} or {s}, not sure which.',
    'Two cards are attached, {r} and {s}. I do not know which one is ours.',
    'Exact copy needed, candidates {r} and {s}; not identified.',
    'Either {r} or {s} is the installed one. Please help identify.',
    'It is one of {r}/{s}; we have not checked.',
    'The file has {r} and {s} for this position and I cannot choose.',
    'Replacement for {r} or {s} - unknown which.',
    'Both {r} and {s} could be the one; nobody knows.',
    'Like-for-like, but {r} and {s} both match the description.'])
REF_TWO_SELECTED = split_bank([
    'Replace like for like with {r}. {s} is for a different line; do not order that one.',
    'Copy {r} exactly. Ignore {s}, it belongs to another machine.',
    'Same as {r}; {s} is the wrong card.', 'Use {r}. {s} was attached by mistake.',
    'It is {r}, not {s}.', 'Exact replacement of {r} (not {s}).', 'Order {r}; disregard {s}.',
    '{r} is the fitted one. {s} is for the other conveyor.', 'Match {r}. {s} does not apply.',
    'Like-for-like {r}. {s} is a different position.', 'Same as {r}. Please ignore {s}.'])
SCOPE = {
    'insufficient_evidence': split_bank([
        'Please order a spare for {c}; I do not know what the part is called.',
        'We need the thing that broke on {c}. No idea what it is.',
        'Replace the part behind the left guard on {c}. I cannot describe it.',
        'Something on {c} is noisy again, please buy the spare.',
        'Need another XR-{k} for {c}. Not sure whether that is a part code or the machine code.',
        'Order the item from bin {k}. The box and ticket are gone.',
        'For {c} we need either a belt or a gearbox - not decided yet.',
        'It is the {p} or the {q} on {c}, maintenance has not diagnosed which.',
        'Buy the part that drives the shaft on {c}. Could be a motor or a reducer.',
        'Same as last time for {c}, whatever we ordered then.',
        'The photo did not attach. Please get the part I circled for {c}.',
        'Either the {p} or the {q} has failed on {c}; we need one of them.',
        'Replacement for the broken bit on {c}. Unknown item.']),
    'mixed_scope': split_bank([
        'For {c} order {a} {p} and {a2} {q}; both are needed.',
        'Please buy {a} {p} plus {a2} {q} for {c}.',
        'We need {a} {p} together with {a2} {q} on {c}.',
        '{C}: one {p} AND one {q}, please.',
        'Order {a} {p} for {c}, and also book a technician to fit it.',
        'Buy {a} {p} and a mounting bracket for {c}.',
        'Need {a} {p} plus the connection cable for {c}; both items to be purchased.',
        '{C} needs {a} {p} and {a2} {q} in the same order.',
        'Please supply {a} {p} and also a can of grease for {c}.',
        'Order {a} {p} and {a2} {q}; we are replacing both on {c}.',
        'Get {a} {p} for {c}, plus installation labour.',
        'Kit for {c}: {a} {p} and {a2} {q}.']),
    'no_suitable_group': split_bank([
        'Order a pneumatic cylinder for {c}, 40 mm bore, 200 mm stroke.',
        'We need a frequency inverter for the motor on {c}; the motor itself stays.',
        'Please buy a photoelectric sensor for {c}; it must be optical, not inductive.',
        'Five litres of gear oil for the gearbox on {c}. Oil only.',
        'Book a vibration analysis visit for {c}. No parts.',
        'A capacitive proximity sensor for {c} (capacitive, not inductive).',
        'Order a chain sprocket for {c}, 20 teeth.',
        'Need a new pulley for the belt drive on {c}; the belt is fine.',
        'Buy a solenoid valve, 24 V DC, for {c}.',
        'A flexible coupling between the motor and gearbox on {c}. Coupling only.',
        'Hydraulic pump for {c}.', 'Replacement contactor for the motor starter on {c}.',
        'Safety light curtain for {c}.', 'An ultrasonic level sensor for the hopper on {c}.']),
}
MODEL_CODES = ['ZV-420', 'KT-9310', 'QG-805', 'MX 7-A', 'RT/45', 'VB-1710', 'NS12-P', 'LM-330', 'TX-2210', 'P-4471',
               'G-88', 'HB 260', 'S-1904', 'D-560-T', 'AM-17', 'Q12-X', 'BRK-77', 'R-3302', 'CX-5', 'E-990']


def write_values(writer, group_style, specs, keys, rng, *, role='current'):
    """Write several facts as one list; returns nothing, spans are recorded."""
    joiner, end = rng.choice(JOINERS[group_style])
    writer.add(rng.choice(SPEC_LEADS[group_style]))
    for i, key in enumerate(keys):
        if i:
            writer.add(joiner if i < len(keys) - 1 or joiner != ', ' else rng.choice([', ', ' and ']))
        writer.phrase(rng.choice(FORMS[key][group_style]), key, role, specs[key], rng)
    writer.add(end or '.')


def inline(writer, template, style, key, value, rng, role, *, extra=None):
    """Write a bank sentence whose "{x}" (and "{y}") are recorded fact phrases."""
    text = template.replace('{t}', TERMS[key])
    parts = text.split('{x}')
    writer.add(parts[0])
    writer.phrase(rng.choice(FORMS[key][style]), key, role, value, rng, allow_words=False)
    rest = parts[1]
    if '{y}' in rest:
        a, b = rest.split('{y}')
        writer.add(a)
        writer.phrase(rng.choice(FORMS[key][style]), key, role, extra, rng, allow_words=False)
        writer.add(b)
    else:
        writer.add(rest)


def resolve(group, state):
    """Attribute labels from intended state, following the supplied-400 conventions."""
    keys = CATEGORIES[group][1]
    refs = state['reference_components']
    chosen = next((r for r in refs if r['component_id'] == state['selected_reference_component']), None)
    labels = {}
    for key in keys:
        if key in state['contradictory_current_values']:
            labels[key] = 'conflicting'
        elif key in state['ambiguous_current_values']:
            labels[key] = 'unclear'
        elif key in state['current_specifications']:
            current = state['current_specifications'][key]
            reference = chosen.get(key) if (chosen and state['reference_substitution_allowed']) else None
            labels[key] = 'conflicting' if reference is not None and reference != current else 'satisfied'
        elif state['reference_substitution_allowed']:
            sources = [chosen] if chosen else refs
            found = {r[key] for r in sources if key in r}
            labels[key] = 'missing' if not found else 'satisfied' if len(found) == 1 else 'unclear'
        else:
            labels[key] = 'missing'
    states = list(labels.values()) + (['unclear'] if state['unresolved_component_selection'] else [])
    return labels, next(s for s in ('conflicting', 'unclear', 'missing', 'satisfied') if s in states)


SUPPORTED_MODES = (
    ['complete'] * 6 + ['complete_restated'] * 2 + ['missing_one'] * 3 + ['missing_two', 'model_only', 'model_plus_partial'] +
    ['text_conflict'] * 3 + ['corrected'] * 2 + ['obsolete_only', 'negated', 'negated_plus', 'ambiguous'] +
    ['unrelated'] * 2 + ['unitless', 'hp', 'field_error'] + ['ref_full'] * 2 + ['ref_agree', 'ref_conflict', 'ref_partial',
    'ref_incomplete'] + ['ref_disclaimed_partial'] * 2 + ['ref_disclaimed_complete', 'ref_two', 'ref_two_selected'])


# Revision 2, training role only (11 November 2026). The first training run
# still gave disclaimed references substitution permission and read a second
# live value as another item's on held-out phrasings. These extra phrasings are
# never used for development, calibration or evaluation families.
TRAIN_EXTRA = {
    'REF_DISCLAIMED': [
        '{r} is attached for information only - do not copy it.', '{r} was the old part; we want something new, not a copy.',
        'Ignore {r}, it is just the history card.', '{r} is no longer fitted and must not be matched.',
        'The {r} card is outdated and irrelevant for this order.', 'We are not ordering another {r}; the record is only background.',
        '{r} belonged to the previous setup, so please do not use its data.', 'Do not base the order on {r}; it is the scrapped unit.',
        '{r} is shown for comparison only. It is not what we are buying.', 'For context only: {r} was fitted before the conversion.',
        '{r} is the part we took out. The new one is different.', 'Please treat {r} as background - not a like-for-like request.',
        "{r} is not the requirement; it's the old version.", 'The attached {r} is out of date; nothing should be copied from it.',
        '{r} was replaced last year and does not apply any more.', 'Reference {r} is historical, not a specification for this order.',
        'Old record {r} attached by habit - ignore it.', '{r} describes what we had, not what we need.',
        'Not a copy of {r}: the design has changed.', '{r} is only there so you can see what was removed.',
        'Do not order to {r}. That one is obsolete.', 'We moved away from {r}; use only the details I gave.',
        '{r} is the previous model and should be disregarded.', 'Only the values above count; {r} is background.'],
    'REF_FULL': [
        'Same part as {r} again, please.', 'Exactly like {r}, which is the fitted one.', 'Please order the same as {r}.',
        'Like for like: copy {r}.', 'Swap in an identical unit per {r}.', '{r} is what is fitted - same again.',
        'Copy the attached {r} exactly.', 'Another one identical to {r}, please.', 'Replace with the same as {r}.',
        'Use {r} as the specification; same part.', 'Identical replacement for {r}.', 'Match the installed {r}.'],
    'CONFLICT': [
        'The second job card says {x} for this same part, and both are still live.',
        'However the drawing for this part shows {x}; neither has been corrected.',
        'Production asked for {x} for the same part. We have not resolved which.',
        'The latest email says {x} for this replacement. Both values are open.',
        'Stores also has {x} written for this exact part - unresolved.',
        'The old and new sheets disagree: the other one says {x}, and both are current.',
        'The planner wrote {x} for this part too. Nobody has confirmed either.',
        'On the purchase note it is {x} for the same item; both stand.',
        'My colleague insists on {x} for this replacement; we have not agreed.',
        'The service report gives {x} for this same unit, still unresolved.',
        'Also listed for this part: {x}. Both are active and conflicting.',
        'The second measurement of this part gave {x}. Neither has been withdrawn.',
        'Engineering says {x} for this same order line; not reconciled.',
        'There is a second, equally valid note for this part saying {x}.',
        'Another current instruction for this replacement states {x}.',
        'The supplier datasheet for this part lists {x}; we have not decided.'],
    'CORRECTED': [
        'The {x} on the old sheet was a mistake and has been corrected.', 'Previously {x}, now changed - ignore the old figure.',
        'Not {x} any more; that was before the change.', 'The {x} value is withdrawn.', '{x} was the old spec and is cancelled.',
        'Earlier note said {x}; it has been superseded.', 'The superseded value {x} no longer applies.',
        'Old figure {x} crossed out.'],
    'UNRELATED': [
        'The neighbouring unit is {x}; we are not touching that one.', 'Another machine nearby uses {x}, unrelated.',
        'The spare for the other line is {x} - not for this order.', 'The second motor on site is {x}, ignore it.',
        'The adjacent conveyor uses {x}; it stays.', 'A different station has {x}. Not relevant here.',
        'The other gearbox in the cabinet is {x}, not being replaced.', 'For the twin line, {x} - not part of this request.'],
}
BANK_NAMES = {id(CONFLICT): 'CONFLICT', id(CORRECTED): 'CORRECTED', id(UNRELATED): 'UNRELATED',
              id(REF_DISCLAIMED): 'REF_DISCLAIMED', id(REF_FULL): 'REF_FULL', id(NEGATED): 'NEGATED',
              id(OBSOLETE_ONLY): 'OBSOLETE_ONLY'}
SUPPORTED_MODES_2 = (SUPPORTED_MODES + ['text_conflict'] * 4 + ['ref_disclaimed_partial'] * 5 +
                     ['ref_disclaimed_complete'] * 2 + ['ref_full'] * 2 + ['ref_agree'] * 2 +
                     ['corrected', 'unrelated'])


def blank():
    return {'current_specifications': {}, 'reference_components': [], 'selected_reference_component': None,
            'reference_substitution_allowed': False, 'contradictory_current_values': {},
            'ambiguous_current_values': {}, 'excluded_evidence': [], 'unresolved_component_selection': False}


def supported(group, mode, style, rng, token, extra=None):
    kind, keys = CATEGORIES[group]
    extra = extra or {}

    def pick(bank):
        """Choose from a held-out style group; revision 2 adds training-only phrasings."""
        return rng.choice(bank[style] + (extra.get(BANK_NAMES[id(bank)], []) if style == 'train' else []))
    specs = values(group, rng)
    changed = rng.choice(keys)
    state, fields = blank(), {}
    writer = Writer()
    noun = rng.choice(NOUNS[group])
    context = rng.choice(CONTEXTS[style])
    intro = rng.choice(INTROS[style])
    writer.add(intro.replace('{a}', article(noun)).replace('{A}', article(noun).capitalize()).replace('{n}', noun)
               .replace('{c}', context).replace('{C}', context[0].upper() + context[1:]))
    writer.add(' ')
    order = list(keys)
    rng.shuffle(order)
    refs = []

    def card(facts, letter, note='Installed part record'):
        return {'component_id': f'{token[:4].upper()}-{letter}', 'kind': kind, **facts, 'description': note}

    def current(keys_to_write):
        state['current_specifications'].update({k: specs[k] for k in keys_to_write})
        if keys_to_write:
            write_values(writer, style, specs, keys_to_write, rng)

    if mode in ('complete', 'field_error', 'complete_restated'):
        current(order)
        if mode == 'field_error':
            field = rng.choice(list(FIELD_ERRORS))
            fields[field] = FIELD_ERRORS[field][0]
            writer.add(' ' + rng.choice(FIELD_ERRORS[field][1][style]))
    elif mode in ('missing_one', 'missing_two'):
        dropped = order[:1] if mode == 'missing_one' or len(order) == 2 else order[:2]
        current([k for k in order if k not in dropped])
        for k in dropped:
            note = rng.choice(MISSING[style]).replace('{t}', TERMS[k])
            if note:
                writer.add(' ' + note[0].upper() + note[1:])
    elif mode == 'model_only':
        before, after = rng.choice(MODEL_ONLY[style]).split('{m}')
        writer.add(before)
        writer.mark(rng.choice(MODEL_CODES), None, 'type', None)
        writer.add(after)
    elif mode == 'model_plus_partial':
        template = rng.choice(MODEL_PLUS[style])
        before, after = template.split('{m}')
        writer.add(before)
        writer.mark(rng.choice(MODEL_CODES), None, 'type', None)
        writer.add(after + ' ')
        current(order[1:])
        writer.add(' ' + rng.choice([m for m in MISSING[style] if m] or ['The {t} is unknown.']).replace('{t}', TERMS[order[0]]))
    elif mode == 'text_conflict':
        current(order)
        alt = alternative(changed, specs[changed], rng, specs)
        state['contradictory_current_values'] = {changed: [specs[changed], alt]}
        writer.add(' ')
        inline(writer, pick(CONFLICT), style, changed, alt, rng, 'current')
    elif mode == 'corrected':
        current(order)
        alt = alternative(changed, specs[changed], rng, specs)
        state['excluded_evidence'] = [{'reason': 'obsolete', 'facts': {changed: alt}}]
        writer.add(' ')
        inline(writer, pick(CORRECTED), style, changed, alt, rng, 'obsolete')
    elif mode in ('obsolete_only', 'negated'):
        current([k for k in order if k != changed])
        writer.add(' ')
        bank, role = (OBSOLETE_ONLY, 'obsolete') if mode == 'obsolete_only' else (NEGATED, 'negated')
        inline(writer, pick(bank), style, changed, specs[changed], rng, role)
        state['excluded_evidence'] = [{'reason': role, 'facts': {changed: specs[changed]}}]
    elif mode == 'negated_plus':
        current(order)
        alt = alternative(changed, specs[changed], rng, specs)
        writer.add(' ')
        inline(writer, rng.choice(NEGATED_PLUS[style]), style, changed, alt, rng, 'negated')
        state['excluded_evidence'] = [{'reason': 'negated', 'facts': {changed: alt}}]
    elif mode == 'ambiguous':
        current([k for k in order if k != changed])
        alt = alternative(changed, specs[changed], rng, specs)
        state['ambiguous_current_values'] = {changed: [specs[changed], alt]}
        writer.add(' ')
        inline(writer, rng.choice(AMBIGUOUS[style]), style, changed, specs[changed], rng, 'ambiguous', extra=alt)
    elif mode == 'unrelated':
        current(order)
        alt = alternative(changed, specs[changed], rng, specs)
        state['excluded_evidence'] = [{'reason': 'other_item', 'facts': {changed: alt}}]
        writer.add(' ')
        inline(writer, pick(UNRELATED), style, changed, alt, rng, 'background')
    elif mode == 'unitless':
        candidates = [k for k in order if k not in ('phases', 'gear_ratio', 'belt_profile', 'output_type')]
        changed = rng.choice(candidates)
        current([k for k in order if k != changed])
        state['ambiguous_current_values'] = {changed: {'number': specs[changed], 'unit': None}}
        template = rng.choice(UNITLESS[style]).replace('{t}', TERMS[changed])
        before, after = template.split('{x}')
        writer.add(' ' + before)
        writer.mark(f'{specs[changed]:g}', changed, 'current', specs[changed])
        writer.add(after)
    elif mode == 'hp':
        if group != 'MG_MOTORS':
            return supported(group, 'missing_one', style, rng, token)
        current([k for k in order if k != 'power_kw'])
        template = rng.choice(HP[style])
        before, after = template.split('{x}')
        writer.add(' ' + before)
        writer.mark(f'{rng.choice([0.5, 1, 1.5, 2, 3, 5, 7.5, 10])} hp'.replace('.0 ', ' '), 'power_kw', 'current', None)
        writer.add(after)
    elif mode.startswith('ref_'):
        state['reference_substitution_allowed'] = mode not in ('ref_disclaimed_partial', 'ref_disclaimed_complete')
        if mode in ('ref_full', 'ref_agree', 'ref_conflict', 'ref_partial', 'ref_incomplete',
                    'ref_disclaimed_partial', 'ref_disclaimed_complete'):
            facts = {k: v for k, v in specs.items() if not (mode in ('ref_partial', 'ref_incomplete') and k == changed)}
            if mode == 'ref_disclaimed_complete':
                facts = {k: alternative(k, v, rng, specs) for k, v in specs.items()}
            refs = [card(facts, rng.choice('ABCDEFGH'), 'Former installation record' if 'disclaimed' in mode else 'Installed part record')]
            r = refs[0]['component_id']
            state['selected_reference_component'] = r if state['reference_substitution_allowed'] else None
            if mode == 'ref_full':
                writer.add(pick(REF_FULL).replace('{r}', r))
            elif mode == 'ref_agree':
                writer.add(pick(REF_FULL).replace('{r}', r) + ' ')
                current(order[:rng.randint(1, len(order))])
            elif mode == 'ref_conflict':
                alt = alternative(changed, specs[changed], rng, specs)
                state['current_specifications'][changed] = alt
                template = rng.choice(REF_CONFLICT[style]).replace('{r}', r)
                before, after = template.split('{x}')
                writer.add(before)
                writer.phrase(rng.choice(FORMS[changed][style]), changed, 'current', alt, rng, allow_words=False)
                writer.add(after)
            elif mode == 'ref_partial':
                state['current_specifications'][changed] = specs[changed]
                inline(writer, rng.choice(REF_PARTIAL[style]).replace('{r}', r), style, changed, specs[changed], rng, 'current')
            elif mode == 'ref_incomplete':
                writer.add(rng.choice(REF_INCOMPLETE[style]).replace('{r}', r).replace('{t}', TERMS[changed]))
            else:
                # Disclaimed record: stated facts agree with it, or describe new values.
                stated = order if mode == 'ref_disclaimed_complete' else order[:rng.randint(1, len(order) - 1)]
                current(stated)
                writer.add(' ' + pick(REF_DISCLAIMED).replace('{r}', r))
                state['excluded_evidence'] = [{'reason': 'disclaimed_reference', 'component_id': r}]
        else:
            first, second = rng.sample('ABCDEFGH', 2)
            other = dict(specs)
            other[changed] = alternative(changed, specs[changed], rng, specs)
            refs = [card(specs, first), card(other, second)]
            r, s = refs[0]['component_id'], refs[1]['component_id']
            if mode == 'ref_two':
                state['unresolved_component_selection'] = True
                writer.add(rng.choice(REF_TWO[style]).replace('{r}', r).replace('{s}', s))
            else:
                state['selected_reference_component'] = r
                writer.add(rng.choice(REF_TWO_SELECTED[style]).replace('{r}', r).replace('{s}', s))
            rng.shuffle(refs)
    else:
        raise AssertionError(mode)
    state['reference_components'] = refs
    labels, overall = resolve(group, state)
    return writer, state, labels, overall, fields, context


def scope_case(group, style, rng):
    """Authored scope sentences; category words appear as distractors."""
    template = rng.choice(SCOPE[group][style])
    context = rng.choice(CONTEXTS[style])
    p, q = rng.sample([rng.choice(NOUNS[g]) for g in CATEGORIES], 2)
    text = (template.replace('{c}', context).replace('{C}', context[0].upper() + context[1:])
            .replace('{p}', p).replace('{q}', q).replace('{a2}', article(q)).replace('{a}', article(p))
            .replace('{k}', str(rng.randrange(10, 99))))
    writer = Writer()
    writer.add(text)
    return writer, blank(), {}, None, {}, context


def build_case(split, number, revision=1):
    style = STYLE_GROUP[split]
    if revision != 1 and split != 'train':
        raise ValueError('Only training families have a second revision')
    salt = '' if revision == 1 else f':r{revision}'
    token = hashlib.sha256(f'{VERSION}:{SEED}:{split}:{number}{salt}'.encode()).hexdigest()[:12]
    rng = random.Random(token)
    group = rng.choices(list(CATEGORIES) + list(SCOPE), weights=[14] * 6 + [5, 5, 6])[0]
    if group in SCOPE:
        mode = 'scope'
        writer, state, labels, overall, fields, context = scope_case(group, style, rng)
    else:
        mode = rng.choice(SUPPORTED_MODES if revision == 1 else SUPPORTED_MODES_2)
        writer, state, labels, overall, fields, context = supported(group, mode, style, rng, token,
                                                                    TRAIN_EXTRA if revision == 2 else None)
    family = 'NAT-F-' + token.upper()
    case_id = 'NAT-' + token.upper() + '-0'
    request = {'item_id': case_id, 'description': writer.text, 'quantity': rng.choice([1, 2, 3, 4, 5, 6, 8, 10]),
               'unit': 'EA', 'plant': rng.choice(['1100', '1200']),
               'required_date': rng.choice(['2026-12-04', '2026-12-18', '2027-01-15', '2027-02-12', '2027-03-05'])}
    request.update(fields)
    if state['reference_components']:
        request['reference_id'] = 'REF-NAT-' + token.upper()
    annotated = [s for s in writer.spans]
    case = {'case_id': case_id, 'family_id': family, 'primary': True, 'variant': 0, 'split': split,
            'contract_version': 'pilot-0.4', 'author': AUTHOR, 'review_status': 'constructed_not_independently_reviewed',
            'request': request, 'reference_fixture': state['reference_components'],
            'writing_template_id': f'{VERSION}:{style}:{mode}' + ('' if revision == 1 else f':r{revision}'),
            'semantic_labels': {'group': group, 'identification': overall, 'attributes': labels},
            'intended_facts': {**state, 'blueprint': mode, 'scenario_context': context},
            'tags': ['synthetic', 'natural_wording', 'slice:' + mode, 'style:' + style]}
    if group in CATEGORIES:
        case['evidence_annotations'] = {'version': SPAN_VERSION,
                                        'description_sha256': hashlib.sha256(writer.text.encode()).hexdigest(),
                                        'spans': annotated}
    return case


def check(case, config):
    """Writer-recorded spans must reproduce exactly; labels must route."""
    text = case['request']['description']
    if len(text) > 600:
        raise ValueError('Overlong natural request')
    for s in case.get('evidence_annotations', {}).get('spans', []):
        if text[s['start']:s['end']] != s['surface']:
            raise ValueError('Writer span does not match its text')
    spans = sorted(case.get('evidence_annotations', {}).get('spans', []), key=lambda s: s['start'])
    if any(a['end'] > b['start'] for a, b in zip(spans, spans[1:])):
        raise ValueError('Overlapping writer spans')
    case['expected'] = expected_route(case, config)
    return case


def build(directory, *, config_path):
    """Write all five roles with an exact 80/20 family split and a loadable manifest."""
    directory = Path(directory)
    if directory.exists():
        raise ValueError('Preserve an existing natural corpus')
    config = json.loads(Path(config_path).read_text())
    directory.mkdir(parents=True)
    parts, files = {}, {}
    for split in SPLITS:
        rows = [check(build_case(split, n), config) for n in range(FAMILIES[split])]
        if len({c['family_id'] for c in rows}) != len(rows):
            raise ValueError('Repeated natural family')
        path = directory / (split + '.jsonl')
        path.write_text(''.join(stable_json(c) + '\n' for c in rows))
        parts[split] = {'file': path.name, 'cases': len(rows), 'families': len(rows),
                        'annotated_cases': sum('evidence_annotations' in c for c in rows),
                        'group_counts': dict(Counter(c['semantic_labels']['group'] for c in rows)),
                        'ready_cases': sum(c['expected']['status'] == 'ready_for_buyer_review' for c in rows),
                        'style_group': STYLE_GROUP[split]}
    (directory / 'pilot_config.json').write_text(json.dumps(config, indent=2) + '\n')
    manifest = {'version': DATA_VERSION, 'generator': VERSION, 'scope': 'synthetic_prototype',
                'source_generator_sha256': file_sha256(__file__), 'partitions': parts,
                'total_families': sum(FAMILIES.values()), 'evaluation_fraction': .2,
                'split_unit': 'scenario_family', 'style_holdout': True,
                'original_400_retained': False, 'exposed_author100_included': False,
                'previous_evaluation_included': False, 'independently_human_reviewed': False,
                'files_sha256': {p.name: file_sha256(p) for p in sorted(directory.iterdir())}}
    (directory / 'dataset-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


COMBINED_GENERATOR = 'ownership-comparison-data-1+' + VERSION


def combine(directory, generated, natural):
    """Add the natural corpus to the existing cohort, role by role, without edits.

    Every existing case keeps its bytes and role. The combined inventory keeps
    an exact 80/20 family split, and the manifest records both parents.
    """
    from .evidence_data import load_partition
    directory, generated, natural = map(Path, (directory, generated, natural))
    if directory.exists():
        raise ValueError('Preserve existing combined data')
    directory.mkdir(parents=True)
    parts, families, config = {}, {}, None
    for role in SPLITS:
        rows = []
        for parent in (generated, natural):
            local, cases, _ = load_partition(parent, role, allowed_splits={role})
            if config is not None and local != config:
                raise ValueError('Different purchasing contracts')
            config = local
            rows.extend(cases)
        for c in rows:
            if families.setdefault(c['family_id'], role) != role:
                raise ValueError('Combined family role overlap')
        if len({c['case_id'] for c in rows}) != len(rows):
            raise ValueError('Combined case identity overlap')
        path = directory / (role + '.jsonl')
        path.write_text(''.join(stable_json(c) + '\n' for c in rows))
        parts[role] = {'file': path.name, 'cases': len(rows), 'families': len({c['family_id'] for c in rows}),
                       'annotated_cases': sum('evidence_annotations' in c for c in rows),
                       'natural_cases': sum(c.get('author') == AUTHOR for c in rows),
                       'group_counts': dict(Counter(c['semantic_labels']['group'] for c in rows))}
    if len(families) != 5 * parts['prototype_evaluation']['families']:
        raise ValueError('Combined inventory must retain an exact 80/20 family split')
    (directory / 'pilot_config.json').write_text(json.dumps(config, indent=2) + '\n')
    manifest = {'version': DATA_VERSION, 'generator': COMBINED_GENERATOR, 'scope': 'synthetic_prototype',
                'source_generator_sha256': file_sha256(__file__),
                'parent_manifest_sha256': [file_sha256(p / 'dataset-manifest.json') for p in (generated, natural)],
                'generated_parent_manifest_sha256': file_sha256(generated / 'dataset-manifest.json'),
                'partitions': parts, 'total_families': len(families), 'evaluation_fraction': .2,
                'split_unit': 'scenario_family', 'original_400_retained': True, 'style_holdout': True,
                'natural_revision': json.loads((natural / 'dataset-manifest.json').read_text()).get('natural_revision', 1),
                'exposed_author100_included': False, 'previous_evaluation_included': False,
                'independently_human_reviewed': False,
                'files_sha256': {p.name: file_sha256(p) for p in sorted(directory.iterdir())}}
    (directory / 'dataset-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def build_revision2(directory, first, supplied_root):
    """Replace only the training families; every other role is copied byte for byte.

    Development, calibration, threshold-selection and reserved families stay
    those of the first corpus, so results remain comparable across runs.
    """
    from .evidence_data import load_partition
    directory, first = Path(directory), Path(first)
    if directory.exists():
        raise ValueError('Preserve an existing natural corpus')
    config = json.loads((first / 'pilot_config.json').read_text())
    previous = json.loads((first / 'dataset-manifest.json').read_text())
    for role in SPLITS:
        load_partition(first, role, allowed_splits={role})
    directory.mkdir(parents=True)
    rows = [check(build_case('train', n, revision=2), config) for n in range(FAMILIES['train'])]
    old_families = {json.loads(line)['family_id'] for role in SPLITS
                    for line in (first / (role + '.jsonl')).read_text().splitlines()}
    if len({c['family_id'] for c in rows}) != len(rows) or {c['family_id'] for c in rows} & old_families:
        raise ValueError('Revision-2 training families must be new and unique')
    # Gold-span views of the supplied natural 400 (same families, training role).
    from .supplied_spans import views as supplied_views
    _, generated, _ = load_partition(Path(supplied_root), 'train', allowed_splits={'train'})
    gold, gold_audit = supplied_views(generated)
    rows = rows + gold
    (directory / 'train.jsonl').write_text(''.join(stable_json(c) + '\n' for c in rows))
    for role in SPLITS[1:]:
        (directory / (role + '.jsonl')).write_bytes((first / (role + '.jsonl')).read_bytes())
    (directory / 'pilot_config.json').write_bytes((first / 'pilot_config.json').read_bytes())
    parts = dict(previous['partitions'])
    natural_rows = [c for c in rows if c['author'] == AUTHOR]
    parts['train'] = {'file': 'train.jsonl', 'cases': len(rows), 'families': len(natural_rows),
                      'supplied_gold_span_views': gold_audit,
                      'annotated_cases': sum('evidence_annotations' in c for c in rows),
                      'group_counts': dict(Counter(c['semantic_labels']['group'] for c in rows)),
                      'ready_cases': sum(c['expected']['status'] == 'ready_for_buyer_review' for c in natural_rows),
                      'style_group': 'train', 'revision': 2,
                      'blueprints': dict(Counter(c['intended_facts']['blueprint'] for c in natural_rows))}
    manifest = {**previous, 'source_generator_sha256': file_sha256(__file__), 'partitions': parts,
                'natural_revision': 2, 'first_revision_manifest_sha256': file_sha256(first / 'dataset-manifest.json'),
                'files_sha256': {p.name: file_sha256(p) for p in sorted(directory.iterdir())}}
    (directory / 'dataset-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest
