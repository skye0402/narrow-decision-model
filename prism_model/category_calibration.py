"""Error-aware, pooled category acceptance; no inference-time text rules."""
import hashlib
import json
import math
import random

from .metrics import wilson
from .evidence_data import make_case

VERSION = 'pooled-category-acceptance-1'
CUTOFFS = (.50, .60, .70, .80, .85, .90, .925, .95, .975, .99, .995, .999)
AUTHOR = 'assistant_category_calibration_20261115'
PRODUCTS = {
 'MG_MOTORS': ('electric motor', 'motor', '5.5 kW, 400 V, three-phase',
               'power_kw', 'voltage_v', 'phases'),
 'MG_CONVEYOR': ('conveyor roller', 'roller', 'diameter 50 mm, length 500 mm',
                 'diameter_mm', 'length_mm'),
 'MG_GEARBOXES': ('gearbox', 'gearbox', 'ratio 5:1, shaft diameter 20 mm',
                  'gear_ratio', 'shaft_diameter_mm'),
 'MG_BELTS': ('drive belt', 'belt', 'width 25 mm, length 750 mm, profile SPB',
              'width_mm', 'length_mm', 'belt_profile'),
 'MG_BEARINGS': ('ball bearing', 'bearing', 'bore 25 mm, outside diameter 52 mm, width 15 mm',
                 'bore_mm', 'outer_diameter_mm', 'width_mm'),
 'MG_SENSORS': ('inductive proximity sensor', 'sensor', 'sensing distance 8 mm, supply 24 V, NPN output',
                'sensing_distance_mm', 'supply_voltage_v', 'output_type'),
}
FRAMES = {
 'calibration': ('Could purchasing obtain {item} for station {station}? {details}',
                 'Failure report at station {station}: we require {item}. {details}',
                 'For the spare-part cabinet at {station}, put {item} on the order. {details}',
                 'I am requesting {item} for repair job {station}. {details}'),
 'threshold_selection': ('Please add {item} to the purchase list for station {station}. {details}',
                         'Workshop {station} needs {item} delivered. {details}',
                         'New order line for asset {station}: {item}. {details}',
                         'For maintenance task {station}, source {item}. {details}'),
}


def challenge(config, role):
    """Declared intent supplies labels; whole families/frames stay in one role."""
    if role not in FRAMES:
        raise ValueError('Only declared fitting/verification roles are allowed')
    cases = []
    groups = list(PRODUCTS)
    for group, (noun, short, specs, *attrs) in PRODUCTS.items():
        for number in range(32):
            rng = random.Random(f'category-facts:{role}:{group}:{number}')
            value = {a: rng.choice({'power_kw': (1.5, 3, 5.5, 7.5), 'voltage_v': (230, 400, 415),
                                   'phases': (1, 3), 'diameter_mm': (40, 50, 63, 76),
                                   'length_mm': (300, 500, 750, 900), 'gear_ratio': (5, 8, 10, 15),
                                   'shaft_diameter_mm': (16, 20, 25, 32), 'width_mm': (12, 15, 20, 25),
                                   'belt_profile': ('SPA', 'SPB', 'AT5'), 'bore_mm': (15, 20, 25, 30),
                                   'outer_diameter_mm': (42, 47, 52, 62), 'sensing_distance_mm': (4, 5, 8, 10),
                                   'supply_voltage_v': (12, 24), 'output_type': ('PNP', 'NPN')}[a]) for a in attrs}
            surface = lambda a: {'phases': f'{value[a]}-phase', 'gear_ratio': f'ratio {value[a]}:1',
                                  'power_kw': f'{value[a]} kW', 'voltage_v': f'{value[a]} V',
                                  'supply_voltage_v': f'supply {value[a]} V', 'output_type': f'{value[a]} output',
                                  'belt_profile': f'profile {value[a]}'}.get(a, a.replace('_mm','').replace('_',' ') + f' {value[a]} mm')
            specs = ', '.join(surface(a) for a in attrs)
            changed = 'voltage_v' if group == 'MG_MOTORS' else attrs[0]
            alternate = {'voltage_v': 110, 'diameter_mm': 89, 'gear_ratio': 20,
                         'width_mm': 32, 'bore_mm': 40, 'sensing_distance_mm': 15}[changed]
            conflict = (f'The required {changed.replace("_mm", "").replace("_v", "").replace("_", " ")} is also '
                        f'{alternate}{":1" if changed == "gear_ratio" else " V" if changed == "voltage_v" else " mm"}.')
            mode = number % 4
            details = ('Required specifications: ' + specs + '.' if mode == 0 else
                       'The identifying specifications are not available yet.' if mode == 1 else
                       'Required specifications: ' + specs + '. ' +
                       conflict if mode == 2 else
                       'This is for the conveyor motor assembly; that context is not an additional purchase. '
                       'Required specifications: ' + specs + '.')
            # Each family has one preselected primary; values are intentionally
            # simple so the challenge tests category/scope transfer, not units.
            text = FRAMES[role][number % 4].format(item=noun, details=details, station=f'{role[:1]}-{number+101}')
            labels = {a: 'satisfied' for a in attrs}
            if mode == 1: labels = {a: 'missing' for a in attrs}
            if mode == 2: labels[attrs[0] if group != 'MG_MOTORS' else 'voltage_v'] = 'conflicting'
            overall = 'missing' if mode == 1 else 'conflicting' if mode == 2 else 'satisfied'
            token = hashlib.sha256(f'category:{role}:{group}:{number}'.encode()).hexdigest()[:12].upper()
            case = make_case(config, token, 0, role, text, group, overall, labels, [], None, 'category_challenge')
            case.update(author=AUTHOR, primary=True)
            case['intended_facts']['category_calibration_values'] = value
            cases.append(case)
    outsiders = ('motor starter contactor', 'a hydraulic pump', 'a pneumatic valve',
                 'a capacitive proximity sensor', 'a motor control inverter', 'a conveyor belt scraper')
    for group in ('no_suitable_group', 'mixed_scope', 'insufficient_evidence'):
        for number in range(32):
            a = groups[number % 6]; b = groups[(number + 1 + (number // 6) % 5) % 6]
            first, second = PRODUCTS[a][0], PRODUCTS[b][0]
            if group == 'no_suitable_group':
                item = outsiders[number % len(outsiders)]
                details = 'Only this component is requested, not the motor or conveyor it belongs to.'
                if number % 2:
                    item = f'inspection service for the {first}'
                    details = 'Labor only; no replacement goods are being ordered.'
            elif group == 'mixed_scope':
                item = f'{first} and {second}'
                details = 'Both products are ordered on this one line.'
            else:
                item = f'either {first} or {second}'
                details = 'The required item has not been identified; do not choose between them yet.'
            text = FRAMES[role][number % 4].format(item=item, details=details, station=f'{role[:1]}-{number+201}')
            token = hashlib.sha256(f'category:{role}:{group}:{number}'.encode()).hexdigest()[:12].upper()
            case = make_case(config, token, 0, role, text, group, None, {}, [], None, 'category_challenge')
            case.update(author=AUTHOR, primary=True)
            cases.append(case)
    return cases


def acceptance(rows, cutoff, outcomes):
    kept = [r for r in rows if r['confidence'] >= cutoff]
    good = sum(r['choice'] == r['label'] for r in kept)
    per = {}
    for outcome in outcomes:
        local = [r for r in kept if r['choice'] == outcome]
        per[outcome] = wilson(sum(r['choice'] == r['label'] for r in local), len(local))
    interval = wilson(good, len(kept))
    return {'threshold': cutoff, 'accepted': len(kept), 'correct': good,
            'errors': len(kept)-good, 'precision': interval, 'per_outcome': per,
            'eligible': len(kept) >= 150 and (interval['estimate'] or 0.) >= .98
                        and (interval['lower'] or 0.) >= .90}


def fit(rows, outcomes):
    if not rows or len({r['family_id'] for r in rows}) != len(rows):
        raise ValueError('Require one independent primary per calibration family')
    if any(not r['primary'] or r['split'] != 'calibration' or
           r['choice'] not in outcomes or r['label'] not in outcomes or
           type(r['confidence']) not in (int, float) or not math.isfinite(r['confidence']) or
           not 0 <= r['confidence'] <= 1 for r in rows):
        raise ValueError('Only valid, primary calibration observations can fit the rule')
    if not any(r['choice'] != r['label'] for r in rows):
        raise ValueError('Require observed category mistakes; do not reuse a sample minimum')
    candidates = [acceptance(rows, t, outcomes) for t in CUTOFFS]
    chosen = next((r for r in candidates if r['eligible']), None)
    return {'version': VERSION, 'threshold': None if chosen is None else chosen['threshold'],
            'observations': len(rows), 'observed_errors': sum(r['choice'] != r['label'] for r in rows),
            'candidates': candidates, 'selected': chosen,
            'selection': 'lowest predeclared fixed-grid cutoff meeting pooled precision/support targets'}
