"""Training-only facts-to-text views with writer-recorded source intervals.

These are augmentations of existing training families, not new independent
families. They never read or rewrite development, calibration or reserved text.
Original requests, including the supplied 400, stay intact in the parent corpus.
"""
import copy
import hashlib
import random

from .data_v2 import PRODUCTS
from .evidence_span_data import VERSION, Writer
from .task_data import words

PREFIXES = {
    'power_kw': ('power ', 'rated output ', 'power rating ', 'motor output '),
    'voltage_v': ('voltage ', 'operating voltage ', 'electrical rating ', 'supply '),
    'phases': ('connection ', 'phase count ', 'electrical connection '),
    'diameter_mm': ('diameter ', 'roller dia ', 'outside diameter '),
    'length_mm': ('length ', 'overall length ', 'L=', 'working length '),
    'gear_ratio': ('ratio ', 'reduction ', 'gear reduction '),
    'shaft_diameter_mm': ('shaft diameter ', 'shaft dia ', 'output shaft '),
    'width_mm': ('width ', 'overall width ', 'W=', 'section width '),
    'belt_profile': ('profile ', 'belt profile ', 'section ', 'cross section '),
    'bore_mm': ('bore ', 'inner diameter ', 'ID ', 'inside diameter '),
    'outer_diameter_mm': ('outer diameter ', 'outside dia ', 'OD ', 'external diameter '),
    'sensing_distance_mm': ('sensing distance ', 'range ', 'detection range ', 'switching distance '),
    'supply_voltage_v': ('supply ', 'supply voltage ', 'power supply ', 'operating voltage '),
    'output_type': ('output type ', 'output configuration ', 'switching output ', 'output '),
}
SUFFIX_LABELS = {
    'power_kw': (' output rating', ' motor power'),
    'voltage_v': (' operating voltage', ' electrical supply'),
    'phases': (' connection', ' electrical system'),
    'diameter_mm': (' roller diameter', ' diameter'),
    'length_mm': (' overall length', ' working length'),
    'gear_ratio': (' reduction ratio', ' gear ratio'),
    'shaft_diameter_mm': (' output shaft diameter', ' shaft diameter'),
    'width_mm': (' overall width', ' section width'),
    'belt_profile': (' belt profile', ' cross section'),
    'bore_mm': (' bore diameter', ' inside diameter'),
    'outer_diameter_mm': (' outside diameter', ' external diameter'),
    'sensing_distance_mm': (' detection range', ' sensing distance'),
    'supply_voltage_v': (' supply voltage', ' power supply'),
    'output_type': (' switching output', ' output configuration'),
}
UNRESOLVED_REFERENCE = (
    'No component has been selected; candidates are {candidates}. ',
    'The choice between {candidates} remains open. ',
    'Do not assume a component: {candidates} are still under consideration. ',
    'Component selection is pending for {candidates}. ',
    'The linked record lists {candidates}; the requester has not chosen one. ',
    'We cannot yet identify which of {candidates} is being replaced. ',
    'The maintenance team must confirm a selection from {candidates}. ',
    'There is no confirmed component among {candidates}. ',
)
CLAUSES = {
    'current': ('For the item on this order: ', 'Deliver the replacement with ', 'Requested ratings: ',
                'The purchased unit needs ', 'Use these figures for the ordered component: ',
                'Specify the new unit as ', 'The replacement rating must be ', 'For delivery: '),
    'background': ('Another device, excluded from this order, has ',
                   'The adjacent equipment will stay installed; its ratings are ',
                   'These are figures for a different component, not the purchase: ',
                   'For information, the separate unit runs at ',
                   'Do not apply these figures to the purchase; they belong to another item: ',
                   'The second machine is not being replaced and uses ',
                   'An unrelated component elsewhere is rated ',
                   'Keep the other assembly, which has ',
                   'Only the neighboring device has ',
                   'A different piece of equipment, remaining in service, uses ',
                   'For the other installation only: ',
                   'The figures for the retained neighboring component are '),
    'obsolete': ('Canceled figures for this component: ', 'The earlier revision specified ',
                 'Disregard the superseded rating of ', 'Before the change, this item had ',
                 'This withdrawn specification no longer applies: ', 'Old figures, now obsolete: '),
    'negated': ('Do not deliver ', 'The rejected alternative has ', 'Exclude the version with ',
                'For this order we have ruled out ', 'This rating is expressly unwanted: '),
    'ambiguous': ('Still awaiting a choice: ', 'The rating has not been confirmed: ',
                  'The requester has left open ', 'An unresolved proposal specifies ',
                  'Confirmation is required before choosing '),
}


def marked_fact(writer, key, value, role, rng, *, version=1):
    """Choose the complete value/unit before insertion; no numeric text search."""
    value_first = version == 2 and rng.random() < .5
    if not value_first:
        writer.append(rng.choice(PREFIXES[key]))
    number = f'{value:g}' if isinstance(value, (float, int)) else str(value)
    if isinstance(value, (float, int)) and value < 1000 and rng.random() < .2:
        number = words(value)
    suffix = (rng.choice((' kW', 'kW', ' kilowatts')) if key == 'power_kw' else
              rng.choice((' V', 'V', ' volts')) if key in ('voltage_v', 'supply_voltage_v') else
              rng.choice(('-phase', ' phases', ' ph', '~')) if key == 'phases' else
              ':1' if key == 'gear_ratio' else
              rng.choice((' mm', 'mm', ' millimetres', ' millimeters')) if key.endswith('_mm') else '')
    writer.marked(number + suffix, attribute=key, role=role, value=value)
    if value_first:
        writer.append(rng.choice(SUFFIX_LABELS[key]))


def view(case, *, epoch, seed=83, version=1):
    if version not in (1, 2):
        raise ValueError('Undeclared attribution wording version')
    if case['split'] != 'train':
        raise ValueError('Attribution wording views are training-only')
    if case.get('author') not in ('assistant_relational_curriculum', 'assistant_facts_first_broadening',
                                  'assistant_writer_recorded_spans'):
        return None
    state = case.get('intended_facts', {})
    group = case['semantic_labels']['group']
    if group not in PRODUCTS or 'current_specifications' not in state:
        return None
    # Retain the original annotated writer cases in the training corpus. Their
    # noncurrent facts are not redundantly reconstructed from incomplete state.
    if 'evidence_annotations' in case:
        return None
    rng = random.Random(f'attribution-view:{seed}:{epoch}:{case["family_id"]}')
    writer = Writer()
    item = rng.choice(PRODUCTS[group][2])
    writer.append(rng.choice(('Please buy {item}. ', 'Replacement {item} required. ',
                              'Purchase request for {item}. ', 'Source {item} for maintenance. ')).format(item=item))
    if rng.random() < .6:
        code = rng.choice(('MX-', 'TR-', 'Z', '')) + str(rng.randrange(100, 9999))
        writer.append(rng.choice(('Type ', 'Catalog model ', 'Product identifier ')))
        writer.marked(code, attribute=None, role='type', value=code)
        writer.append('. ')
    refs = case['reference_fixture']
    if refs:
        if state['reference_substitution_allowed']:
            writer.append(rng.choice(('Match the recorded unit exactly. ', 'Like-for-like replacement. ',
                                      'Keep the existing specifications. ', 'Use the installed component as the specification. ')))
            selected = state['selected_reference_component']
            if selected is not None and len(refs) > 1:
                writer.append('The requested component is ' + selected + '. ')
            elif version == 2 and selected is None and len(refs) > 1:
                candidates = ' or '.join(r['component_id'] for r in refs)
                writer.append(rng.choice(UNRESOLVED_REFERENCE).format(candidates=candidates))
        else:
            writer.append(rng.choice(('The linked record concerns other equipment and must not supply missing ratings. ',
                                      'The attached record is background only, for a different item. ')))
    current = state['current_specifications']
    conflicts = state.get('contradictory_current_values', {})
    ambiguous = state.get('ambiguous_current_values', {})
    chunks = [('current', [(key, value) for key, value in current.items() if key not in conflicts and key not in ambiguous])]
    for key, values in conflicts.items():
        chunks.extend(('current', [(key, value)]) for value in values)
    for key, values in ambiguous.items():
        chunks.append(('ambiguous', [(key, value) for value in values]))
    ignored = state.get('ignored_specifications', {})
    if ignored:
        role = 'obsolete' if state['blueprint'].startswith('obsolete') else 'background'
        chunks.append((role, list(ignored.items())))
    # Add semantically irrelevant facts to varied training families so ownership
    # cannot be learned only from one dedicated scenario template.
    available = state.get('base_specifications', current)
    if available and rng.random() < .75:
        key = rng.choice(list(available))
        chunks.append((rng.choice(('background', 'background', 'obsolete', 'negated')), [(key, available[key])]))
    rng.shuffle(chunks)
    for role, facts in chunks:
        if not facts:
            continue
        writer.append(rng.choice(CLAUSES[role]))
        rng.shuffle(facts)
        for i, (key, value) in enumerate(facts):
            if i:
                writer.append(' or ' if role == 'ambiguous' else rng.choice((', ', '; ')))
            marked_fact(writer, key, value, role, rng, version=version)
        writer.append('. ')
    for key, value in state.get('unsupported_units', {}).items():
        if key != 'power_hp':
            raise ValueError('Unknown unsupported-unit training state')
        writer.append('For the ordered part, power ')
        writer.marked(f'{value:g} hp', attribute='power_kw', role='current', value=value)
        writer.append('. ')
    writer.append('Job location: line ' + str(rng.randrange(1, 9999)) + '.')
    result = copy.deepcopy(case)
    result['case_id'] = case['case_id'] + ('-AV' if version == 1 else '-AV2-') + str(epoch)
    result['request']['item_id'] = result['case_id']
    result['primary'] = False
    result['request']['description'] = writer.text
    result['evidence_annotations'] = {'version': VERSION, 'description_sha256': hashlib.sha256(writer.text.encode()).hexdigest(),
                                      'spans': writer.spans}
    result['training_view'] = {'version': 'attribution-view-' + str(version), 'epoch': epoch, 'parent_case_id': case['case_id'],
                               'counts_as_independent_family': False}
    return result
