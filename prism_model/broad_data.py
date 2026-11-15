"""Crossed fictional facts and narrative frames for the broadened experiment.

Reuses the checked normalized scenario-state constructor, not previous corpora.
The writer never reads the exposed 100 requests or their predictions. Authorship
is still this assistant's synthetic generation, not an independent human author.
"""
from collections import Counter
import copy
import json
from pathlib import Path
import random
import zipfile

from . import evidence_data as base
from .data_v2 import PRODUCTS, META
from .provenance import file_sha256
from .task_data import expected_route, stable_json, words

ROOT=Path(__file__).resolve().parents[1]
COUNTS=base.COUNTS
MODES=tuple('reference_incomplete' if m=='reference_only' else m for m in base.MODES if m!='policy_counterfactual')+(
    'unitless','reference_background','reference_common_candidates',
    'reference_conflict','reference_partial_conflict','ambiguous_value',
    'missing_current_pair','missing_current_pair','missing_current_pair')
FRAMES={
 'train': ('Need {item} for {context}. {body}', 'Please buy {item}. {body} It goes in {context}.',
           '{context}: order {item}. {body}', 'Maintenance request for {item}: {body}',
           'The part being ordered is {item}, for {context}. {body}',
           'Can purchasing get {item}? {body} Location: {context}.',
           '{body} This is a request for {item} in {context}.',
           'A replacement {item} is needed at {context}. {body}'),
 'development': ('For the repair at {context}, procure {item}. {body}',
                 'Workshop message: {item} wanted. {body} Equipment: {context}.',
                 'Order line for {context} / {item}. {body}',
                 'The technician asks for {item}. {body} Job at {context}.'),
 'calibration': ('Purchasing note from {context}: {item}. {body}',
                 'Spare required: {item} serving {context}. {body}',
                 'Arrange a supply of {item} for {context}. {body}'),
 'threshold_selection': ('To complete work at {context}, get {item}. {body}',
                         '{item} is the purchase for {context}; {body}',
                         'Please raise an order for {item}. {body} Application: {context}.'),
 'prototype_evaluation': ('For the upcoming job on {context}, the requested spare is {item}. {body}',
                          'Part requisition / {item} / equipment {context}. {body}',
                          'We are sourcing {item} to repair {context}. {body}',
                          'Procure {item}; use the following note from {context}. {body}',
                          'This order is for {item}. {body} The equipment is {context}.'),
}
CONTEXTS=('the seal test stand','the film cutter','the laboratory lift','the ceramic press',
          'the carton feeder','the weighing hopper','the drying chamber','the assembly cell',
          'the timber scanner','the water treatment skid','the fabric inspection line',
          'the heat-sealing station','the pallet elevator','the test conveyor','the component washer',
          'the sorting station','the wrapping machine','the robot tooling fixture','the dosing line',
          'the sample transport rig','the foam cutting table','the filter cleaning unit')
TERMS={'power_kw':('power','rated output'),'voltage_v':('voltage','operating voltage'),
       'diameter_mm':('roller diameter','roller dia'),'length_mm':('length','overall length'),
       'shaft_diameter_mm':('shaft diameter','shaft dia'),'width_mm':('width','overall width'),
       'bore_mm':('bore','inner diameter'),'outer_diameter_mm':('outer diameter','outside dia'),
       'sensing_distance_mm':('sensing distance','detection range'),'supply_voltage_v':('supply voltage','power supply')}
UNITS={'power_kw':'kW','voltage_v':'V','supply_voltage_v':'V',
       **{k:'mm' for k in ('diameter_mm','length_mm','shaft_diameter_mm','width_mm','bore_mm','outer_diameter_mm','sensing_distance_mm')}}


def fact(key,value,rng,role,*,unitless=False):
    numeric=isinstance(value,(float,int))
    n=words(value) if numeric and rng.random()<.15 else f'{value:g}' if numeric else value
    if key=='phases':return rng.choice((f'{n} phase',f'{n} ph','single-phase' if value==1 else 'three-phase'))
    if key=='gear_ratio':return rng.choice((f'ratio {n}:1',f'{n}:1 reduction',f'gear reduction {n} to one'))
    if key=='belt_profile':return rng.choice((f'profile {value}',f'{value} belt profile',f'section {value}'))
    if key=='output_type':return rng.choice((f'{value} output',f'output type {value}',f'{value} switching output'))
    name=rng.choice(TERMS[key]);unit='' if unitless else UNITS[key]
    if unit and rng.random()<.15:unit={'kW':'kilowatts','V':'volts','mm':'millimetres'}[unit]
    styles={
      'train':(f'{name} {n} {unit}',f'{n} {unit} {name}',f'{name}={n}{unit}'),
      'development':(f'{name} is {n} {unit}',f'{name}: {n} {unit}'),
      'calibration':(f'{name} specified as {n} {unit}',f'{name} / {n} {unit}'),
      'threshold_selection':(f'{name} to be {n} {unit}',f'{name} requirement {n} {unit}'),
      'prototype_evaluation':(f'{name} of {n} {unit}',f'{name} measured at {n} {unit}',f'{name} rated {n} {unit}')}
    return rng.choice(styles[role]).strip()


def parts(specs,rng,role,*,unitless=False):
    keys=list(specs);rng.shuffle(keys)
    return rng.choice((', ','; ',' / ')).join(fact(k,specs[k],rng,role,unitless=unitless) for k in keys)


def overall(case,config):
    labels=base.intended_attribute_labels(case,config)
    state=case['intended_facts']
    if (state['reference_substitution_allowed'] and len(state['reference_components'])>1
            and state['selected_reference_component'] is None and 'conflicting' not in labels.values()):
        return labels,'unclear'
    return labels,base.compose(list(labels.values()))


def state_family(config,group,mode,number,seed,role):
    source={**config,'references':{}}
    equivalent={'complete':'omit','unitless':'complete','reference_background':'reference_without_same',
                'reference_incomplete':'reference_only','missing_current_pair':'omit',
                'reference_common_candidates':'reference_ambiguous'}.get(mode,mode)
    cases=base.scenario_family(source,group,equivalent,number,seed,role)
    if group in META:return cases
    rng=random.Random(f'broad-state:{seed}:{number}')
    required=PRODUCTS[group][1]
    negative_variant=rng.randrange(2)
    for c in cases:
        c.pop('policy_id',None)
        s=c['intended_facts']
        if mode in ('unitless','reference_background','reference_common_candidates','reference_incomplete','missing_current_pair'):s['blueprint']=mode
        # Short, shuffled component identifiers avoid spending tokens on hashes.
        names=rng.sample(('A1','B2','C3','D4','E5','F6'),len(s['reference_components']))
        ids={r['component_id']:names[i] for i,r in enumerate(s['reference_components'])}
        for r in s['reference_components']:r['component_id']=ids[r['component_id']]
        if s['selected_reference_component'] is not None:s['selected_reference_component']=ids[s['selected_reference_component']]
        if c['reference_fixture']:c['reference_fixture']=copy.deepcopy(s['reference_components'])
        if mode=='unitless' and c['variant']==negative_variant:
            s['ambiguous_current_values']={k:[v] for k,v in s['current_specifications'].items() if k in UNITS}
        if mode=='missing_current_pair':
            s['current_specifications']=dict(s['base_specifications'])
            s['current_specifications'].pop(required[c['variant']%len(required)])
        if mode=='reference_incomplete':
            s['current_specifications']={}
            if c['variant']==negative_variant:
                for ref in s['reference_components']:ref.pop(s['target_attribute'])
            c['reference_fixture']=copy.deepcopy(s['reference_components'])
        if mode=='reference_background':
            s['reference_substitution_allowed']=False;s['selected_reference_component']=None
            current=dict(s['base_specifications'])
            if c['variant']==negative_variant:current.pop(s['target_attribute'])
            s['current_specifications']=current
            for ref in s['reference_components']:
                for key in required:ref[key]=base.alternative_value(key,ref[key],rng)
            c['reference_fixture']=copy.deepcopy(s['reference_components'])
        if mode=='reference_common_candidates':
            first=s['reference_components'][0]
            for i,ref in enumerate(s['reference_components']):
                ref.update({k:first[k] for k in required});ref['mounting_variant']='X' if i==0 else 'Y'
            s['base_specifications']={k:first[k] for k in required}
            s['current_specifications']={};s['reference_substitution_allowed']=True
            s['selected_reference_component']=None if c['variant']==negative_variant else first['component_id']
            s['ambiguous_current_values']={};c['reference_fixture']=copy.deepcopy(s['reference_components'])
        labels,identification=overall(c,config)
        c['semantic_labels'].update(attributes=labels,identification=identification)
    return cases


def product_body(case,rng,role):
    s=case['intended_facts'];mode=s['blueprint'];current=s['current_specifications'];fragments=[]
    unitless=mode=='unitless' and bool(s['ambiguous_current_values'])
    if s['reference_components']:
        if s['reference_substitution_allowed']:
            phrases={
                'train':('Like-for-like replacement.','The replacement must match the existing part.','Use an identical spare.'),
                'development':('Please match the installed unit exactly.','Same specification as the installed component.'),
                'calibration':('An identical replacement to the existing component is wanted.',),
                'threshold_selection':('The intended purchase is an exact replacement of the existing unit.',),
                'prototype_evaluation':('No specification change: match the existing component.','The existing unit should be replaced with its exact equivalent.')}
            fragments.append(rng.choice(phrases[role]))
        else:fragments.append(rng.choice(('The attached record is equipment history, not a requirement to match it.',
                                          'The old record is only background. This is a new specification.',
                                          'A historical reference is attached; do not copy its ratings into this purchase.')))
        refs=s['reference_components'];selected=s['selected_reference_component']
        if len(refs)>1 and selected is None:
            a,b=(r['component_id'] for r in refs)
            phrases={
                'train':(f'{a} and {b} are listed; which one is needed has not been confirmed.',
                         f'I cannot choose between {a} and {b} from these records.'),
                'development':(f'We still need someone to decide whether {a} or {b} is the correct unit.',),
                'calibration':(f'The selection of {a} versus {b} remains open.',),
                'threshold_selection':(f'No component has been selected from {a} and {b}.',),
                'prototype_evaluation':(f'The records name {a} and {b}, with no indication which to order.',)}
            fragments.append(rng.choice(phrases[role]))
        elif len(refs)>1 and (not current or mode=='reference_common_candidates'):
            chosen=next(r for r in refs if r['component_id']==selected)
            other=next(r for r in refs if r['component_id']!=selected)
            if mode=='reference_selected' and mode!='reference_common_candidates' and rng.random()<.65:
                key=s['target_attribute']
                fragments.append(('Choose the version with '+fact(key,chosen[key],rng,role)+'.') if rng.random()<.5
                                 else 'Do not choose the version with '+fact(key,other[key],rng,role)+'.')
            else:fragments.append(rng.choice((f'Select {selected}, not the other candidate.',f'The requested unit is {selected}.')))
    if current:
        text=parts(current,rng,role,unitless=unitless)
        fragments.append(rng.choice(('For the requested part: '+text+'.',text+'.','Purchase details: '+text+'.')))
    if unitless:fragments.append(rng.choice(('The measurement units were not recorded.','I have not confirmed the units of these figures.')))
    if mode=='roller_dimensions':
        f=s['base_specifications']
        if s['ambiguous_current_values']:
            fragments.append(f"Size note: {f['diameter_mm']} x {f['length_mm']} mm; the roles of those numbers are not confirmed.")
        else:fragments.append(f"Roller size Ø{f['diameter_mm']} × {f['length_mm']} mm.")
    for key,values in s['contradictory_current_values'].items():
        fragments.append(rng.choice(('Another current instruction requires ','The same purchase also calls for ','A second active note specifies '))+fact(key,values[1],rng,role)+rng.choice(('.', '. Both instructions are still current.')))
    if mode=='text_conflict' and not s['contradictory_current_values']:
        key=s['target_attribute'];old=base.alternative_value(key,s['base_specifications'][key],rng)
        fragments.insert(0,'A withdrawn earlier note stated '+fact(key,old,rng,role)+'. It has been superseded by the current requirement.')
    if mode=='ambiguous_value':
        for key,values in s['ambiguous_current_values'].items():
            fragments.append(f'The choice is {fact(key,values[0],rng,role)} or {fact(key,values[1],rng,role)}; we have not decided which.')
    if s['unsupported_units']:fragments.append(f"The only power figure on hand is {s['unsupported_units']['power_hp']} hp.")
    if mode=='negated':
        key=s['target_attribute'];alternate=base.alternative_value(key,s['base_specifications'][key],rng)
        fragments.append('Do not order the '+fact(key,alternate,rng,role)+' alternative.')
    if s['ignored_specifications']:
        text=parts(s['ignored_specifications'],rng,role)
        note=(rng.choice((f'Obsolete old-unit record: {text}.',f'The previous design had {text}; it is superseded.'))
              if mode.startswith('obsolete') else rng.choice((f'Unrelated equipment elsewhere uses {text}.',f'A different part has {text}; those are not this part\'s requirements.')))
        position=rng.randrange(len(fragments)+1);fragments.insert(position,note);s['background_position']=position
    if not fragments:fragments.append(rng.choice(('That is all the information currently available.','No other identifying details are in the note.')))
    return ' '.join(fragments)


def scope_body(group,rng,role):
    a,b=rng.sample(list(PRODUCTS),2);first=rng.choice(PRODUCTS[a][2]);second=rng.choice(PRODUCTS[b][2])
    extras=rng.choice(('a separate cable','a mounting bracket','installation labor','a pneumatic cylinder'))
    styles={
      'insufficient_evidence':(
        f'We need either {first} or {second}, but the part has not been identified.',
        'A replacement for the failed component; nobody can tell me what kind it is yet.',
        f'The technician has not decided whether the item is {first} or {second}.',
        'The missing photo was the only identification of the requested component.',
        f'Part selection pending: {first} / {second}. One will be needed, not both.'),
      'mixed_scope':(
        f'Please order {first} and {second}; both are purchases.',
        f'Buy {first}, and also order {extras} separately on this line.',
        f'The order includes {first} plus {extras}.',
        f'Two things are needed together: {first} and {second}.',
        f'Arrange supply of {first} along with fitting services.'),
      'no_suitable_group':(
        'A pneumatic cylinder is the item to buy, not the motor it operates beside.',
        'Hydraulic pump replacement only; the drive motor is staying in place.',
        'Service-only motor inspection. Do not supply replacement goods.',
        'A new cable for the proximity switch, cable only; keep the sensor.',
        'Lubrication service for the conveyor, without replacement parts.',
        'Office printer paper for the workshop.',
        'A control valve, not a gearbox or an electric motor.')}
    if role=='train':
        more={
          'insufficient_evidence':(
            f'We do not know whether to request {first} or {second}; someone must identify it.',
            'An unidentified spare is wanted. The drawing and description are both missing.',
            f'Could be {first}, could be {second}. There is no confirmed item yet.',
            'Replacement type VX230 required; the kind of part is not known.'),
          'mixed_scope':(
            f'Supply {first} as well as {second}.',
            f'{first} and {extras} are both to be bought.',
            f'This line orders {first} plus another item, {second}.',
            f'Goods and labor required together: {first}, then installation service.'),
          'no_suitable_group':(
            'Buy a hydraulic pump; leave the attached electric motor alone.',
            'Order a cable for the inductive sensor, without buying a sensor.',
            'A pneumatic control valve is the only requested good.',
            'Conveyor motor inspection labor only; no replacement components.',
            'Workshop printer paper is the item to procure.',
            'A mounting bracket for the drive is needed; keep the gearbox.',
            'Service the proximity switch wiring. This is labor, not a parts order.')}
        return rng.choice((styles[group][0],*more[group]))
    # The held-out scope banks use distinct inner wording, not just a new prefix.
    index={'train':0,'development':1,'calibration':2,'threshold_selection':3,'prototype_evaluation':4}[role]
    bank=styles[group]
    return bank[index]


def family(config,group,mode,number,seed,role):
    cases=state_family(config,group,mode,number,seed,role)
    for c in cases:
        rng=random.Random(f'broad-writing:{seed}:{number}:{c["variant"]}')
        item=(rng.choice(PRODUCTS[group][2]) if group in PRODUCTS else 'the requested purchase')
        if group in PRODUCTS and (mode=='bare_type' or rng.random()<.25):item+=' type '+rng.choice(('230','400',f'VX-{rng.randrange(100,9999)}'))
        context=rng.choice(CONTEXTS)+' '+str(rng.randrange(1,90))
        body=product_body(c,rng,role) if group in PRODUCTS else scope_body(group,rng,role)
        frame_index=rng.randrange(len(FRAMES[role]));text=FRAMES[role][frame_index].format(item=item,context=context,body=body)
        if group in PRODUCTS and rng.random()<.2:
            other=rng.choice([g for g in PRODUCTS if g!=group]);noun=rng.choice(PRODUCTS[other][2])
            text+=f' The existing {noun} is not being purchased.'
        if group in PRODUCTS and rng.random()<.15:
            text+=rng.choice((' Our own crew will fit it; installation is not being ordered.',
                              'Only this part is to be bought, not a cable or bracket.'))
        c['request'].update(description=text,quantity=rng.choice((1,2,3,4,6)),plant=rng.choice(config['plants']))
        c['writing_template_id']=f'broad-narrative:{role}:{frame_index}'
        c['author']='assistant_facts_first_broadening';c['tags']=['synthetic','crossed_evidence','slice:'+mode]
        c['expected']=expected_route(c,config)
    return cases


def generate(directory,*,seed=251105,counts=None):
    directory=Path(directory)
    if directory.exists():raise ValueError('Use a new corpus directory')
    counts=dict(COUNTS if counts is None else counts)
    if set(counts)!=set(base.SPLITS) or sum(counts.values())!=5*counts['prototype_evaluation']:
        raise ValueError('Exactly 20 percent of new scenario families must be reserved evaluation')
    with zipfile.ZipFile(ROOT/'data/prism-ai-training-complement-2026-11-05.zip') as z:
        native='ai_training_complement_2026_11_05/native/'
        original=[json.loads(x) for x in z.read(native+'train.jsonl').decode().splitlines()]
        config=json.loads(z.read(native+'pilot_config.json'))
    directory.mkdir(parents=True);(directory/'pilot_config.json').write_text(json.dumps(config,indent=2)+'\n')
    total=sum(counts.values());numbers=list(range(total));random.Random(seed).shuffle(numbers);cursor=0
    groups=(*PRODUCTS,*PRODUCTS,*META)
    manifest={'version':base.VERSION,'scope':'synthetic_prototype','seed':seed,'prospective_families':total,
              'split_unit':'scenario_family','new_model_development_fraction':.8,'new_evaluation_fraction':.2,
              'original_supplied_training_cases':400,'original_archive_sha256':file_sha256(ROOT/'data/prism-ai-training-complement-2026-11-05.zip'),
              'source_generator_sha256':file_sha256(__file__),'state_constructor_sha256':file_sha256(base.__file__),
              'exposed_author100_used_for_generation':False,'independent_human_authorship':False,
              'authorship_limit':'Assistant-authored synthetic expansion designed after aggregate failure review; no author/generator independence claim.',
              'partitions':{},'files_sha256':{}}
    for role,count in counts.items():
        cases=[]
        for i,number in enumerate(numbers[cursor:cursor+count]):
            group=groups[i%len(groups)];mode=MODES[(i//len(groups))%len(MODES)]
            cases.extend(family(config,group,mode,number,seed,role))
        cursor+=count
        if role=='train':cases.extend(copy.deepcopy(original))
        random.Random(f'broad-shuffle:{seed}:{role}').shuffle(cases)
        path=directory/(role+'.jsonl');path.write_text(''.join(stable_json(c)+'\n' for c in cases))
        manifest['partitions'][role]={'file':path.name,'cases':len(cases),'new_families':count,
                                      'families':len({c['family_id'] for c in cases}),'original_cases':400 if role=='train' else 0,
                                      'group_counts':dict(Counter(c['semantic_labels']['group'] for c in cases)),
                                      'identification_counts':dict(Counter(str(c['semantic_labels']['identification']) for c in cases))}
    manifest['files_sha256']={p.name:file_sha256(p) for p in directory.iterdir() if p.is_file()}
    (directory/'dataset-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest
