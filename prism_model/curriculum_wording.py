"""Separate narrative/scope wording for the relational corpus's five roles."""
import random
import re
FRAMES={
 'train':('Get {item} for {context}. {body}','Order request from {context}: {item}. {body}',
          '{item} required by the crew at {context}. {body}','Procurement of {item} for {context}: {body}',
          'Please obtain {item}; job location {context}. {body}',
          'Spare: {item} / {body} ({context})','{item}; {body} For {context}.'),
 'development':('Material wanted at {context}: {item}. {body}',
                'Buying request for {item} at {context}. {body}',
                '{context} needs us to purchase {item}. {body}',
                '{item} wanted / {context} / {body}'),
 'calibration':('For the work order at {context}, the item to obtain is {item}. {body}',
                'Purchase entry from {context}. Required item: {item}. {body}',
                'Please arrange {item} to be supplied to {context}. {body}',
                'Item {item} - job {context} - {body}'),
 'threshold_selection':('We have a requisition for {item}, for use at {context}. {body}',
                        'At {context}, purchasing is asked to source {item}. {body}',
                        'The current buying line concerns {item} for {context}. {body}',
                        '{context}: purchase {item}; {body}'),
 'prototype_evaluation':('The repair team at {context} has requested {item}. {body}',
                         '{item} is being sourced for the work at {context}. {body}',
                         'Send purchasing this requirement from {context}: {item}. {body}',
                         'For {context}, we are looking to buy {item}. {body}',
                         '{item} / repair job {context} / {body}')}


def technical_specs(specs,rng):
    """Unambiguous industrial shorthand from intended facts, not an extractor."""
    fragments=[];keys=list(specs);rng.shuffle(keys)
    for key in keys:
        v=specs[key];n=f'{v:g}' if isinstance(v,(int,float)) else v
        fragments.append({'power_kw':f'{n}kW','voltage_v':f'{n}V','phases':f'{n}ph',
            'diameter_mm':f'roller dia {n}mm','length_mm':f'L={n}mm',
            'gear_ratio':f'ratio {n}:1','shaft_diameter_mm':f'shaft dia {n}mm',
            'width_mm':f'W={n}mm','belt_profile':f'profile {n}',
            'bore_mm':f'ID {n}mm','outer_diameter_mm':f'OD {n}mm',
            'sensing_distance_mm':f'range {n}mm','supply_voltage_v':f'{n}V supply','output_type':str(n)}[key])
    return rng.choice((' / ',', ','; ')).join(fragments)


def vary_units(text,seed,number):
    """Label-preserving spelling variations shared across a family's variants."""
    rng=random.Random(f'curriculum-units:{seed}:{number}')
    if rng.random()<.4:
        text=re.sub(r'\b(?:3 phase|3 ph|three-phase)\b','3~',text)
        text=re.sub(r'\b(?:1 phase|1 ph|single-phase)\b','1~',text)
    if rng.random()<.4:text=re.sub(r'(\d) (kW|V|mm)\b',r'\1\2',text)
    if rng.random()<.25:text=re.sub(r'\bkW\b','kW'.lower(),text)
    return text

SCOPE={
 'train':{
  'insufficient_evidence':('A spare is needed, but there is no clear identification of it.',
    'Someone wrote type VX230; nobody knows which kind of part that means.',
    'We have a choice between {a} and {b}. The choice is still open.',
    'Ask maintenance whether we want {a} or {b}; buy one after they decide.'),
  'mixed_scope':('This purchase line covers {a} plus {b}, both to be supplied.',
    'Include {a} and a separate mounting bracket in the purchase.',
    'Purchase {a} together with paid installation work.',
    'Two ordered items: {a}; then {b}. Neither is merely background.'),
  'no_suitable_group':('Only a hydraulic cylinder is to be purchased; retain the electric motor.',
    'Procure a cable gland for the sensor wiring, not the sensor itself.',
    'We are buying a cleaning service for the conveyor, without any goods.',
    'A replacement PLC input card is the only required item.')},
 'development':{
  'insufficient_evidence':('The breakdown note does not say which component to purchase.',
    'The material could be {a} or {b}; we cannot identify the right one.',
    'The spare-part description is lost. There is no known purchase category.',
    'Please resolve the alternative {a} versus {b} before ordering one.'),
  'mixed_scope':('Supply {a}, plus purchase {b} as another item.',
    'The requisition requests {a} and a new cable as two purchases.',
    'We want {a} supplied and an outside contractor to install it.',
    'Both {a} and {b} are needed from purchasing on this line.'),
  'no_suitable_group':('The requested goods are a pneumatic valve; the nearby motor stays.',
    'Buy the connector for the proximity switch, with no new switch.',
    'Contract a technician to inspect the drive; do not order parts.',
    'Purchase replacement ink cartridges for the workshop printer.')},
 'calibration':{
  'insufficient_evidence':('No one has established what sort of replacement we need.',
    'Our choice of {a} versus {b} is undecided. Only one will be bought.',
    'The note says replacement unit, without identifying the unit.',
    'We cannot yet tell whether the requested spare is {a} or {b}.'),
  'mixed_scope':('This line purchases {a} as well as {b}.',
    'Arrange supply of {a} and of a mounting plate for it.',
    'Buy {a} and order fitting labor from the supplier.',
    'Purchasing should deliver two different items, {a} and {b}.'),
  'no_suitable_group':('The order is solely for a hydraulic pump; no electric motor is wanted.',
    'We need the sensor extension lead only. Keep the installed sensor.',
    'Order an electrical testing service, with no replacement equipment.',
    'A carton of office paper is the only workshop purchase.')},
 'threshold_selection':{
  'insufficient_evidence':('We lack the information needed to name the failed part being bought.',
    'Either {a} or {b} will be required; confirmation is pending.',
    'An unspecified item needs replacing. Its identity has not been supplied.',
    'There is one purchase, but we still have to choose {a} or {b}.'),
  'mixed_scope':('Please source {a} and source {b}; both are actual purchases.',
    'Supply {a} along with an ordered fastening kit.',
    'The order contains {a} and a separate installation service.',
    'Create one line covering two goods: {a} and {b}.'),
  'no_suitable_group':('Buy a pressure regulator, keeping the existing motor and gearbox.',
    'The mounting rail for the switch is wanted, without a switch.',
    'Purchase annual conveyor inspection labor only, with no spares.',
    'Order safety labels for the workshop; no machine components are needed.')},
 'prototype_evaluation':{
  'insufficient_evidence':('The requester has not identified the item that should be bought.',
    'There is a pending decision between {a} and {b}; we need one of them.',
    'A failed piece of equipment needs a spare, but its type is unknown.',
    'We have not determined whether purchasing should get {a} or {b}.'),
  'mixed_scope':('We are ordering both {a} and {b} as distinct items.',
    'This is a purchase of {a} plus a separate power cable.',
    'Please purchase {a} including a separately ordered fitting service.',
    'Two supplies are requested together, namely {a} and {b}.'),
  'no_suitable_group':('A hydraulic hose is to be ordered; the drive motor is not being replaced.',
    'Get the protective cover for the sensor, not a replacement sensor.',
    'The purchase is a machinery cleaning contract, not replacement goods.',
    'The only order is a set of printer toner cartridges.')},
}
