"""Facts-first synthetic data with disjoint writing templates and optional IDs."""
from collections import Counter
import copy
import json
from pathlib import Path
import random

from .orchestrator import validate_request
from .provenance import checked_file, file_sha256, verify_files
from .task_data import opaque, stable_json, expected_route, question_examples, words

VERSION = 'task-data-2'
SPLITS = ('train','development','calibration','threshold_selection','prototype_evaluation')
COUNTS = dict(zip(SPLITS,(2400,500,600,900,1500)))
PRODUCTS = {
 'MG_MOTORS': ('motor', ('power_kw','voltage_v','phases'), ('electric motor','replacement motor','conveyor drive motor','AC motor')),
 'MG_CONVEYOR': ('roller', ('diameter_mm','length_mm'), ('conveyor roller','idler roller','replacement roller','mechanical conveyor roller')),
 'MG_GEARBOXES': ('gearbox', ('gear_ratio','shaft_diameter_mm'), ('gearbox','speed reducer','reduction gearbox','gear reducer')),
 'MG_BELTS': ('belt', ('width_mm','length_mm','belt_profile'), ('conveyor belt','replacement belt','drive belt','belt')),
 'MG_BEARINGS': ('bearing', ('bore_mm','outer_diameter_mm','width_mm'), ('bearing','ball bearing','roller bearing','replacement bearing')),
 'MG_SENSORS': ('sensor', ('sensing_distance_mm','supply_voltage_v','output_type'), ('proximity sensor','inductive sensor','proximity switch','inductive proximity sensor')),
}
META = ('insufficient_evidence','mixed_scope','no_suitable_group')
GROUPS = (*PRODUCTS,*META)
DESCRIPTIONS = {
 'MG_MOTORS':'Standalone replacement electric motors, including conveyor drive motors.',
 'MG_CONVEYOR':'Mechanical conveyor or idler rollers; excludes roller bearings.',
 'MG_GEARBOXES':'Gearboxes, gear reducers and speed reducers.',
 'MG_BELTS':'Replacement conveyor or drive belts.',
 'MG_BEARINGS':'Mechanical ball or roller bearings.',
 'MG_SENSORS':'Inductive proximity sensors and proximity switches.',
 'insufficient_evidence':'Requested item or choice of item remains unresolved.',
 'mixed_scope':'Multiple kinds of goods or goods plus services requested together.',
 'no_suitable_group':'Identifiable purchase outside the supplied categories, including service-only requests.'}
TEMPLATES = {
 'train': ('{item}. {spec}', 'Need {item}. {spec}', 'Please supply {item}; {spec}',
           'Order: {item}. {spec}', '{item} required for replacement. {spec}',
           'Replacing a failed {item}. {spec}', 'Maintenance needs {item}. {spec}', '{spec} — requested item: {item}.'),
 'development': ('Purchase request / {item} / {spec}', 'We are replacing the {item}; details: {spec}',
                 'Item wanted: {item}. Details: {spec}', 'Can you provide {item}? {spec}'),
 'calibration': ('Requested spare: {item}; {spec}', 'Requisition: {item}. Specification: {spec}'),
 'threshold_selection': ('Buying {item}. Technical details: {spec}', '{item} is needed. Available information: {spec}'),
 'prototype_evaluation': ('For the next repair, get {item}. {spec}', 'Please arrange {item} with the following: {spec}',
                          'The part to procure is {item}; {spec}', 'Replacement part entry\n{item}\n{spec}',
                          'Workshop order — {item}; stated details: {spec}')}
SCENARIOS = ('complete','partial','bare_type','same_reference','reference_without_same','ambiguous_reference',
 'selected_reference','reference_conflict','text_conflict','irrelevant','negated_context','wrong_reference_kind')


def config_v2(base):
 c=copy.deepcopy(base);c['contract_version']='pilot-0.4';c['catalog_version']='PILOT-CATALOG-3'
 c['catalog']=DESCRIPTIONS.copy()
 c['identification_rules']=('Assess only required attributes. Use relevant affirmative facts; ignore negated/unrelated values. '
   'An unknown type/model number does not supply specifications or units. Accept written numbers and explicit unit variants. '
   'Like-for-like may use one selected matching-kind reference with required facts; otherwise require explicit specifications. '
   'Resolve candidates by relevant constraints, never position. Conflict overrides unclear, then missing, then satisfied. '
   'Do not convert hp to kW. Bare roller dimensions are ambiguous; diameter and length must be identified.')
 for key,(kind,required,_) in PRODUCTS.items():
  pid=c['default_policies'].get(key,'PILOT-'+kind.upper()+'-1')
  c['default_policies'][key]=pid;c['policies'][pid]={'group':key,'kind':kind,'required':list(required)}
 for group, (kind,required,_) in PRODUCTS.items():
  if group in ('MG_MOTORS','MG_CONVEYOR'):continue
  facts=values(random.Random(kind));facts={k:facts[k] for k in required}
  c['references']['REF-'+kind.upper()+'-EXAMPLE']=[{'component_id':kind.upper()+'-A','kind':kind,**facts}]
 return c


def values(rng):
 return {'power_kw':rng.choice((.75,1.5,2.2,3,4,5.5,7.5,11)), 'voltage_v':rng.choice((110,200,230,400,415,480)),
  'phases':rng.choice((1,3)), 'diameter_mm':rng.choice((25,40,50,60,89)), 'length_mm':rng.choice((200,400,500,750,1000)),
  'gear_ratio':rng.choice((5,10,15,20,30,40)), 'shaft_diameter_mm':rng.choice((20,25,30,35,40)),
  'width_mm':rng.choice((10,15,20,25,30)), 'belt_profile':rng.choice(('A','B','C','SPA','SPB')),
  'bore_mm':rng.choice((10,15,20,25,30)), 'outer_diameter_mm':rng.choice((40,47,52,62,72)),
  'sensing_distance_mm':rng.choice((2,4,5,8,10)), 'supply_voltage_v':rng.choice((12,24,48)), 'output_type':rng.choice(('PNP','NPN'))}


def attribute_text(key,value,rng):
 number=words(value) if isinstance(value,(int,float)) and rng.random()<.08 else str(value)
 return {'power_kw':f'{number} kW', 'voltage_v':f'{number} V', 'phases':('single-phase' if value==1 else rng.choice(('three-phase','3 ph','3~'))),
  'diameter_mm':f'diameter {number} mm', 'length_mm':f'length {number} mm', 'gear_ratio':f'ratio {number}:1',
  'shaft_diameter_mm':f'shaft diameter {number} mm','width_mm':f'width {number} mm','belt_profile':f'profile {value}',
  'bore_mm':f'bore {number} mm', 'outer_diameter_mm':f'outer diameter {number} mm',
  'sensing_distance_mm':f'sensing distance {number} mm', 'supply_voltage_v':f'supply {number} V','output_type':f'{value} output'}[key]


def different(value):
 if isinstance(value,str):return 'NPN' if value=='PNP' else 'PNP' if value=='NPN' else 'Z'
 return 1 if value==3 else 3 if value==1 else value+7


def family(config,group,mode,index,seed,split):
 rng=random.Random(f'{seed}:{index}');facts=values(rng)
 kind,required,names=PRODUCTS.get(group,('motor',('power_kw','voltage_v','phases'),PRODUCTS['MG_MOTORS'][2]))
 component={'component_id':'PART-'+opaque(f'{seed}:{index}')[:7],'kind':kind,**{k:facts[k] for k in required}}
 static_keys=[k for k,v in config['references'].items() if len(v)==1 and v[0]['kind']==kind and not config.get('reference_integrity',{}).get(k)]
 static=rng.choice(static_keys) if static_keys and rng.random()<.3 else None
 if static:
  component=copy.deepcopy(config['references'][static][0]);facts.update({k:component[k] for k in required})
 ref_id=static or 'REF-'+opaque(f'ref:{seed}:{index}')
 changed=rng.choice(required);other=different(component[changed]);omitted=set(rng.sample(list(required),rng.randint(1,len(required))))
 result=[]
 templates=TEMPLATES[split]
 for variant in (0,1):
  local=random.Random(f'{seed}:{index}:{variant}');name=local.choice(names)
  type_code=local.choice(('230','400','ZX-'+str(local.randint(100,9999)),'M'+str(local.randint(10,9999)),str(local.randint(100,99999))))
  item=name+(' type '+type_code if mode=='bare_type' or local.random()<.45 else '')
  pairs=[attribute_text(k,facts[k],local) for k in required];local.shuffle(pairs)
  full=', '.join(pairs);spec=full;ident='satisfied';components=[];reference=None
  if mode in ('partial','bare_type'):
   missing=set(required) if mode=='bare_type' else omitted
   spec=', '.join(attribute_text(k,facts[k],local) for k in required if k not in missing);ident='missing'
  elif mode in ('same_reference','reference_without_same','ambiguous_reference','selected_reference','reference_conflict','wrong_reference_kind'):
   reference=ref_id;components=[copy.deepcopy(component)]
   spec=local.choice(('Same model as the existing part.','Like-for-like replacement.','Replace with an identical unit.'))
   if mode=='reference_without_same':spec='Model and remaining specifications have not been confirmed.';ident='missing'
   if mode in ('ambiguous_reference','selected_reference'):
    components.append({**component,'component_id':component['component_id']+'-B',changed:other});local.shuffle(components)
    if mode=='ambiguous_reference':ident='unclear'
    else:spec+=' Choose the '+attribute_text(changed,component[changed],local)+' version.'
   if mode=='reference_conflict':spec+=' Required '+attribute_text(changed,other,local)+'.';ident='conflicting'
   if mode=='wrong_reference_kind':
    other_group=local.choice([g for g in PRODUCTS if g!=group]);other_kind,other_req,_=PRODUCTS[other_group]
    components=[{'component_id':'OTHER-PART','kind':other_kind,**{k:facts[k] for k in other_req}}];ident='conflicting'
  elif mode=='text_conflict':
   spec=full+'. Also required: '+attribute_text(changed,other,local)+'.';ident='conflicting'
  elif mode=='irrelevant':
   spec='The old drawing lists '+attribute_text(changed,other,local)+' for a different part. Requested specifications: '+full+'.'
  template=templates[(index+variant)%len(templates)]
  text=template.format(item=item,spec=spec).strip()
  if mode=='negated_context':
   other_group=local.choice([g for g in PRODUCTS if g!=group]);other_name=local.choice(PRODUCTS[other_group][2])
   text='Do not order a '+other_name+'. '+text
  if local.random()<.15:text+=' For conveyor CV-'+str(local.randint(1,99))+'.'
  if group in META:
   a,b=local.sample(list(PRODUCTS),2);first=local.choice(PRODUCTS[a][2]);second=local.choice(PRODUCTS[b][2])
   ident=None;reference=None;components=[]
   if group=='insufficient_evidence':
    item=local.choice(('an unidentified spare part','a replacement unit of unknown kind',f'either a {first} or a {second}; the choice is not decided'))
   elif group=='mixed_scope':item=local.choice((f'both a {first} and a {second}',f'a {first} plus installation services'))
   else:item=local.choice(('printer paper','an office chair','a replacement hydraulic pump','a control valve','a laptop',
      'motor inspection service only, no replacement goods','conveyor maintenance labor only, no parts'))
   if group=='no_suitable_group' and local.random()<.75:item+=local.choice((' / stock code ',' / request code ',' / vendor type '))+type_code
   text=template.format(item=item,spec='').strip()
  request={'item_id':'V2-'+opaque(f'{seed}:{index}:{variant}'),'description':text,'quantity':local.choice((1,2,3,4,6)),
   'unit':'EA','plant':local.choice(config['plants']),'required_date':f'2026-12-{local.randint(1,28):02d}'}
  if reference:request['reference_id']=reference
  if variant and local.random()<.10:request[local.choice(('quantity','plant','required_date'))]=local.choice((None,))
  case={'case_id':request['item_id'],'family_id':'FAM2-'+opaque(f'{seed}:{index}'),'variant':variant,'primary':variant==0,
   'split':split,'contract_version':config['contract_version'],'author':'facts_first_generator_v2','review_status':'not_independently_reviewed',
   'generator_version':VERSION,'writing_template_id':f'{split}-{(index+variant)%len(templates)}',
   'intended_facts':{'blueprint':mode,'kind':kind,'required_attributes':list(required),'component':component,'changed_attribute':changed},
   'request':request,'reference_fixture':components,'semantic_labels':{'group':group,'identification':ident},
   'tags':['synthetic','blueprint:'+mode,'writing:'+split]}
  case['expected']=expected_route(case,config);result.append(case)
 return result


def generate(directory,base_config,*,counts=None,seed=77231):
 directory=Path(directory)
 if directory.exists():raise ValueError('Use a new corpus directory')
 counts=COUNTS if counts is None else counts
 if set(counts)!=set(SPLITS) or any(type(n) is not int or n<9 for n in counts.values()):raise ValueError('At least nine families per role')
 config=config_v2(json.loads(Path(base_config).read_text()));directory.mkdir(parents=True)
 (directory/'pilot_config.json').write_text(json.dumps(config,indent=2)+'\n')
 manifest={'version':VERSION,'scope':'synthetic_prototype','seed':seed,'generator_sha256':file_sha256(__file__),
  'split_unit':'scenario_family_and_writing_template','variants_per_family':2,'primary_variant':0,'gold_labels_reviewed':False,
  'external_training_data_used':False,'independent_human_challenge_present':False,'outgoing_data_license':'not_selected; local/private',
  'data_origin':'Declared fictional facts, new local synthetic writing templates; no external corpus',
  'partitions':{},'files_sha256':{}}
 seen=set();number=0
 for split,n in counts.items():
  cases=[]
  for index in range(n):
   # Supported categories each get about 13%; each scope outcome about 7%.
   cycle=(*PRODUCTS,*PRODUCTS,*META);group=cycle[index%len(cycle)]
   mode=SCENARIOS[(index//len(cycle)+list(PRODUCTS).index(group) if group in PRODUCTS else index)%len(SCENARIOS)]
   for attempt in range(200):
    generated=family(config,group,mode,number,seed,split);number+=1
    keys=[stable_json({'description':c['request']['description'],'reference':c['reference_fixture'],
      'reference_id':c['request'].get('reference_id')}) for c in generated]
    if len(set(keys))==2 and not (set(keys)&seen):break
   else:raise ValueError(f'Cannot produce a distinct family: {split}, {group}, {mode}, index {index}')
   seen.update(keys);cases.extend(generated)
  random.Random(f'shuffle:{seed}:{split}').shuffle(cases)
  if split=='development':
   root=Path(__file__).resolve().parents[1]
   for source in ('data/user_feedback_cases.jsonl','data/smoke_cases.jsonl'):
    for original in map(json.loads,(root/source).read_text().splitlines()):
     labels=original['expected']
     # Only unchanged supported semantics; newly supported gearbox/belt cases
     # must not retain the old catalog's outside-scope labels.
     if labels['group'] not in ('MG_MOTORS','MG_CONVEYOR') or labels['identification'] is None:continue
     request=copy.deepcopy(original['request']);reference=config['references'].get(request.get('reference_id'),[])
     extra={'case_id':'EXPOSED-'+original['case_id'],'family_id':'EXPOSED-'+original['family_id'],
      'variant':0,'primary':True,'split':split,'contract_version':config['contract_version'],
      'author':original.get('author','assistant_drafted'),'review_status':'not_independently_reviewed','generator_version':VERSION,
      'writing_template_id':'exposed_feedback','intended_facts':{'blueprint':'exposed_feedback'},
      'request':request,'reference_fixture':copy.deepcopy(reference),
      'semantic_labels':{'group':labels['group'],'identification':labels['identification']},'tags':['exposed_development']}
     if original.get('policy_id'):extra['policy_id']=original['policy_id']
     extra['expected']=expected_route(extra,config);cases.append(extra)
  path=directory/(split+'.jsonl');path.write_text(''.join(stable_json(c)+'\n' for c in cases))
  manifest['partitions'][split]={'file':path.name,'families':len({c['family_id'] for c in cases}),'generated_families':n,'cases':len(cases),'question_rows':len(question_examples(cases,config)),
   'group_counts':dict(Counter(c['semantic_labels']['group'] for c in cases)),
   'identification_counts':dict(Counter(c['semantic_labels']['identification'] for c in cases if c['semantic_labels']['identification'])),
   'writing_template_ids':sorted({c['writing_template_id'] for c in cases})}
 manifest['files_sha256']={p.name:file_sha256(p) for p in sorted(directory.iterdir()) if p.is_file()}
 (directory/'dataset-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest


def load_partition(directory,split,*,allowed_splits):
 if split not in SPLITS or split not in allowed_splits:raise ValueError('Partition role not permitted')
 directory=Path(directory);manifest=json.loads(checked_file(directory,'dataset-manifest.json').read_text())
 if manifest['version']!=VERSION:raise ValueError('Wrong dataset version')
 filename=manifest['partitions'][split]['file']
 verify_files(directory,{k:manifest['files_sha256'][k] for k in (filename,'pilot_config.json')})
 config=json.loads((directory/'pilot_config.json').read_text())
 cases=[json.loads(line) for line in checked_file(directory,filename).read_text().splitlines()]
 if len(cases)!=manifest['partitions'][split]['cases']:raise ValueError('Wrong partition count')
 for case in cases:
  validate_request(case['request'])
  if case['split']!=split or case['contract_version']!=config['contract_version']:raise ValueError('Wrong case role/config')
  labels=case['semantic_labels']
  if labels['group'] not in config['catalog'] or labels['identification'] not in (set(config['identification_options']) if labels['group'] in PRODUCTS else {None}):raise ValueError('Invalid labels')
  if case['expected']!=expected_route(case,config):raise ValueError('Label routing mismatch')
 return config,cases,manifest
