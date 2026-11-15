"""Fresh role-separated context corpus for full encoder fine-tuning."""
from collections import Counter
import copy
import json
from pathlib import Path
import random
import re
import shutil
from .data_v2 import PRODUCTS,META,SCENARIOS,family
from .provenance import file_sha256
from .task_data import question_examples,expected_route,stable_json

FRAMES={
 'train':('Equipment background: {context}. Requested purchase: {body}',
  '{body} This is for work on {context}; the equipment itself is not being ordered.',
  'The existing {context} is background information. Order note: {body}',
  'Maintenance is working on {context}. This requisition says: {body}',
  'Job context, not an extra purchase: {context}. {body}',
  '{body} Installation context only: {context}.'),
 'calibration':('Requisition wording: {body} For context, this supports {context}.',
  'Site background concerns {context}. The requested order is described as follows: {body}'),
 'threshold_selection':('Parts request — {body} Equipment mentioned only for context: {context}.',
  'Repair context: {context}. The buying request itself reads: {body}'),
 'prototype_evaluation':('Regarding repairs on {context}, here is the actual order note: {body}',
  'Please assess this purchasing line: {body} The {context} explains where it will be used.',
  '{body} For avoidance of ambiguity, {context} is the installed equipment rather than another requested item.',
  'We are servicing {context}; only the following request is for procurement: {body}',
  'Order text: {body}\nBackground equipment, outside the requested scope: {context}.')}


def paraphrase(text,rng):
 if rng.random()<.6:text=re.sub(r'(?<![\w.])(\d+(?:\.\d+)?)\s*kW\b',r'rated power \1 kW',text)
 if rng.random()<.5:text=re.sub(r'(?<![\w.])(\d+(?:\.\d+)?)\s*V\b',r'voltage \1 V',text)
 if rng.random()<.4:text=text.replace('three-phase','3-phase').replace('single-phase','single phase')
 if rng.random()<.4:text=re.sub(r'(?<!outer )(?<!shaft )diameter (\d+) mm',r'\1 mm diameter',text)
 if rng.random()<.4:text=re.sub(r'length (\d+) mm',r'\1 mm long',text)
 if rng.random()<.4:text=re.sub(r'\b(PNP|NPN) output',r'output type \1',text)
 return text


def generate(original,destination):
 original,destination=map(Path,(original,destination))
 if destination.exists():raise ValueError('Use a new full-finetuning dataset directory')
 destination.mkdir(parents=True);config=json.loads((original/'pilot_config.json').read_text())
 (destination/'pilot_config.json').write_text(json.dumps(config,indent=2)+'\n')
 train=[json.loads(line) for line in (original/'train.jsonl').read_text().splitlines()]
 development=[json.loads(line) for line in (original/'development.jsonl').read_text().splitlines()]
 challenge=original.parent/'natural-challenge/cases.jsonl'
 for case in map(json.loads,challenge.read_text().splitlines()):
  case=copy.deepcopy(case);case['split']='development';case['writing_template_id']='exposed_v2_natural_context';development.append(case)
 api=json.loads(Path('research/task-v2-api-results.json').read_text())
 for row in api['results']:
  if not row['case_id'].startswith('complete_'):continue
  key='CRITICAL-'+row['case_id'];labels={'group':row['expected']['group'],'identification':'satisfied'}
  case={'case_id':key,'family_id':key,'primary':True,'variant':0,'split':'development','contract_version':config['contract_version'],
   'author':'exposed_api_development','review_status':'not_independently_reviewed','writing_template_id':'exposed_api_complete',
   'intended_facts':{'blueprint':'exposed_complete_request'},'request':row['request'],'reference_fixture':[],'semantic_labels':labels}
  case['expected']=expected_route(case,config);development.append(case)
 groups=(*PRODUCTS,*META);partitions={'train':train,'development':development};number=200000
 contexts=('drive motor assembly','roller conveyor','belt conveyor','gearbox assembly','bearing housing','sensor installation')
 for role,n in [('train',1350),('calibration',600),('threshold_selection',900),('prototype_evaluation',1500)]:
  cases=[]
  for index in range(n):
   group=groups[index%len(groups)];mode=SCENARIOS[(index//len(groups)+index%len(groups))%len(SCENARIOS)]
   generated=family(config,group,mode,number,104729,'train');number+=1
   for variant,case in enumerate(generated):
    rng=random.Random(f'context:104729:{role}:{index}:{variant}')
    body=paraphrase(case['request']['description'],rng);context=rng.choice(contexts)
    frame=FRAMES[role][(index+variant)%len(FRAMES[role])]
    case['request']['description']=frame.format(body=body,context=context)
    case['case_id']='C3-'+case['case_id'];case['family_id']='C3-'+case['family_id']
    case.update(split=role,author='facts_first_context_generator_v3',writing_template_id=f'v3-{role}-{(index+variant)%len(FRAMES[role])}')
    case['intended_facts']['background_equipment']=context;case['expected']=expected_route(case,config)
    cases.append(case)
  if role=='train':partitions['train'].extend(cases)
  else:partitions[role]=cases
 m={'version':'task-data-2','generator_variant':'full_encoder_context_3','scope':'synthetic_prototype',
  'generator_path':'prism_model/data_v3.py','generator_sha256':file_sha256(__file__),
  'parent_dataset_manifest_sha256':file_sha256(original/'dataset-manifest.json'),'seed':104729,'gold_labels_reviewed':False,
  'data_origin':'Local facts-first synthesis; former v0.2 natural test and API examples explicitly promoted to exposed development.',
  'independent_human_challenge_present':False,'outgoing_data_license':'not_selected; local/private',
  'split_unit':'scenario_family_and_outer_context_writing_template','variants_per_family':2,'primary_variant':0,
  'partitions':{},'files_sha256':{}}
 seen=set()
 for role,cases in partitions.items():
  ids={c['family_id'] for c in cases}
  if ids&seen:raise ValueError('Family overlap')
  seen|=ids;random.Random(f'shuffle:3:{role}').shuffle(cases)
  path=destination/(role+'.jsonl');path.write_text(''.join(stable_json(c)+'\n' for c in cases))
  m['partitions'][role]={'file':path.name,'cases':len(cases),'families':len(ids),'question_rows':len(question_examples(cases,config)),
   'group_counts':dict(Counter(c['semantic_labels']['group'] for c in cases)),
   'identification_counts':dict(Counter(c['semantic_labels']['identification'] for c in cases if c['semantic_labels']['identification'])),
   'writing_template_ids':sorted({c['writing_template_id'] for c in cases})}
 m['files_sha256']={p.name:file_sha256(p) for p in destination.iterdir() if p.is_file()}
 (destination/'dataset-manifest.json').write_text(json.dumps(m,indent=2)+'\n');return m
