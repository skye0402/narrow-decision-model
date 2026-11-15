"""Development-driven training-only augmentation; reserved partitions unchanged."""
from collections import Counter
import copy
import json
from pathlib import Path
import random
import shutil
from .data_v2 import PRODUCTS,values,attribute_text
from .provenance import file_sha256
from .task_data import expected_route,opaque,stable_json,question_examples


def derive(original,destination,seed=99223):
 original,destination=map(Path,(original,destination))
 if destination.exists():raise ValueError('Use a new derived dataset directory')
 destination.mkdir(parents=True)
 for p in original.iterdir():
  if p.is_file():shutil.copyfile(p,destination/p.name)
 config=json.loads((destination/'pilot_config.json').read_text())
 manifest=json.loads((destination/'dataset-manifest.json').read_text())
 excluded={case['request']['description'].casefold().strip() for case in map(json.loads,(original/'development.jsonl').read_text().splitlines())}
 existing=[json.loads(line) for line in (destination/'train.jsonl').read_text().splitlines()]
 seen={c['request']['description'].casefold().strip() for c in existing}
 additions=[]
 templates=(
  '{item} type {code} for {context} replacement',
  'Need {item}, model {code}, to replace the failed part on the {context}.',
  'For the {context}, the spare needed is a {item} ({code}).',
  '{item} replacement / {context} / type {code}',
  'Order {quantity} pieces: {item} {code}; use on {context}.',
  'Maintenance spare request: {item}, {code}. Equipment: {context}.',
  'The {context} needs a new {item}; part designation {code}.',
  '{item} {code} required — replacing a worn part in {context}.')
 contexts=('conveyor actor','conveyor actuator','conveyor drive','conveyor line','packaging machine','motor assembly',
  'roller conveyor','belt conveyor','production line','sorting station','conveyor system','drive station')
 for group,(kind,required,names) in PRODUCTS.items():
  for family in range(200):
   rng=random.Random(f'{seed}:{group}:{family}');facts=values(rng);complete=family%2==0
   code=rng.choice(('230','400','ZX-'+str(rng.randint(100,9999)),'M'+str(rng.randint(100,9999)),str(rng.randint(100,99999))))
   for variant in (0,1):
    for attempt in range(100):
     item=rng.choice((*names,kind,kind) if kind!='sensor' else names);context=rng.choice(contexts);quantity=rng.choice((1,2,3,4,6))
     template=rng.choice(templates);text=template.format(item=item,code=code,context=context,quantity=quantity)
     if complete:
      specifications=[attribute_text(k,facts[k],rng) for k in required];rng.shuffle(specifications)
      text+='; '+', '.join(specifications)
     if rng.random()<.25:text=text[0].upper()+text[1:]
     key=text.casefold().strip()
     if key not in seen and key not in excluded:break
    else:raise ValueError('Cannot produce distinct natural request')
    seen.add(key)
    request={'item_id':'NATURAL-'+opaque(f'{seed}:{group}:{family}:{variant}'),'description':text,
     'quantity':quantity,'unit':'EA','plant':rng.choice(config['plants']),'required_date':f'2026-12-{rng.randint(1,28):02d}'}
    case={'case_id':request['item_id'],'family_id':'NATURAL-FAM-'+opaque(f'{seed}:{group}:{family}'),
     'variant':variant,'primary':variant==0,'split':'train','contract_version':config['contract_version'],
     'author':'facts_first_natural_request_augmentation','review_status':'not_independently_reviewed',
     'generator_version':'task-data-2b','writing_template_id':'natural-train-'+str(templates.index(template)),
     'intended_facts':{'blueprint':'natural_short_context','kind':kind,'required_attributes':list(required),
      'supplied_attributes':{k:facts[k] for k in required} if complete else {},'opaque_type_code':code,'context':context},
     'request':request,'reference_fixture':[],'semantic_labels':{'group':group,'identification':'satisfied' if complete else 'missing'},
     'tags':['synthetic','training_only_augmentation','context_not_requested_product']}
    case['expected']=expected_route(case,config);additions.append(case)
 all_cases=existing+additions;random.Random(seed).shuffle(all_cases)
 (destination/'train.jsonl').write_text(''.join(stable_json(c)+'\n' for c in all_cases))
 manifest.update(generator_path='prism_model/data_v2b.py',generator_sha256=file_sha256(__file__),
   parent_v2_dataset_manifest_sha256=file_sha256(original/'dataset-manifest.json'),
   augmentation={'seed':seed,'added_training_families':1200,'added_training_cases':len(additions),
    'development_and_reserved_partition_bytes_unchanged':True,'exact_development_descriptions_excluded':True,
    'reason':'First candidate fails the exposed motor-with-conveyor-context example; final evaluation not opened.'})
 partition=manifest['partitions']['train'];partition.update(families=len({c['family_id'] for c in all_cases}),cases=len(all_cases),
  question_rows=len(question_examples(all_cases,config)),group_counts=dict(Counter(c['semantic_labels']['group'] for c in all_cases)),
  identification_counts=dict(Counter(c['semantic_labels']['identification'] for c in all_cases if c['semantic_labels']['identification'])),
  writing_template_ids=sorted({c['writing_template_id'] for c in all_cases}))
 manifest['files_sha256']['train.jsonl']=file_sha256(destination/'train.jsonl')
 for role in ('development','calibration','threshold_selection','prototype_evaluation'):
  if file_sha256(destination/(role+'.jsonl'))!=file_sha256(original/(role+'.jsonl')):raise ValueError('Non-training data changed')
 (destination/'dataset-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest
