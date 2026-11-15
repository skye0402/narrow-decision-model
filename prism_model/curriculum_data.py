"""Combined relational corpus with an exact overall family-level 80/20 split."""
from collections import Counter
import copy
import json
from pathlib import Path
import random

from . import broad_data,reference_curriculum,curriculum_wording
from .data_v2 import PRODUCTS,META
from .evidence_data import load_partition,SPLITS
from .provenance import file_sha256
from .task_data import expected_route,stable_json

ROOT=Path(__file__).resolve().parents[1]
PARENT_MANIFEST_SHA256='48a4746bdb1e024104132d26fedc70678c608620b6a784263b17e612e4a44bd7'
COUNTS=dict(zip(SPLITS,(3900,600,600,900,2725)))
NON_REFERENCE=('omit','bare_type','unitless','text_conflict','obsolete','obsolete_one',
               'unrelated','unrelated_one','negated','ambiguous_value','hp','roller_dimensions','reference_background')


def ordinary_family(config,group,mode,number,seed,role):
    cases=broad_data.family(config,group,mode,number,seed,role)
    for c in cases:
        rng=random.Random(f'curriculum-writing:{seed}:{number}:{c["variant"]}')
        frame=rng.randrange(len(curriculum_wording.FRAMES[role]))
        if group in META:
            a,b=rng.sample(list(PRODUCTS),2)
            names={'a':rng.choice(PRODUCTS[a][2]),'b':rng.choice(PRODUCTS[b][2])}
            body=rng.choice(curriculum_wording.SCOPE[role][group]).format(**names)
            item='the requested purchase'
        else:
            item=rng.choice(PRODUCTS[group][2]);body=broad_data.product_body(c,rng,role)
            if mode=='bare_type' or rng.random()<.25:item+=' type '+rng.choice(('230','400','VX-'+str(rng.randrange(100,9999))))
        text=curriculum_wording.FRAMES[role][frame].format(item=item,body=body,context=f'the repair cell {number}')
        c['request']['description']=curriculum_wording.vary_units(text,seed,number)
        c['writing_template_id']=f'curriculum-narrative:{role}:{frame}'
        c['author']='assistant_relational_curriculum';c['tags']=['synthetic','relational_corpus','slice:'+c['intended_facts']['blueprint']]
        c['expected']=expected_route(c,config)
    return cases


def generate(directory,parent,*,seed=251107):
    directory,parent=map(Path,(directory,parent))
    if directory.exists():raise ValueError('Use a new corpus directory')
    if file_sha256(parent/'dataset-manifest.json')!=PARENT_MANIFEST_SHA256:raise ValueError('Previous corpus changed')
    config,old_train,previous=load_partition(parent,'train',allowed_splits={'train'})
    _,old_development,_=load_partition(parent,'development',allowed_splits={'development'})
    assert len(old_train)==8200 and len(old_development)==1200
    retained_families=len({c['family_id'] for c in old_train+old_development})
    total=retained_families+sum(COUNTS.values())
    if total!=5*COUNTS['prototype_evaluation']:raise ValueError('Overall family split must be exactly 80/20')
    numbers=list(range(sum(COUNTS.values())));random.Random(seed).shuffle(numbers);cursor=0
    directory.mkdir(parents=True)
    (directory/'pilot_config.json').write_bytes((parent/'pilot_config.json').read_bytes())
    manifest={'version':'evidence-data-1','scope':'synthetic_prototype','seed':seed,
        'generator':'relational-curriculum-1','source_generator_sha256':file_sha256(__file__),
        'family_constructor_sha256':file_sha256(reference_curriculum.__file__),
        'wording_source_sha256':file_sha256(curriculum_wording.__file__),
        'previous_training_manifest_sha256':PARENT_MANIFEST_SHA256,
        'original_archive_sha256':previous['original_archive_sha256'],
        'original_supplied_training_cases':400,'retained_gradient_training_cases':len(old_train),
        'retained_development_cases':len(old_development),'retained_families':retained_families,
        'new_families':sum(COUNTS.values()),'total_families':total,
        'split_unit':'scenario_family','model_development_fraction':.8,'evaluation_fraction':.2,
        'exposed_author100_used_for_generation':False,'previous_evaluation_used_for_gradients':False,
        'independent_human_authorship':False,
        'authorship_limit':'Assistant-authored constructed scenarios; familiar mechanisms; no independent author or operational reliability claim.',
        'partitions':{},'files_sha256':{}}
    for role,count in COUNTS.items():
        cases=[];occurrences=Counter()
        for i,number in enumerate(numbers[cursor:cursor+count]):
            group=(*PRODUCTS,*PRODUCTS,*META)[i%15]
            occurrence=occurrences[group];occurrences[group]+=1
            if group in META:rows=ordinary_family(config,group,'scope',number,seed,role)
            else:
                slot=occurrence%8;attributes=PRODUCTS[group][1]
                attribute=attributes[(occurrence//8+slot)%len(attributes)]
                mode='rating_crossing' if slot<4 else 'selection_crossing' if slot==4 else 'availability_crossing' if slot==5 else NON_REFERENCE[(occurrence//8*2+slot-6)%len(NON_REFERENCE)]
                if slot<6:
                    technical=random.Random(f'curriculum-style:{seed}:{number}').random()<.4
                    rows=reference_curriculum.family(config,group,attribute,mode,number,seed,role,frames=curriculum_wording.FRAMES,
                        current_formatter=curriculum_wording.technical_specs if technical else None)
                    for c in rows:
                        c['writing_template_id']='curriculum-narrative:'+role+':'+c['writing_template_id'].rsplit(':',1)[1]
                        c['request']['description']=curriculum_wording.vary_units(c['request']['description'],seed,number)
                else:rows=ordinary_family(config,group,mode,number,seed,role)
            cases.extend(rows)
        cursor+=count
        retained=old_train if role=='train' else old_development if role=='development' else []
        cases.extend(copy.deepcopy(retained));random.Random(f'curriculum-shuffle:{seed}:{role}').shuffle(cases)
        path=directory/(role+'.jsonl');path.write_text(''.join(stable_json(c)+'\n' for c in cases))
        manifest['partitions'][role]={'file':path.name,'cases':len(cases),'families':len({c['family_id'] for c in cases}),
            'new_families':count,'retained_cases':len(retained),
            'group_counts':dict(Counter(c['semantic_labels']['group'] for c in cases)),
            'identification_counts':dict(Counter(str(c['semantic_labels']['identification']) for c in cases)),
            'blueprint_counts':dict(Counter(c['intended_facts'].get('blueprint','unannotated') for c in cases))}
    manifest['files_sha256']={p.name:file_sha256(p) for p in directory.iterdir() if p.is_file()}
    (directory/'dataset-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
