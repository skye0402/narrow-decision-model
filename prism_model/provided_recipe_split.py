"""Recipe-held-out internal validation using only the original 400 records.

No new request text, annotation changes, test examples, or training runs. Each
supported-product writing recipe stays wholly in one development fold across
all categories. The separately authored scope cases are family-stratified.
"""
from collections import Counter,defaultdict
import copy
import hashlib
import json
from pathlib import Path
import random
import zipfile

from .provenance import file_sha256
from .task_data import stable_json

META={'insufficient_evidence','mixed_scope','no_suitable_group'}


def membership(cases):
    normalized=sorted(({**c,'split':'train'} for c in cases),key=lambda c:c['case_id'])
    return hashlib.sha256(stable_json(normalized).encode()).hexdigest()


def recipe_folds(cases,*,folds=3,seed=73):
    if folds!=3:raise ValueError('This protocol predeclares three folds')
    if len(cases)!=400 or len({c['family_id'] for c in cases})!=400:raise ValueError('Expected 400 distinct supplied families')
    recipes=defaultdict(list);scope=defaultdict(list)
    for case in cases:
        if case['semantic_labels']['group'] in META:scope[case['semantic_labels']['group']].append(case)
        else:recipes[case['writing_template_id']].append(case)
    by_label=defaultdict(list)
    for name,rows in recipes.items():
        labels={c['semantic_labels']['identification'] for c in rows}
        if len(labels)!=1:raise ValueError('Recipe labels changed; review grouping before use')
        by_label[next(iter(labels))].append(name)
    if set(by_label)!={'satisfied','missing','conflicting','unclear'}:raise ValueError('Missing evidence outcome')
    held=[set() for _ in range(folds)];scope_ids=[set() for _ in range(folds)]
    rng=random.Random(seed)
    for label in sorted(by_label):
        names=sorted(by_label[label]);rng.shuffle(names)
        if len(names)<folds:raise ValueError('Insufficient distinct recipes for each outcome')
        for index,name in enumerate(names):held[index%folds].add(name)
    for group in sorted(scope):
        rows=sorted(scope[group],key=lambda c:c['family_id']);rng.shuffle(rows)
        for index,c in enumerate(rows):scope_ids[index%folds].add(c['case_id'])
    result=[]
    for index in range(folds):
        development=[];train=[]
        for original in cases:
            case=copy.deepcopy(original)
            is_dev=case['writing_template_id'] in held[index] or case['case_id'] in scope_ids[index]
            case['split']='development' if is_dev else 'train'
            (development if is_dev else train).append(case)
        assert membership(train+development)==membership(cases)
        train_recipes={c['writing_template_id'] for c in train if c['semantic_labels']['group'] not in META}
        dev_recipes={c['writing_template_id'] for c in development if c['semantic_labels']['group'] not in META}
        assert not train_recipes&dev_recipes
        assert not {c['family_id'] for c in train}&{c['family_id'] for c in development}
        result.append({'fold':index,'train':train,'development':development,'held_out_supported_recipes':sorted(held[index])})
    assert Counter(c['case_id'] for fold in result for c in fold['development'])==Counter(c['case_id'] for c in cases)
    return result


def prepare(archive,output):
    archive,output=Path(archive),Path(output)
    if output.exists():raise ValueError('Keep previous cohorts; use a new directory')
    with zipfile.ZipFile(archive) as z:
        root='ai_training_complement_2026_11_05/native/'
        cases=[json.loads(x) for x in z.read(root+'train.jsonl').decode().splitlines()]
        config=z.read(root+'pilot_config.json');original_manifest=z.read(root+'dataset-manifest.json')
    data=recipe_folds(cases);output.mkdir(parents=True)
    report={'scope':'internal_validation_design_only','source_archive_sha256':file_sha256(archive),
            'provided_cases_sha256':membership(cases),'source_cases':400,'new_cases_generated':0,
            'external_100_used_in_partitions':False,'training_or_inference_run':False,
            'generator_sha256':file_sha256(__file__),'folds':[],
            'limit':'Supported-product evidence recipes are disjoint; authored scope cases are family-stratified, not a writing-style holdout.'}
    for fold in data:
        target=output/f"fold-{fold['fold']}";target.mkdir()
        (target/'pilot_config.json').write_bytes(config)
        manifest={'version':'evidence-data-1','scope':'user_supplied_training_only','provided_case_count':400,
                  'sole_task_training_source':True,'input_archive_sha256':report['source_archive_sha256'],
                  'provided_cases_sha256':membership(cases),'source_native_manifest_sha256':hashlib.sha256(original_manifest).hexdigest(),
                  'external_test_included':False,'old_task_data_or_weights_used':False,'gold_labels_reviewed':False,
                  'internal_selection_split':'three-fold supported-recipe holdout; separately stratified authored scope',
                  'fold':fold['fold'],'seed':73,'held_out_supported_recipes':fold['held_out_supported_recipes'],
                  'partitions':{},'files_sha256':{}}
        for role in ('train','development'):
            rows=fold[role];path=target/(role+'.jsonl');path.write_text(''.join(stable_json(c)+'\n' for c in rows))
            manifest['partitions'][role]={'file':path.name,'cases':len(rows),'families':len(rows),
                                         'group_counts':dict(Counter(c['semantic_labels']['group'] for c in rows)),
                                         'identification_counts':dict(Counter(str(c['semantic_labels']['identification']) for c in rows))}
        manifest['files_sha256']={p.name:file_sha256(p) for p in target.iterdir() if p.is_file()}
        (target/'dataset-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        report['folds'].append({'fold':fold['fold'],'train':len(fold['train']),'development':len(fold['development']),
                               'held_out_supported_recipes':fold['held_out_supported_recipes'],
                               'supported_recipe_overlap':0,'family_overlap':0,'membership_preserved':True,
                               'manifest_sha256':file_sha256(target/'dataset-manifest.json')})
    (output/'split-audit.json').write_text(json.dumps(report,indent=2)+'\n');return report
