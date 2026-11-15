"""Frozen evaluation of the provided-data candidate on an explicit future set.

No training, calibration fitting, dataset search, or automatic test adaptation.
The accepted input is native case JSONL, not an inferred annotation vocabulary.
"""
from collections import Counter
from datetime import datetime, timezone
import json
import hashlib
import importlib.metadata
from pathlib import Path
import platform

from .evidence_data import question_examples
from .evidence_orchestrator import EvidenceOrchestrator, InputLimitError
from .evidence_runtime import EvidenceJudge, EvidenceRuntime, inference_hashes
from .calibration import apply_temperatures
from .metrics import classification, calibration_metrics, wilson
from .model_v4 import verify_bundle
from .orchestrator import resolve_reference, validate_request
from .provenance import file_sha256
from .rules_v4 import RulesV4
from .task_data import case_config, stable_json
from .task_iteration import atomic_json, routing_metrics, predicted_group_identification
from .task_model import prediction_records, question_metrics

VERSION='provided-evaluation-1'
ROOT=Path(__file__).resolve().parents[1]
META={'mixed_scope','insufficient_evidence','no_suitable_group'}
TEST_ROLES={'test','external_test','prototype_evaluation','independent_challenge'}
TARGETS={'supported_group_accuracy':.90,'oracle_identification_macro_f1':.85,
         'actual_identification_macro_f1':.85,'raw_ready_precision':.95,'raw_ready_wilson_lower':.90,
         'minimum_distinct_raw_ready':150,'raw_ready_recall':.60,'safety_false_ready':0}


def source_hashes():
    names=('prism_model/provided_evaluation.py','prism_model/metrics.py','prism_model/task_data.py',
           'prism_model/data_v2.py','prism_model/provenance.py','uv.lock','requirements/training.txt',
           'prism_model/task_iteration.py','prism_model/rules_v4.py','prism_model/rules_v3.py',
           'prism_model/rules_v2.py','prism_model/rules.py','scripts/evaluate_provided.py')
    return {**inference_hashes(),**{n:file_sha256(ROOT/n) for n in names}}


def training_records(bundle):
    archive=ROOT/'data/prism-ai-training-complement-2026-11-05.zip'
    import zipfile
    report=json.loads((Path(bundle)/'training-report.json').read_text())
    if file_sha256(archive)!=report['input_archive_sha256']:raise ValueError('Training archive changed')
    with zipfile.ZipFile(archive) as z:
        body=z.read('ai_training_complement_2026_11_05/native/train.jsonl')
    return [json.loads(line) for line in body.decode().splitlines() if line.strip()]


def validate_cases(cases,config,training,*,mechanism=False):
    if not cases:raise ValueError('No evaluation cases')
    identities=set();families={};roles=set();texts=set()
    for c in cases:
        for key in ('case_id','family_id','request','reference_fixture','semantic_labels','expected','primary','split','contract_version'):
            if key not in c:raise ValueError('Native case field missing: '+key)
        if any(type(c[k]) is not str or not c[k] for k in ('case_id','family_id')):raise ValueError('Invalid case/family identity')
        if c['case_id'] in identities:raise ValueError('Duplicate case identity')
        identities.add(c['case_id']);roles.add(c['split'])
        if type(c['primary']) is not bool:raise ValueError('Explicit primary variant required')
        families[c['family_id']]=families.get(c['family_id'],0)+c['primary']
        if c['contract_version']!=config['contract_version']:raise ValueError('Different policy contract')
        if not isinstance(c['reference_fixture'],list):raise ValueError('Native reference fixture must be a list')
        if any(k in c for k in ('config_overrides','thresholds','temperature','reference_integrity','verified_reference','resolved_reference')):
            raise ValueError('Unsupported trusted fixture override; adapt and review the loader before scoring')
        if c.get('policy_id') and c['policy_id'] not in config['policies']:raise ValueError('Unknown policy fixture')
        gold=c['semantic_labels'];expected=c['expected']
        if not isinstance(gold,dict) or not isinstance(expected,dict):raise ValueError('Native annotations must be objects')
        if gold.get('group') not in (None,*config['catalog']):raise ValueError('Unknown group label')
        if gold.get('identification') not in (None,*config['identification_options']):raise ValueError('Unknown identification label')
        if expected.get('http_status') not in (200,422):raise ValueError('Unsupported expected HTTP status')
        valid=True
        try:validate_request(c['request'])
        except ValueError:valid=False
        if expected['http_status']!=(200 if valid else 422):raise ValueError('Expected HTTP status disagrees with request schema')
        if expected['http_status']==422 and any(expected.get(k) is not None for k in ('status','group','identification')):
            raise ValueError('Schema failures cannot have semantic readiness labels')
        if expected['http_status']==200 and expected.get('status') is None:raise ValueError('Expected routing status missing')
        if expected.get('status')=='ready_for_buyer_review' and (gold.get('group') in META or gold.get('identification')!='satisfied'):
            raise ValueError('Ready annotation disagrees with semantic labels')
        for k in ('group','identification'):
            if expected.get(k)!=gold.get(k):raise ValueError('Inconsistent semantic and routing annotations')
        for k in ('clarification_ids','notice_ids'):
            if not isinstance(expected.get(k),list) or len(set(expected[k]))!=len(expected[k]):raise ValueError('Invalid expected correction/notice IDs')
        if expected.get('status') not in (None,'clarification_needed','buyer_review_required','evaluation_unavailable','ready_for_buyer_review'):
            raise ValueError('Unknown native status label')
        if isinstance(c['request'],dict) and isinstance(c['request'].get('description'),str):
            texts.add(' '.join(c['request']['description'].casefold().split()))
        attributes=gold.get('attributes',{})
        if not isinstance(attributes,dict):raise ValueError('Attribute labels must be an object')
        if attributes:
            group=gold['group'];pid=c.get('policy_id') or config['default_policies'].get(group)
            if not pid or set(attributes)!=set(config['policies'][pid]['required']):raise ValueError('Incomplete per-attribute labels')
            if any(v not in config['identification_options'] for v in attributes.values()):raise ValueError('Unknown attribute label')
            if gold['identification']=='satisfied' and any(v!='satisfied' for v in attributes.values()):raise ValueError('Satisfied overall annotation conflicts with attributes')
    if any(n!=1 for n in families.values()):raise ValueError('Each family must have one preselected primary case')
    overlap={'case_ids':len(identities&{c['case_id'] for c in training}),
             'families':len(set(families)&{c['family_id'] for c in training}),
             'normalized_descriptions':len(texts&{' '.join(c['request']['description'].casefold().split()) for c in training})}
    if not mechanism and (not roles<=TEST_ROLES or any(overlap.values())):
        raise ValueError('Not an unseen test set: role or supplied-training overlap')
    if mechanism and not identities<={c['case_id'] for c in training}:
        raise ValueError('Mechanism checks must use only supplied training records')
    if mechanism:
        originals={c['case_id']:stable_json({**c,'split':'train'}) for c in training}
        if any(stable_json({**c,'split':'train'})!=originals[c['case_id']] for c in cases):
            raise ValueError('Mechanism records differ from the supplied archive')
    return {'cases':len(cases),'families':len(families),'roles':sorted(roles),'training_overlap':overlap}


def freeze(bundle,output):
    bundle,output=Path(bundle),Path(output)
    if output.resolve().is_relative_to(bundle.resolve()):raise ValueError('Keep the freeze outside the model')
    if output.exists():raise ValueError('Keep the existing evaluation freeze')
    manifest=verify_bundle(bundle)
    report=json.loads((bundle/'training-report.json').read_text())
    if report['training']['training_cases']!=400 or not report['only_supplied_training_data_used'] or report['calibrated']:
        raise ValueError('Expected the uncalibrated all-400 candidate')
    training=training_records(bundle)
    value={'version':VERSION,'created_at':datetime.now(timezone.utc).isoformat(),
           'model_artifact_identity':file_sha256(bundle/'bundle-manifest.json'),'model_identity':manifest['model_identity'],
           'input_archive_sha256':report['input_archive_sha256'],'training_families':len(training),
           'evaluation_source_sha256':source_hashes(),'test_contents_opened_by_this_freeze':False,
           'procedure':'Raw argmax group plus individually composed global/required-attribute judgments, T=1 and zero diagnostic gates; no fitted thresholds',
           'primary_unit':'one preselected primary request per scenario family',
           'comparators':['frozen RulesV4, separate from neural inference'],
           'targets':dict(TARGETS),
           'limits':'AI-authored annotations are provisional unless separately reviewed. This runner cannot establish calibrated operational reliability.'}
    atomic_json(output,value);return value


def assessments(targets,routing,supported_accuracy,oracle_f1,actual_f1,safety_count,safety_false):
    def above(value,target):return 'inconclusive' if value is None else 'pass' if value>=target else 'fail'
    precision=routing['ready_precision']
    ready=('inconclusive' if routing['predicted_ready']<targets['minimum_distinct_raw_ready'] else
           'pass' if precision['estimate']>=targets['raw_ready_precision'] and precision['lower']>=targets['raw_ready_wilson_lower'] else 'fail')
    return {'supported_group_accuracy':above(supported_accuracy,targets['supported_group_accuracy']),
            'oracle_identification_macro_f1':above(oracle_f1,targets['oracle_identification_macro_f1']),
            'actual_identification_macro_f1':above(actual_f1,targets['actual_identification_macro_f1']),
            'raw_ready_precision_and_support':ready,
            'raw_ready_recall':above(routing['ready_recall'],targets['raw_ready_recall']),
            'safety_false_ready':'inconclusive' if not safety_count else 'pass' if safety_false==0 else 'fail'}


def decision_routing_metrics(rows):
    result=routing_metrics(rows)
    # Matching "Ready" statuses cannot excuse selecting the wrong product group.
    called=[r for r in rows if r['prediction']['status']=='ready_for_buyer_review']
    correct=sum(r['expected']['status']=='ready_for_buyer_review'
                and r['prediction']['group']==r['expected']['group']
                and r['prediction']['identification']==r['expected']['identification'] for r in called)
    result['status_only_ready_precision']=result['ready_precision']
    result['status_only_ready_recall']=result['ready_recall']
    result['ready_precision']=wilson(correct,len(called))
    result['ready_recall']=correct/result['truly_ready'] if result['truly_ready'] else None
    result['false_ready']=len(called)-correct
    result['ready_correctness']='Expected Ready plus correct product group and identification'
    return result


def evaluate(bundle,cases_path,freeze_path,output,*,mechanism=False):
    bundle,cases_path,freeze_path,output=map(Path,(bundle,cases_path,freeze_path,output))
    if output.resolve().is_relative_to(bundle.resolve()):raise ValueError('Keep results outside the model')
    if output.exists():raise ValueError('Keep prior results; use a new output directory')
    frozen=json.loads(freeze_path.read_text())
    if frozen.get('version')!=VERSION or frozen['evaluation_source_sha256']!=source_hashes():raise ValueError('Evaluation procedure changed')
    if frozen.get('targets')!=TARGETS:raise ValueError('Predeclared targets changed')
    if frozen['model_artifact_identity']!=file_sha256(bundle/'bundle-manifest.json'):raise ValueError('Frozen candidate changed')
    runtime=EvidenceRuntime(bundle,require_calibration=False)
    payload=cases_path.read_bytes();cases_sha=hashlib.sha256(payload).hexdigest()
    cases=[json.loads(line) for line in payload.decode().splitlines() if line.strip()]
    audit=validate_cases(cases,runtime.config,training_records(bundle),mechanism=mechanism)
    output.mkdir(parents=True)
    atomic_json(output/'started.json',{'cases_sha256':cases_sha,'freeze_sha256':file_sha256(freeze_path),
                                     'scope':'training_data_mechanism_only' if mechanism else 'user_supplied_test_evaluation'})
    temperatures={q:1. for q in runtime.temperatures}
    gates={('group' if q=='material_group' else q):{v:0. for v in (runtime.config['catalog'] if q=='material_group' else runtime.config['identification_options'])} for q in temperatures}
    routed=[];rules=[];records=[];cache={};question_errors=[]
    for c in cases:
        valid=True
        try:validate_request(c['request'])
        except ValueError:valid=False
        local=case_config(c,runtime.config) if valid else runtime.config
        judge=EvidenceJudge(runtime.model,temperatures,gates,local,records=cache)
        result=EvidenceOrchestrator(local,judge,gates).evaluate(c['request'],policy_id=c.get('policy_id'),lookup_failure=c.get('lookup_failure',False))
        def row(result):
            differences={k:{'expected':v,'predicted':result.get(k)} for k,v in c['expected'].items()
                         if (set(v)!=set(result.get(k,[])) if k in ('clarification_ids','notice_ids') else v!=result.get(k))}
            return {'case_id':c['case_id'],'family_id':c['family_id'],'primary':c['primary'],
                    'expected':c['expected'],'prediction':result,'differences':differences}
        routed.append(row(result))
        rules.append(row(EvidenceOrchestrator(local,RulesV4()).evaluate(c['request'],policy_id=c.get('policy_id'),lookup_failure=c.get('lookup_failure',False))))
        if valid and c['semantic_labels']['group'] is not None:
            reference=resolve_reference(c['request'],local,c.get('lookup_failure',False))
            if reference['status']!='unavailable' and not reference['integrity_issues']:
                for e in question_examples([c],runtime.config):
                    gold_group=c['semantic_labels']['group'];pid=c.get('policy_id') or local['default_policies'].get(gold_group)
                    wrong_kind=pid and len(reference['components'])==1 and reference['components'][0]['kind']!=local['policies'][pid]['kind']
                    if wrong_kind and e['row']['question_id']!='material_group':continue
                    try:judge.judge(e['row'])  # Same frozen model/cache; no fitting.
                    except InputLimitError:
                        question_errors.append({'case_id':c['case_id'],'question_id':e['row']['question_id'],'error':'input_too_long'})
                        continue
                    r=cache[e['row']['text_sha256']]
                    records.extend(prediction_records([e],[r['logits']]))
        if len(routed)%10==0:print(json.dumps({'event':'provided_evaluation_cases','done':len(routed),'total':len(cases)}),flush=True)
    primary=[r for r in routed if r['primary']];primary_rules=[r for r in rules if r['primary']]
    group_rows=[r for r in primary if r['expected']['group'] is not None]
    absent='no_group_decision';labels=[*runtime.config['catalog'],absent]
    groups=classification([r['expected']['group'] for r in group_rows],
                          [r['prediction']['group'] or absent for r in group_rows],labels)
    groups['macro_f1_over_catalog']=sum(groups['per_label'][k]['f1'] for k in runtime.config['catalog'])/len(runtime.config['catalog'])
    supported=[r for r in group_rows if r['expected']['group'] not in META]
    routing=decision_routing_metrics(primary);oracle=question_metrics([r for r in records if r['primary']])
    safety=[r for r in primary if r['expected']['group'] in META or r['expected']['identification'] in ('conflicting','unclear') or r['expected']['status']=='evaluation_unavailable']
    safety_false=sum(r['prediction']['status']=='ready_for_buyer_review' for r in safety)
    actual_id=predicted_group_identification(primary,runtime.config['identification_options'])
    probability_records=apply_temperatures(records,temperatures)
    verdict={'supported_group_accuracy':sum(r['prediction']['group']==r['expected']['group'] for r in supported)/len(supported) if supported else None,
             'oracle_identification_macro_f1':oracle.get('replacement_identification',{}).get('macro_f1'),
             'raw_ready_precision_evidence':'sufficient' if routing['predicted_ready']>=frozen['targets']['minimum_distinct_raw_ready'] else 'inconclusive: too few distinct Ready predictions',
             'safety_cases':len(safety),'safety_false_ready':safety_false,
             'operational_reliability_qualified':False,'calibrated':False}
    report={'scope':'training_data_mechanism_only' if mechanism else 'user_supplied_test_against_provided_annotations',
            'signature':json.loads((output/'started.json').read_text()),'model_artifact_identity':runtime.artifact_identity,
            'data_audit':audit,'primary_raw_routing':routing,'all_variant_raw_routing':decision_routing_metrics(routed),
            'group_metrics':groups,'predicted_group_identification':actual_id,
            'oracle_policy_question_metrics':oracle,'rules_primary_routing':decision_routing_metrics(primary_rules),'diagnostics':verdict,
            'authors':dict(Counter(c.get('author','unspecified') for c in cases)),
            'evaluation_runtime':{'platform':platform.platform(),'architecture':platform.machine(),'python':platform.python_version(),
                                  **{name:importlib.metadata.version(name) for name in ('torch','transformers','tokenizers')},
                                  'device':'cpu','precision':'fp32'},
            'annotation_review_status':dict(Counter(c.get('review_status','unspecified') for c in cases)),
            'target_assessments':assessments(frozen['targets'],routing,verdict['supported_group_accuracy'],verdict['oracle_identification_macro_f1'],actual_id['macro_f1_over_true_identification_outcomes'] if actual_id['observations'] else None,len(safety),safety_false),
            'uncalibrated_probability_diagnostics':{q:calibration_metrics([r for r in probability_records if r['primary'] and r['question_id']==q]) for q in sorted({r['question_id'] for r in probability_records})},
            'oracle_question_errors':question_errors,'training_performed':False,'thresholds_or_temperature_fitted':False,'qualifies_operational_reliability':False,
            'notes':'Zero gates expose raw model decisions and must not be deployed as calibrated thresholds. No probability is asserted for the composed decision.'}
    if file_sha256(cases_path)!=cases_sha:raise ValueError('Input changed during evaluation')
    atomic_json(output/'routing.json',routed);atomic_json(output/'rules-routing.json',rules)
    atomic_json(output/'oracle-question-records.json',records);atomic_json(output/'report.json',report)
    return report
