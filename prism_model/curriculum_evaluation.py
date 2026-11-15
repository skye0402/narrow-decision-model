"""Calibration and frozen evaluation for the relational continuation.

Uses the unchanged decision runtime. Ready metrics require the right product
and identification; no test or exposed-regression outputs select checkpoints.
"""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import tempfile

from . import evidence_iteration,offline_v4,provided_evaluation,rules_v4
from .calibration import apply_temperatures,fit_temperatures
from .evidence_data import load_partition,question_examples
from .evidence_iteration import collect,select_gate
from .evidence_orchestrator import EvidenceOrchestrator
from .evidence_runtime import EvidenceJudge,EvidenceRuntime,inference_hashes
from .metrics import wilson
from .model_v4 import manifest
from .provenance import file_sha256
from .provided_evaluation import TARGETS,assessments,decision_routing_metrics,validate_cases
from .rules_v4 import RulesV4
from .task_data import case_config
from .task_iteration import atomic_json,predicted_group_identification
from .task_model import question_metrics
from .training import emit

VERSION='reference-curriculum-evaluation-1'
ROOT=Path(__file__).resolve().parents[1]


def source_hashes():
    # Binding the old metric helpers does not change their previous freeze.
    return {**provided_evaluation.source_hashes(),**{str(Path(p).relative_to(ROOT)):file_sha256(p) for p in
            (__file__,evidence_iteration.__file__,offline_v4.__file__,ROOT/'scripts/evaluate_curriculum.py')}}


def raw_gates(runtime):
    return {('group' if q=='material_group' else q):
            {v:0. for v in (runtime.config['catalog'] if q=='material_group' else runtime.config['identification_options'])}
            for q in runtime.temperatures}


def joint_gate_eligible(metrics):
    return (metrics['predicted_ready']>=150 and metrics['false_ready']==0
            and (metrics['ready_recall'] or 0.)>=.60)


def replay(runtime,cases,records,thresholds,temperatures,*,cache=None):
    cache=cache if cache is not None else {r['input_sha256']:r for r in records}
    output=[]
    for c in cases:
        local=case_config(c,runtime.config)
        judge=EvidenceJudge(runtime.model,temperatures,thresholds,local,mode=runtime.decision_mode,records=cache)
        result=EvidenceOrchestrator(local,judge,thresholds).evaluate(c['request'],policy_id=c.get('policy_id'),
                                                                  lookup_failure=c.get('lookup_failure',False))
        if result.get('error'):raise ValueError('Routed evaluation failed: '+result['error'])
        differences={k:{'expected':v,'predicted':result.get(k)} for k,v in c['expected'].items()
                     if (set(v)!=set(result.get(k,[])) if k in ('clarification_ids','notice_ids') else v!=result.get(k))}
        output.append({'case_id':c['case_id'],'family_id':c['family_id'],'primary':c['primary'],
            'blueprint':c.get('intended_facts',{}).get('blueprint','external_unannotated'),
            'expected':c['expected'],'prediction':result,'differences':differences,
            'model_question_count':len(judge.trace),'question_judgments':judge.trace})
    return output


def fit(raw,dataset,output,cache):
    raw,dataset,output,cache=map(Path,(raw,dataset,output,cache))
    if output.exists():raise ValueError('Use a new calibrated output')
    runtime=EvidenceRuntime(raw,require_calibration=False)
    report=json.loads((raw/'training-report.json').read_text())
    if report['kind']!='relational_warm_full_encoder_training' or not report['runtime']['cuda_execution_verified']:
        raise ValueError('Expected the L4 broader-data candidate')
    identity=file_sha256(dataset/'dataset-manifest.json')
    if report['dataset_manifest']['parent_dataset_manifest_sha256']!=identity:raise ValueError('Different cohort')
    _,calibration,m=load_partition(dataset,'calibration',allowed_splits={'calibration'})
    records,cal_time=collect(runtime,calibration,cache/'calibration-predictions.json',m['files_sha256']['calibration.jsonl'])
    temperatures,diagnostics=fit_temperatures(records)
    _,selection,_=load_partition(dataset,'threshold_selection',allowed_splits={'threshold_selection'})
    records,gate_time=collect(runtime,selection,cache/'threshold-predictions.json',m['files_sha256']['threshold_selection.jsonl'])
    calibrated=apply_temperatures(records,temperatures);gates={};gate_diagnostics={}
    for q in sorted(temperatures):
        rows=[r for r in calibrated if r['question_id']==q];key='group' if q=='material_group' else q
        gates[key]={};gate_diagnostics[key]={}
        for outcome in rows[0]['option_ids']:
            settings=(.99,.95,50) if q=='material_group' else (.99,.95,100) if q=='replacement_identification' and outcome=='satisfied' else (.99,.90,30) if outcome=='satisfied' else (.97,.90,30)
            selected=select_gate(rows,outcome,precision=settings[0],lower=settings[1],support=settings[2])
            gates[key][outcome]=selected['threshold'];gate_diagnostics[key][outcome]=selected
    primary=[c for c in selection if c['primary']];prediction_cache={r['input_sha256']:r for r in records}
    routed=replay(runtime,primary,records,gates,temperatures,cache=prediction_cache);joint=decision_routing_metrics(routed)
    initial=gates['replacement_identification']['satisfied'];enabled=joint_gate_eligible(joint)
    if not enabled and initial is not None:
        rows=[r for r in calibrated if r['primary'] and r['question_id']=='replacement_identification' and r['choice']=='satisfied']
        for cutoff in sorted({r['confidence'] for r in rows if r['confidence']>=initial}):
            # Search only threshold-selection data; each global gate retains its
            # own 100-family and 99%/95%-lower precision requirements.
            selected=[r for r in rows if r['confidence']>=cutoff]
            interval=wilson(sum(r['label']=='satisfied' for r in selected),len(selected))
            if len(selected)<100 or interval['estimate']<.99 or interval['lower']<.95:continue
            trial={k:dict(v) for k,v in gates.items()};trial['replacement_identification']['satisfied']=cutoff
            trial_rows=replay(runtime,primary,records,trial,temperatures,cache=prediction_cache)
            candidate=decision_routing_metrics(trial_rows)
            if joint_gate_eligible(candidate):gates,routed,joint,enabled=trial,trial_rows,candidate,True;break
    before_disable=joint
    if not enabled:
        gates['replacement_identification']['satisfied']=None
        routed=replay(runtime,primary,records,gates,temperatures,cache=prediction_cache)
        joint=decision_routing_metrics(routed)
    value={'version':'evidence-calibration-1','scope':'synthetic_prototype','created_at':datetime.now(timezone.utc).isoformat(),
           'model_identity':runtime.manifest['model_identity'],'config_sha256':file_sha256(raw/'pilot_config.json'),
           'inference_source_sha256':inference_hashes(),'fitting_source_sha256':file_sha256(__file__),
           'offline_source_sha256':file_sha256(offline_v4.__file__),'temperatures':temperatures,'thresholds':gates,
           'gate_diagnostics':gate_diagnostics,'calibration_diagnostics':diagnostics,
           'source_dataset_manifest_sha256':identity,'fit_partitions':['calibration','threshold_selection'],
           'primary_variant':0,'joint_ready_gate_enabled':enabled,'joint_threshold_primary_diagnostics':joint,
           'joint_before_possible_disable':before_disable,'joint_eligibility':{'minimum_ready_families':150,'false_ready':0,'minimum_recall':.60},
           'runtime':{'calibration':cal_time,'threshold':gate_time},'independently_reviewed':False,'deployment_qualified':False}
    atomic_json(cache/'calibration-fit.json',value);atomic_json(cache/'threshold-routing.json',routed)
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.broad-calibration-',dir=output.parent) as temp:
        stage=Path(temp)/'model';shutil.copytree(raw,stage);atomic_json(stage/'calibration.json',value)
        manifest(stage,calibrated_for_synthetic_prototype=True);EvidenceRuntime(stage);stage.rename(output)
    emit('curriculum_calibration_frozen',joint_ready_gate_enabled=enabled,joint=joint)
    return value


def freeze(bundle,dataset,output):
    bundle,dataset,output=map(Path,(bundle,dataset,output))
    if output.exists():raise ValueError('Preserve the prior freeze')
    runtime=EvidenceRuntime(bundle)
    identity=file_sha256(dataset/'dataset-manifest.json')
    if runtime.calibration['source_dataset_manifest_sha256']!=identity:raise ValueError('Different calibration cohort')
    if runtime.calibration['fitting_source_sha256']!=file_sha256(__file__):raise ValueError('Fitting source changed')
    if runtime.calibration['offline_source_sha256']!=file_sha256(offline_v4.__file__):raise ValueError('Prediction worker changed')
    report=json.loads((bundle/'training-report.json').read_text())
    if not report['runtime']['cuda_execution_verified'] or report['exposed_author100_used']:raise ValueError('Wrong training provenance')
    value={'version':VERSION,'created_at':datetime.now(timezone.utc).isoformat(),
           'model_artifact_identity':runtime.artifact_identity,'model_identity':runtime.manifest['model_identity'],
           'dataset_manifest_sha256':identity,'evaluation_source_sha256':source_hashes(),'targets':dict(TARGETS),
           'evaluation_role':'prototype_evaluation','primary_unit':'One preselected primary request per family',
           'procedures':['raw composed diagnostic, T=1 and zero gates','frozen calibrated composed path','separate frozen RulesV4'],
           'independently_authored':False,'independently_reviewed':False,
           'limits':'Synthetic same-assistant generalization evidence; external reliability remains unqualified'}
    atomic_json(output,value);return value


def summarize(rows,cases,options,oracle_f1):
    routing=decision_routing_metrics(rows);by_id={c['case_id']:c for c in cases}
    supported=[r for r in rows if r['expected']['identification'] is not None]
    accuracy=sum(r['prediction']['group']==r['expected']['group'] for r in supported)/len(supported) if supported else None
    actual=predicted_group_identification(rows,options)
    safety=[r for r in rows if r['expected']['group'] in provided_evaluation.META
            or r['expected']['identification'] in ('conflicting','unclear')
            or r['expected']['status']=='evaluation_unavailable']
    safety_false=sum(r['prediction']['status']=='ready_for_buyer_review' for r in safety)
    slices={}
    for r in rows:
        key=by_id[r['case_id']].get('intended_facts',{}).get('blueprint','not_annotated')
        slices.setdefault(key,[]).append(r)
    return {'routing':routing,'supported_group_accuracy':accuracy,'actual_identification':actual,
            'oracle_identification_macro_f1':oracle_f1,'safety_cases':len(safety),'safety_false_ready':safety_false,
            'assessment':assessments(TARGETS,routing,accuracy,oracle_f1,actual['macro_f1_over_true_identification_outcomes'],len(safety),safety_false),
            'slices':{k:decision_routing_metrics(v) for k,v in slices.items()}}


def evaluate(bundle,dataset,freeze_path,output,*,cases_path=None,exposed_regression=False):
    bundle,dataset,freeze_path,output=map(Path,(bundle,dataset,freeze_path,output))
    if output.exists():raise ValueError('Preserve prior results')
    frozen=json.loads(freeze_path.read_text())
    if frozen['version']!=VERSION or frozen['evaluation_source_sha256']!=source_hashes() or frozen['targets']!=TARGETS:
        raise ValueError('Frozen procedure changed')
    runtime=EvidenceRuntime(bundle)
    if runtime.artifact_identity!=frozen['model_artifact_identity']:raise ValueError('Frozen model changed')
    if file_sha256(dataset/'dataset-manifest.json')!=frozen['dataset_manifest_sha256']:raise ValueError('Cohort changed')
    training=[]
    for role in ('train','development','calibration','threshold_selection'):
        _,rows,m=load_partition(dataset,role,allowed_splits={role});training.extend(rows)
    if cases_path is None:
        _,cases,m=load_partition(dataset,'prototype_evaluation',allowed_splits={'prototype_evaluation'})
        input_path=dataset/'prototype_evaluation.jsonl';scope='reserved_synthetic_evaluation'
    else:
        if not exposed_regression:raise ValueError('External-path evaluation must explicitly acknowledge exposure')
        input_path=Path(cases_path);cases=[json.loads(x) for x in input_path.read_text().splitlines()]
        scope='exposed_author100_regression_not_blind'
    audit=validate_cases(cases,runtime.config,training)
    output.mkdir(parents=True)
    atomic_json(output/'started.json',{'scope':scope,'cases_sha256':file_sha256(input_path),
                                     'freeze_sha256':file_sha256(freeze_path),'input_audit':audit})
    records,timing=collect(runtime,cases,output/'oracle-predictions.json',file_sha256(input_path))
    primary=[c for c in cases if c['primary']];oracle=question_metrics([r for r in records if r['primary']])
    cache={r['input_sha256']:r for r in records}
    raw=replay(runtime,cases,records,raw_gates(runtime),{q:1. for q in runtime.temperatures},cache=cache)
    calibrated=replay(runtime,cases,records,runtime.thresholds,runtime.temperatures,cache=cache)
    rules=[]
    for c in cases:
        local=case_config(c,runtime.config);p=EvidenceOrchestrator(local,RulesV4()).evaluate(c['request'],policy_id=c.get('policy_id'),lookup_failure=c.get('lookup_failure',False))
        differences={k:{'expected':v,'predicted':p.get(k)} for k,v in c['expected'].items()
                     if (set(v)!=set(p.get(k,[])) if k in ('clarification_ids','notice_ids') else v!=p.get(k))}
        rules.append({'case_id':c['case_id'],'family_id':c['family_id'],'primary':c['primary'],
                      'expected':c['expected'],'prediction':p,'differences':differences})
    primary_ids={c['case_id'] for c in primary}
    report={'version':VERSION,'scope':scope,'freeze_sha256':file_sha256(freeze_path),
            'model_artifact_identity':runtime.artifact_identity,'input_sha256':file_sha256(input_path),
            'input_audit':audit,'oracle_primary_questions':oracle,'cpu_prediction_runtime':timing,
            'independently_reviewed':False,'deployment_qualified':False}
    for name,rows in (('raw',raw),('calibrated',calibrated),('rules',rules)):
        atomic_json(output/(name+'-routing.json'),rows)
        chosen=[r for r in rows if r['case_id'] in primary_ids]
        report[name]={'primary':summarize(chosen,cases,runtime.config['identification_options'],
                                        oracle['replacement_identification']['macro_f1'] if name!='rules' else None),
                      'all_variants_routing':decision_routing_metrics(rows)}
    atomic_json(output/'report.json',report)
    emit('curriculum_evaluation_complete',scope=scope,raw=report['raw']['primary']['routing'],calibrated=report['calibrated']['primary']['routing'])
    return report
