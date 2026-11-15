"""CPU calibration/freeze/evaluation stages with explicit partition roles."""
import copy
from datetime import datetime,timezone
import json
import random
from pathlib import Path
import shutil
import tempfile
from time import perf_counter

import torch

from .calibration import VERSION,apply_temperatures,fit_temperatures,fit_outcome_gates,probabilities
from .metrics import classification,calibration_metrics,wilson
from .orchestrator import Orchestrator,READY
from .provenance import file_sha256
from .task_data import case_config,load_partition,question_examples
from .task_model import load_task_model,prediction_records,question_metrics,verify_task_bundle,write_bundle_manifest
from .task_runtime import CalibratedJudge,TaskRuntime,inference_hashes
from .training import emit


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def collect_records(model,config,cases,*,cache_path,identity,partition_sha,untrained_head=None):
    """Use the same single-row CPU path as serving, with resumable metadata only."""
    examples=question_examples(cases,config)
    cache_path=Path(cache_path)
    signature={'model_identity':identity,'partition_sha256':partition_sha,'inference_source_sha256':inference_hashes()}
    if untrained_head is not None:signature['untrained_baseline_seed']=1729
    records=[]
    if cache_path.exists():
        cached=json.loads(cache_path.read_text())
        if cached['signature']!=signature:raise ValueError('Prediction cache identity changed')
        records=cached['records']
        if len(records)>len(examples):raise ValueError('Prediction cache has extra rows')
        for cached,example in zip(records,examples):
            if cached['input_sha256']!=example['row']['text_sha256'] or cached['target']!=example['target']:
                raise ValueError('Prediction cache data changed')
    started=perf_counter();initial=len(records)
    model.eval()
    for index in range(len(records),len(examples)):
        example=examples[index]
        with torch.inference_mode():
            if untrained_head is None:values=model(example['row']).cpu().tolist();baseline=None
            else:
                features=model.features(example['row'])
                values=model.head(features.context,features.options).cpu().tolist()
                baseline=untrained_head(features.context,features.options).cpu().tolist()
        observation=prediction_records([example],[values])[0]
        if baseline is not None:observation['untrained_logits']=baseline
        records.append(observation)
        if (index+1)%64==0 or index+1==len(examples):
            atomic_json(cache_path,{'signature':signature,'records':records,'complete':index+1==len(examples)})
            emit('cpu_predictions',rows=index+1,total=len(examples),cache=str(cache_path))
    return records,{'new_rows':len(records)-initial,'seconds':perf_counter()-started,'total_rows':len(records),'batch_size':1}


def replay(cases,config,records,thresholds,temperatures,*,model=None):
    cache={r['input_sha256']:r for r in records}
    output=[]
    for case in cases:
        local=case_config(case,config)
        judge=CalibratedJudge(model,temperatures,records=cache)
        result=Orchestrator(local,judge,thresholds).evaluate(case['request'],policy_id=case.get('policy_id'))
        if result.get('error')=='decision_evaluation_failed':raise ValueError('Routed inference failed during evaluation')
        differences={}
        for key,expected in case['expected'].items():
            actual=result[key]
            equal=set(actual)==set(expected) if key in ('clarification_ids','notice_ids') else actual==expected
            if not equal:differences[key]={'expected':expected,'predicted':actual}
        output.append({'case_id':case['case_id'],'family_id':case['family_id'],'primary':case['primary'],
            'blueprint':case['intended_facts']['blueprint'],'expected':case['expected'],'prediction':result,'differences':differences})
    return output


def routing_metrics(rows):
    predicted=[r for r in rows if r['prediction']['status']==READY]
    true_ready=[r for r in rows if r['expected']['status']==READY]
    correct_ready=sum(r['expected']['status']==READY for r in predicted)
    return {'cases':len(rows),'exact_matches':sum(not r['differences'] for r in rows),
        'status_accuracy':sum(r['expected']['status']==r['prediction']['status'] for r in rows)/len(rows) if rows else None,
        'ready_precision':wilson(correct_ready,len(predicted)),
        'ready_recall':correct_ready/len(true_ready) if true_ready else None,'truly_ready':len(true_ready),
        'predicted_ready':len(predicted),'false_ready':len(predicted)-correct_ready,
        'ready_rate':len(predicted)/len(rows) if rows else None,
        'clarification_rate':sum(r['prediction']['status']=='clarification_needed' for r in rows)/len(rows) if rows else None,
        'buyer_review_rate':sum(r['prediction']['status']=='buyer_review_required' for r in rows)/len(rows) if rows else None}


def family_intervals(rows,replicates=500,seed=20261103):
    groups={}
    for row in rows:
        current=groups.setdefault(row['family_id'],[0,0,0])
        called=row['prediction']['status']==READY;true=row['expected']['status']==READY
        current[0]+=called and true;current[1]+=called;current[2]+=true
    families=list(groups.values());rng=random.Random(seed)
    precision=[];recall=[]
    for _ in range(replicates):
        sampled=rng.choices(families,k=len(families))
        tp=sum(x[0] for x in sampled);called=sum(x[1] for x in sampled);true=sum(x[2] for x in sampled)
        if called:precision.append(tp/called)
        if true:recall.append(tp/true)
    def interval(values):
        values=sorted(values)
        return {'lower':values[int(.025*(len(values)-1))],'upper':values[int(.975*(len(values)-1))]} if values else None
    return {'method':'percentile bootstrap resampling whole scenario families','replicates':replicates,'seed':seed,
        'families':len(families),'precision_95_percent':interval(precision),'recall_95_percent':interval(recall)}


def predicted_group_identification(rows,options):
    eligible=[r for r in rows if r['expected']['identification'] is not None]
    absent='incorrect_group_or_not_evaluated'
    predicted=[r['prediction']['identification'] if r['prediction']['group']==r['expected']['group'] and r['prediction']['identification'] is not None else absent for r in eligible]
    metrics=classification([r['expected']['identification'] for r in eligible],predicted,[*options,absent])
    metrics['macro_f1_over_true_identification_outcomes']=sum(metrics['per_label'][key]['f1'] for key in options)/len(options)
    metrics['incorrect_group_or_not_evaluated']=predicted.count(absent)
    metrics['includes_deterministic_type_checks']=True
    return metrics


def fit_and_freeze(raw_bundle,dataset_dir,output_dir,cache_dir):
    raw_bundle,dataset_dir,output_dir,cache_dir=map(Path,(raw_bundle,dataset_dir,output_dir,cache_dir))
    if output_dir.exists():raise ValueError('Use a new frozen runtime directory')
    model,config,manifest=load_task_model(raw_bundle)
    report=json.loads((raw_bundle/'training-report.json').read_text())
    if report['dataset_manifest']['parent_dataset_manifest_sha256']!=file_sha256(dataset_dir/'dataset-manifest.json'):
        raise ValueError('Calibration corpus differs from the declared training cohort')
    _,calibration_cases,data_manifest=load_partition(dataset_dir,'calibration',allowed_splits={'calibration'})
    calibration_records,cal_runtime=collect_records(model,config,calibration_cases,cache_path=cache_dir/'calibration-predictions.json',
        identity=manifest['model_identity'],partition_sha=data_manifest['files_sha256']['calibration.jsonl'])
    temperatures,calibration_diagnostics=fit_temperatures(calibration_records)
    emit('temperatures_fitted',temperatures=temperatures)
    _,threshold_cases,_=load_partition(dataset_dir,'threshold_selection',allowed_splits={'threshold_selection'})
    threshold_records,threshold_runtime=collect_records(model,config,threshold_cases,cache_path=cache_dir/'threshold-predictions.json',
        identity=manifest['model_identity'],partition_sha=data_manifest['files_sha256']['threshold_selection.jsonl'])
    calibrated=apply_temperatures(threshold_records,temperatures)
    gates,gate_diagnostics=fit_outcome_gates(calibrated)
    primary=[case for case in threshold_cases if case['primary']]
    joint_rows=replay(primary,config,threshold_records,gates,temperatures,model=model)
    joint=routing_metrics(joint_rows)
    # Selectively tighten the satisfied gate on threshold-selection data only.
    # Preserve its per-outcome support requirement and require zero false Ready
    # among at least 35 distinct primary families in this diagnostic sample.
    initial=gates['replacement_identification'].get('satisfied')
    candidates=sorted({r['confidence'] for r in calibrated if r['primary'] and r['question_id']=='replacement_identification' and r['choice']=='satisfied' and initial is not None and r['confidence']>=initial})
    found=initial is not None and joint['false_ready']==0 and joint['predicted_ready']>=35
    if not found and initial is not None:
        for candidate in candidates:
            support=[r for r in calibrated if r['primary'] and r['question_id']=='replacement_identification' and r['choice']=='satisfied' and r['confidence']>=candidate]
            if len(support)<50:continue
            if any(r['label']!='satisfied' for r in support):continue
            trial=copy.deepcopy(gates);trial['replacement_identification']['satisfied']=candidate
            trial_rows=replay(primary,config,threshold_records,trial,temperatures,model=model)
            metrics=routing_metrics(trial_rows)
            if metrics['false_ready']==0 and metrics['predicted_ready']>=35:
                gates,joint,joint_rows=trial,metrics,trial_rows;found=True;break
    if not found:
        gates['replacement_identification']['satisfied']=None
        joint_rows=replay(primary,config,threshold_records,gates,temperatures,model=model)
        joint=routing_metrics(joint_rows)
    calibration={'version':VERSION,'scope':'synthetic_prototype','created_at':datetime.now(timezone.utc).isoformat(),
        'model_identity':manifest['model_identity'],'config_sha256':file_sha256(raw_bundle/'pilot_config.json'),
        'inference_source_sha256':inference_hashes(),'temperatures':temperatures,'thresholds':gates,
        'fit_partitions':['calibration','threshold_selection'],'primary_variant':0,'split_unit':'scenario_family',
        'source_dataset_manifest_sha256':file_sha256(dataset_dir/'dataset-manifest.json'),
        'calibration_file_sha256':data_manifest['files_sha256']['calibration.jsonl'],
        'threshold_file_sha256':data_manifest['files_sha256']['threshold_selection.jsonl'],
        'calibration_diagnostics':calibration_diagnostics,'gate_diagnostics':gate_diagnostics,
        'joint_threshold_primary_diagnostics':joint,'joint_ready_gate_enabled':found,
        'runtime':{'calibration':cal_runtime,'threshold_selection':threshold_runtime},
        'independently_reviewed':False,'deployment_qualified':False,'qualifies_model_quality':False}
    cache_dir.mkdir(parents=True,exist_ok=True)
    atomic_json(cache_dir/'calibration-fit.json',calibration)
    atomic_json(cache_dir/'threshold-routing.json',joint_rows)
    output_dir.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.runtime-freeze-',dir=output_dir.parent) as temporary:
        staging=Path(temporary)/'model';shutil.copytree(raw_bundle,staging)
        atomic_json(staging/'calibration.json',calibration)
        write_bundle_manifest(staging,calibrated_for_synthetic_prototype=True,calibration_sha256=file_sha256(staging/'calibration.json'))
        verify_task_bundle(staging)
        runtime=TaskRuntime(staging)
        runtime.evaluate({'item_id':'freeze-check','description':'Need a replacement motor.','quantity':1,'unit':'EA','plant':'1100','required_date':'2026-12-15'})
        staging.rename(output_dir)
    emit('runtime_bundle_frozen',directory=str(output_dir),joint_ready_gate_enabled=found,threshold_primary=joint)
    return calibration


def evaluate_frozen(bundle,dataset_dir,output_dir):
    bundle,dataset_dir,output_dir=map(Path,(bundle,dataset_dir,output_dir))
    runtime=TaskRuntime(bundle)
    output_dir.mkdir(parents=True,exist_ok=True)
    from . import rules as rules_module
    signature={'artifact_identity':runtime.artifact_identity,'dataset_manifest_sha256':file_sha256(dataset_dir/'dataset-manifest.json'),
        'evaluation_source_sha256':file_sha256(Path(__file__)),'rules_source_sha256':file_sha256(rules_module.__file__),
        'untrained_baseline_seed':1729}
    lock=output_dir/'evaluation-started.json'
    if lock.exists():
        previous=json.loads(lock.read_text())
        if previous['signature']!=signature:raise ValueError('An evaluation already started with different frozen inputs')
    else:atomic_json(lock,{'signature':signature,'started_at':datetime.now(timezone.utc).isoformat()})
    if (output_dir/'evaluation-report.json').exists():raise ValueError('Frozen evaluation is already complete; read its report')
    config,cases,manifest=load_partition(dataset_dir,'prototype_evaluation',allowed_splits={'prototype_evaluation'})
    from .neural import OptionScorer
    with torch.random.fork_rng():
        torch.manual_seed(1729);untrained_head=OptionScorer(runtime.model.encoder.config.hidden_size).eval()
    records,cpu_runtime=collect_records(runtime.model,config,cases,cache_path=output_dir/'evaluation-predictions.json',
        identity=runtime.manifest['model_identity'],partition_sha=manifest['files_sha256']['prototype_evaluation.jsonl'],untrained_head=untrained_head)
    calibrated=apply_temperatures(records,runtime.temperatures)
    routed=replay(cases,config,records,runtime.thresholds,runtime.temperatures,model=runtime.model)
    from .rules import RulesJudge
    rules=[]
    for case in cases:
        result=Orchestrator(case_config(case,config),RulesJudge()).evaluate(case['request'],policy_id=case.get('policy_id'))
        differences={k:{'expected':v,'predicted':result[k]} for k,v in case['expected'].items()
            if (set(result[k])!=set(v) if k in ('clarification_ids','notice_ids') else result[k]!=v)}
        rules.append({'case_id':case['case_id'],'family_id':case['family_id'],'primary':case['primary'],
            'blueprint':case['intended_facts']['blueprint'],'expected':case['expected'],'prediction':result,'differences':differences})
    primary_records=[r for r in calibrated if r['primary']]
    untrained=apply_temperatures([{**r,'logits':r['untrained_logits']} for r in records if r['primary']],
        {'material_group':1.,'replacement_identification':1.})
    report={'kind':'first_task_prototype_evaluation','scope':'single_generator_synthetic','signature':signature,
        'cases':len(cases),'families':len({c['family_id'] for c in cases}),
        'model_oracle_policy_questions':question_metrics(primary_records),
        'untrained_shared_head_oracle_policy_questions':question_metrics(untrained),
        'untrained_baseline_note':'Seed 1729 random shared head on the same pretrained encoder; no fitted gates or adaptation.',
        'model_primary_routing':routing_metrics([r for r in routed if r['primary']]),
        'model_all_variant_routing':routing_metrics(routed),
        'model_primary_predicted_group_identification':predicted_group_identification([r for r in routed if r['primary']],config['identification_options']),
        'model_family_resampled_uncertainty':family_intervals(routed),
        'rules_primary_routing':routing_metrics([r for r in rules if r['primary']]),
        'rules_all_variant_routing':routing_metrics(rules),
        'rules_version':rules_module.RULES_VERSION,
        'calibration_by_question':{q:calibration_metrics([r for r in primary_records if r['question_id']==q])
            for q in sorted({r['question_id'] for r in primary_records})},
        'failure_counts_by_blueprint':{b:sum(bool(r['differences']) for r in routed if r['blueprint']==b) for b in sorted({r['blueprint'] for r in routed})},
        'cpu_prediction_runtime':cpu_runtime,'independently_reviewed':False,'qualifies_model_quality':False,'qualifies_cpu_budget':False}
    atomic_json(output_dir/'routing-results.json',routed);atomic_json(output_dir/'rules-results.json',rules)
    atomic_json(output_dir/'evaluation-report.json',report)
    emit('frozen_evaluation_complete',model_primary=report['model_primary_routing'],rules_primary=report['rules_primary_routing'])
    return report
