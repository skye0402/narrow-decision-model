"""CPU fitting and frozen evaluation for the prospective 80/20 evidence corpus."""
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import tempfile
from time import perf_counter

from .calibration import apply_temperatures,fit_temperatures
from .evidence_data import load_partition,question_examples
from .evidence_runtime import EvidenceJudge,EvidenceRuntime,inference_hashes
from .evidence_orchestrator import EvidenceOrchestrator
from .metrics import calibration_metrics,wilson
from .model_v4 import manifest
from .provenance import file_sha256
from .rules_v4 import RulesV4,VERSION as RULES_VERSION
from .task_data import case_config
from .task_iteration import atomic_json,routing_metrics,family_intervals,predicted_group_identification
from .task_model import prediction_records,question_metrics
from .training import emit
from . import offline_v4


def collect(runtime,cases,path,partition_sha):
    path=Path(path)
    signature={'model_identity':runtime.manifest['model_identity'],'partition_sha256':partition_sha,
               'inference_source_sha256':inference_hashes(),'fitting_source_sha256':file_sha256(__file__),
               'worker_source_sha256':file_sha256(offline_v4.__file__)}
    examples=question_examples(cases,runtime.config,attributes=runtime.task_mode=='attributes');records=[]
    if path.exists():
        cached=json.loads(path.read_text())
        if cached['signature']!=signature:raise ValueError('Changed prediction cache identity')
        records=cached['records']
        if len(records)>len(examples):raise ValueError('Extra cached predictions')
        for r,e in zip(records,examples):
            if r['input_sha256']!=e['row']['text_sha256'] or any(r[k]!=e[k] for k in ('label','case_id','family_id','primary','target')):
                raise ValueError('Changed cache input or cohort metadata')
    started=perf_counter();initial=len(records)
    values=offline_v4.predictions(runtime.directory,(e['row'] for e in examples[initial:]))
    for index,logits in enumerate(values,start=initial):
        records.append(prediction_records([examples[index]],[logits])[0])
        if (index+1)%128==0 or index+1==len(examples):
            atomic_json(path,{'signature':signature,'records':records,'complete':index+1==len(examples)})
            emit('evidence_cpu_predictions',rows=index+1,total=len(examples),partition=path.stem)
    return records,{'new_rows':len(records)-initial,'seconds':perf_counter()-started,'total_rows':len(records),
                   'batch_size':1,'offline_workers':3,'torch_threads_per_worker':2}


def replay(runtime,cases,records,thresholds,temperatures,*,mode=None,cache=None):
    cache=cache if cache is not None else {r['input_sha256']:r for r in records}
    output=[]
    for case in cases:
        local=case_config(case,runtime.config)
        judge=EvidenceJudge(runtime.model,temperatures,thresholds,local,mode=mode or runtime.decision_mode,records=cache)
        result=EvidenceOrchestrator(local,judge,thresholds).evaluate(case['request'],policy_id=case.get('policy_id'))
        if result.get('error'):raise ValueError('Routed evaluation failed: '+result['error'])
        differences={}
        for key,wanted in case['expected'].items():
            actual=result.get(key)
            if (set(actual or [])!=set(wanted) if key in ('clarification_ids','notice_ids') else actual!=wanted):
                differences[key]={'expected':wanted,'predicted':actual}
        output.append({'case_id':case['case_id'],'family_id':case['family_id'],'primary':case['primary'],
            'blueprint':case['intended_facts']['blueprint'],'expected':case['expected'],'prediction':result,
            'differences':differences,'model_question_count':len(judge.trace),
            'question_judgments':judge.trace})
    return output


def select_gate(records,outcome,*,precision,lower,support):
    rows=[r for r in records if r['primary'] and r['choice']==outcome]
    if len({r['family_id'] for r in rows})!=len(rows):raise ValueError('Repeated gate family')
    for cutoff in sorted({r['confidence'] for r in rows}):
        selected=[r for r in rows if r['confidence']>=cutoff]
        interval=wilson(sum(r['label']==outcome for r in selected),len(selected))
        if len(selected)>=support and interval['estimate']>=precision and interval['lower']>=lower:
            return {'threshold':cutoff,'support':len(selected),'precision':interval,
                    'minimum_observed_confidence':min(r['confidence'] for r in rows),
                    'minimum_support':support,'target_precision':precision,'minimum_lower_bound':lower}
    return {'threshold':None,'available_predictions':len(rows),'minimum_support':support,
            'target_precision':precision,'minimum_lower_bound':lower,'reason':'insufficient_support_or_precision'}


def fit(raw,dataset,output,cache):
    raw,dataset,output,cache=map(Path,(raw,dataset,output,cache))
    if output.exists():raise ValueError('Use a new calibrated bundle directory')
    runtime=EvidenceRuntime(raw,require_calibration=False)
    report=json.loads((raw/'training-report.json').read_text())
    if report['dataset_manifest']['scope']!='synthetic_prototype' or not report['runtime']['cuda_execution_verified']:
        raise ValueError('CPU mechanism artifacts are not candidate training runs')
    if report['dataset_manifest']['parent_dataset_manifest_sha256']!=file_sha256(dataset/'dataset-manifest.json'):
        raise ValueError('Data differs from training cohort')
    _,cases,m=load_partition(dataset,'calibration',allowed_splits={'calibration'})
    calibration,cal_time=collect(runtime,cases,cache/'calibration-predictions.json',m['files_sha256']['calibration.jsonl'])
    temperatures,diagnostics=fit_temperatures(calibration);emit('evidence_temperatures',values=temperatures)
    _,cases,_=load_partition(dataset,'threshold_selection',allowed_splits={'threshold_selection'})
    records,selection_time=collect(runtime,cases,cache/'threshold-predictions.json',m['files_sha256']['threshold_selection.jsonl'])
    calibrated=apply_temperatures(records,temperatures);gates={};gate_diagnostics={}
    for q in sorted(temperatures):
        rows=[r for r in calibrated if r['question_id']==q];key='group' if q=='material_group' else q
        gates[key]={};gate_diagnostics[key]={}
        for outcome in rows[0]['option_ids']:
            settings=(.99,.95,50) if q=='material_group' else (.99,.95,100) if q=='replacement_identification' and outcome=='satisfied' else (.99,.90,30) if outcome=='satisfied' else (.97,.90,30)
            selected=select_gate(rows,outcome,precision=settings[0],lower=settings[1],support=settings[2])
            gates[key][outcome]=selected['threshold'];gate_diagnostics[key][outcome]=selected
    primary=[c for c in cases if c['primary']];prediction_cache={r['input_sha256']:r for r in records}
    routed=replay(runtime,primary,records,gates,temperatures,cache=prediction_cache);joint=routing_metrics(routed)
    enabled=joint['predicted_ready']>=60 and joint['false_ready']==0
    initial=gates['replacement_identification']['satisfied']
    if not enabled and initial is not None:
        rows=[r for r in calibrated if r['primary'] and r['question_id']=='replacement_identification' and r['choice']=='satisfied']
        for cutoff in sorted({r['confidence'] for r in rows if r['confidence']>=initial}):
            selected=[r for r in rows if r['confidence']>=cutoff]
            interval=wilson(sum(r['label']=='satisfied' for r in selected),len(selected))
            if len(selected)<100 or interval['estimate']<.99 or interval['lower']<.95:continue
            trial={k:dict(v) for k,v in gates.items()};trial['replacement_identification']['satisfied']=cutoff
            trial_rows=replay(runtime,primary,records,trial,temperatures,cache=prediction_cache);candidate=routing_metrics(trial_rows)
            if candidate['false_ready']==0 and candidate['predicted_ready']>=60:
                gates,routed,joint,enabled=trial,trial_rows,candidate,True;break
    if not enabled:
        gates['replacement_identification']['satisfied']=None
        routed=replay(runtime,primary,records,gates,temperatures,cache=prediction_cache);joint=routing_metrics(routed)
    value={'version':'evidence-calibration-1','scope':'synthetic_prototype','created_at':datetime.now(timezone.utc).isoformat(),
           'model_identity':runtime.manifest['model_identity'],'config_sha256':file_sha256(raw/'pilot_config.json'),
           'inference_source_sha256':inference_hashes(),'fitting_source_sha256':file_sha256(__file__),
           'offline_source_sha256':file_sha256(offline_v4.__file__),'temperatures':temperatures,'thresholds':gates,
           'gate_diagnostics':gate_diagnostics,'calibration_diagnostics':diagnostics,
           'source_dataset_manifest_sha256':file_sha256(dataset/'dataset-manifest.json'),
           'fit_partitions':['calibration','threshold_selection'],'primary_variant':0,
           'joint_ready_gate_enabled':enabled,'joint_threshold_primary_diagnostics':joint,
           'runtime':{'calibration':cal_time,'threshold':selection_time},
           'independently_reviewed':False,'deployment_qualified':False}
    atomic_json(cache/'calibration-fit.json',value);atomic_json(cache/'threshold-routing.json',routed)
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.evidence-freeze-',dir=output.parent) as temp:
        staging=Path(temp)/'model';shutil.copytree(raw,staging);atomic_json(staging/'calibration.json',value)
        manifest(staging,calibrated_for_synthetic_prototype=True);EvidenceRuntime(staging);staging.rename(output)
    emit('evidence_calibration_frozen',ready_gate_enabled=enabled,joint=joint)
    return value


def freeze_comparison(dataset,holistic,attributes,baseline,output):
    dataset,holistic,attributes,baseline,output=map(Path,(dataset,holistic,attributes,baseline,output))
    if output.exists():raise ValueError('Comparison already frozen; read it')
    identity=file_sha256(dataset/'dataset-manifest.json');candidates=[]
    parents=[];budgets=[]
    for name,directory,expected_mode in [('A',holistic,'holistic'),('B',attributes,'attributes')]:
        runtime=EvidenceRuntime(directory)
        if runtime.task_mode!=expected_mode or runtime.calibration['source_dataset_manifest_sha256']!=identity:
            raise ValueError('Comparison candidates do not share the declared cohort')
        if runtime.calibration['fitting_source_sha256']!=file_sha256(__file__) or runtime.calibration['offline_source_sha256']!=file_sha256(offline_v4.__file__):
            raise ValueError('Fitting/worker code changed after calibration')
        report=json.loads((directory/'training-report.json').read_text())
        parents.append(report['parent_artifact_identity']);budgets.append(report['training']['optimization_rows_per_epoch'])
        candidates.append({'name':name,'task_mode':expected_mode,'directory':str(directory),
                           'artifact_identity':runtime.artifact_identity,'model_identity':runtime.manifest['model_identity']})
    if len(set(parents))!=1 or len(set(budgets))!=1 or parents[0]!=file_sha256(baseline/'bundle-manifest.json'):
        raise ValueError('Unmatched starting checkpoint or update budget')
    from .model_v3 import RuntimeV3
    previous=RuntimeV3(baseline)
    candidates.append({'name':'v0.3','task_mode':'frozen_holistic','directory':str(baseline),
                       'artifact_identity':previous.artifact_identity,'model_identity':previous.manifest['model_identity']})
    from . import rules_v4
    value={'version':'evidence-comparison-freeze-1','created_at':datetime.now(timezone.utc).isoformat(),
           'dataset_manifest_sha256':identity,'evaluation_source_sha256':file_sha256(__file__),
           'rules_source_sha256':file_sha256(rules_v4.__file__),
           'candidates':candidates,'fit_roles':['train','development','calibration','threshold_selection'],
           'evaluation_role':'prototype_evaluation','independently_reviewed':False}
    atomic_json(output,value);return value


def evaluate(bundle,dataset,output,comparison_freeze):
    bundle,dataset,output,comparison_freeze=map(Path,(bundle,dataset,output,comparison_freeze))
    runtime=EvidenceRuntime(bundle);freeze=json.loads(comparison_freeze.read_text())
    if freeze.get('version')!='evidence-comparison-freeze-1' or {r['name'] for r in freeze['candidates']}!={'A','B','v0.3'}:
        raise ValueError('Both candidates and baseline must be frozen before testing')
    if freeze['evaluation_source_sha256']!=file_sha256(__file__):raise ValueError('Evaluator changed after comparison freeze')
    for candidate in freeze['candidates']:
        if file_sha256(Path(candidate['directory'])/'bundle-manifest.json')!=candidate['artifact_identity']:
            raise ValueError('A frozen comparison artifact changed')
    if freeze['dataset_manifest_sha256']!=file_sha256(dataset/'dataset-manifest.json'):
        raise ValueError('Changed comparison dataset')
    if runtime.artifact_identity not in {m['artifact_identity'] for m in freeze['candidates']}:
        raise ValueError('Candidate was not frozen before evaluation')
    output.mkdir(parents=True,exist_ok=True)
    signature={'artifact_identity':runtime.artifact_identity,'dataset_manifest_sha256':freeze['dataset_manifest_sha256'],
               'comparison_freeze_sha256':file_sha256(comparison_freeze),'evaluation_source_sha256':file_sha256(__file__),
               'inference_source_sha256':inference_hashes()}
    from . import rules_v4
    if freeze['rules_source_sha256']!=file_sha256(rules_v4.__file__):raise ValueError('Comparator changed after comparison freeze')
    signature['rules_source_sha256']=file_sha256(rules_v4.__file__)
    lock=output/'evaluation-started.json'
    if lock.exists():
        if json.loads(lock.read_text())!=signature:raise ValueError('Frozen evaluation changed')
    else:atomic_json(lock,signature)
    if (output/'evaluation-report.json').exists():raise ValueError('Evaluation complete; read saved report')
    _,cases,m=load_partition(dataset,'prototype_evaluation',allowed_splits={'prototype_evaluation'})
    records,timing=collect(runtime,cases,output/'evaluation-predictions.json',m['files_sha256']['prototype_evaluation.jsonl'])
    calibrated=apply_temperatures(records,runtime.temperatures)
    routed=replay(runtime,cases,records,runtime.thresholds,runtime.temperatures)
    rule_rows=[]
    for c in cases:
        result=EvidenceOrchestrator(case_config(c,runtime.config),RulesV4()).evaluate(c['request'],policy_id=c.get('policy_id'))
        differences={k:{'expected':v,'predicted':result.get(k)} for k,v in c['expected'].items() if (set(v)!=set(result.get(k,[])) if k in ('clarification_ids','notice_ids') else v!=result.get(k))}
        rule_rows.append({'case_id':c['case_id'],'family_id':c['family_id'],'primary':c['primary'],'expected':c['expected'],'prediction':result,'differences':differences})
    primary=[r for r in routed if r['primary']]
    report={'scope':'single_author_synthetic_80_20_family_holdout','signature':signature,'cases':len(cases),'families':len(primary),
        'task_mode':runtime.task_mode,'decision_mode':runtime.decision_mode,
        'oracle_policy_question_metrics':question_metrics([r for r in calibrated if r['primary']]),
        'primary_routing':routing_metrics(primary),'all_variant_routing':routing_metrics(routed),
        'primary_predicted_group_identification':predicted_group_identification(primary,runtime.config['identification_options']),
        'rules_primary_routing':routing_metrics([r for r in rule_rows if r['primary']]),
        'rules_version':RULES_VERSION,
        'family_resampled_uncertainty':family_intervals(routed),
        'failure_counts_by_blueprint':dict(Counter(r['blueprint'] for r in routed if r['differences'])),
        'calibration_by_question':{q:calibration_metrics([r for r in calibrated if r['primary'] and r['question_id']==q]) for q in runtime.temperatures},
        'cpu_prediction_runtime':timing,'independently_reviewed':False,'qualifies_cpu_budget':False}
    if runtime.task_mode=='attributes':
        holistic=replay(runtime,cases,records,runtime.thresholds,runtime.temperatures,mode='holistic')
        report['same_weights_holistic_primary_routing']=routing_metrics([r for r in holistic if r['primary']])
        atomic_json(output/'same-weights-holistic-routing.json',holistic)
    atomic_json(output/'routing-results.json',routed);atomic_json(output/'rules-results.json',rule_rows)
    atomic_json(output/'evaluation-report.json',report)
    emit('evidence_evaluation_complete',primary=report['primary_routing'])
    return report
