"""Partition-controlled calibration and one frozen evaluation of v0.3."""
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import tempfile
from time import perf_counter

import torch
from .calibration import apply_temperatures,fit_temperatures
from .data_v2 import load_partition,PRODUCTS
from .metrics import wilson,calibration_metrics
from .model_v3 import RuntimeV3,manifest,verify_bundle,inference_hashes
from .neural import OptionScorer
from .orchestrator import Orchestrator
from .provenance import file_sha256
from .rules_v3 import RulesV3,VERSION as RULES_VERSION
from .task_data import case_config,question_examples
from .task_iteration import atomic_json,replay,routing_metrics,family_intervals,predicted_group_identification
from .task_model import prediction_records,question_metrics
from .training import emit
from . import offline_v3


def collect(runtime,cases,path,partition_sha):
 path=Path(path);signature={'model_identity':runtime.manifest['model_identity'],'partition_sha256':partition_sha,'inference_source_sha256':inference_hashes(),
  'prediction_source_sha256':file_sha256(__file__),'offline_source_sha256':file_sha256(offline_v3.__file__),
  'shared_evaluation_sha256':file_sha256(replay.__code__.co_filename)}
 examples=question_examples(cases,runtime.config);records=[]
 if path.exists():
  cached=json.loads(path.read_text())
  if cached['signature']!=signature:raise ValueError('Prediction cache identity changed')
  records=cached['records']
  if len(records)>len(examples):raise ValueError('Cache has extra records')
  for record,example in zip(records,examples):
   if record['input_sha256']!=example['row']['text_sha256'] or record['target']!=example['target']:raise ValueError('Cache data changed')
 started=perf_counter();initial=len(records)
 scored=offline_v3.predictions(runtime.directory,(e['row'] for e in examples[initial:]),workers=3)
 for index,(values,random_values) in enumerate(scored,start=initial):
  e=examples[index]
  record=prediction_records([e],[values])[0];record['random_head_logits']=random_values;records.append(record)
  if (index+1)%128==0 or index+1==len(examples):
   atomic_json(path,{'signature':signature,'records':records,'complete':index+1==len(examples)})
   emit('v3_cpu_predictions',rows=index+1,total=len(examples),partition=path.stem)
 return records,{'new_rows':len(records)-initial,'seconds':perf_counter()-started,'total_rows':len(records),'batch_size':1,'offline_workers':3,'torch_threads_per_worker':2}


def select_gate(records,outcome,precision,lower,support):
 rows=[r for r in records if r['primary'] and r['choice']==outcome]
 if len({r['family_id'] for r in rows})!=len(rows):raise ValueError('Gate support repeats a family')
 candidates=sorted({*[i/100 for i in range(101)],*[r['confidence'] for r in rows]})
 for cutoff in candidates:
  selected=[r for r in rows if r['confidence']>=cutoff];interval=wilson(sum(r['label']==outcome for r in selected),len(selected))
  if len(selected)>=support and interval['estimate']>=precision and interval['lower']>=lower:
   return {'threshold':cutoff,'support':len(selected),'precision':interval,'target_precision':precision,'minimum_lower_bound':lower}
 return {'threshold':None,'available_predictions':len(rows),'reason':'insufficient_support_or_precision','minimum_support':support}


def fit(raw,dataset,output,cache):
 raw,dataset,output,cache=map(Path,(raw,dataset,output,cache))
 if output.exists():raise ValueError('Use a new calibrated bundle directory')
 runtime=RuntimeV3(raw,require_calibration=False)
 training=json.loads((raw/'training-report.json').read_text())
 if training['dataset_manifest']['parent_dataset_manifest_sha256']!=file_sha256(dataset/'dataset-manifest.json'):raise ValueError('Dataset differs from training cohort')
 _,cases,m=load_partition(dataset,'calibration',allowed_splits={'calibration'})
 calibration,cal_time=collect(runtime,cases,cache/'calibration-predictions.json',m['files_sha256']['calibration.jsonl'])
 temperatures,diagnostics=fit_temperatures(calibration);emit('v3_temperatures',values=temperatures)
 _,cases,_=load_partition(dataset,'threshold_selection',allowed_splits={'threshold_selection'})
 records,threshold_time=collect(runtime,cases,cache/'threshold-predictions.json',m['files_sha256']['threshold_selection.jsonl'])
 calibrated=apply_temperatures(records,temperatures);gates={'group':{},'replacement_identification':{}};gate_diagnostics={}
 for question in ('material_group','replacement_identification'):
  rows=[r for r in calibrated if r['question_id']==question];key='group' if question=='material_group' else question;gate_diagnostics[key]={}
  for option in sorted({key for r in rows for key in r['option_ids']}):
   settings=(.99,.95,50) if key=='group' else (.99,.95,100) if option=='satisfied' else (.97,.90,50)
   chosen=select_gate(rows,option,*settings);gates[key][option]=chosen['threshold'];gate_diagnostics[key][option]=chosen
 primary=[c for c in cases if c['primary']]
 routed=replay(primary,runtime.config,records,gates,temperatures,model=runtime.model);joint=routing_metrics(routed)
 enabled=joint['predicted_ready']>=60 and joint['false_ready']==0
 initial=gates['replacement_identification']['satisfied']
 if not enabled and initial is not None:
  choices=sorted({r['confidence'] for r in calibrated if r['primary'] and r['question_id']=='replacement_identification' and r['choice']=='satisfied' and r['confidence']>=initial})
  for value in choices:
   selected=[r for r in calibrated if r['primary'] and r['question_id']=='replacement_identification' and r['choice']=='satisfied' and r['confidence']>=value]
   interval=wilson(sum(r['label']=='satisfied' for r in selected),len(selected))
   if len(selected)<100 or interval['estimate']<.99 or interval['lower']<.95:continue
   trial={key:dict(values) for key,values in gates.items()};trial['replacement_identification']['satisfied']=value
   trial_rows=replay(primary,runtime.config,records,trial,temperatures,model=runtime.model);metrics=routing_metrics(trial_rows)
   if metrics['predicted_ready']>=60 and metrics['false_ready']==0:gates=trial;joint=metrics;routed=trial_rows;enabled=True;break
 if not enabled:
  gates['replacement_identification']['satisfied']=None
  routed=replay(primary,runtime.config,records,gates,temperatures,model=runtime.model);joint=routing_metrics(routed)
 value={'version':'task-calibration-3','scope':'synthetic_prototype','created_at':datetime.now(timezone.utc).isoformat(),
  'model_identity':runtime.manifest['model_identity'],'config_sha256':file_sha256(raw/'pilot_config.json'),
  'inference_source_sha256':inference_hashes(),'fitting_source_sha256':file_sha256(__file__),'offline_source_sha256':file_sha256(offline_v3.__file__),
  'temperatures':temperatures,'thresholds':gates,'gate_diagnostics':gate_diagnostics,
  'calibration_diagnostics':diagnostics,'source_dataset_manifest_sha256':file_sha256(dataset/'dataset-manifest.json'),
  'fit_partitions':['calibration','threshold_selection'],'primary_variant':0,'joint_threshold_primary_diagnostics':joint,
  'joint_ready_gate_enabled':enabled,'runtime':{'calibration':cal_time,'threshold':threshold_time},
  'independently_reviewed':False,'deployment_qualified':False}
 atomic_json(cache/'calibration-fit.json',value);atomic_json(cache/'threshold-routing.json',routed)
 output.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='.v2-freeze-',dir=output.parent) as temporary:
  staging=Path(temporary)/'model';shutil.copytree(raw,staging);atomic_json(staging/'calibration.json',value)
  manifest(staging,calibrated_for_synthetic_prototype=True);RuntimeV3(staging);staging.rename(output)
 emit('v3_calibration_frozen',ready_gate_enabled=enabled,joint=joint);return value


def evaluate(bundle,dataset,output):
 bundle,dataset,output=map(Path,(bundle,dataset,output));runtime=RuntimeV3(bundle);output.mkdir(parents=True,exist_ok=True)
 from . import rules_v3
 signature={'artifact_identity':runtime.artifact_identity,'dataset_manifest_sha256':file_sha256(dataset/'dataset-manifest.json'),
  'evaluation_source_sha256':file_sha256(__file__),'rules_source_sha256':file_sha256(rules_v3.__file__),'offline_source_sha256':file_sha256(offline_v3.__file__),
  'shared_evaluation_sha256':file_sha256(replay.__code__.co_filename)}
 lock=output/'evaluation-started.json'
 if lock.exists():
  if json.loads(lock.read_text())['signature']!=signature:raise ValueError('Frozen evaluation inputs changed')
 else:atomic_json(lock,{'signature':signature,'started_at':datetime.now(timezone.utc).isoformat()})
 if (output/'evaluation-report.json').exists():raise ValueError('Evaluation complete; read its report')
 _,cases,m=load_partition(dataset,'prototype_evaluation',allowed_splits={'prototype_evaluation'})
 records,timing=collect(runtime,cases,output/'evaluation-predictions.json',m['files_sha256']['prototype_evaluation.jsonl'])
 calibrated=apply_temperatures(records,runtime.temperatures)
 routed=replay(cases,runtime.config,records,runtime.thresholds,runtime.temperatures,model=runtime.model)
 rule_rows=[]
 for case in cases:
  p=Orchestrator(case_config(case,runtime.config),RulesV3()).evaluate(case['request'])
  diffs={key:{'expected':value,'predicted':p.get(key)} for key,value in case['expected'].items() if (set(value)!=set(p.get(key,[])) if key in ('clarification_ids','notice_ids') else value!=p.get(key))}
  rule_rows.append({'case_id':case['case_id'],'family_id':case['family_id'],'primary':case['primary'],'expected':case['expected'],
    'prediction':p,'differences':diffs})
 primary=[r for r in calibrated if r['primary']]
 random_records=apply_temperatures([{**r,'logits':r['random_head_logits']} for r in records if r['primary']],{'material_group':1.,'replacement_identification':1.})
 report={'scope':'single_author_synthetic_writing_holdout','signature':signature,'cases':len(cases),'families':len({c['family_id'] for c in cases}),
  'model_oracle_policy_questions':question_metrics(primary),'random_head_on_selected_encoder':question_metrics(random_records),
  'random_baseline_note':'Random shared scorer on the fully fine-tuned encoder. This measures a random head, not a fully unadapted model.',
  'model_primary_routing':routing_metrics([r for r in routed if r['primary']]),'model_all_variant_routing':routing_metrics(routed),
  'rules_primary_routing':routing_metrics([r for r in rule_rows if r['primary']]),'rules_version':RULES_VERSION,
  'model_primary_predicted_group_identification':predicted_group_identification([r for r in routed if r['primary']],runtime.config['identification_options']),
  'family_resampled_uncertainty':family_intervals(routed),
  'calibration_by_question':{q:calibration_metrics([r for r in primary if r['question_id']==q]) for q in ('material_group','replacement_identification')},
  'failure_counts_by_blueprint':dict(Counter(r['blueprint'] for r in routed if r['differences'])),
  'cpu_prediction_runtime':timing,'independently_reviewed':False,'qualifies_cpu_budget':False}
 atomic_json(output/'routing-results.json',routed);atomic_json(output/'rules-results.json',rule_rows);atomic_json(output/'evaluation-report.json',report)
 emit('v3_evaluation_complete',model=report['model_primary_routing'],rules=report['rules_primary_routing']);return report
