"""Evidence-task fine-tuning from the verified v0.3 encoder and head."""
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import tempfile
from time import perf_counter

import torch
from torch import nn
from .calibration import probabilities
from .evidence_data import load_partition,question_examples
from .model_v2 import differentiable_logits,manifest as write_manifest
from .neural import FrozenDecisionModel,OptionScorer,load_head,save_head,load_local_encoder
from .provenance import file_sha256,verify_files,verify_source,checked_file
from collections import Counter
from .model_v3 import load_model as load_parent
from .task_model import feature_batches,score_features,prediction_records,question_metrics,permute_options
from .task_runtime import TaskRuntime,inference_hashes as base_hashes
from .training import emit,select_device,state_sha256,ENCODER_FILES

VERSION='task-bundle-4'


def inference_hashes():
 root=Path(__file__).resolve().parents[1]
 names=('prism_model/model_v4.py','prism_model/evidence_data.py','prism_model/evidence_format.py')
 return {**base_hashes(),**{name:file_sha256(root/name) for name in names}}


def manifest(directory,**extra):return write_manifest(directory,version=VERSION,**extra)


def verify_bundle(directory):
 directory=Path(directory);m=json.loads(checked_file(directory,'bundle-manifest.json').read_text())
 if m.get('version')!=VERSION or m.get('complete') is not True:raise ValueError('Wrong full-finetuned bundle')
 verify_files(directory,m['files_sha256'],exact=True,exclude=('bundle-manifest.json',),
  required=('encoder/pytorch_model.bin','adaptation.json','head/head.pt','head/head.json','pilot_config.json','training-report.json'))
 names=('encoder/pytorch_model.bin','encoder/config.json','encoder/tokenizer.json','encoder/tokenizer_config.json',
  'encoder/special_tokens_map.json','head/head.pt','head/head.json','adaptation.pt','adaptation.json','pilot_config.json')
 identity=hashlib.sha256(json.dumps({k:m['files_sha256'][k] for k in names},sort_keys=True,separators=(',',':')).encode()).hexdigest()
 if m['model_identity']!=identity:raise ValueError('Model identity mismatch')
 return m


def encoder_identity(directory,upstream):
 names=('config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','pytorch_model.bin')
 return {'kind':'fully_finetuned_encoder','upstream_identity':upstream,'files_sha256':{n:file_sha256(Path(directory)/n) for n in names}}


def load_model(directory,threads=2):
 from transformers import AutoModel,AutoTokenizer
 directory=Path(directory);m=verify_bundle(directory);torch.set_num_threads(threads)
 metadata=json.loads((directory/'adaptation.json').read_text());expected=metadata['encoder_identity']
 identity=encoder_identity(directory/'encoder',expected['upstream_identity'])
 if identity!=expected:raise ValueError('Fine-tuned encoder identity changed')
 tokenizer=AutoTokenizer.from_pretrained(directory/'encoder',local_files_only=True,trust_remote_code=False)
 encoder,info=AutoModel.from_pretrained(directory/'encoder',local_files_only=True,trust_remote_code=False,weights_only=True,
  torch_dtype=torch.float32,attn_implementation='sdpa',output_loading_info=True)
 if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise ValueError('Fine-tuned weights not fully restored')
 encoder.config.reference_compile=False;head,hm=load_head(directory/'head',identity)
 return FrozenDecisionModel(encoder,tokenizer,head,max_tokens=hm['max_tokens']).eval(),json.loads((directory/'pilot_config.json').read_text()),m


def critical_summary(records):
 rows=[r for r in records if r['case_id'].startswith(('REVIEW-','EXPOSED-','CRITICAL-'))]
 false_ready=sum(r['question_id']=='replacement_identification' and r['choice']=='satisfied' and r['label']!='satisfied' for r in rows)
 return {'rows':len(rows),'correct':sum(r['choice']==r['label'] for r in rows),'false_raw_ready':false_ready,
         'all_correct':bool(rows) and all(r['choice']==r['label'] for r in rows)}


def train(*,data_dir,warm_model,output_dir,source_root,source_manifest,device='cuda',epochs=4,seed=73,task_mode='attributes'):
 data_dir,warm_model,output_dir=map(Path,(data_dir,warm_model,output_dir))
 if task_mode not in ('holistic','attributes'):raise ValueError('Unknown task mode')
 if output_dir.exists():raise ValueError('Use a new full-training output')
 verify_source(source_root,source_manifest)
 config,cases,m=load_partition(data_dir,'train',allowed_splits={'train','development'})
 _,development,_=load_partition(data_dir,'development',allowed_splits={'train','development'})
 if {c['family_id'] for c in cases}&{c['family_id'] for c in development}:raise ValueError('Train/development family overlap')
 if any((data_dir/(s+'.jsonl')).exists() for s in ('calibration','threshold_selection','prototype_evaluation')):raise ValueError('Evaluation labels mounted into optimizer')
 torch.set_num_threads(2);torch.manual_seed(seed);torch.set_float32_matmul_precision('highest')
 selected,hardware=select_device(device,'NVIDIA L4')
 if selected.type=='cuda' and not torch.cuda.is_bf16_supported():raise ValueError('BF16 is required for this GPU experiment')
 model,parent_config,parent_manifest=load_parent(warm_model)
 if parent_config!=config:raise ValueError('Warm model and task config differ')
 upstream=json.loads((warm_model/'adaptation.json').read_text())['encoder_identity']['upstream_identity']
 model=model.to(selected);encoder,tokenizer,head=model.encoder,model.tokenizer,model.head
 encoder.requires_grad_(True);head.requires_grad_(True)
 original=state_sha256(encoder);examples=question_examples(cases,config,attributes=task_mode=='attributes');dev=question_examples(development,config,attributes=task_mode=='attributes')
 counts=Counter((e['row']['question_id'],e['label']) for e in examples)
 totals=Counter(e['row']['question_id'] for e in examples)
 labels={q:len({label for question,label in counts if question==q}) for q in totals}
 loss_weights={key:min(4.,max(.5,math.sqrt(totals[key[0]]/(labels[key[0]]*count)))) for key,count in counts.items()}
 initial=question_metrics(prediction_records(dev,score_features(head,feature_batches(model,dev,8))))
 optimizer=torch.optim.AdamW([{'params':encoder.parameters(),'lr':1e-5},{'params':head.parameters(),'lr':1e-4}],weight_decay=.01)
 target_rows=len(question_examples(cases,config,attributes=True))
 batch_size=8;accumulate=4;batches=math.ceil(target_rows/batch_size);steps=math.ceil(batches/accumulate)*epochs;warmup=max(1,int(.1*steps))
 def rate(step):
  if step<warmup:return (step+1)/warmup
  return .5*(1+math.cos(math.pi*min(1,(step-warmup)/max(1,steps-warmup))))
 scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,rate);sampler=random.Random(seed)
 best=-1.;best_rank=None;best_critical=False;best_encoder=None;best_head=None;best_epoch=0;stale=0;history=[];started=perf_counter()
 emit('v4_training_started',device=str(selected),train_cases=len(cases),train_rows=len(examples),development_rows=len(dev),encoder_parameters=sum(p.numel() for p in encoder.parameters()))
 for epoch in range(epochs):
  encoder.eval();head.train();order=list(range(len(examples)))
  if len(order)<target_rows:order+=sampler.choices(range(len(examples)),k=target_rows-len(order))
  sampler.shuffle(order);loss_sum=0.;optimizer.zero_grad(set_to_none=True)
  for batch,start in enumerate(range(0,len(order),batch_size)):
   chosen=[examples[i] for i in order[start:start+batch_size]];rows=[];targets=[]
   for e in chosen:
    row=e['row']
    rows.append(row);targets.append(row['option_ids'].index(e['label']))
   with torch.autocast(device_type=selected.type,dtype=torch.bfloat16,enabled=selected.type=='cuda'):
    logits=differentiable_logits(model,rows)
    losses=nn.functional.cross_entropy(logits,torch.tensor(targets,device=selected),reduction='none')
    weights=torch.tensor([loss_weights[(e['row']['question_id'],e['label'])] for e in chosen],device=selected)
    loss=(losses*weights).mean()
   if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite full-encoder loss')
   group_start=(batch//accumulate)*accumulate;group_size=min(accumulate*batch_size,len(order)-group_start*batch_size)
   (loss*len(rows)/group_size).backward();loss_sum+=float(loss.detach())*len(rows)
   if (batch+1)%accumulate==0 or batch+1==batches:
    nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
   if (batch+1)%128==0:emit('v4_training_batches',epoch=epoch+1,batch=batch+1,total=batches)
  model.eval();records=prediction_records(dev,score_features(head,feature_batches(model,dev,8)))
  metrics=question_metrics(records);score=sum(x['macro_f1'] for x in metrics.values())/len(metrics)
  critical_cases=critical_summary(records)
  eligible=metrics['material_group']['accuracy']>=.90 and metrics['replacement_identification']['macro_f1']>=.85
  critical=critical_cases['all_correct'] and eligible
  rank=(eligible,-critical_cases['false_raw_ready'],critical_cases['correct']/max(1,critical_cases['rows']),score)
  history.append({'epoch':epoch+1,'loss':loss_sum/len(order),'metrics':metrics,'score':score,'critical_gate':critical,'critical_cases':critical_cases})
  emit('v4_epoch',epoch=epoch+1,score=score,critical_gate=critical,group_accuracy=metrics['material_group']['accuracy'],identification_f1=metrics['replacement_identification']['macro_f1'])
  if best_rank is None or rank>best_rank:
   best_rank=rank
   best=score;best_critical=critical;best_epoch=epoch+1;stale=0
   best_encoder={k:v.detach().cpu().clone() for k,v in encoder.state_dict().items()};best_head={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
  else:stale+=1
  if stale>=3:break
 encoder.load_state_dict(best_encoder);head.load_state_dict(best_head);model.eval()
 final_records=prediction_records(dev,score_features(head,feature_batches(model,dev,8)))
 reference_examples=dev[:24]+[e for e in dev if e['case_id'].startswith(('REVIEW-','EXPOSED-','CRITICAL-'))]
 with torch.inference_mode():references=[{'row':e['row'],'logits':model(e['row']).cpu().tolist()} for e in reference_examples]
 changed=state_sha256(encoder)!=original
 if not changed:raise ValueError('Full encoder did not change')
 hardware.update(gradient_precision='bf16 autocast' if selected.type=='cuda' else 'fp32',weights_and_development_precision='fp32, TF32 disabled',
  gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(selected) if selected.type=='cuda' else None)
 report={'kind':'evidence_full_encoder_training','task_mode':task_mode,'parent_model_identity':parent_manifest['model_identity'],'parent_artifact_identity':file_sha256(warm_model/'bundle-manifest.json'),'scope':'synthetic_prototype','runtime':hardware,'source_manifest':source_manifest,'dataset_manifest':m,
  'training':{'seed':seed,'training_cases':len(cases),'development_cases':len(development),'selected_epoch':best_epoch,'critical_gate':best_critical,
   'unique_training_question_rows':len(examples),'optimization_rows_per_epoch':target_rows,'class_weighting':'inverse_sqrt_per_question_clipped_0.5_4.0',
   'encoder_changed':changed,'trained_encoder_parameters':sum(p.numel() for p in encoder.parameters()),'trained_head_parameters':sum(p.numel() for p in head.parameters()),
   'source_commit':source_manifest['source_commit'],'optimizer_partitions':['train'],'selection_partition':'development','history':history,'seconds':perf_counter()-started},
  'initial_development_metrics':initial,'selected_development_metrics':question_metrics(final_records),'upstream_encoder_identity':upstream,
  'warm_head_sha256':file_sha256(warm_model/'head/head.pt'),'gold_labels_reviewed':False,'calibrated':False,'qualifies_model_quality':False,'qualifies_cpu_budget':False}
 output_dir.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='.full-encoder-',dir=output_dir.parent) as temporary:
  staging=Path(temporary)/'model';(staging/'encoder').mkdir(parents=True)
  for name in ENCODER_FILES:shutil.copyfile(checked_file(warm_model/'encoder',name),staging/'encoder'/name)
  encoder.to('cpu').save_pretrained(staging/'encoder',safe_serialization=False)
  identity=encoder_identity(staging/'encoder',upstream)
  save_head(staging/'head',head,identity,max_tokens=512,training_info=report['training'])
  torch.save({},staging/'adaptation.pt')
  objects=[('adaptation.json',{'mode':'full_encoder_evidence','task_mode':task_mode,'encoder_identity':identity}),('training-report.json',report),
   ('development-predictions.json',final_records),('source-manifest.json',source_manifest),('dataset-manifest.json',m),('pilot_config.json',config)]
  for name,value in objects:(staging/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
  manifest(staging);cpu,_,_=load_model(staging);checks=[]
  with torch.inference_mode():
   for item in references:
    actual=cpu(item['row']).cpu();expected=torch.tensor(item['logits']);checks.append({'choice_match':int(actual.argmax())==int(expected.argmax()),'max_logit_drift':float((actual-expected).abs().max())})
  if not all(x['choice_match'] for x in checks):raise ValueError('Fresh CPU reload changed a reference/critical choice')
  report['fresh_cpu_reload']={'rows':len(checks),'all_choices_match':True,'max_logit_drift':max(x['max_logit_drift'] for x in checks),'checks':checks}
  (staging/'training-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');manifest(staging);verify_bundle(staging);staging.rename(output_dir)
 emit('v4_training_complete',selected_epoch=best_epoch,critical_gate=best_critical,group_accuracy=report['selected_development_metrics']['material_group']['accuracy'],
  identification_f1=report['selected_development_metrics']['replacement_identification']['macro_f1'],cpu_reload_max_drift=report['fresh_cpu_reload']['max_logit_drift'])
 return report
