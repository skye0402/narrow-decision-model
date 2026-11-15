"""Full encoder fine-tuning and standalone CPU weights for v0.3."""
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
from .data_v2 import load_partition
from .model_v2 import differentiable_logits,manifest as write_manifest
from .neural import FrozenDecisionModel,OptionScorer,load_head,save_head,load_local_encoder
from .provenance import file_sha256,verify_files,verify_source,checked_file
from .task_data import question_examples
from .task_model import feature_batches,score_features,prediction_records,question_metrics,permute_options
from .task_runtime import TaskRuntime,inference_hashes as base_hashes
from .training import emit,select_device,state_sha256,ENCODER_FILES

VERSION='task-bundle-3'


def inference_hashes():return {**base_hashes(),'prism_model/model_v3.py':file_sha256(__file__)}


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


class RuntimeV3(TaskRuntime):
 def __init__(self,directory,*,threads=2,require_calibration=True):
  self.directory=Path(directory);self.model,self.config,self.manifest=load_model(directory,threads)
  self.artifact_identity=file_sha256(self.directory/'bundle-manifest.json');self.calibration=None
  if 'calibration.json' in self.manifest['files_sha256']:
   c=json.loads((self.directory/'calibration.json').read_text());self.calibration=c
   if c['version']!='task-calibration-3' or c['model_identity']!=self.manifest['model_identity']:raise ValueError('Wrong v0.3 calibration')
   if c['config_sha256']!=file_sha256(self.directory/'pilot_config.json') or c['inference_source_sha256']!=inference_hashes():raise ValueError('Calibration implementation/config changed')
   if c['independently_reviewed'] is not False or c['deployment_qualified'] is not False:raise ValueError('Unsupported qualification')
   if set(c['temperatures'])!={'material_group','replacement_identification'} or any(type(t) not in (int,float) or not math.isfinite(t) or not .25<=t<=10 for t in c['temperatures'].values()):raise ValueError('Invalid temperatures')
   for key,choices in (('group',self.config['catalog']),('replacement_identification',self.config['identification_options'])):
    if set(c['thresholds'][key])!=set(choices) or any(t is not None and (type(t) not in (int,float) or not math.isfinite(t) or not 0<=t<=1) for t in c['thresholds'][key].values()):raise ValueError('Invalid thresholds')
  elif require_calibration:raise ValueError('Fit and freeze v0.3 calibration first')
  self.temperatures=self.calibration['temperatures'] if self.calibration else {'material_group':1.,'replacement_identification':1.}
  self.thresholds=self.calibration['thresholds'] if self.calibration else {}
 def metadata(self):
  value=super().metadata();value.update(model_name='Requisition Ready v0.3',training_cases=json.loads((self.directory/'training-report.json').read_text())['training']['training_cases']);return value
 def evaluate(self,*args,**kwargs):
  result=super().evaluate(*args,**kwargs);result['model']['name']='Requisition Ready v0.3';return result


def critical_gate(records):
 rows=[r for r in records if r['case_id'].startswith(('EXPOSED-FEEDBACK-','CRITICAL-'))]
 return len(rows)==20 and all(r['choice']==r['label'] for r in rows)


def train(*,data_dir,encoder_dir,warm_head,output_dir,source_root,source_manifest,device='cuda',epochs=6,seed=41):
 data_dir,encoder_dir,warm_head,output_dir=map(Path,(data_dir,encoder_dir,warm_head,output_dir))
 if output_dir.exists():raise ValueError('Use a new full-training output')
 verify_source(source_root,source_manifest)
 config,cases,m=load_partition(data_dir,'train',allowed_splits={'train','development'})
 _,development,_=load_partition(data_dir,'development',allowed_splits={'train','development'})
 if {c['family_id'] for c in cases}&{c['family_id'] for c in development}:raise ValueError('Train/development family overlap')
 if any((data_dir/(s+'.jsonl')).exists() for s in ('calibration','threshold_selection','prototype_evaluation')):raise ValueError('Evaluation labels mounted into optimizer')
 torch.set_num_threads(2);torch.manual_seed(seed);torch.set_float32_matmul_precision('highest')
 selected,hardware=select_device(device,'NVIDIA L4')
 if selected.type=='cuda' and not torch.cuda.is_bf16_supported():raise ValueError('BF16 is required for this GPU experiment')
 encoder,tokenizer,upstream,_=load_local_encoder(encoder_dir);encoder.config.reference_compile=False
 head,_=load_head(warm_head,upstream)
 model=FrozenDecisionModel(encoder,tokenizer,head).to(selected);encoder.requires_grad_(True);head.requires_grad_(True)
 original=state_sha256(encoder);examples=question_examples(cases,config);dev=question_examples(development,config)
 initial=question_metrics(prediction_records(dev,score_features(head,feature_batches(model,dev,8))))
 optimizer=torch.optim.AdamW([{'params':encoder.parameters(),'lr':1e-5},{'params':head.parameters(),'lr':1e-4}],weight_decay=.01)
 batch_size=8;accumulate=4;batches=math.ceil(len(examples)/batch_size);steps=math.ceil(batches/accumulate)*epochs;warmup=max(1,int(.1*steps))
 def rate(step):
  if step<warmup:return (step+1)/warmup
  return .5*(1+math.cos(math.pi*min(1,(step-warmup)/max(1,steps-warmup))))
 scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,rate);sampler=random.Random(seed)
 best=-1.;best_critical=False;best_encoder=None;best_head=None;best_epoch=0;stale=0;history=[];started=perf_counter()
 emit('v3_training_started',device=str(selected),train_cases=len(cases),train_rows=len(examples),development_rows=len(dev),encoder_parameters=sum(p.numel() for p in encoder.parameters()))
 for epoch in range(epochs):
  encoder.eval();head.train();order=list(range(len(examples)));sampler.shuffle(order);loss_sum=0.;optimizer.zero_grad(set_to_none=True)
  for batch,start in enumerate(range(0,len(order),batch_size)):
   chosen=[examples[i] for i in order[start:start+batch_size]];rows=[];targets=[]
   for e in chosen:
    row=e['row']
    if sampler.random()<.5:
     permutation=list(range(len(row['option_ids'])));sampler.shuffle(permutation);row=permute_options(row,permutation)
    rows.append(row);targets.append(row['option_ids'].index(e['label']))
   with torch.autocast(device_type=selected.type,dtype=torch.bfloat16,enabled=selected.type=='cuda'):
    logits=differentiable_logits(model,rows);loss=nn.functional.cross_entropy(logits,torch.tensor(targets,device=selected))
   if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite full-encoder loss')
   group_start=(batch//accumulate)*accumulate;group_size=min(accumulate*batch_size,len(order)-group_start*batch_size)
   (loss*len(rows)/group_size).backward();loss_sum+=float(loss.detach())*len(rows)
   if (batch+1)%accumulate==0 or batch+1==batches:
    nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
   if (batch+1)%128==0:emit('v3_training_batches',epoch=epoch+1,batch=batch+1,total=batches)
  model.eval();records=prediction_records(dev,score_features(head,feature_batches(model,dev,8)))
  metrics=question_metrics(records);score=sum(x['macro_f1'] for x in metrics.values())/len(metrics)
  critical=critical_gate(records) and metrics['material_group']['accuracy']>=.90 and metrics['replacement_identification']['macro_f1']>=.85
  history.append({'epoch':epoch+1,'loss':loss_sum/len(examples),'metrics':metrics,'score':score,'critical_gate':critical})
  emit('v3_epoch',epoch=epoch+1,score=score,critical_gate=critical,group_accuracy=metrics['material_group']['accuracy'],identification_f1=metrics['replacement_identification']['macro_f1'])
  if (critical and not best_critical) or (critical==best_critical and score>best+1e-5):
   best=score;best_critical=critical;best_epoch=epoch+1;stale=0
   best_encoder={k:v.detach().cpu().clone() for k,v in encoder.state_dict().items()};best_head={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
  else:stale+=1
  if stale>=3:break
 encoder.load_state_dict(best_encoder);head.load_state_dict(best_head);model.eval()
 final_records=prediction_records(dev,score_features(head,feature_batches(model,dev,8)))
 reference_examples=dev[:24]+[e for e in dev if e['case_id'].startswith(('EXPOSED-FEEDBACK-','CRITICAL-'))]
 with torch.inference_mode():references=[{'row':e['row'],'logits':model(e['row']).cpu().tolist()} for e in reference_examples]
 changed=state_sha256(encoder)!=original
 if not changed:raise ValueError('Full encoder did not change')
 hardware.update(gradient_precision='bf16 autocast' if selected.type=='cuda' else 'fp32',weights_and_development_precision='fp32, TF32 disabled',
  gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(selected) if selected.type=='cuda' else None)
 report={'kind':'full_encoder_training','scope':'synthetic_prototype','runtime':hardware,'source_manifest':source_manifest,'dataset_manifest':m,
  'training':{'seed':seed,'training_cases':len(cases),'development_cases':len(development),'selected_epoch':best_epoch,'critical_gate':best_critical,
   'encoder_changed':changed,'trained_encoder_parameters':sum(p.numel() for p in encoder.parameters()),'trained_head_parameters':sum(p.numel() for p in head.parameters()),
   'source_commit':source_manifest['source_commit'],'optimizer_partitions':['train'],'selection_partition':'development','history':history,'seconds':perf_counter()-started},
  'initial_development_metrics':initial,'selected_development_metrics':question_metrics(final_records),'upstream_encoder_identity':upstream,
  'warm_head_sha256':file_sha256(warm_head/'head.pt'),'gold_labels_reviewed':False,'calibrated':False,'qualifies_model_quality':False,'qualifies_cpu_budget':False}
 output_dir.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='.full-encoder-',dir=output_dir.parent) as temporary:
  staging=Path(temporary)/'model';(staging/'encoder').mkdir(parents=True)
  for name in ENCODER_FILES:shutil.copyfile(checked_file(encoder_dir,name),staging/'encoder'/name)
  encoder.to('cpu').save_pretrained(staging/'encoder',safe_serialization=False)
  identity=encoder_identity(staging/'encoder',upstream)
  save_head(staging/'head',head,identity,max_tokens=512,training_info=report['training'])
  torch.save({},staging/'adaptation.pt')
  objects=[('adaptation.json',{'mode':'full_encoder_standalone','encoder_identity':identity}),('training-report.json',report),
   ('development-predictions.json',final_records),('source-manifest.json',source_manifest),('dataset-manifest.json',m),('pilot_config.json',config)]
  for name,value in objects:(staging/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
  manifest(staging);cpu,_,_=load_model(staging);checks=[]
  with torch.inference_mode():
   for item in references:
    actual=cpu(item['row']).cpu();expected=torch.tensor(item['logits']);checks.append({'choice_match':int(actual.argmax())==int(expected.argmax()),'max_logit_drift':float((actual-expected).abs().max())})
  if not all(x['choice_match'] for x in checks):raise ValueError('Fresh CPU reload changed a reference/critical choice')
  report['fresh_cpu_reload']={'rows':len(checks),'all_choices_match':True,'max_logit_drift':max(x['max_logit_drift'] for x in checks),'checks':checks}
  (staging/'training-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');manifest(staging);verify_bundle(staging);staging.rename(output_dir)
 emit('v3_training_complete',selected_epoch=best_epoch,critical_gate=best_critical,group_accuracy=report['selected_development_metrics']['material_group']['accuracy'],
  identification_f1=report['selected_development_metrics']['replacement_identification']['macro_f1'],cpu_reload_max_drift=report['fresh_cpu_reload']['max_logit_drift'])
 return report
