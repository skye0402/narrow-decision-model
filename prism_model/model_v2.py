"""Versioned partial-encoder adaptation and CPU loading; leaves v0.1 intact."""
import copy
import hashlib
import json
from pathlib import Path
import platform
import random
import shutil
import tempfile
from time import perf_counter

import torch
from torch import nn

from .data_v2 import load_partition
from .neural import FrozenDecisionModel,OptionScorer,load_local_encoder,load_head,save_head,tokenize_row
from .provenance import checked_file,file_sha256,verify_files,verify_source
from .task_data import question_examples
from .task_model import (augment_options,feature_batches,pack_features,packed_logits,prediction_records,
    question_metrics,score_features,permute_options)
from .task_runtime import TaskRuntime,inference_hashes as original_inference_hashes
from .training import ENCODER_FILES,emit,select_device,synchronize,state_sha256

VERSION='task-bundle-2'
ROOT=Path(__file__).resolve().parents[1]


def inference_hashes():
 return {**original_inference_hashes(), 'prism_model/model_v2.py':file_sha256(__file__)}


def digest_state(state):
 h=hashlib.sha256()
 for key,value in sorted(state.items()):
  h.update(key.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
  h.update(value.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()


def manifest(directory,**extra):
 directory=Path(directory)
 files={p.relative_to(directory).as_posix():file_sha256(p) for p in sorted(directory.rglob('*')) if p.is_file() and p.name!='bundle-manifest.json'}
 identity_names=('encoder/pytorch_model.bin','encoder/config.json','encoder/tokenizer.json','encoder/tokenizer_config.json',
   'encoder/special_tokens_map.json','head/head.pt','head/head.json','adaptation.pt','adaptation.json','pilot_config.json')
 identity=hashlib.sha256(json.dumps({k:files[k] for k in identity_names},sort_keys=True,separators=(',',':')).encode()).hexdigest()
 result={'version':VERSION,'scope':'synthetic_prototype','complete':True,'model_identity':identity,
  'calibrated_for_synthetic_prototype':False,'qualifies_model_quality':False,'qualifies_cpu_budget':False,
  **extra,'files_sha256':files}
 (directory/'bundle-manifest.json').write_text(json.dumps(result,indent=2)+'\n');return result


def verify_bundle(directory):
 directory=Path(directory);result=json.loads(checked_file(directory,'bundle-manifest.json').read_text())
 if result['version']!=VERSION or result['complete'] is not True:raise ValueError('Invalid v0.2 bundle')
 verify_files(directory,result['files_sha256'],exact=True,exclude=('bundle-manifest.json',),
  required=('head/head.pt','adaptation.pt','adaptation.json','pilot_config.json','training-report.json','source-manifest.json'))
 identity_names=('encoder/pytorch_model.bin','encoder/config.json','encoder/tokenizer.json','encoder/tokenizer_config.json',
   'encoder/special_tokens_map.json','head/head.pt','head/head.json','adaptation.pt','adaptation.json','pilot_config.json')
 identity=hashlib.sha256(json.dumps({k:result['files_sha256'][k] for k in identity_names},sort_keys=True,separators=(',',':')).encode()).hexdigest()
 if identity!=result['model_identity']:raise ValueError('Model identity mismatch')
 return result


def load_model(directory,threads=2):
 directory=Path(directory);bundle=verify_bundle(directory);torch.set_num_threads(threads)
 encoder,tokenizer,identity,_=load_local_encoder(directory/'encoder');encoder.config.reference_compile=False
 head,metadata=load_head(directory/'head',identity)
 adaptation=json.loads((directory/'adaptation.json').read_text())
 state=torch.load(directory/'adaptation.pt',map_location='cpu',weights_only=True)
 expected={k for k in encoder.state_dict() if any(k.startswith(f'layers.{i}.') for i in adaptation['layer_indices'])}
 if set(state)!=expected or adaptation['state_sha256']!=digest_state(state):raise ValueError('Invalid adapted layer state')
 if any(not bool(torch.isfinite(value).all()) for value in state.values()):raise ValueError('Nonfinite adapted weights')
 result=encoder.load_state_dict(state,strict=False)
 if result.unexpected_keys:raise ValueError('Unexpected adapted layer key')
 model=FrozenDecisionModel(encoder,tokenizer,head,max_tokens=metadata['max_tokens']).eval()
 return model,json.loads((directory/'pilot_config.json').read_text()),bundle


class RuntimeV2(TaskRuntime):
 def __init__(self,directory,*,threads=2,require_calibration=True):
  self.directory=Path(directory);self.model,self.config,self.manifest=load_model(directory,threads)
  self.artifact_identity=file_sha256(self.directory/'bundle-manifest.json');self.calibration=None
  if 'calibration.json' in self.manifest['files_sha256']:
   self.calibration=json.loads((self.directory/'calibration.json').read_text());c=self.calibration
   if c['version']!='task-calibration-2' or c['model_identity']!=self.manifest['model_identity']:raise ValueError('Wrong calibration identity')
   if c['config_sha256']!=file_sha256(self.directory/'pilot_config.json') or c['inference_source_sha256']!=inference_hashes():raise ValueError('Calibration source/config changed')
   if c['independently_reviewed'] is not False or c['deployment_qualified'] is not False:raise ValueError('Unsupported qualification')
   import math
   if set(c['temperatures'])!={'material_group','replacement_identification'} or any(type(t) not in (int,float) or not math.isfinite(t) or not .25<=t<=10 for t in c['temperatures'].values()):raise ValueError('Invalid temperatures')
   for key,choices in (('group',self.config['catalog']),('replacement_identification',self.config['identification_options'])):
    if set(c['thresholds'][key])!=set(choices) or any(t is not None and (type(t) not in (int,float) or not math.isfinite(t) or not 0<=t<=1) for t in c['thresholds'][key].values()):raise ValueError('Invalid thresholds')
  elif require_calibration:raise ValueError('Fit and freeze v0.2 calibration before serving')
  self.temperatures=self.calibration['temperatures'] if self.calibration else {'material_group':1.,'replacement_identification':1.}
  self.thresholds=self.calibration['thresholds'] if self.calibration else {}

 def metadata(self):
  value=super().metadata();value.update(model_name='Requisition Ready v0.2',training_cases=json.loads((self.directory/'training-report.json').read_text())['training']['training_cases'])
  return value

 def evaluate(self,*args,**kwargs):
  value=super().evaluate(*args,**kwargs);value['model']['name']='Requisition Ready v0.2';return value


def trainable_layers(encoder,count=4):
 if count<1 or count>len(encoder.layers):raise ValueError('Invalid layer count')
 encoder.requires_grad_(False);indices=list(range(len(encoder.layers)-count,len(encoder.layers)))
 for index in indices:encoder.layers[index].requires_grad_(True)
 return indices


def differentiable_logits(model,rows):
 encoded=[tokenize_row(model.tokenizer,row,model.max_tokens) for row in rows]
 device=next(model.parameters()).device;width=max(x[0]['input_ids'].shape[1] for x in encoded)
 inputs={}
 for key in encoded[0][0]:
  pad=model.tokenizer.pad_token_id if key=='input_ids' else 0
  tensor=torch.full((len(rows),width),pad,dtype=encoded[0][0][key].dtype,device=device)
  for i,(tokens,_) in enumerate(encoded):tensor[i,:tokens[key].shape[1]]=tokens[key][0].to(device)
  inputs[key]=tensor
 hidden=model.encoder(**inputs).last_hidden_state
 count=max(len(r['option_ids']) for r in rows);all_scores=[]
 for i,(tokens,mask) in enumerate(encoded):
  weights=mask.to(device=device,dtype=hidden.dtype)
  options=weights@hidden[i,:mask.shape[1]]/weights.sum(dim=1,keepdim=True)
  scores=model.head(hidden[i,0],options)
  all_scores.append(nn.functional.pad(scores,(0,count-len(scores)),value=torch.finfo(scores.dtype).min))
 return torch.stack(all_scores)


def _metric_score(metrics):return sum(x['macro_f1'] for x in metrics.values())/len(metrics)


def feedback_gate(predictions):
 rows=[r for r in predictions if r['question_id']=='material_group' and r['case_id'].startswith('EXPOSED-FEEDBACK-')]
 return len(rows)==4 and all(r['choice']==r['label'] for r in rows)


def train(*,data_dir,encoder_dir,output_dir,source_root,source_manifest,device='cuda',seed=23,
          head_epochs=40,adapt_epochs=6,layers=4,batch_size=8):
 data_dir,encoder_dir,output_dir=map(Path,(data_dir,encoder_dir,output_dir))
 if output_dir.exists():raise ValueError('Use a new output')
 verify_source(source_root,source_manifest)
 config,cases,data_manifest=load_partition(data_dir,'train',allowed_splits={'train','development'})
 _,development,_=load_partition(data_dir,'development',allowed_splits={'train','development'})
 if {c['family_id'] for c in cases}&{c['family_id'] for c in development}:raise ValueError('Family leakage')
 if any((data_dir/(s+'.jsonl')).exists() for s in ('calibration','threshold_selection','prototype_evaluation')):raise ValueError('Non-optimizer labels mounted')
 torch.set_num_threads(2);torch.manual_seed(seed);torch.set_float32_matmul_precision('high')
 selected,hardware=select_device(device,'NVIDIA L4')
 encoder,tokenizer,identity,loading=load_local_encoder(encoder_dir);encoder.config.reference_compile=False
 model=FrozenDecisionModel(encoder,tokenizer,OptionScorer(encoder.config.hidden_size),max_tokens=512).to(selected)
 original=state_sha256(encoder);train_examples=question_examples(cases,config);dev=question_examples(development,config)
 augmented=augment_options(train_examples,seed)
 started=perf_counter()
 def progress(done,total):
  if done%512==0 or done==total:emit('v2_features',rows=done,total=total)
 training_features=feature_batches(model,augmented,batch_size,progress);dev_features=feature_batches(model,dev,batch_size,progress)
 untrained=question_metrics(prediction_records(dev,score_features(model.head,dev_features)))
 packed,valid=pack_features(training_features);targets=torch.tensor([e['target'] for e in augmented],device=selected)
 optimizer=torch.optim.AdamW(model.head.parameters(),lr=.001,weight_decay=.01)
 rng=torch.Generator().manual_seed(seed);history=[];best=-1.;best_head=None;stale=0;baseline_epoch=0;best_feedback=False
 for epoch in range(head_epochs):
  model.head.train();order=torch.randperm(len(augmented),generator=rng).tolist();loss_sum=0.
  for start in range(0,len(order),64):
   indices=torch.tensor(order[start:start+64],device=selected);optimizer.zero_grad(set_to_none=True)
   logits=packed_logits(model.head,packed[indices],valid[indices]);loss=nn.functional.cross_entropy(logits,targets[indices])
   if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite baseline loss')
   loss.backward();nn.utils.clip_grad_norm_(model.head.parameters(),1.);optimizer.step();loss_sum+=float(loss.detach())*len(indices)
  predictions=prediction_records(dev,score_features(model.head,dev_features));metrics=question_metrics(predictions);score=_metric_score(metrics)
  feedback=feedback_gate(predictions) and metrics['material_group']['accuracy']>=.90 and metrics['replacement_identification']['macro_f1']>=.85
  history.append({'stage':'head_only','epoch':epoch+1,'loss':loss_sum/len(augmented),'metrics':metrics,'score':score,'exposed_feedback_gate':feedback})
  emit('v2_head_epoch',epoch=epoch+1,score=score,group_accuracy=metrics['material_group']['accuracy'],identification_f1=metrics['replacement_identification']['macro_f1'],feedback_gate=feedback)
  if (feedback and not best_feedback) or (feedback==best_feedback and score>best+1e-5):best=score;best_feedback=feedback;baseline_epoch=epoch+1;best_head={k:v.detach().cpu().clone() for k,v in model.head.state_dict().items()};stale=0
  else:stale+=1
  if stale>=6:break
 model.head.load_state_dict(best_head);baseline_metrics=question_metrics(prediction_records(dev,score_features(model.head,dev_features)))
 baseline_seconds=perf_counter()-started
 del packed,valid,training_features,dev_features,optimizer
 if selected.type=='cuda':torch.cuda.empty_cache()
 indices=trainable_layers(encoder,layers);prefixes=tuple(f'layers.{i}.' for i in indices)
 baseline_layers={k:v.detach().cpu().clone() for k,v in encoder.state_dict().items() if k.startswith(prefixes)}
 frozen_hash=digest_state({k:v for k,v in encoder.state_dict().items() if not k.startswith(prefixes)})
 best_layers=baseline_layers;chosen='head_only';best_adapt_epoch=0;stale=0
 optimizer=torch.optim.AdamW([{'params':[p for p in encoder.parameters() if p.requires_grad],'lr':1e-5},
                             {'params':model.head.parameters(),'lr':1e-4}],weight_decay=.01)
 started=perf_counter();sampler=random.Random(seed)
 for epoch in range(adapt_epochs):
  model.head.train();encoder.eval();order=list(range(len(train_examples)));sampler.shuffle(order)
  optimizer.zero_grad(set_to_none=True);loss_sum=0.;batches=(len(order)+batch_size-1)//batch_size
  for batch,start in enumerate(range(0,len(order),batch_size)):
   selected_examples=[train_examples[i] for i in order[start:start+batch_size]];rows=[];labels=[]
   for example in selected_examples:
    row=example['row']
    if sampler.random()<.5:
     permutation=list(range(len(row['option_ids'])));sampler.shuffle(permutation);row=permute_options(row,permutation)
    rows.append(row);labels.append(row['option_ids'].index(example['label']))
   logits=differentiable_logits(model,rows);loss=nn.functional.cross_entropy(logits,torch.tensor(labels,device=selected))
   if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite adaptation loss')
   group_start=(batch//4)*4;accumulation=min(4,batches-group_start)
   (loss/accumulation).backward();loss_sum+=float(loss.detach())*len(rows)
   if (batch+1)%4==0 or batch+1==batches:
    nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);optimizer.step();optimizer.zero_grad(set_to_none=True)
   if (batch+1)%128==0:emit('v2_adapt_batches',epoch=epoch+1,batch=batch+1,total=batches)
  model.eval();features=feature_batches(model,dev,batch_size)
  dev_predictions=prediction_records(dev,score_features(model.head,features));metrics=question_metrics(dev_predictions);score=_metric_score(metrics);del features
  feedback=feedback_gate(dev_predictions) and metrics['material_group']['accuracy']>=.90 and metrics['replacement_identification']['macro_f1']>=.85
  history.append({'stage':'last_layers','epoch':epoch+1,'loss':loss_sum/len(train_examples),'metrics':metrics,'score':score,'exposed_feedback_gate':feedback})
  emit('v2_adapt_epoch',epoch=epoch+1,score=score,group_accuracy=metrics['material_group']['accuracy'],identification_f1=metrics['replacement_identification']['macro_f1'])
  if (feedback and not best_feedback) or (feedback==best_feedback and score>best+1e-5):
   best=score;best_feedback=feedback;best_head={k:v.detach().cpu().clone() for k,v in model.head.state_dict().items()}
   best_layers={k:v.detach().cpu().clone() for k,v in encoder.state_dict().items() if k.startswith(prefixes)}
   chosen='last_four_layers';best_adapt_epoch=epoch+1;stale=0
  else:stale+=1
  if stale>=2:break
 adaptation_seconds=perf_counter()-started
 if digest_state({k:v for k,v in encoder.state_dict().items() if not k.startswith(prefixes)})!=frozen_hash:raise ValueError('Frozen lower layers changed')
 model.head.load_state_dict(best_head);encoder.load_state_dict(best_layers,strict=False);model.eval()
 selected_encoder_unchanged=state_sha256(encoder)==original
 predictions=prediction_records(dev,score_features(model.head,feature_batches(model,dev,batch_size)))
 reference=[]
 with torch.inference_mode():
  for example in dev[:24]:reference.append({'row':example['row'],'logits':model(example['row']).cpu().tolist()})
 hardware.update(torch=torch.__version__,platform=platform.platform(),precision='fp32; TF32 allowed during CUDA training',
   gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(selected) if selected.type=='cuda' else None)
 report={'kind':'task_v2_training','scope':'synthetic_prototype','runtime':hardware,'source_manifest':source_manifest,'dataset_manifest':data_manifest,
  'training':{'seed':seed,'training_cases':len(cases),'development_cases':len(development),'source_commit':source_manifest['source_commit'],
   'selected_mode':chosen,'selected_head_epoch':baseline_epoch,'selected_adaptation_epoch':best_adapt_epoch,'adapted_layer_indices':indices,
   'selected_development_feedback_gate':best_feedback,
   'selected_encoder_unchanged':selected_encoder_unchanged,
   'lower_layers_unchanged':True,'upstream_encoder_state_sha256':original,'baseline_seconds':baseline_seconds,
   'adaptation_seconds':adaptation_seconds,'history':history,'optimizer_labels':['train'],'checkpoint_selection_labels':['development']},
  'untrained_development_metrics':untrained,'head_only_development_metrics':baseline_metrics,
  'selected_development_metrics':question_metrics(predictions),'encoder_identity':identity,'loading_info':loading,
  'gold_labels_reviewed':False,'calibrated':False,'qualifies_model_quality':False,'qualifies_cpu_budget':False}
 output_dir.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='.v2-output-',dir=output_dir.parent) as temporary:
  staging=Path(temporary)/'model';(staging/'encoder').mkdir(parents=True)
  for name in ENCODER_FILES:shutil.copyfile(checked_file(encoder_dir,name),staging/'encoder'/name)
  save_head(staging/'head',model.head,identity,max_tokens=512,training_info=report['training'])
  torch.save(best_layers,staging/'adaptation.pt')
  adaptation={'layer_indices':indices,'state_sha256':digest_state(best_layers),'selected_mode':chosen,'base_weights_unchanged_on_disk':True}
  for name,value in (('adaptation.json',adaptation),('training-report.json',report),('development-predictions.json',predictions),
     ('source-manifest.json',source_manifest),('dataset-manifest.json',data_manifest),('pilot_config.json',config)):
   (staging/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
  manifest(staging);cpu,_,_=load_model(staging);drifts=[];choices=[]
  with torch.inference_mode():
   for item in reference:
    actual=cpu(item['row']).cpu();expected=torch.tensor(item['logits']);drifts.append(float((actual-expected).abs().max()));choices.append(int(actual.argmax())==int(expected.argmax()))
  if not all(choices):raise ValueError('Fresh CPU reload changed selected choices')
  report['fresh_cpu_reload']={'rows':len(reference),'all_choices_match':all(choices),'max_logit_drift':max(drifts),'reference_rows':reference}
  (staging/'training-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');manifest(staging);verify_bundle(staging);staging.rename(output_dir)
 emit('v2_training_complete',selected_mode=chosen,metrics=report['selected_development_metrics'],fresh_cpu_reload=report['fresh_cpu_reload']['max_logit_drift'])
 return report
