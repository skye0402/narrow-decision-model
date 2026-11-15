"""Frozen-encoder task training, portable bundles, and CPU inference primitives."""
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import shutil
import tempfile
from time import perf_counter

import torch
from torch import nn

from .metrics import classification
from .neural import (DecisionFeatures,FrozenDecisionModel,NeuralJudge,OptionScorer,
    load_head,load_local_encoder,save_head,tokenize_row)
from .provenance import checked_file,file_sha256,verify_files,verify_source
from .task_data import load_partition,question_examples
from .training import ENCODER_FILES,cpu_roundtrip,emit,select_device,state_sha256,synchronize

TASK_BUNDLE_VERSION='task-bundle-1'


def permute_options(row,order):
    if sorted(order)!=list(range(len(row['option_ids']))):raise ValueError('Invalid option permutation')
    text=row['text'].rsplit('\nOPTIONS\n',1)[0]+'\nOPTIONS\n'
    spans=[];ids=[]
    for index,old_index in enumerate(order):
        old=row['option_spans'][old_index]
        answer=row['text'][old['start']:old['end']]
        key=row['option_ids'][old_index];prefix=f'{index} '
        start=len(text)+len(prefix)
        spans.append({'option_id':key,'start':start,'end':start+len(answer)})
        ids.append(key);text+=prefix+answer+'\n'
    return {**row,'text':text,'option_ids':ids,'option_spans':spans,'text_sha256':hashlib.sha256(text.encode()).hexdigest()}


def augment_options(examples,seed):
    rng=random.Random(seed);result=list(examples)
    for example in examples:
        order=list(range(len(example['row']['option_ids'])))
        rng.shuffle(order)
        if order==list(range(len(order))):order=order[1:]+order[:1]
        row=permute_options(example['row'],order)
        result.append({**example,'row':row,'target':row['option_ids'].index(example['label']),'option_augmented':True})
    return result


def feature_batches(model,examples,batch_size=8,progress=None):
    """Encode question rows together; answers share their question's sequence."""
    if type(batch_size) is not int or batch_size<1:raise ValueError('Invalid feature batch size')
    encoder=model.encoder;encoder.eval()
    if hasattr(encoder.config,'reference_compile'):encoder.config.reference_compile=False
    device=next(encoder.parameters()).device
    output=[]
    for start in range(0,len(examples),batch_size):
        batch=examples[start:start+batch_size]
        encoded=[tokenize_row(model.tokenizer,e['row'],model.max_tokens) for e in batch]
        width=max(item[0]['input_ids'].shape[1] for item in encoded)
        keys=encoded[0][0].keys()
        inputs={}
        for key in keys:
            fill=model.tokenizer.pad_token_id if key=='input_ids' else 0
            tensor=torch.full((len(batch),width),fill,dtype=encoded[0][0][key].dtype)
            for i,(tokens,_) in enumerate(encoded):tensor[i,:tokens[key].shape[1]]=tokens[key][0]
            inputs[key]=tensor.to(device)
        with torch.no_grad():
            hidden=encoder(**inputs).last_hidden_state
            if not bool(torch.isfinite(hidden).all()):raise ValueError('Non-finite encoder features')
            for i,(example,(tokens,mask)) in enumerate(zip(batch,encoded)):
                weights=mask.to(device=device,dtype=hidden.dtype)
                states=hidden[i,:weights.shape[1]]
                options=weights@states/weights.sum(dim=1,keepdim=True)
                output.append(DecisionFeatures(hidden[i,0].detach().clone(),options.detach().clone(),
                    tuple(example['row']['option_ids']),tokens['input_ids'].shape[1],example['row']['text_sha256']))
        if progress is not None:progress(min(start+len(batch),len(examples)),len(examples))
    return output


def pack_features(features):
    if not features:raise ValueError('No training features')
    n=len(features);width=max(len(f.option_ids) for f in features);hidden=features[0].context.shape[0]
    device=features[0].context.device
    packed=torch.zeros((n,width,2*hidden),device=device,dtype=torch.float32)
    valid=torch.zeros((n,width),device=device,dtype=torch.bool)
    for i,f in enumerate(features):
        count=len(f.option_ids)
        packed[i,:count]=torch.cat((f.context.expand(count,-1),f.options),dim=-1)
        valid[i,:count]=True
    return packed,valid


def packed_logits(head,features,valid):
    scores=head.network(features).squeeze(-1)
    if scores.shape!=valid.shape or not bool(valid.any(dim=1).all()):raise ValueError('Invalid packed options')
    return scores.masked_fill(~valid,torch.finfo(scores.dtype).min)


def score_features(head,features,batch_size=128):
    head.eval();output=[]
    with torch.inference_mode():
        for start in range(0,len(features),batch_size):
            batch=features[start:start+batch_size];packed,valid=pack_features(batch)
            logits=packed_logits(head,packed,valid).cpu()
            for i,f in enumerate(batch):output.append(logits[i,:len(f.option_ids)].tolist())
    return output


def prediction_records(examples,logits):
    if len(examples)!=len(logits):raise ValueError('Prediction row count mismatch')
    records=[]
    for example,values in zip(examples,logits):
        row=example['row']
        if len(values)!=len(row['option_ids']) or not all(torch.isfinite(torch.tensor(values))):
            raise ValueError('Invalid task logits')
        choice=max(range(len(values)),key=values.__getitem__)
        records.append({k:example[k] for k in ('case_id','family_id','primary','target','label')}|
            {'question_id':row['question_id'],'option_ids':row['option_ids'],'input_sha256':row['text_sha256'],
             'logits':values,'choice':row['option_ids'][choice]})
    return records


def question_metrics(records):
    result={}
    for question in sorted({r['question_id'] for r in records}):
        rows=[r for r in records if r['question_id']==question]
        labels=sorted({k for row in rows for k in row['option_ids']})
        result[question]=classification([r['label'] for r in rows],[r['choice'] for r in rows],labels)
    return result


def model_identity(directory):
    directory=Path(directory)
    files={name:file_sha256(checked_file(directory,name)) for name in (
        'encoder/config.json','encoder/tokenizer.json','encoder/tokenizer_config.json',
        'encoder/special_tokens_map.json','encoder/pytorch_model.bin','head/head.pt','head/head.json','pilot_config.json')}
    return hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def write_bundle_manifest(directory,**extra):
    directory=Path(directory)
    manifest={'version':TASK_BUNDLE_VERSION,'scope':'synthetic_prototype','complete':True,
        'model_identity':model_identity(directory),'calibrated_for_synthetic_prototype':False,
        'qualifies_model_quality':False,'qualifies_cpu_budget':False,**extra,
        'files_sha256':{p.relative_to(directory).as_posix():file_sha256(p) for p in sorted(directory.rglob('*'))
                        if p.is_file() and p.name!='bundle-manifest.json'}}
    (directory/'bundle-manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    return manifest


def verify_task_bundle(directory):
    directory=Path(directory)
    manifest=json.loads(checked_file(directory,'bundle-manifest.json').read_text())
    if (manifest.get('version')!=TASK_BUNDLE_VERSION or manifest.get('scope')!='synthetic_prototype'
            or manifest.get('complete') is not True):raise ValueError('Unsupported task model bundle')
    verify_files(directory,manifest['files_sha256'],exact=True,exclude=('bundle-manifest.json',),
        required=('encoder/pytorch_model.bin','head/head.pt','head/head.json','pilot_config.json','training-report.json','source-manifest.json'))
    if manifest['model_identity']!=model_identity(directory):raise ValueError('Task model identity mismatch')
    return manifest


def load_task_model(directory,*,threads=2):
    directory=Path(directory);manifest=verify_task_bundle(directory)
    torch.set_num_threads(threads)
    encoder,tokenizer,identity,_=load_local_encoder(directory/'encoder')
    if hasattr(encoder.config,'reference_compile'):encoder.config.reference_compile=False
    head,metadata=load_head(directory/'head',identity)
    model=FrozenDecisionModel(encoder,tokenizer,head,max_tokens=metadata['max_tokens']).eval()
    config=json.loads((directory/'pilot_config.json').read_text())
    return model,config,manifest


def run_task_training(*,data_dir,encoder_dir,output_dir,source_root,source_manifest,
        device='cuda',seed=17,epochs=40,patience=6,batch_size=64,feature_batch_size=8,learning_rate=0.001):
    """Optimize train only; choose epochs on development. Other labels are absent."""
    output_dir,data_dir,encoder_dir=(Path(p).resolve() for p in (output_dir,data_dir,encoder_dir))
    if output_dir.exists() or any(output_dir.is_relative_to(p) or p.is_relative_to(output_dir) for p in (data_dir,encoder_dir,Path(source_root).resolve())):
        raise ValueError('Use a new output directory separate from source and inputs')
    if any(type(n) is not int or n<1 for n in (epochs,patience,batch_size,feature_batch_size)):
        raise ValueError('Positive training settings required')
    verify_source(source_root,source_manifest)
    config,train_cases,dataset=load_partition(data_dir,'train',allowed_splits={'train','development'})
    _,dev_cases,_=load_partition(data_dir,'development',allowed_splits={'train','development'})
    if {c['family_id'] for c in train_cases}&{c['family_id'] for c in dev_cases}:raise ValueError('Train/development family leakage')
    # The optimizer artifact must not contain any other partition's case files.
    if any((data_dir/(name+'.jsonl')).exists() for name in ('calibration','threshold_selection','prototype_evaluation')):
        raise ValueError('Do not mount calibration/evaluation labels into the optimizer')
    torch.set_num_threads(2);torch.manual_seed(seed);torch.set_float32_matmul_precision('highest')
    selected,runtime=select_device(device,'NVIDIA L4')
    runtime.update(python=platform.python_version(),platform=platform.platform(),torch=torch.__version__,
        transformers=importlib.metadata.version('transformers'),precision='fp32',threads=torch.get_num_threads(),
        reference_compile=False,feature_batch_size=feature_batch_size)
    if selected.type=='cuda':torch.cuda.reset_peak_memory_stats(selected)
    encoder,tokenizer,identity,loading=load_local_encoder(encoder_dir)
    encoder.config.reference_compile=False
    model=FrozenDecisionModel(encoder,tokenizer,OptionScorer(encoder.config.hidden_size)).to(selected,dtype=torch.float32)
    before=state_sha256(encoder);initial_head=state_sha256(model.head)
    canonical=question_examples(train_cases,config)
    train=augment_options(canonical,seed)
    dev=question_examples(dev_cases,config)
    # Check the actual pinned encoder's batched path before using its cache.
    check=feature_batches(model,dev[:4],batch_size=4)
    feature_drift=[]
    for example,batched in zip(dev[:4],check):
        single=model.features(example['row'])
        torch.testing.assert_close(batched.context,single.context,rtol=1e-3,atol=1e-3)
        torch.testing.assert_close(batched.options,single.options,rtol=1e-3,atol=1e-3)
        feature_drift.append(float((batched.options-single.options).abs().max()))
    emit('task_training_started',device=str(selected),train_cases=len(train_cases),train_rows=len(train),development_rows=len(dev))
    synchronize(selected);started=perf_counter()
    def progress(done,total):
        if done==total or done%256==0:emit('features_cached',rows=done,total=total)
    train_features=feature_batches(model,train,feature_batch_size,progress)
    dev_features=feature_batches(model,dev,feature_batch_size,progress)
    synchronize(selected);extraction_seconds=perf_counter()-started
    untrained=prediction_records(dev,score_features(model.head,dev_features))
    untrained_metrics=question_metrics(untrained)
    packed,valid=pack_features(train_features)
    targets=torch.tensor([e['target'] for e in train],device=selected)
    optimizer=torch.optim.AdamW(model.head.parameters(),lr=learning_rate,weight_decay=0.01)
    generator=torch.Generator().manual_seed(seed)
    history=[];best_score=-1.;best_state=None;best_epoch=None;stale=0
    synchronize(selected);started=perf_counter()
    for epoch in range(epochs):
        model.train();order=torch.randperm(len(train),generator=generator).tolist();loss_sum=0.
        for start in range(0,len(order),batch_size):
            indices=torch.tensor(order[start:start+batch_size],device=selected)
            optimizer.zero_grad(set_to_none=True)
            logits=packed_logits(model.head,packed[indices],valid[indices])
            loss=nn.functional.cross_entropy(logits,targets[indices])
            if not bool(torch.isfinite(loss)):raise ValueError('Non-finite task loss')
            loss.backward();nn.utils.clip_grad_norm_(model.head.parameters(),1.)
            optimizer.step();loss_sum+=float(loss.detach())*len(indices)
        predictions=prediction_records(dev,score_features(model.head,dev_features))
        metrics=question_metrics(predictions)
        score=sum(m['macro_f1'] for m in metrics.values())/len(metrics)
        row={'epoch':epoch+1,'training_loss':loss_sum/len(train),'development_metrics':metrics,'selection_score':score}
        history.append(row)
        emit('task_epoch',epoch=epoch+1,loss=row['training_loss'],selection_score=score,
            group_accuracy=metrics['material_group']['accuracy'],identification_macro_f1=metrics['replacement_identification']['macro_f1'])
        if score>best_score+1e-5:
            best_score=score;best_epoch=epoch+1;stale=0
            best_state={k:v.detach().cpu().clone() for k,v in model.head.state_dict().items()}
        else:stale+=1
        if stale>=patience:break
    synchronize(selected);training_seconds=perf_counter()-started
    model.head.load_state_dict(best_state);model.eval()
    if state_sha256(encoder)!=before or any(p.grad is not None for p in encoder.parameters()):raise ValueError('Frozen encoder changed')
    if state_sha256(model.head)==initial_head:raise ValueError('Head did not update')
    predictions=prediction_records(dev,score_features(model.head,dev_features))
    references=[]
    with torch.inference_mode():
        for example in dev[:16]:
            row=example['row']
            references.append({'text_sha256':row['text_sha256'],'option_ids':row['option_ids'],'logits':model(row).cpu().tolist()})
    if selected.type=='cuda':runtime.update(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(selected),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(selected))
    training_info={'scope':'synthetic_prototype','seed':seed,'learning_rate':learning_rate,'weight_decay':0.01,
        'max_epochs':epochs,'selected_epoch':best_epoch,'patience':patience,'batch_size':batch_size,
        'frozen_encoder':True,'canonical_training_rows':len(canonical),'augmented_training_rows':len(train),
        'optimizer_partitions':['train'],'selection_partition':'development','calibration_or_evaluation_labels_mounted':False,
        'data_manifest_sha256':file_sha256(data_dir/'dataset-manifest.json'),'source_commit':source_manifest['source_commit']}
    output_dir.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.task-training-',dir=output_dir.parent) as temporary:
        staging=Path(temporary)/'model';(staging/'encoder').mkdir(parents=True)
        for name in ENCODER_FILES:shutil.copyfile(checked_file(encoder_dir,name),staging/'encoder'/name)
        head_metadata=save_head(staging/'head',model.head,identity,max_tokens=model.max_tokens,training_info=training_info)
        _,cpu_checks=cpu_roundtrip(staging/'encoder',staging/'head',dev[:16],references)
        report={'kind':'first_task_model_training','scope':'synthetic_prototype','runtime':runtime,
            'source_manifest':source_manifest,'dataset_manifest':dataset,'encoder_identity':identity,'loading_info':loading,
            'training':{**training_info,'epochs_run':len(history),'history':history,'best_development_score':best_score,
                'encoder_unchanged':True,'encoder_state_sha256':before,'head_changed':True,
                'feature_extraction_seconds':extraction_seconds,'head_training_seconds':training_seconds},
            'head':head_metadata,'untrained_development_metrics':untrained_metrics,'selected_development_metrics':question_metrics(predictions),
            'batch_feature_max_absolute_drift':max(feature_drift),'cpu_reload':cpu_checks,'reference_logits':references,
            'gold_labels_reviewed':False,'calibrated':False,'qualifies_model_quality':False,'qualifies_cpu_budget':False}
        for name,value in (('training-report.json',report),('development-predictions.json',predictions),
                           ('source-manifest.json',source_manifest),('dataset-manifest.json',dataset),('pilot_config.json',config)):
            (staging/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
        write_bundle_manifest(staging)
        verify_task_bundle(staging)
        staging.rename(output_dir)
    emit('task_artifact_complete',selected_epoch=best_epoch,group_accuracy=report['selected_development_metrics']['material_group']['accuracy'],
        identification_macro_f1=report['selected_development_metrics']['replacement_identification']['macro_f1'],calibrated=False)
    return report
