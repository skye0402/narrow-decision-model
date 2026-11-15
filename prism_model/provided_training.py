"""Fresh task training using only the user's pinned 400-request archive.

Select an epoch count on an internal 320/80 split, then initialize again from
the untouched upstream encoder and a fresh head and fit all 400. The final
artifact is uncalibrated. No external test or previous task weights are read.
"""
from collections import Counter
import gc
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

from .evidence_data import load_partition, question_examples
from .evidence_selection import critical_pipeline_summary
from .model_v2 import differentiable_logits
from .model_v4 import encoder_identity, load_model, manifest, verify_bundle
from .neural import FrozenDecisionModel, OptionScorer, load_local_encoder, save_head
from .provenance import checked_file, file_sha256, verify_source
from .task_model import feature_batches, prediction_records, question_metrics, score_features
from .task_data import stable_json
from .training import ENCODER_FILES, emit, select_device, state_sha256


def initialize(encoder_dir, device, seed):
    torch.manual_seed(seed)
    encoder, tokenizer, upstream, _ = load_local_encoder(encoder_dir)
    encoder.config.reference_compile=False
    # No adaptation/head from an earlier procurement model is accepted.
    head = OptionScorer(encoder.config.hidden_size)
    initial_head = state_sha256(head)
    model = FrozenDecisionModel(encoder, tokenizer, head, max_tokens=512).to(device)
    model.encoder.requires_grad_(True); model.head.requires_grad_(True)
    return model, upstream, initial_head


def supplied_cases_hash(cases):
    """Internal role assignment may change; request, labels and fixtures may not."""
    normalized=sorted(({**c,'split':'train'} for c in cases),key=lambda c:c['case_id'])
    return hashlib.sha256(stable_json(normalized).encode()).hexdigest()


def fit_phase(model, cases, development, config, *, epochs, seed, phase):
    device = next(model.parameters()).device
    examples = question_examples(cases, config, attributes=True)
    dev = question_examples(development, config, attributes=True)
    counts = Counter((e['row']['question_id'], e['label']) for e in examples)
    totals = Counter(e['row']['question_id'] for e in examples)
    kinds = {q:len({label for question,label in counts if question==q}) for q in totals}
    weights = {k:min(4.,max(.5,math.sqrt(totals[k[0]]/(kinds[k[0]]*n)))) for k,n in counts.items()}
    optimizer = torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':1e-5},
                                   {'params':model.head.parameters(),'lr':1e-3}],weight_decay=.01)
    batch_size=8; accumulate=4; batches=math.ceil(len(examples)/batch_size)
    updates=math.ceil(batches/accumulate)*epochs; warmup=max(1,int(.1*updates))
    def rate(step):
        if step<warmup:return (step+1)/warmup
        return .5*(1+math.cos(math.pi*min(1,(step-warmup)/max(1,updates-warmup))))
    schedule=torch.optim.lr_scheduler.LambdaLR(optimizer,rate)
    sampler=random.Random(seed); history=[]; best_rank=None; best_state=None; best_epoch=0; stale=0
    before=state_sha256(model.encoder); started=perf_counter(); selected_diagnostics=[]
    for epoch in range(epochs):
        model.train(); model.encoder.eval(); order=list(range(len(examples))); sampler.shuffle(order)
        total_loss=0.; optimizer.zero_grad(set_to_none=True)
        for batch,start in enumerate(range(0,len(order),batch_size)):
            chosen=[examples[i] for i in order[start:start+batch_size]]
            rows=[e['row'] for e in chosen]
            targets=torch.tensor([e['target'] for e in chosen],device=device)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                logits=differentiable_logits(model,rows)
                losses=nn.functional.cross_entropy(logits,targets,reduction='none')
                factors=torch.tensor([weights[(e['row']['question_id'],e['label'])] for e in chosen],device=device)
                loss=(losses*factors).mean()
            if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite training loss')
            first=(batch//accumulate)*accumulate
            group_size=min(accumulate*batch_size,len(order)-first*batch_size)
            (loss*len(chosen)/group_size).backward(); total_loss+=float(loss.detach())*len(chosen)
            if (batch+1)%accumulate==0 or batch+1==batches:
                nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); schedule.step(); optimizer.zero_grad(set_to_none=True)
        row={'epoch':epoch+1,'loss':total_loss/len(examples)}
        model.eval()
        if development:
            records=prediction_records(dev,score_features(model.head,feature_batches(model,dev,8)))
            metrics=question_metrics(records)
            routed,diagnostics=critical_pipeline_summary(model,development,records,config,'attributes',known_only=False)
            eligible=metrics['material_group']['accuracy']>=.90 and metrics['replacement_identification']['macro_f1']>=.85
            score=sum(x['macro_f1'] for x in metrics.values())/len(metrics)
            rank=(eligible,-routed['false_raw_ready'],routed['correct']/max(1,routed['rows']),score)
            row.update(metrics=metrics,routed=routed,selection_rank=list(rank))
            if best_rank is None or rank>best_rank:
                best_rank=rank; best_epoch=epoch+1; stale=0; selected_diagnostics=diagnostics
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:stale+=1
            emit('provided_epoch',phase=phase,epoch=epoch+1,loss=row['loss'],
                 group_accuracy=metrics['material_group']['accuracy'],identification_f1=metrics['replacement_identification']['macro_f1'],
                 false_raw_ready=routed['false_raw_ready'],semantic_correct=routed['correct'],semantic_rows=routed['rows'])
        else:
            best_epoch=epoch+1
            emit('provided_epoch',phase=phase,epoch=epoch+1,loss=row['loss'])
        history.append(row)
        if development and stale>=6:break
    if best_state is not None:model.load_state_dict(best_state)
    model.eval()
    if state_sha256(model.encoder)==before:raise ValueError('Encoder did not change')
    return {'phase':phase,'epochs_run':len(history),'selected_epoch':best_epoch,'training_cases':len(cases),
            'development_cases':len(development),'question_rows':len(examples),'history':history,
            'seconds':perf_counter()-started,'encoder_changed':True},selected_diagnostics


def save_bundle(model, config, upstream, encoder_dir, output, source_manifest, dataset_manifest, report, verification_rows):
    output=Path(output)
    if output.exists():raise ValueError('Use a new bundle output')
    references=[]
    with torch.inference_mode():
        for row in verification_rows:references.append((row,model(row).detach().cpu()))
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.provided-training-',dir=output.parent) as temp:
        stage=Path(temp)/'model';(stage/'encoder').mkdir(parents=True)
        for name in ENCODER_FILES:shutil.copyfile(checked_file(encoder_dir,name),stage/'encoder'/name)
        model.encoder.to('cpu').save_pretrained(stage/'encoder',safe_serialization=False)
        identity=encoder_identity(stage/'encoder',upstream)
        save_head(stage/'head',model.head,identity,max_tokens=512,training_info=report['training'])
        torch.save({},stage/'adaptation.pt')
        objects={'adaptation.json':{'mode':'full_encoder_evidence','task_mode':'attributes','encoder_identity':identity},
                 'training-report.json':report,'source-manifest.json':source_manifest,'dataset-manifest.json':dataset_manifest,
                 'pilot_config.json':config}
        for name,value in objects.items():(stage/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
        manifest(stage); cpu,_,_=load_model(stage); checks=[]
        with torch.inference_mode():
            for row,expected in references:
                actual=cpu(row).cpu(); checks.append({'input_sha256':row['text_sha256'],
                    'same_choice':int(actual.argmax())==int(expected.argmax()),'max_logit_drift':float((actual-expected).abs().max())})
        if not all(x['same_choice'] for x in checks):raise ValueError('CPU reload changed a verification choice')
        report['fresh_cpu_reload']={'rows':len(checks),'all_choices_match':True,'max_logit_drift':max(x['max_logit_drift'] for x in checks),'checks':checks}
        (stage/'training-report.json').write_text(json.dumps(report,indent=2)+'\n');manifest(stage);verify_bundle(stage);stage.rename(output)
    return report


def train(*, data_dir, encoder_dir, output_dir, source_root, source_manifest, device='cuda', max_epochs=20, seed=73):
    data_dir,encoder_dir,output_dir=map(Path,(data_dir,encoder_dir,output_dir))
    if type(max_epochs) is not int or max_epochs<1:raise ValueError('Positive epoch count required')
    if output_dir.exists():raise ValueError('Use a new training output')
    for protected in (data_dir,encoder_dir,Path(source_root)):
        if output_dir.resolve().is_relative_to(protected.resolve()) or protected.resolve().is_relative_to(output_dir.resolve()):
            raise ValueError('Output must be separate from inputs and source')
    verify_source(source_root,source_manifest)
    config,cases,m=load_partition(data_dir,'train',allowed_splits={'train','development'})
    _,development,_=load_partition(data_dir,'development',allowed_splits={'train','development'})
    if m['scope'] not in ('user_supplied_training_only','provided_training_cpu_mechanism'):raise ValueError('Wrong data scope')
    if not m['sole_task_training_source'] or m.get('external_test_included') is not False:raise ValueError('Exclusive source required')
    expected_count=400 if m['scope']=='user_supplied_training_only' else m['provided_case_count']
    all_cases=cases+development
    if len(all_cases)!=expected_count or len({c['family_id'] for c in all_cases})!=len(all_cases):raise ValueError('Unexpected cases/family overlap')
    if supplied_cases_hash(all_cases)!=m['provided_cases_sha256']:raise ValueError('Cases differ from the supplied archive')
    if set(m['partitions'])!={'train','development'}:raise ValueError('Only internal training/development roles allowed')
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    selected,hardware=select_device(device,'NVIDIA L4')
    if selected.type=='cuda' and not torch.cuda.is_bf16_supported():raise ValueError('BF16 required')
    model,upstream,initial_head=initialize(encoder_dir,selected,seed)
    initial_encoder=state_sha256(model.encoder)
    emit('provided_training_started',train_cases=len(cases),development_cases=len(development),archive_sha256=m['input_archive_sha256'])
    selection,diagnostics=fit_phase(model,cases,development,config,epochs=max_epochs,seed=seed,phase='internal_selection')
    output_dir.mkdir(parents=True)
    (output_dir/'internal-development-results.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    common={'kind':'provided_archive_full_encoder_training','task_mode':'attributes','scope':m['scope'],
        'initialization':{'kind':'untouched_upstream_encoder_and_fresh_random_head','encoder_identity':upstream,
            'encoder_state_sha256':initial_encoder,'fresh_head_state_sha256':initial_head,'seed':seed,'prior_task_weights_used':False},
        'runtime':hardware,'source_commit':source_manifest['source_commit'],'input_archive_sha256':m['input_archive_sha256'],
        'only_supplied_training_data_used':True,'external_test_used':False,'calibrated':False,
        'qualifies_model_quality':False,'qualifies_cpu_budget':False,'independently_reviewed':False,
        'recipe':{'encoder_lr':1e-5,'head_lr':1e-3,'optimizer':'AdamW','weight_decay':.01,'batch_size':8,'accumulation':4,
            'gradient_clip':1.,'schedule':'10 percent warmup then cosine','class_weights':'inverse_sqrt_per_question_clipped_0.5_4.0'},
        'trained_encoder_parameters':sum(p.numel() for p in model.encoder.parameters()),'trained_head_parameters':sum(p.numel() for p in model.head.parameters())}
    verification=[e['row'] for e in question_examples(development,config)[:20]]
    selection_report={**common,'training':selection,'validation_scope':'internal model selection only; not the future test set',
                      'development_case_results':diagnostics}
    save_bundle(model,config,upstream,encoder_dir,output_dir/'selection-model',source_manifest,m,selection_report,verification)
    del model;gc.collect()
    if selected.type=='cuda':torch.cuda.empty_cache()
    model,again,again_head=initialize(encoder_dir,selected,seed)
    if again!=upstream or again_head!=initial_head or state_sha256(model.encoder)!=initial_encoder:raise ValueError('Refit initialization changed')
    refit,_=fit_phase(model,all_cases,[],config,epochs=selection['selected_epoch'],seed=seed,phase='refit_all_supplied_cases')
    final_report={**common,'training':refit,'epoch_selection':selection,
        'validation_scope':'All 400 supplied cases used for final gradients; independent test and calibration pending',
        'final_training_roles':['train','development']}
    save_bundle(model,config,upstream,encoder_dir,output_dir/'model',source_manifest,m,final_report,verification)
    (output_dir/'run-report.json').write_text(json.dumps({'selection':selection_report,'final':final_report},indent=2)+'\n')
    emit('provided_training_complete',selected_epoch=selection['selected_epoch'],final_training_cases=len(all_cases),
         calibrated=False,test_set_used=False,cpu_reload_max_drift=final_report['fresh_cpu_reload']['max_logit_drift'])
    return final_report
