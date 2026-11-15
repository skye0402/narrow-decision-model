"""Relational continuation: whole-family batches and balanced typed outcomes.

Only the training role receives gradients. The warm checkpoint is our previous
original-encoder/fresh-head model; previous test data or gates are never loaded.
"""
from collections import Counter
import json
import math
from pathlib import Path
import random
from time import perf_counter

import torch
from torch import nn

from .evidence_data import load_partition, question_examples
from .evidence_selection import critical_pipeline_summary
from .model_v2 import differentiable_logits
from .provided_training import save_bundle
from .model_v4 import load_model
from .broad_training import selection_summary as previous_selection
from .provenance import file_sha256, verify_source
from .task_model import feature_batches, prediction_records, question_metrics, score_features
from .training import emit, select_device, state_sha256


def selection_summary(rows,options,oracle_f1):
    value=previous_selection(rows,options,oracle_f1)
    safety=[r for r in rows if r['expected']['group'] in ('mixed_scope','no_suitable_group','insufficient_evidence')
            or r['expected']['identification'] in ('conflicting','unclear') or r['expected']['status']=='evaluation_unavailable']
    safety_false=sum(r['prediction']['status']=='ready_for_buyer_review' for r in safety)
    routing=value['routing'];precision=routing['ready_precision']['estimate'] or 0.;recall=routing['ready_recall'] or 0.
    base=(value['group']['accuracy']>=.90 and oracle_f1>=.85 and value['actual_identification_macro_f1']>=.85 and recall>=.60)
    ready_f1=2*precision*recall/(precision+recall) if precision+recall else 0.
    value.update(base_eligible=base,eligible=base and precision>=.95 and safety_false==0,
                 safety_cases=len(safety),safety_false_ready=safety_false,
                 selection_rank=[base,-safety_false,-routing['false_ready'],ready_f1,value['actual_identification_macro_f1'],oracle_f1,routing['exact_matches']/len(rows)])
    return value


def family_batches(examples,seed,batch_size=8):
    """Never separate a family/question's counterfactuals into different batches."""
    groups={}
    for i,e in enumerate(examples):groups.setdefault((e['family_id'],e['row']['question_id']),[]).append(i)
    chunks=list(groups.values())
    if any(len(g)>batch_size for g in chunks):raise ValueError('Family/question exceeds batch capacity')
    random.Random(seed).shuffle(chunks);batches=[];current=[]
    for chunk in chunks:
        if len(current)+len(chunk)>batch_size:batches.append(current);current=[]
        current.extend(chunk)
    if current:batches.append(current)
    return batches


def choice_losses(logits,targets,option_counts,smoothing=.05):
    # Different questions offer four or nine answers. Never spread smoothing
    # mass onto the padding used to batch their variable-sized output spaces.
    return torch.stack([nn.functional.cross_entropy(logits[i,:count].unsqueeze(0),targets[i:i+1],
                                                   reduction='sum',label_smoothing=smoothing)
                        for i,count in enumerate(option_counts)])


def train(*, data_dir, warm_dir, output_dir, source_root, source_manifest,
          device='cuda', epochs=3, seed=74):
    data_dir,warm_dir,output_dir=map(Path,(data_dir,warm_dir,output_dir))
    if type(epochs) is not int or not 1<=epochs<=3:raise ValueError('Predeclared budget is at most three epochs')
    if output_dir.exists():raise ValueError('Use a new training output')
    for protected in (data_dir,warm_dir,Path(source_root)):
        if output_dir.resolve().is_relative_to(protected.resolve()) or protected.resolve().is_relative_to(output_dir.resolve()):
            raise ValueError('Output must be separate from inputs and source')
    verify_source(source_root,source_manifest)
    config,cases,manifest=load_partition(data_dir,'train',allowed_splits={'train','development'})
    _,development,_=load_partition(data_dir,'development',allowed_splits={'train','development'})
    if set(manifest['partitions'])!={'train','development'}:raise ValueError('Only optimizer and epoch-selection roles may be mounted')
    if manifest['scope'] not in ('synthetic_prototype','curriculum_cpu_mechanism'):raise ValueError('Wrong data scope')
    if manifest.get('exposed_author100_included') is not False:raise ValueError('Exposed test must remain absent')
    if manifest.get('prior_task_weights_used') is not True:raise ValueError('Declare the warm task checkpoint')
    if manifest.get('previous_evaluation_included') is not False:raise ValueError('Previous tests must remain absent')
    if {c['family_id'] for c in cases}&{c['family_id'] for c in development}:raise ValueError('Development family leak')
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    selected,hardware=select_device(device,'NVIDIA L4')
    if selected.type=='cuda' and not torch.cuda.is_bf16_supported():raise ValueError('BF16 required')
    torch.manual_seed(seed)
    model,warm_config,warm_manifest=load_model(warm_dir)
    if warm_config!=config:raise ValueError('Different warm policy configuration')
    if file_sha256(warm_dir/'bundle-manifest.json')!=manifest['warm_model_artifact_identity']:raise ValueError('Warm checkpoint changed')
    parent_report=json.loads((warm_dir/'training-report.json').read_text())
    if parent_report['kind']!='broadened_fresh_full_encoder_training' or parent_report['exposed_author100_used'] or parent_report['heldout_evaluation_used']:
        raise ValueError('Unexpected warm training ancestry')
    if (warm_dir/'calibration.json').exists():raise ValueError('Warm weights must not include fitted gates')
    if parent_report['dataset_manifest']['parent_dataset_manifest_sha256']!=manifest['previous_training_manifest_sha256']:
        raise ValueError('Warm training and retained corpus differ')
    upstream=json.loads((warm_dir/'adaptation.json').read_text())['encoder_identity']['upstream_identity']
    model=model.to(selected);model.encoder.requires_grad_(True);model.head.requires_grad_(True)
    initial_head=state_sha256(model.head)
    encoder_dir=warm_dir/'encoder'
    before=state_sha256(model.encoder)
    examples=question_examples(cases,config);dev=question_examples(development,config)
    counts=Counter((e['row']['question_id'],e['label']) for e in examples)
    totals=Counter(e['row']['question_id'] for e in examples)
    kinds={q:len({label for question,label in counts if question==q}) for q in totals}
    weights={k:min(8.,max(.25,totals[k[0]]/(kinds[k[0]]*n))) for k,n in counts.items()}
    optimizer=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':5e-6},
                                 {'params':model.head.parameters(),'lr':1e-4}],weight_decay=.05)
    batch_size=8;accumulate=4
    epoch_orders=[family_batches(examples,seed+epoch,batch_size) for epoch in range(epochs)]
    updates=sum(math.ceil(len(order)/accumulate) for order in epoch_orders);warmup=max(1,int(.1*updates))
    def rate(step):
        if step<warmup:return (step+1)/warmup
        return .5*(1+math.cos(math.pi*min(1,(step-warmup)/max(1,updates-warmup))))
    schedule=torch.optim.lr_scheduler.LambdaLR(optimizer,rate)
    output_dir.mkdir(parents=True);checkpoints=output_dir/'checkpoints';checkpoints.mkdir()
    history=[];best_rank=None;best_epoch=0;best_diagnostics=[];started=perf_counter()
    emit('curriculum_training_started',training_cases=len(cases),development_cases=len(development),
         training_rows=len(examples),epochs=epochs,parent_dataset_manifest_sha256=manifest['parent_dataset_manifest_sha256'])
    for epoch,order in enumerate(epoch_orders):
        model.train();model.encoder.eval();batches=len(order)
        total_loss=0.;optimizer.zero_grad(set_to_none=True)
        seen_rows=0
        for batch,indices in enumerate(order):
            chosen=[examples[i] for i in indices];seen_rows+=len(chosen)
            targets=torch.tensor([e['target'] for e in chosen],device=selected)
            with torch.autocast(device_type=selected.type,dtype=torch.bfloat16,enabled=selected.type=='cuda'):
                logits=differentiable_logits(model,[e['row'] for e in chosen])
                losses=choice_losses(logits,targets,[len(e['row']['option_ids']) for e in chosen])
                factors=torch.tensor([weights[(e['row']['question_id'],e['label'])] for e in chosen],device=selected)
                loss=(losses*factors).mean()
            if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite loss')
            first=(batch//accumulate)*accumulate
            group_size=sum(len(order[b]) for b in range(first,min(first+accumulate,batches)))
            (loss*len(chosen)/group_size).backward();total_loss+=float(loss.detach())*len(chosen)
            if (batch+1)%accumulate==0 or batch+1==batches:
                nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();schedule.step();optimizer.zero_grad(set_to_none=True)
            if (batch+1)%400==0:
                emit('curriculum_training_progress',epoch=epoch+1,batch=batch+1,batches=batches,loss=total_loss/seen_rows)
        model.eval()
        records=prediction_records(dev,score_features(model.head,feature_batches(model,dev,8)))
        oracle=question_metrics([r for r in records if r['primary']])
        _,diagnostics=critical_pipeline_summary(model,development,records,config,'attributes',known_only=False)
        by_id={c['case_id']:c for c in development}
        primary=[r for r in diagnostics if by_id[r['case_id']]['primary']]
        summary=selection_summary(primary,config['identification_options'],oracle['replacement_identification']['macro_f1'])
        rank=tuple(summary['selection_rank'])
        checkpoint=checkpoints/f'epoch-{epoch+1}.pt'
        torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},checkpoint)
        row={'epoch':epoch+1,'loss':total_loss/len(examples),'selection':summary,
             'oracle_question_metrics':oracle,'checkpoint_sha256':file_sha256(checkpoint)}
        history.append(row)
        if best_rank is None or rank>best_rank:
            best_rank=rank;best_epoch=epoch+1;best_diagnostics=diagnostics
        (checkpoints/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        emit('curriculum_training_epoch',**{k:row[k] for k in ('epoch','loss')},selected_epoch=best_epoch,
             eligible=summary['eligible'],group_accuracy=summary['group']['accuracy'],
             actual_identification_macro_f1=summary['actual_identification_macro_f1'],
             ready_precision=summary['routing']['ready_precision']['estimate'],ready_recall=summary['routing']['ready_recall'],
             false_ready=summary['routing']['false_ready'],safety_false_ready=summary['safety_false_ready'])
    model.load_state_dict(torch.load(checkpoints/f'epoch-{best_epoch}.pt',map_location=selected,weights_only=True));model.eval()
    if state_sha256(model.encoder)==before:raise ValueError('Encoder did not change')
    if state_sha256(model.head)==initial_head:raise ValueError('Decision head did not change')
    training={'epochs_run':len(history),'selected_epoch':best_epoch,'training_cases':len(cases),
              'development_cases':len(development),'question_rows':len(examples),'history':history,
              'seconds':perf_counter()-started,'encoder_changed':True,'head_changed':True,'final_refit_on_development':False,
              'selection_scope':'One preselected primary per development family; composed actual routing',
              'selected_development_eligible':history[best_epoch-1]['selection']['eligible']}
    report={'kind':'relational_warm_full_encoder_training','task_mode':'attributes',
            'dataset_manifest':manifest,'training':training,'runtime':hardware,
            'source_commit':source_manifest['source_commit'],'calibrated':False,
            'initialization':{'kind':'previous_own_broader_model','parent_artifact_identity':file_sha256(warm_dir/'bundle-manifest.json'),'parent_model_identity':warm_manifest['model_identity'],'encoder_identity':upstream,
                'encoder_state_sha256':before,'head_state_sha256':initial_head,'seed':seed,'prior_task_weights_used':True},
            'gradient_roles':['train'],'checkpoint_selection_roles':['development'],
            'exposed_author100_used':False,'heldout_evaluation_used':False,'independently_reviewed':False,
            'qualifies_model_quality':False,'qualifies_cpu_budget':False,
            'recipe':{'encoder_lr':5e-6,'head_lr':1e-4,'optimizer':'AdamW','weight_decay':.05,
                'batch_size':8,'accumulation':4,'gradient_clip':1.,'schedule':'10 percent warmup then cosine',
                'class_weights':'inverse_frequency_per_question_clipped_0.25_8','label_smoothing':.05,'counterfactual_family_question_batches':True},
            'trained_encoder_parameters':sum(p.numel() for p in model.encoder.parameters()),
            'trained_head_parameters':sum(p.numel() for p in model.head.parameters())}
    checks=[];seen=Counter()
    for e in dev:
        q=e['row']['question_id']
        if seen[q]<2:checks.append(e['row']);seen[q]+=1
    save_bundle(model,config,upstream,encoder_dir,output_dir/'model',source_manifest,manifest,report,checks)
    (checkpoints/'selected-development-results.json').write_text(json.dumps(best_diagnostics,indent=2)+'\n')
    (checkpoints/'run-report.json').write_text(json.dumps(report,indent=2)+'\n')
    emit('curriculum_training_complete',selected_epoch=best_epoch,development_eligible=training['selected_development_eligible'],
         cpu_reload_max_drift=report['fresh_cpu_reload']['max_logit_drift'],calibrated=False)
    return report
