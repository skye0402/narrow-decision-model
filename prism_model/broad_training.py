"""Fresh upstream training, disjoint narrative development, composed selection.

Only the training role receives gradients. Development selects one of four
predeclared epochs; calibration, threshold and evaluation roles are absent from
the training artifact. The exposed author100 is never read by this module.
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
from .metrics import classification
from .model_v2 import differentiable_logits
from .provided_evaluation import decision_routing_metrics
from .provided_training import initialize, save_bundle
from .provenance import file_sha256, verify_source
from .task_iteration import predicted_group_identification
from .task_model import feature_batches, prediction_records, question_metrics, score_features
from .training import emit, select_device, state_sha256


def selection_summary(rows, options, oracle_f1):
    """Joint Ready correctness counts a wrong product as wrong even if ready."""
    routing=decision_routing_metrics(rows)
    group=classification([r['expected']['group'] for r in rows],
                         [r['prediction']['group'] for r in rows],
                         sorted({r['expected']['group'] for r in rows}|{r['prediction']['group'] for r in rows if r['prediction']['group']}))
    actual=predicted_group_identification(rows,options)['macro_f1_over_true_identification_outcomes']
    safety=[r for r in rows if r['expected']['group']=='mixed_scope' or r['expected']['identification'] in ('conflicting','unclear')]
    safety_false=sum(r['prediction']['status']=='ready_for_buyer_review' for r in safety)
    precision=routing['ready_precision']['estimate'] or 0.
    recall=routing['ready_recall'] or 0.
    eligible=(group['accuracy']>=.90 and oracle_f1>=.85 and actual>=.85
              and precision>=.95 and recall>=.60 and safety_false==0)
    ready_f1=2*precision*recall/(precision+recall) if precision+recall else 0.
    # A never-Ready model cannot become eligible or win by zero false positives.
    balanced=(group['macro_f1']+oracle_f1+actual+ready_f1)/4
    rank=(eligible,balanced,-routing['false_ready'],routing['exact_matches']/len(rows))
    return {'eligible':eligible,'selection_rank':list(rank),'group':group,
            'actual_identification_macro_f1':actual,'oracle_identification_macro_f1':oracle_f1,
            'safety_cases':len(safety),'safety_false_ready':safety_false,'routing':routing}


def train(*, data_dir, encoder_dir, output_dir, source_root, source_manifest,
          device='cuda', epochs=4, seed=73):
    data_dir,encoder_dir,output_dir=map(Path,(data_dir,encoder_dir,output_dir))
    if type(epochs) is not int or not 1<=epochs<=4:raise ValueError('Predeclared budget is at most four epochs')
    if output_dir.exists():raise ValueError('Use a new training output')
    for protected in (data_dir,encoder_dir,Path(source_root)):
        if output_dir.resolve().is_relative_to(protected.resolve()) or protected.resolve().is_relative_to(output_dir.resolve()):
            raise ValueError('Output must be separate from inputs and source')
    verify_source(source_root,source_manifest)
    config,cases,manifest=load_partition(data_dir,'train',allowed_splits={'train','development'})
    _,development,_=load_partition(data_dir,'development',allowed_splits={'train','development'})
    if set(manifest['partitions'])!={'train','development'}:raise ValueError('Only optimizer and epoch-selection roles may be mounted')
    if manifest['scope'] not in ('synthetic_prototype','broad_cpu_mechanism'):raise ValueError('Wrong data scope')
    if manifest.get('exposed_author100_included') is not False:raise ValueError('Exposed test must remain absent')
    if manifest.get('prior_task_weights_used') is not False:raise ValueError('Use untouched upstream weights')
    if {c['family_id'] for c in cases}&{c['family_id'] for c in development}:raise ValueError('Development family leak')
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    selected,hardware=select_device(device,'NVIDIA L4')
    if selected.type=='cuda' and not torch.cuda.is_bf16_supported():raise ValueError('BF16 required')
    model,upstream,initial_head=initialize(encoder_dir,selected,seed)
    before=state_sha256(model.encoder)
    examples=question_examples(cases,config);dev=question_examples(development,config)
    counts=Counter((e['row']['question_id'],e['label']) for e in examples)
    totals=Counter(e['row']['question_id'] for e in examples)
    kinds={q:len({label for question,label in counts if question==q}) for q in totals}
    weights={k:min(4.,max(.5,math.sqrt(totals[k[0]]/(kinds[k[0]]*n)))) for k,n in counts.items()}
    optimizer=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':1e-5},
                                 {'params':model.head.parameters(),'lr':1e-3}],weight_decay=.01)
    batch_size=8;accumulate=4;batches=math.ceil(len(examples)/batch_size)
    updates=math.ceil(batches/accumulate)*epochs;warmup=max(1,int(.1*updates))
    def rate(step):
        if step<warmup:return (step+1)/warmup
        return .5*(1+math.cos(math.pi*min(1,(step-warmup)/max(1,updates-warmup))))
    schedule=torch.optim.lr_scheduler.LambdaLR(optimizer,rate)
    output_dir.mkdir(parents=True);checkpoints=output_dir/'checkpoints';checkpoints.mkdir()
    history=[];best_rank=None;best_epoch=0;best_diagnostics=[];sampler=random.Random(seed);started=perf_counter()
    emit('broad_training_started',training_cases=len(cases),development_cases=len(development),
         training_rows=len(examples),epochs=epochs,parent_dataset_manifest_sha256=manifest['parent_dataset_manifest_sha256'])
    for epoch in range(epochs):
        model.train();model.encoder.eval();order=list(range(len(examples)));sampler.shuffle(order)
        total_loss=0.;optimizer.zero_grad(set_to_none=True)
        for batch,start in enumerate(range(0,len(order),batch_size)):
            chosen=[examples[i] for i in order[start:start+batch_size]]
            targets=torch.tensor([e['target'] for e in chosen],device=selected)
            with torch.autocast(device_type=selected.type,dtype=torch.bfloat16,enabled=selected.type=='cuda'):
                logits=differentiable_logits(model,[e['row'] for e in chosen])
                losses=nn.functional.cross_entropy(logits,targets,reduction='none')
                factors=torch.tensor([weights[(e['row']['question_id'],e['label'])] for e in chosen],device=selected)
                loss=(losses*factors).mean()
            if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite loss')
            first=(batch//accumulate)*accumulate
            group_size=min(accumulate*batch_size,len(order)-first*batch_size)
            (loss*len(chosen)/group_size).backward();total_loss+=float(loss.detach())*len(chosen)
            if (batch+1)%accumulate==0 or batch+1==batches:
                nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();schedule.step();optimizer.zero_grad(set_to_none=True)
            if (batch+1)%400==0:
                emit('broad_training_progress',epoch=epoch+1,batch=batch+1,batches=batches,loss=total_loss/min(len(order),(batch+1)*batch_size))
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
        emit('broad_training_epoch',**{k:row[k] for k in ('epoch','loss')},selected_epoch=best_epoch,
             eligible=summary['eligible'],group_accuracy=summary['group']['accuracy'],
             actual_identification_macro_f1=summary['actual_identification_macro_f1'],
             ready_precision=summary['routing']['ready_precision']['estimate'],ready_recall=summary['routing']['ready_recall'],
             false_ready=summary['routing']['false_ready'],safety_false_ready=summary['safety_false_ready'])
    model.load_state_dict(torch.load(checkpoints/f'epoch-{best_epoch}.pt',map_location=selected,weights_only=True));model.eval()
    if state_sha256(model.encoder)==before:raise ValueError('Encoder did not change')
    training={'epochs_run':len(history),'selected_epoch':best_epoch,'training_cases':len(cases),
              'development_cases':len(development),'question_rows':len(examples),'history':history,
              'seconds':perf_counter()-started,'encoder_changed':True,'final_refit_on_development':False,
              'selection_scope':'One preselected primary per development family; composed actual routing',
              'selected_development_eligible':history[best_epoch-1]['selection']['eligible']}
    report={'kind':'broadened_fresh_full_encoder_training','task_mode':'attributes',
            'dataset_manifest':manifest,'training':training,'runtime':hardware,
            'source_commit':source_manifest['source_commit'],'calibrated':False,
            'initialization':{'kind':'untouched_upstream_encoder_and_fresh_random_head','encoder_identity':upstream,
                'encoder_state_sha256':before,'fresh_head_state_sha256':initial_head,'seed':seed,'prior_task_weights_used':False},
            'gradient_roles':['train'],'checkpoint_selection_roles':['development'],
            'exposed_author100_used':False,'heldout_evaluation_used':False,'independently_reviewed':False,
            'qualifies_model_quality':False,'qualifies_cpu_budget':False,
            'recipe':{'encoder_lr':1e-5,'head_lr':1e-3,'optimizer':'AdamW','weight_decay':.01,
                'batch_size':8,'accumulation':4,'gradient_clip':1.,'schedule':'10 percent warmup then cosine',
                'class_weights':'inverse_sqrt_per_question_clipped_0.5_4.0'},
            'trained_encoder_parameters':sum(p.numel() for p in model.encoder.parameters()),
            'trained_head_parameters':sum(p.numel() for p in model.head.parameters())}
    checks=[];seen=Counter()
    for e in dev:
        q=e['row']['question_id']
        if seen[q]<2:checks.append(e['row']);seen[q]+=1
    save_bundle(model,config,upstream,encoder_dir,output_dir/'model',source_manifest,manifest,report,checks)
    (checkpoints/'selected-development-results.json').write_text(json.dumps(best_diagnostics,indent=2)+'\n')
    (checkpoints/'run-report.json').write_text(json.dumps(report,indent=2)+'\n')
    emit('broad_training_complete',selected_epoch=best_epoch,development_eligible=training['selected_development_eligible'],
         cpu_reload_max_drift=report['fresh_cpu_reload']['max_logit_drift'],calibrated=False)
    return report
