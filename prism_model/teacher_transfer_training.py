"""Four-epoch, training-only transfer into the fixed ten-layer student."""
import json
import math
from pathlib import Path
import shutil
from time import perf_counter

import torch

from .comparison_quality import evaluate
from .comparison_runtime import ComparisonRuntime
from .comparison_training import rank, training_weights
from .curriculum_training import family_batches
from .evidence_data import load_partition
from .isolated_attribution import attribution_rows, configure_stage, isolated_loss
from .ownership_bundle import load_model, save_bundle, write_manifest
from .ownership_data import augment_training, examples
from .provenance import file_sha256, verify_source
from .teacher_transfer import (COEFFICIENTS, STUDENT_IDENTITY, TEACHER_IDENTITY,
    EncoderCapture, cached_readout_loss, collect_readout, extraction_transfer)
from .training import emit, select_device, state_sha256


WORDING_STUDENT_IDENTITY = '866f497aed698ff963a643bd2cf7f7904d723c3120909b1c8a8f0368e083d4db'
PLAIN_STUDENT_IDENTITY = 'e66fa6adcda140ee223a17b50ff54eb4e6e05c0faba96a616ac029fcee80cde1'
# Natural-wording continuation, 11 November 2026. The plain-transfer selected
# weights and the frozen teacher are unchanged; both were rebound to the
# revised inference source (ownership_bundle.rebind_bundle).
NATURAL_STUDENT_IDENTITY = 'e19641a1959f760cd9ec535ff24e0fd11af31d6fa1aca60e27ec80f8bb312229'
NATURAL_TEACHER_IDENTITY = '50fcacc87b9d81d926e44082462dcab4b4a67d03022b9f7845b5a5a149276d58'
NATURAL_PARENT_COHORT = '8fcf2f3900090b73baebe0c1740ab9c147a08f476acd44ad50feb752cd93f5d4'
NATURAL_COUNTS = {'train': 30952, 'development': 5064, 'natural_train': 4000, 'natural_development': 600}
# Round two warm-starts from round one's selected weights (rebound for the
# Ready-only gate change) and uses the revision-2 natural training families.
NATURAL_RUNS = {
    1: {'student': NATURAL_STUDENT_IDENTITY, 'teacher': NATURAL_TEACHER_IDENTITY, 'warm_cohort': NATURAL_PARENT_COHORT},
    2: {'student': '24794759533cd4814fffd52ebdbee8fc6a46ed08aaf35bc64a3fbaacaedbdd15',
        'teacher': 'b3f022d98adc50beea90658bde61c0641d2dde5470668eb3a191345b07ff4465',
        'warm_cohort': 'b6f21cf812b4436d1813f7f2d8c36c6601b432d24c38387aaed6a2a25ecc9344'},
}


def transfer_rows(cases, config, tokenizer, *, epoch, seed=83, view_version=1):
    rows, audit = attribution_rows(cases, config, tokenizer, epoch=epoch, seed=seed, view_version=view_version)
    originals = [augment_training(c, epoch, seed) for c in cases
                 if c['primary'] or c.get('author') == 'ai-cedar']
    # A supplied request with a gold-span view is supervised by that view; its
    # teacher-only extraction rows would teach the teacher's errors back.
    from .supplied_spans import AUTHOR as GOLD_VIEW
    viewed = {c['case_id'].removesuffix('-gold') for c in cases if c.get('author') == GOLD_VIEW}
    added = [e for e in examples(originals, config)
             if e['row']['question_id'].startswith('attribute:') and e['annotation'] is None
             and e['case_id'] not in viewed]
    identities = [(e['case_id'], e['row']['question_id']) for e in [*rows, *added]]
    if len(set(identities)) != len(identities):
        raise ValueError('Repeated teacher supervision row')
    return [*rows, *added], {**audit, 'unannotated_teacher_attribute_rows': len(added)}


def gold_loss(output, rows, choice_weights, auxiliary_weights):
    valid = [i for i, e in enumerate(rows)
             if not e['row']['question_id'].startswith('attribute:') or e['annotation'] is not None]
    if not valid:
        return output['bio'].sum() * 0.
    selected = {name: ([value[i] for i in valid] if isinstance(value, list) else value[valid])
                for name, value in output.items()}
    return isolated_loss(selected, [rows[i] for i in valid], choice_weights, auxiliary_weights, attribution=True)


def train(*, data_dir, warm_dir, teacher_dir, output_dir, source_root, source_manifest,
          device='cuda', seed=83, max_batches=None, view_version=1):
    if view_version not in (1, 2, 3):
        raise ValueError('Undeclared transfer wording version')
    student_identity = {1: STUDENT_IDENTITY, 2: WORDING_STUDENT_IDENTITY, 3: PLAIN_STUDENT_IDENTITY}[view_version]
    teacher_identity = TEACHER_IDENTITY
    data_dir, warm_dir, teacher_dir, output_dir = map(Path, (data_dir, warm_dir, teacher_dir, output_dir))
    if output_dir.exists() or (max_batches is not None and (device != 'cpu' or type(max_batches) is not int or max_batches < 1)):
        raise ValueError('Use a new output; truncated transfer is a CPU mechanism check only')
    verify_source(source_root, source_manifest)
    from .natural_data import AUTHOR as NATURAL_AUTHOR, COMBINED_GENERATOR
    declared = json.loads((Path(data_dir)/'dataset-manifest.json').read_text())
    natural = declared['generator'] == COMBINED_GENERATOR
    revision = declared.get('natural_revision', 1) if natural else None
    if natural:
        # The deployed workflow passes view version 3; the dataset selects the
        # natural continuation, which keeps version-3 plain views unchanged.
        if view_version != 3 or revision not in NATURAL_RUNS:
            raise ValueError('The natural continuation extends plain-view version 3')
        student_identity, teacher_identity = NATURAL_RUNS[revision]['student'], NATURAL_RUNS[revision]['teacher']
    if (file_sha256(warm_dir/'bundle-manifest.json') != student_identity
            or file_sha256(teacher_dir/'bundle-manifest.json') != teacher_identity):
        raise ValueError('Declared teacher or student artifact changed')
    model, config, _ = load_model(warm_dir)
    teacher, teacher_config, _ = load_model(teacher_dir)
    for name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'):
        if file_sha256(warm_dir/'encoder'/name) != file_sha256(teacher_dir/'encoder'/name):
            raise ValueError('Teacher and student tokenizer bytes differ')
    warm_report = json.loads((warm_dir/'training-report.json').read_text())
    teacher_report = json.loads((teacher_dir/'training-report.json').read_text())
    local, cases, manifest = load_partition(data_dir, 'train', allowed_splits={'train', 'development'})
    _, development, _ = load_partition(data_dir, 'development', allowed_splits={'train', 'development'})
    parent_cohort = manifest.get('generated_parent_manifest_sha256') if natural else manifest['full_cohort_manifest_sha256']
    if (config != local or teacher_config != config or set(manifest['partitions']) != {'train', 'development'}
            or manifest['generator'] != (COMBINED_GENERATOR if natural else 'ownership-comparison-data-1')
            or manifest['exposed_author100_included']
            or manifest['previous_evaluation_included'] or {c['family_id'] for c in cases} & {c['family_id'] for c in development}
            or teacher_report['dataset_manifest']['full_cohort_manifest_sha256'] != parent_cohort
            or warm_report['dataset_manifest']['full_cohort_manifest_sha256'] != (
                NATURAL_RUNS[revision]['warm_cohort'] if natural else parent_cohort)
            or (natural and parent_cohort != NATURAL_PARENT_COHORT)):
        raise ValueError('Transfer may use only the declared unchanged training/development cohort')
    if model.encoder.config.num_hidden_layers != 10 or teacher.encoder.config.num_hidden_layers != 22:
        raise ValueError('The declared teacher/student depth changed')
    gold_views = sum(c.get('author') == 'ai-cedar-gold-span-view' for c in cases)
    if max_batches is None and natural and (len(cases) != NATURAL_COUNTS['train'] + gold_views
            or gold_views != {1: 0, 2: 311}[revision]
            or len(development) != NATURAL_COUNTS['development'] or not manifest['original_400_retained']
            or sum(c.get('author') == NATURAL_AUTHOR for c in cases) != NATURAL_COUNTS['natural_train']
            or sum(c.get('author') == NATURAL_AUTHOR for c in development) != NATURAL_COUNTS['natural_development']
            or sum(c.get('author')=='ai-cedar' for c in cases) != 400):
        raise ValueError('The natural continuation must keep the generated cohort and add the declared natural cases')
    if max_batches is None and not natural and (manifest != warm_report['dataset_manifest']
            or manifest != teacher_report['dataset_manifest'] or len(cases) != 26952
            or len(development) != 4464 or not manifest['original_400_retained']
            or sum(c.get('author')=='ai-cedar' for c in cases) != 400):
        raise ValueError('The full transfer run must preserve the exact trained corpus and all supplied cases')
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision('highest')
    selected, hardware = select_device(device, 'NVIDIA L4')
    if selected.type == 'cuda' and not torch.cuda.is_bf16_supported():
        raise ValueError('Declared L4 BF16 training is unavailable')
    model.to(selected)
    teacher.to(selected).eval().requires_grad_(False)
    teacher_initial = state_sha256(teacher)
    student_capture, teacher_capture = EncoderCapture(model.encoder), EncoderCapture(teacher.encoder)
    tracked = {'encoder': model.encoder, **{name: getattr(model.head, name) for name in
        ('bio', 'owner', 'state', 'permission', 'reference', 'original_scorer', 'decision')}}
    initial = {name: state_sha256(module) for name, module in tracked.items()}
    rows, audit = transfer_rows(cases, config, model.tokenizer, epoch=1, seed=seed, view_version=view_version)
    original = examples(cases, config)
    choices, weights = training_weights([*original, *rows], model.tokenizer)
    weights = {name: value.to(selected) for name, value in weights.items()}
    output_dir.mkdir(parents=True)
    checkpoints = output_dir/'checkpoints';checkpoints.mkdir()
    history, best, chosen_epoch, feature_cache, feature_audit = [], None, None, None, None
    started = perf_counter()
    for epoch in range(1, 5):
        attribution = epoch <= 2
        if epoch == 2:
            rows, audit = transfer_rows(cases, config, model.tokenizer, epoch=epoch, seed=seed, view_version=view_version)
        elif epoch == 3:
            rows = [e for e in original if e['row']['question_id'].startswith('attribute:')]
            audit = {'original_cases': len(cases), 'added_views': 0, 'added_independent_families': 0}
        batches = family_batches(rows, seed+epoch, batch_size=8)
        if max_batches is not None:
            batches = batches[:max_batches]
        configure_stage(model, attribution=attribution)
        if epoch == 3:
            extraction_epoch = chosen_epoch if view_version == 3 else 2
            extraction_checkpoint = checkpoints/('epoch-'+str(extraction_epoch))
            if view_version == 3:
                # Use the same selected extraction features for the learned
                # control. Load into existing modules to retain capture hooks.
                restored, _, _ = load_model(extraction_checkpoint)
                model.load_state_dict(restored.state_dict(), strict=True)
                del restored
            # Cache all designated training rows; a bounded preflight mounts a
            # separate small mechanism dataset, never the reserved cohort.
            student_capture.clear();teacher_capture.clear()
            feature_cache, feature_audit = collect_readout(model, teacher, rows,
                family_batches(rows, seed+epoch, batch_size=8), progress=emit)
            feature_audit.update(student_extraction_epoch=extraction_epoch,
                student_extraction_checkpoint_identity=file_sha256(extraction_checkpoint/'bundle-manifest.json'),
                teacher_artifact_identity=teacher_identity, training_file_sha256=manifest['files_sha256']['train.jsonl'])
            (checkpoints/'readout-cache-audit.json').write_text(json.dumps(feature_audit, indent=2)+'\n')
            configure_stage(model, attribution=False)
        stage_initial = {name: state_sha256(module) for name, module in tracked.items()}
        if epoch in (1, 3):
            groups = ([{'params': model.encoder.parameters(), 'lr': 1e-5},
                       {'params': [p for p in model.head.parameters() if p.requires_grad], 'lr': 3e-4}]
                      if attribution else [{'params': model.head.decision.parameters(), 'lr': 1e-4}])
            optimizer = torch.optim.AdamW(groups, weight_decay=.05)
            updates = math.ceil(len(batches)/4)*2
            warmup = max(1, int(.1*updates))
            def schedule(step):
                return ((step+1)/warmup if step < warmup else
                    .5*(1+math.cos(math.pi*min(1., (step-warmup)/max(1,updates-warmup)))))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        optimizer.zero_grad(set_to_none=True)
        emit('transfer_stage_started', epoch=epoch, attribution=attribution, question_rows=len(rows),
             batches=len(batches), view_audit=audit, teacher_training_only=True)
        total_loss, seen, transfer_totals = 0., 0, {name: 0. for name in COEFFICIENTS}
        for batch, indices in enumerate(batches):
            current = [rows[i] for i in indices]
            if attribution:
                with torch.autocast(device_type=selected.type, dtype=torch.bfloat16, enabled=selected.type=='cuda'):
                    output = model.forward_batch([e['row'] for e in current])
                    with torch.no_grad():
                        teacher_output = teacher.forward_batch([e['row'] for e in current])
                    transfer = extraction_transfer(output, teacher_output, current, student_capture.hidden, teacher_capture.hidden)
                    loss = gold_loss(output, current, choices, weights) + sum(COEFFICIENTS[k]*v for k,v in transfer.items())
                for name, value in transfer.items():
                    transfer_totals[name] += float(value.detach())*len(current)
            else:
                # FP32 cached features and actual trainable head, no encoder pass.
                loss = cached_readout_loss(model, feature_cache, indices, rows, choices)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite teacher transfer loss')
            start = batch//4*4
            count = sum(len(x) for x in batches[start:start+4])
            (loss*len(indices)/count).backward()
            if (batch+1)%4 == 0 or batch+1 == len(batches):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
                optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach())*len(current);seen += len(current)
            student_capture.clear();teacher_capture.clear()
            if (batch+1)%256 == 0:
                emit('transfer_batches', epoch=epoch, batch=batch+1, total=len(batches))
        if any(p.grad is not None for p in teacher.parameters()) or state_sha256(teacher) != teacher_initial:
            raise ValueError('Frozen teacher changed')
        changed = {name: state_sha256(module) != stage_initial[name] for name, module in tracked.items()}
        if (attribution and changed['decision']) or (not attribution and any(v for k,v in changed.items() if k!='decision')):
            raise ValueError('Transfer changed a module declared frozen')
        summary, records, cache = evaluate(model, development, config)
        student_capture.clear()
        history.append({'epoch': epoch, 'stage': 'attribution_transfer' if attribution else 'cached_learned_readout',
            'loss': total_loss/seen, 'view_audit': audit, 'stage_changed_modules': changed,
            'mean_transfer_terms': {k:v/seen for k,v in transfer_totals.items()} if attribution else None,
            'teacher_unchanged': True, 'development': summary})
        report = {'kind': 'isolated_attribution_comparison_training',
            'training_recipe': ('teacher-transfer-natural-4' if revision == 1 else 'teacher-transfer-natural-5') if natural else {1:'teacher-transfer-1', 2:'teacher-transfer-wording-2',
                                3:'teacher-transfer-plain-3'}[view_version],
            'attribution_view_version': view_version,
            'runtime': hardware, 'source_manifest': source_manifest, 'dataset_manifest': manifest,
            'warm_artifact_identity': student_identity, 'teacher_artifact_identity': teacher_identity,
            'teacher_training_only': True, 'teacher_state_sha256': teacher_initial,
            'transfer_coefficients': COEFFICIENTS, 'transfer_temperature': 2., 'readout_cache_audit': feature_audit,
            'training': {'seed': seed, 'epochs_run': epoch, 'max_batches': max_batches, 'history': history,
                'training_cases': len(cases), 'development_cases': len(development),
                'architecture': warm_report['training']['architecture'],
                'changed_modules': {name: state_sha256(module) != initial[name] for name,module in tracked.items()}},
            'selected_development': summary, 'author100_used': False, 'reserved_evaluation_used': False,
            'calibrated': False, 'full_pipeline_qualified': False, 'deployment_qualified': False}
        destination = checkpoints/('epoch-'+str(epoch))
        save_bundle(destination, model, config, report, source_manifest)
        (destination/'development-results.json').write_text(json.dumps(records, allow_nan=False)+'\n')
        write_manifest(destination)
        torch.save(cache.storage, checkpoints/('epoch-'+str(epoch)+'-development-cache.pt'))
        del cache
        score = rank(summary)
        if best is None or score > best:
            best, chosen_epoch = score, epoch
        (checkpoints/'history.json').write_text(json.dumps(history, indent=2)+'\n')
        emit('transfer_epoch', epoch=epoch, pipelines={k:{n:v[n] for n in ('eligible','base_eligible',
            'safety_false_ready','actual_identification_macro_f1','oracle_identification_macro_f1','routing')}
            for k,v in summary['pipelines'].items()}, ownership=summary['exact_source_ownership'])
    student_capture.close();teacher_capture.close()
    chosen = checkpoints/('epoch-'+str(chosen_epoch))
    cpu, _, _ = load_model(chosen)
    reference, _, _ = load_model(chosen);reference.to(selected)
    from .task_data import case_config
    parity = []
    for case in [c for c in development if c['primary']][:8]:
        for mode in ('learned','hybrid'):
            kwargs = {'config':case_config(case,config),'policy_id':case.get('policy_id'),
                      'lookup_failure':case.get('lookup_failure',False)}
            actual = ComparisonRuntime(cpu,config,'reload',mode=mode,diagnostic=True).evaluate(case['request'],**kwargs)
            expected = ComparisonRuntime(reference,config,'reload',mode=mode,diagnostic=True).evaluate(case['request'],**kwargs)
            parity.append(all(actual.get(k)==expected.get(k) for k in ('group','identification','status','clarification_ids','notice_ids')))
    if not all(parity):
        raise ValueError('Complete transfer pipeline changed after CPU reload')
    report = json.loads((chosen/'training-report.json').read_text())
    report['training'].update(selected_epoch=chosen_epoch,epochs_run=4,history=history,seconds=perf_counter()-started)
    report['fresh_cpu_reload'] = {'complete_pipeline_comparisons':len(parity),'all_match':all(parity)}
    report['runtime']['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(selected) if selected.type=='cuda' else None
    report['final_teacher_unchanged'] = state_sha256(teacher)==teacher_initial
    if not report['final_teacher_unchanged']:
        raise ValueError('Teacher changed after final reload verification')
    report['readout_cache_audit'] = feature_audit
    shutil.copytree(chosen,output_dir/'model')
    (output_dir/'model/training-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    write_manifest(output_dir/'model')
    (checkpoints/'run-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    emit('transfer_training_complete',selected_epoch=chosen_epoch)
    return report
