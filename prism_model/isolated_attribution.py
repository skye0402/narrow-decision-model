"""Train source extraction without attribute-decision gradient interference."""
import json
import math
from pathlib import Path
import shutil
from time import perf_counter

import torch

from .attribution_views import view
from .comparison_quality import evaluate
from .comparison_runtime import ComparisonRuntime
from .comparison_training import auxiliary_loss, rank, training_weights
from .curriculum_training import choice_losses, family_batches
from .evidence_data import load_partition
from .ownership_bundle import load_model, save_bundle, write_manifest
from .ownership_data import augment_training, examples
from .ownership_format import factor_targets
from .provenance import file_sha256, verify_source
from .training import emit, select_device, state_sha256

WARM_IDENTITY = 'eab72635b5b39f3ad62f4171c814b6ca10fcbcfd4b4875384dba2a0045d6688b'


def attribution_rows(cases, config, tokenizer, *, epoch, seed=83, max_tokens=512, view_version=1):
    if view_version not in (1, 2, 3):
        raise ValueError('Undeclared attribution view version')
    originals = [augment_training(c, epoch, seed) for c in cases
                 if c['primary'] or 'evidence_annotations' in c or c.get('author') == 'ai-cedar']
    added, omitted = [], []
    for c in cases:
        if not c['primary']:
            continue
        if view_version == 3:
            from .unmarked_views import view as plain_view
            candidate = plain_view(c, epoch=epoch, seed=seed)
        else:
            candidate = view(c, epoch=epoch, seed=seed, version=view_version)
        if candidate is None:
            continue
        rows = examples([candidate], config)
        maximum = max(len(tokenizer(e['row']['text'], truncation=False)['input_ids']) for e in rows)
        if maximum > max_tokens:
            omitted.append({'parent_case_id': c['case_id'], 'tokens': maximum})
        else:
            added.append(candidate)
    rows = [e for e in examples([*originals, *added], config)
            if not e['row']['question_id'].startswith('attribute:') or e['annotation'] is not None]
    return rows, {'original_cases': len(originals), 'added_views': len(added), 'omitted_overlength_views': omitted,
                  'added_independent_families': 0, 'supplied_source_cases': sum(c.get('author') == 'ai-cedar' for c in originals)}


def isolated_loss(output, rows, choice_weights, auxiliary_weights, *, attribution):
    terms = []
    for i, e in enumerate(rows):
        attribute = e['row']['question_id'].startswith('attribute:')
        if attribution and attribute:
            if e['annotation'] is None:
                raise ValueError('Unannotated attribute rows have no extraction supervision')
            targets = factor_targets(output['mappings'][i], e['annotation'], e['row']['question_id'].removeprefix('attribute:'))
            value = sum(multiplier * auxiliary_loss(output[name][i, :len(targets[name])],
                        targets[name].to(output[name].device), auxiliary_weights[name])
                        for name, multiplier in (('bio', 1.), ('owner', 2.), ('state', 1.)))
        else:
            width = len(e['row']['option_ids'])
            value = choice_losses(output['decisions'][i:i + 1, :width],
                                  torch.tensor([e['target']], device=output['decisions'].device), [width], smoothing=.05)[0]
            value = value * choice_weights[(e['row']['question_id'], e['label'])]
        if attribution and e['reference_target'] is not None:
            target, ids = e['reference_target'], output['component_ids'][i]
            permission = torch.tensor([target['permission']], device=output['permission'].device)
            value = value + torch.nn.functional.cross_entropy(output['permission'][i:i + 1], permission,
                                                               weight=auxiliary_weights['permission'])
            pointer = ids.index(target['component_id']) if target['mode'] == 'selected' else len(ids) + int(target['mode'] == 'unresolved')
            value = value + torch.nn.functional.cross_entropy(output['reference'][i][None],
                              torch.tensor([pointer], device=output['reference'][i].device))
        terms.append(value)
    return torch.stack(terms).mean()


def configure_stage(model, *, attribution):
    model.requires_grad_(False)
    if attribution:
        model.encoder.requires_grad_(True)
        for name in ('bio', 'owner', 'state', 'permission', 'reference', 'original_scorer'):
            getattr(model.head, name).requires_grad_(True)
        model.head.reference_specials.requires_grad_(True)
    else:
        model.head.decision.requires_grad_(True)
    model.train()
    if not attribution:
        model.encoder.eval()


def train(*, data_dir, warm_dir, output_dir, source_root, source_manifest,
          device='cuda', seed=83, max_batches=None, compact=False):
    data_dir, warm_dir, output_dir = map(Path, (data_dir, warm_dir, output_dir))
    if output_dir.exists() or (max_batches is not None and (device != 'cpu' or type(max_batches) is not int or max_batches < 1)):
        raise ValueError('Use a new output; truncated runs are CPU mechanism checks only')
    verify_source(source_root, source_manifest)
    if type(compact) is not bool:
        raise ValueError('Declare the compact experiment explicitly')
    warm_identity = WARM_IDENTITY
    if compact:
        from .compact_encoder import WARM_IDENTITY as compact_warm_identity
        warm_identity = compact_warm_identity
    if file_sha256(warm_dir / 'bundle-manifest.json') != warm_identity:
        raise ValueError('The declared raw warm artifact changed')
    model, config, _ = load_model(warm_dir)
    architecture = None
    if compact:
        from .compact_encoder import reduce_depth
        architecture = reduce_depth(model)
    local, cases, manifest = load_partition(data_dir, 'train', allowed_splits={'train', 'development'})
    _, development, _ = load_partition(data_dir, 'development', allowed_splits={'train', 'development'})
    if (config != local or set(manifest['partitions']) != {'train', 'development'}
            or manifest.get('generator') != 'ownership-comparison-data-1'
            or manifest['exposed_author100_included'] or manifest['previous_evaluation_included']
            or {c['family_id'] for c in cases} & {c['family_id'] for c in development}):
        raise ValueError('Only the unchanged training/development cohort is allowed')
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision('highest')
    selected, hardware = select_device(device, 'NVIDIA L4')
    if selected.type == 'cuda' and not torch.cuda.is_bf16_supported():
        raise ValueError('BF16 is required for the declared L4 experiment')
    model.to(selected)
    tracked = {'encoder': model.encoder, **{name: getattr(model.head, name) for name in
               ('bio', 'owner', 'state', 'permission', 'reference', 'original_scorer', 'decision')}}
    initial = {name: state_sha256(module) for name, module in tracked.items()}
    rows, audit = attribution_rows(cases, config, model.tokenizer, epoch=1, seed=seed)
    original = examples(cases, config)
    choices, weights = training_weights([*original, *rows], model.tokenizer)
    weights = {key: value.to(selected) for key, value in weights.items()}
    output_dir.mkdir(parents=True)
    checkpoint_root = output_dir / 'checkpoints'
    checkpoint_root.mkdir()
    history, best, chosen_epoch = [], None, None
    started = perf_counter()
    optimizer = None
    for epoch in range(1, 4):
        attribution = epoch < 3
        if epoch == 2:
            rows, audit = attribution_rows(cases, config, model.tokenizer, epoch=epoch, seed=seed)
        elif epoch == 3:
            rows = [e for e in original if e['row']['question_id'].startswith('attribute:')]
            audit = {'original_cases': len(cases), 'added_views': 0, 'added_independent_families': 0}
        batches = family_batches(rows, seed + epoch, batch_size=8)
        if max_batches is not None:
            batches = batches[:max_batches]
        configure_stage(model, attribution=attribution)
        stage_initial = {name: state_sha256(module) for name, module in tracked.items()}
        if epoch in (1, 3):
            groups = ([{'params': model.encoder.parameters(), 'lr': 1e-5},
                       {'params': [p for p in model.head.parameters() if p.requires_grad], 'lr': 3e-4}] if attribution
                      else [{'params': model.head.decision.parameters(), 'lr': 1e-4}])
            optimizer = torch.optim.AdamW(groups, weight_decay=.05)
            updates = math.ceil(len(batches) / 4) * (2 if attribution else 1)
            warmup = max(1, int(.1 * updates))
            def schedule(step):
                return ((step + 1) / warmup if step < warmup else
                        .5 * (1 + math.cos(math.pi * min(1., (step - warmup) / max(1, updates - warmup)))))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        optimizer.zero_grad(set_to_none=True)
        emit('isolated_stage_started', epoch=epoch, attribution=attribution, question_rows=len(rows),
             batches=len(batches), added_views=audit['added_views'],
             omitted_overlength_views=len(audit.get('omitted_overlength_views', [])))
        total_loss, seen = 0., 0
        for batch, indices in enumerate(batches):
            current = [rows[i] for i in indices]
            with torch.autocast(device_type=selected.type, dtype=torch.bfloat16, enabled=selected.type == 'cuda'):
                output = model.forward_batch([e['row'] for e in current])
                loss = isolated_loss(output, current, choices, weights, attribution=attribution)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite isolated attribution loss')
            start = batch // 4 * 4
            count = sum(len(x) for x in batches[start:start + 4])
            (loss * len(indices) / count).backward()
            if (batch + 1) % 4 == 0 or batch + 1 == len(batches):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach()) * len(current)
            seen += len(current)
            if (batch + 1) % 256 == 0:
                emit('isolated_batches', epoch=epoch, batch=batch + 1, total=len(batches))
        stage_changed = {name: state_sha256(module) != stage_initial[name] for name, module in tracked.items()}
        if (attribution and stage_changed['decision']) or (not attribution and any(
                changed for name, changed in stage_changed.items() if name != 'decision')):
            raise ValueError('A stage changed a module declared frozen')
        summary, records, cache = evaluate(model, development, config)
        history.append({'epoch': epoch, 'stage': 'attribution' if attribution else 'learned_readout',
                        'loss': total_loss / seen, 'view_audit': audit, 'stage_changed_modules': stage_changed,
                        'development': summary})
        report = {'kind': 'isolated_attribution_comparison_training', 'runtime': hardware,
                  'source_manifest': source_manifest, 'dataset_manifest': manifest, 'warm_artifact_identity': warm_identity,
                  'training': {'seed': seed, 'epochs_run': epoch, 'max_batches': max_batches, 'history': history,
                    'training_cases': len(cases), 'development_cases': len(development),
                    'architecture': architecture,
                    'changed_modules': {name: state_sha256(module) != initial[name] for name, module in tracked.items()}},
                  'selected_development': summary, 'author100_used': False, 'reserved_evaluation_used': False,
                  'calibrated': False, 'full_pipeline_qualified': False, 'deployment_qualified': False}
        destination = checkpoint_root / ('epoch-' + str(epoch))
        save_bundle(destination, model, config, report, source_manifest)
        (destination / 'development-results.json').write_text(json.dumps(records, allow_nan=False) + '\n')
        write_manifest(destination)
        torch.save(cache.storage, checkpoint_root / ('epoch-' + str(epoch) + '-development-cache.pt'))
        del cache
        score = rank(summary)
        if best is None or score > best:
            best, chosen_epoch = score, epoch
        (checkpoint_root / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
        emit('isolated_epoch', epoch=epoch, pipelines={key: {k: v for k, v in value.items() if k in
             ('eligible', 'base_eligible', 'safety_false_ready', 'actual_identification_macro_f1',
              'oracle_identification_macro_f1', 'routing')} for key, value in summary['pipelines'].items()},
             ownership=summary['exact_source_ownership'])
    chosen = checkpoint_root / ('epoch-' + str(chosen_epoch))
    cpu, _, _ = load_model(chosen)
    reference, _, _ = load_model(chosen)
    reference.to(selected)
    from .task_data import case_config
    parity = []
    for case in [c for c in development if c['primary']][:8]:
        for mode in ('learned', 'hybrid'):
            kwargs = {'config': case_config(case, config), 'policy_id': case.get('policy_id'),
                      'lookup_failure': case.get('lookup_failure', False)}
            actual = ComparisonRuntime(cpu, config, 'reload', mode=mode, diagnostic=True).evaluate(case['request'], **kwargs)
            expected = ComparisonRuntime(reference, config, 'reload', mode=mode, diagnostic=True).evaluate(case['request'], **kwargs)
            parity.append(all(actual.get(k) == expected.get(k) for k in ('group', 'identification', 'status', 'clarification_ids', 'notice_ids')))
    if not all(parity):
        raise ValueError('A complete pipeline result changed after CPU reload')
    report = json.loads((chosen / 'training-report.json').read_text())
    report['training'].update(selected_epoch=chosen_epoch, epochs_run=3, history=history, seconds=perf_counter() - started)
    report['fresh_cpu_reload'] = {'complete_pipeline_comparisons': len(parity), 'all_match': all(parity)}
    report['runtime']['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(selected) if selected.type == 'cuda' else None
    shutil.copytree(chosen, output_dir / 'model')
    (output_dir / 'model/training-report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    write_manifest(output_dir / 'model')
    (checkpoint_root / 'run-report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    emit('isolated_training_complete', selected_epoch=chosen_epoch)
    return report
