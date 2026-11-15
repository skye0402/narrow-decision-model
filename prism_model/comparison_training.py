"""Full-corpus joint training; select on both real composed development paths."""
from collections import Counter
import json
import math
from pathlib import Path
import shutil
from time import perf_counter

import torch
from torch import nn

from .comparison_quality import evaluate
from .comparison_runtime import ComparisonRuntime
from .curriculum_training import choice_losses, family_batches
from .evidence_data import load_partition
from .ownership_bundle import load_model, save_bundle, write_manifest
from .ownership_data import examples
from .ownership_format import BIO_LABELS, OWNERS, STATES, factor_targets
from .ownership_model import OwnershipHead, OwnershipModel
from .provenance import file_sha256, verify_source
from .span_bundle import load_model as load_warm
from .span_format import layout
from .training import emit, select_device, state_sha256

WARM_IDENTITY = '08371fb9ec8e7631e5b80a40627758bc2527065c1a8ca3a0d2f1669cf47a45e5'


def class_weights(counts, size):
    total, present = sum(counts.values()), len(counts)
    return torch.tensor([min(8., max(.25, math.sqrt(total / (present * counts[i])))) if counts[i] else 1.
                         for i in range(size)])


def auxiliary_loss(logits, labels, weights):
    return (nn.functional.cross_entropy(logits, labels, weight=weights)
            if bool((labels != -100).any()) else logits.sum() * 0.)


def training_weights(rows, tokenizer):
    counts = {k: Counter() for k in ('bio', 'owner', 'state', 'permission')}
    choice, totals = Counter(), Counter()
    for e in rows:
        key = e['row']['question_id']
        choice[(key, e['label'])] += 1
        totals[key] += 1
        if e['annotation'] is not None:
            attribute = key.removeprefix('attribute:')
            mapping = layout(tokenizer, e['row'], attribute)
            for name, targets in factor_targets(mapping, e['annotation'], attribute).items():
                counts[name].update(int(x) for x in targets if int(x) >= 0)
        if e['reference_target'] is not None:
            counts['permission'][e['reference_target']['permission']] += 1
    kinds = Counter(key for key, _ in choice)
    choices = {key: min(8., max(.25, math.sqrt(totals[key[0]] / (kinds[key[0]] * n)))) for key, n in choice.items()}
    weights = {name: class_weights(counts[name], size) for name, size in
               (('bio', len(BIO_LABELS)), ('owner', len(OWNERS)), ('state', len(STATES)), ('permission', 2))}
    return choices, weights


def joint_loss(output, rows, choice_weights, auxiliary_weights):
    device = output['decisions'].device
    targets = torch.tensor([e['target'] for e in rows], device=device)
    losses = choice_losses(output['decisions'], targets, [len(e['row']['option_ids']) for e in rows], smoothing=.05)
    factors = torch.tensor([choice_weights[(e['row']['question_id'], e['label'])] for e in rows], device=device)
    total = []
    for i, e in enumerate(rows):
        value = losses[i] * factors[i]
        if e['annotation'] is not None:
            attribute = e['row']['question_id'].removeprefix('attribute:')
            labels = factor_targets(output['mappings'][i], e['annotation'], attribute)
            for name, multiplier in (('bio', 1.), ('owner', 2.), ('state', 1.)):
                target = labels[name].to(device)
                value = value + multiplier * auxiliary_loss(output[name][i, :len(target)], target, auxiliary_weights[name])
        if target := e['reference_target']:
            permission = torch.tensor([target['permission']], device=device)
            value = value + nn.functional.cross_entropy(output['permission'][i:i + 1], permission,
                                                        weight=auxiliary_weights['permission'])
            ids = output['component_ids'][i]
            pointer = (ids.index(target['component_id']) if target['mode'] == 'selected'
                       else len(ids) + int(target['mode'] == 'unresolved'))
            value = value + nn.functional.cross_entropy(output['reference'][i].unsqueeze(0),
                                                       torch.tensor([pointer], device=device))
        total.append(value)
    return torch.stack(total).mean()


def rank(summary):
    hybrid, learned = summary['pipelines']['hybrid'], summary['pipelines']['learned']
    other = summary['exact_source_ownership']['other_item']
    return [*hybrid['selection_rank'], other['recall'], other['precision'],
            learned['base_eligible'], -learned['safety_false_ready'], learned['actual_identification_macro_f1']]


def train(*, data_dir, warm_dir, output_dir, source_root, source_manifest,
          device='cuda', epochs=2, seed=79, max_batches=None):
    data_dir, warm_dir, output_dir = map(Path, (data_dir, warm_dir, output_dir))
    if output_dir.exists() or type(epochs) is not int or not 1 <= epochs <= 2:
        raise ValueError('Use new output and at most the declared two epochs')
    if max_batches is not None and device != 'cpu':
        raise ValueError('A partial batch run is a CPU mechanism check only')
    verify_source(source_root, source_manifest)
    config, cases, manifest = load_partition(data_dir, 'train', allowed_splits={'train', 'development'})
    _, development, _ = load_partition(data_dir, 'development', allowed_splits={'train', 'development'})
    if (set(manifest['partitions']) != {'train', 'development'} or manifest.get('generator') != 'ownership-comparison-data-1'
            or manifest['exposed_author100_included'] or manifest['previous_evaluation_included']):
        raise ValueError('Only the declared training/development roles may be mounted')
    if {c['family_id'] for c in cases} & {c['family_id'] for c in development}:
        raise ValueError('Family leak')
    if file_sha256(warm_dir / 'bundle-manifest.json') != WARM_IDENTITY:
        raise ValueError('The declared raw warm model changed')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(seed)
    selected, hardware = select_device(device, 'NVIDIA L4')
    if selected.type == 'cuda' and not torch.cuda.is_bf16_supported():
        raise ValueError('BF16 required on L4')
    warm, previous_config, _ = load_warm(warm_dir)
    if previous_config != config:
        raise ValueError('Different purchasing policy')
    head = OwnershipHead(warm.head.hidden_size, warm.head.original_scorer, warm.head.decision, warm.head.head_size)
    model = OwnershipModel(warm.encoder, warm.tokenizer, head).to(selected)
    del warm
    model.requires_grad_(True)
    tracked = {'encoder': model.encoder, **{name: getattr(model.head, name) for name in
               ('bio', 'owner', 'state', 'permission', 'reference', 'original_scorer', 'decision')}}
    initial = {name: state_sha256(module) for name, module in tracked.items()}
    rows = examples(cases, config, epoch=1, seed=seed)
    choices, weights = training_weights(rows, model.tokenizer)
    weights = {name: value.to(selected) for name, value in weights.items()}
    plans = [family_batches(rows, seed + epoch, batch_size=8) for epoch in range(epochs)]
    if max_batches is not None:
        plans = [batches[:max_batches] for batches in plans]
    updates = sum(math.ceil(len(b) / 4) for b in plans)
    warmup = max(1, int(.1 * updates))
    optimizer = torch.optim.AdamW([{'params': model.encoder.parameters(), 'lr': 5e-6},
                                   {'params': model.head.parameters(), 'lr': 5e-4}], weight_decay=.05)
    def rate(step):
        return ((step + 1) / warmup if step < warmup else
                .5 * (1 + math.cos(math.pi * min(1., (step - warmup) / max(1, updates - warmup)))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    output_dir.mkdir(parents=True)
    checkpoint_root = output_dir / 'checkpoints'
    checkpoint_root.mkdir()
    history, best_rank, best_epoch = [], None, 0
    started = perf_counter()
    semantic_order = [(e['case_id'], e['row']['question_id'], e['target']) for e in rows]
    emit('comparison_training_started', training_cases=len(cases), development_cases=len(development),
         question_rows=len(rows), epochs=epochs, device=str(selected))
    for epoch, batches in enumerate(plans, 1):
        if epoch > 1:
            del rows
            rows = examples(cases, config, epoch=epoch, seed=seed)
            if semantic_order != [(e['case_id'], e['row']['question_id'], e['target']) for e in rows]:
                raise ValueError('A training wording view changed decision labels/order')
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, seen = 0., 0
        for batch, indices in enumerate(batches):
            chosen = [rows[i] for i in indices]
            with torch.autocast(device_type=selected.type, dtype=torch.bfloat16, enabled=selected.type == 'cuda'):
                output = model.forward_batch([e['row'] for e in chosen])
                loss = joint_loss(output, chosen, choices, weights)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite full comparison loss')
            start = batch // 4 * 4
            count = sum(len(x) for x in batches[start:start + 4])
            (loss * len(indices) / count).backward()
            if (batch + 1) % 4 == 0 or batch + 1 == len(batches):
                nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach()) * len(chosen)
            seen += len(chosen)
            if (batch + 1) % 256 == 0:
                emit('comparison_batches', epoch=epoch, batch=batch + 1, total=len(batches))
        summary, records, cache = evaluate(model, development, config)
        history.append({'epoch': epoch, 'loss': total_loss / seen, 'development': summary})
        report = {'kind': 'full_ownership_comparison_training', 'runtime': hardware,
                  'source_manifest': source_manifest, 'dataset_manifest': manifest,
                  'warm_artifact_identity': WARM_IDENTITY, 'training': {'seed': seed, 'epochs_run': epoch,
                   'training_cases': len(cases), 'development_cases': len(development), 'question_rows': len(rows),
                   'max_batches': max_batches, 'history': history,
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
        if best_rank is None or score > best_rank:
            best_rank, best_epoch = score, epoch
        (checkpoint_root / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
        emit('comparison_epoch', epoch=epoch, pipelines={key: {k: v for k, v in value.items() if k in
             ('eligible', 'base_eligible', 'safety_false_ready', 'actual_identification_macro_f1', 'oracle_identification_macro_f1', 'routing')}
             for key, value in summary['pipelines'].items()}, exact_source_ownership=summary['exact_source_ownership'])
    chosen = checkpoint_root / ('epoch-' + str(best_epoch))
    cpu, _, _ = load_model(chosen)
    reference, _, _ = load_model(chosen)
    reference.to(selected)
    parity = []
    for case in [c for c in development if c['primary']][:8]:
        from .task_data import case_config
        local = case_config(case, config)
        for mode in ('learned', 'hybrid'):
            expected = ComparisonRuntime(reference, config, 'reload', mode=mode, diagnostic=True).evaluate(case['request'], config=local)
            actual = ComparisonRuntime(cpu, config, 'reload', mode=mode, diagnostic=True).evaluate(case['request'], config=local)
            keys = ('group', 'identification', 'status', 'clarification_ids', 'notice_ids')
            parity.append(all(actual.get(k) == expected.get(k) for k in keys))
    if not all(parity):
        raise ValueError('GPU-to-CPU reload changed a complete pipeline result')
    report = json.loads((chosen / 'training-report.json').read_text())
    report['training'].update(selected_epoch=best_epoch, epochs_run=epochs, history=history, seconds=perf_counter() - started)
    report['fresh_cpu_reload'] = {'complete_pipeline_comparisons': len(parity), 'all_match': all(parity)}
    report['runtime']['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(selected) if selected.type == 'cuda' else None
    shutil.copytree(chosen, output_dir / 'model')
    (output_dir / 'model/training-report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    write_manifest(output_dir / 'model')
    (checkpoint_root / 'run-report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    emit('comparison_training_complete', selected_epoch=best_epoch, pipelines=report['selected_development']['pipelines'])
    return report
