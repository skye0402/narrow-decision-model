"""Bounded full-encoder prototype with auxiliary, writer-recorded BIO loss."""
from collections import Counter
import json
import math
from pathlib import Path
import shutil
from time import perf_counter

import torch
from torch import nn

from .curriculum_training import family_batches
from .evidence_data import load_partition
from .evidence_span_data import ROLES
from .metrics import classification
from .model_v4 import load_model as load_warm
from .provenance import file_sha256, verify_source
from .span_bundle import load_model, save_bundle, write_manifest
from .span_format import BIO_LABELS, auxiliary_targets, layout
from .span_model import AttributedDecisionModel, EvidenceHead
from .training import emit, select_device, state_sha256
from scripts.prepare_span_data import target_examples

WARM_IDENTITY = 'b59e4f11cee5ecc01b2ce9f4406a910e2c887f994b637823f37313d61f022d2e'


def bio_spans(tags):
    result = set()
    current, start = None, None
    for i, name in enumerate([*tags, 'O']):
        prefix, role = name.split('-', 1) if name != 'O' else ('O', None)
        if current is not None and (prefix != 'I' or role != current):
            result.add((start, i, current))
            current, start = None, None
        if role is not None and current is None:
            current, start = role, i
    return result


def evaluate(model, examples, batch_size=8):
    model.eval()
    records = []
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            chosen = examples[start:start + batch_size]
            decisions, tokens, mappings = model.forward_batch([e['row'] for e in chosen])
            if not bool(torch.isfinite(decisions).all()) or not bool(torch.isfinite(tokens).all()):
                raise ValueError('Nonfinite prototype development predictions')
            for i, (e, mapping) in enumerate(zip(chosen, mappings)):
                count = len(mapping['offsets'])
                expected = auxiliary_targets(mapping, e['annotation'], e['attribute'])
                predicted = tokens[i, :count].argmax(-1).cpu()
                valid = expected != -100
                gold_spans = bio_spans([BIO_LABELS[int(x)] for x in expected[valid]])
                actual_spans = bio_spans([BIO_LABELS[int(x)] for x in predicted[valid]])
                choice = int(decisions[i, :len(e['row']['option_ids'])].argmax())
                records.append({**{k: e[k] for k in ('case_id', 'family_id', 'primary', 'attribute', 'label')},
                                'choice': e['row']['option_ids'][choice], 'input_sha256': e['row']['text_sha256'],
                                'logits': decisions[i, :len(e['row']['option_ids'])].float().cpu().tolist(),
                                'gold_spans': sorted(gold_spans), 'predicted_spans': sorted(actual_spans)})
    primary = [r for r in records if r['primary']]
    semantics = classification([r['label'] for r in primary], [r['choice'] for r in primary],
                               ['satisfied', 'missing', 'conflicting', 'unclear'])
    predicted_ready = [r for r in primary if r['choice'] == 'satisfied']
    true_ready = sum(r['label'] == 'satisfied' for r in primary)
    correct_ready = sum(r['label'] == 'satisfied' for r in predicted_ready)
    precision = correct_ready / len(predicted_ready) if predicted_ready else 0.
    recall = correct_ready / true_ready if true_ready else 0.
    safety_false = sum(r['choice'] == 'satisfied' and r['label'] in ('conflicting', 'unclear') for r in primary)
    spans = {}
    for role in ROLES:
        correct = predicted = expected = 0
        for r in primary:
            gold = {tuple(s) for s in r['gold_spans'] if s[2] == role}
            actual = {tuple(s) for s in r['predicted_spans'] if s[2] == role}
            correct += len(gold & actual)
            predicted += len(actual)
            expected += len(gold)
        spans[role] = {'correct': correct, 'predicted': predicted, 'expected': expected,
                       'precision': correct / predicted if predicted else 0.,
                       'recall': correct / expected if expected else 0.,
                       'f1': 2 * correct / (predicted + expected) if predicted + expected else 0.}
    total_correct = sum(s['correct'] for s in spans.values())
    total_spans = sum(s['predicted'] + s['expected'] for s in spans.values())
    micro_f1 = 2 * total_correct / total_spans if total_spans else 0.
    base = semantics['macro_f1'] >= .85 and recall >= .60
    false = len(predicted_ready) - correct_ready
    eligible = base and precision >= .95 and safety_false == 0 and micro_f1 >= .90 and spans['current']['recall'] >= .90
    summary = {'primary_families': len(primary), 'decisions': semantics,
               'satisfied_predictions': len(predicted_ready), 'true_satisfied': true_ready,
               'satisfied_precision': precision, 'satisfied_recall': recall,
               'false_satisfied': false, 'safety_false_satisfied': safety_false,
               'span_metrics': spans, 'span_micro_f1': micro_f1, 'base_eligible': base,
               'prototype_eligible': eligible,
               'selection_rank': [base, -safety_false, -false, semantics['macro_f1'], micro_f1]}
    return summary, records


def class_weights(counts, size):
    total = sum(counts.values())
    present = len(counts)
    return torch.tensor([min(8., max(.25, math.sqrt(total / (present * counts[i])))) if counts[i] else 1.
                         for i in range(size)])


def train(*, data_dir, warm_dir, output_dir, source_root, source_manifest,
          device='cuda', epochs=2, seed=78, max_batches=None):
    data_dir, warm_dir, output_dir = map(Path, (data_dir, warm_dir, output_dir))
    if output_dir.exists() or type(epochs) is not int or not 1 <= epochs <= 2:
        raise ValueError('Use a new output and the predeclared two-epoch maximum')
    if max_batches is not None and device != 'cpu':
        raise ValueError('Batch limits are CPU mechanism checks only')
    verify_source(source_root, source_manifest)
    config, cases, manifest = load_partition(data_dir, 'train', allowed_splits={'train', 'development'})
    _, development, _ = load_partition(data_dir, 'development', allowed_splits={'train', 'development'})
    if set(manifest['partitions']) != {'train', 'development'} or manifest.get('writer_recorded_spans') is not True:
        raise ValueError('Only annotated optimizer/selection roles may be mounted')
    if manifest['exposed_author100_included'] or manifest['previous_evaluation_included']:
        raise ValueError('Previous evaluation cannot train or select this model')
    if {c['family_id'] for c in cases} & {c['family_id'] for c in development}:
        raise ValueError('Train/development family overlap')
    if file_sha256(warm_dir / 'bundle-manifest.json') != WARM_IDENTITY or (warm_dir / 'calibration.json').exists():
        raise ValueError('The declared raw warm model changed')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(seed)
    selected, hardware = select_device(device, 'NVIDIA L4')
    if selected.type == 'cuda' and not torch.cuda.is_bf16_supported():
        raise ValueError('BF16 required on the declared GPU')
    previous, old_config, _ = load_warm(warm_dir)
    if old_config != config:
        raise ValueError('Policy configuration changed')
    head = EvidenceHead(previous.encoder.config.hidden_size, previous.head)
    model = AttributedDecisionModel(previous.encoder, previous.tokenizer, head).to(selected)
    del previous
    model.encoder.requires_grad_(True)
    model.head.original_scorer.requires_grad_(False)
    before_encoder, before_head = state_sha256(model.encoder), state_sha256(model.head)
    before_attribution = state_sha256(model.head.attribution)
    before_decision = state_sha256(model.head.decision)
    before_original = state_sha256(model.head.original_scorer)
    examples, dev = target_examples(cases, config), target_examples(development, config)
    token_counts = Counter()
    for e in examples:
        mapping = layout(model.tokenizer, e['row'], e['attribute'])
        token_counts.update(int(x) for x in auxiliary_targets(mapping, e['annotation'], e['attribute']) if int(x) != -100)
    token_weights = class_weights(token_counts, len(BIO_LABELS)).to(selected)
    choice_weights = class_weights(Counter(e['target'] for e in examples), 4).to(selected)
    batches = [family_batches(examples, seed + epoch, batch_size=8) for epoch in range(epochs)]
    if max_batches is not None:
        batches = [b[:max_batches] for b in batches]
    updates = sum(math.ceil(len(b) / 4) for b in batches)
    warmup = max(1, int(.1 * updates))
    optimizer = torch.optim.AdamW([{'params': model.encoder.parameters(), 'lr': 1e-5},
                                   {'params': [p for p in model.head.parameters() if p.requires_grad], 'lr': 5e-4}], weight_decay=.01)
    def rate(step):
        return (step + 1) / warmup if step < warmup else .5 * (1 + math.cos(math.pi * min(1., (step - warmup) / max(1, updates - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    history, best_rank, best_epoch = [], None, 0
    output_dir.mkdir(parents=True)
    checkpoints = output_dir / 'checkpoints'
    checkpoints.mkdir()
    started = perf_counter()
    emit('span_training_started', train_cases=len(cases), development_cases=len(development), device=str(selected))
    for epoch, epoch_batches in enumerate(batches, 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = decision_sum = token_sum = seen = 0
        for batch, indices in enumerate(epoch_batches):
            chosen = [examples[i] for i in indices]
            with torch.autocast(device_type=selected.type, dtype=torch.bfloat16, enabled=selected.type == 'cuda'):
                logits, tokens, mappings = model.forward_batch([e['row'] for e in chosen])
                losses, auxiliary = [], []
                for i, (e, mapping) in enumerate(zip(chosen, mappings)):
                    assert len(e['row']['option_ids']) == 4
                    losses.append(nn.functional.cross_entropy(logits[i:i+1, :4], torch.tensor([e['target']], device=selected),
                                                              label_smoothing=.05, reduction='sum') * choice_weights[e['target']])
                    labels = auxiliary_targets(mapping, e['annotation'], e['attribute']).to(selected)
                    auxiliary.append(nn.functional.cross_entropy(tokens[i, :len(labels)], labels, weight=token_weights))
                decision_loss, token_loss = torch.stack(losses).mean(), torch.stack(auxiliary).mean()
                loss = decision_loss + token_loss
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite joint training loss')
            group_start = batch // 4 * 4
            group_rows = sum(len(x) for x in epoch_batches[group_start:group_start + 4])
            (loss * len(indices) / group_rows).backward()
            if (batch + 1) % 4 == 0 or batch + 1 == len(epoch_batches):
                nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += float(loss.detach()) * len(indices)
            decision_sum += float(decision_loss.detach()) * len(indices)
            token_sum += float(token_loss.detach()) * len(indices)
            seen += len(indices)
            if (batch + 1) % 128 == 0:
                emit('span_training_batches', epoch=epoch, batch=batch + 1, total=len(epoch_batches))
        summary, records = evaluate(model, dev)
        history.append({'epoch': epoch, 'loss': loss_sum / seen, 'decision_loss': decision_sum / seen,
                        'token_loss': token_sum / seen, 'selection': summary})
        report = {'kind': 'auxiliary_evidence_attribute_prototype', 'scope': 'synthetic_attribute_mechanism',
                  'runtime': hardware, 'source_manifest': source_manifest, 'dataset_manifest': manifest,
                  'warm_artifact_identity': WARM_IDENTITY, 'training': {'seed': seed, 'epochs_run': epoch,
                   'training_cases': len(cases), 'development_cases': len(development), 'history': history,
                   'encoder_changed': state_sha256(model.encoder) != before_encoder,
                   'head_changed': state_sha256(model.head) != before_head, 'max_batches': max_batches},
                  'selected_development': summary, 'author100_used': False, 'reserved_evaluation_used': False,
                  'full_pipeline_qualified': False, 'deployment_qualified': False, 'calibrated': False}
        report['training'].update(attribution_head_changed=state_sha256(model.head.attribution) != before_attribution,
                                  decision_head_changed=state_sha256(model.head.decision) != before_decision,
                                  original_scorer_changed=state_sha256(model.head.original_scorer) != before_original,
                                  trained_encoder_parameters=sum(p.numel() for p in model.encoder.parameters() if p.requires_grad),
                                  trained_head_parameters=sum(p.numel() for p in model.head.parameters() if p.requires_grad))
        path = checkpoints / ('epoch-' + str(epoch))
        save_bundle(path, model, config, report, source_manifest)
        (path / 'development-results.json').write_text(json.dumps(records, indent=2) + '\n')
        write_manifest(path)
        if best_rank is None or summary['selection_rank'] > best_rank:
            best_rank, best_epoch = summary['selection_rank'], epoch
        (checkpoints / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
        emit('span_epoch', epoch=epoch, **{k: v for k, v in summary.items() if k not in ('span_metrics', 'decisions')})
    model.eval()
    # Reload the selected checkpoint independently on CPU and compare fixed rows
    # against a GPU re-read, so the check also covers selected-checkpoint identity.
    selected_path = checkpoints / ('epoch-' + str(best_epoch))
    restored, _, _ = load_model(selected_path)
    # If the final epoch was not selected, use a freshly loaded selected model
    # on the GPU too. No comparison is made against different epoch weights.
    reference_model, _, _ = load_model(selected_path)
    reference_model.to(selected)
    checks = []
    with torch.inference_mode():
        for e in dev[:16]:
            expected = reference_model(e['row']).cpu()
            actual = restored(e['row']).cpu()
            checks.append({'same_choice': int(actual.argmax()) == int(expected.argmax()),
                           'max_logit_drift': float((actual - expected).abs().max())})
    if not all(c['same_choice'] for c in checks):
        raise ValueError('Fresh CPU reload changed selected decisions')
    report = json.loads((selected_path / 'training-report.json').read_text())
    report['training'].update(epochs_run=epochs, selected_epoch=best_epoch, history=history, seconds=perf_counter() - started)
    report['fresh_cpu_reload'] = {'rows': len(checks), 'all_choices_match': True,
                                  'max_logit_drift': max(c['max_logit_drift'] for c in checks)}
    report['runtime']['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(selected) if selected.type == 'cuda' else None
    destination = output_dir / 'model'
    shutil.copytree(selected_path, destination)
    (destination / 'training-report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    write_manifest(destination)
    (checkpoints / 'run-report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    emit('span_training_complete', selected_epoch=best_epoch, prototype_eligible=report['selected_development']['prototype_eligible'])
    return report
