"""Training-only teacher losses and exact frozen-head feature packing.

Nothing in this module is an inference parser or a deployment fallback.
Teacher tensors are always detached. Attribute decision gradients remain out
of the extraction stage; their supervised and distillation losses belong to
the later frozen-feature readout stage.
"""
import hashlib

import torch
from torch.nn import functional as F

from .ownership_format import factor_targets

STUDENT_IDENTITY = 'b0922b09210a3b08ad79fdc762c8a3a08bf666b8348c77a23c514bcd66ed6422'
TEACHER_IDENTITY = '28b45104f0888b61382c72b05436dd1646b5e54e8382cdf4107d7e5c59ba7682'
COEFFICIENTS = {'hidden': .5, 'choice': .5, 'bio': .5, 'owner': 1., 'state': .5,
                'reference': .25, 'permission': .25}


class EncoderCapture:
    def __init__(self, encoder):
        self.hidden = None
        self.handle = encoder.register_forward_hook(self.store)

    def store(self, module, args, output):
        self.hidden = output.last_hidden_state

    def clear(self):
        self.hidden = None

    def close(self):
        self.handle.remove()
        self.clear()


def masked_kl(student, teacher, mask, *, temperature=2.):
    if student.shape != teacher.shape or student.ndim < 2 or mask.shape != student.shape[:-1]:
        raise ValueError('Teacher vocabulary, token mask or shape changed')
    if not .25 <= temperature <= 10 or mask.dtype != torch.bool:
        raise ValueError('Invalid teacher loss temperature or mask')
    if not bool(torch.isfinite(student).all()) or not bool(torch.isfinite(teacher).all()):
        raise ValueError('Nonfinite teacher loss input')
    if not bool(mask.any()):
        return student.sum() * 0.
    student = student[mask].float() / temperature
    teacher = teacher.detach()[mask].float() / temperature
    return F.kl_div(student.log_softmax(-1), teacher.softmax(-1), reduction='batchmean') * temperature ** 2


def correct_choice_kl(student, teacher, targets):
    """Use training targets to reject a teacher's wrong choice, never relabel it."""
    if student.ndim != 2 or targets.shape != student.shape[:1]:
        raise ValueError('Wrong training target geometry')
    return masked_kl(student, teacher, teacher.detach().argmax(-1) == targets)


def hidden_alignment(student, teacher, mask):
    if student.ndim != 3 or student.shape != teacher.shape or mask.shape != student.shape[:2]:
        raise ValueError('Teacher hidden states are not aligned')
    if mask.dtype != torch.bool or not bool(mask.any()):
        raise ValueError('Hidden alignment needs visible input tokens')
    if not bool(torch.isfinite(student).all()) or not bool(torch.isfinite(teacher).all()):
        raise ValueError('Nonfinite teacher hidden states')
    return (1. - F.cosine_similarity(student[mask].float(), teacher.detach()[mask].float(), dim=-1)).mean()


def extraction_transfer(student, teacher, rows, student_hidden, teacher_hidden):
    """Return separate losses; a caller must record its declared coefficients."""
    if len(rows) != len(student['mappings']) or len(rows) != len(teacher['mappings']):
        raise ValueError('Teacher batch membership differs')
    zero = student_hidden.sum() * 0.
    terms = {name: [] for name in ('choice', 'bio', 'owner', 'state', 'reference', 'permission')}
    width = student_hidden.shape[1]
    lengths = []
    for i, e in enumerate(rows):
        mapping, other = student['mappings'][i], teacher['mappings'][i]
        if (mapping['offsets'] != other['offsets']
                or not torch.equal(mapping['description_mask'], other['description_mask'])
                or student['component_ids'][i] != teacher['component_ids'][i]):
            raise ValueError('Teacher tokenizer or source/candidate mapping differs')
        n = len(mapping['offsets'])
        lengths.append(n)
        if e['row']['question_id'].startswith('attribute:'):
            mask = mapping['description_mask'].to(student_hidden.device)
            gold = (factor_targets(mapping, e['annotation'], e['row']['question_id'].removeprefix('attribute:'))
                    if e['annotation'] is not None else None)
            chosen = teacher['bio'][i, :n].detach().argmax(-1)
            masks = {'bio': mask, 'owner': mask & (chosen != 0),
                     'state': mask & ((chosen == 1) | (chosen == 2))}
            for name in ('bio', 'owner', 'state'):
                valid = masks[name]
                if gold is not None:
                    target = gold[name].to(valid.device)
                    valid = valid & (target >= 0) & (teacher[name][i, :n].detach().argmax(-1) == target)
                terms[name].append(masked_kl(student[name][i, :n], teacher[name][i, :n], valid))
            # Deliberately do not use attribute decision logits in this stage.
        else:
            count = len(e['row']['option_ids'])
            terms['choice'].append(correct_choice_kl(student['decisions'][i:i+1, :count],
                teacher['decisions'][i:i+1, :count], torch.tensor([e['target']], device=student_hidden.device)))
        target = e['reference_target']
        ids = student['component_ids'][i]
        if target is not None and ids:
            index = ids.index(target['component_id']) if target['mode'] == 'selected' else len(ids) + int(target['mode'] == 'unresolved')
            for name, wanted in (('reference', index), ('permission', target['permission'])):
                a, b = student[name][i][None], teacher[name][i][None]
                terms[name].append(correct_choice_kl(a, b, torch.tensor([wanted], device=a.device)))
    mask = torch.arange(width, device=student_hidden.device)[None] < torch.tensor(lengths, device=student_hidden.device)[:, None]
    return {'hidden': hidden_alignment(student_hidden, teacher_hidden, mask),
            **{name: torch.stack(values).mean() if values else zero for name, values in terms.items()}}


def pack_features(features, hidden_size):
    """Store one shared context and each option vector without changing values."""
    if features.dtype != torch.float32:
        raise ValueError('Collect frozen readout features in FP32')
    if features.ndim != 2 or features.shape[1] != 6 * hidden_size + 7 or len(features) < 2:
        raise ValueError('Unexpected frozen decision-head input')
    context = features[:, :-hidden_size]
    if not torch.equal(context, context[:1].expand_as(context)):
        raise ValueError('Option contexts differ; refusing lossy feature packing')
    return context[0].detach().float().cpu().clone(), features[:, -hidden_size:].detach().float().cpu().clone()


def unpack_features(context, options):
    if context.ndim != 2 or options.ndim != 3 or len(context) != len(options):
        raise ValueError('Wrong frozen-feature batch geometry')
    return torch.cat((context[:, None, :].expand(-1, options.shape[1], -1), options), -1)


def collect_readout(model, teacher, rows, batches, *, progress=None):
    """Cache only designated training rows under frozen, FP32 inference."""
    if any(p.requires_grad for p in model.encoder.parameters()):
        raise ValueError('Freeze extraction before collecting readout features')
    if any(p.requires_grad for p in teacher.parameters()) or teacher.training:
        raise ValueError('Teacher must be frozen and in evaluation mode')
    if any(not e['row']['question_id'].startswith('attribute:') for e in rows):
        raise ValueError('Readout cache accepts only attribute questions')
    if not rows or not batches or any(not batch for batch in batches):
        raise ValueError('Nonempty readout collection plan required')
    records, verified, maximum_error, seen = [None] * len(rows), 0, 0., 0
    device = next(model.head.parameters()).device
    model.eval()
    with torch.inference_mode():
        for indices in batches:
            chosen = [rows[i] for i in indices]
            captured = []
            def capture(module, args):
                captured.append(pack_features(args[0], model.head.hidden_size))
            hook = model.head.decision.register_forward_pre_hook(capture)
            try:
                student_output = model.forward_batch([e['row'] for e in chosen])
            finally:
                hook.remove()
            teacher_output = teacher.forward_batch([e['row'] for e in chosen])
            if len(captured) != len(chosen):
                raise ValueError('Readout feature capture lost a row')
            for local, index in enumerate(indices):
                if records[index] is not None:
                    raise ValueError('Repeated training row during feature collection')
                context, options = captured[local]
                count = len(chosen[local]['row']['option_ids'])
                if len(options) != count:
                    raise ValueError('Feature option vocabulary changed')
                if verified < 16:
                    features = unpack_features(context[None].to(device), options[None].to(device))[0]
                    actual = model.head.decision(features).squeeze(-1)
                    expected = student_output['decisions'][local, :count]
                    maximum_error = max(maximum_error, float((actual-expected).abs().max()))
                    if not torch.equal(actual, expected):
                        raise ValueError('FP32 cached head roundtrip changed logits')
                    verified += 1
                records[index] = (context, options, teacher_output['decisions'][local, :count].detach().float().cpu().clone())
            seen += len(indices)
            if progress is not None and seen % 1024 == 0:
                progress('teacher_readout_cache', rows=seen, total=len(rows))
    if any(x is None for x in records):
        raise ValueError('Incomplete training feature cache')
    digest = hashlib.sha256()
    for e, tensors in zip(rows, records):
        digest.update(e['row']['text_sha256'].encode())
        digest.update(str(e['target']).encode())
        for tensor in tensors:
            digest.update(tensor.contiguous().numpy().tobytes())
    audit = {'version': 'frozen-readout-feature-cache-1', 'rows': len(rows), 'precision': 'fp32',
             'content_sha256': digest.hexdigest(), 'real_head_roundtrips': verified,
             'maximum_logit_error': maximum_error, 'teacher_detached': True,
             'all_collection_rows_once': True}
    return records, audit


def cached_readout_loss(model, cache, indices, rows, choice_weights):
    from .curriculum_training import choice_losses
    device = next(model.head.parameters()).device
    chosen = [rows[i] for i in indices]
    contexts = torch.stack([cache[i][0] for i in indices]).to(device)
    options = torch.stack([cache[i][1] for i in indices]).to(device)
    teacher = torch.stack([cache[i][2] for i in indices]).to(device)
    logits = model.head.decision(unpack_features(contexts, options)).squeeze(-1)
    targets = torch.tensor([e['target'] for e in chosen], device=device)
    weights = torch.tensor([choice_weights[(e['row']['question_id'], e['label'])] for e in chosen], device=device)
    gold = (choice_losses(logits, targets, [len(e['row']['option_ids']) for e in chosen], smoothing=.05) * weights).mean()
    return gold + .5 * correct_choice_kl(logits, teacher, targets)
