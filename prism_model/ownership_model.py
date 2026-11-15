"""Shared encoder for learned decisions and the extraction-plus-code path."""
import torch
import re
from torch import nn

from .evidence_span_data import ROLES
from .ownership_format import BIO_LABELS, OWNERS, STATES, component_layout
from .span_model import batch_layout
from .evidence_orchestrator import InputLimitError


def classifier(hidden, output, width):
    return nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, width), nn.GELU(), nn.Linear(width, output))


class OwnershipHead(nn.Module):
    def __init__(self, hidden_size, original_scorer, decision, head_size=128):
        super().__init__()
        self.hidden_size, self.head_size = hidden_size, head_size
        self.original_scorer, self.decision = original_scorer, decision
        self.bio = classifier(hidden_size, len(BIO_LABELS), head_size)
        self.owner = classifier(hidden_size, len(OWNERS), head_size)
        self.state = classifier(hidden_size, len(STATES), head_size)
        self.permission = classifier(hidden_size, 2, head_size)
        self.reference = classifier(4 * hidden_size, 1, head_size)
        self.reference_specials = nn.Parameter(torch.randn(2, hidden_size) * .02)

    def forward(self, hidden, options, descriptions, references, attribute_questions, components):
        bio, owner, state = self.bio(hidden), self.owner(hidden), self.state(hidden)
        span_probability = bio.softmax(-1)
        value = span_probability[:, :, 1] + span_probability[:, :, 2]
        type_probability = span_probability[:, :, 3] + span_probability[:, :, 4]
        owner_probability = owner.softmax(-1)
        state_probability = state.softmax(-1)
        outputs, pointer = [], []
        for i, is_attribute in enumerate(attribute_questions):
            masks = options[i].to(hidden.dtype)
            answer_vectors = masks @ hidden[i] / masks.sum(-1, keepdim=True)
            if is_attribute:
                requested = value[i] * owner_probability[i, :, 0]
                role_mass = torch.stack((requested * state_probability[i, :, 0],
                                         value[i] * owner_probability[i, :, 1],
                                         requested * state_probability[i, :, 1],
                                         requested * state_probability[i, :, 2],
                                         requested * state_probability[i, :, 3], type_probability[i]), -1)
                visible = descriptions[i].to(hidden.dtype)
                weights = role_mass[:, ROLES.index('current')] * visible
                current = weights @ hidden[i] / weights.sum().clamp_min(1e-8)
                ref = references[i].to(hidden.dtype)
                reference = ref @ hidden[i] / ref.sum().clamp_min(1)
                presence = (role_mass * visible[:, None]).amax(0)
                context = torch.cat((hidden[i, 0], current, reference, (current - reference).abs(),
                                     current * reference, presence, ref.any().to(hidden.dtype).reshape(1)))
                features = torch.cat((context.expand(len(answer_vectors), -1), answer_vectors), -1)
                outputs.append(self.decision(features).squeeze(-1))
            else:
                outputs.append(self.original_scorer(hidden[i, 0], answer_vectors))
            vectors = []
            for mask in components[i]:
                mask = mask.to(device=hidden.device, dtype=hidden.dtype)
                vectors.append(mask @ hidden[i] / mask.sum())
            candidates = torch.stack([*vectors, *self.reference_specials.to(hidden.dtype)])
            context = hidden[i, 0].expand(len(candidates), -1)
            features = torch.cat((context, candidates, (context - candidates).abs(), context * candidates), -1)
            pointer.append(self.reference(features).squeeze(-1))
        width = max(len(scores) for scores in outputs)
        decisions = torch.stack([nn.functional.pad(scores, (0, width - len(scores)), value=torch.finfo(scores.dtype).min)
                                 for scores in outputs])
        return {'decisions': decisions, 'bio': bio, 'owner': owner, 'state': state,
                'permission': self.permission(hidden[:, 0]), 'reference': pointer}


class OwnershipModel(nn.Module):
    def __init__(self, encoder, tokenizer, head, *, max_tokens=512):
        super().__init__()
        if encoder.config.hidden_size != head.hidden_size or not 1 <= max_tokens <= encoder.config.max_position_embeddings:
            raise ValueError('Invalid ownership encoder/head dimensions')
        self.encoder, self.tokenizer, self.head, self.max_tokens = encoder, tokenizer, head, max_tokens

    def forward_batch(self, rows):
        try:
            inputs, options, descriptions, references, mappings = batch_layout(self, rows)
        except ValueError as error:
            found = re.fullmatch(r'Input has (\d+) tokens; limit is (\d+); no truncation', str(error))
            if found:
                for row in rows:
                    count = len(self.tokenizer(row['text'], truncation=False)['input_ids'])
                    if count > self.max_tokens:
                        raise InputLimitError(row['question_id'], count, self.max_tokens) from error
            raise
        components = [component_layout(row, mapping, inputs['input_ids'].shape[1]) for row, mapping in zip(rows, mappings)]
        hidden = self.encoder(**inputs).last_hidden_state
        result = self.head(hidden, options, descriptions, references,
                           [r['question_id'].startswith('attribute:') for r in rows], [x[1] for x in components])
        result['mappings'] = mappings
        result['component_ids'] = [x[0] for x in components]
        return result

    def forward(self, row):
        return self.forward_batch([row])['decisions'][0, :len(row['option_ids'])]
