"""Our auxiliary evidence-attribution head and learned typed decision scorer."""
import torch
from torch import nn

from .evidence_span_data import ROLES
from .span_format import BIO_LABELS, layout


class EvidenceHead(nn.Module):
    def __init__(self, hidden_size, original_scorer, head_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.original_scorer = original_scorer
        self.attribution = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, head_size),
                                         nn.GELU(), nn.Linear(head_size, len(BIO_LABELS)))
        self.decision = nn.Sequential(nn.LayerNorm(6 * hidden_size + len(ROLES) + 1),
                                      nn.Linear(6 * hidden_size + len(ROLES) + 1, head_size),
                                      nn.GELU(), nn.Linear(head_size, 1))

    def forward(self, hidden, option_mask, description_mask, reference_mask, attribute_question):
        token_logits = self.attribution(hidden)
        outputs = []
        for i in range(len(hidden)):
            options = option_mask[i].to(hidden.dtype)
            options = options @ hidden[i] / options.sum(-1, keepdim=True)
            if not attribute_question[i]:
                outputs.append(self.original_scorer(hidden[i, 0], options))
                continue
            probabilities = token_logits[i].softmax(-1)
            role_mass = torch.stack([probabilities[:, BIO_LABELS.index('B-' + role)]
                                     + probabilities[:, BIO_LABELS.index('I-' + role)] for role in ROLES], -1)
            visible = description_mask[i].to(hidden.dtype)
            weights = role_mass[:, ROLES.index('current')] * visible
            current = weights @ hidden[i] / weights.sum().clamp_min(1e-8)
            ref_weights = reference_mask[i].to(hidden.dtype)
            reference = ref_weights @ hidden[i] / ref_weights.sum().clamp_min(1)
            presence = (role_mass * visible[:, None]).amax(0)
            context = torch.cat((hidden[i, 0], current, reference, (current - reference).abs(),
                                 current * reference, presence, ref_weights.any().to(hidden.dtype).reshape(1)))
            features = torch.cat((context.expand(len(options), -1), options), -1)
            outputs.append(self.decision(features).squeeze(-1))
        width = max(len(x) for x in outputs)
        logits = torch.stack([nn.functional.pad(x, (0, width - len(x)), value=torch.finfo(x.dtype).min) for x in outputs])
        return logits, token_logits


def batch_layout(model, rows):
    encoded = [layout(model.tokenizer, row, row['question_id'].removeprefix('attribute:')
                      if row['question_id'].startswith('attribute:') else None, model.max_tokens) for row in rows]
    device = next(model.encoder.parameters()).device
    width = max(x['inputs']['input_ids'].shape[1] for x in encoded)
    inputs = {}
    for key in encoded[0]['inputs']:
        pad = model.tokenizer.pad_token_id if key == 'input_ids' else 0
        tensor = torch.full((len(rows), width), pad, dtype=encoded[0]['inputs'][key].dtype, device=device)
        for i, item in enumerate(encoded):
            tensor[i, :item['inputs'][key].shape[1]] = item['inputs'][key][0].to(device)
        inputs[key] = tensor
    options, descriptions, references = [], [], []
    for item in encoded:
        missing = width - len(item['description_mask'])
        options.append(nn.functional.pad(item['option_mask'], (0, missing)).to(device))
        descriptions.append(nn.functional.pad(item['description_mask'], (0, missing)).to(device))
        references.append(nn.functional.pad(item['reference_mask'], (0, missing)).to(device))
    return inputs, options, descriptions, references, encoded


class AttributedDecisionModel(nn.Module):
    def __init__(self, encoder, tokenizer, head, *, max_tokens=512):
        super().__init__()
        if encoder.config.hidden_size != head.hidden_size or not 1 <= max_tokens <= encoder.config.max_position_embeddings:
            raise ValueError('Invalid encoder/head dimensions or token budget')
        self.encoder, self.tokenizer, self.head, self.max_tokens = encoder, tokenizer, head, max_tokens

    def forward_batch(self, rows):
        inputs, options, descriptions, references, mappings = batch_layout(self, rows)
        hidden = self.encoder(**inputs).last_hidden_state
        logits, tokens = self.head(hidden, options, descriptions, references,
                                   [row['question_id'].startswith('attribute:') for row in rows])
        return logits, tokens, mappings

    def forward(self, row):
        return self.forward_batch([row])[0][0, :len(row['option_ids'])]
