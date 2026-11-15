"""Development-only learned readout; no numeric extraction or equality rules.

The only located value is a known field in trusted structured input. The entire
request description is presented to learned attention, without gold spans.
This module is not used by saved decision models or the application runtime.
"""
import copy
import json
import math

import torch
from torch import nn

from .neural import tokenize_row
from .renderer import compact


def json_field_span(value, path, offset=0):
    """Locate a field by structure in our canonical JSON, never by value search."""
    if not path:
        return offset, offset + len(compact(value))
    first, *rest = path
    cursor = offset + 1
    if isinstance(value, dict):
        for index, key in enumerate(sorted(value)):
            if index:
                cursor += 1
            prefix = compact(key) + ':'
            if key == first:
                start, end = json_field_span(value[key], rest, cursor + len(prefix))
                # Include the key only for the leaf reference field.
                return (cursor, end) if not rest else (start, end)
            cursor += len(prefix) + len(compact(value[key]))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            if index:
                cursor += 1
            if index == first:
                return json_field_span(item, rest, cursor)
            cursor += len(compact(item))
    raise ValueError('Requested structural path is absent')


def structural_masks(tokenizer, row, attribute, max_tokens=512):
    inputs, option_mask = tokenize_row(tokenizer, row, max_tokens)
    text = row['text']
    prefix = '\nEVIDENCE '
    start = text.index(prefix) + len(prefix)
    end = text.index('\nPOLICY ', start)
    source = text[start:end]
    evidence = json.loads(source)
    if compact(evidence) != source:
        raise ValueError('Unexpected noncanonical evidence serialization')
    components = evidence['resolved_reference']['components']
    if len(components) != 1 or attribute not in components[0]:
        raise ValueError('Mechanism requires a sole trusted field')
    # Description includes its key and the complete quoted value. No numeric
    # parser or teacher-supplied request span participates in these masks.
    spans = [json_field_span(evidence, ['description'], start),
             json_field_span(evidence, ['resolved_reference', 'components', 0, attribute], start)]
    encoded = tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True,
                        truncation=False)
    if encoded['input_ids'] != inputs['input_ids'][0].tolist():
        raise ValueError('Offset and model tokenization differ')
    masks = []
    for a, b in spans:
        mask = [not special and y > x and x < b and y > a
                for (x, y), special in zip(encoded['offset_mapping'], encoded['special_tokens_mask'])]
        covered = {i for active, (x, y) in zip(mask, encoded['offset_mapping']) if active
                   for i in range(max(a, x), min(b, y))}
        if not any(mask) or any(i not in covered and not text[i].isspace() for i in range(a, b)):
            raise ValueError('Incomplete known-field token coverage')
        masks.append(torch.tensor(mask, dtype=torch.bool))
    if bool((masks[0] & masks[1]).any()):
        raise ValueError('Request and trusted field masks overlap')
    return inputs, option_mask, masks[0], masks[1], spans


def encode(model, examples, batch_size=8, progress=None):
    """Cache frozen CPU contextual tokens, CLS and offered-answer representations."""
    output = []
    model.eval()
    device = next(model.encoder.parameters()).device
    for begin in range(0, len(examples), batch_size):
        chosen = examples[begin:begin + batch_size]
        encoded = [structural_masks(model.tokenizer, e['row'], e['attribute'], model.max_tokens)
                   for e in chosen]
        width = max(x[0]['input_ids'].shape[1] for x in encoded)
        inputs = {}
        for key in encoded[0][0]:
            pad = model.tokenizer.pad_token_id if key == 'input_ids' else 0
            tensor = torch.full((len(chosen), width), pad, dtype=encoded[0][0][key].dtype, device=device)
            for i, item in enumerate(encoded):
                tensor[i, :item[0][key].shape[1]] = item[0][key][0].to(device)
            inputs[key] = tensor
        with torch.inference_mode():
            states = model.encoder(**inputs).last_hidden_state.cpu()
            if not bool(torch.isfinite(states).all()):
                raise ValueError('Nonfinite encoder output')
            for state, (tokens, options, request, reference, spans) in zip(states, encoded):
                hidden = state[:tokens['input_ids'].shape[1]]
                weights = options.float()
                # Float32 copies avoid inference tensors entering training graphs.
                output.append({'context': hidden[0].clone(),
                               'options': (weights @ hidden / weights.sum(1, keepdim=True)).clone(),
                               'request': hidden[request].clone(),
                               'reference': hidden[reference].mean(0).clone()})
        if progress:
            progress(len(output), len(examples))
    # Leave inference_mode before copying tensors used by autograd.
    return [{k: v.clone() for k, v in f.items()} for f in output]


def pack(features):
    size = len(features)
    hidden = features[0]['context'].shape[0]
    width = max(f['request'].shape[0] for f in features)
    requests = torch.zeros(size, width, hidden)
    visible = torch.zeros(size, width, dtype=torch.bool)
    for i, f in enumerate(features):
        count = f['request'].shape[0]
        requests[i, :count] = f['request']
        visible[i, :count] = True
    return {'context': torch.stack([f['context'] for f in features]),
            'options': torch.stack([f['options'] for f in features]),
            'reference': torch.stack([f['reference'] for f in features]),
            'request': requests, 'visible': visible}


class Readout(nn.Module):
    def __init__(self, base_head, relational=False, heads=4, width=32):
        super().__init__()
        self.base = copy.deepcopy(base_head)
        self.relational, self.heads, self.width = relational, heads, width
        if relational:
            hidden = base_head.hidden_size
            self.normalization = nn.LayerNorm(hidden)
            self.projection = nn.Linear(hidden, heads * width)
            self.option_projection = nn.Linear(hidden, 64)
            self.residual = nn.Sequential(nn.LayerNorm(4 * heads * width + 64),
                                          nn.Linear(4 * heads * width + 64, 128),
                                          nn.GELU(), nn.Linear(128, 1))
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

    def forward(self, features):
        context, options = features['context'], features['options']
        batch, choices, hidden = options.shape
        base = self.base.network(torch.cat((context[:, None, :].expand(-1, choices, -1), options), -1)).squeeze(-1)
        if not self.relational:
            return base
        reference = self.projection(self.normalization(features['reference'])).reshape(batch, self.heads, self.width)
        request = self.projection(self.normalization(features['request'])).reshape(batch, -1, self.heads, self.width)
        similarity = torch.einsum('bhd,bthd->bht', reference, request) / math.sqrt(self.width)
        similarity = similarity.masked_fill(~features['visible'][:, None, :], float('-inf'))
        attention = similarity.softmax(-1)
        attended = torch.einsum('bht,bthd->bhd', attention, request)
        pair = torch.cat((reference.flatten(1), attended.flatten(1),
                          (reference - attended).abs().flatten(1), (reference * attended).flatten(1)), -1)
        option = self.option_projection(options)
        delta = self.residual(torch.cat((pair[:, None, :].expand(-1, choices, -1), option), -1)).squeeze(-1)
        return base + delta
