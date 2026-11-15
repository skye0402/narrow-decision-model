"""Stable option ordering for the versioned evidence-model interface.

This normalizes equivalent permutations before ordinary joint encoding. It is
not NDM block attention and does not imply invariance to changing answer text.
"""
import json

from .task_model import permute_options


def canonical_row(row):
    descriptions = [json.loads(row['text'][s['start']:s['end']]) for s in row['option_spans']]
    if len(set(descriptions)) != len(descriptions):
        raise ValueError('Duplicate answer descriptions are ambiguous')
    order = sorted(range(len(descriptions)), key=lambda i: descriptions[i])
    return permute_options(row, order)
