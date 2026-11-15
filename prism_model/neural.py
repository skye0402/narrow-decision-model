"""Frozen text encoder plus our variable-option scorer (prototype, uncalibrated).

Optional dependencies come from the pinned uv benchmark group. Importing the
rules application does not import this module or require PyTorch.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn

from .orchestrator import Judgment
from .provenance import file_sha256
from .renderer import RENDERER_VERSION

HEAD_VERSION = "option-scorer-1"
ARTIFACT_VERSION = "decision-head-prototype-1"


def tokenize_row(tokenizer, row, max_tokens):
    """Map JSON-quoted answer spans to fast-tokenizer offsets; never truncate."""
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer with character offsets is required")
    text, ids, spans = row["text"], row["option_ids"], row["option_spans"]
    if row["renderer_version"] != RENDERER_VERSION or not text.startswith(f"FORMAT {RENDERER_VERSION}\n"):
        raise ValueError("Unsupported renderer version")
    if hashlib.sha256(text.encode()).hexdigest() != row["text_sha256"]:
        raise ValueError("Rendered text hash mismatch")
    if len(ids) < 2 or any(type(key) is not str or not key for key in ids) or len(set(ids)) != len(ids):
        raise ValueError("Offer at least two unique nonempty option IDs")
    if len(spans) != len(ids) or "\nOPTIONS\n" not in text:
        raise ValueError("Invalid option spans")
    cursor = text.rindex("\nOPTIONS\n") + len("\nOPTIONS\n")
    for index, (key, span) in enumerate(zip(ids, spans)):
        start, end = span["start"], span["end"]
        prefix = f"{index} "
        if (span["option_id"] != key or type(start) is not int or type(end) is not int
                or start != cursor + len(prefix) or not start < end < len(text)
                or text[cursor:start] != prefix or text[end:end+1] != "\n"):
            raise ValueError("Option metadata does not match rendered positions")
        if not isinstance(json.loads(text[start:end]), str):
            raise ValueError("Option description must be a JSON string")
        cursor = end + 1
    if cursor != len(text):
        raise ValueError("Unexpected text after offered options")

    encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True,
                        return_special_tokens_mask=True, truncation=False, padding=False)
    if encoded["input_ids"].shape[1] > max_tokens:
        raise ValueError(f"Input has {encoded['input_ids'].shape[1]} tokens; limit is {max_tokens}; no truncation")
    if tokenizer.cls_token_id is None or int(encoded["input_ids"][0, 0]) != tokenizer.cls_token_id:
        raise ValueError("The scorer requires a leading CLS token")
    offsets = encoded.pop("offset_mapping")[0].tolist()
    special = encoded.pop("special_tokens_mask")[0].tolist()
    active = encoded["attention_mask"][0].tolist()
    masks = []
    for span in spans:
        start, end = span["start"], span["end"]
        mask = [bool(on and not sp and b > a and a < end and b > start)
                for (a, b), sp, on in zip(offsets, special, active)]
        covered = set()
        for selected, (a, b) in zip(mask, offsets):
            if selected:
                covered.update(range(max(start, a), min(end, b)))
        if not any(mask) or any(i not in covered and not text[i].isspace() for i in range(start, end)):
            raise ValueError("Tokenizer offsets do not cover the complete answer description")
        masks.append(mask)
    option_mask = torch.tensor(masks, dtype=torch.bool)
    if bool((option_mask.sum(dim=0) > 1).any()):
        raise ValueError("A token overlaps multiple answer descriptions")
    return dict(encoded), option_mask


@dataclass(frozen=True)
class DecisionFeatures:
    context: torch.Tensor
    options: torch.Tensor
    option_ids: tuple[str, ...]
    token_count: int
    text_sha256: str


class OptionScorer(nn.Module):
    """Shared MLP: [CLS representation; mean answer-span representation] -> score."""

    def __init__(self, hidden_size, head_hidden_size=128):
        super().__init__()
        if any(type(size) is not int or size < 1 for size in (hidden_size, head_hidden_size)):
            raise ValueError("Head dimensions must be positive integers")
        self.hidden_size, self.head_hidden_size = hidden_size, head_hidden_size
        self.network = nn.Sequential(nn.LayerNorm(2 * hidden_size),
                                     nn.Linear(2 * hidden_size, head_hidden_size),
                                     nn.GELU(), nn.Linear(head_hidden_size, 1))

    def forward(self, context, options):
        if context.shape != (self.hidden_size,) or options.ndim != 2 or options.shape[1] != self.hidden_size or options.shape[0] < 2:
            raise ValueError("Expected one context vector and at least two answer vectors")
        features = torch.cat((context.expand(options.shape[0], -1), options), dim=-1)
        return self.network(features).squeeze(-1)


class FrozenDecisionModel(nn.Module):
    """One row per forward; train only our head, keeping the encoder in eval mode."""

    def __init__(self, encoder, tokenizer, head, *, max_tokens=512):
        super().__init__()
        if type(max_tokens) is not int or not 1 <= max_tokens <= encoder.config.max_position_embeddings:
            raise ValueError("Invalid token budget for this encoder")
        if encoder.config.hidden_size != head.hidden_size:
            raise ValueError("Encoder and head dimensions differ")
        self.encoder, self.tokenizer, self.head, self.max_tokens = encoder, tokenizer, head, max_tokens
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def features(self, row):
        inputs, mask = tokenize_row(self.tokenizer, row, self.max_tokens)
        device = next(self.encoder.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        mask = mask.to(device)
        # no_grad keeps cached features usable by autograd when training the head.
        with torch.no_grad():
            hidden = self.encoder(**inputs).last_hidden_state[0]
            if not bool(torch.isfinite(hidden).all()):
                raise ValueError("Non-finite encoder representations")
            weights = mask.to(hidden.dtype)
            options = weights @ hidden / weights.sum(dim=1, keepdim=True)
        return DecisionFeatures(hidden[0].detach(), options.detach(), tuple(row["option_ids"]),
                                inputs["input_ids"].shape[1], row["text_sha256"])

    def forward(self, row):
        features = self.features(row)
        return self.head(features.context, features.options)


class NeuralJudge:
    """Probabilistic adapter for the existing orchestrator; no thresholds supplied."""
    probabilistic = True

    def __init__(self, model):
        self.model = model

    def judge(self, row):
        self.model.eval()
        with torch.inference_mode():
            logits = self.model(row)
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("Non-finite answer scores")
            probabilities = torch.softmax(logits, dim=0).tolist()
        if not all(math.isfinite(value) for value in probabilities):
            raise ValueError("Non-finite probabilities")
        distribution = dict(zip(row["option_ids"], probabilities))
        return Judgment(row["option_ids"][int(logits.argmax())], distribution)

    def judge_group(self, row, context):
        return self.judge(row)

    def judge_requirement(self, row, context, policy):
        return self.judge(row)


def load_local_encoder(directory):
    """Restore the verified local fp32/CPU backbone and bind its exact files."""
    from transformers import AutoModel, AutoTokenizer

    directory = Path(directory)
    provenance = json.loads((directory / "provenance.json").read_text())
    names = ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "pytorch_model.bin")
    hashes = {name: file_sha256(directory / name) for name in names}
    expected = provenance["upstream_files"]["pytorch_model.bin"]["lfs"]
    if hashes["pytorch_model.bin"] != expected["sha256"] or (directory / "pytorch_model.bin").stat().st_size != expected["size"]:
        raise ValueError("Local encoder weights do not match downloaded provenance")
    identity = {"repository": provenance["repository"], "revision": provenance["revision"],
                "declared_license": provenance["declared_license"], "files_sha256": hashes}
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    encoder, loading = AutoModel.from_pretrained(directory, local_files_only=True, trust_remote_code=False,
        weights_only=True, output_loading_info=True, torch_dtype=torch.float32, attn_implementation="sdpa")
    if any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("Encoder weights were not fully restored")
    return encoder.to("cpu").eval(), tokenizer, identity, loading


def save_head(directory, head, encoder_identity, *, max_tokens, training_info):
    """Save only our head; the encoder stays separately pinned. Never overwrite."""
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("Invalid token limit")
    if not all(bool(torch.isfinite(value).all()) for value in head.state_dict().values()):
        raise ValueError("Cannot save non-finite head weights")
    metadata = {"artifact_version": ARTIFACT_VERSION, "head_version": HEAD_VERSION,
                "renderer_version": RENDERER_VERSION, "hidden_size": head.hidden_size,
                "head_hidden_size": head.head_hidden_size, "max_tokens": max_tokens,
                "temperature": 1.0, "calibrated": False, "deployment_qualified": False,
                "encoder_identity": encoder_identity, "training_info": training_info}
    # Validate metadata before creating the artifact directory.
    json.dumps(metadata, allow_nan=False)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    weights = directory / "head.pt"
    torch.save({key: value.detach().cpu() for key, value in head.state_dict().items()}, weights)
    metadata["weights_sha256"] = file_sha256(weights)
    (directory / "head.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return metadata


def load_head(directory, encoder_identity):
    """Load tensor weights only; refuse a different encoder/tokenizer or format."""
    directory = Path(directory)
    metadata = json.loads((directory / "head.json").read_text())
    if (metadata["artifact_version"] != ARTIFACT_VERSION or metadata["head_version"] != HEAD_VERSION
            or metadata["renderer_version"] != RENDERER_VERSION):
        raise ValueError("Unsupported head artifact version")
    if metadata["encoder_identity"] != encoder_identity:
        raise ValueError("Head belongs to a different encoder/tokenizer artifact")
    if metadata["temperature"] != 1.0 or metadata["calibrated"] is not False or metadata["deployment_qualified"] is not False:
        raise ValueError("This prototype loader accepts only uncalibrated artifacts")
    if type(metadata["max_tokens"]) is not int or metadata["max_tokens"] < 1:
        raise ValueError("Invalid artifact token limit")
    if file_sha256(directory / "head.pt") != metadata["weights_sha256"]:
        raise ValueError("Head weight digest mismatch")
    head = OptionScorer(metadata["hidden_size"], metadata["head_hidden_size"])
    head.load_state_dict(torch.load(directory / "head.pt", map_location="cpu", weights_only=True), strict=True)
    if not all(bool(torch.isfinite(value).all()) for value in head.state_dict().values()):
        raise ValueError("Non-finite saved head weights")
    return head.eval(), metadata
