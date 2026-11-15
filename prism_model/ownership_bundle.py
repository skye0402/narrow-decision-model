"""Versioned shared model bundles for the two complete comparison paths."""
import hashlib
import json
from pathlib import Path

import torch

from .neural import OptionScorer
from .provenance import checked_file, file_sha256, verify_files
from .ownership_format import BIO_LABELS, OWNERS, STATES
from .ownership_model import OwnershipModel, OwnershipHead

VERSION = 'ownership-comparison-model-1'
ROOT = Path(__file__).resolve().parents[1]


def inference_hashes():
    from .comparison_runtime import inference_hashes as pipeline_hashes
    return {**pipeline_hashes(), 'prism_model/ownership_bundle.py': file_sha256(__file__)}



def write_manifest(directory):
    directory = Path(directory)
    files = {p.relative_to(directory).as_posix(): file_sha256(p) for p in sorted(directory.rglob('*'))
             if p.is_file() and p.name != 'bundle-manifest.json'}
    weights = {k: v for k, v in files.items() if k.startswith('encoder/') or k in ('heads.pt', 'model.json', 'pilot_config.json')}
    identity = hashlib.sha256(json.dumps(weights, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    result = {'version': VERSION, 'complete': True, 'scope': 'synthetic_full_comparison',
              'files_sha256': files, 'model_identity': identity,
              'calibrated': False, 'full_pipeline_qualified': False, 'deployment_qualified': False}
    (directory / 'bundle-manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def verify_bundle(directory):
    directory = Path(directory)
    manifest = json.loads(checked_file(directory, 'bundle-manifest.json').read_text())
    if manifest['version'] != VERSION or manifest['complete'] is not True:
        raise ValueError('Wrong ownership comparison model')
    verify_files(directory, manifest['files_sha256'], exact=True, exclude=('bundle-manifest.json',),
                 required=('encoder/pytorch_model.bin', 'encoder/config.json', 'heads.pt', 'model.json',
                           'pilot_config.json', 'training-report.json'))
    weights = {k: v for k, v in manifest['files_sha256'].items()
               if k.startswith('encoder/') or k in ('heads.pt', 'model.json', 'pilot_config.json')}
    identity = hashlib.sha256(json.dumps(weights, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if identity != manifest['model_identity']:
        raise ValueError('Prototype model identity changed')
    return manifest


def save_bundle(directory, model, config, report, source_manifest):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    encoder = directory / 'encoder'
    model.encoder.save_pretrained(encoder, safe_serialization=False,
                                 state_dict={k: v.detach().cpu() for k, v in model.encoder.state_dict().items()})
    model.tokenizer.save_pretrained(encoder)
    torch.save({k: v.detach().cpu() for k, v in model.head.state_dict().items()}, directory / 'heads.pt')
    metadata = {'version': VERSION, 'hidden_size': model.head.hidden_size, 'head_size': model.head.head_size,
                'original_scorer_hidden_size': model.head.original_scorer.head_hidden_size,
                'max_tokens': model.max_tokens, 'bio_labels': list(BIO_LABELS), 'owners': list(OWNERS), 'states': list(STATES),
                'inference_source_sha256': inference_hashes(), 'calibrated': False,
                'full_pipeline_qualified': False, 'deployment_qualified': False}
    for name, value in (('model.json', metadata), ('pilot_config.json', config),
                        ('training-report.json', report), ('source-manifest.json', source_manifest)):
        (directory / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    return write_manifest(directory)


def rebind_bundle(source, destination, reason):
    """Bind unchanged trained weights to revised inference code, keeping provenance.

    Encoder, tokenizer, heads and policy bytes are copied and re-verified. Only
    model.json changes: it records the new inference hashes and the parent
    artifact. The result has a new model identity and needs new calibration.
    """
    import shutil
    source, destination = Path(source), Path(destination)
    manifest = verify_bundle(source)
    if destination.exists():
        raise ValueError('Preserve an existing rebound bundle')
    shutil.copytree(source, destination)
    for name in ('calibration-learned.json', 'calibration-hybrid.json'):
        (destination / name).unlink(missing_ok=True)
    metadata = json.loads((destination / 'model.json').read_text())
    metadata['rebinding'] = {'parent_artifact_identity': file_sha256(source / 'bundle-manifest.json'),
                             'parent_model_identity': manifest['model_identity'], 'reason': reason,
                             'parent_inference_source_sha256': metadata['inference_source_sha256'],
                             'weights_changed': False}
    metadata['inference_source_sha256'] = inference_hashes()
    (destination / 'model.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    result = write_manifest(destination)
    for name, digest in manifest['files_sha256'].items():
        if (name.startswith('encoder/') or name in ('heads.pt', 'pilot_config.json')) and result['files_sha256'][name] != digest:
            raise ValueError('Rebinding changed trained weights or policy')
    return result


def load_model(directory, threads=2):
    from transformers import AutoModel, AutoTokenizer
    directory = Path(directory)
    manifest = verify_bundle(directory)
    metadata = json.loads((directory / 'model.json').read_text())
    if metadata['inference_source_sha256'] != inference_hashes() or (metadata['bio_labels'] != list(BIO_LABELS) or metadata['owners'] != list(OWNERS) or metadata['states'] != list(STATES)):
        raise ValueError('Prototype inference source or label schema changed')
    if any(metadata[key] is not False for key in ('calibrated', 'full_pipeline_qualified', 'deployment_qualified')):
        raise ValueError('Unsupported prototype qualification')
    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(directory / 'encoder', local_files_only=True, trust_remote_code=False)
    encoder, loading = AutoModel.from_pretrained(directory / 'encoder', local_files_only=True, trust_remote_code=False,
                                                 weights_only=True, torch_dtype=torch.float32, attn_implementation='sdpa',
                                                 output_loading_info=True)
    if any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('Prototype encoder did not restore completely')
    encoder.config.reference_compile = False
    original = OptionScorer(metadata['hidden_size'], metadata['original_scorer_hidden_size'])
    decision = torch.nn.Sequential(torch.nn.LayerNorm(6 * metadata['hidden_size'] + 7),
        torch.nn.Linear(6 * metadata['hidden_size'] + 7, metadata['head_size']), torch.nn.GELU(),
        torch.nn.Linear(metadata['head_size'], 1))
    head = OwnershipHead(metadata['hidden_size'], original, decision, metadata['head_size'])
    head.load_state_dict(torch.load(directory / 'heads.pt', map_location='cpu', weights_only=True), strict=True)
    if not all(bool(torch.isfinite(p).all()) for p in head.parameters()):
        raise ValueError('Nonfinite prototype head')
    model = OwnershipModel(encoder, tokenizer, head, max_tokens=metadata['max_tokens']).eval()
    return model, json.loads((directory / 'pilot_config.json').read_text()), manifest
