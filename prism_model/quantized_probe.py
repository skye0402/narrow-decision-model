"""Bounded CPU quantization probe; never reuse FP32 decision calibration."""
import hashlib
import json

import torch

from .provenance import file_sha256

VERSION = 'encoder-dynamic-int8-probe-1'


def float_state_sha256(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def prepare(model, raw_model_identity):
    if torch.__version__.split('+')[0] != '2.9.1' or 'qnnpack' not in torch.backends.quantized.supported_engines:
        raise ValueError('This probe is pinned to PyTorch 2.9.1 with QNNPACK')
    if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('Start from the verified CPU FP32 model')
    model.eval()
    model.requires_grad_(False)
    heads_before = float_state_sha256(model.head)
    original = sum(isinstance(m, torch.nn.Linear) for m in model.encoder.modules())
    if not original:
        raise ValueError('No encoder linear layers to quantize')
    torch.backends.quantized.engine = 'qnnpack'
    model.encoder = torch.ao.quantization.quantize_dynamic(model.encoder, {torch.nn.Linear},
                                                           dtype=torch.qint8, inplace=True)
    digest = hashlib.sha256()
    converted = 0
    for name, layer in sorted(model.encoder.named_modules()):
        if not isinstance(layer, torch.ao.nn.quantized.dynamic.Linear):
            continue
        converted += 1
        weight, bias = layer.weight(), layer.bias()
        metadata = {'name': name, 'shape': list(weight.shape), 'dtype': str(weight.dtype), 'qscheme': str(weight.qscheme())}
        if weight.qscheme() in (torch.per_tensor_affine, torch.per_tensor_symmetric):
            metadata.update(scale=weight.q_scale(), zero_point=weight.q_zero_point())
        else:
            metadata.update(axis=weight.q_per_channel_axis(), scales=weight.q_per_channel_scales().tolist(),
                            zero_points=weight.q_per_channel_zero_points().tolist())
        digest.update(json.dumps(metadata, sort_keys=True).encode())
        digest.update(weight.int_repr().contiguous().numpy().tobytes())
        if bias is not None:
            digest.update(bias.detach().cpu().contiguous().numpy().tobytes())
    if converted != original or any(isinstance(m, torch.nn.Linear) for m in model.encoder.modules()):
        raise ValueError('Encoder linear conversion was incomplete')
    if float_state_sha256(model.head) != heads_before:
        raise ValueError('The probe changed a decision or extraction head')
    recipe = {'version': VERSION, 'raw_model_identity': raw_model_identity,
              'recipe_source_sha256': file_sha256(__file__), 'backend': 'qnnpack',
              'torch_base_version': torch.__version__.split('+')[0], 'scope': 'encoder_linear_layers_only',
              'quantized_encoder_linears': converted, 'heads_precision': 'fp32',
              'heads_sha256': heads_before, 'quantized_linear_state_sha256': digest.hexdigest()}
    identity = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return model, {**recipe, 'variant_model_identity': identity, 'torch_build': torch.__version__,
                    'calibrated': False, 'deployment_qualified': False}
