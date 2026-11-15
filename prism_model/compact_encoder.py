"""One declared depth-reduction experiment; no policy or evidence shortcuts."""
import torch

from .training import state_sha256

LAYERS = (0, 1, 2, 6, 7, 8, 12, 13, 14, 21)
WARM_IDENTITY = '28b45104f0888b61382c72b05436dd1646b5e54e8382cdf4107d7e5c59ba7682'


def reduce_depth(model):
    """Keep fixed blocks with the same global/local attention pattern on reload."""
    encoder, config = model.encoder, model.encoder.config
    if (getattr(config, 'model_type', None) != 'modernbert'
            or config.num_hidden_layers != 22 or len(encoder.layers) != 22
            or config.global_attn_every_n_layers != 3 or config.hidden_size != 768):
        raise ValueError('The compact experiment requires the declared 22-layer encoder')
    if any(old % 3 != new % 3 for new, old in enumerate(LAYERS)):
        raise ValueError('Depth reduction changed the attention pattern')
    for old in LAYERS:
        expected = (-1, -1) if old % 3 == 0 else (config.local_attention // 2,) * 2
        if encoder.layers[old].attn.local_attention != expected:
            raise ValueError('Unexpected warm attention layout')
    before = sum(p.numel() for p in model.parameters())
    head_sha = state_sha256(model.head)
    selected = [encoder.layers[i] for i in LAYERS]
    encoder.layers = torch.nn.ModuleList(selected)
    config.num_hidden_layers = len(LAYERS)
    for index, layer in enumerate(encoder.layers):
        layer.attn.layer_id = index
    after = sum(p.numel() for p in model.parameters())
    if after >= before or state_sha256(model.head) != head_sha:
        raise ValueError('Depth reduction must remove blocks and preserve all heads')
    return {'version': 'fixed-depth-reduction-1', 'original_layers': 22,
            'retained_original_indices': list(LAYERS), 'layers': len(LAYERS),
            'parameters_before': before, 'parameters_after': after,
            'heads_sha256_before_and_after': head_sha,
            'global_local_attention_pattern_preserved': True,
            'no_training_or_quality_claim': True}
