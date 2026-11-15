"""AI Core entrypoint for the full learned-versus-hybrid training candidate."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prism_model.comparison_training import train

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--warm-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--max-batches', type=int)
    a = p.parse_args()
    train(data_dir=a.data_dir, warm_dir=a.warm_dir, output_dir=a.output_dir,
          source_root=a.source_manifest.parent, source_manifest=json.loads(a.source_manifest.read_text()),
          device=a.device, epochs=a.epochs, max_batches=a.max_batches)
