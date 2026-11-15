"""AI Core container entrypoint. CUDA is mandatory unless --device cpu is explicit."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prism_model.training import emit, run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/input/data"))
    parser.add_argument("--encoder-dir", type=Path, default=Path("/input/encoder"))
    parser.add_argument("--output-dir", type=Path, default=Path("/output/model"))
    parser.add_argument("--source-manifest", type=Path, default=ROOT/"source-manifest.json")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--expected-gpu", default="NVIDIA L4")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()
    import torch
    torch.set_num_interop_threads(1)
    manifest = json.loads(args.source_manifest.read_text())
    run_training(data_dir=args.data_dir, encoder_dir=args.encoder_dir, output_dir=args.output_dir,
        source_root=ROOT, source_manifest=manifest, device=args.device, expected_gpu=args.expected_gpu,
        steps=args.steps, seed=args.seed, threads=args.threads, max_tokens=args.max_tokens)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        emit("training_failed", error_type=type(error).__name__)
        sys.exit(1)
