"""Export exact committed source for the complete ownership comparison experiment."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prism_model.provenance import SOURCE_VERSION, file_sha256, verify_source


def prepare(root, entrypoint='train_comparison', view_version=1):
    root = Path(root)
    destination = root / 'input/source'
    if destination.exists():
        raise ValueError('Preserve the previous source export')
    audit = json.loads((root / 'data-audit.json').read_text())
    if (audit['dataset_manifest_sha256'] != file_sha256(root / 'data/dataset-manifest.json')
            or audit['optimizer_manifest_sha256'] != file_sha256(root / 'input/data/dataset-manifest.json')
            or not audit['original_400_identical'] or not audit['all_family_primaries_unique']
            or audit['maximum_tokens'] > 512 or audit['evaluation_fraction'] != .2):
        raise ValueError('Complete data audit changed or failed')
    if entrypoint not in ('train_comparison', 'train_attribution', 'train_teacher_transfer'):
        raise ValueError('Undeclared training entrypoint')
    if entrypoint in ('train_attribution', 'train_teacher_transfer'):
        extra = json.loads((root / 'isolated-training-data-audit.json').read_text())
        if (extra['source_data_manifest_sha256'] != file_sha256(root / 'data/dataset-manifest.json')
                or extra.get('attribution_view_version', 1) != view_version
                or extra['source_training_sha256'] != file_sha256(ROOT / 'prism_model/isolated_attribution.py')
                or extra['source_views_sha256'] != file_sha256(ROOT / 'prism_model/attribution_views.py')
                or any(e['maximum_tokens'] > 512 or e['view_audit']['supplied_source_cases'] != 400 for e in extra['epochs'])):
            raise ValueError('Isolated attribution input audit changed or failed')
        if entrypoint == 'train_teacher_transfer' and (extra.get('teacher_transfer') is not True
                or extra['source_transfer_training_sha256'] != file_sha256(ROOT / 'prism_model/teacher_transfer_training.py')
                or extra['source_teacher_loss_sha256'] != file_sha256(ROOT / 'prism_model/teacher_transfer.py')):
            raise ValueError('Teacher transfer source audit changed or failed')
        if view_version == 3 and (extra.get('source_unmarked_views_sha256')
                != file_sha256(ROOT / 'prism_model/unmarked_views.py')):
            raise ValueError('Plain-fact writer source audit changed or failed')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    modules = subprocess.check_output(['git', 'ls-files', 'prism_model'], cwd=ROOT, text=True).splitlines()
    names = [p for p in modules if p.endswith('.py')] + ['scripts/train_model.py', 'scripts/' + entrypoint + '.py', 'pyproject.toml', 'uv.lock', 'requirements/training.txt']
    contents = {name: subprocess.check_output(['git', 'show', revision + ':' + name], cwd=ROOT) for name in names}
    if any((ROOT / name).read_bytes() != value for name, value in contents.items()):
        raise ValueError('Commit source before export')
    if 'prism_model/comparison_training.py' not in contents:
        raise ValueError('Commit the new trainer first')
    for name, value in contents.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    manifest = {'version': SOURCE_VERSION, 'source_commit': revision,
                'files_sha256': {name: file_sha256(destination / name) for name in names}}
    (destination / 'source-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    verify_source(destination, manifest)
    return {'source_commit': revision, 'source_manifest_sha256': file_sha256(destination / 'source-manifest.json'),
            'files': len(names)}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT / '.cache/ownership-comparison-20261106')
    p.add_argument('--entrypoint', choices=('train_comparison', 'train_attribution', 'train_teacher_transfer'), default='train_comparison')
    p.add_argument('--view-version', type=int, choices=(1, 2, 3), default=1)
    args = p.parse_args()
    print(json.dumps(prepare(args.root, args.entrypoint, args.view_version)))
