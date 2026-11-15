"""Download (or use a local copy of) a pinned public-release artifact."""
import argparse, hashlib, json, tarfile, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--archive',type=Path)
p.add_argument('--training',action='store_true')
a=p.parse_args()
m=json.loads((ROOT/'artifacts.json').read_text())
name='training-inputs.tar.gz' if a.training else 'prism-cpu-demo.tar.gz'
archive=a.archive or ROOT/'.cache/downloads'/name
if not archive.exists():
 archive.parent.mkdir(parents=True,exist_ok=True)
 url=f"https://github.com/{m['repository']}/releases/download/{m['release_tag']}/{name}"
 print('Downloading',url,flush=True)
 urllib.request.urlretrieve(url,archive)
if hashlib.sha256(archive.read_bytes()).hexdigest()!=m['artifacts'][name]['sha256']:
 raise SystemExit('Archive checksum mismatch; nothing extracted.')
dest=ROOT/'.cache' if a.training else ROOT/'.cache/category-gate-20261115/demo-update'
target=dest/('training-data' if a.training else 'model')
if target.exists(): raise SystemExit('Target exists; preserve the installed artifact.')
dest.mkdir(parents=True,exist_ok=True)
with tarfile.open(archive) as t: t.extractall(dest,filter='data')
print('Installed',target)
