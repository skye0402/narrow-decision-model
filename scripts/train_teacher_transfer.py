"""Run the declared two extraction-transfer and two cached-readout epochs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prism_model.teacher_transfer_training import train

p=argparse.ArgumentParser(description=__doc__)
for name in ('data-dir','warm-dir','teacher-dir','output-dir','source-root','source-manifest'):
    p.add_argument('--'+name,required=True)
p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
p.add_argument('--seed',type=int,default=83)
p.add_argument('--max-batches',type=int)
p.add_argument('--view-version',type=int,choices=(1,2,3),default=1)
a=p.parse_args()
train(data_dir=a.data_dir,warm_dir=a.warm_dir,teacher_dir=a.teacher_dir,output_dir=a.output_dir,
      source_root=a.source_root,source_manifest=json.loads(Path(a.source_manifest).read_text()),
      device=a.device,seed=a.seed,max_batches=a.max_batches,view_version=a.view_version)
