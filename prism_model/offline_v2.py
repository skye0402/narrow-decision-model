"""Parallel offline workers; each prediction is the normal one-row CPU path."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

_runtime=None
_random_head=None


def initialize(bundle):
 global _runtime,_random_head
 import torch
 from .model_v2 import RuntimeV2
 from .neural import OptionScorer
 torch.set_num_interop_threads(1)
 _runtime=RuntimeV2(bundle,threads=2,require_calibration=False)
 with torch.random.fork_rng():
  torch.manual_seed(1729);_random_head=OptionScorer(_runtime.model.head.hidden_size).eval()


def score(row):
 import torch
 with torch.inference_mode():
  features=_runtime.model.features(row)
  return (_runtime.model.head(features.context,features.options).cpu().tolist(),
          _random_head(features.context,features.options).cpu().tolist())


def predictions(bundle,rows,workers=3):
 with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
    initializer=initialize,initargs=(str(bundle),)) as executor:
  yield from executor.map(score,rows,chunksize=8)
