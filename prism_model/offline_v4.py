"""Independent native one-row/two-thread CPU workers, with no labels in workers."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

_model=None


def initialize(directory):
    global _model
    import torch
    from .model_v4 import load_model
    torch.set_num_interop_threads(1)
    _model,_,_=load_model(directory,threads=2)


def score(row):
    import torch
    from .evidence_format import canonical_row
    if canonical_row(row)['text_sha256']!=row['text_sha256']:raise ValueError('Unnormalized worker input')
    with torch.inference_mode():return _model(row).cpu().tolist()


def predictions(directory,rows,workers=3):
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize,initargs=(str(directory),)) as pool:
        yield from pool.map(score,rows,chunksize=8)
