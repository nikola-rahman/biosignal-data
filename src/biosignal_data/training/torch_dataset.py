"""A PyTorch IterableDataset over an ExampleSource (needs the `train` extra: torch).

dataset = TorchExamples(source)
loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=8)
for epoch in range(epochs):
    dataset.set_epoch(epoch)  # takes effect with persistent_workers=False (the default)
    for example in loader: ...  # the model gets example.signals only

Each data loader worker of each distributed rank reads its own part of the recordings.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

from biosignal_data.training.loader import Example, ExampleSource


class TorchExamples(IterableDataset[Example]):
    def __init__(self, source: ExampleSource) -> None:
        super().__init__()
        self.source = source
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[Example]:
        info = get_worker_info()
        worker, workers = (info.id, info.num_workers) if info is not None else (0, 1)
        rank, ranks = (
            (dist.get_rank(), dist.get_world_size())
            if dist.is_available() and dist.is_initialized()
            else (0, 1)
        )
        yield from self.source.examples(self.epoch, rank * workers + worker, ranks * workers)
