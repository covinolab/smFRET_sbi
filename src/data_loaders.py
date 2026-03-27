"""
Efficient HDF5 data loader for training with large datasets.

Uses chunk-based loading with contiguous reads for maximum I/O performance.
Supports multi-worker DataLoader for parallel decompression.
"""

import random
from pathlib import Path
from typing import Union, List, Optional, Tuple, Iterator

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import IterableDataset


class H5Dataset(IterableDataset):
    """
    Efficient iterable dataset for HDF5 files with chunk-based loading.

    Loads contiguous chunks and shuffles for randomization. Uses only slice
    reads (no fancy indexing) for maximum I/O performance.

    Shuffling strategy:
        1. Shuffle chunk order (which chunks to read first)
        2. Load chunk_step chunks at once into RAM
        3. Shuffle samples within the loaded data
        4. Yield batches from shuffled RAM data

    This achieves good randomization while maintaining fast sequential I/O.

    Multi-worker support:
        Chunks are partitioned across workers (round-robin), so each worker
        reads different chunks. This enables parallel decompression for
        compressed HDF5 files.

    Parameters
    ----------
    file : str or Path
        Path to HDF5 file containing 'theta' and 'x' datasets.
    batch_size : int, optional
        Size of yielded batches. If None, yields individual samples.
    chunk_size : int
        Number of samples per chunk. Should match HDF5 chunk size for best I/O.
    chunk_step : int
        Number of chunks to load at once. Larger = more RAM but better shuffle.
    shuffle : bool
        Whether to shuffle chunk order and samples within loaded data.
    chunk_indices : list, optional
        Which chunk indices to use (for train/val split by chunk).
        If None, uses all chunks.

    Example
    -------
    >>> # Create train/val split by chunks
    >>> num_chunks = 100
    >>> train_chunks = list(range(90))
    >>> val_chunks = list(range(90, 100))
    >>>
    >>> train_dataset = H5Dataset(
    ...     'data.h5', batch_size=128, chunk_size=1000, chunk_step=10,
    ...     shuffle=True, chunk_indices=train_chunks
    ... )
    >>>
    >>> # Wrap in DataLoader for multi-worker prefetching
    >>> train_loader = DataLoader(
    ...     train_dataset, batch_size=None, num_workers=8,
    ...     pin_memory=True, prefetch_factor=4
    ... )
    """

    def __init__(
        self,
        file: Union[str, Path],
        batch_size: int = None,
        chunk_size: int = 1000,
        chunk_step: int = 10,
        shuffle: bool = False,
        chunk_indices: Optional[List[int]] = None,
        drop_last: bool = False,
    ):
        super().__init__()
        self.file_path = Path(file)
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.chunk_step = chunk_step
        self.shuffle = shuffle
        self.drop_last = drop_last

        with h5py.File(self.file_path, 'r') as f:
            self._total_len = len(f['theta'])

        self._file = None

        self._num_total_chunks = (self._total_len + chunk_size - 1) // chunk_size
        self._chunk_list = list(chunk_indices) if chunk_indices is not None else list(range(self._num_total_chunks))
        self._len = sum(self._chunk_span(ci) for ci in self._chunk_list)

    # ------------------------------------------------------------------
    # Chunk helpers
    # ------------------------------------------------------------------

    def _chunk_bounds(self, ci: int) -> Tuple[int, int]:
        """Return (start, end) sample indices for chunk ci."""
        start = ci * self.chunk_size
        return start, min(start + self.chunk_size, self._total_len)

    def _chunk_span(self, ci: int) -> int:
        """Return the number of samples in chunk ci."""
        start, end = self._chunk_bounds(ci)
        return end - start

    @staticmethod
    def _partition_chunks(chunk_list: List[int], num_workers: int, worker_id: int) -> List[int]:
        """Round-robin partition of chunk_list for a given worker."""
        return [c for i, c in enumerate(chunk_list) if i % num_workers == worker_id]

    @staticmethod
    def _merge_contiguous(bounds: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Merge adjacent [start, end) intervals to minimise slice reads."""
        merged: List[Tuple[int, int]] = []
        for start, end in sorted(bounds):
            if merged and merged[-1][1] == start:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        return merged

    def _get_file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.file_path, 'r', swmr=True, libver='latest')
        return self._file

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __len__(self) -> int:
        return self._len

    def num_batches(self, num_workers: int = 1) -> int:
        """Return number of batches yielded in one full epoch.

        This mirrors the current iteration behavior, including:
        - worker chunk partitioning (round-robin)
        - chunk grouping via ``chunk_step``
        - batch splitting with optional ``drop_last``

        Parameters
        ----------
        num_workers : int
            Number of DataLoader workers used with this dataset.
        """
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")

        total_batches = 0
        for worker_id in range(num_workers):
            worker_chunks = self._partition_chunks(self._chunk_list, num_workers, worker_id)
            for i in range(0, len(worker_chunks), self.chunk_step):
                group_samples = sum(self._chunk_span(ci) for ci in worker_chunks[i:i + self.chunk_step])
                if self.batch_size is None:
                    total_batches += group_samples
                elif self.drop_last:
                    total_batches += group_samples // self.batch_size
                else:
                    total_batches += (group_samples + self.batch_size - 1) // self.batch_size

        return total_batches

    def __getitem__(self, i: Union[int, slice]) -> Tuple[Tensor, Tensor]:
        """Random access for initialization (uses contiguous reads)."""
        f = self._get_file()
        theta, x = f['theta'][i], f['x'][i]
        return (
            torch.from_numpy(theta.astype(np.float32)),
            torch.from_numpy(x.astype(np.float32)),
        )

    def _get_worker_chunks(self) -> List[int]:
        """Return chunks assigned to this worker."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return self._chunk_list
        return self._partition_chunks(self._chunk_list, worker_info.num_workers, worker_info.id)

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        f = self._get_file()

        worker_chunks = self._get_worker_chunks()
        if not worker_chunks:
            return

        bounds = [self._chunk_bounds(ci) for ci in worker_chunks]

        if self.shuffle:
            random.shuffle(bounds)

        for i in range(0, len(bounds), self.chunk_step):
            group = bounds[i:i + self.chunk_step]
            intervals = self._merge_contiguous(group)

            theta = np.concatenate([f['theta'][s:e] for s, e in intervals])
            x = np.concatenate([f['x'][s:e] for s, e in intervals])

            theta = torch.from_numpy(theta.astype(np.float32))
            x = torch.from_numpy(x.astype(np.float32))

            if self.shuffle:
                order = torch.randperm(len(theta))
                theta, x = theta[order], x[order]

            if self.batch_size is None:
                yield from zip(theta, x)
            else:
                for tb, xb in zip(theta.split(self.batch_size), x.split(self.batch_size)):
                    if self.drop_last and len(tb) < self.batch_size:
                        continue
                    yield tb, xb

    def __del__(self):
        if self._file is not None:
            self._file.close()
