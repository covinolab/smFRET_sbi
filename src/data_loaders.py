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

    LOG_IPT_EPS = 1e-12  # ε added inside log(Δt + ε); sub-picosecond, well below any real timing scale.

    def __init__(
        self,
        file: Union[str, Path],
        batch_size: int = None,
        chunk_size: int = 1000,
        chunk_step: int = 10,
        shuffle: bool = False,
        chunk_indices: Optional[List[int]] = None,
        drop_last: bool = False,
        length_buckets: Optional[List[int]] = None,
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
            self._n_max_photons = int(f.attrs.get('n_max_photons',
                                                  f['interphoton_times'].shape[-1]))
            # set-based mode: ipt has shape (N_sims, num_traces_per_set, N_max)
            self._is_set_mode = f['interphoton_times'].ndim == 3

        # Length buckets: each batch is trimmed (along the photon axis) to the
        # smallest bucket >= its local max. A small fixed set of bucket sizes
        # bounds the number of distinct compiled graphs under torch.compile.
        # The top bucket is always pinned at n_max_photons so the trim is a
        # no-op when a batch happens to contain a max-length trace.
        if length_buckets is None:
            buckets: List[int] = []
            b = 256
            while b < self._n_max_photons:
                buckets.append(b)
                b *= 2
            buckets.append(self._n_max_photons)
        else:
            buckets = sorted({int(b) for b in length_buckets if int(b) > 0})
            if not buckets or buckets[-1] < self._n_max_photons:
                buckets.append(self._n_max_photons)
        self._length_buckets: Tuple[int, ...] = tuple(buckets)

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
    def _bucket_up(n: int, buckets: Tuple[int, ...]) -> int:
        """Smallest bucket >= n; falls back to n if it exceeds every bucket."""
        for b in buckets:
            if n <= b:
                return b
        return n

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

    @property
    def n_max_photons(self) -> int:
        """Padding budget per trace (last axis of interphoton_times)."""
        return self._n_max_photons

    @property
    def is_set_mode(self) -> bool:
        """Whether each sample is a stack of num_traces_per_set traces."""
        return self._is_set_mode

    @classmethod
    def _arrivals_to_x(
        cls,
        ipt: np.ndarray,
        chans: np.ndarray,
        lengths: np.ndarray,
    ) -> Tensor:
        """Convert (interphoton_times, channels, lengths) → single `x` tensor.

        x has 4 channels per photon: (log(Δt + ε), G_onehot, R_onehot, mask).
        The last channel is the validity mask (1=real photon, 0=padding) —
        PhotonTCN extracts it with `mask = x[..., -1].bool()`.

        Inputs (numpy):
            ipt     : shape (..., N_max), float
            chans   : shape (..., N_max), uint8 in {0, 1}
            lengths : shape (...),        integer
        Returns (torch): shape (..., N_max, 4), float32
        """
        n_max = ipt.shape[-1]
        ipt_f32 = ipt.astype(np.float32, copy=False)
        log_ipt = np.log(ipt_f32 + np.float32(cls.LOG_IPT_EPS))
        g_oh = (chans == 0).astype(np.float32)
        r_oh = (chans == 1).astype(np.float32)
        idx = np.arange(n_max, dtype=np.int64)
        mask_bool = idx < lengths[..., None]  # (..., n_max)
        mask_f = mask_bool.astype(np.float32)
        # Zero out features in padded positions (defensive; the mask channel is
        # what PhotonTCN's pool gates on, but other consumers may not check).
        log_ipt = log_ipt * mask_f
        g_oh = g_oh * mask_f
        r_oh = r_oh * mask_f
        x = np.stack([log_ipt, g_oh, r_oh, mask_f], axis=-1)
        return torch.from_numpy(x)

    def __getitem__(self, i: Union[int, slice]) -> Tuple[Tensor, Tensor]:
        """Random access for initialization (uses contiguous reads)."""
        f = self._get_file()
        theta = torch.from_numpy(f['theta'][i].astype(np.float32))
        ipt = np.asarray(f['interphoton_times'][i])
        chans = np.asarray(f['channels'][i])
        lengths = np.asarray(f['lengths'][i], dtype=np.int64)
        x = self._arrivals_to_x(ipt, chans, lengths)
        return theta, x

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

            theta_np = np.concatenate([f['theta'][s:e] for s, e in intervals])
            ipt_np = np.concatenate([f['interphoton_times'][s:e] for s, e in intervals])
            chans_np = np.concatenate([f['channels'][s:e] for s, e in intervals])
            lengths_np = np.concatenate([f['lengths'][s:e] for s, e in intervals])
            lengths_np = lengths_np.astype(np.int64, copy=False)

            # Group-level bucket trim: drop zero-padding past the bucketed max
            # for the whole loaded buffer before the log/one-hot work in
            # _arrivals_to_x. Bucketed (not exact) so it stays >= every
            # subsequent per-batch bucket.
            group_max = int(lengths_np.max()) if lengths_np.size else 0
            group_bucket = self._bucket_up(group_max, self._length_buckets)
            if group_bucket < ipt_np.shape[-1]:
                ipt_np = ipt_np[..., :group_bucket]
                chans_np = chans_np[..., :group_bucket]

            theta = torch.from_numpy(theta_np.astype(np.float32))
            x = self._arrivals_to_x(ipt_np, chans_np, lengths_np)
            lengths = torch.from_numpy(lengths_np)

            if self.shuffle:
                order = torch.randperm(len(theta))
                theta = theta[order]
                x = x[order]
                lengths = lengths[order]

            if self.batch_size is None:
                yield from zip(theta, x)
            else:
                theta_splits = theta.split(self.batch_size)
                x_splits = x.split(self.batch_size)
                lengths_splits = lengths.split(self.batch_size)
                for tb, xb, lb in zip(theta_splits, x_splits, lengths_splits):
                    if self.drop_last and len(tb) < self.batch_size:
                        continue
                    # Per-batch bucket trim along the photon axis (dim -2).
                    # local_bucket is drawn from a fixed set, so the encoder
                    # sees at most len(length_buckets) distinct shapes — keeps
                    # torch.compile from recompiling per batch.
                    local_max = int(lb.max().item())
                    local_bucket = self._bucket_up(local_max, self._length_buckets)
                    if local_bucket < xb.shape[-2]:
                        xb = xb[..., :local_bucket, :].contiguous()
                    yield tb, xb

    def __del__(self):
        if self._file is not None:
            self._file.close()
