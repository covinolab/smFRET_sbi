"""
Efficient HDF5 data loader for training with large datasets.

Uses chunk-based loading with contiguous reads for maximum I/O performance.
Supports multi-worker DataLoader for parallel decompression.
"""

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
        
        # Get dataset length without keeping file open
        with h5py.File(self.file_path, 'r') as f:
            self._total_len = len(f['theta'])
        
        self._file = None
        
        # Calculate number of chunks
        self._num_total_chunks = (self._total_len + chunk_size - 1) // chunk_size
        
        if chunk_indices is not None:
            self._chunk_list = list(chunk_indices)
        else:
            self._chunk_list = list(range(self._num_total_chunks))
        
        # Calculate length based on assigned chunks
        self._len = 0
        for ci in self._chunk_list:
            start = ci * chunk_size
            end = min(start + chunk_size, self._total_len)
            self._len += (end - start)
    
    def _get_file(self) -> h5py.File:
        """Lazy file handle (reopened per worker process)."""
        if self._file is None:
            self._file = h5py.File(self.file_path, 'r')
        return self._file
    
    def __len__(self) -> int:
        return self._len
    
    def __getitem__(self, i: Union[int, slice]) -> Tuple[Tensor, Tensor]:
        """Random access for initialization (uses contiguous reads)."""
        f = self._get_file()
        theta, x = f['theta'][i], f['x'][i]
        return (
            torch.from_numpy(theta.astype(np.float32)),
            torch.from_numpy(x.astype(np.float32))
        )
    
    def _get_worker_chunks(self) -> List[int]:
        """Get chunks assigned to this worker (for multi-worker support)."""
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # Single-process loading
            return self._chunk_list
        
        # Multi-worker: partition chunks across workers (round-robin)
        num_workers = worker_info.num_workers
        worker_id = worker_info.id
        return [c for i, c in enumerate(self._chunk_list) if i % num_workers == worker_id]
    
    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        f = self._get_file()
        
        # Get chunks for this worker
        worker_chunks = self._get_worker_chunks()
        if len(worker_chunks) == 0:
            return
        
        # Build chunk boundaries [(start, end), ...]
        chunks = torch.tensor([
            (ci * self.chunk_size, min((ci + 1) * self.chunk_size, self._total_len))
            for ci in worker_chunks
        ])
        
        # Shuffle chunk order
        if self.shuffle:
            chunks = chunks[torch.randperm(len(chunks))]
        
        # Process chunks in groups of chunk_step
        for slices in chunks.split(self.chunk_step):
            # Merge contiguous slices for efficient I/O
            slices = sorted(slices.tolist())
            merged = []
            for s in slices:
                if merged and merged[-1][1] == s[0]:
                    merged[-1][1] = s[1]
                else:
                    merged.append(list(s))
            
            # Load merged chunks (contiguous slice reads = FAST)
            theta = np.concatenate([f['theta'][i:j] for i, j in merged])
            x = np.concatenate([f['x'][i:j] for i, j in merged])
            
            theta = torch.from_numpy(theta.astype(np.float32))
            x = torch.from_numpy(x.astype(np.float32))
            
            # Shuffle samples within loaded data
            if self.shuffle:
                order = torch.randperm(len(theta))
                theta, x = theta[order], x[order]
            
            # Yield batches
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
