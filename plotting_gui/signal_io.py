# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 19:00:11 2025

@author: markablonczy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional

import h5py
import numpy as np


@dataclass
class HDF5Metadata:
    path: str
    n_samples: int
    n_channels: int
    sample_rate: float
    channel_names: Sequence[str]

class HDF5SignalSource:
    """
    Thin wrapper around an Intan-style HDF5 file for read-only access.

    It encapsulates:
    - file opening/closing
    - dataset lookup
    - segment extraction
    - basic metadata (fs, channel names, etc.)
    """

    def __init__(self, h5: h5py.File, dataset, metadata: HDF5Metadata):
        self._h5 = h5
        self._dataset = dataset
        self.metadata = metadata

    @classmethod
    def open(
        cls,
        path: str,
        dataset_key: str,
        sample_rate_attr: str = "sample_rate",
        channel_names_key: Optional[str] = None,
    ) -> "HDF5SignalSource":
        """
        Open an HDF5 file and construct a HDF5SignalSource.

        Parameters
        ----------
        path : str
            Path to the HDF5 file.
        dataset_key : str
            Key inside the HDF5 that holds the main data array
            (e.g. "data" or "signals").
        sample_rate_attr : str
            Attribute name or dataset key for sample rate.
        channel_names_key : Optional[str]
            Dataset key for channel names, if available.

        Returns
        -------
        HDF5SignalSource
        """
        h5 = h5py.File(path, "r")

        dset = h5[dataset_key]
        n_channels, n_samples = dset.shape

        # Try to get sample rate from attribute or a separate dataset
        fs = None
        if sample_rate_attr in dset.attrs:
            fs = float(dset.attrs[sample_rate_attr])
        elif sample_rate_attr in h5:
            fs = float(h5[sample_rate_attr][()])
        else:
            fs = 100.0

        # Channel names
        if channel_names_key is not None and channel_names_key in h5:
            # Assume 1D bytes/str dataset
            raw_names = h5[channel_names_key][:]
            channel_names = [n.decode("utf-8") if isinstance(n, bytes) else str(n)
                             for n in raw_names]
        else:
            channel_names = [f"ch{idx}" for idx in range(n_channels)]

        meta = HDF5Metadata(
            path=path,
            n_samples=n_samples,
            n_channels=n_channels,
            sample_rate=fs,
            channel_names=channel_names,
        )


        return cls(h5, dset, meta)

    def get_segment(self, ch_idx: int, sa: int, sb: int) -> np.ndarray:
        """
        Return segment [sa:sb) for channel ch_idx as a 1D numpy array.
    
        Layout assumed: (channels, samples).
        """
        if ch_idx < 0 or ch_idx >= self.metadata.n_channels:
            raise IndexError(f"Channel index {ch_idx} out of range")
    
        sa_clamped = max(0, min(sa, self.metadata.n_samples))
        sb_clamped = max(sa_clamped, min(sb, self.metadata.n_samples))
    
        # HDF5 layout: (channels, samples)
        seg = self._dataset[ch_idx, sa_clamped:sb_clamped]
        return np.asarray(seg)


    def close(self) -> None:
        try:
            self._h5.close()
        except Exception:
            pass

    def get_channel_decimated(self, ch_idx: int, stride: int) -> np.ndarray:
        """
        Return a decimated copy of the full channel:
        samples 0:n:stride for channel ch_idx, as a 1D np.ndarray.
        Useful for global stats / y-range estimation.
        """
        if stride <= 0:
            stride = 1

        if ch_idx < 0 or ch_idx >= self.metadata.n_channels:
            raise IndexError(f"Channel index {ch_idx} out of range")

        # Dataset layout: (samples, channels)
        seg = self._dataset[0:self.metadata.n_samples:stride, ch_idx]
        return np.asarray(seg)

    def get_segment_decimated(
        self,
        ch_idx: int,
        sa: int,
        sb: int,
        stride: int,
    ) -> np.ndarray:
        if stride <= 1:
            return self.get_segment(ch_idx, sa, sb)
    
        sa_clamped = max(0, min(sa, self.metadata.n_samples))
        sb_clamped = max(sa_clamped, min(sb, self.metadata.n_samples))
    
        # assuming layout (channels, samples) as in open()
        seg = self._dataset[ch_idx, sa_clamped:sb_clamped:stride]
        return np.asarray(seg)
