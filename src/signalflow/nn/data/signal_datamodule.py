# src/signalflow/nn/data/signal_datamodule.py
"""
Data module for signal-based temporal classification.

Key concept: Windows are created ONLY for signal timestamps, not for all data.
This ensures the model learns to validate detected signals, not predict on every bar.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L


class SignalWindowDataset(Dataset):
    """Dataset that creates windows ONLY at signal timestamps.
    
    For each signal at time t for pair P, creates a window from P's history:
    [t-window_size+1, t] features from pair P only.
    
    This is the core dataset for training signal validators.
    
    Args:
        features_df: DataFrame with [pair, timestamp, feature_1, ...].
        signals_df: DataFrame with [pair, timestamp, label] - ONLY signal rows.
        window_size: Number of timesteps in each window.
        feature_cols: List of feature columns (auto-detected if None).
        pair_col: Name of pair column.
        ts_col: Name of timestamp column.
        label_col: Name of label column.
        
    Example:
        >>> # Signal for BTCUSDT at 10:30 -> window from BTCUSDT [10:01-10:30]
        >>> # Signal for ETHUSDT at 10:30 -> window from ETHUSDT [10:01-10:30]
        >>> dataset = SignalWindowDataset(
        ...     features_df=all_features,
        ...     signals_df=labeled_signals,
        ...     window_size=30,
        ... )
    """
    
    def __init__(
        self,
        features_df: pl.DataFrame,
        signals_df: pl.DataFrame,
        window_size: int = 60,
        feature_cols: Optional[list[str]] = None,
        pair_col: str = "pair",
        ts_col: str = "timestamp",
        label_col: str = "label",
    ):
        self.window_size = window_size
        self.pair_col = pair_col
        self.ts_col = ts_col
        self.label_col = label_col
        
        # Determine feature columns
        exclude_cols = {pair_col, ts_col, label_col, "signal", "signal_type"}
        if feature_cols is not None:
            self.feature_cols = feature_cols
        else:
            self.feature_cols = [
                c for c in features_df.columns 
                if c not in exclude_cols
            ]
        
        # Build per-pair data structures
        self._build_pair_data(features_df)
        
        # Build windows for each signal
        self._build_signal_windows(signals_df)
    
    def _build_pair_data(self, features_df: pl.DataFrame):
        """Build per-pair feature matrices and timestamp indices."""
        self.pair_data: dict[str, dict] = {}
        
        for pair in features_df[self.pair_col].unique().to_list():
            # Get this pair's data, sorted by timestamp
            pair_df = (
                features_df
                .filter(pl.col(self.pair_col) == pair)
                .sort(self.ts_col)
            )
            
            # Feature matrix for this pair
            feature_matrix = pair_df.select(self.feature_cols).to_numpy().astype(np.float32)
            
            # Timestamp -> index mapping for this pair
            timestamps = pair_df[self.ts_col].to_list()
            ts_to_idx = {ts: idx for idx, ts in enumerate(timestamps)}
            
            self.pair_data[pair] = {
                "feature_matrix": feature_matrix,
                "ts_to_idx": ts_to_idx,
                "n_rows": len(timestamps),
            }
    
    def _build_signal_windows(self, signals_df: pl.DataFrame):
        """Build windows only for signal timestamps, using same-pair history."""
        self.windows = []  # List of (pair, window_start_idx, window_end_idx, label)
        
        skipped_unknown_pair = 0
        skipped_no_timestamp = 0
        skipped_insufficient = 0
        
        for row in signals_df.iter_rows(named=True):
            pair = row[self.pair_col]
            ts = row[self.ts_col]
            label = row[self.label_col]
            
            # Check pair exists
            if pair not in self.pair_data:
                skipped_unknown_pair += 1
                continue
            
            pair_info = self.pair_data[pair]
            
            # Find timestamp index within this pair's data
            if ts not in pair_info["ts_to_idx"]:
                skipped_no_timestamp += 1
                continue
            
            signal_idx = pair_info["ts_to_idx"][ts]
            
            # Need at least window_size bars before (including current)
            if signal_idx < self.window_size - 1:
                skipped_insufficient += 1
                continue
            
            # Window indices within this pair's feature matrix
            window_start = signal_idx - self.window_size + 1
            window_end = signal_idx + 1  # exclusive
            
            self.windows.append((pair, window_start, window_end, label))
        
        total_skipped = skipped_unknown_pair + skipped_no_timestamp + skipped_insufficient
        if total_skipped > 0:
            print(f"SignalWindowDataset: {len(self.windows)} valid windows")
            print(f"  Skipped: {skipped_unknown_pair} (unknown pair), "
                  f"{skipped_no_timestamp} (timestamp not found), "
                  f"{skipped_insufficient} (insufficient history)")
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair, window_start, window_end, label = self.windows[idx]
        
        # Extract window from THIS PAIR's feature matrix
        feature_matrix = self.pair_data[pair]["feature_matrix"]
        window = feature_matrix[window_start:window_end]  # [window_size, num_features]
        
        return (
            torch.tensor(window, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


@dataclass
class SignalDataModule(L.LightningDataModule):
    """Lightning DataModule for signal validation training.
    
    Creates train/val/test splits from labeled signals.
    Windows are created ONLY at signal timestamps.
    
    Args:
        features_df: Full feature history [pair, timestamp, features...].
        signals_df: Labeled signals [pair, timestamp, label] - only signal rows.
        window_size: Temporal window size.
        train_val_test_split: Split ratios.
        split_strategy: How to split data.
        batch_size: Batch size.
        num_workers: DataLoader workers.
        feature_cols: Feature columns (auto-detected if None).
        
    Example:
        >>> # signals_df comes from: detector -> labeler
        >>> datamodule = SignalDataModule(
        ...     features_df=all_features,
        ...     signals_df=labeled_signals,  # Only signal timestamps!
        ...     window_size=60,
        ...     batch_size=64,
        ... )
        >>> trainer.fit(model, datamodule)
    """
    
    features_df: pl.DataFrame
    signals_df: pl.DataFrame
    window_size: int = 60
    train_val_test_split: tuple[float, float, float] = (0.7, 0.15, 0.15)
    split_strategy: Literal["temporal", "random", "pair"] = "temporal"
    batch_size: int = 32
    num_workers: int = 4
    feature_cols: Optional[list[str]] = None
    
    pair_col: str = "pair"
    ts_col: str = "timestamp"
    label_col: str = "label"
    
    # Internal state
    train_signals: Optional[pl.DataFrame] = field(default=None, init=False)
    val_signals: Optional[pl.DataFrame] = field(default=None, init=False)
    test_signals: Optional[pl.DataFrame] = field(default=None, init=False)
    
    def __post_init__(self):
        super().__init__()
    
    def setup(self, stage: Optional[str] = None):
        """Split signals into train/val/test."""
        train_ratio, val_ratio, test_ratio = self.train_val_test_split
        
        if self.split_strategy == "temporal":
            self._temporal_split(train_ratio, val_ratio, test_ratio)
        elif self.split_strategy == "random":
            self._random_split(train_ratio, val_ratio, test_ratio)
        elif self.split_strategy == "pair":
            self._pair_split(train_ratio, val_ratio, test_ratio)
        else:
            raise ValueError(f"Unknown split_strategy: {self.split_strategy}")
        
        print(f"Data split: train={len(self.train_signals)}, "
              f"val={len(self.val_signals)}, test={len(self.test_signals)}")
    
    def _temporal_split(self, train_ratio: float, val_ratio: float, test_ratio: float):
        """Split by time - earliest for train, latest for test."""
        # Sort by timestamp
        sorted_signals = self.signals_df.sort(self.ts_col)
        n = len(sorted_signals)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        self.train_signals = sorted_signals[:train_end]
        self.val_signals = sorted_signals[train_end:val_end]
        self.test_signals = sorted_signals[val_end:]
    
    def _random_split(self, train_ratio: float, val_ratio: float, test_ratio: float):
        """Random shuffle split."""
        n = len(self.signals_df)
        indices = np.random.permutation(n)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        self.train_signals = self.signals_df[indices[:train_end].tolist()]
        self.val_signals = self.signals_df[indices[train_end:val_end].tolist()]
        self.test_signals = self.signals_df[indices[val_end:].tolist()]
    
    def _pair_split(self, train_ratio: float, val_ratio: float, test_ratio: float):
        """Split by pair - some pairs for train, others for val/test."""
        pairs = self.signals_df[self.pair_col].unique().to_list()
        np.random.shuffle(pairs)
        
        n_pairs = len(pairs)
        train_end = int(n_pairs * train_ratio)
        val_end = int(n_pairs * (train_ratio + val_ratio))
        
        train_pairs = set(pairs[:train_end])
        val_pairs = set(pairs[train_end:val_end])
        test_pairs = set(pairs[val_end:])
        
        self.train_signals = self.signals_df.filter(
            pl.col(self.pair_col).is_in(list(train_pairs))
        )
        self.val_signals = self.signals_df.filter(
            pl.col(self.pair_col).is_in(list(val_pairs))
        )
        self.test_signals = self.signals_df.filter(
            pl.col(self.pair_col).is_in(list(test_pairs))
        )
    
    def _create_dataset(self, signals: pl.DataFrame) -> SignalWindowDataset:
        """Create dataset for given signals."""
        return SignalWindowDataset(
            features_df=self.features_df,
            signals_df=signals,
            window_size=self.window_size,
            feature_cols=self.feature_cols,
            pair_col=self.pair_col,
            ts_col=self.ts_col,
            label_col=self.label_col,
        )
    
    def train_dataloader(self) -> DataLoader:
        dataset = self._create_dataset(self.train_signals)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        dataset = self._create_dataset(self.val_signals)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self) -> DataLoader:
        dataset = self._create_dataset(self.test_signals)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )