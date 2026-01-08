from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


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
        
        self._build_pair_data(features_df)
        
        self._build_signal_windows(signals_df)
    
    def _build_pair_data(self, features_df: pl.DataFrame):
        """Build per-pair feature matrices and timestamp indices."""
        self.pair_data: dict[str, dict] = {}
        
        if self.ts_col in features_df.columns:
            ts_dtype = features_df.schema.get(self.ts_col)
            if isinstance(ts_dtype, pl.Datetime) and ts_dtype.time_zone is not None:
                features_df = features_df.with_columns(
                    pl.col(self.ts_col).dt.replace_time_zone(None)
                )
        
        for pair in features_df[self.pair_col].unique().to_list():
            pair_df = (
                features_df
                .filter(pl.col(self.pair_col) == pair)
                .sort(self.ts_col)
            )
            
            feature_matrix = pair_df.select(self.feature_cols).to_numpy().astype(np.float32)
            
            nan_count = np.isnan(feature_matrix).sum()
            if nan_count > 0:
                feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
            
            timestamps = pair_df[self.ts_col].to_list()
            ts_to_idx = {ts: idx for idx, ts in enumerate(timestamps)}
            
            self.pair_data[pair] = {
                "feature_matrix": feature_matrix,
                "ts_to_idx": ts_to_idx,
                "n_rows": len(timestamps),
            }
    
    def _build_signal_windows(self, signals_df: pl.DataFrame):
        """Build windows only for signal timestamps, using same-pair history."""
        self.windows = []
        self.valid_signal_indices = []  
        
        if self.ts_col in signals_df.columns:
            ts_dtype = signals_df.schema.get(self.ts_col)
            if isinstance(ts_dtype, pl.Datetime) and ts_dtype.time_zone is not None:
                signals_df = signals_df.with_columns(
                    pl.col(self.ts_col).dt.replace_time_zone(None)
                )
        
        skipped_unknown_pair = 0
        skipped_no_timestamp = 0
        skipped_insufficient = 0
        
        debug_shown = False
        
        for signal_row_idx, row in enumerate(signals_df.iter_rows(named=True)):
            pair = row[self.pair_col]
            ts = row[self.ts_col]
            label = row[self.label_col]
            
            if pair not in self.pair_data:
                skipped_unknown_pair += 1
                continue
            
            pair_info = self.pair_data[pair]
            
            if ts not in pair_info["ts_to_idx"]:
                skipped_no_timestamp += 1
                if not debug_shown and signal_row_idx < 5:
                    sample_feature_ts = list(pair_info["ts_to_idx"].keys())[0] if pair_info["ts_to_idx"] else None
                    print(f"  DEBUG: Signal ts type={type(ts)}, value={ts}")
                    print(f"  DEBUG: Feature ts type={type(sample_feature_ts)}, value={sample_feature_ts}")
                    debug_shown = True
                continue
            
            signal_idx = pair_info["ts_to_idx"][ts]
            
            if signal_idx < self.window_size - 1:
                skipped_insufficient += 1
                continue
            
            window_start = signal_idx - self.window_size + 1
            window_end = signal_idx + 1  # exclusive
            
            self.windows.append((pair, window_start, window_end, label))
            self.valid_signal_indices.append(signal_row_idx)
        
        total_skipped = skipped_unknown_pair + skipped_no_timestamp + skipped_insufficient
        print(f"SignalWindowDataset: {len(self.windows)} valid windows from {signals_df.height} signals")
        if total_skipped > 0:
            print(f"  Skipped: {skipped_unknown_pair} (unknown pair), "
                  f"{skipped_no_timestamp} (timestamp not found), "
                  f"{skipped_insufficient} (insufficient history)")
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair, window_start, window_end, label = self.windows[idx]
        
        feature_matrix = self.pair_data[pair]["feature_matrix"]
        window = feature_matrix[window_start:window_end]  
        
        return (
            torch.tensor(window, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )

