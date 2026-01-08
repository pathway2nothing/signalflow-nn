from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


class SignalWindowDataset(Dataset):
    """Dataset that creates windows ONLY at signal timestamps.
    
    For each signal at time t for pair P, creates a window from P's history:
    [t - span, t] features from pair P only, sampled with a stride.
    
    Args:
        features_df: DataFrame with [pair, timestamp, feature_1, ...].
        signals_df: DataFrame with [pair, timestamp, label] - ONLY signal rows.
        window_size: Number of timesteps in each window (output size).
        window_timeframe: Stride for sampling history (dilation). 
                          1 = consecutive rows, 5 = every 5th row.
        feature_cols: List of feature columns (auto-detected if None).
        pair_col: Name of pair column.
        ts_col: Name of timestamp column.
        label_col: Name of label column.
    """
    
    def __init__(
        self,
        features_df: pl.DataFrame,
        signals_df: pl.DataFrame,
        window_size: int = 60,
        window_timeframe: int = 1,
        feature_cols: Optional[list[str]] = None,
        pair_col: str = "pair",
        ts_col: str = "timestamp",
        label_col: str = "label",
    ):
        self.window_size = window_size
        self.window_timeframe = window_timeframe
        self.pair_col = pair_col
        self.ts_col = ts_col
        self.label_col = label_col
        
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
        """Build windows only for signal timestamps using dilated history."""
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
        
        required_span = (self.window_size - 1) * self.window_timeframe
        
        for signal_row_idx, row in enumerate(signals_df.iter_rows(named=True)):
            pair = row[self.pair_col]
            ts = row[self.ts_col]
            label = row[self.label_col] if self.label_col in row else 0
            
            if pair not in self.pair_data:
                skipped_unknown_pair += 1
                continue
            
            pair_info = self.pair_data[pair]
            
            if ts not in pair_info["ts_to_idx"]:
                skipped_no_timestamp += 1
                continue
            
            signal_idx = pair_info["ts_to_idx"][ts]
            
            if signal_idx < required_span:
                skipped_insufficient += 1
                continue
            
            window_start = signal_idx - required_span
            window_end = signal_idx + 1
            
            self.windows.append((pair, window_start, window_end, label))
            self.valid_signal_indices.append(signal_row_idx)
        
        print(f"SignalWindowDataset: {len(self.windows)} valid windows from {signals_df.height} signals")
        if (skipped_unknown_pair + skipped_no_timestamp + skipped_insufficient) > 0:
            print(f"  Skipped: {skipped_unknown_pair} (unknown pair), "
                  f"{skipped_no_timestamp} (no timestamp), "
                  f"{skipped_insufficient} (insufficient history)")
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair, window_start, window_end, label = self.windows[idx]
        
        feature_matrix = self.pair_data[pair]["feature_matrix"]
        
        # Apply stride slice: start:stop:step
        window = feature_matrix[window_start:window_end:self.window_timeframe]
        
        return (
            torch.tensor(window, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )