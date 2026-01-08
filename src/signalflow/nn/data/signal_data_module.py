from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional
import os

import polars as pl
from torch.utils.data import DataLoader
import lightning as L

from signalflow.nn.data.signal_window_dataset import SignalWindowDataset
from signalflow.nn.data.ts_preprocessor import TimeSeriesPreprocessor

@dataclass
class SignalDataModule(L.LightningDataModule):
    """
    Lightning DataModule that handles Time-Series Scaling and Signal Splitting.
    """
    
    features_df: pl.DataFrame
    signals_df: pl.DataFrame
    preprocessor: TimeSeriesPreprocessor
    window_size: int = 60
    window_timeframe: int = 1
    train_val_test_split: tuple[float, float, float] = (0.7, 0.15, 0.15)
    split_strategy: Literal["temporal", "random", "pair"] = "temporal"
    batch_size: int = 32
    num_workers: int = 4
    feature_cols: Optional[list[str]] = None
    
    pair_col: str = "pair"
    ts_col: str = "timestamp"
    label_col: str = "label"
    
    train_signals: Optional[pl.DataFrame] = field(default=None, init=False)
    val_signals: Optional[pl.DataFrame] = field(default=None, init=False)
    test_signals: Optional[pl.DataFrame] = field(default=None, init=False)
    processed_features: Optional[pl.DataFrame] = field(default=None, init=False)

    def __post_init__(self):
        super().__init__()
        if self.feature_cols is None:
            exclude = {self.pair_col, self.ts_col, self.label_col, "signal", "signal_type"}
            self.feature_cols = [c for c in self.features_df.columns if c not in exclude]

    def setup(self, stage: Optional[str] = None):
        if self.train_signals is not None:
            return

        train_ratio, val_ratio, test_ratio = self.train_val_test_split
        
        if self.split_strategy == "temporal":
            self._temporal_split(train_ratio, val_ratio, test_ratio)
        elif self.split_strategy == "random":
            self._random_split(train_ratio, val_ratio, test_ratio)
        elif self.split_strategy == "pair":
            self._pair_split(train_ratio, val_ratio, test_ratio)
        else:
            raise ValueError(f"Unknown split_strategy: {self.split_strategy}")

        print(f"[SignalDataModule] Split: Train={len(self.train_signals)}, "
              f"Val={len(self.val_signals)}, Test={len(self.test_signals)}")
        
        if self.split_strategy == "temporal":
            last_train_ts = self.train_signals.select(pl.col(self.ts_col).max()).item()
            print(f"[SignalDataModule] Fitting preprocessor on data <= {last_train_ts}")
            train_features_subset = self.features_df.filter(
                pl.col(self.ts_col) <= last_train_ts
            )
        else:
            print("[SignalDataModule] Warning: Non-temporal split used. Fitting on full train pairs.")
            train_features_subset = self.features_df 

        print("[SignalDataModule] Fitting scaler...")
        self.preprocessor.fit(train_features_subset, feature_cols=self.feature_cols)

        print("[SignalDataModule] Transforming all features...")
        self.processed_features = self.preprocessor.transform(self.features_df)
        
        if isinstance(self.processed_features, pl.LazyFrame):
            self.processed_features = self.processed_features.collect()

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

    def _create_dataset(self, signals: pl.DataFrame) -> SignalWindowDataset:
        if self.processed_features is None:
            raise RuntimeError("Features have not been processed. Run setup() first.")
            
        return SignalWindowDataset(
            features_df=self.processed_features,
            signals_df=signals,
            window_size=self.window_size,
            window_timeframe=self.window_timeframe,
            feature_cols=self.feature_cols,
            pair_col=self.pair_col,
            ts_col=self.ts_col,
            label_col=self.label_col,
        )

    def _temporal_split(self, train_ratio, val_ratio, test_ratio):
        sorted_signals = self.signals_df.sort(self.ts_col)
        n = len(sorted_signals)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        self.train_signals = sorted_signals[:train_end]
        self.val_signals = sorted_signals[train_end:val_end]
        self.test_signals = sorted_signals[val_end:]

    def _random_split(self, train_ratio, val_ratio, test_ratio):
        shuffled = self.signals_df.sample(fraction=1.0, shuffle=True, seed=42)
        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        self.train_signals = shuffled[:train_end]
        self.val_signals = shuffled[train_end:val_end]
        self.test_signals = shuffled[val_end:]

    def _pair_split(self, train_ratio, val_ratio, test_ratio):
        pairs = self.signals_df[self.pair_col].unique().to_list()
        import random
        random.seed(42)
        random.shuffle(pairs)
        
        n = len(pairs)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_pairs = pairs[:train_end]
        val_pairs = pairs[train_end:val_end]
        test_pairs = pairs[val_end:]
        
        self.train_signals = self.signals_df.filter(pl.col(self.pair_col).is_in(train_pairs))
        self.val_signals = self.signals_df.filter(pl.col(self.pair_col).is_in(val_pairs))
        self.test_signals = self.signals_df.filter(pl.col(self.pair_col).is_in(test_pairs))

    def save_preprocessor(self, path: str):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        self.preprocessor.save(path)