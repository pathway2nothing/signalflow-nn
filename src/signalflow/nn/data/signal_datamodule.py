# src/signalflow/nn/data/signal_datamodule.py
"""
LightningDataModule for temporal signal validation.

Input: Multiple time series with same time index but different trading pairs.
Each pair has same features (already normalized) and signals.

At signal moment: extract window of features + label → training example.
"""
import lightning as L
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Literal, Optional
import numpy as np


class SignalWindowDataset(Dataset):
    """Dataset for extracting temporal windows around signals
    
    Each sample: (features_window, label)
    - features_window: [window_size, n_features] from time series before signal
    - label: target value at signal moment
    
    Args:
        features_df: DataFrame with columns [pair, timestamp, feature_1, ..., feature_n]
        signals_df: DataFrame with columns [pair, timestamp, signal, label]
        window_size: Number of timesteps to look back
        feature_cols: List of feature column names (if None, auto-detect)
        pair_col: Name of pair column. Default: 'pair'
        ts_col: Name of timestamp column. Default: 'timestamp'
        label_col: Name of label column. Default: 'label'
    """
    
    def __init__(
        self,
        features_df: pl.DataFrame,
        signals_df: pl.DataFrame,
        window_size: int,
        feature_cols: Optional[list[str]] = None,
        pair_col: str = "pair",
        ts_col: str = "timestamp",
        label_col: str = "label",
    ):
        self.features_df = features_df
        self.signals_df = signals_df
        self.window_size = window_size
        self.pair_col = pair_col
        self.ts_col = ts_col
        self.label_col = label_col
        
        # Auto-detect feature columns if not provided
        if feature_cols is None:
            # Exclude pair, timestamp, label
            exclude_cols = {pair_col, ts_col, label_col, "signal", "signal_type"}
            self.feature_cols = [
                col for col in features_df.columns 
                if col not in exclude_cols
            ]
        else:
            self.feature_cols = feature_cols
        
        self.n_features = len(self.feature_cols)
        
        # Build index: list of (pair, signal_timestamp, label)
        self._build_index()
    
    def _build_index(self):
        """Build index of valid signals with enough history"""
        self.samples = []
        
        # Group by pair for efficient processing
        for pair in self.signals_df[self.pair_col].unique():
            pair_features = self.features_df.filter(
                pl.col(self.pair_col) == pair
            ).sort(self.ts_col)
            
            pair_signals = self.signals_df.filter(
                pl.col(self.pair_col) == pair
            ).sort(self.ts_col)
            
            # Convert to numpy for faster indexing
            timestamps = pair_features[self.ts_col].to_numpy()
            features_array = pair_features.select(self.feature_cols).to_numpy()
            
            # For each signal, check if we have enough history
            for row in pair_signals.iter_rows(named=True):
                signal_ts = row[self.ts_col]
                label = row[self.label_col]
                
                # Find signal position in features
                signal_idx = np.searchsorted(timestamps, signal_ts)
                
                # Check if we have enough history
                if signal_idx >= self.window_size:
                    # Extract window: [signal_idx - window_size : signal_idx]
                    start_idx = signal_idx - self.window_size
                    end_idx = signal_idx
                    
                    window = features_array[start_idx:end_idx]
                    
                    # Store sample
                    self.samples.append({
                        "pair": pair,
                        "timestamp": signal_ts,
                        "window": window,
                        "label": label,
                    })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            features: [window_size, n_features]
            label: scalar (int or float)
        """
        sample = self.samples[idx]
        
        # Convert to tensors
        features = torch.FloatTensor(sample["window"])  # [window_size, n_features]
        label = torch.tensor(sample["label"], dtype=torch.long)  # Assuming classification
        
        return features, label
    
    def get_sample_info(self, idx: int) -> dict:
        """Get metadata for a sample (for debugging)"""
        return {
            "pair": self.samples[idx]["pair"],
            "timestamp": self.samples[idx]["timestamp"],
        }


class SignalDataModule(L.LightningDataModule):
    """LightningDataModule for temporal signal validation
    
    Handles train/val/test splits and dataloaders for signal windows.
    
    Args:
        features_df: DataFrame with features [pair, timestamp, feature_1, ...]
        signals_df: DataFrame with signals [pair, timestamp, signal, label]
        window_size: Number of timesteps in each window
        train_val_test_split: Tuple of (train_frac, val_frac, test_frac). Default: (0.7, 0.15, 0.15)
        split_strategy: How to split data. Options:
            - 'temporal': Split by time (oldest → train, middle → val, newest → test)
            - 'random': Random split
            - 'pair': Split by trading pair (some pairs for train, others for val/test)
        batch_size: Batch size for dataloaders. Default: 32
        num_workers: Number of workers for dataloaders. Default: 4
        feature_cols: List of feature columns (if None, auto-detect)
        pair_col: Name of pair column. Default: 'pair'
        ts_col: Name of timestamp column. Default: 'timestamp'
        label_col: Name of label column. Default: 'label'
    
    Example:
        >>> # Prepare data
        >>> features_df = pl.DataFrame({
        ...     'pair': ['BTCUSDT', 'BTCUSDT', ...],
        ...     'timestamp': [...],
        ...     'rsi': [...],
        ...     'sma_10': [...],
        ... })
        >>> signals_df = pl.DataFrame({
        ...     'pair': ['BTCUSDT', ...],
        ...     'timestamp': [...],
        ...     'signal': [1, ...],
        ...     'label': [1, 0, ...],  # 1 = profitable, 0 = unprofitable
        ... })
        >>> 
        >>> # Create datamodule
        >>> dm = SignalDataModule(
        ...     features_df=features_df,
        ...     signals_df=signals_df,
        ...     window_size=60,
        ...     batch_size=32,
        ... )
        >>> dm.setup('fit')
        >>> 
        >>> # Use with trainer
        >>> trainer = L.Trainer(...)
        >>> trainer.fit(model, dm)
    """
    
    def __init__(
        self,
        features_df: pl.DataFrame,
        signals_df: pl.DataFrame,
        window_size: int,
        train_val_test_split: tuple[float, float, float] = (0.7, 0.15, 0.15),
        split_strategy: Literal["temporal", "random", "pair"] = "temporal",
        batch_size: int = 32,
        num_workers: int = 4,
        feature_cols: Optional[list[str]] = None,
        pair_col: str = "pair",
        ts_col: str = "timestamp",
        label_col: str = "label",
        shuffle_train: bool = True,
        pin_memory: bool = True,
    ):
        super().__init__()
        
        # Validate split
        assert sum(train_val_test_split) == 1.0, "Split fractions must sum to 1.0"
        
        self.features_df = features_df.sort([pair_col, ts_col])
        self.signals_df = signals_df.sort([pair_col, ts_col])
        self.window_size = window_size
        self.train_val_test_split = train_val_test_split
        self.split_strategy = split_strategy
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.feature_cols = feature_cols
        self.pair_col = pair_col
        self.ts_col = ts_col
        self.label_col = label_col
        self.shuffle_train = shuffle_train
        self.pin_memory = pin_memory
        
        # Datasets (created in setup)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """Split data and create datasets"""
        
        if stage == "fit" or stage is None:
            train_signals, val_signals, test_signals = self._split_signals()
            
            # Create datasets
            self.train_dataset = SignalWindowDataset(
                features_df=self.features_df,
                signals_df=train_signals,
                window_size=self.window_size,
                feature_cols=self.feature_cols,
                pair_col=self.pair_col,
                ts_col=self.ts_col,
                label_col=self.label_col,
            )
            
            self.val_dataset = SignalWindowDataset(
                features_df=self.features_df,
                signals_df=val_signals,
                window_size=self.window_size,
                feature_cols=self.feature_cols,
                pair_col=self.pair_col,
                ts_col=self.ts_col,
                label_col=self.label_col,
            )
            
            self.test_dataset = SignalWindowDataset(
                features_df=self.features_df,
                signals_df=test_signals,
                window_size=self.window_size,
                feature_cols=self.feature_cols,
                pair_col=self.pair_col,
                ts_col=self.ts_col,
                label_col=self.label_col,
            )
        
        if stage == "test":
            _, _, test_signals = self._split_signals()
            
            self.test_dataset = SignalWindowDataset(
                features_df=self.features_df,
                signals_df=test_signals,
                window_size=self.window_size,
                feature_cols=self.feature_cols,
                pair_col=self.pair_col,
                ts_col=self.ts_col,
                label_col=self.label_col,
            )
    
    def _split_signals(self) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Split signals into train/val/test based on strategy"""
        
        train_frac, val_frac, test_frac = self.train_val_test_split
        
        if self.split_strategy == "temporal":
            # Sort by timestamp and split
            signals_sorted = self.signals_df.sort(self.ts_col)
            n = len(signals_sorted)
            
            train_end = int(n * train_frac)
            val_end = int(n * (train_frac + val_frac))
            
            train_signals = signals_sorted[:train_end]
            val_signals = signals_sorted[train_end:val_end]
            test_signals = signals_sorted[val_end:]
            
        elif self.split_strategy == "random":
            # Random split
            n = len(self.signals_df)
            indices = np.random.permutation(n)
            
            train_end = int(n * train_frac)
            val_end = int(n * (train_frac + val_frac))
            
            train_idx = indices[:train_end]
            val_idx = indices[train_end:val_end]
            test_idx = indices[val_end:]
            
            train_signals = self.signals_df[train_idx]
            val_signals = self.signals_df[val_idx]
            test_signals = self.signals_df[test_idx]
            
        elif self.split_strategy == "pair":
            # Split by trading pair
            pairs = self.signals_df[self.pair_col].unique().to_list()
            n_pairs = len(pairs)
            
            train_end = int(n_pairs * train_frac)
            val_end = int(n_pairs * (train_frac + val_frac))
            
            train_pairs = pairs[:train_end]
            val_pairs = pairs[train_end:val_end]
            test_pairs = pairs[val_end:]
            
            train_signals = self.signals_df.filter(pl.col(self.pair_col).is_in(train_pairs))
            val_signals = self.signals_df.filter(pl.col(self.pair_col).is_in(val_pairs))
            test_signals = self.signals_df.filter(pl.col(self.pair_col).is_in(test_pairs))
        
        else:
            raise ValueError(f"Unknown split_strategy: {self.split_strategy}")
        
        return train_signals, val_signals, test_signals
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    
    def get_feature_dim(self) -> int:
        """Get number of features"""
        if self.train_dataset is None:
            # Infer from features_df
            exclude_cols = {self.pair_col, self.ts_col, self.label_col, "signal", "signal_type"}
            feature_cols = [col for col in self.features_df.columns if col not in exclude_cols]
            return len(feature_cols)
        return self.train_dataset.n_features
    
    def get_num_classes(self) -> int:
        """Get number of unique classes"""
        return self.signals_df[self.label_col].n_unique()
    
    def get_class_distribution(self, split: str = "train") -> dict:
        """Get class distribution for a split
        
        Args:
            split: 'train', 'val', or 'test'
        
        Returns:
            Dictionary mapping class -> count
        """
        if split == "train":
            dataset = self.train_dataset
        elif split == "val":
            dataset = self.val_dataset
        elif split == "test":
            dataset = self.test_dataset
        else:
            raise ValueError(f"Unknown split: {split}")
        
        if dataset is None:
            raise ValueError("Dataset not initialized. Call setup() first.")
        
        # Count labels
        labels = [sample["label"] for sample in dataset.samples]
        unique, counts = np.unique(labels, return_counts=True)
        
        return dict(zip(unique.tolist(), counts.tolist()))