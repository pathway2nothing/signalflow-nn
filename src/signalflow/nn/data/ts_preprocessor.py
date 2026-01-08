import polars as pl
import numpy as np
import pickle
from typing import Dict, List, Optional, Union, Tuple
from loguru import logger
from dataclasses import dataclass

@dataclass
class ScalerConfig:
    method: str = "robust"  # 'robust', 'standard', 'minmax'
    scope: str = "group"    # 'global' (all data), 'group' (per asset_id)

class TimeSeriesPreprocessor:
    """
    High-performance Time-Series Preprocessor built on Polars.
    Handles per-asset scaling, sequence windowing, and tensor generation for RNNs/Transformers.
    """

    def __init__(
        self,
        feature_configs: Dict[str, ScalerConfig] = None,
        default_config: ScalerConfig = ScalerConfig(method="robust", scope="group"),
        time_col: str = "timestamp",
        group_col: str = "asset_id",
        fill_strategy: str = "forward"
    ):
        """
        Args:
            feature_configs: Dict mapping specific columns to ScalerConfig.
            default_config: Default scaling strategy for numeric columns not in feature_configs.
            time_col: Name of the timestamp column.
            group_col: Name of the grouping column (e.g., 'symbol', 'asset_id').
            fill_strategy: Strategy to handle NaNs ('forward', 'zero', 'median').
        """
        self.feature_configs = feature_configs or {}
        self.default_config = default_config
        self.time_col = time_col
        self.group_col = group_col
        self.fill_strategy = fill_strategy
        
        # State storage for fitted parameters (mean, std, min, max, median, iqr)
        # Structure: { feature_name: { 'global': params, 'groups': { asset_id: params } } }
        self.fitted_params: Dict[str, Dict] = {}
        self.feature_names: List[str] = []

    def fit(self, df: pl.DataFrame, feature_cols: List[str]) -> "TimeSeriesPreprocessor":
        """
        Computes statistics for scaling based on the training set.
        CRITICAL: Use only TRAINING data here to avoid look-ahead bias.
        """
        self.feature_names = feature_cols
        logger.info(f"Fitting preprocessor on {len(df)} rows, {len(feature_cols)} features.")

        for col in feature_cols:
            config = self.feature_configs.get(col, self.default_config)
            self.fitted_params[col] = {"config": config}

            if config.scope == "global":
                self._fit_global(df, col, config.method)
            elif config.scope == "group":
                self._fit_grouped(df, col, config.method)
            else:
                raise ValueError(f"Unknown scope: {config.scope}")

        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Applies the fitted transformations to the dataframe.
        """
        logger.info("Transforming data...")
        
        # 1. Handle Missing Values (Polars is very fast at this)
        if self.fill_strategy == "forward":
            df = df.with_columns([pl.col(c).forward_fill().backward_fill() for c in self.feature_names])
        elif self.fill_strategy == "zero":
            df = df.with_columns([pl.col(c).fill_null(0) for c in self.feature_names])

        # 2. Apply Scaling
        # We build a list of expressions to execute lazily if possible
        expressions = []
        
        # For grouped scaling, we might need to join params. 
        # But for performance on large DataFrames, mapping or join is better.
        # Let's iterate features.
        
        df_lazy = df.lazy()
        
        for col in self.feature_names:
            params = self.fitted_params.get(col)
            if not params:
                logger.warning(f"Feature {col} was not fitted. Skipping.")
                continue

            config = params["config"]
            
            if config.scope == "global":
                expr = self._get_transform_expr(pl.col(col), params["stats"], config.method)
                df_lazy = df_lazy.with_columns(expr.alias(col))
            
            elif config.scope == "group":
                # For grouped scaling, we need to apply logic per asset.
                # Efficient approach: Join the stats DataFrame and calculate.
                stats_df = params["stats_df"]  # This stores mean/std per asset_id
                
                # Join stats to the main frame
                # Suffix columns to avoid collision, e.g., "_mean", "_std"
                df_lazy = df_lazy.join(stats_df.lazy(), on=self.group_col, how="left")
                
                # Apply calculation
                col_expr = pl.col(col)
                if config.method == "standard":
                    mean_col = pl.col(f"{col}_mean")
                    std_col = pl.col(f"{col}_std")
                    # (x - mean) / std
                    expr = (col_expr - mean_col) / (std_col + 1e-8)
                elif config.method == "robust":
                    median_col = pl.col(f"{col}_median")
                    iqr_col = pl.col(f"{col}_iqr")
                    expr = (col_expr - median_col) / (iqr_col + 1e-8)
                elif config.method == "minmax":
                    min_col = pl.col(f"{col}_min")
                    max_col = pl.col(f"{col}_max")
                    expr = (col_expr - min_col) / (max_col - min_col + 1e-8)
                
                df_lazy = df_lazy.with_columns(expr.alias(col))
                
                # Drop stats cols to clean up
                stats_cols = stats_df.columns
                stats_cols.remove(self.group_col)
                df_lazy = df_lazy.drop(stats_cols)

        return df_lazy.collect()

    def make_windows(
        self, 
        df: pl.DataFrame, 
        events: pl.DataFrame, 
        window_size: int,
        future_window: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts 3D tensors for neural networks efficiently.
        
        Args:
            df: Transformed DataFrame (Long format: [timestamp, asset_id, features...])
            events: DataFrame containing [timestamp, asset_id] of signals.
            window_size: Lookback period (Sequence Length).
            
        Returns:
            X: (num_events, window_size, num_features)
            y: (num_events,) or None (targets)
        """
        logger.info(f"Extracting {len(events)} windows of size {window_size}...")
        
        # Ensure sorting
        df = df.sort([self.group_col, self.time_col])
        
        # Convert to numpy for fast slicing (Polars -> Numpy is zero-copy for numeric)
        # We need a mapping of (asset_id) -> (array_indices)
        
        # Strategy:
        # 1. Group df by asset_id to get contiguous memory blocks.
        # 2. Use searchsorted to find indices of events.
        # 3. Slice numpy arrays.
        
        # Check if asset_id is string or int. If string, encode it for performance or use dictionary map.
        assets = df[self.group_col].unique().to_list()
        feature_data = {}
        time_data = {}
        
        # Partition data by asset for O(1) access
        # This is memory intensive but fast. For HUGE data, use lazy slicing.
        partitioned = df.partition_by(self.group_col, as_dict=True)
        
        X_list = []
        valid_indices = []
        
        # Convert feature columns to float32 numpy matrix per asset
        asset_arrays = {}
        asset_times = {}
        
        for asset, sub_df in partitioned.items():
            # (Time, Features)
            asset_arrays[asset] = sub_df.select(self.feature_names).to_numpy().astype(np.float32)
            asset_times[asset] = sub_df.select(self.time_col).to_series().to_numpy()

        # Iterate events
        # events should have columns: [timestamp, asset_id]
        event_rows = events.to_dicts()
        
        for i, event in enumerate(event_rows):
            asset = event[self.group_col]
            ts = event[self.time_col]
            
            if asset not in asset_arrays:
                continue
                
            times = asset_times[asset]

            idx = np.searchsorted(times, ts, side='right') - 1
            
            # Check bounds
            if idx < window_size - 1:
                # Not enough history (warmup period)
                continue
                
            # Slice: [idx - window + 1 : idx + 1]
            start_idx = idx - window_size + 1
            end_idx = idx + 1
            
            window_data = asset_arrays[asset][start_idx:end_idx]
            
            if window_data.shape[0] == window_size:
                X_list.append(window_data)
                valid_indices.append(i)

        if not X_list:
            return np.array([]), np.array([])
            
        X = np.stack(X_list) # (N, L, F)
        logger.success(f"Tensor created: {X.shape}")
        
        return X, np.array(valid_indices)

    # --- Internal Helpers ---

    def _fit_global(self, df: pl.DataFrame, col: str, method: str):
        if method == "standard":
            stats = df.select([pl.col(col).mean(), pl.col(col).std()]).to_dict(as_series=False)
            self.fitted_params[col]["stats"] = {"mean": stats[col + "_mean"][0], "std": stats[col + "_std"][0]}
        elif method == "robust":
            stats = df.select([pl.col(col).median(), pl.col(col).quantile(0.75) - pl.col(col).quantile(0.25)]).to_dict(as_series=False)
            self.fitted_params[col]["stats"] = {"median": stats[col + "_median"][0], "iqr": stats[col][0]} # key might vary slightly depending on alias
        # ... implement minmax

    def _fit_grouped(self, df: pl.DataFrame, col: str, method: str):
        # Calculate stats grouped by asset_id
        if method == "standard":
            stats_df = df.group_by(self.group_col).agg([
                pl.col(col).mean().alias(f"{col}_mean"),
                pl.col(col).std().alias(f"{col}_std")
            ])
        elif method == "robust":
            stats_df = df.group_by(self.group_col).agg([
                pl.col(col).median().alias(f"{col}_median"),
                (pl.col(col).quantile(0.75) - pl.col(col).quantile(0.25)).alias(f"{col}_iqr")
            ])
        elif method == "minmax":
            stats_df = df.group_by(self.group_col).agg([
                pl.col(col).min().alias(f"{col}_min"),
                pl.col(col).max().alias(f"{col}_max")
            ])
            
        self.fitted_params[col]["stats_df"] = stats_df

    def _get_transform_expr(self, col_expr, stats, method):
        if method == "standard":
            return (col_expr - stats["mean"]) / (stats["std"] + 1e-8)
        elif method == "robust":
            return (col_expr - stats["median"]) / (stats["iqr"] + 1e-8)
        # ...
        return col_expr

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "TimeSeriesPreprocessor":
        with open(path, "rb") as f:
            return pickle.load(f)