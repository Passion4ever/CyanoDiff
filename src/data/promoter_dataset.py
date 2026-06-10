"""Promoter fine-tuning datasets and expression discretizers."""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset

from src.data.genome_dataset import encode_sequence


LOG_EXPR_MAX = 6.028  # 99.9th percentile of log10(reads+1) on the 7120 training set


def normalize_expression(reads):
    return np.log10(reads + 1) / LOG_EXPR_MAX


def denormalize_expression(normalized):
    return 10 ** (normalized * LOG_EXPR_MAX) - 1


class ExpressionBinner:
    """Bin continuous log-expression into equal-frequency quantile classes."""

    def __init__(self, n_bins=3):
        self.n_bins = n_bins
        self.bin_edges = None
        self.bin_labels = None

    def fit(self, log_expressions):
        quantiles = np.linspace(0, 100, self.n_bins + 1)
        self.bin_edges = np.percentile(log_expressions, quantiles)
        # Open the outer edges so transform() never drops a test-set outlier.
        self.bin_edges[0] = -np.inf
        self.bin_edges[-1] = np.inf
        self.bin_labels = self._default_labels(self.n_bins)

    def transform(self, log_expressions):
        # np.digitize returns 1-based bin index; the [1:-1] slice drops the
        # sentinel ±inf edges so the result lives in [0, n_bins-1].
        bin_ids = np.digitize(log_expressions, self.bin_edges[1:-1])
        return np.clip(bin_ids, 0, self.n_bins - 1).astype(np.int64)

    def get_bin_stats(self, log_expressions):
        bin_ids = self.transform(log_expressions)
        stats = {}
        for i in range(self.n_bins):
            mask = bin_ids == i
            vals = log_expressions[mask]
            reads = 10 ** vals - 1
            stats[self.bin_labels[i]] = {
                'count': int(mask.sum()),
                'reads_range': f'{reads.min():.0f}-{reads.max():.0f}',
                'log_expr_range': f'{vals.min():.2f}-{vals.max():.2f}',
            }
        return stats

    def save(self, path):
        np.save(path, self.bin_edges)

    def load(self, path):
        self.bin_edges = np.load(path)
        self.n_bins = len(self.bin_edges) - 1
        self.bin_labels = self._default_labels(self.n_bins)
        return self

    @staticmethod
    def _default_labels(n):
        if n == 3:
            return ['low', 'mid', 'high']
        if n == 5:
            return ['very_low', 'low', 'mid', 'high', 'very_high']
        return [f'bin{i}' for i in range(n)]


class QuantileNormalizer:
    """Rank-based [0, 1] mapping. Kept for legacy configs that pass continuous cond."""

    def __init__(self):
        self.sorted_values = None

    def fit(self, log_expressions):
        self.sorted_values = np.sort(log_expressions)

    def transform(self, log_expressions):
        ranks = np.searchsorted(self.sorted_values, log_expressions, side='right')
        return (ranks / len(self.sorted_values)).astype(np.float32)

    def inverse_transform(self, quantiles):
        quantiles = np.asarray(quantiles)
        indices = np.clip(
            (quantiles * len(self.sorted_values)).astype(int),
            0, len(self.sorted_values) - 1,
        )
        return 10 ** self.sorted_values[indices] - 1

    def save(self, path):
        np.save(path, self.sorted_values)

    def load(self, path):
        self.sorted_values = np.load(path)
        return self


class PromoterDataset(Dataset):
    """Promoter sequences paired with an expression condition.

    Pass `binner` for discrete class ids (LongTensor) or `normalizer` for
    continuous quantile scalars. With neither, falls back to log/LOG_EXPR_MAX
    normalization (the v1 scheme).
    """

    def __init__(self, csv_path, binner=None, normalizer=None):
        self.df = pd.read_csv(csv_path)
        self.discrete = binner is not None

        self.sequences = torch.stack([
            encode_sequence(row['sequence']) for _, row in self.df.iterrows()
        ])  # [N, 81]

        log_expr = self.df['log_expression'].values
        if binner is not None:
            self.expressions = torch.tensor(binner.transform(log_expr), dtype=torch.long)
        elif normalizer is not None:
            self.expressions = torch.tensor(normalizer.transform(log_expr), dtype=torch.float32)
        else:
            reads = self.df['reads_wt0'].values
            self.expressions = torch.tensor(normalize_expression(reads), dtype=torch.float32)

        cond_type = 'discrete' if self.discrete else 'continuous'
        print(f"loaded {csv_path}: n={len(self)}, cond={cond_type}, "
              f"range=[{self.expressions.min():.3f}, {self.expressions.max():.3f}]")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.expressions[idx]
