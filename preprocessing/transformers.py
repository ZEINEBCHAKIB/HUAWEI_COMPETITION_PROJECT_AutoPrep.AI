from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freqs_: dict | None = None

    def fit(self, X: pd.Series, y=None):
        s = pd.Series(X)
        counts = s.value_counts(dropna=False)
        total = counts.sum()
        self.freqs = (counts / total).to_dict()
        return self

    def transform(self, X: pd.Series):
        s = pd.Series(X)
        return s.map(self.freqs).fillna(0.0).astype(float)


class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing: float = 10.0):
        self.smoothing = smoothing
        self.global_mean_: float | None = None
        self.mapping_: dict | None = None

    def fit(self, X: pd.Series, y: pd.Series):
        df = pd.DataFrame({'x': X, 'y': y})
        self.global_mean_ = float(df['y'].mean())
        stats = df.groupby('x')['y'].agg(['mean', 'count'])
        smoothing = 1 / (1 + np.exp(-(stats['count'] - self.smoothing)))
        self.mapping_ = (self.global_mean_ * (1 - smoothing) + stats['mean'] * smoothing).to_dict()
        return self

    def transform(self, X: pd.Series):
        return pd.Series(X).map(self.mapping_).fillna(self.global_mean_).astype(float)


class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, method: str = 'iqr', z_thresh: float = 3.0):
        self.method = method
        self.z_thresh = z_thresh
        self.params_: dict | None = None

    def fit(self, X: pd.Series, y=None):
        s = pd.Series(pd.to_numeric(X, errors='coerce'))
        if self.method == 'zscore':
            mu = s.mean()
            sd = s.std()
            self.params_ = {'mu': float(mu), 'sd': float(sd)}
        else:
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            self.params_ = {'low': float(q1 - 1.5 * iqr), 'high': float(q3 + 1.5 * iqr)}
        return self

    def transform(self, X: pd.Series):
        s = pd.Series(pd.to_numeric(X, errors='coerce'))
        if self.method == 'zscore':
            mu = self.params_['mu']
            sd = self.params_['sd'] or 1.0
            low = mu - self.z_thresh * sd
            high = mu + self.z_thresh * sd
            return s.clip(lower=low, upper=high)
        else:
            return s.clip(lower=self.params_['low'], upper=self.params_['high'])
