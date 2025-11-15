from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

from .stats import infer_types, column_stats, dataset_wide_stats
from .transformers import FrequencyEncoder, TargetMeanEncoder, OutlierCapper
from .llm_advisor import advise


@dataclass
class PreprocessResult:
    processed_df: pd.DataFrame
    decisions: Dict[str, Any]
    before_stats: Dict[str, Any]
    after_stats: Dict[str, Any]


class AutoPreprocessor:
    def __init__(
        self,
        target_column: Optional[str] = None,
        low_card_threshold: int = 10,
        high_missing_threshold: float = 0.3,
        apply_outlier_treatment: bool = True,
        scaling_enabled: bool = True,
        scaling_method: Optional[str] = None,
        llm_model: Optional[str] = None,
    ):
        self.target_column = target_column
        self.low_card_threshold = low_card_threshold
        self.high_missing_threshold = high_missing_threshold
        self.apply_outlier_treatment = apply_outlier_treatment
        self.scaling_enabled = scaling_enabled
        self.scaling_method = scaling_method  # None='auto', 'standard', or 'minmax'
        self.llm_model = llm_model

    def _choose_scaling_method(self, series: pd.Series) -> str:
        """
        Auto-detect scaling method based on skewness.
        - If |skewness| > 0.5: use MinMaxScaler (better for skewed data)
        - Otherwise: use StandardScaler (better for normal distributions)
        """
        try:
            # Remove NaN values for skewness calculation
            clean_series = pd.to_numeric(series, errors='coerce').dropna()
            if len(clean_series) < 2:
                return 'standard'
            
            skewness = scipy_stats.skew(clean_series)
            # If data is significantly skewed, use MinMax; otherwise StandardScaler
            if abs(skewness) > 0.5:
                return 'minmax'
            else:
                return 'standard'
        except Exception:
            return 'standard'

    def fit_transform(self, df: pd.DataFrame) -> PreprocessResult:
        df = df.copy()
        if self.target_column and self.target_column not in df.columns:
            self.target_column = None

        types = infer_types(df)

        before = {
            'dataset': dataset_wide_stats(df, types),
            'columns': column_stats(df, types),
        }

        # Drop rules
        drops: List[str] = []
        for col in df.columns:
            s = df[col]
            t = types[col]
            if t == 'id':
                drops.append(col)
                continue
            if s.nunique(dropna=True) <= 1:
                drops.append(col)
                continue
            if s.isna().mean() > self.high_missing_threshold:
                drops.append(col)
                continue
        df = df.drop(columns=drops) if drops else df
        for d in drops:
            types.pop(d, None)

        # Prepare decisions log
        decisions: Dict[str, Any] = {
            'global': {
                'low_card_threshold': self.low_card_threshold,
                'high_missing_threshold': self.high_missing_threshold,
                'outliers': self.apply_outlier_treatment,
                'scaling': self.scaling_enabled,
                'scaling_method': self.scaling_method or 'auto-detect',
            },
            'drops': drops,
            'features': {},
        }

        # Target
        y = None
        if self.target_column:
            y = df[self.target_column]

        # Per-feature decisions and transforms
        processed = pd.DataFrame(index=df.index)
        numeric_scaler = None
        if self.scaling_enabled:
            numeric_scaler = {'standard': StandardScaler(), 'minmax': MinMaxScaler()}
        
        # Track if datetime features already created (to avoid duplicates)
        datetime_features_created = False

        for col in df.columns:
            if col == self.target_column:
                continue
            t = types[col]
            s = df[col]
            stats_entry = next((c for c in before['columns'] if c['name'] == col), {})
            decision = advise(stats_entry, model=self.llm_model)
            decisions['features'][col] = decision

            # Imputation
            if decision['imputation'] == 'drop':
                continue
            if t == 'numeric':
                if decision['imputation'] == 'median':
                    imp = SimpleImputer(strategy='median')
                elif decision['imputation'] == 'mean':
                    imp = SimpleImputer(strategy='mean')
                elif decision['imputation'] == 'knn':
                    imp = KNNImputer(n_neighbors=5)
                else:
                    imp = SimpleImputer(strategy='median')
                col_values = pd.to_numeric(s, errors='coerce').values.reshape(-1, 1)
                s_imp = imp.fit_transform(col_values).ravel()
                s_proc = pd.Series(s_imp, index=df.index)

                # Outliers
                if self.apply_outlier_treatment:
                    capper = OutlierCapper(method='iqr')
                    s_proc = capper.fit_transform(s_proc)

                # Scaling
                if self.scaling_enabled:
                    # Determine scaling method
                    if self.scaling_method is None:
                        # Auto-detect based on skewness
                        scaler_key = self._choose_scaling_method(df[col])
                    else:
                        # Use user-selected method
                        scaler_key = self.scaling_method
                    
                    scaler = numeric_scaler.get(scaler_key, StandardScaler())
                    s_proc = scaler.fit_transform(s_proc.values.reshape(-1, 1)).ravel()
                processed[col] = s_proc

            elif t == 'categorical':
                imp = SimpleImputer(strategy='most_frequent')
                s_imp = imp.fit_transform(s.values.reshape(-1, 1)).ravel()
                enc = decision.get('encoding') or ('onehot' if s.nunique() <= self.low_card_threshold else 'frequency')
                if enc == 'onehot':
                    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                    arr = ohe.fit_transform(np.array(s_imp, dtype=object).reshape(-1,1))
                    ohe_cols = [f"{col}__{c}" for c in ohe.categories_[0]]
                    processed[ohe_cols] = arr
                elif enc == 'frequency':
                    fe = FrequencyEncoder()
                    processed[col] = fe.fit_transform(pd.Series(s_imp))
                elif enc == 'target' and y is not None and y.dtype.kind in ('i','u','f'):
                    tme = TargetMeanEncoder()
                    processed[col] = tme.fit_transform(pd.Series(s_imp), pd.to_numeric(y, errors='coerce'))
                else:
                    # fallback: frequency
                    fe = FrequencyEncoder()
                    processed[col] = fe.fit_transform(pd.Series(s_imp))

            elif t == 'datetime':
                # Only create datetime features once (from first datetime column)
                if not datetime_features_created:
                    # Parse datetime and extract features
                    s_parsed = pd.to_datetime(s, errors='coerce')
                    
                    # Impute missing values with mode or median date
                    if s_parsed.isna().any():
                        mode_date = s_parsed.mode()
                        fill_date = mode_date.iloc[0] if len(mode_date) > 0 else pd.Timestamp.now()
                        s_parsed = s_parsed.fillna(fill_date)
                    
                    # Extract temporal features
                    year_vals = s_parsed.dt.year
                    month_vals = s_parsed.dt.month
                    day_vals = s_parsed.dt.day
                    day_of_week_vals = s_parsed.dt.dayofweek  # 0=Monday, 6=Sunday
                    is_weekend_vals = (s_parsed.dt.dayofweek >= 5).astype(int)
                    
                    # Apply scaling to numeric temporal features if enabled
                    if self.scaling_enabled:
                        # Determine scaling method for temporal features
                        if self.scaling_method is None:
                            # Auto-detect based on skewness of year (representative feature)
                            scaler_key = self._choose_scaling_method(pd.Series(year_vals))
                        else:
                            scaler_key = self.scaling_method
                        
                        scaler = numeric_scaler.get(scaler_key, StandardScaler())
                        
                        # Scale all numeric temporal features
                        year_vals = scaler.fit_transform(year_vals.values.reshape(-1, 1)).ravel()
                        month_vals = scaler.fit_transform(month_vals.values.reshape(-1, 1)).ravel()
                        day_vals = scaler.fit_transform(day_vals.values.reshape(-1, 1)).ravel()
                        day_of_week_vals = scaler.fit_transform(day_of_week_vals.values.reshape(-1, 1)).ravel()
                        # is_weekend is binary, no need to scale
                    
                    processed["year"] = year_vals
                    processed["month"] = month_vals
                    processed["day"] = day_vals
                    processed["day_of_week"] = day_of_week_vals
                    processed["is_weekend"] = is_weekend_vals
                    
                    # Part of day based on hour (if time available)
                    if s_parsed.dt.hour.notna().any():
                        hour = s_parsed.dt.hour
                        # Define part_day: Morning: 6-12, Afternoon: 12-18, Evening: 18-24
                        # Night (0-6) is also considered evening for simplicity
                        def categorize_hour(h):
                            if 6 <= h < 12:
                                return 'morning'
                            elif 12 <= h < 18:
                                return 'afternoon'
                            else:  # 18-24 or 0-6
                                return 'evening'
                        part_day_cat = hour.apply(categorize_hour)
                        
                        # One-hot encode part_day with all categories (force all 3 columns)
                        part_day_series = pd.Categorical(part_day_cat, categories=['morning', 'afternoon', 'evening'])
                        part_day_dummies = pd.get_dummies(part_day_series, prefix="part_day")
                        for col_name in part_day_dummies.columns:
                            processed[col_name] = part_day_dummies[col_name].astype(int)
                    # If no time info, no part_day features created
                    
                    # Mark datetime features as created
                    datetime_features_created = True
                # Skip other datetime columns
            else:
                # leave
                processed[col] = df[col]

        after_types = infer_types(processed)
        after = {
            'dataset': dataset_wide_stats(processed, after_types),
            'columns': column_stats(processed, after_types),
        }

        return PreprocessResult(
            processed_df=processed,
            decisions=decisions,
            before_stats=before,
            after_stats=after,
        )
