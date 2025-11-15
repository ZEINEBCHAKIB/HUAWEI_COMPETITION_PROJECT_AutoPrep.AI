from __future__ import annotations
import re
from typing import Dict, Any, List

import numpy as np
import pandas as pd


ID_PATTERN = re.compile(r"^(id|.*_id|code|.*_code|.*_number|.*_num)$", re.IGNORECASE)


def infer_types(df: pd.DataFrame) -> Dict[str, str]:
    types: Dict[str, str] = {}
    for col in df.columns:
        s = df[col]
        
        # Check if column name matches ID pattern FIRST (case-insensitive)
        # This catches transaction_id, user_id, etc. regardless of data type
        if ID_PATTERN.match(col):
            types[col] = 'id'
            continue
        
        # Check if column has very high uniqueness (almost all unique values)
        if s.nunique(dropna=True) > 0.9 * s.shape[0]:
            types[col] = 'id'
            continue
        
        # Now check data types
        if s.dtype.kind in ('i', 'u', 'f'):
            types[col] = 'numeric'
        elif np.issubdtype(s.dtype, np.datetime64):
            types[col] = 'datetime'
        elif s.dtype == 'O' or s.dtype.name == 'category':
            # try parse datetime
            sample = s.dropna().head(50).astype(str)
            dt_count = 0
            for val in sample:
                try:
                    pd.to_datetime(val)
                    dt_count += 1
                except Exception:
                    pass
            if len(sample) > 0 and dt_count / len(sample) > 0.7:
                types[col] = 'datetime'
            else:
                types[col] = 'categorical'
        else:
            types[col] = 'categorical'
    return types


def column_stats(df: pd.DataFrame, types: Dict[str, str]) -> List[Dict[str, Any]]:
    stats: List[Dict[str, Any]] = []
    for col in df.columns:
        t = types.get(col, 'categorical')
        s = df[col]
        miss = float(s.isna().mean())
        card = int(s.nunique(dropna=True))
        entry: Dict[str, Any] = {
            'name': col,
            'type': t,
            'missing_rate': miss,
            'cardinality': card,
        }
        if t == 'numeric':
            vals = pd.to_numeric(s, errors='coerce')
            entry.update({
                'mean': float(vals.mean(skipna=True)) if vals.notna().any() else None,
                'median': float(vals.median(skipna=True)) if vals.notna().any() else None,
                'std': float(vals.std(skipna=True)) if vals.notna().any() else None,
                'skew': float(vals.skew(skipna=True)) if vals.notna().any() else 0.0,
                'min': float(vals.min(skipna=True)) if vals.notna().any() else None,
                'max': float(vals.max(skipna=True)) if vals.notna().any() else None,
            })
        stats.append(entry)
    return stats


def dataset_wide_stats(df: pd.DataFrame, types: Dict[str, str]) -> Dict[str, Any]:
    res = {
        'n_rows': int(df.shape[0]),
        'n_cols': int(df.shape[1]),
        'missing_overall': float(df.isna().mean().mean()) if df.size else 0.0,
        'type_counts': {typ: sum(1 for t in types.values() if t == typ) for typ in set(types.values())},
    }
    return res
