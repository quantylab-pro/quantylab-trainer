"""Load and split stock single-swing datasets."""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd


def _resolve_dataset_dir(base_path: str, dataset: str) -> str:
    if os.path.isabs(dataset):
        return dataset
    return os.path.join(base_path, "data", dataset)


def load_stock_sequences(
    base_path: str,
    dataset: str,
    start_date: str | None = None,
    end_date: str | None = None,
    min_rows: int = 120,
) -> list[tuple[str, pd.DataFrame, np.ndarray]]:
    """Load aligned environment/features and return one sequence per stock."""

    dataset_dir = _resolve_dataset_dir(base_path, dataset)
    env_path = os.path.join(dataset_dir, "environment.csv")
    feature_path = os.path.join(dataset_dir, "training_scaled.csv")
    code_path = os.path.join(dataset_dir, "stock_codes.csv")
    for path in (env_path, feature_path, code_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"stock 데이터 파일이 없습니다: {path}")

    env = pd.read_csv(env_path)
    features = pd.read_csv(feature_path).to_numpy(dtype=np.float32)
    codes = pd.read_csv(code_path)["stock_code"].astype(str).str.zfill(6)
    if not (len(env) == len(features) == len(codes)):
        raise ValueError(
            f"stock 데이터 정렬 불일치: env={len(env)}, features={len(features)}, codes={len(codes)}"
        )

    env = env.copy()
    env["stock_code"] = codes.to_numpy()
    env["date"] = env["date"].astype(str).str.zfill(8)
    if start_date is not None:
        env_mask = env["date"] >= str(start_date).zfill(8)
    else:
        env_mask = np.ones(len(env), dtype=bool)
    if end_date is not None:
        env_mask &= env["date"] <= str(end_date).zfill(8)

    selected = env.loc[env_mask].copy().reset_index(drop=True)
    selected_features = features[env_mask]
    sequences = []
    for code, group in selected.groupby("stock_code", sort=True):
        indices = group.index.to_numpy()
        if len(indices) < min_rows:
            continue
        # The builder writes stock-contiguous rows. Sorting here makes the
        # loader robust to manually assembled datasets as well.
        order = np.argsort(group["date"].to_numpy())
        group = group.iloc[order].reset_index(drop=True)
        group_features = selected_features[indices][order]
        sequences.append((str(code).zfill(6), group, group_features))

    if not sequences:
        raise ValueError("요청한 기간에 min_rows를 충족하는 stock sequence가 없습니다.")
    return sequences
