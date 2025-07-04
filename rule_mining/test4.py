import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from itertools import combinations
from typing import Optional


class Phi2Calculator:
    def compute_phi2(self, table: np.ndarray) -> Optional[float]:
        arr = np.asarray(table, dtype=float)
        if arr.size == 0 or arr.sum() == 0:
            return None
        R, C = arr.shape
        d = min(R, C)
        if d <= 1:
            return 0.0
        T = arr.sum()
        rs = arr.sum(axis=1)
        cs = arr.sum(axis=0)
        sparse = coo_matrix(arr)
        chi2 = sum((o * o) / (rs[i] * cs[j] / T) for i, j, o in zip(sparse.row, sparse.col, sparse.data)) - T
        phi2 = chi2 / (T * (d - 1))
        return phi2

    def compute_expected_phi2(self, table: np.ndarray) -> Optional[float]:
        arr = np.asarray(table, dtype=float)
        if arr.size == 0 or arr.sum() == 0:
            return None
        R, C = arr.shape
        d = min(R, C)
        T = arr.sum()
        if d <= 1 or T <= 1:
            return None
        expected_phi = ((C - 1) * (R - 1)) / ((T - 1) * (d - 1))
        return expected_phi

    def normalize_phi(self, phi2: Optional[float], expected_phi: Optional[float]) -> float:
        if phi2 is None or expected_phi in (None, 1.0):
            return 0.0
        return max((phi2 - expected_phi) / (1 - expected_phi), 0.0)


def analyze_csv(filepath: str):
    df = pd.read_csv(filepath, dtype=str)  # 以字符串形式读取所有列
    calc = Phi2Calculator()

    for col1, col2 in combinations(df.columns, 2):
        contingency = pd.crosstab(df[col1], df[col2])
        table = contingency.to_numpy()

        phi2 = calc.compute_phi2(table)
        expected_phi = calc.compute_expected_phi2(table)
        # expected_phi = 0.333333
        normalized = calc.normalize_phi(phi2, expected_phi)

        print(f"==== {col1} vs {col2} ====")
        print(f"φ²: {phi2:.6f}" if phi2 is not None else "φ²: None")
        print(f"Expected φ²: {expected_phi:.6f}" if expected_phi is not None else "Expected φ²: None")
        print(f"Normalized φ²: {normalized:.6f}")
        print()


# ======= 用法示例 =======
# 替换为你自己的 CSV 文件路径
analyze_csv("data_test.csv")
