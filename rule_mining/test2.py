import numpy as np
from scipy.sparse import coo_matrix
from typing import Optional


class Phi2Calculator:
    def compute_phi2(self, table: np.ndarray) -> Optional[float]:
        """
        计算 φ² 统计量。

        :param table: 列联表
        :return: phi² 值；无效时返回 None
        """
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
        """
        计算期望 φ²。

        :param table: 列联表
        :return: 期望 φ² 值；无效时返回 None
        """
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
        """
        归一化 φ² 到 [0,1]。

        :param phi2: 原始 φ²
        :param expected_phi: 期望 φ²
        :return: 归一化值，最低为 0
        """
        if phi2 is None or expected_phi in (None, 1.0):
            return 0.0
        return max((phi2 - expected_phi) / (1 - expected_phi), 0.0)


# ===== 示例使用 =====
table = [
    [1, 4,2,0],
    [0, 1,0,2]
]

calculator = Phi2Calculator()
phi2 = calculator.compute_phi2(table)
expected_phi = calculator.compute_expected_phi2(table)
normalized_phi = calculator.normalize_phi(phi2, expected_phi)

print(f"φ²: {phi2:.6f}")
print(f"Expected φ²: {expected_phi:.6f}")
print(f"Normalized φ²: {normalized_phi:.6f}")
