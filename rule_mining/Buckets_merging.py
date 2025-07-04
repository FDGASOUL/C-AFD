import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from scipy.sparse import coo_matrix
from rule_mining.ColumnLayoutRelationData import ColumnLayoutRelationData

logger = logging.getLogger(__name__)

class HashCompressPhi2Analyzer:
    def __init__(self, layout_data: ColumnLayoutRelationData, target_index: int, target_num_clusters: int):
        self.layout_data = layout_data
        self.target_index = target_index
        self.target_num_clusters = target_num_clusters
        self.column_vectors = [np.array(col, dtype=int) for col in layout_data.get_column_vectors()]
        self.null_value_id = layout_data.get_null_value_id()
        self.n_cols = len(self.column_vectors)
        self.original_target_vec = self.column_vectors[target_index].copy()

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
        chi2 = sum((o * o) / (rs[i] * cs[j] / T)
                   for i, j, o in zip(sparse.row, sparse.col, sparse.data)) - T
        phi2 = chi2 / (T * (d - 1))
        return phi2

    def build_contingency_table(self, col_idx: int, target_vec: np.ndarray) -> Tuple[np.ndarray, List[Any]]:
        lhs_vec = self.column_vectors[col_idx]
        mask = (lhs_vec != self.null_value_id) & (target_vec != self.null_value_id)
        lhs_vals = lhs_vec[mask]
        rhs_vals = target_vec[mask]
        unique_rhs = sorted(set(rhs_vals.tolist()))
        col_map = {val: i for i, val in enumerate(unique_rhs)}
        lhs_unique, lhs_inv = np.unique(lhs_vals, return_inverse=True)
        rhs_mapped = np.array([col_map[val] for val in rhs_vals])
        data = np.ones(len(lhs_vals), dtype=int)
        mat = coo_matrix((data, (lhs_inv, rhs_mapped)),
                         shape=(len(lhs_unique), len(unique_rhs)))
        return mat.toarray(), unique_rhs

    def compress_with_hashing(self, vec: np.ndarray) -> np.ndarray:
        """
        将原始列的值映射为哈希后的整数，范围为 [0, target_num_clusters-1]
        空值保持不变。
        """
        hashed = np.array([
            v if v == self.null_value_id else hash(v) % self.target_num_clusters
            for v in vec
        ])
        return hashed

    def analyze_compression_effect(self) -> None:
        print(f"目标列（索引 {self.target_index}）原始类数为：{len(np.unique(self.original_target_vec))}")

        phi2_before = {}
        phi2_after = {}

        compressed_vec = self.compress_with_hashing(self.original_target_vec)
        print(f"哈希压缩后目标列类数为：{len(np.unique(compressed_vec))}")

        for i in range(self.n_cols):
            if i == self.target_index:
                continue

            # 原始 φ²
            table_before, _ = self.build_contingency_table(i, self.original_target_vec)
            phi_before = self.compute_phi2(table_before)
            phi2_before[i] = phi_before

            # 压缩后 φ²
            table_after, _ = self.build_contingency_table(i, compressed_vec)
            phi_after = self.compute_phi2(table_after)
            phi2_after[i] = phi_after

            delta = None if phi_before is None or phi_after is None else phi_after - phi_before
            print(f"列 {i} 与目标列 φ²：原始={phi_before}，压缩后={phi_after}，变化量={delta}")
