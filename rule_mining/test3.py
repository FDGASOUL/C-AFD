import logging
import numpy as np
from typing import List, Tuple, Set, Optional, Dict, Any
from scipy.sparse import coo_matrix
from rule_mining.Cluster_merging import Incorporate
from rule_mining.ColumnLayoutRelationData import ColumnLayoutRelationData

logger = logging.getLogger(__name__)

class Phi2Merger:
    def __init__(self, layout_data: ColumnLayoutRelationData, target_index: int):
        self.layout_data = layout_data
        self.target_index = target_index
        self.column_vectors = [np.array(col, dtype=int) for col in layout_data.get_column_vectors()]
        self.column_data = layout_data.get_column_data()
        self.null_value_id = layout_data.get_null_value_id()
        self.n_cols = len(self.column_vectors)
        self.target_vec = self.column_vectors[target_index].copy()

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

    def merge_classes(self, target_vec: np.ndarray, merge_pairs: Set[Tuple[Any, Any]]) -> np.ndarray:
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            parent[find(x)] = find(y)

        for a, b in merge_pairs:
            union(a, b)

        return np.array([find(v) if v != self.null_value_id else v for v in target_vec])

    def analyze_and_merge(self) -> int:
        print(f"目标列（索引 {self.target_index}）初始类数为：{len(np.unique(self.target_vec))}")

        phi2_before = {}
        phi2_after = {}
        intersect_pairs: Optional[Set[Tuple[Any, Any]]] = None
        early_stop = False

        for i in range(self.n_cols):
            if i == self.target_index:
                continue  # 跳过目标列自身

            table, unique_vals = self.build_contingency_table(i, self.target_vec)
            phi2 = self.compute_phi2(table)
            phi2_before[i] = phi2
            print(f"[初始] 列 {i} 与目标列的 φ² = {phi2}")

            pairs = set(Incorporate().get_mergeable_column_pairs(table))
            if intersect_pairs is None:
                intersect_pairs = pairs
            else:
                intersect_pairs &= pairs

            if not intersect_pairs:
                print(f"\n在处理列 {i} 时交集为空，提前终止合并与后续 φ² 分析。")
                early_stop = True
                break

        if early_stop or not intersect_pairs:
            print("\n未找到共同可合并对，跳过合并与后续 φ² 分析。")
            return

        print(f"\n所有列联表共有的可合并对（交集）为：{intersect_pairs}")

        # 合并值
        new_target_vec = self.merge_classes(self.target_vec, intersect_pairs)
        # 合并后的类数
        new_unique_vals = np.unique(new_target_vec)
        print(f"\n合并后的目标列（索引 {self.target_index}）类数为：{len(new_unique_vals)}")
        # 压缩了多少
        print(f"合并后目标列的压缩数为：{len(np.unique(self.target_vec)) - len(new_unique_vals)}")

        for i in range(self.n_cols):
            if i == self.target_index:
                continue
            table, _ = self.build_contingency_table(i, new_target_vec)
            phi2 = self.compute_phi2(table)
            phi2_after[i] = phi2
            before = phi2_before[i]
            delta = None if before is None or phi2 is None else phi2 - before
            print(f"[合并后] 列 {i} 与目标列的 φ² = {phi2}，变化量 = {delta}")

        return len(new_unique_vals)
