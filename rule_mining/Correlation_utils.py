import logging
from math import sqrt
from typing import List, Tuple, Optional, Any

import numpy as np
from scipy.sparse import coo_matrix

from rule_mining.Cluster_old import Incorporate

# 获取日志实例
logger = logging.getLogger(__name__)


class CorrelationCalculator:
    """
    相关性计算工具类。

    提供列联表构建、期望频数计算、脏格检查及基于 φ² 的相关性评估方法。

    类属性：
        upper_threshold (float): 相关性上限阈值。
        lower_threshold (float): 相关性下限阈值。
    """
    upper_threshold: float = 0.61
    lower_threshold: float = 0.01

    def __init__(self, column_layout_data: Any) -> None:
        """
        初始化 CorrelationCalculator。

        :param column_layout_data: ColumnLayoutRelationData 实例
        """
        self.column_vectors: List[List[int]] = column_layout_data.get_column_vectors()
        self.column_data: List[Any] = column_layout_data.get_column_data()
        self.schema: List[str] = column_layout_data.get_schema()
        self.null_value_id: int = column_layout_data.get_null_value_id()

    def _get_column_name(self, index: int) -> str:
        """
        获取列名。

        :param index: 列索引（0-based）
        :return: 列名
        """
        return self.schema[index]

    def build_linked_table(
            self,
            lhs: List[int],
            rhs: int
    ) -> np.ndarray:
        """
        构建列联表。

        步骤：
        1. 获取 LHS 的 PLI 列表（支持多属性交叉）。
        2. 提取 RHS 列向量并映射非空值至列索引。
        3. 使用 COO 稀疏矩阵聚合计数。
        4. 转为密集矩阵并过滤全零簇。

        :param lhs: LHS 属性索引列表（0-based）
        :param rhs: RHS 属性索引（0-based）
        :return: NumPy 数组，形状 (n_clusters, n_unique_rhs_values)
        """
        # 获取 PLI 列表
        if len(lhs) == 1:
            plis = self.column_data[lhs[0]]["PLI"]
        else:
            plis = self._cross_plis([self.column_data[i]["PLI"] for i in lhs])

        # 提取 RHS 向量并映射值
        rhs_vec = self.column_vectors[rhs]
        unique_vals = sorted({rhs_vec[pos] for cluster in plis for pos in cluster
                              if rhs_vec[pos] != self.null_value_id})
        val2col = {val: idx for idx, val in enumerate(unique_vals)}

        # 构造 COO 数据
        rows, cols = [], []
        for cid, cluster in enumerate(plis):
            for pos in cluster:
                val = rhs_vec[pos]
                if val != self.null_value_id:
                    rows.append(cid)
                    cols.append(val2col[val])
        data = np.ones(len(rows), dtype=int)

        mat = coo_matrix((data, (rows, cols)), shape=(len(plis), len(unique_vals)))
        dense = mat.toarray()
        mask = dense.sum(axis=1) > 0
        return dense[mask]

    def compute_expected_frequencies(
            self,
            table: Any
    ) -> np.ndarray:
        """
        计算期望频数。

        E[i,j] = row_sum[i] * col_sum[j] / total_sum

        :param table: 实际频数表
        :return: 期望频数矩阵 (float32)
        """
        arr = np.asarray(table, dtype=np.float32)
        rt = arr.sum(axis=1, keepdims=True)
        ct = arr.sum(axis=0, keepdims=True)
        total = rt.sum() or 1.0
        return (rt * ct) / total

    def _check_expected_frequencies(self, expected: np.ndarray) -> bool:
        """
        检查期望频数表中 >=5 的格子比例是否 >=80%。

        :param expected: 期望频数表
        :return: True 表示大部分格子满足 >=5 条件
        """
        return np.mean(expected > 5) >= 0.8

    def _check_linked_table(
            self,
            linked: np.ndarray,
            lhs: List[int]
    ) -> str:
        """
        检查列联表是否疑似 FD 或需要归并。

        :param linked: 列联表数组
        :param lhs: LHS 属性索引列表
        :return: "suspected_fd", "suspected_fd_more" 或 "incorporated"
        """
        R, C = linked.shape
        # 行抽样检查
        rows = linked if R <= 10 else linked[np.random.choice(R, 10, replace=False), :]
        rs = rows.sum(axis=1)
        rr = rows.max(axis=1) / np.where(rs == 0, 1, rs)
        if np.any(rr >= 0.9):
            return "suspected_fd"
        # 列抽样检查
        cols = linked if C <= 10 else linked[:, np.random.choice(C, 10, replace=False)]
        cs = cols.sum(axis=0)
        cr = cols.max(axis=0) / np.where(cs == 0, 1, cs)
        if np.any(cr >= 0.9):
            return "suspected_fd"
        return "incorporated"

    def _cross_plis(
            self,
            pli_list: List[List[List[int]]]
    ) -> List[List[int]]:
        """
        交叉多个 PLI，合并共同位置，保留所有簇。

        :param pli_list: 多属性 PLI 列表
        :return: 交叉后的 PLI 列表
        """
        from collections import defaultdict
        pli_maps = [{pos: idx for idx, cluster in enumerate(pli) for pos in cluster}
                    for pli in pli_list]
        merged = defaultdict(set)
        for pos in pli_maps[0]:
            key = tuple(pm.get(pos, -1) for pm in pli_maps)
            merged[key].add(pos)
        return [sorted(cluster) for cluster in merged.values()]

    def compute_phi_stats(
            self,
            table: np.ndarray
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        计算 φ² 统计量和期望 φ²。

        :param table: 列联表
        :return: (phi2, expected_phi)；无效返回 (None, None)
        """
        arr = np.asarray(table, dtype=float)
        if arr.size == 0 or arr.sum() == 0:
            return None, None
        R, C = arr.shape
        d = min(R, C)
        if d <= 1:
            return 0.0, None
        T = arr.sum()
        rs = arr.sum(axis=1)
        cs = arr.sum(axis=0)
        sparse = coo_matrix(arr)
        chi2 = sum((o * o) / (rs[i] * cs[j] / T) for i, j, o in zip(sparse.row, sparse.col, sparse.data)) - T
        phi2 = chi2 / (T * (d - 1))
        expected_phi = ((C - 1) * (R - 1)) / ((T - 1) * (d - 1)) if T > 1 else None
        return phi2, expected_phi

    def normalize_phi(
            self,
            phi2: Optional[float],
            expected_phi: Optional[float]
    ) -> float:
        """
        归一化 φ² 到 [0,1]。

        :param phi2: 原始 φ²
        :param expected_phi: 期望 φ²
        :return: 归一化值，最低为 0
        """
        if phi2 is None or expected_phi in (None, 1.0):
            return 0.0
        return max((phi2 - expected_phi) / (1 - expected_phi), 0.0)

    def check_dependency_direction(
            self,
            linked: np.ndarray
    ) -> str:
        """
        基于加权平均比较行/列方向，判断依赖方向。

        :param linked: 列联表
        :return: "mutual" / "left_to_right" / "right_to_left" / "invalid"
        """
        arr = np.asarray(linked, dtype=float)
        total = arr.sum()
        if total == 0:
            logger.warning("列联表总数为零")
            return "invalid"
        rs = arr.sum(axis=1)
        rm = arr.max(axis=1)
        row_score = rm[rs > 0].sum() / total
        cs = arr.sum(axis=0)
        cm = arr.max(axis=0)
        col_score = cm[cs > 0].sum() / total
        logger.info(f"行方向得分: {row_score:.4f}, 列方向得分: {col_score:.4f}")
        avg = (row_score + col_score) / 2
        if abs(row_score - col_score) < 0.003 * avg:
            return "mutual"
        return "left_to_right" if row_score > col_score else "right_to_left"

    def determine_correlation_result(
            self,
            phi2_norm: float,
            lhs: List[int],
            linked: np.ndarray
    ) -> Any:
        """
        根据归一化 φ² 和 LHS 列数，决定最终相关性结果。

        :param phi2_norm: 归一化 φ²
        :param lhs: LHS 属性索引列表
        :param linked: 列联表
        :return: 布尔或字符串结果
        """
        if len(lhs) == 1:
            if phi2_norm >= self.upper_threshold:
                return self.check_dependency_direction(linked)
            if phi2_norm < self.lower_threshold:
                return "invalid"
            return "pending"
        else:
            if phi2_norm >= self.upper_threshold:
                return True
            if phi2_norm < self.lower_threshold:
                return "invalid"
            return "pending"

    def compute_correlation(
            self,
            rhs: int,
            lhs: List[int]
    ) -> Any:
        """
        计算 RHS<-LHS 的相关性。

        流程：
        1. 构建列联表
        2. 计算期望频数并检查
        3. 若不满足则尝试聚类归并
        4. 基于 φ² 归一化和方向/强弱阈值决策

        :param rhs: RHS 属性索引
        :param lhs: LHS 属性索引列表
        :return: 相关性判断结果
        """
        linked = self.build_linked_table(lhs, rhs)
        if linked.size == 0:
            logger.warning("列联表为空，返回 invalid")
            return "invalid"
        exp_freq = self.compute_expected_frequencies(linked)
        if len(lhs) > 2 and not self._check_expected_frequencies(exp_freq):
            logger.warning("期望分布频数表不符合要求，无法计算相关性。")
            return False
        phi2_pre, exp_phi = self.compute_phi_stats(linked)
        logger.info(
            f"计算列 {[self._get_column_name(col) for col in lhs]} 和列 {self._get_column_name(rhs)} 之间的φ²: {phi2_pre}")
        logger.info(
            f"计算列 {[self._get_column_name(col) for col in lhs]} 和列 {self._get_column_name(rhs)} 之间的期望: {exp_phi}")
        norm_phi = self.normalize_phi(phi2_pre, exp_phi)
        rhs_name = self._get_column_name(rhs)
        lhs_names = [self._get_column_name(col) for col in lhs]
        logger.info(
            f"计算列 {lhs_names} 和列 {rhs_name} 之间的相关性 (φ²): {norm_phi}")
        return self.determine_correlation_result(norm_phi, lhs, linked)
