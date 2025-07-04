import logging
from typing import List, Tuple, Any

import numpy as np

# 获取日志实例
logger = logging.getLogger(__name__)


class Incorporate:
    """
    表格列聚类工具类（仅列归并分析，不修改原始数据），用于返回可合并的列对。

    类属性：
        similarity_threshold (float): 聚类阈值
        small_ratio_threshold (float): 小簇占比阈值
        big_k (int): 大簇初始簇数
    """
    similarity_threshold: float = 0.001
    small_ratio_threshold: float = 0.001
    big_k: int = 5

    @staticmethod
    def dpmeans_with_groups(
            X: np.ndarray,
            lambda_: float,
            init_centers_idx: List[int],
            small_idx: List[int],
            max_iter: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        分层 DP-Means 聚类：仅聚类，不输出合并结果。

        :param X: 输入样本矩阵，shape=(n_samples, n_features)
        :param lambda_: 最大距离阈值
        :param init_centers_idx: 初始大簇样本索引列表
        :param small_idx: 小簇样本索引列表，总被分配至现有簇
        :param max_iter: 最大迭代次数
        :return: (labels, centers)，labels 为每样本簇 ID，centers 为簇心矩阵
        """
        n_samples, _ = X.shape
        if n_samples < len(init_centers_idx):
            return np.arange(n_samples, dtype=int), X.copy()

        centers: List[np.ndarray] = [X[i].copy() for i in init_centers_idx]
        labels = np.zeros(n_samples, dtype=int)
        small_idx_set = set(small_idx)

        # 最大列总和的索引在 init_centers_idx 中的位置（作为归并目标）
        largest_sum_idx = init_centers_idx[-1]  # 默认最后一个是最大，因为已按 col_sum 排序
        target_cluster_id = init_centers_idx.index(largest_sum_idx)

        for _ in range(max_iter):
            changed = False
            # 分配阶段
            for i in range(n_samples):
                if i in small_idx_set:
                    # 小簇强制合并到目标簇
                    labels[i] = target_cluster_id
                    continue

                dists2 = np.sum((np.vstack(centers) - X[i]) ** 2, axis=1)
                j_min = int(np.argmin(dists2))
                if dists2[j_min] > lambda_:
                    centers.append(X[i].copy())
                    labels[i] = len(centers) - 1
                    changed = True
                else:
                    labels[i] = j_min

            # 更新阶段：重新计算每个簇的簇心（忽略小样本）
            new_centers: List[np.ndarray] = []
            for k in range(len(centers)):
                members_idx = [i for i in range(n_samples) if labels[i] == k and i not in small_idx_set]
                if members_idx:
                    members = X[members_idx]
                    new_centers.append(members.mean(axis=0))
                else:
                    # 没有中/大样本属于该簇，保留原簇心
                    new_centers.append(centers[k])

            centers = new_centers
            if not changed:
                break

        return labels, np.vstack(centers)

    def cluster_columns(self, matrix: Any) -> np.ndarray:
        """
        对观测表的列进行聚类，返回每列的聚类标签。

        :param matrix: 原始计数表，二维可转换为 numpy 数组
        :return: 列聚类标签数组，length 为列数
        """
        arr = np.array(matrix, dtype=float)
        n_cols = arr.shape[1]
        if n_cols <= 1:
            return np.zeros(n_cols, dtype=int)

        # 计算每列总和，用于识别“大簇”和“小簇”的初始化
        col_sums = arr.sum(axis=0)
        total = col_sums.sum() or 1
        big_idx = list(np.argsort(col_sums)[-self.big_k:])  # 取总和最大的 big_k 列作为初始大簇中心
        small_idx = list(np.where(col_sums / total < self.small_ratio_threshold)[0])
        # 对列进行归一化，使每列和为 1
        norm_data = self.normalize_columns(arr).T  # 转置后每行为一个“样本”

        labels, _ = self.dpmeans_with_groups(
            norm_data, self.similarity_threshold, big_idx, small_idx
        )
        return labels

    @staticmethod
    def normalize_columns(matrix: np.ndarray) -> np.ndarray:
        """
        对列归一化，使每列和为1。

        :param matrix: 输入矩阵
        :return: 归一化矩阵
        """
        arr = matrix.astype(float)
        sums = arr.sum(axis=0, keepdims=True)
        sums[sums == 0] = 1
        return arr / sums

    def get_mergeable_column_pairs(self, table: Any) -> List[Tuple[int, int]]:
        """
        根据列聚类结果，返回所有可合并的列对 (i, j)，其中 i < j，
        同一簇内的列两两组合并，如果一个簇有 k 列，则产生 C(k,2) 个列对。

        :param table: 原始观测计数表，二维可转换为 numpy 数组
        :return: List[Tuple[int, int]]，每个元组表示一对可合并的列索引
        """
        arr = np.array(table, dtype=float)
        labels = self.cluster_columns(arr)
        # 将同一簇内的列索引分组
        cluster_to_cols = {}
        for idx, lbl in enumerate(labels):
            cluster_to_cols.setdefault(lbl, []).append(idx)

        merge_pairs: List[Tuple[int, int]] = []
        # 对每个簇，生成所有列对
        for cols_in_cluster in cluster_to_cols.values():
            if len(cols_in_cluster) <= 1:
                continue
            # 若簇内有多列，则两两组合
            for i in range(len(cols_in_cluster)):
                for j in range(i + 1, len(cols_in_cluster)):
                    merge_pairs.append((cols_in_cluster[i], cols_in_cluster[j]))

        return merge_pairs
