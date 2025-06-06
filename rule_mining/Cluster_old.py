import logging
from typing import List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

# 获取日志实例
logger = logging.getLogger(__name__)


class Incorporate:
    """
    表格归并工具类。

    基于行/列聚类对观测计数表进行分级归并，
    并可计算归并后的期望计数表及脏格比例。

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
        分层 DP-Means 聚类：
        - 大簇(init_centers_idx)、小簇(small_idx)和中簇策略

        :param X: 输入样本矩阵，shape=(n_samples, n_features)
        :param lambda_: 最大距离阈值
        :param init_centers_idx: 初始大簇样本索引列表
        :param small_idx: 小簇样本索引列表，总被分配至现有簇
        :param max_iter: 最大迭代次数
        :return: (labels, centers)，labels为每样本簇ID，centers为簇心矩阵
        """
        n_samples, _ = X.shape
        if n_samples < len(init_centers_idx):
            return np.arange(n_samples, dtype=int), X.copy()

        centers: List[np.ndarray] = [X[i].copy() for i in init_centers_idx]
        labels = np.zeros(n_samples, dtype=int)

        for _ in range(max_iter):
            changed = False
            # 分配阶段
            for i in range(n_samples):
                dists2 = np.sum((np.vstack(centers) - X[i]) ** 2, axis=1)
                j_min = int(np.argmin(dists2))
                if i in small_idx:
                    labels[i] = j_min
                else:
                    if dists2[j_min] > lambda_ and i not in init_centers_idx:
                        centers.append(X[i].copy())
                        labels[i] = len(centers) - 1
                        changed = True
                    else:
                        labels[i] = j_min
            # 更新阶段
            new_centers: List[np.ndarray] = []
            for k in range(len(centers)):
                members = X[labels == k]
                new_centers.append(members.mean(axis=0) if len(members) else centers[k])
            centers = new_centers
            if not changed:
                break
        return labels, np.vstack(centers)

    def cluster_rows(self, matrix: Any) -> np.ndarray:
        """
        对观测表的行进行聚类。

        :param matrix: 原始计数表，二维可转换为 numpy 数组
        :return: 行聚类标签数组，length 为行数
        """
        arr = np.array(matrix, dtype=float)
        if arr.shape[0] <= 1:
            return np.zeros(arr.shape[0], dtype=int)

        row_sums = arr.sum(axis=1)
        total = row_sums.sum() or 1
        big_idx = list(np.argsort(row_sums)[-self.big_k:])
        small_idx = list(np.where(row_sums / total < self.small_ratio_threshold)[0])
        norm_data = self.normalize_rows(arr)
        labels, _ = self.dpmeans_with_groups(
            norm_data, self.similarity_threshold, big_idx, small_idx
        )
        return labels

    def cluster_columns(self, matrix: Any) -> np.ndarray:
        """
        对观测表的列进行聚类。

        :param matrix: 原始计数表
        :return: 列聚类标签数组，length 为列数
        """
        arr = np.array(matrix, dtype=float)
        if arr.shape[1] <= 1:
            return np.zeros(arr.shape[1], dtype=int)

        col_sums = arr.sum(axis=0)
        total = col_sums.sum() or 1
        big_idx = list(np.argsort(col_sums)[-self.big_k:])
        small_idx = list(np.where(col_sums / total < self.small_ratio_threshold)[0])
        norm_data = self.normalize_columns(arr).T
        labels, _ = self.dpmeans_with_groups(
            norm_data, self.similarity_threshold, big_idx, small_idx
        )
        return labels

    @staticmethod
    def normalize_rows(matrix: np.ndarray) -> np.ndarray:
        """
        对行归一化，使每行和为1。

        :param matrix: 输入矩阵
        :return: 归一化矩阵
        """
        arr = matrix.astype(float)
        sums = arr.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1
        return arr / sums

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

    @staticmethod
    def merge_by_clusters(
            arr: np.ndarray,
            row_labels: np.ndarray,
            col_labels: np.ndarray
    ) -> np.ndarray:
        """
        按聚类标签对矩阵聚合归并。

        无 Python 循环，使用稀疏 COO 矩阵映射方式。

        :param arr: 原始观测矩阵，shape=(N, M)
        :param row_labels: 行标签数组，length N
        :param col_labels: 列标签数组，length M
        :return: 归并后矩阵，shape=(R, C)
        """
        sparse = coo_matrix(arr)
        rows = row_labels[sparse.row]
        cols = col_labels[sparse.col]
        data = sparse.data

        unique_r = np.unique(row_labels)
        unique_c = np.unique(col_labels)
        r_idx = np.searchsorted(unique_r, rows)
        c_idx = np.searchsorted(unique_c, cols)

        merged = coo_matrix((data, (r_idx, c_idx)), shape=(unique_r.size, unique_c.size))
        return merged.toarray()

    def merge_tables(self, actual_table: Any) -> np.ndarray:
        """
        自动选择行或列先归并，返回合并后的计数表。

        :param actual_table: 原始观测计数表
        :return: 聚类归并后的计数矩阵
        """
        arr = np.array(actual_table, dtype=float)
        if arr.shape[0] > arr.shape[1]:
            rows_lbl = self.cluster_rows(arr)
            cols_lbl = self.cluster_columns(arr)
        else:
            cols_lbl = self.cluster_columns(arr)
            rows_lbl = self.cluster_rows(arr)
        return self.merge_by_clusters(arr, rows_lbl, cols_lbl)
