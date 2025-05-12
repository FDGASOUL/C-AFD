import math
import logging
import time
import copy
import numpy as np
import pandas as pd
from scipy.stats.contingency import association

# 获取日志实例
logger = logging.getLogger(__name__)


def compute_expected_frequencies(crosstab):
    """
    根据实际分布频数表计算期望分布频数表。
    :param crosstab: 实际分布频数表（二维列表）。
    :return: 期望分布频数表（二维列表）。
    """
    row_totals = [sum(row) for row in crosstab]  # 每一行的总和
    col_totals = [sum(col) for col in zip(*crosstab)]  # 每一列的总和
    total = sum(row_totals)  # 整个列联表的总和

    # 构建期望频数表
    expected_frequencies = [
        [(row_total * col_total) / total for col_total in col_totals]
        for row_total in row_totals
    ]

    return expected_frequencies


def compute_normalized_phi_squared_one(linked_table, expected_frequencies):
    # 计算总观测数
    total = sum(sum(row) for row in linked_table)
    if total == 0:
        msg = "总观测数为零，设置 φ² 为 0。"
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return "invalid"

    # 计算卡方统计量
    chi_squared = 0.0
    for i, row in enumerate(linked_table):
        for j, observed in enumerate(row):
            expected = expected_frequencies[i][j]
            if expected > 0:
                chi_squared += ((observed - expected) ** 2) / expected

    # 自由度 d
    d1 = len(linked_table)
    d2 = len(linked_table[0]) if d1 > 0 else 0
    d = min(d1, d2)
    if d <= 1:
        msg = "自由度为零，设置 φ² 为 0。"
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return "invalid"

    # 计算 φ² 指标
    phi_squared = chi_squared / (total * (d - 1))

    print(f"φ² = {phi_squared:.3f}")

    # # 定义 N, M, K
    # N = total
    # M = len(linked_table[0])
    # K = len(linked_table)
    #
    # # 期望的 φ²
    # expected_phi = ((M - 1) * (K - 1)) / ((N - 1) * (d - 1))
    #
    # if expected_phi == 1:
    #     normalized_phi_squared = 0
    # else:
    #     normalized_phi_squared = (phi_squared - expected_phi) / (1 - expected_phi)
    #
    # normalized_phi_squared_plus = max(normalized_phi_squared, 0)
    #
    # msg = f"归一化的 φ² = {normalized_phi_squared_plus:.3f}"
    # if logger:
    #     logger.info(msg)
    # else:
    #     print(msg)

    return phi_squared


def compute_normalized_phi_squared_two(linked_table):
    # 计算总观测数
    total = sum(sum(row) for row in linked_table)

    # 自由度 d
    d1 = len(linked_table)
    d2 = len(linked_table[0]) if d1 > 0 else 0
    d = min(d1, d2)
    # 定义 N, M, K
    N = total
    M = len(linked_table[0])
    K = len(linked_table)

    # 期望的 φ²
    expected_phi = ((M - 1) * (K - 1)) / ((N - 1) * (d - 1))

    # if expected_phi == 1:
    #     normalized_phi_squared = 0
    # else:
    #     normalized_phi_squared = (phi_squared - expected_phi) / (1 - expected_phi)
    #
    # normalized_phi_squared_plus = max(normalized_phi_squared, 0)
    #
    # msg = f"归一化的 φ² = {normalized_phi_squared_plus:.3f}"
    # if logger:
    #     logger.info(msg)
    # else:
    #     print(msg)

    return expected_phi


class Incorporate:
    """
    归并工具类（使用聚类方法归并，仅归并实际计数表，然后依据归并结果计算期望计数表）。
    """
    similarity_threshold = 0.001  # DP-Means 阈值，可调整
    small_ratio_threshold = 0.001  # 小簇占比阈值，例如5%
    big_k = 5  # 大簇数量

    @staticmethod
    def dpmeans_with_groups(X, lambda_, init_centers_idx, small_idx, max_iter=100):
        """
        按大簇、中簇、小簇策略的 DP-Means：
        - init_centers_idx: 大簇行/列索引，初始质心
        - small_idx: 小簇行/列索引，总是硬分配，不新建簇
        """
        n_samples, _ = X.shape
        # 若样本数不足，则每行一簇
        if n_samples < len(init_centers_idx):
            labels = np.arange(n_samples, dtype=int)
            return labels, X.copy()

        # 初始化质心列表
        centers = [X[i].copy() for i in init_centers_idx]
        labels = np.zeros(n_samples, dtype=int)

        for _ in range(max_iter):
            changed = False
            # 分配阶段
            for i in range(n_samples):
                x = X[i]
                dists2 = np.sum((np.vstack(centers) - x) ** 2, axis=1)
                j_min = np.argmin(dists2)
                if i in small_idx:
                    # 小簇：总是归并到最近簇
                    labels[i] = j_min
                else:
                    # 大簇和中簇
                    if dists2[j_min] > lambda_ and i not in init_centers_idx:
                        # 中簇且超阈值，新建簇
                        centers.append(x.copy())
                        labels[i] = len(centers) - 1
                        changed = True
                    else:
                        labels[i] = j_min
            # 更新阶段
            new_centers = []
            for k in range(len(centers)):
                members = X[labels == k]
                if len(members) > 0:
                    new_centers.append(members.mean(axis=0))
                else:
                    new_centers.append(centers[k])
            centers = new_centers
            if not changed:
                break
        return labels, np.vstack(centers)

    def cluster_rows(self, matrix):
        arr = np.array(matrix, dtype=float)
        n_rows = arr.shape[0]
        if n_rows <= 1:
            return np.zeros(n_rows, dtype=int)
        # 计算每行总和
        row_sums = arr.sum(axis=1)
        total = row_sums.sum()
        # 大簇：前 big_k 行
        big_idx = np.argsort(row_sums)[-self.big_k:]
        # 小簇：占比 < small_ratio_threshold
        small_idx = np.where(row_sums / total < self.small_ratio_threshold)[0]
        # 中簇：剩余
        # 对行归一化
        norm_data = self.normalize_rows(arr)
        labels, _ = self.dpmeans_with_groups(
            norm_data,
            lambda_=self.similarity_threshold,
            init_centers_idx=big_idx.tolist(),
            small_idx=small_idx.tolist()
        )
        return labels

    def cluster_columns(self, matrix):
        arr = np.array(matrix, dtype=float)
        n_cols = arr.shape[1]
        if n_cols <= 1:
            return np.zeros(n_cols, dtype=int)
        # 计算每列总和
        col_sums = arr.sum(axis=0)
        total = col_sums.sum()
        big_idx = np.argsort(col_sums)[-self.big_k:]
        small_idx = np.where(col_sums / total < self.small_ratio_threshold)[0]
        # 对列归一化并转置
        norm_data = self.normalize_columns(arr).T
        labels, _ = self.dpmeans_with_groups(
            norm_data,
            lambda_=self.similarity_threshold,
            init_centers_idx=big_idx.tolist(),
            small_idx=small_idx.tolist()
        )
        return labels

    @staticmethod
    def normalize_rows(matrix):
        arr = np.array(matrix, dtype=float)
        row_sums = arr.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return arr / row_sums

    @staticmethod
    def normalize_columns(matrix):
        arr = np.array(matrix, dtype=float)
        col_sums = arr.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        return arr / col_sums

    @staticmethod
    def merge_by_clusters(matrix, row_labels, col_labels):
        arr = np.array(matrix, dtype=float)
        unique_row = np.unique(row_labels)
        unique_col = np.unique(col_labels)
        merged = np.zeros((len(unique_row), len(unique_col)))
        for i, rl in enumerate(unique_row):
            row_inds = np.where(row_labels == rl)[0]
            for j, cl in enumerate(unique_col):
                col_inds = np.where(col_labels == cl)[0]
                merged[i, j] = arr[np.ix_(row_inds, col_inds)].sum()
        return merged

    @staticmethod
    def compute_expected(table):
        arr = np.array(table, dtype=float)
        row_totals = arr.sum(axis=1, keepdims=True)
        col_totals = arr.sum(axis=0, keepdims=True)
        grand_total = arr.sum()
        if grand_total == 0:
            return np.zeros_like(arr)
        return (row_totals * col_totals) / grand_total

    @staticmethod
    def compute_dirty_ratio(matrix, threshold_value=5):
        arr = np.array(matrix, dtype=float)
        total_cells = arr.size
        dirty_count = np.sum(arr < threshold_value)
        return dirty_count / total_cells

    def merge_tables(self, actual_table):
        actual_arr = np.array(actual_table, dtype=float)
        # 判断行多还是列多，多的先归并
        if actual_arr.shape[0] > actual_arr.shape[1]:
            row_labels = self.cluster_rows(actual_arr)
            col_labels = self.cluster_columns(actual_arr)
        else:
            col_labels = self.cluster_columns(actual_arr)
            row_labels = self.cluster_rows(actual_arr)
        merged_actual = self.merge_by_clusters(actual_arr, row_labels, col_labels)
        return merged_actual.tolist()


if __name__ == "__main__":
    data = pd.read_csv("data/rwd/hospital.csv")
    # 随机抽样5000
    data = data.sample(n=5000, random_state=42)
    # 取指定两列构建列联表
    contingency_table = pd.crosstab(data.iloc[:, 3], data.iloc[:, 8])

    # cramers_v = association(contingency_table, method="cramer")
    # 转为二维列表
    contingency_table = contingency_table.values.tolist()

    # 计算crameV
    # cramers_v = association(contingency_table, method="cramer")

    # print(f"cramers_v = {cramers_v:.3f}")

    # 计算期望计数
    expected_frequencies = compute_expected_frequencies(contingency_table)

    start_time = time.time()
    # 计算相关性
    phi_squared = compute_normalized_phi_squared_one(contingency_table, expected_frequencies)
    expected_phi = compute_normalized_phi_squared_two(contingency_table)
    if expected_phi == 1:
        normalized_phi_squared = 0
    else:
        normalized_phi_squared = (phi_squared - expected_phi) / (1 - expected_phi)

    normalized_phi_squared_plus = max(normalized_phi_squared, 0)
    runtime = time.time() - start_time
    print(f"计算耗时：{runtime:.3f}秒")
    print(f"归一化的 φ² = {normalized_phi_squared_plus:.3f}")

    # 初始化归并器
    start_time = time.time()
    incorporator = Incorporate()
    # 归并操作
    merged_actual = incorporator.merge_tables(contingency_table)

    runtime = time.time() - start_time
    print(f"归并耗时：{runtime:.3f}秒")

    # # 计算crameV
    # cramers_v = association(merged_actual, method="cramer")
    #
    # print(f"cramers_v = {cramers_v:.3f}")

    # 计算期望计数表
    merged_expected = compute_expected_frequencies(merged_actual)
    # 计算相关性
    phi_squared = compute_normalized_phi_squared_one(merged_actual, merged_expected)
    normalized_phi_squared_plus = (phi_squared - expected_phi) / (1 - expected_phi)

    print(f"归一化的 φ² = {normalized_phi_squared_plus:.3f}")
