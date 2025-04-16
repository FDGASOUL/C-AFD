import math
import logging
import copy
import numpy as np
import pandas as pd
from scipy.stats.contingency import association
from umap import UMAP  # 代码中虽然有引入，但未实际使用
from sklearn.cluster import Birch
import time

# 获取日志实例
logger = logging.getLogger(__name__)


class Incorporate:
    """
    归并工具类（使用 UMAP + Birch 进行聚类归并，仅归并实际计数表，
    然后依据归并结果计算期望计数表）。
    """
    # UMAP 降维后的最大维度（如有需要使用 UMAP 降维，可以调整此参数）
    umap_n_components = 50

    # Birch 参数：
    # threshold：控制 CF 条目合并的距离阈值（数值越小，簇越紧密，簇的数量通常越多）
    # branching_factor：控制树中每个节点最多包含的子节点数
    birch_threshold = 0.5
    birch_branching_factor = 50

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

    def cluster_rows(self, matrix):
        """
        对矩阵的行进行聚类，使用 Birch 算法对归一化后的行向量聚类。
        """
        norm_data = self.normalize_rows(matrix)
        # 使用 Birch 构建 CF 树，此处 n_clusters 设置为 None 表示不进行全局再聚类，
        # 直接返回 CF 树叶节点中每个数据点对应的簇标签
        clusterer = Birch(threshold=self.birch_threshold,
                          branching_factor=self.birch_branching_factor,
                          n_clusters=None)
        labels = clusterer.fit_predict(norm_data)
        return labels

    def cluster_columns(self, matrix):
        """
        对矩阵的列进行聚类：先对列归一化，再对转置后的数据使用 Birch 聚类。
        """
        norm_data = self.normalize_columns(matrix).T

        if norm_data.shape[0] <= 1:
            return np.array([0])

        clusterer = Birch(threshold=self.birch_threshold,
                          branching_factor=self.birch_branching_factor,
                          n_clusters=None)
        labels = clusterer.fit_predict(norm_data)
        return labels

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
        expected = (row_totals * col_totals) / grand_total
        return expected

    def merge_tables(self, actual_table, expected_table=None):
        actual_arr = np.array(actual_table, dtype=float)

        # 用 UMAP + Birch 对行和列分别聚类（此处 UMAP 降维代码未作修改，如果需要可以先降维再聚类）
        row_labels = self.cluster_rows(actual_arr)
        col_labels = self.cluster_columns(actual_arr)

        # 如果没能合并任何行或列，就返回 False
        if (len(np.unique(row_labels)) == actual_arr.shape[0] and
                len(np.unique(col_labels)) == actual_arr.shape[1]):
            logger.info("聚类归并无效：未发现足够相似的行或列")
            return False

        merged_actual = self.merge_by_clusters(actual_arr, row_labels, col_labels)
        merged_expected = self.compute_expected(merged_actual)

        return merged_actual.tolist(), merged_expected.tolist()


if __name__ == "__main__":
    data = pd.read_csv("data/rwd/hospital.csv")
    data = data.sample(n=5000, random_state=42)
    # 取指定两列构建列联表
    contingency_table = pd.crosstab(data.iloc[:, 13], data.iloc[:, 14])
    contingency_table = contingency_table.values.tolist()

    # 计算原始 Cramér's V
    cramer_v_orig = association(contingency_table, method="cramer")
    print(f"原始 Cramér's V = {cramer_v_orig:.3f}")

    # 计时
    start_time = time.time()

    incorporator = Incorporate()
    result = incorporator.merge_tables(contingency_table)
    if not result:
        print("未能成功归并，请检查聚类参数或数据。")
    else:
        merged_actual, merged_expected = result

        # 重新计算 Cramér's V
        observed = np.array(merged_actual)
        expected = np.array(merged_expected)
        chi2 = np.sum((observed - expected) ** 2 / np.where(expected == 0, 1e-10, expected))
        n = observed.sum()
        min_dim = min(observed.shape[0] - 1, observed.shape[1] - 1)
        cramers_v_new = np.sqrt(chi2 / (n * min_dim)) if n * min_dim > 0 else 0
        print(f"归并后 Cramér's V = {cramers_v_new:.3f}")
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"归并耗时: {execution_time:.2f} 秒")
