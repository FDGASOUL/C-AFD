import math
import logging
import time
import copy
import numpy as np
import pandas as pd
from scipy.stats.contingency import association
from sklearn.cluster import AgglomerativeClustering
# 获取日志实例
logger = logging.getLogger(__name__)


class Incorporate:
    """
    归并工具类（使用聚类方法归并，仅归并实际计数表，然后依据归并结果计算期望计数表）。
    """
    # similarity_threshold 用于聚类时的距离阈值，
    # 较小的值要求归一化后向量更加相似。
    similarity_threshold = 0.5  # 可根据需要调整

    @staticmethod
    def normalize_rows(matrix):
        """
        对二维列表中的每一行归一化（转为概率分布）。

        :param matrix: 二维列表或 NumPy 数组
        :return: 归一化后的 NumPy 数组，每行之和为 1（若某行全为0，则归一化后仍为0）
        """
        arr = np.array(matrix, dtype=float)
        row_sums = arr.sum(axis=1, keepdims=True)
        # 避免除以0：将和为0的行设置为1，归一化后该行仍为0
        row_sums[row_sums == 0] = 1
        return arr / row_sums

    @staticmethod
    def normalize_columns(matrix):
        """
        对二维列表中的每一列归一化（转为概率分布）。

        :param matrix: 二维列表或 NumPy 数组
        :return: 归一化后的 NumPy 数组，每列之和为 1（若某列全为0，则归一化后仍为0）
        """
        arr = np.array(matrix, dtype=float)
        col_sums = arr.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        return arr / col_sums

    def cluster_rows(self, matrix):
        """
        对矩阵的行进行聚类：先归一化每一行（视为分布），再使用 AgglomerativeClustering
        （余弦距离，平均连接）得到聚类标签。

        :param matrix: 二维列表或 NumPy 数组，形状为 (n_rows, n_features)
        :return: 聚类标签（NumPy 数组）
        """
        norm_data = self.normalize_rows(matrix)
        n_rows = norm_data.shape[0]

        if n_rows <= 1:
            return np.array([0])
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage='average',
            distance_threshold=self.similarity_threshold
        )
        labels = clustering.fit_predict(norm_data)
        return labels

    def cluster_columns(self, matrix):
        """
        对矩阵的列进行聚类：先对每列归一化（视为分布），
        再对转置后的数据进行聚类，得到聚类标签。

        :param matrix: 二维列表或 NumPy 数组，形状为 (n_rows, n_columns)
        :return: 聚类标签（NumPy 数组）
        """
        # 对列进行归一化：转置后每一行对应原矩阵的一列
        norm_data = self.normalize_columns(matrix).T
        n_cols = norm_data.shape[0]
        if n_cols <= 1:
            return np.array([0])
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=self.similarity_threshold
        )
        labels = clustering.fit_predict(norm_data)
        return labels

    @staticmethod
    def merge_by_clusters(matrix, row_labels, col_labels):
        """
        根据行和列聚类标签对矩阵进行归并。对于每个（行簇, 列簇）的组合，
        将原始矩阵中对应区域的值累加。

        :param matrix: 二维 NumPy 数组
        :param row_labels: 行聚类标签（NumPy 数组）
        :param col_labels: 列聚类标签（NumPy 数组）
        :return: 归并后的二维数组
        """
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
        """
        根据归并后的实际计数表计算期望计数表。
        使用公式：expected_{ij} = (row_total_i * col_total_j) / grand_total

        :param table: 归并后的实际计数表（二维 NumPy 数组）
        :return: 期望计数表（二维 NumPy 数组）
        """
        arr = np.array(table, dtype=float)
        row_totals = arr.sum(axis=1, keepdims=True)
        col_totals = arr.sum(axis=0, keepdims=True)
        grand_total = arr.sum()
        # 当grand_total为0时，避免除法错误
        if grand_total == 0:
            return np.zeros_like(arr)
        expected = (row_totals * col_totals) / grand_total
        return expected

    @staticmethod
    def compute_dirty_ratio(matrix, threshold_value=5):
        """
        计算矩阵中小于 threshold_value 的格子所占比例。

        :param matrix: 二维 NumPy 数组
        :param threshold_value: 临界值（默认5）
        :return: 小于 threshold_value 的格子比例（0-1之间）
        """
        arr = np.array(matrix, dtype=float)
        total_cells = arr.size
        dirty_count = np.sum(arr < threshold_value)
        return dirty_count / total_cells

    def merge_tables(self, actual_table, expected_table=None):
        """
        归并操作：仅对实际分布列联表进行归并，
        然后利用归并后的实际计数表计算期望计数表。
        归并过程：
          1. 使用聚类方法对实际计数表的行和列分别进行分组，
             得到各方向的聚类标签。
          2. 根据聚类标签对实际计数表进行归并。
          3. 根据归并后的实际计数表计算期望计数表。
          4. 如果期望计数表中小于 5 的格子比例超过 20%，则认为归并失败，
             返回 False；否则返回归并后的 (actual, expected) 表。

        :param actual_table: 实际分布列联表（二维列表）。
        :param expected_table: 可选参数，此处忽略，直接根据实际表计算期望表。
        :return: 若归并成功，返回归并后的 (actual, expected) 表，否则返回 False。
        """
        # 对实际计数表做深拷贝
        # actual_copy = copy.deepcopy(actual_table)
        # 转换为 NumPy 数组
        actual_arr = np.array(actual_table, dtype=float)

        # 使用实际计数表做聚类（行和列均依据实际分布）
        row_labels = self.cluster_rows(actual_arr)
        col_labels = self.cluster_columns(actual_arr)
        #
        # print(row_labels)
        # print(col_labels)
        # 如果聚类结果与原始行列数完全一致，说明未能归并
        if len(np.unique(row_labels)) == actual_arr.shape[0] and len(np.unique(col_labels)) == actual_arr.shape[1]:
            logger.info("聚类归并无效：未发现足够相似的行或列")
            return False

        # 根据聚类标签对实际计数表进行归并
        merged_actual = self.merge_by_clusters(actual_arr, row_labels, col_labels)

        # 根据归并后的实际计数表计算期望计数表
        merged_expected = self.compute_expected(merged_actual)

        return merged_actual.tolist(), merged_expected.tolist()


if __name__ == "__main__":
    data = pd.read_csv("data/rwd/hospital.csv")
    # 随机抽样5000
    data = data.sample(n=5000, random_state=42)
    # 取指定两列构建列联表
    contingency_table = pd.crosstab(data.iloc[:, 13], data.iloc[:, 14])

    # 转为二维列表
    contingency_table = contingency_table.values.tolist()
    start_time = time.time()
    cramer_v = association(contingency_table, method="cramer")
    print(f"Cramér's V = {cramer_v:.3f}")
    runtime = time.time() - start_time
    print(f"计算Cramér's V耗时：{runtime:.3f}秒")
    # print(contingency_table)
    # 初始化归并器
    start_time = time.time()
    incorporator = Incorporate()
    # 归并操作
    merged_actual, merged_expected = incorporator.merge_tables(contingency_table)

    runtime = time.time() - start_time
    print(f"归并耗时：{runtime:.3f}秒")

    #计算crameV
    # 计算Cramér's V系数
    observed = np.array(merged_actual)
    expected = np.array(merged_expected)

    # 计算卡方统计量
    chi2 = np.sum((observed - expected) ** 2 / np.where(expected == 0, 1e-10, expected))

    # 计算样本总量
    n = observed.sum()

    # 计算最小维度 (行数-1 和 列数-1 中的较小值)
    min_dim = min(observed.shape[0] - 1, observed.shape[1] - 1)

    # 计算Cramér's V（处理除零保护）
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if n * min_dim > 0 else 0
    print(f"Cramér's V = {cramers_v:.3f}")


