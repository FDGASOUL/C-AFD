# TODO: 对计算结果的缓存，是否需要？都什么结果能够用上缓存？
# TODO: 一次检测正反的方向，前面代码需要改变，只影响速度，不影响结果
import math

import numpy as np
from scipy.stats import fisher_exact

from Incorporate_into import Incorporate
import logging

# 获取日志实例
logger = logging.getLogger(__name__)


class CorrelationCalculator:
    """
    相关性计算工具类。
    用于封装不同的相关性计算方法，以便在项目中复用。
    """
    directional_threshold = 0.7  # 方向判断阈值

    def __init__(self, column_layout_data):
        """
        初始化相关性计算器。
        :param column_layout_data: ColumnLayoutRelationData 的实例，包含列的数值索引和列数据。
        """
        self.columnVectors = column_layout_data.get_column_vectors()
        self.columnData = column_layout_data.get_column_data()
        self.schema = column_layout_data.get_schema()  # 获取列名映射

    def _get_column_name(self, column_index):
        """
        根据列索引获取列名。
        :param column_index: 列索引。
        :return: 对应的列名。
        """
        return self.schema[column_index]

    # TODO: 应该采取左部PLI，右部vectors的方式，但是PLI应该怎么交叉呢？
    def build_linked_table(self, lhs, rhs):
        """
        构建列联表（linked table），左部 (lhs) 使用 PLI，右部 (rhs) 使用 columnVectors。

        :param lhs: 左部属性索引列表（可以是单个属性，也可以是多个属性）。
        :param rhs: 右部属性索引（单个属性）。
        :return: 一个包含映射关系的二维数组，表示列联表。
        """

        # 1. 计算 LHS 的 PLI
        if len(lhs) == 1:
            pli_lhs = self.columnData[lhs[0]]["PLI"]
        else:
            pli_lhs = self._cross_plis([self.columnData[col]["PLI"] for col in lhs])

        # 2. **筛选前 5 大簇**
        pli_lhs = sorted(pli_lhs, key=len, reverse=True)[:5]  # 按大小降序，取前 5 个

        # 3. 计算 RHS 的 columnVectors
        column_vectors_rhs = self.columnVectors[rhs]

        # 4. 获取 RHS 唯一值，初始化映射表
        unique_values = set()
        for cluster in pli_lhs:
            for pos in cluster:
                unique_values.add(column_vectors_rhs[pos])
        rhs_index = {value: 0 for value in unique_values}

        # 5. 构建列联表
        crosstab_list = []
        for cluster in pli_lhs:
            cluster_map = rhs_index.copy()
            for position in cluster:
                value_rhs = column_vectors_rhs[position]
                cluster_map[value_rhs] = cluster_map.get(value_rhs, 0) + 1
            crosstab_list.append(cluster_map)

        # 6. 转换为二维数组
        crosstab = []
        for cluster_map in crosstab_list:
            crosstab.append([cluster_map.get(key, 0) for key in sorted(rhs_index.keys())])

        return crosstab

    def build_linked_table_old(self, column_a, column_b):
        """
        利用 Probing Table 构建列连表（linked table）。
        :param column_a: 列 A 的索引。
        :param column_b: 列 B 的索引或组合（可以是单个索引，也可以是集合）。
        :return: 一个包含映射关系的二维数组，表示列连表。
        """
        pli_a = self.columnData[column_a]["PLI"]

        if len(column_b) == 1:
            # 如果只有一个列，直接获取其 columnVectors
            column_vectors_b = self.columnVectors[column_b[0]]
        else:
            # 对多个列的 columnVectors 进行交叉，形成新 columnVectors
            column_vectors_b = self._cross_column_vectors([self.columnVectors[col] for col in column_b])

        # 根据 pli_a 中所有实际出现的行构建 column_b 的索引，避免出现空索引的情况
        unique_values = set()
        for cluster in pli_a:
            for pos in cluster:
                value = column_vectors_b[pos]
                unique_values.add(value)
        lhs_index = {value: 0 for value in unique_values}

        # 填充列联表：每个 cluster 克隆一份初始化后的索引，并更新计数
        crosstab_list = []
        for cluster in pli_a:
            cluster_map = lhs_index.copy()
            for position in cluster:
                value_b = column_vectors_b[position]
                cluster_map[value_b] = cluster_map.get(value_b, 0) + 1
            crosstab_list.append(cluster_map)

        # 转换为二维数组：按照排序后的索引键形成每一行的计数列表
        crosstab = []
        for cluster_map in crosstab_list:
            crosstab.append([cluster_map.get(key, 0) for key in sorted(lhs_index.keys())])

        return crosstab

    def compute_expected_frequencies(self, crosstab):
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

    def _check_expected_frequencies(self, expected_frequencies):
        """
        检查期望分布频数表中是否超过 80% 的格子的期望计数大于 5。
        :param expected_frequencies: 期望分布频数表。
        :return: 如果满足条件，返回 True；否则，返回 False。
        """
        valid_count = sum(1 for row in expected_frequencies for value in row if value > 5)
        total_cells = len(expected_frequencies) * len(expected_frequencies[0])
        return valid_count / total_cells >= 0.8

    def _cross_plis(self, pli_list):
        """
        交叉多个 PLI 形成新的 PLI（不过滤单元素簇）。
        :param pli_list: 需要交叉的 PLI 列表。
        :return: 交叉后的 PLI
        """
        from collections import defaultdict

        # 建立行号到簇 ID 的映射
        pli_maps = [{pos: idx for idx, cluster in enumerate(pli) for pos in cluster} for pli in pli_list]

        # 计算交叉结果
        merged_clusters = defaultdict(set)
        for pos in pli_maps[0]:  # 遍历第一个 PLI 的行号
            key = tuple(pli_map.get(pos, -1) for pli_map in pli_maps)  # 形成唯一标识
            merged_clusters[key].add(pos)

        # 不再过滤单元素簇，保持所有簇
        return [sorted(list(cluster)) for cluster in merged_clusters.values()]

    def check_dependency_direction(self, rhs_column, lhs_columns):
        """
        检查函数依赖的方向是否正确（按行计算）。
        :param lhs_columns: 左部属性列表。
        :param rhs_column: 右部属性。
        :return: 如果所有行都满足方向阈值返回 True，否则返回 False。
        """
        # 构建列联表
        crosstab = self.build_linked_table(lhs_columns, rhs_column)
        logger.info(f"目前的左部属性列表：{lhs_columns}")
        logger.info(f"目前的右部属性：{rhs_column}")

        # 记录不符合条件的行
        failed_rows = []

        # 遍历行联表（按行计算）
        for row_index, row in enumerate(crosstab):
            row_sum = sum(row)  # 当前行的总数
            if row_sum == 0:
                continue  # 跳过没有数据的行
            max_value = max(row)  # 当前行的最大值
            ratio = max_value / row_sum
            # logger.info(f"行 {row_index} 的 max_value / row_sum = {ratio}")

            if ratio < self.directional_threshold:
                failed_rows.append((row_index, ratio))  # 记录不符合条件的行

        # 如果有不符合条件的行，返回 False，并输出不符合条件的行信息
        if failed_rows:
            logger.info(f"以下行不满足阈值：{failed_rows}")
            return False

        # 所有行都满足阈值，返回 True
        return True

    # def check_dependency_direction(self, lhs_columns, rhs_column):
    #     """
    #     检查函数依赖的方向是否正确。
    #     :param lhs_columns: 左部属性列表。
    #     :param rhs_column: 右部属性。
    #     :return: 如果所有列都满足方向阈值，返回一个包含正向和反向方向判断结果的字典。
    #     """
    #     # 构建列联表
    #     crosstab = self.build_linked_table(lhs_columns, rhs_column)
    #
    #     # 判断正向（左部决定右部）方向是否满足阈值
    #     forward_check = True
    #     for col_index in range(len(crosstab[0])):
    #         column_sum = sum(row[col_index] for row in crosstab)  # 当前列的总数
    #         if column_sum == 0:
    #             continue  # 跳过没有数据的列
    #         max_value = max(row[col_index] for row in crosstab)  # 当前列的最大值
    #         if max_value / column_sum < self.directional_threshold:
    #             forward_check = False  # 如果有列不满足阈值，标记为 False
    #             break
    #
    #     # 判断反向（右部决定左部）方向是否满足阈值
    #     reverse_check = True
    #     for row_index in range(len(crosstab)):
    #         row_sum = sum(crosstab[row_index])  # 当前行的总数
    #         if row_sum == 0:
    #             continue  # 跳过没有数据的行
    #         max_value = max(crosstab[row_index])  # 当前行的最大值
    #         if max_value / row_sum < self.directional_threshold:
    #             reverse_check = False  # 如果有行不满足阈值，标记为 False
    #             break
    #
    #     return {
    #         "forward_check": forward_check,  # 正向判断结果
    #         "reverse_check": reverse_check  # 反向判断结果
    #     }

    # def compute_correlation(self, column_a, column_b):
    #     """
    #     计算两个列之间的相关性。
    #     :param column_a: 列 A 的索引。
    #     :param column_b: 列 B 的索引或组合。
    #     :return: 相关性值。
    #     """
    #     # 构建列连表
    #     linked_table = self.build_linked_table(column_a, column_b)
    #     expected_frequencies = self.compute_expected_frequencies(linked_table)
    #     # 检查期望分布频数表
    #     if not self._check_expected_frequencies(expected_frequencies):
    #         logger.warning("期望分布频数表中超过 20% 的格子期望计数未大于 5，进行归并操作。")
    #         inc = Incorporate()
    #         result = inc.merge_tables(linked_table, expected_frequencies)
    #         if not result:
    #             logger.warning("归并操作失败，改用最大似然比卡方检验计算相关性。")
    #
    #             # 改用最大似然比卡方检验
    #             total = sum(sum(row) for row in linked_table)
    #             if total == 0:
    #                 logger.warning("总观测数为零，设置 φ² 为 0。")
    #                 return 0
    #
    #             g_squared = 0
    #             for i, row in enumerate(linked_table):
    #                 for j, observed in enumerate(row):
    #                     expected = expected_frequencies[i][j]
    #                     if observed > 0 and expected > 0:
    #                         g_squared += 2 * observed * math.log(observed / expected)
    #                     elif observed > 0 and expected == 0:
    #                         logger.warning(f"期望频数为 0（{i}, {j}），无法计算最大似然比卡方统计量。")
    #
    #             chi_squared = 0
    #             for i, row in enumerate(linked_table):
    #                 for j, observed in enumerate(row):
    #                     expected = expected_frequencies[i][j]
    #                     if expected > 0:
    #                         chi_squared += ((observed - expected) ** 2) / expected
    #                     else:
    #                         logger.warning(f"期望频数为零（{i}, {j}），跳过此格子。")
    #
    #             d1, d2 = len(linked_table), len(linked_table[0])
    #             d = min(d1, d2)
    #             if d <= 1:
    #                 logger.warning("自由度为零，设置 φ² 为 0。")
    #                 return 0
    #
    #             phi_squared = g_squared / (total * (d - 1))
    #             phi_squared_1 = chi_squared / (total * (d - 1))
    #
    #             # # 使用creamV
    #             # # 保留四位小数
    #             # g_ratio = np.round(g_squared / (total * (d - 1)), 4)
    #             # chi_ratio = np.round(chi_squared / (total * (d - 1)), 4)
    #             #
    #             # # 计算平方根
    #             # phi_squared = np.sqrt(g_ratio)
    #             # phi_squared_1 = np.sqrt(chi_ratio)
    #
    #             column_a_name = self._get_column_name(column_a)
    #             column_b_names = [self._get_column_name(col) for col in column_b]
    #             logger.info(
    #                 f"计算列 {column_a_name} 和列 {column_b_names} 之间的相关性 (φ², 基于最大似然比卡方): {phi_squared}, 基于原卡方：{phi_squared_1}")
    #             return phi_squared
    #
    #         # 如果归并成功，更新表格和期望频数
    #         linked_table, expected_frequencies = result
    #
    #     # 总观测数
    #     total = sum(sum(row) for row in linked_table)
    #     if total == 0:
    #         logger.warning("总观测数为零，设置 φ² 为 0。")
    #         return 0
    #
    #     # 计算 χ²
    #     chi_squared = 0
    #     for i, row in enumerate(linked_table):
    #         for j, observed in enumerate(row):
    #             expected = expected_frequencies[i][j]
    #             if expected > 0:
    #                 chi_squared += ((observed - expected) ** 2) / expected
    #             else:
    #                 logger.warning(f"期望频数为零（{i}, {j}），跳过此格子。")
    #
    #     # 计算 φ²
    #     d1, d2 = len(linked_table), len(linked_table[0])
    #     d = min(d1, d2)
    #     if d <= 1:
    #         logger.warning("自由度为零，设置 φ² 为 0。")
    #         return 0
    #
    #     phi_squared = chi_squared / (total * (d - 1))
    #
    #     column_a_name = self._get_column_name(column_a)
    #     column_b_names = [self._get_column_name(col) for col in column_b]
    #     logger.info(f"计算列 {column_a_name} 和列 {column_b_names} 之间的相关性 (φ²): {phi_squared}")
    #     return phi_squared

    def compute_correlation(self, column_a, column_b):
        """
        计算两个列之间的相关性。
        :param column_a: 列 A 的索引。
        :param column_b: 列 B 的索引或组合。
        :return: 相关性值。
        """
        # 构建列联表
        linked_table = self.build_linked_table(column_b, column_a)
        if not linked_table:
            logger.warning("列联表为空，无法计算相关性。")
            return 0

        expected_frequencies = self.compute_expected_frequencies(linked_table)

        # 检查期望分布频数表
        if not self._check_expected_frequencies(expected_frequencies) :
            logger.warning("期望分布频数表中超过 20% 的格子期望计数未大于 5，进行归并操作。")
            inc = Incorporate()
            result = inc.merge_tables(linked_table, expected_frequencies)
            if not result:
                logger.warning("归并操作失败。")
                return 0
            #
            #         # # 改用最大似然比卡方检验
            #         # total = sum(sum(row) for row in linked_table)
            #         #
            #         # if total == 0:
            #         #     logger.warning("总观测数为零，设置 φ² 为 0。")
            #         #     return 0
            #         #
            #         # g_squared = 0
            #         # for i, row in enumerate(linked_table):
            #         #     for j, observed in enumerate(row):
            #         #         expected = expected_frequencies[i][j]
            #         #         if observed > 0 and expected > 0:
            #         #             g_squared += 2 * observed * math.log(observed / expected)
            #         #         elif observed > 0 and expected == 0:
            #         #             logger.warning(f"期望频数为 0（{i}, {j}），无法计算最大似然比卡方统计量。")
            #         #
            #         # chi_squared = 0
            #         # for i, row in enumerate(linked_table):
            #         #     for j, observed in enumerate(row):
            #         #         expected = expected_frequencies[i][j]
            #         #         if expected > 0:
            #         #             chi_squared += ((observed - expected) ** 2) / expected
            #         #         else:
            #         #             logger.warning(f"期望频数为零（{i}, {j}），跳过此格子。")
            #         #
            #         # d1, d2 = len(linked_table), len(linked_table[0])
            #         # d = min(d1, d2)
            #         # if d <= 1:
            #         #     logger.warning("自由度为零，设置 φ² 为 0。")
            #         #     return 0
            #         #
            #         # phi_squared = g_squared / (total * (d - 1))
            #         # phi_squared_1 = chi_squared / (total * (d - 1))
            #
            #         # # 使用creamV
            #         # # 保留四位小数
            #         # g_ratio = np.round(g_squared / (total * (d - 1)), 4)
            #         # chi_ratio = np.round(chi_squared / (total * (d - 1)), 4)
            #         #
            #         # # 计算平方根
            #         # phi_squared = np.sqrt(g_ratio)
            #         # phi_squared_1 = np.sqrt(chi_ratio)
            #
            #         # column_a_name = self._get_column_name(column_a)
            #         # column_b_names = [self._get_column_name(col) for col in column_b]
            #         # logger.info(
            #         #     f"计算列 {column_a_name} 和列 {column_b_names} 之间的相关性 (φ², 基于最大似然比卡方): {phi_squared}, 基于原卡方：{phi_squared_1}")
            #         # return phi_squared_1
            #
            # 如果归并成功，更新表格和期望频数
            logger.info("归并操作成功。")
            linked_table, expected_frequencies = result

        # 总观测数
        total = sum(sum(row) for row in linked_table)
        if total == 0:
            logger.warning("总观测数为零，设置 φ² 为 0。")
            return 0

        # 计算 χ²
        chi_squared = 0
        for i, row in enumerate(linked_table):
            for j, observed in enumerate(row):
                expected = expected_frequencies[i][j]
                if expected > 0:
                    chi_squared += ((observed - expected) ** 2) / expected
                else:
                    logger.warning(f"期望频数为零（{i}, {j}），跳过此格子。")

        # 计算 φ²
        d1, d2 = len(linked_table), len(linked_table[0])
        d = min(d1, d2)
        if d <= 1:
            logger.warning("自由度为零，设置 φ² 为 0。")
            return 0

        phi_squared = chi_squared / (total * (d - 1))

        # 使用creamV
        cramer_v = np.sqrt(phi_squared)

        column_a_name = self._get_column_name(column_a)
        column_b_names = [self._get_column_name(col) for col in column_b]
        logger.info(f"计算列 {column_a_name} 和列 {column_b_names} 之间的相关性 (φ²): {cramer_v}")
        return cramer_v
