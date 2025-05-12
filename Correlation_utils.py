import logging
import random
import time
import numpy as np
from line_profiler import profile
from scipy.sparse import coo_matrix

# from Incorporate_into_DBSCAN import Incorporate
# from Incorporate_into_hierarchical import Incorporate
# from Incorporate_into_dpmeans import Incorporate
from Incorporate_into_dpmeans_top import Incorporate

# 获取日志实例
logger = logging.getLogger(__name__)


class CorrelationCalculator:
    """
    相关性计算工具类。
    用于封装不同的相关性计算方法，以便在项目中复用。
    """
    upper_threshold = 0.61  # 上限阈值
    lower_threshold = 0.01  # 下限阈值

    def __init__(self, column_layout_data):
        """
        初始化相关性计算器。
        :param column_layout_data: ColumnLayoutRelationData 的实例，包含列的数值索引和列数据。
        """
        self.columnVectors = column_layout_data.get_column_vectors()
        self.columnData = column_layout_data.get_column_data()
        self.schema = column_layout_data.get_schema()  # 获取列名映射
        self.null_value_id = column_layout_data.get_null_value_id()  # 获取空值标识符

    def _get_column_name(self, column_index):
        """
        根据列索引获取列名。
        :param column_index: 列索引。
        :return: 对应的列名。
        """
        return self.schema[column_index]

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
        # pli_lhs = sorted(pli_lhs, key=len, reverse=True)[:20]  # 按大小降序，取前 5 个

        # 3. 计算 RHS 的 columnVectors
        column_vectors_rhs = self.columnVectors[rhs]

        # 4. 获取 RHS 唯一值，初始化映射表
        unique_values = set()
        for cluster in pli_lhs:
            for pos in cluster:
                val = column_vectors_rhs[pos]
                if val != self.null_value_id:  # 跳过空值
                    unique_values.add(val)
        rhs_index = {value: 0 for value in unique_values}

        # 5. 构建列联表
        crosstab_list = []
        for cluster in pli_lhs:
            cluster_map = rhs_index.copy()
            for position in cluster:
                val = column_vectors_rhs[position]
                if val == self.null_value_id:  # 跳过空值
                    continue
                cluster_map[val] = cluster_map.get(val, 0) + 1
            if all(count == 0 for count in cluster_map.values()):
                continue
            crosstab_list.append(cluster_map)

        # 6. 转换为二维数组
        crosstab = []
        for cluster_map in crosstab_list:
            crosstab.append([cluster_map.get(key, 0) for key in sorted(rhs_index.keys())])

        return crosstab

    def build_linked_table_new_new(self, lhs, rhs):
        """
        构建列联表（linked table），左部 (lhs) 使用 PLI，右部 (rhs) 使用 columnVectors。
        优化思路：将“簇×值”的计数问题转换为稀疏矩阵聚合，只遍历非零项。

        :param lhs: 左部属性索引列表（单个或多个属性）。
        :param rhs: 右部属性索引（单个属性）。
        :return: NumPy 二维数组，shape=(n_clusters, n_unique_rhs_values)。
        """
        # 1. 计算 LHS 的 PLI 列表
        if len(lhs) == 1:
            pli_lhs = self.columnData[lhs[0]]["PLI"]
        else:
            pli_lhs = self._cross_plis([self.columnData[col]["PLI"] for col in lhs])

        # 2. 取出 RHS 向量
        column_vectors_rhs = self.columnVectors[rhs]

        # 3. 收集所有非空的 RHS 唯一值，并排序以稳定列顺序
        unique_values = sorted({
            column_vectors_rhs[pos]
            for cluster in pli_lhs
            for pos in cluster
            if column_vectors_rhs[pos] != self.null_value_id
        })
        n_vals = len(unique_values)
        # 为每个 RHS 值分配一个列索引
        val2col = {val: idx for idx, val in enumerate(unique_values)}

        # 4. 构造稀疏计数对 (cluster_idx, value_idx)
        rows = []  # 存放簇索引
        cols = []  # 存放 RHS 值的列索引
        for cluster_idx, cluster in enumerate(pli_lhs):
            for pos in cluster:
                val = column_vectors_rhs[pos]
                if val == self.null_value_id:
                    continue
                rows.append(cluster_idx)
                cols.append(val2col[val])
        data = np.ones(len(rows), dtype=int)

        # 5. 用 coo_matrix 在 C 层进行累加
        n_clusters = len(pli_lhs)
        cmat = coo_matrix((data, (rows, cols)),
                          shape=(n_clusters, n_vals),
                          dtype=int)

        # 6. 转为稠密数组，并过滤掉全零的簇
        linked = cmat.toarray()  # shape (n_clusters, n_vals)
        nonzero_mask = linked.sum(axis=1) > 0  # 找到至少有一个非零的簇
        linked = linked[nonzero_mask]

        return linked



    def build_linked_table_new(self, lhs, rhs):
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

        # 2. 加权抽样簇：如果簇的总数大于 50，则根据簇大小的权重抽样 50 个簇，否则全部选取
        if len(pli_lhs) > 50:
            sizes = [len(cluster) for cluster in pli_lhs]
            total_size = sum(sizes)
            # 计算权重，权重之和为 1
            weights = [size / total_size for size in sizes]
            indices = np.random.choice(range(len(pli_lhs)), size=50, replace=False, p=weights)
            sampled_clusters = [pli_lhs[i] for i in indices]
        else:
            sampled_clusters = pli_lhs

        # 3. 对每个簇再进行抽样操作：每个簇中抽取 100 条数据（如果不足 100 条则全部选取）
        sampled_clusters = [random.sample(cluster, 100) if len(cluster) > 100 else cluster for cluster in
                            sampled_clusters]

        # 4. 计算 RHS 的 columnVectors
        column_vectors_rhs = self.columnVectors[rhs]

        # 5. 获取 RHS 唯一值，初始化映射表
        unique_values = set()
        for cluster in sampled_clusters:
            for pos in cluster:
                unique_values.add(column_vectors_rhs[pos])
        rhs_index = {value: 0 for value in unique_values}

        # 6. 构建列联表
        crosstab_list = []
        for cluster in sampled_clusters:
            # 这里对每个簇先复制初始的映射结构
            cluster_map = rhs_index.copy()
            for position in cluster:
                value_rhs = column_vectors_rhs[position]
                cluster_map[value_rhs] = cluster_map.get(value_rhs, 0) + 1
            crosstab_list.append(cluster_map)

        # 7. 转换为二维数组，按 RHS 唯一值字典序排列
        sorted_keys = sorted(rhs_index.keys())
        crosstab = []
        for cluster_map in crosstab_list:
            crosstab.append([cluster_map.get(key, 0) for key in sorted_keys])

        return crosstab

    # todo:精度问题
    def compute_expected_frequencies(self, crosstab):
        """
        根据实际分布频数表计算期望分布频数表。
        :param crosstab: 实际分布频数表（二维列表）。
        :return: 期望分布频数表（二维列表）。
        """
        # row_totals = [sum(row) for row in crosstab]  # 每一行的总和
        # col_totals = [sum(col) for col in zip(*crosstab)]  # 每一列的总和
        # total = sum(row_totals)  # 整个列联表的总和
        #
        # # 构建期望频数表
        # expected_frequencies = [
        #     [(row_total * col_total) / total for col_total in col_totals]
        #     for row_total in row_totals
        # ]
        #
        # return expected_frequencies
        arr = np.array(crosstab, dtype=float)
        row_totals = arr.sum(axis=1)
        col_totals = arr.sum(axis=0)
        total = row_totals.sum()
        # 向量化外积
        return np.outer(row_totals, col_totals) / total

    def _check_expected_frequencies(self, expected_frequencies):
        """
        检查期望分布频数表中是否超过 80% 的格子的期望计数大于 5。
        :param expected_frequencies: 期望分布频数表。
        :return: 如果满足条件，返回 True；否则，返回 False。
        """
        # valid_count = sum(1 for row in expected_frequencies for value in row if value > 5)
        # total_cells = len(expected_frequencies) * len(expected_frequencies[0])
        # return valid_count / total_cells >= 0.8
        arr = np.array(expected_frequencies, dtype=float)
        # 计算大于 5 的比例
        return np.mean(arr > 5) >= 0.8

    def _check_linked_table(self, linked_table, lhs):
        """
        检查列联表是否有效。
        :param linked_table: 列联表（二位列表）。
        :return: 根据检查结果返回 "suspected_fd" 或 "incorporated"
        """
        # 随机抽取 10 行（不足 10 行则全抽）
        if len(linked_table) < 10:
            sampled_rows = linked_table
        else:
            sampled_rows = random.sample(linked_table, 10)
        # 检查每行：判断最大值占整行和是否达到或超过 90%
        for row in sampled_rows:
            row_sum = sum(row)
            # 如果总和为零，则跳过该行检查
            if row_sum == 0:
                continue
            if max(row) / row_sum >= 0.9:
                if len(lhs) == 1:
                    logger.warning("疑似FD，不进行聚类合并")
                    return "suspected_fd"
                else:
                    logger.warning("疑似多左部FD，舍弃")
                    return "suspected_fd_more"

        # 检查列
        n_cols = len(linked_table[0])
        # 如果列数小于10则取所有列，否则随机抽取10列
        if n_cols < 10:
            # 使用 zip(*) 转置数据
            sampled_columns = list(zip(*linked_table))
        else:
            # 随机抽取 10 个列索引
            indices = random.sample(range(n_cols), 10)
            sampled_columns = []
            for j in indices:
                # 构造第 j 列：遍历每一行取第 j 个元素
                col = [row[j] for row in linked_table]
                sampled_columns.append(col)
        # 检查每列是否有 90% 以上的占比
        for col in sampled_columns:
            col_sum = sum(col)
            if col_sum == 0:
                continue  # 避免除以零
            if max(col) / col_sum >= 0.9:
                if len(lhs) == 1:
                    logger.warning("疑似FD，不进行聚类合并")
                    return "suspected_fd"
                else:
                    logger.warning("疑似多右部FD，舍弃")
                    return "suspected_fd_more"

        return "incorporated"

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

    def check_dependency_direction(self, rhs_column, lhs_columns, linked_table):
        """
        检查函数依赖方向（加权平均比较行列方向）
        :param lhs_columns: 左部属性列表（潜在决定因素）
        :param rhs_column: 右部属性（被决定属性）
        :param linked_table: 预先构建的列联表
        :return: 方向判断结果字符串
        """
        total = sum(sum(row) for row in linked_table)
        if total == 0:
            logger.warning("列联表总数为零")
            return "invalid"

        # 计算行方向（左→右）加权平均
        row_avg = sum(
            (max(row) / row_sum) * (row_sum / total)
            for row in linked_table
            if (row_sum := sum(row)) > 0
        )

        # 计算列方向（右→左）加权平均
        col_avg = sum(
            (max(col) / col_sum) * (col_sum / total)
            for col in zip(*linked_table)  # 转置列联表
            if (col_sum := sum(col)) > 0
        )

        logger.info(f"行方向得分: {row_avg:.2f}, 列方向得分: {col_avg:.2f}")

        # 判断依赖方向
        avg = (row_avg + col_avg) / 2
        if abs(row_avg - col_avg) < 0.003 * avg:
        # if abs(row_avg - col_avg) < 1e-6:  # 浮点数相等判断
            return "mutual"
        elif row_avg > col_avg:
            return "left_to_right"
        else:
            return "right_to_left"
    @profile
    # kernprof -l -v Config.py
    def compute_correlation(self, column_a, column_b):
        """
        计算两个列之间的相关性。
        :param column_a: 列 A 的索引。
        :param column_b: 列 B 的索引或组合。
        :return: 相关性值。
        """
        # 构建列联表
        linked_table = self.build_linked_table(column_b, column_a)
        # 打印归并前列联表维度

        if not linked_table or len(linked_table[0]) == 0:
            logger.warning("列联表为空，无法计算相关性。")
            return "invalid"
        logger.info(f"归并前列联表维度: {len(linked_table)} 行, {len(linked_table[0])} 列")

        expected_frequencies = self.compute_expected_frequencies(linked_table)
        # 检查期望分布频数表
        if not self._check_expected_frequencies(expected_frequencies):
            logger.warning("期望分布频数表不符合要求，无法计算相关性。")
            result = self._check_linked_table(linked_table, column_b)
            if result == "incorporated":
                total = sum(sum(row) for row in linked_table)
                # 计算 χ²
                chi_squared = 0
                for i, row in enumerate(linked_table):
                    for j, observed in enumerate(row):
                        expected = expected_frequencies[i][j]
                        if expected > 0:
                            chi_squared += ((observed - expected) ** 2) / expected
                d1, d2 = len(linked_table), len(linked_table[0])
                d = min(d1, d2)
                # 打印 χ²
                logger.info(f"归并前计算列 {column_a} 和列 {column_b} 之间的 χ²: {chi_squared}")
                if d <= 1:
                    logger.warning("自由度为零，设置 φ² 为 0。")
                    return "invalid"

                phi_squared = chi_squared / (total * (d - 1))

                N = total
                M = len(linked_table[0])
                K = len(linked_table)

                # 计算
                expected_phi = ((M - 1) * (K - 1)) / ((N - 1) * (d - 1))

                if expected_phi == 1:
                    normalized_phi_squared = 0
                else:
                    normalized_phi_squared = (phi_squared - expected_phi) / (1 - expected_phi)

                normalized_phi_squared_plus = max(normalized_phi_squared, 0)

                column_a_name = self._get_column_name(column_a)
                column_b_names = [self._get_column_name(col) for col in column_b]
                logger.info(
                    f"归并前计算列 {column_a_name} 和列 {column_b_names} 之间的相关性 (φ²): {normalized_phi_squared_plus}")

                incorporate = Incorporate()
                linked_table = incorporate.merge_tables(linked_table)
                # 打印归并后列联表维度
                logger.info(f"归并后列联表维度: {len(linked_table)} 行, {len(linked_table[0])} 列")
                expected_frequencies = self.compute_expected_frequencies(linked_table)

                # 总观测数
                total = sum(sum(row) for row in linked_table)
                if total == 0:
                    logger.warning("总观测数为零，设置 φ² 为 0。")
                    return "invalid"

                # 计算 χ²
                chi_squared = 0
                for i, row in enumerate(linked_table):
                    for j, observed in enumerate(row):
                        expected = expected_frequencies[i][j]
                        if expected > 0:
                            chi_squared += ((observed - expected) ** 2) / expected

                # 计算 φ²
                d1, d2 = len(linked_table), len(linked_table[0])
                d = min(d1, d2)
                # 打印 χ²
                logger.info(f"归并后计算列 {column_a} 和列 {column_b} 之间的 χ²: {chi_squared}")
                if d <= 1:
                    logger.warning("自由度为零，设置 φ² 为 0。")
                    return "invalid"

                phi_squared = chi_squared / (total * (d - 1))

                logger.info(f"归并后计算列 {column_a_name} 和列 {column_b_names} 之间的未修正相关性 (φ²): {phi_squared}")

                if expected_phi == 1:
                    normalized_phi_squared = 0
                else:
                    normalized_phi_squared = (phi_squared - expected_phi) / (1 - expected_phi)

                normalized_phi_squared_plus = max(normalized_phi_squared, 0)

                column_a_name = self._get_column_name(column_a)
                column_b_names = [self._get_column_name(col) for col in column_b]
                logger.info(
                    f"计算列 {column_a_name} 和列 {column_b_names} 之间的相关性 (φ²): {normalized_phi_squared_plus}")

                if len(column_b) == 1:
                    # 添加方向检查逻辑
                    if normalized_phi_squared_plus >= self.upper_threshold:
                        direction_check = self.check_dependency_direction(rhs_column=column_a, lhs_columns=column_b,
                                                                          linked_table=linked_table)
                        return direction_check
                    elif normalized_phi_squared_plus < self.lower_threshold:
                        return "invalid"
                    else:
                        return "pending"
                else:
                    if normalized_phi_squared_plus >= self.upper_threshold:
                        return True
                    elif normalized_phi_squared_plus < self.lower_threshold:
                        return "invalid"
                    else:
                        return "pending"
            elif result == "suspected_fd_more":
                return False
        else:
            logger.info("期望分布频数表符合要求，开始计算相关性。")

        # 总观测数
        total = sum(sum(row) for row in linked_table)
        if total == 0:
            logger.warning("总观测数为零，设置 φ² 为 0。")
            return "invalid"

        # # 计算 χ²
        # chi_squared = 0
        # for i, row in enumerate(linked_table):
        #     for j, observed in enumerate(row):
        #         expected = expected_frequencies[i][j]
        #         if expected > 0:
        #             chi_squared += ((observed - expected) ** 2) / expected
        #
        # logger.info(f"111计算列 {column_a} 和列 {column_b} 之间的 χ²: {chi_squared}")

        # 先将 Python 嵌套列表转换为稀疏矩阵
        arr = np.array(linked_table, dtype=float)
        row_sum = arr.sum(axis=1)
        col_sum = arr.sum(axis=0)
        T = row_sum.sum()

        # 构造 coo_matrix，只遍历非零项
        sparse = coo_matrix(arr)
        # 计算期望：E_ij = row_sum[i] * col_sum[j] / T
        # 对于每一个 (i,j,o) in sparse:
        #   contrib = o^2 / E_ij
        chi2_num = 0.0
        for i, j, o in zip(sparse.row, sparse.col, sparse.data):
            E = row_sum[i] * col_sum[j] / T
            # E>0 保证
            chi2_num += (o * o) / E

        # 最终 χ²
        chi_squared = chi2_num - T

        logger.info(f"111计算列 {column_a} 和列 {column_b} 之间的 χ²: {chi_squared}")

        # 计算 φ²
        d1, d2 = len(linked_table), len(linked_table[0])
        d = min(d1, d2)
        if d <= 1:
            logger.warning("自由度为零，设置 φ² 为 0。")
            return "invalid"

        phi_squared = chi_squared / (total * (d - 1))

        N = total
        M = len(linked_table[0])
        K = len(linked_table)

        # 计算
        expected_phi = ((M - 1) * (K - 1)) / ((N - 1) * (d - 1))

        if expected_phi == 1:
            normalized_phi_squared = 0
        else:
            normalized_phi_squared = (phi_squared - expected_phi) / (1 - expected_phi)

        normalized_phi_squared_plus = max(normalized_phi_squared, 0)

        column_a_name = self._get_column_name(column_a)
        column_b_names = [self._get_column_name(col) for col in column_b]
        logger.info(f"计算列 {column_a_name} 和列 {column_b_names} 之间的未修正相关性 (φ²): {phi_squared}")
        logger.info(f"计算列 {column_a_name} 和列 {column_b_names} 之间的相关性 (φ²): {normalized_phi_squared_plus}")

        if len(column_b) == 1:
            # 添加方向检查逻辑
            if normalized_phi_squared_plus >= self.upper_threshold:
                direction_check = self.check_dependency_direction(rhs_column=column_a, lhs_columns=column_b,
                                                                  linked_table=linked_table)
                return direction_check
            elif normalized_phi_squared_plus < self.lower_threshold:
                return "invalid"
            else:
                return "pending"
        else:
            if normalized_phi_squared_plus >= self.upper_threshold:
                return True
            elif normalized_phi_squared_plus < self.lower_threshold:
                return "invalid"
            else:
                return "pending"
