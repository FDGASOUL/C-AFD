# TODO: java代码中忽略了单个的簇，是否需要借鉴？
# TODO: java代码中采用了列表逐层的方式，是否需要借鉴？
# TODO: 使用data数据后，速度很慢，具体慢在哪里？计算pli耗费时间不多，为每个搜索空间设置上下文环境用了一半时间,具体来说add_combination添加属性组合到依赖树中占据大量时间
# TODO: φ²的计算公式需不需要除d-1？GPT:如果希望最后的结果除以 (d−1)(d−1)，其实是将 φ² 的计算转换为 Cramér's V 的一种变形（或与之相关的指标）。这可能是为了便于在不同维度的数据中比较关联强度。
class CorrelationCalculator:
    """
    相关性计算工具类。
    用于封装不同的相关性计算方法，以便在项目中复用。
    """

    def __init__(self, column_layout_data):
        """
        初始化相关性计算器。
        :param column_layout_data: ColumnLayoutRelationData 的实例，包含列的数值索引和列数据。
        """
        self.columnVectors = column_layout_data.get_column_vectors()
        self.columnData = column_layout_data.get_column_data()

    def build_linked_table(self, column_a, column_b):
        """
        利用 Probing Table 构建列连表（linked table）。
        :param column_a: 列 A 的索引。
        :param column_b: 列 B 的索引或组合（可以是单个索引，也可以是集合）。
        :return: 一个包含映射关系的字典，表示列连表。
        """
        if isinstance(column_b, int):
            column_b = {column_b}

        pli_a = self.columnData[column_a]["PLI"]

        if len(column_b) == 1:
            # 如果只有一个列，直接获取其 columnVectors
            column_vectors_b = self.columnVectors[list(column_b)[0]]
        else:
            # 对多个列的 columnVectors 进行交叉，形成新 columnVectors
            column_vectors_b = self._cross_column_vectors([self.columnVectors[col] for col in column_b])

        # 初始化 column_b 索引
        lhs_index = {value: 0 for value in set(column_vectors_b) if value != -1}

        # 填充列联表
        crosstab_list = []
        for cluster in pli_a:
            cluster_map = lhs_index.copy()
            for position in cluster:
                value_b = column_vectors_b[position]
                if value_b != -1:
                    cluster_map[value_b] = cluster_map.get(value_b, 0) + 1
            crosstab_list.append(cluster_map)

        # 转换为二维数组
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

    def _cross_column_vectors(self, vectors_list):
        """
        对多个列的 columnVectors 进行交叉，生成新 Vectors。
        :param vectors_list: 列表，每个元素是一个列的 Vectors。
        :return: 交叉后的新 Vectors。
        """
        if not vectors_list:
            return []

        num_rows = len(vectors_list[0])
        cross_vectors = [-1] * num_rows
        cluster_map = {}
        next_cluster_id = 1

        for row_index in range(num_rows):
            cluster_key = tuple(vectors[row_index] for vectors in vectors_list)
            if -1 in cluster_key:
                continue  # 跳过空值

            if cluster_key not in cluster_map:
                cluster_map[cluster_key] = next_cluster_id
                next_cluster_id += 1

            cross_vectors[row_index] = cluster_map[cluster_key]

        return cross_vectors

    def compute_correlation(self, column_a, column_b):
        """
        计算两个列之间的相关性。
        :param column_a: 列 A 的索引。
        :param column_b: 列 B 的索引或组合。
        :return: 相关性值。
        """
        # 构建列连表
        linked_table = self.build_linked_table(column_a, column_b)
        expected_frequencies = self.compute_expected_frequencies(linked_table)

        # 检查期望分布频数表
        if not self._check_expected_frequencies(expected_frequencies):
            raise ValueError("期望分布频数表中超过 20% 的格子期望计数未大于 5，需要进行归并操作。")

        # 总观测数
        total = sum(sum(row) for row in linked_table)

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
        phi_squared = chi_squared / (total * (d - 1))

        # print(f"计算列 {column_a} 和列 {list(column_b)} 之间的相关性 (φ²): {phi_squared}")
        return phi_squared


