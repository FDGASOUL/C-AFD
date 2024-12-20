import pandas as pd
from collections import defaultdict


# TODO: 对于原本就是整数的列，是否需要转换columnVectors数字标识符？
# TODO: probing_table是否能用上，是否要去除单个元素的cluster？单例簇对于依赖发现来说没有实际意义（一个值不可能约束其他值），Probing Table 通过排除单例簇，降低了计算复杂度。
class ColumnLayoutRelationData:
    def __init__(self, data):
        """
        初始化 ColumnLayoutRelationData 类，构建列布局关系数据。
        :param data: pandas DataFrame，包含抽样后的数据。
        """
        self.schema = list(data.columns)  # 模式：列名列表
        self.columnVectors = self._compute_column_vectors(data)  # 每列的数值索引表示
        self.columnData = self._compute_column_data()  # 包含列的元信息、PLI 和 Probing Table

    def _compute_column_vectors(self, data):
        """
        将原始数据表中的字符串值转化为整数值数组表示。
        :param data: pandas DataFrame，包含抽样后的数据。
        :return: List[IntList]，每列的数值索引。
        """
        column_vectors = []
        value_to_id = defaultdict(lambda: len(value_to_id))
        nullValueId = -1

        for column in data.columns:
            column_vector = []
            for value in data[column]:
                if pd.isnull(value):
                    column_vector.append(nullValueId)
                else:
                    column_vector.append(value_to_id[value])
            column_vectors.append(column_vector)
        return column_vectors

    def _compute_column_data(self):
        """
        计算每列的 PLI 和 Probing Table。

        :return: 列的元信息，包括 PLI 和 Probing Table。
        """
        column_data = []
        nullValueId = 0

        for col_vector in self.columnVectors:
            equivalence_classes = defaultdict(list)
            for index, value in enumerate(col_vector):
                equivalence_classes[value].append(index)

            pli = list(equivalence_classes.values())
            probing_table = [nullValueId] * len(col_vector)  # 初始化 Probing Table
            next_cluster_id = 1  # 每列的簇编号从 1 开始

            for cluster in pli:
                for position in cluster:
                    probing_table[position] = next_cluster_id
                next_cluster_id += 1

            column_data.append({
                "PLI": pli,
                "ProbingTable": probing_table
            })

        return column_data

    def get_schema(self):
        """
        获取数据的列名（模式）。
        :return: 数据的列名列表。
        """
        return self.schema

    def get_column_vectors(self):
        """
        获取列的数值索引表示。
        :return: List[IntList]，每列的数值索引。
        """
        return self.columnVectors

    def get_column_data(self):
        """
        获取列的元信息，包括 PLI 和 Probing Table。
        :return: List[Dict]，列的元信息。
        """
        return self.columnData

    def num_rows(self):
        """
        获取数据的行数。
        :return: 数据的行数。
        """
        return len(self.columnVectors[0]) if self.columnVectors else 0

    def num_columns(self):
        """
        获取数据的列数。
        :return: 数据的列数。
        """
        return len(self.columnVectors)
