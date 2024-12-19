import pandas as pd
from collections import defaultdict


class ColumnLayoutRelationData:
    def __init__(self, data):
        """
        初始化 ColumnLayoutRelationData 类，构建列布局关系数据。
        :param data: pandas DataFrame，包含抽样后的数据。
        """
        self.schema = list(data.columns)  # 模式：列名列表
        self.columnTypes = self._infer_column_types(data)  # 每列的数据类型元信息
        self.columnVectors = self._compute_column_vectors(data)  # 每列的数值索引表示
        self.columnData = self._compute_column_data()  # 包含列的元信息、PLI 和 Probing Table

    def _infer_column_types(self, data):
        """
        推断每列的数据类型。
        :param data: pandas DataFrame，包含抽样后的数据。
        :return: Dict，列名到数据类型的映射。
        """
        column_types = {}
        for column in data.columns:
            column_types[column] = data[column].dtype
        return column_types

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

        for col_vector in self.columnVectors:
            equivalence_classes = defaultdict(list)
            for index, value in enumerate(col_vector):
                equivalence_classes[value].append(index)

            pli = list(equivalence_classes.values())
            probing_table = [cluster for cluster in pli if len(cluster) > 1]

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

    def get_column_types(self):
        """
        获取数据的列类型元信息。
        :return: Dict，列名到数据类型的映射。
        """
        return self.columnTypes

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
