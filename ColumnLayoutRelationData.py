import pandas as pd
from collections import defaultdict


#TODO: 有index没有table，Vectors是什么？
class ColumnLayoutRelationData:
    def __init__(self, data):
        """
        初始化 ColumnLayoutRelationData 类，构建列布局关系数据。
        :param data: pandas DataFrame，包含抽样后的数据。
        """
        self.data = data
        self.pli_indices = self._compute_pli()

    def _compute_pli(self):
        """
        计算每列的 Position List Index (PLI)。
        :return: 字典，每列名对应的 PLI。
        """
        pli_indices = {}
        for column in self.data.columns:
            equivalence_classes = defaultdict(list)
            for index, value in self.data[column].items():  # 修改 iteritems() 为 items()
                equivalence_classes[value].append(index)
            pli_indices[column] = list(equivalence_classes.values())
        return pli_indices

    def get_pli(self, column_name):
        """
        获取指定列的 PLI。
        :param column_name: 列名。
        :return: 指定列的 PLI。
        """
        if column_name not in self.pli_indices:
            raise ValueError(f"Column '{column_name}' does not exist in the data.")
        return self.pli_indices[column_name]

    def get_all_pli(self):
        """
        获取所有列的 PLI。
        :return: 包含所有列的 PLI 的字典。
        """
        return self.pli_indices

    def get_schema(self):
        """
        获取数据的列名（模式）。
        :return: 数据的列名列表。
        """
        return list(self.data.columns)

    def num_rows(self):
        """
        获取数据的行数。
        :return: 数据的行数。
        """
        return len(self.data)

    def num_columns(self):
        """
        获取数据的列数。
        :return: 数据的列数。
        """
        return len(self.data.columns)
