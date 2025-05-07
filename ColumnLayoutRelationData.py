import pandas as pd
from collections import defaultdict
import logging

# 获取日志实例
logger = logging.getLogger(__name__)


# TODO: 通过排除单例簇，降低了计算复杂度，同时解决了index的问题。目前只排除PLI中的单列簇，构造交叉表时对RHS使用PLI，而LHS使用Vectors，是否合理，应该是LHS的单列簇没有用
# TODO: 对数字可以不计算vectors
class ColumnLayoutRelationData:
    def __init__(self, data):
        """
        初始化 ColumnLayoutRelationData 类，构建列布局关系数据。
        :param data: pandas DataFrame，包含抽样后的数据。
        """
        self.schema = list(data.columns)  # 模式：列名列表
        self.null_value_id = -1
        self.columnVectors = self._compute_column_vectors(data)  # 每列的数值索引表示
        self.columnData = self._compute_column_data()  # 包含列的元信息、PLI 和 Probing Table

    def _compute_column_vectors(self, data):
        """
        将原始数据表中的值转化为整数索引表示。
        空值使用 self.null_value_id 标识（-1）。
        :param data: pandas DataFrame
        :return: List[List[int]]
        """
        column_vectors = []
        # value_to_id 每遇到一个新的非空值，就分配一个新的整数 id
        value_to_id = defaultdict(lambda: len(value_to_id))

        for col in data.columns:
            vec = []
            for val in data[col]:
                if pd.isnull(val):
                    vec.append(self.null_value_id)
                else:
                    vec.append(value_to_id[val])
            column_vectors.append(vec)

        return column_vectors

    def _compute_column_data(self):
        """
        计算每列的 PLI（排除单例簇且不含空值簇）。
        :return: List[Dict]，每个 dict 仅包含 "PLI"
        """
        column_data = []

        for vec in self.columnVectors:
            eq_classes = defaultdict(list)
            # 构建等价类
            for idx, v in enumerate(vec):
                eq_classes[v].append(idx)

            # 排除空值簇（key == self.null_value_id），以及单例簇（len<=1）
            pli = [
                cluster
                for val, cluster in eq_classes.items()
                if val != self.null_value_id and len(cluster) > 1
            ]

            column_data.append({"PLI": pli})

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

    def num_columns(self):
        """
        获取数据的列数。
        :return: 数据的列数。
        """
        return len(self.columnVectors)

    def get_null_value_id(self):
        return self.null_value_id
