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
        self.columnVectors = self._compute_column_vectors(data)  # 每列的数值索引表示
        self.columnData = self._compute_column_data()  # 包含列的元信息、PLI 和 Probing Table
        # self.testData = data

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

            # 获取 PLI，排除单列簇（只包含一个元素的簇）
            pli = [cluster for cluster in equivalence_classes.values() if len(cluster) > 1]

            # 初始化 Probing Table，所有位置初始为 0
            probing_table = [nullValueId] * len(col_vector)
            next_cluster_id = 1  # 非单列簇的簇编号从 1 开始

            # 为非单列簇赋值
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

    def num_columns(self):
        """
        获取数据的列数。
        :return: 数据的列数。
        """
        return len(self.columnVectors)
