import logging
from collections import defaultdict
from typing import List, Dict, Any

import pandas as pd

# 获取日志实例
logger = logging.getLogger(__name__)


class ColumnLayoutRelationData:
    """
    列布局关系数据处理器。

    将清洗后的 DataFrame 转换为整型索引形式，并提取列级特征（如 PLI）。

    Attributes:
        schema (List[str]): 数据模式（列名列表）。
        null_value_id (int): 空值标识符。
        column_vectors (List[List[int]]): 列向量集合，每列用整型索引表示。
        column_data (List[Dict[str, Any]]): 列特征数据集合，包含 PLI 等。
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        初始化 ColumnLayoutRelationData。

        :param data: 清洗后的原始数据，pandas DataFrame。
        """
        self.schema: List[str] = list(data.columns)
        self.null_value_id: int = -1
        self.column_vectors: List[List[int]] = self._compute_column_vectors(data)
        self.column_data: List[Dict[str, Any]] = self._compute_column_data()

    def _compute_column_vectors(self, data: pd.DataFrame) -> List[List[int]]:
        """
        将原始数据转换为整型索引表示。

        每列为独立映射：
            1. 非空值分配唯一整型 ID。
            2. 空值标记为 null_value_id。

        :param data: 待转换的 DataFrame。
        :return: 二维列表形式的列向量集合。
        """
        column_vectors: List[List[int]] = []
        for column in data.columns:
            value_registry: Dict[Any, int] = {}
            next_id = 0
            vector: List[int] = []
            for value in data[column]:
                if pd.isnull(value):
                    vector.append(self.null_value_id)
                else:
                    if value not in value_registry:
                        value_registry[value] = next_id
                        next_id += 1
                    vector.append(value_registry[value])
            column_vectors.append(vector)
        return column_vectors

    def _compute_column_data(self) -> List[Dict[str, Any]]:
        """
        计算列特征数据（目前仅 PLI）。

        PLI 计算规则：
            1. 排除空值对应位置。
            2. 排除单元素簇（长度<=1）。

        :return: 每列特征数据字典列表。
        """
        feature_collection: List[Dict[str, Any]] = []
        for vector in self.column_vectors:
            positions: Dict[int, List[int]] = defaultdict(list)
            for idx, val in enumerate(vector):
                positions[val].append(idx)
            valid_clusters: List[List[int]] = [pos_list for val, pos_list in positions.items()
                                               if val != self.null_value_id and len(pos_list) > 1]
            feature_collection.append({"PLI": valid_clusters})
        return feature_collection

    def get_schema(self) -> List[str]:
        """
        获取数据模式（列名列表）。

        :return: 列名列表。
        """
        return self.schema

    def get_column_vectors(self) -> List[List[int]]:
        """
        获取列向量表示。

        :return: 列向量集合。
        """
        return self.column_vectors

    def get_column_data(self) -> List[Dict[str, Any]]:
        """
        获取列特征数据。

        :return: 列特征数据集合。
        """
        return self.column_data

    def num_columns(self) -> int:
        """
        获取列数。

        :return: 列的数量。
        """
        return len(self.column_vectors)

    def get_null_value_id(self) -> int:
        """
        获取空值标识符。

        :return: 空值标识符（int）。
        """
        return self.null_value_id
