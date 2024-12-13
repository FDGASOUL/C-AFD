import csv
import random
from collections import defaultdict


class ColumnLayoutRelationData:
    """
    列布局关系数据类，包含属性向量（列数据）和相关的架构信息。
    """

    def __init__(self, schema, column_data, column_vectors):
        self.schema = schema  # 存储列的模式信息
        self.column_data = column_data  # 列数据对象
        self.column_vectors = column_vectors  # 属性向量

    @staticmethod
    def create_from(file_input_generator, is_null_equal_null, max_cols, max_rows):
        """
        从文件输入生成器创建 ColumnLayoutRelationData 实例。

        :param file_input_generator: 文件输入生成器对象
        :param is_null_equal_null: 是否认为两个不同的 NULL 值相等
        :param max_cols: 最大列数
        :param max_rows: 最大行数
        :return: ColumnLayoutRelationData 实例
        """

        # 读取数据为df
        relational_input = file_input_generator

        # 创建模式
        # schema = RelationSchema(file_input_generator.relation_name, is_null_equal_null)

        # 准备数值索引
        value_dictionary = defaultdict(lambda: 0)  # 未知值默认为 0
        unknown_value_id = 0
        next_value_id = 1

        # 准备列向量
        num_columns = relational_input.number_of_columns
        if max_cols > 0:
            num_columns = min(num_columns, max_cols)

        column_vectors = [[] for _ in range(num_columns)]  # 初始化列向量列表

        # 填充列向量
        row_num = 0
        random.seed(23)  # 随机数种子
        for row in relational_input:  # 遍历所有行
            if max_rows <= 0 or row_num < max_rows:  # 未达到最大行限制
                for index, field in enumerate(row[:num_columns]):
                    if field is None:  # 空值处理
                        column_vectors[index].append(-1)
                    else:
                        if field not in value_dictionary:
                            value_dictionary[field] = next_value_id
                            next_value_id += 1
                        column_vectors[index].append(value_dictionary[field])
            else:  # 超过最大行限制，随机替换
                position = random.randint(0, row_num)
                if position < max_rows:
                    for index, field in enumerate(row[:num_columns]):
                        if field is None:
                            column_vectors[index][position] = -1
                        else:
                            if field not in value_dictionary:
                                value_dictionary[field] = next_value_id
                                next_value_id += 1
                            column_vectors[index][position] = value_dictionary[field]
            row_num += 1

        # 清空数值索引
        value_dictionary = None

        # 创建列数据对象
        column_data_list = []
        for index, vector in enumerate(column_vectors):
            column = Column(schema, relational_input.column_names[index], index)
            pli = PositionListIndex.create_for(vector, is_null_equal_null)
            probing_table = pli.get_probing_table(True)
            column_data = ColumnData(column, probing_table, pli)
            column_data_list.append(column)

        # 构建 ColumnLayoutRelationData 对象
        column_data_array = [None] * len(column_vectors)
        for data in column_data_list:
            schema.columns.append(data.column)  # 添加到架构中
            column_data_array[data.column.index] = data

        return ColumnLayoutRelationData(schema, column_data_array, column_vectors)

    def get_column_data(self):
        """
        返回所有列数据对象。
        """
        return self.column_data

    def get_column_data_by_index(self, column_index):
        """
        根据索引返回列数据对象。
        """
        return self.column_data[column_index]

    def get_column_vectors(self):
        """
        返回列向量。
        """
        return self.column_vectors

    def get_num_rows(self):
        """
        返回行数。
        """
        return len(self.column_data[0].probing_table)

    def get_tuple(self, tuple_index):
        """
        返回指定索引处的元组（行）。
        """
        num_columns = self.schema.get_num_columns()
        return [self.column_data[i].probing_table[tuple_index] for i in range(num_columns)]

        # 获取某列的 PLI
        column_name = 3
        pli = relation_data.get_pli(column_name)
        print(f"PLI for column '{column_name}': {pli}")

        # 获取所有列的 PLI
        all_pli = relation_data.get_all_pli()
        print("All PLIs:", all_pli)

        # 查看数据模式
        schema = relation_data.get_schema()
        print("Schema:", schema)

        # 获取行数和列数
        print("Number of rows:", relation_data.num_rows())
        print("Number of columns:", relation_data.num_columns())
