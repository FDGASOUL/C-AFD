import random
from collections import defaultdict
from typing import List

class ColumnData:
    def __init__(self, column, probing_table, pli):
        self.column = column
        self.probing_table = probing_table
        self.pli = pli

    def get_probing_table(self):
        return self.probing_table

    def get_column(self):
        return self.column


class Column:
    def __init__(self, schema, name, index):
        self.schema = schema
        self.name = name
        self.index = index

    def get_index(self):
        return self.index


class PositionListIndex:
    @staticmethod
    def create_for(column_vector, is_null_equal_null):
        # Simplified placeholder implementation of PLI creation
        clusters = defaultdict(list)
        for i, value in enumerate(column_vector):
            clusters[value].append(i)
        probing_table = [value for value, cluster in clusters.items() if len(cluster) > 1]
        return PositionListIndex(probing_table)

    def __init__(self, probing_table):
        self.probing_table = probing_table

    def get_probing_table(self, unique=False):
        return self.probing_table


class RelationSchema:
    def __init__(self, relation_name, is_null_equal_null):
        self.relation_name = relation_name
        self.is_null_equal_null = is_null_equal_null
        self.columns = []

    def get_num_columns(self):
        return len(self.columns)


class ColumnLayoutRelationData:
    def __init__(self, schema, column_data, column_vectors):
        self.schema = schema
        self.column_data = column_data
        self.column_vectors = column_vectors

    @classmethod
    def create_from(cls, file_input_generator, is_null_equal_null, max_cols, max_rows):
        relational_input = file_input_generator.generate_new_copy()
        schema = RelationSchema(relational_input.relation_name(), is_null_equal_null)

        value_dictionary = {}
        unknown_value_id = 0
        next_value_id = 1
        null_value_id = -1

        num_columns = relational_input.number_of_columns()
        if max_cols > 0:
            num_columns = min(num_columns, max_cols)

        column_vectors = [[] for _ in range(num_columns)]

        row_num = 0
        random.seed(23)

        while relational_input.has_next():
            row = relational_input.next()
            if max_rows <= 0 or row_num < max_rows:
                for index, field in enumerate(row[:num_columns]):
                    if field is None:
                        column_vectors[index].append(null_value_id)
                    else:
                        if field not in value_dictionary:
                            value_dictionary[field] = next_value_id
                            next_value_id += 1
                        column_vectors[index].append(value_dictionary[field])
            else:
                position = random.randint(0, row_num)
                if position < max_rows:
                    for index, field in enumerate(row[:num_columns]):
                        if field is None:
                            column_vectors[index][position] = null_value_id
                        else:
                            if field not in value_dictionary:
                                value_dictionary[field] = next_value_id
                                next_value_id += 1
                            column_vectors[index][position] = value_dictionary[field]
            row_num += 1

        column_data_list = []
        for index, vector in enumerate(column_vectors):
            column = Column(schema, relational_input.column_names()[index], index)
            pli = PositionListIndex.create_for(vector, schema.is_null_equal_null)
            column_data = ColumnData(column, pli.get_probing_table(True), pli)
            column_data_list.append(column)

        column_data = column_data_list
        schema.columns = column_data_list

        return cls(schema, column_data, column_vectors)

    def get_column_data(self):
        return self.column_data

    def get_column_data_by_index(self, column_index):
        return self.column_data[column_index]

    def get_column_vectors(self):
        return self.column_vectors

    def get_num_rows(self):
        return len(self.column_data[0].get_probing_table())

    def get_tuple(self, tuple_index):
        return [
            self.column_data[column_index].get_probing_table()[tuple_index]
            for column_index in range(self.schema.get_num_columns())
        ]
