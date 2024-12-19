# TODO: 获取到的data比较臃肿，是否应该只要部分数据，对速度会有影响吗？
# TODO: java代码中忽略了单个的簇，是否需要借鉴？
class CorrelationCalculator:
    """
    相关性计算工具类。
    用于封装不同的相关性计算方法，以便在项目中复用。
    """

    def __init__(self, data):
        """
        初始化相关性计算器。
        :param data: 输入数据字典，包括原始数据和 PLI 索引。
                     data.data 是原始数据，data.pli_indices 是列的 PLI 索引。
        """
        self.raw_data = data.data
        self.pli_indices = data.pli_indices

    def build_linked_table(self, column_a, column_b):
        """
        利用 PLI 索引构建列连表（linked table）。
        :param column_a: 列 A 的索引。
        :param column_b: 列 B 的索引或组合（可以是单个索引，也可以是集合）。
        :return: 一个包含映射关系的字典，表示列连表。
        """
        # 如果 column_b 是单个索引，将其转为集合
        if isinstance(column_b, int):
            column_b = {column_b}

        # 获取 PLI 数据
        pli_a = self.pli_indices[column_a]

        if len(column_b) == 1:
            # 如果只有一个列，直接获取其 PLI
            pli_b = self.pli_indices[next(iter(column_b))]
        else:
            # 对多个列的 PLI 进行交叉，形成新 PLI
            pli_b = self._cross_pli([self.pli_indices[col] for col in column_b])

        # 构建列连表
        linked_table = {}
        for block_a in pli_a:
            for block_b in pli_b:
                # 交集是 A 的块与 B 的块重叠部分
                intersection = set(block_a) & set(block_b)
                if intersection:
                    linked_table[tuple(block_a)] = linked_table.get(tuple(block_a), set()).union(intersection)

        return linked_table

    def _cross_pli(self, pli_list):
        """
        对多个 PLI 进行交叉，生成新 PLI。
        :param pli_list: 列表，每个元素是一个列的 PLI。
        :return: 交叉后的新 PLI，形式为列表嵌套列表。
        """
        if not pli_list:
            return []

        # 初始交叉结果为第一个列的 PLI
        result = [list(block) for block in pli_list[0]]

        # 逐列交叉
        for pli in pli_list[1:]:
            new_result = []
            for block_a in result:
                for block_b in pli:
                    intersection = set(block_a) & set(block_b)
                    if intersection:
                        new_result.append(list(intersection))
            result = new_result

        return result

    def compute_correlation(self, column_a, column_b):
        """
        计算两个列之间的相关性。
        :param column_a: 列 A 的索引。
        :param column_b: 列 B 的索引或组合。
        :return: 相关性值。
        """
        # 构建列连表
        linked_table = self.build_linked_table(column_a, column_b)
        if isinstance(column_b, (set, frozenset)) and 1 in column_b:
            return 0.9
        # TODO: 添加实际相关性计算逻辑，使用例如皮尔逊相关系数或 Cramér's V 等方法。
        return 0.5  # 示例返回值
