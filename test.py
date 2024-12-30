import math


class Incorporate:
    """
    归并工具类。
    """

    @staticmethod
    def are_rows_similar(row1, row2, similarity_threshold):
        """
        判断两行是否分布相似，使用 KL 散度。
        :param row1: 第一行。
        :param row2: 第二行。
        :param similarity_threshold: 相似度阈值（KL 散度越小越相似）。
        :return: 布尔值，表示两行是否相似。
        """
        divergence = Incorporate._kl_divergence(row1, row2)
        return divergence

    @staticmethod
    def are_columns_similar(table, col1, col2, similarity_threshold):
        """
        判断两列是否分布相似，使用 KL 散度。
        :param table: 列联表。
        :param col1: 第一列索引。
        :param col2: 第二列索引。
        :param similarity_threshold: 相似度阈值（KL 散度越小越相似）。
        :return: 布尔值，表示两列是否相似。
        """
        col1_values = [row[col1] for row in table]
        col2_values = [row[col2] for row in table]
        divergence = Incorporate._kl_divergence(col1_values, col2_values)
        return divergence <= similarity_threshold

    @staticmethod
    def _kl_divergence(vec1, vec2):
        """
        计算两个概率分布之间的 KL 散度。
        :param vec1: 第一个分布（向量）。
        :param vec2: 第二个分布（向量）。
        :return: KL 散度值。
        """
        # 转换为概率分布
        sum1 = sum(vec1)
        sum2 = sum(vec2)
        if sum1 == 0 or sum2 == 0:
            raise ValueError("向量和不能为零。")
        prob1 = [x / sum1 for x in vec1]
        prob2 = [y / sum2 for y in vec2]

        # 避免出现 log(0) 的问题，添加一个很小的平滑值
        epsilon = 1e-10
        prob1 = [p + epsilon for p in prob1]
        prob2 = [q + epsilon for q in prob2]

        # 计算 KL 散度
        divergence = sum(p * math.log(p / q) for p, q in zip(prob1, prob2))
        return divergence


row1 = [10, 20, 30]
row2 = [2, 4, 6]
threshold = 0.5

result = Incorporate.are_rows_similar(row1, row2, threshold)
print(result)  # 输出是否相似

