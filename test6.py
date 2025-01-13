from scipy.stats import fisher_exact

# 2x2列联表
table = [[1458, 160],
         [4957, 93425]]

# 执行Fisher确切检验
odds_ratio, p_value = fisher_exact(table)

print(f"Odds Ratio: {odds_ratio}")
print(f"P-value: {p_value}")

# 判断是否拒绝原假设
alpha = 0.05
if p_value <= alpha:
    print("拒绝原假设：两列数据是相关的。")
else:
    print("不能拒绝原假设：两列数据是独立的。")


    # def compute_correlation(self, column_a, column_b):
    #     """
    #     计算两个列之间的相关性。
    #     :param column_a: 列 A 的索引。
    #     :param column_b: 列 B 的索引或组合。
    #     :return: 相关性值。
    #     """
    #     # 构建列连表
    #     linked_table = self.build_linked_table(column_a, column_b)
    #
    #     # 检查是否为 2x2 表
    #     if len(linked_table) != 2 or len(linked_table[0]) != 2:
    #         logger.warning("Yule's Q 仅适用于 2x2 表。返回 0。")
    #         return 0
    #
    #     # 提取四个格子的观察值
    #     a = linked_table[0][0]
    #     b = linked_table[0][1]
    #     c = linked_table[1][0]
    #     d = linked_table[1][1]
    #
    #     # 检查总观测数是否为零
    #     if a + b + c + d == 0:
    #         logger.warning("总观测数为 0，相关性设置为 0。")
    #         return 0
    #
    #     # 检查分母是否为零
    #     numerator = a * d - b * c
    #     denominator = a * d + b * c
    #     if denominator == 0:
    #         logger.warning("Yule's Q 的分母为 0，相关性设置为 0。")
    #         return 0
    #
    #     # 计算 Yule's Q
    #     yules_q = numerator / denominator
    #
    #     column_a_name = self._get_column_name(column_a)
    #     column_b_names = [self._get_column_name(col) for col in column_b]
    #     logger.info(f"计算列 {column_a_name} 和列 {column_b_names} 之间的相关性 (Yule's Q): {yules_q}")
    #     return yules_q
