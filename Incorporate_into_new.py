import math
import logging

# 获取日志实例
logger = logging.getLogger(__name__)

class Incorporate:
    """
    归并工具类。
    """
    # similarity_threshold 表示最大允许的 JS 散度，越低表示要求两分布越接近。
    similarity_threshold = 0.01

    @staticmethod
    def _js_divergence(vec1, vec2):
        """
        计算两个分布之间的 JS 散度。
        JS(P||Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)，其中 M = (P + Q) / 2。

        :param vec1: 第一个向量（数值列表）。
        :param vec2: 第二个向量（数值列表）。
        :return: JS 散度值。
        """
        # 转换为概率分布
        sum1 = sum(vec1)
        sum2 = sum(vec2)
        if sum1 == 0 or sum2 == 0:
            raise ValueError("向量和不能为零。")
        prob1 = [x / sum1 for x in vec1]
        prob2 = [x / sum2 for x in vec2]

        # 为避免 log(0) 问题，添加平滑值
        epsilon = 1e-10
        prob1 = [p + epsilon for p in prob1]
        prob2 = [q + epsilon for q in prob2]

        # 计算平均分布 M
        M = [(p + q) / 2 for p, q in zip(prob1, prob2)]

        # 计算 KL 散度（信息熵距离）
        kl1 = sum(p * math.log(p / m) for p, m in zip(prob1, M))
        kl2 = sum(q * math.log(q / m) for q, m in zip(prob2, M))
        js_divergence = 0.5 * kl1 + 0.5 * kl2
        return js_divergence

    def merge_tables(self, actual_table, expected_table):
        """
        归并操作：对实际分布列联表和期望分布列联表进行归并，
        在传入的表中直接修改数据。不论归并是否完全成功，都直接返回修改后的表格。

        :param actual_table: 实际分布列联表（二维列表）。
        :param expected_table: 期望分布列联表（二维列表）。
        :return: 返回归并后的实际分布表和期望分布表（无论是否完全满足归并条件）。
        """

        # 计算脏格子计数：统计每行、每列脏格子数量及总脏格子数
        def calculate_dirty_counts(table):
            """
            计算每行和每列的脏格子计数和总脏格子数。

            :param table: 期望分布二维列表
            :return: (row_dirty_count, col_dirty_count, total_dirty)
            """
            row_dirty_count = [sum(1 for val in row if val < 5) for row in table]
            col_count = len(table[0])
            col_dirty_count = []
            for j in range(col_count):
                cnt = sum(1 for row in table if row[j] < 5)
                col_dirty_count.append(cnt)
            total_dirty = sum(row_dirty_count)
            return row_dirty_count, col_dirty_count, total_dirty

        # 初始化脏格子计数和表的尺寸
        row_dirty_count, col_dirty_count, total_dirty_cells = calculate_dirty_counts(expected_table)
        row_count = len(expected_table)
        col_count = len(expected_table[0])
        total_cells = row_count * col_count
        threshold = total_cells / 5

        # 归并循环：当脏格子数量过多时尝试归并
        while total_dirty_cells > threshold:
            round_merged = False  # 当前轮次是否有归并动作

            # 尝试归并行
            for i in range(row_count):
                for j in range(i + 1, row_count):
                    eliminated = 0
                    for col in range(col_count):
                        val_i = expected_table[i][col]
                        val_j = expected_table[j][col]
                        # 若其中一个格子小于5且归并后至少达到5，则认为该格子“被消除”
                        if (val_i < 5 or val_j < 5) and (val_i + val_j >= 5):
                            eliminated += 1
                    if eliminated < col_count * 0.2:
                        continue

                    # 判断两行的实际分布是否足够相似
                    js_div = self._js_divergence(actual_table[i], actual_table[j])
                    if js_div > self.similarity_threshold:
                        continue

                    # 满足归并条件，将 j 行累加到 i 行
                    for col in range(col_count):
                        actual_table[i][col] += actual_table[j][col]
                        expected_table[i][col] += expected_table[j][col]
                    # 删除 j 行
                    del actual_table[j]
                    del expected_table[j]
                    row_count -= 1

                    # 重新计算脏格子计数
                    row_dirty_count, col_dirty_count, total_dirty_cells = calculate_dirty_counts(expected_table)
                    round_merged = True
                    break
                if round_merged:
                    break

            # 未进行行归并则尝试列归并
            if not round_merged:
                for i in range(col_count):
                    for j in range(i + 1, col_count):
                        eliminated = 0
                        for row in range(row_count):
                            val_i = expected_table[row][i]
                            val_j = expected_table[row][j]
                            if (val_i < 5 or val_j < 5) and (val_i + val_j >= 5):
                                eliminated += 1
                        if eliminated < row_count * 0.2:
                            continue

                        # 判断两列的实际分布是否足够相似
                        vec_i = [actual_table[row][i] for row in range(row_count)]
                        vec_j = [actual_table[row][j] for row in range(row_count)]
                        js_div = self._js_divergence(vec_i, vec_j)
                        if js_div > self.similarity_threshold:
                            continue

                        # 满足归并条件，对每行将 j 列累加到 i 列，并删除 j 列
                        for row in range(row_count):
                            actual_table[row][i] += actual_table[row][j]
                            expected_table[row][i] += expected_table[row][j]
                            del actual_table[row][j]
                            del expected_table[row][j]
                        col_count -= 1

                        # 重新计算脏格子计数
                        row_dirty_count, col_dirty_count, total_dirty_cells = calculate_dirty_counts(expected_table)
                        round_merged = True
                        break
                    if round_merged:
                        break

            # 如果当前轮次内既未进行行归并也未进行列归并，则退出循环
            if not round_merged:
                break

            # 更新总格子数与阈值（基于最新表格）
            total_cells = row_count * col_count
            threshold = total_cells / 5

        # 无论归并是否达到“成功”条件，都返回操作后的表格
        return actual_table, expected_table
