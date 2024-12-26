import heapq
import math


# TODO: 分布相似度衡量标准，使用余弦相似度和Pearson 相关系数都特别大，难以区分真正相似的分布和不相似的分布。
# TODO: 只允许脏的跟干净的归并，能简单不少
class Incorporate:
    """
    归并工具类。
    """
    similarity_threshold = 0.8  # 相似度阈值

    def merge_tables(self, actual_table, expected_table):
        """
        归并操作：对实际分布列联表和期望分布列联表进行归并。
        :param actual_table: 实际分布列联表（二维列表）。
        :param expected_table: 期望分布列联表（二维列表）。
        :return: 归并后的实际分布表和期望分布表，如果无法归并则返回 False。
        """

        # 预处理阶段
        dirty_cells = []  # 存储脏格子
        row_dirty_count = [0] * len(expected_table)  # 每行脏格子计数
        col_dirty_count = [0] * len(expected_table[0])  # 每列脏格子计数

        for i, row in enumerate(expected_table):
            for j, val in enumerate(row):
                if val < 5:
                    dirty_cells.append((i, j))
                    row_dirty_count[i] += 1
                    col_dirty_count[j] += 1

        # 记录可以归并的行对和列对
        merge_heap = []  # 最大堆存储行/列归并对及其 benefit

        # 寻找行归并对
        dirty_rows = [i for i, count in enumerate(row_dirty_count) if count > 0]
        checked_row_pairs = set()  # 用于记录已经检查过的行对

        for dirty_row_idx in dirty_rows:
            for other_row_idx in range(len(expected_table)):
                if dirty_row_idx != other_row_idx:
                    row_pair = (min(dirty_row_idx, other_row_idx), max(dirty_row_idx, other_row_idx))
                    if row_pair not in checked_row_pairs:
                        checked_row_pairs.add(row_pair)
                        if Incorporate._are_rows_similar(actual_table[row_pair[0]], actual_table[row_pair[1]], self.similarity_threshold):
                            benefit = Incorporate._calculate_row_benefit(expected_table[row_pair[0]], expected_table[row_pair[1]])
                            if benefit > len(expected_table[0]) / 5:
                                heapq.heappush(merge_heap, (-benefit, 'row', row_pair[0], row_pair[1]))

        # 寻找列归并对
        dirty_cols = [j for j, count in enumerate(col_dirty_count) if count > 0]
        checked_col_pairs = set()  # 用于记录已经检查过的列对

        for dirty_col_idx in dirty_cols:
            for other_col_idx in range(len(expected_table[0])):
                if dirty_col_idx != other_col_idx:
                    col_pair = (min(dirty_col_idx, other_col_idx), max(dirty_col_idx, other_col_idx))
                    if col_pair not in checked_col_pairs:
                        checked_col_pairs.add(col_pair)
                        if Incorporate._are_columns_similar(actual_table, col_pair[0], col_pair[1], self.similarity_threshold):
                            benefit = Incorporate._calculate_column_benefit(expected_table, col_pair[0], col_pair[1])
                            if benefit > len(expected_table) / 5:
                                heapq.heappush(merge_heap, (-benefit, 'col', col_pair[0], col_pair[1]))

        # 开始归并循环
        total_dirty_cells = len(dirty_cells)
        while total_dirty_cells > len(expected_table) * len(expected_table[0]) / 5:
            if not merge_heap:
                print("没有可归并的行或列，无法完成归并。")
                return False

            # 选择 benefit 最大的归并对
            _, merge_type, idx1, idx2 = heapq.heappop(merge_heap)

            if merge_type == 'row':
                Incorporate._merge_rows(actual_table, expected_table, idx1, idx2)
                row_dirty_count[idx1] += row_dirty_count[idx2]
                del row_dirty_count[idx2]

                # 更新脏格子计数
                for j in range(len(expected_table[0])):
                    if expected_table[idx1][j] >= 5 and (idx1, j) in dirty_cells:
                        dirty_cells.remove((idx1, j))
                        total_dirty_cells -= 1

            elif merge_type == 'col':
                Incorporate._merge_columns(actual_table, expected_table, idx1, idx2)
                col_dirty_count[idx1] += col_dirty_count[idx2]
                del col_dirty_count[idx2]

                # 更新脏格子计数
                for i in range(len(expected_table)):
                    if expected_table[i][idx1] >= 5 and (i, idx1) in dirty_cells:
                        dirty_cells.remove((i, idx1))
                        total_dirty_cells -= 1

        return actual_table, expected_table

    @staticmethod
    def _merge_rows(actual_table, expected_table, row_idx1, row_idx2):
        """
        归并两行。
        :param actual_table: 实际分布列联表。
        :param expected_table: 期望分布列联表。
        :param row_idx1: 第一行索引。
        :param row_idx2: 第二行索引。
        """
        for j in range(len(actual_table[0])):
            actual_table[row_idx1][j] += actual_table[row_idx2][j]
            expected_table[row_idx1][j] += expected_table[row_idx2][j]
        del actual_table[row_idx2]
        del expected_table[row_idx2]

    @staticmethod
    def _merge_columns(actual_table, expected_table, col_idx1, col_idx2):
        """
        归并两列。
        :param actual_table: 实际分布列联表。
        :param expected_table: 期望分布列联表。
        :param col_idx1: 第一列索引。
        :param col_idx2: 第二列索引。
        """
        for i in range(len(actual_table)):
            actual_table[i][col_idx1] += actual_table[i][col_idx2]
            expected_table[i][col_idx1] += expected_table[i][col_idx2]
            del actual_table[i][col_idx2]
            del expected_table[i][col_idx2]

    @staticmethod
    def _are_rows_similar(row1, row2, similarity_threshold):
        """
        判断两行是否分布相似，使用余弦相似度。
        :param row1: 第一行。
        :param row2: 第二行。
        :param similarity_threshold: 相似度阈值。
        :return: 布尔值，表示两行是否相似。
        """
        similarity = Incorporate._cosine_similarity(row1, row2)
        return similarity >= similarity_threshold

    @staticmethod
    def _are_columns_similar(table, col1, col2, similarity_threshold):
        """
        判断两列是否分布相似，使用余弦相似度。
        :param table: 列联表。
        :param col1: 第一列索引。
        :param col2: 第二列索引。
        :param similarity_threshold: 相似度阈值。
        :return: 布尔值，表示两列是否相似。
        """
        col1_values = [row[col1] for row in table]
        col2_values = [row[col2] for row in table]
        similarity = Incorporate._cosine_similarity(col1_values, col2_values)
        return similarity >= similarity_threshold

    @staticmethod
    def _cosine_similarity(vec1, vec2):
        """
        计算两个向量的余弦相似度。
        :param vec1: 第一个向量。
        :param vec2: 第二个向量。
        :return: 余弦相似度值。
        """
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        return dot_product / (magnitude1 * magnitude2)

    @staticmethod
    def _calculate_row_benefit(row1, row2):
        """
        计算两行归并的 benefit。
        :param row1: 第一行。
        :param row2: 第二行。
        :return: 归并带来的 benefit。
        """
        benefit = 0
        for j in range(len(row1)):
            if row1[j] < 5 and row2[j] < 5:
                if row1[j] + row2[j] >= 5:
                    # 两个脏格子合并后变为非脏格子
                    benefit += 2
                else:
                    # 两个脏格子合并仍是脏格子
                    benefit += 1
            elif row1[j] < 5 or row2[j] < 5:
                # 仅一个是脏格子
                benefit += 1
        return benefit

    @staticmethod
    def _calculate_column_benefit(table, col1_idx, col2_idx):
        """
        计算两列归并的 benefit。
        :param table: 期望分布的列联表。
        :param col1_idx: 第一列的索引。
        :param col2_idx: 第二列的索引。
        :return: 归并带来的 benefit。
        """
        benefit = 0
        for row in table:
            val1, val2 = row[col1_idx], row[col2_idx]
            if val1 < 5 and val2 < 5:
                if val1 + val2 >= 5:
                    # 两个脏格子合并后变为非脏格子
                    benefit += 2
                else:
                    # 两个脏格子合并仍是脏格子
                    benefit += 1
            elif val1 < 5 or val2 < 5:
                # 仅一个是脏格子
                benefit += 1
        return benefit

