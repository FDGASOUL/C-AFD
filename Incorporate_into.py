import heapq
import math
import logging

# 获取日志实例
logger = logging.getLogger(__name__)


class Incorporate:
    """
    归并工具类。
    """
    similarity_threshold = 0.01  # 相似度阈值

    def merge_tables(self, actual_table, expected_table):
        """
        归并操作：对实际分布列联表和期望分布列联表进行归并。
        :param actual_table: 实际分布列联表（二维列表）。
        :param expected_table: 期望分布列联表（二维列表）。
        :return: 归并后的实际分布表和期望分布表，如果无法归并则返回 False。
        """

        # 初始化映射表
        row_mapping = {i: i for i in range(len(expected_table))}  # 行索引映射
        col_mapping = {j: j for j in range(len(expected_table[0]))}  # 列索引映射

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

        # 初始化行列长度
        row_longest = len(expected_table)
        col_longest = len(expected_table[0])

        # 寻找行归并对
        dirty_rows = [i for i, count in enumerate(row_dirty_count) if count > 0]
        merge_heap = Incorporate.find_merge_pairs(dirty_rows, range(len(expected_table)), actual_table, expected_table,
                                                  is_row=True,
                                                  similarity_threshold=self.similarity_threshold,
                                                  benefit_threshold=col_longest / 5)

        # 寻找列归并对
        dirty_cols = [j for j, count in enumerate(col_dirty_count) if count > 0]
        merge_heap += Incorporate.find_merge_pairs(dirty_cols, range(len(expected_table[0])), actual_table,
                                                   expected_table,
                                                   is_row=False,
                                                   similarity_threshold=self.similarity_threshold,
                                                   benefit_threshold=row_longest / 5)

        # 开始归并循环
        total_dirty_cells = len(dirty_cells)

        while total_dirty_cells > row_longest * col_longest / 5:
            if not merge_heap:
                logger.info("没有可归并的行或列，无法完成归并。")
                return False

            # 选择 benefit 最大的归并对
            benefit, merge_type, idx1, idx2 = heapq.heappop(merge_heap)

            # 通过映射表获取最新的索引
            idx1 = row_mapping[idx1] if merge_type == 'row' else col_mapping[idx1]
            idx2 = row_mapping[idx2] if merge_type == 'row' else col_mapping[idx2]

            if idx1 == idx2:  # 已经归并过，跳过
                continue

            # 计算归并对其他行/列的影响
            dependencies = Incorporate.find_dirty_cell_dependencies(expected_table, idx1, idx2,
                                                                    axis=0 if merge_type == 'row' else 1)

            if merge_type == 'row':
                # 归并行
                Incorporate._merge(actual_table, expected_table, idx1, idx2, axis=0)

                # 更新行数
                row_longest -= 1

                # 创建一个列表存储 idx1 和 idx2 中为脏行的行索引
                dirty_rows_in_merge = [idx for idx in (idx1, idx2) if row_dirty_count[idx] > 0]

                # 更新每行脏格子计数
                row_dirty_count[idx1] = row_dirty_count[idx1] + row_dirty_count[idx2] + benefit
                row_dirty_count[idx2] = 0

                # 更新 merge_heap 中的相关归并对
                merge_heap = Incorporate.update_dirty_count_and_heap(row_dirty_count, dirty_rows_in_merge,
                                                                     expected_table,
                                                                     merge_heap, dependencies, idx1, 'row', col_longest, row_longest)

                # 更新映射表
                row_mapping = Incorporate.update_mapping(row_mapping, idx1, idx2)

                # 更新脏格子总数
                total_dirty_cells += benefit

            else:
                # 归并列
                Incorporate._merge(actual_table, expected_table, idx1, idx2, axis=1)

                # 更新列数
                col_longest -= 1

                # 创建一个列表存储 idx1 和 idx2 中为脏列的列索引
                dirty_cols_in_merge = [idx for idx in (idx1, idx2) if col_dirty_count[idx] > 0]

                # 更新每列脏格子计数
                col_dirty_count[idx1] = col_dirty_count[idx2] + col_dirty_count[idx1] + benefit
                col_dirty_count[idx2] = 0

                # 更新 merge_heap 中的相关归并对
                merge_heap = Incorporate.update_dirty_count_and_heap(col_dirty_count, dirty_cols_in_merge,
                                                                     expected_table,
                                                                     merge_heap, dependencies, idx1, 'col', row_longest, col_longest)

                # 更新映射表
                col_mapping = Incorporate.update_mapping(col_mapping, idx1, idx2)

                # 更新脏格子总数
                total_dirty_cells += benefit

        # 最终清理阶段
        actual_table = [row for i, row in enumerate(actual_table) if row_mapping[i] == i]
        expected_table = [row for i, row in enumerate(expected_table) if row_mapping[i] == i]
        for row in actual_table:
            row[:] = [val for j, val in enumerate(row) if col_mapping[j] == j]
        for row in expected_table:
            row[:] = [val for j, val in enumerate(row) if col_mapping[j] == j]

        return actual_table, expected_table

    @staticmethod
    def _merge(actual_table, expected_table, idx1, idx2, axis):
        """
        归并函数。
        :param actual_table: 实际分布列联表。
        :param expected_table: 期望分布列联表。
        :param idx1: 第一行/列索引。
        :param idx2: 第二行/列索引。
        :param axis: 指定归并方向，0 表示行，1 表示列。
        """
        if axis == 0:  # 行归并
            for j in range(len(actual_table[0])):
                actual_table[idx1][j] += actual_table[idx2][j]
                expected_table[idx1][j] += expected_table[idx2][j]
        elif axis == 1:  # 列归并
            for i in range(len(actual_table)):
                actual_table[i][idx1] += actual_table[i][idx2]
                expected_table[i][idx1] += expected_table[i][idx2]

    @staticmethod
    def _are_rows_similar(row1, row2, similarity_threshold):
        """
        判断两行是否分布相似，使用 KL 散度。
        :param row1: 第一行。
        :param row2: 第二行。
        :param similarity_threshold: 相似度阈值（KL 散度越小越相似）。
        :return: 布尔值，表示两行是否相似。
        """
        divergence = Incorporate._kl_divergence(row1, row2)
        return divergence <= similarity_threshold

    @staticmethod
    def _are_columns_similar(table, col1, col2, similarity_threshold):
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

    # 更新映射表函数
    @staticmethod
    def update_mapping(mapping, idx1, idx2):
        """
        更新映射表。
        :param mapping: 映射表。
        :param idx1: 第一个索引。
        :param idx2: 第二个索引。
        """
        for key, val in mapping.items():
            if val == idx2:
                mapping[key] = idx1
        return mapping

    @staticmethod
    def find_merge_pairs(dirty_indices, other_indices, actual_table, expected_table, is_row, similarity_threshold,
                         benefit_threshold):
        """
        寻找可归并的行对或列对并存入堆。
        :param dirty_indices: 脏的行或列索引列表。
        :param other_indices: 其他行或列索引列表。
        :param actual_table: 实际分布表。
        :param expected_table: 期望分布表。
        :param is_row: 是否为行（True 表示行，False 表示列）。
        :param similarity_threshold: 相似度阈值。
        :param benefit_threshold: 归并收益阈值。
        :return: 可归并对的堆列表。
        """
        merge_heap = []
        checked_pairs = set()

        for dirty_idx in dirty_indices:
            for other_idx in other_indices:
                if dirty_idx != other_idx:
                    pair = (min(dirty_idx, other_idx), max(dirty_idx, other_idx))
                    if pair not in checked_pairs:
                        checked_pairs.add(pair)
                        # 先计算收益
                        if is_row:
                            benefit = Incorporate._calculate_row_benefit(expected_table[pair[0]],
                                                                         expected_table[pair[1]])
                        else:
                            benefit = Incorporate._calculate_column_benefit(expected_table, pair[0], pair[1])

                        # 判断收益是否满足阈值
                        if benefit > benefit_threshold:
                            # 如果满足收益阈值，再判断相似性
                            if is_row:
                                similar = Incorporate._are_rows_similar(actual_table[pair[0]], actual_table[pair[1]],
                                                                        similarity_threshold)
                            else:
                                similar = Incorporate._are_columns_similar(actual_table, pair[0], pair[1],
                                                                           similarity_threshold)

                            # 如果相似性也满足，则加入堆
                            if similar:
                                heapq.heappush(merge_heap, (-benefit, 'row' if is_row else 'col', pair[0], pair[1]))

        return merge_heap

    @staticmethod
    def find_dirty_cell_dependencies(expected_table, idx1, idx2, axis):
        """
        寻找给定行或列索引中的脏格子涉及到的列或行。
        :param expected_table: 期望分布列联表。
        :param idx1: 第一个行/列索引。
        :param idx2: 第二个行/列索引。
        :param axis: 0 表示行，1 表示列。
        :return: 涉及的列索引列表（如果 axis=0）或行索引列表（如果 axis=1）。
        """
        dependencies = set()
        if axis == 0:  # 行
            for j in range(len(expected_table[0])):  # 遍历列
                if expected_table[idx1][j] < 5 or expected_table[idx2][j] < 5:
                    dependencies.add(j)
        elif axis == 1:  # 列
            for i in range(len(expected_table)):  # 遍历行
                if expected_table[i][idx1] < 5 or expected_table[i][idx2] < 5:
                    dependencies.add(i)
        return list(dependencies)

    @staticmethod
    def update_dirty_count_and_heap(dirty_count, dirty_items_in_merge, expected_table, merge_heap, dependencies, idx1,
                                    item_type,
                                    benefit_threshold_1, benefit_threshold_2):
        """
        更新脏格子计数和堆中归并对。
        :param dirty_count: 当前脏格子计数（行或列）。
        :param dirty_items_in_merge: 当前归并的脏行/列索引。
        :param expected_table: 期望表，用于计算 benefit。
        :param merge_heap: 归并堆。
        :param dependencies: 归并对涉及到的行/列。
        :param idx1: 归并后的行/列索引。
        :param item_type: 'row' 或 'col'，用于区分操作类型。
        :param benefit_threshold_1: 归并收益阈值 1。
        :param benefit_threshold_2: 归并收益阈值 2。
        """
        updated_heap = []
        while merge_heap:
            current_benefit, current_type, i1, i2 = heapq.heappop(merge_heap)

            if current_type == item_type:
                # 判断是否有一行/列涉及到刚归并的行/列
                is_i1_dirty = i1 in dirty_items_in_merge
                is_i2_dirty = i2 in dirty_items_in_merge

                if is_i1_dirty or is_i2_dirty:
                    # 跳过条件：另一行是干净行，且 idx1 已经是干净行
                    if ((is_i1_dirty and dirty_count[i2] == 0) or (is_i2_dirty and dirty_count[i1] == 0)) and \
                            dirty_count[idx1] == 0:
                        continue

                    # 重新计算 benefit
                    if dirty_count[i1] != 0 and dirty_count[i2] != 0:
                        if item_type == 'row':
                            current_benefit = Incorporate._calculate_row_benefit(expected_table[i1], expected_table[i2])
                        else:
                            current_benefit = Incorporate._calculate_column_benefit(expected_table, i1, i2)
                    else:
                        current_benefit = max(dirty_count[i1], dirty_count[i2])

                    # 插入堆的条件：满足收益阈值
                    if current_benefit > benefit_threshold_1 / 5:
                        heapq.heappush(updated_heap, (-current_benefit, current_type, i1, i2))
                else:
                    # 不涉及刚归并的行/列，直接重新插入堆
                    heapq.heappush(updated_heap, (-current_benefit, current_type, i1, i2))
            else:
                # 检查是否有至少一行/列在 dependencies 中
                if i1 in dependencies or i2 in dependencies:
                    # 重新计算 benefit
                    if current_type == 'row':
                        current_benefit = Incorporate._calculate_row_benefit(expected_table[i1], expected_table[i2])
                    else:
                        current_benefit = Incorporate._calculate_column_benefit(expected_table, i1, i2)

                    # 插入堆的条件：满足收益阈值 2
                    if current_benefit > benefit_threshold_2 / 5:
                        heapq.heappush(updated_heap, (-current_benefit, current_type, i1, i2))
                else:
                    # 不涉及 dependencies 的对，直接保留
                    heapq.heappush(updated_heap, (-current_benefit, current_type, i1, i2))

        return updated_heap

