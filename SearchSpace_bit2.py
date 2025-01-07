from Correlation_utils import CorrelationCalculator
from itertools import combinations
import logging

# 获取日志实例
logger = logging.getLogger(__name__)


class SearchSpace:
    upper_threshold = 0.95  # 上限阈值
    lower_threshold = 0.05  # 下限阈值

    def __init__(self, column_id):
        """
        初始化搜索空间。
        :param column_id: 属性 ID（列索引）。
        """
        self.column_id = column_id  # 当前搜索空间的目标属性（RHS）
        self.context = None  # 数据上下文（ColumnLayoutRelationData 对象）
        self.remaining_attributes = []  # 可用作组合的属性（LHS）
        self.correlation_calculator = None  # 相关性计算器
        self.discovered_dependencies = []  # 存储已发现的函数依赖
        self.pruned_combinations = set()  # 无效组合集合

    def set_context(self, relation_data):
        """
        设置搜索空间的上下文。
        :param relation_data: ColumnLayoutRelationData 对象。
        """
        self.context = relation_data
        self.correlation_calculator = CorrelationCalculator(relation_data)

        # 获取所有属性的索引列表
        all_attributes = list(range(1, len(relation_data.get_schema()) + 1))

        # 确定剩余属性（可用作 LHS），排除当前属性（RHS）
        self.remaining_attributes = [attr for attr in all_attributes if attr != self.column_id]

    def detect_first_level(self):
        """
        检测第一层（单属性）中的相关性。
        :return: 可以继续上升的属性集合（用二进制表示）。
        """
        valid_attributes = set()
        num_columns = self.context.num_columns()
        initial_combinations = [1 << i for i in range(num_columns) if i != self.column_id - 1]

        for combination in initial_combinations:
            column_b = [combination.bit_length() - 1]

            # 计算相关性
            correlation = self.correlation_calculator.compute_correlation(self.column_id - 1, column_b)

            if correlation > self.upper_threshold:
                schema = self.context.get_schema()
                lhs_column = schema[column_b[0]]
                rhs_column = schema[self.column_id - 1]

                if self.correlation_calculator.check_dependency_direction(self.column_id - 1, column_b):
                    logger.info(f"发现函数依赖: {lhs_column} -> {rhs_column}")
                    self.discovered_dependencies.append(([lhs_column], rhs_column))
                else:
                    valid_attributes.add(combination)

            elif correlation > self.lower_threshold:
                valid_attributes.add(combination)

        return valid_attributes

    def generate_next_combinations(self, base_attributes, target_size, current_level_pruned):
        """
        根据基础属性生成特定大小的候选组合。
        :param base_attributes: 第一层检测出的有效属性集合。
        :param target_size: 下一层的目标组合大小（属性数量）。
        :param current_level_pruned: 当前层的剪枝集合。
        :return: 下一层的组合列表（用二进制表示）。
        """
        next_combinations = set()
        base_list = list(base_attributes)

        # 基于基础属性生成指定大小的组合
        for combination in combinations(base_list, target_size):
            # 将组合转换为二进制表示
            new_combination = sum(combination)

            # 跳过已被剪枝的组合
            if any((new_combination & pruned) == pruned for pruned in current_level_pruned):
                continue

            next_combinations.add(new_combination)

        return next_combinations

    def recursive_discover(self, base_attributes, current_level_combinations, level):
        """
        递归发现函数依赖。
        :param base_attributes: 第一层检测出的有效属性集合。
        :param current_level_combinations: 当前层候选属性组合列表（用二进制表示）。
        :param level: 当前递归层级。
        """
        current_level_pruned = set()

        for combination in current_level_combinations:
            column_b = [idx for idx in range(self.context.num_columns()) if (combination & (1 << idx)) > 0]
            correlation = self.correlation_calculator.compute_correlation(self.column_id - 1, column_b)

            if correlation > self.upper_threshold:
                schema = self.context.get_schema()
                lhs_columns = [schema[attr] for attr in column_b]
                rhs_column = schema[self.column_id - 1]

                logger.info(f"发现函数依赖: {lhs_columns} -> {rhs_column}")
                self.discovered_dependencies.append((lhs_columns, rhs_column))
                current_level_pruned.add(combination)

            elif correlation < self.lower_threshold:
                current_level_pruned.add(combination)

        next_combinations = self.generate_next_combinations(base_attributes, level, current_level_pruned)

        if next_combinations:
            self.recursive_discover(base_attributes, next_combinations, level + 1)

    def discover(self):
        """
        搜索依赖关系。
        """
        if self.column_id != 0 and self.context:
            first_level_attributes = self.detect_first_level()

            if not first_level_attributes:
                logger.info("第一层全部剪枝，不用继续上升。")
                return

            # 生成第二层候选组合
            second_level_combinations = self.generate_next_combinations(first_level_attributes, 2, set())

            if second_level_combinations:
                self.recursive_discover(first_level_attributes, second_level_combinations, 2)
            else:
                logger.info("第二层无有效组合，搜索结束。")
        else:
            logger.info("无法发现依赖：搜索空间未初始化或上下文数据未设置。")

    def get_discovered_dependencies(self):
        """
        获取已发现的所有函数依赖。
        :return: 发现的函数依赖列表。
        """
        return self.discovered_dependencies
