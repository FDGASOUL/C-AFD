from Correlation_utils import CorrelationCalculator


# TODO: 只检查本层剪枝集是否合理
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

    def generate_next_combinations(self, current_level_combinations, current_level_pruned):
        """
        根据当前层的组合生成下一层的组合，仅检查本层的剪枝组合。
        :param current_level_combinations: 当前层的组合列表（用二进制表示）。
        :param current_level_pruned: 当前层的剪枝集合。
        :return: 下一层的组合列表（用二进制表示）。
        """
        next_combinations = set()

        # 两两组合生成下一层
        combinations_list = list(current_level_combinations)
        for i in range(len(combinations_list)):
            for j in range(i + 1, len(combinations_list)):
                # 按位或操作生成新组合
                new_combination = combinations_list[i] | combinations_list[j]

                # 验证新组合是否符合条件
                # 条件1：新组合的属性个数等于当前层的属性个数 + 1
                if bin(new_combination).count('1') != bin(combinations_list[i]).count('1') + 1:
                    continue

                # 条件2：新组合不包含当前层剪枝集合中的任何组合
                if any((new_combination & pruned) == pruned for pruned in current_level_pruned):
                    continue

                # 如果通过验证，加入下一层组合
                next_combinations.add(new_combination)

        return next_combinations

    def recursive_discover(self, current_level_combinations):
        """
        递归发现函数依赖。
        :param current_level_combinations: 当前层候选属性组合列表（用二进制表示）。
        """
        next_level_combinations = set()  # 用于记录下一层需要扩展的组合
        current_level_pruned = set()  # 当前层的剪枝集

        # 对当前层的候选属性组合计算相关性并处理剪枝
        for combination in current_level_combinations:
            # 解析组合，将二进制位转换为属性索引列表
            column_b = [idx for idx in range(self.context.num_columns()) if (combination & (1 << idx)) > 0]

            # 计算组合的相关性
            correlation = self.correlation_calculator.compute_correlation(self.column_id - 1, column_b)

            if correlation > self.upper_threshold:
                # 发现函数依赖，记录
                schema = self.context.get_schema()
                lhs_columns = [schema[attr] for attr in column_b]
                rhs_column = schema[self.column_id - 1]

                if len(column_b) == 1:  # 如果左部属性只有一个，判断方向
                    if self.correlation_calculator.check_dependency_direction(self.column_id - 1, column_b):
                        print(f"发现函数依赖: {lhs_columns} -> {rhs_column}")
                        self.discovered_dependencies.append((lhs_columns, rhs_column))  # 记录发现的依赖
                        current_level_pruned.add(combination)  # 将当前组合加入当前层的剪枝集
                    else:
                        next_level_combinations.add(combination)  # 方向有问题，加入扩展节点
                else:  # 如果左部属性超过一个，直接存储依赖
                    print(f"发现函数依赖: {lhs_columns} -> {rhs_column}")
                    self.discovered_dependencies.append((lhs_columns, rhs_column))  # 记录发现的依赖
                    current_level_pruned.add(combination)  # 将当前组合加入当前层的剪枝集

            elif correlation < self.lower_threshold:
                # 相关性过低，加入当前层的剪枝集
                current_level_pruned.add(combination)
            else:
                # 相关性介于上下阈值之间，保留作为下一层候选
                next_level_combinations.add(combination)

        # 生成下一层组合，仅检查当前层的剪枝集
        next_combinations = self.generate_next_combinations(next_level_combinations, current_level_pruned)

        # 如果存在下一层候选属性，递归处理
        if next_combinations:
            self.recursive_discover(next_combinations)

    def discover(self):
        """
        搜索依赖关系。
        """
        if self.column_id != 0 and self.context:
            num_columns = self.context.num_columns()

            # 初始化第一层：单属性组合（用二进制表示）
            initial_combinations = [1 << i for i in range(num_columns) if i != self.column_id - 1]

            # 开始递归搜索
            self.recursive_discover(initial_combinations)
        else:
            print("无法发现依赖：搜索空间未初始化或上下文数据未设置。")

    def get_discovered_dependencies(self):
        """
        获取已发现的所有函数依赖。
        :return: 发现的函数依赖列表。
        """
        return self.discovered_dependencies