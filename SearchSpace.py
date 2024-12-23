from itertools import combinations
from correlation_utils import CorrelationCalculator


# TODO: 方向问题
# TODO: 过拟合问题，不知道为什么没有减掉，出现了大量的过拟合组合，应该在构建树的时候就限制组合长度，剪枝部分可能也有问题，发现：1与12候选，与13剪枝，但还是会验证12,13
# TODO: 构建树还是比较麻烦，筛选属性更好
class DependencyTreeNode:
    def __init__(self, attributes):
        """
        依赖树节点类，用于维护候选依赖关系。
        :param attributes: 当前节点的属性集合。
        """
        self.attributes = frozenset(attributes)
        self.children = {}  # 子节点映射（key 是单个属性，value 是 DependencyTreeNode）

    def add_combination(self, combination):
        """
        添加属性组合到依赖树中。
        :param combination: 属性组合。
        """
        if not combination:
            return
        first, *rest = combination
        if first not in self.children:
            self.children[first] = DependencyTreeNode(self.attributes | {first})
        self.children[first].add_combination(rest)

    def prune(self, attribute):
        """
        剪枝：移除包含指定属性的所有子节点。
        :param attribute: 需要移除的属性。
        """
        if attribute in self.children:
            del self.children[attribute]
        for child in self.children.values():
            child.prune(attribute)

    def get_next_level_candidates(self):
        """
        获取下一层的候选组合。
        :return: 下一层候选组合列表。
        """
        return [child.attributes for child in self.children.values()]


class SearchSpace:
    upper_threshold = 0.9  # Cramér's V 上限
    lower_threshold = 0.01  # Cramér's V 下限

    def __init__(self, column_id):
        """
        初始化搜索空间。
        :param column_id: 属性 ID（列索引）。
        """
        self.column_id = column_id
        self.context = None
        self.candidate_tree = None
        self.correlation_calculator = None
        self.discovered_dependencies = []  # 存储已发现的函数依赖

    def set_context(self, relation_data):
        """
        设置搜索空间的上下文。
        :param relation_data: ColumnLayoutRelationData 对象。
        """
        self.context = relation_data
        self.correlation_calculator = CorrelationCalculator(relation_data)
        self._initialize_candidate_tree()

    def _initialize_candidate_tree(self):
        """
        初始化候选依赖树。
        """
        num_columns = self.context.num_columns()
        initial_candidates = [col for col in range(num_columns) if col != self.column_id - 1]
        self.candidate_tree = DependencyTreeNode(set())

        MAX_COMBINATION_SIZE = 5  # 限制候选组合最大长度
        for size in range(1, min(len(initial_candidates), MAX_COMBINATION_SIZE) + 1):
            for combination in combinations(initial_candidates, size):
                self.candidate_tree.add_combination(combination)

    def recursive_discover(self, current_level_nodes):
        """
        递归发现函数依赖。
        :param current_level_nodes: 当前层候选属性节点列表。
        """
        next_level_nodes = []  # 下一层候选属性节点

        # 对当前层的候选属性计算相关性并剪枝
        for node in current_level_nodes:
            column_b = list(node.attributes) if isinstance(node.attributes, frozenset) else [node.attributes]
            correlation = self.correlation_calculator.compute_correlation(self.column_id - 1, column_b)
            if correlation > self.upper_threshold:
                schema = self.context.get_schema()
                lhs_columns = [schema[attr] for attr in column_b]
                rhs_column = schema[self.column_id - 1]
                print(f"发现函数依赖: {lhs_columns} -> {rhs_column}")
                self.discovered_dependencies.append((column_b, self.column_id - 1))  # 记录发现的依赖
                self.candidate_tree.prune(node.attributes)  # 剪枝
            elif correlation < self.lower_threshold:
                self.candidate_tree.prune(node.attributes)  # 剪枝
            else:
                next_level_nodes.extend(node.children.values())

        # 如果存在下一层候选属性，递归处理
        if next_level_nodes:
            self.recursive_discover(next_level_nodes)

    def discover(self):
        """
        搜索依赖关系。
        """
        if self.column_id != 0 and self.context:
            schema = self.context.get_schema()
            initial_nodes = list(self.candidate_tree.children.values())
            self.recursive_discover(initial_nodes)
        else:
            print("无法发现依赖：搜索空间未初始化或上下文数据未设置。")

    def get_discovered_dependencies(self):
        """
        获取所有已发现的函数依赖。
        :return: 发现的函数依赖列表。
        """
        return self.discovered_dependencies

    def __hash__(self):
        """
        定义对象的哈希值，使其可用于字典的键。
        """
        return hash(self.column_id)

    def __eq__(self, other):
        """
        定义对象相等性。
        """
        return isinstance(other, SearchSpace) and self.column_id == other.column_id

    def __repr__(self):
        return f"SearchSpace(column_id={self.column_id})"
