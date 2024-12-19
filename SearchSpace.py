from itertools import combinations
from correlation_utils import CorrelationCalculator


# TODO: 方向问题，去除索引属性问题
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
    lower_threshold = 0.3  # Cramér's V 下限

    def __init__(self, column_id):
        """
        初始化搜索空间。
        :param column_id: 属性 ID（列索引）。
        """
        self.column_id = column_id
        self.context = None
        self.candidate_tree = None
        self.correlation_calculator = None

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

        # 按层级构建依赖树
        for size in range(1, len(initial_candidates) + 1):
            for combination in combinations(initial_candidates, size):
                sorted_combination = tuple(sorted(combination))
                self.candidate_tree.add_combination(sorted_combination)

    def recursive_discover(self, current_level_nodes, visited):
        """
        递归发现函数依赖。
        :param current_level_nodes: 当前层候选属性节点列表。
        :param visited: 已发现的依赖。
        """
        next_level_nodes = []  # 下一层候选属性节点

        # 对当前层的候选属性计算相关性并剪枝
        for node in current_level_nodes:
            correlation = self.correlation_calculator.compute_correlation(self.column_id - 1, node.attributes)
            if correlation > self.upper_threshold:
                print(f"发现函数依赖: {node.attributes} -> {self.column_id - 1}")
                visited.append(node.attributes)
                self.candidate_tree.prune(node.attributes)  # 剪枝
            elif correlation < self.lower_threshold:
                self.candidate_tree.prune(node.attributes)  # 剪枝
            else:
                next_level_nodes.extend(node.children.values())

        # 如果存在下一层候选属性，递归处理
        if next_level_nodes:
            self.recursive_discover(next_level_nodes, visited)

    def discover(self):
        """
        搜索依赖关系。
        """
        visited = []  # 存储已知依赖

        if self.column_id != 0 and self.context:
            initial_nodes = list(self.candidate_tree.children.values())
            self.recursive_discover(initial_nodes, visited)

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
