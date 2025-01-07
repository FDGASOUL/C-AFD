from itertools import combinations
from Correlation_utils import CorrelationCalculator
import logging

# 获取日志实例
logger = logging.getLogger(__name__)


# TODO: 方向问题,发现方向有问题，则认为可以继续上升，是否会影响速度？由于只有单个属性会有方向问题，可以利用缓存机制，减少计算量
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
        剪枝：递归移除所有包含指定属性的路径。
        :param attribute: 需要移除的属性。
        """
        # 收集需要删除的子节点
        to_delete = [key for key, child in self.children.items() if attribute in child.attributes]

        # 删除直接子节点
        for key in to_delete:
            del self.children[key]

        # 递归对子节点进行剪枝
        for child in self.children.values():
            child.prune(attribute)

    def get_next_level_candidates(self):
        """
        获取下一层的候选组合。
        :return: 下一层候选组合列表。
        """
        return [child.attributes for child in self.children.values()]


class SearchSpace:
    upper_threshold = 0.95  # 上限阈值
    lower_threshold = 0.05  # 下限阈值

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

        max_combination_size = 4  # 限制候选组合最大长度
        for size in range(1, min(len(initial_candidates), max_combination_size) + 1):
            for combination in combinations(initial_candidates, size):
                self.candidate_tree.add_combination(combination)

    def recursive_discover(self, current_level_nodes):
        """
        递归发现函数依赖。
        :param current_level_nodes: 当前层候选属性节点列表。
        """
        nodes_to_expand = []  # 用于记录需要扩展的节点

        # 对当前层的候选属性计算相关性并处理剪枝
        for node in current_level_nodes:
            column_b = list(node.attributes) if isinstance(node.attributes, frozenset) else [node.attributes]
            correlation = self.correlation_calculator.compute_correlation(self.column_id - 1, column_b)

            # 根据相关性阈值分类处理
            if correlation > self.upper_threshold:
                # 发现函数依赖，记录并剪枝
                schema = self.context.get_schema()
                lhs_columns = [schema[attr] for attr in column_b]
                rhs_column = schema[self.column_id - 1]

                if len(column_b) == 1:  # 如果左部属性只有一个，判断方向
                    if self.correlation_calculator.check_dependency_direction(self.column_id - 1, column_b):
                        logger.info(f"发现函数依赖: {lhs_columns} -> {rhs_column}")
                        self.discovered_dependencies.append((lhs_columns, rhs_column))  # 记录发现的依赖
                        self.candidate_tree.prune(next(iter(node.attributes)))  # 剪枝
                    else:
                        nodes_to_expand.append(node)  # 方向有问题，加入扩展节点
                else:  # 如果左部属性超过一个，直接存储依赖
                    logger.info(f"发现函数依赖: {lhs_columns} -> {rhs_column}")
                    self.discovered_dependencies.append((lhs_columns, rhs_column))  # 记录发现的依赖
                    self.candidate_tree.prune(next(iter(node.attributes)))  # 剪枝

            elif correlation < self.lower_threshold:
                # 相关性过低，直接剪枝
                self.candidate_tree.prune(next(iter(node.attributes)))  # 剪枝
            else:
                # 相关性介于上下阈值之间，记录节点以供下一层扩展
                nodes_to_expand.append(node)

        # 根据记录的节点生成下一层候选节点
        next_level_nodes = []
        for node in nodes_to_expand:
            next_level_nodes.extend(node.children.values())

        # 如果存在下一层候选属性，递归处理
        if next_level_nodes:
            self.recursive_discover(next_level_nodes)

    def discover(self):
        """
        搜索依赖关系。
        """
        if self.column_id != 0 and self.context:
            initial_nodes = list(self.candidate_tree.children.values())
            self.recursive_discover(initial_nodes)
        else:
            logger.info("无法发现依赖：搜索空间未初始化或上下文数据未设置。")

    def get_discovered_dependencies(self):
        """
        获取所有已发现的函数依赖。
        :return: 发现的函数依赖列表。
        """
        return self.discovered_dependencies
