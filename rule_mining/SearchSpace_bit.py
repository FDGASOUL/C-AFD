import logging
from collections import defaultdict
from threading import RLock
from typing import List, Tuple, Set

from rule_mining.Correlation_utils import CorrelationCalculator

# 获取日志实例
logger = logging.getLogger(__name__)

# 全局结论表和线程锁
global_conclusion_table: defaultdict = defaultdict(float)
global_table_lock = RLock()


class SearchSpace:
    """
    搜索空间类：针对单个属性（RHS），递归搜索所有可能的属性组合(LHS)并发现函数依赖。

    Attributes:
        column_id (int): 目标属性索引（从1开始）。
        context: 数据上下文(ColumnLayoutRelationData实例)。
        remaining_attributes (List[int]): 可用作LHS的属性列表。
        correlation_calculator: 相关性计算器实例。
        discovered_dependencies (List[Tuple[List[str], str]]): 发现的函数依赖列表。
        pruned_combinations (Set[int]): 已剪枝的组合(bitmask表示)。
    """

    def __init__(self, column_id: int) -> None:
        """
        初始化SearchSpace实例。

        :param column_id: 目标属性ID（1表示第1列）。
        """
        self.column_id: int = column_id
        self.context = None
        self.remaining_attributes: List[int] = []
        self.correlation_calculator = None
        self.discovered_dependencies: List[Tuple[List[str], str]] = []
        self.pruned_combinations: Set[int] = set()

    def set_context(self, relation_data) -> None:
        """
        设置数据上下文并初始化可用属性及相关性计算器。

        :param relation_data: ColumnLayoutRelationData类型，包含schema和列数据。
        """
        self.context = relation_data
        self.correlation_calculator = CorrelationCalculator(relation_data)

        # 获取所有属性编号（1-based），排除当前RHS属性
        total = self.context.num_columns()
        self.remaining_attributes = [i for i in range(1, total+1) if i != self.column_id]

    def generate_next_combinations(
        self,
        current_level: Set[int],
        pruned: Set[int]
    ) -> Set[int]:
        """
        根据当前层的组合生成下一层组合，跳过剪枝组合。

        :param current_level: 当前层bitmask组合集合。
        :param pruned: 当前层已剪枝bitmask集合。
        :return: 下一层bitmask组合集合。
        """
        next_level: Set[int] = set()
        combos = list(current_level)
        for i in range(len(combos)):
            for j in range(i+1, len(combos)):
                new_combo = combos[i] | combos[j]
                # 新组合大小应为上一组合大小+1
                if bin(new_combo).count('1') != bin(combos[i]).count('1') + 1:
                    continue
                # 跳过包含任何已剪枝子集的组合
                if any((new_combo & p) == p for p in pruned):
                    continue
                next_level.add(new_combo)
        return next_level

    def compute_correlation_with_cache(
        self,
        rhs_idx: int,
        lhs_idxs: List[int]
    ) -> bool:
        """
        使用全局缓存计算相关性，支持单个LHS属性。

        :param rhs_idx: RHS属性0-based索引。
        :param lhs_idxs: LHS属性0-based索引列表，长度必须为1。
        :return: 正向是否满足函数依赖。
        """
        if len(lhs_idxs) != 1:
            raise ValueError("仅支持单属性LHS情况")
        left = lhs_idxs[0]
        key_fwd = ((left,), rhs_idx)
        key_rev = ((rhs_idx,), left)

        schema = self.context.get_schema()
        lhs_name = schema[left]
        rhs_name = schema[rhs_idx]

        # 检查全局缓存
        with global_table_lock:
            if key_fwd in global_conclusion_table:
                logger.info(f"从全局结论表中命中缓存: {lhs_name}->{rhs_name}")
                return global_conclusion_table[key_fwd]

        # 计算相关性
        corr = self.correlation_calculator.compute_correlation(rhs_idx, lhs_idxs)

        # 解析结果
        mapping = {
            'mutual': (True, True),
            'left_to_right': (True, False),
            'right_to_left': (False, True),
            'invalid': (False, False)
        }
        if corr == 'pending':
            res_fwd = 'pending'
            res_rev = 'pending'
        else:
            res_fwd, res_rev = mapping.get(corr, (False, False))

        # 更新全局缓存
        with global_table_lock:
            global_conclusion_table[key_fwd] = res_fwd
            global_conclusion_table[key_rev] = res_rev

        return res_fwd

    def recursive_discover(self, level_combos: Set[int]) -> None:
        """
        递归探索属性组合，发现函数依赖并剪枝。

        :param level_combos: 当前层bitmask组合集合。
        """
        if not level_combos:
            return
        next_level: Set[int] = set()
        pruned: Set[int] = set()

        for combo in level_combos:
            # 解码bitmask为属性0-based索引列表
            lhs = [i for i in range(self.context.num_columns()) if combo & (1<<i)]
            if len(lhs) == 1:
                corr = self.compute_correlation_with_cache(self.column_id-1, lhs)
            else:
                corr = self.correlation_calculator.compute_correlation(self.column_id-1, lhs)

            schema = self.context.get_schema()
            lhs_names = [schema[i] for i in lhs]
            rhs_name = schema[self.column_id-1]

            if corr is True:
                logger.info(f"发现函数依赖: {lhs_names} -> {rhs_name}")
                self.discovered_dependencies.append((lhs_names, rhs_name))
                pruned.add(combo)
            elif corr is False or corr == 'invalid':
                pruned.add(combo)
            else:  # pending
                next_level.add(combo)

        # 生成并递归处理下一层组合
        new_combos = self.generate_next_combinations(next_level, pruned)
        self.recursive_discover(new_combos)

    def discover(self) -> None:
        """
        对当前RHS属性搜索所有LHS组合并发现依赖。
        """
        if not self.context:
            logger.warning("上下文未设置，无法发现依赖。")
            return
        total = self.context.num_columns()
        # 初始层：所有单属性组合的bitmask（排除RHS自身）
        initial = {1<<i for i in range(total) if i != self.column_id-1}
        self.recursive_discover(initial)

    def get_discovered_dependencies(self) -> List[Tuple[List[str], str]]:
        """
        获取所有已发现的函数依赖。

        :return: 依赖列表，每项为( [LHS列名], RHS列名 )
        """
        return self.discovered_dependencies
