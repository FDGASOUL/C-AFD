import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

from Sampler import Sampler
from ColumnLayoutRelationData import ColumnLayoutRelationData
from SearchSpace_bit import SearchSpace
from rule_mining.Incorporate_into import FDAnalyzer
from rule_mining.Buckets_merging import HashCompressPhi2Analyzer
from test3 import Phi2Merger

# 获取日志实例
logger = logging.getLogger(__name__)


def process_search_space(space: SearchSpace) -> List[Tuple[List[int], int]]:
    """
    处理单个搜索空间：执行发现并返回依赖列表。

    :param space: 待处理的搜索空间实例
    :return: 发现的函数依赖列表，每项格式为 (LHS 属性列表, RHS 属性)
    """
    try:
        logger.info(f"Processing search space: {space}")
        space.discover()
        return space.get_discovered_dependencies()
    except Exception:
        logger.exception(f"Error processing search space: {space}")
        return []


def run_sequential(spaces: List[SearchSpace]) -> List[Tuple[List[int], int]]:
    """
    顺序执行一组搜索空间。

    :param spaces: 搜索空间列表
    :return: 汇总所有搜索空间发现的依赖列表
    """
    results: List[Tuple[List[int], int]] = []
    for space in spaces:
        results.extend(process_search_space(space))
    return results


class CAFD:
    """
    CAFD 算法执行类：负责数据抽样、构建搜索空间并发现函数依赖。

    Attributes:
        config: 配置实例，包含数据路径、采样参数等。
        input_path: 原始数据文件完整路径
        groundtruth_path: 真值文件完整路径
    """

    def __init__(self, config) -> None:
        self.config = config
        # 使用 Config 类中预先构建的路径属性
        self.input_path: Path = config.input_path
        self.groundtruth_path: Path = config.groundtruth_path

        # 校验文件存在性
        if not self.input_path.exists():
            logger.error(f"输入文件不存在: {self.input_path}")
            raise FileNotFoundError(f"输入文件不存在: {self.input_path}")
        if not self.groundtruth_path.exists():
            logger.error(f"真值文件不存在: {self.groundtruth_path}")
            raise FileNotFoundError(f"真值文件不存在: {self.groundtruth_path}")

    def execute(self) -> List[Tuple[List[int], int]]:
        """
        执行 CAFD 主流程：
        1. 抽样
        2. 构建列布局关系数据
        3. 初始化搜索空间并发现依赖

        :return: 发现的所有函数依赖列表
        """
        logger.info(f"Executing CAFD on: {self.input_path}")
        logger.info(f"Using ground truth: {self.groundtruth_path}")

        # 数据抽样
        sampler = Sampler(self.config)
        sample = sampler.random_sample()

        # 构建列布局关系数据
        layout_data = ColumnLayoutRelationData(sample)
        schema = layout_data.get_schema()

        analyzer = FDAnalyzer(layout_data)
        analyzer.refine_column_plis()

        # merger = Phi2Merger(layout_data, target_index=1)  # 设定目标列为第6列
        # num = merger.analyze_and_merge()
        #
        # analyzer = HashCompressPhi2Analyzer(layout_data, target_index=1, target_num_clusters=num)
        # analyzer.analyze_compression_effect()


        # 初始化并设置上下文
        spaces: List[SearchSpace] = []
        for attr_id in range(1, len(schema) + 1):
            space = SearchSpace(attr_id)
            space.set_context(layout_data)
            spaces.append(space)

        # 顺序发现依赖
        dependencies = run_sequential(spaces)
        logger.info(f"Total dependencies found: {len(dependencies)}")
        return dependencies
