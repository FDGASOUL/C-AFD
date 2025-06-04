import logging
import random
import math
from pathlib import Path
from itertools import combinations
from typing import List

import numpy as np
import pandas as pd

# 获取日志实例
logger = logging.getLogger(__name__)


def calculate_full_formula(v: float, p: float, delta: float, d: int) -> float:
    """
    根据给定参数计算样本量。

    :param v: (d1-1)*(d2-1)
    :param p: 置信度参数
    :param delta: 误差容限
    :param d: 两列域基数的最小值
    :return: 计算得到的样本量结果
    """
    log_term = math.log(p * math.sqrt(2 * math.pi))
    numerator = math.sqrt(-16 * v * log_term) - 8 * log_term
    denominator = 1.69 * delta * (d - 1) * (v ** (-0.071))
    return numerator / denominator


class Sampler:
    """
    数据集抽样器：负责加载、清洗、计算样本量列表，并基于“众数分段法”选择最终抽样数。

    思路：
      1. 计算所有列对的样本量 sizes；
      2. 丢弃 sizes 中大于数据集总行数的“不现实”值；
      3. 将剩余 sizes 按 [0, max_rows] 区间以 2000 为步长切分若干段（bin）；
      4. 统计每个 bin 中样本量的数量，选取数量最多的那个 bin；
      5. 以该 bin 的中点（或左/右边界）作为最终抽样数；
      6. 对原数据进行随机抽样并返回结果。
    """

    def __init__(self, config) -> None:
        """
        初始化 Sampler。

        :param config: 配置对象，包含以下属性：
            - input_path (Path): 输入 CSV 文件路径
            - input_separator (str): CSV 分隔符
            - has_header (bool): 是否包含表头
            - p (float): 置信度参数
            - delta (float): 误差容限
        """
        self.config = config
        self.input_path: Path = Path(config.input_path)
        self.input_separator: str = config.input_separator
        self.has_header: bool = config.has_header
        self.p: float = config.p
        self.delta: float = config.delta
        self.method: str = 'mode_bin_2000'  # 表示使用 2000 为步长的众数分段法

    def random_sample(self) -> pd.DataFrame:
        logger.info(f"Reading dataset from: {self.input_path}")
        try:
            df = pd.read_csv(
                self.input_path,
                sep=self.input_separator,
                header=0 if self.has_header else None,
                low_memory=False,
                encoding='utf-8'
            )
        except Exception as exc:
            logger.error(f"读取数据失败: {exc}")
            raise ValueError(f"输入数据集读取失败: {exc}") from exc

        clean_df = self._clean_data(df)
        sizes = self._calculate_all_sizes(clean_df)

        if not sizes:
            logger.warning("无法计算任何列对的样本量，返回完整数据集")
            return clean_df

        max_rows = len(clean_df)
        # 丢弃大于 max_rows 的不现实值
        realistic_sizes = [s for s in sizes if s <= max_rows]
        removed_count = len(sizes) - len(realistic_sizes)
        if removed_count > 0:
            logger.info(f"移除大于数据集行数 ({max_rows}) 的不现实样本量：{removed_count} 个")

        if not realistic_sizes:
            # 如果全部值都被移除则退回到原始 sizes 列表
            logger.warning(
                f"所有样本量均大于数据集行数 ({max_rows})，退回到原始 sizes 列表"
            )
            realistic_sizes = sizes.copy()

        # 将 realistic_sizes 按 [0, max_rows] 区间以 2000 为步长分 bin
        bin_width = 2000
        # 生成 bin 边界：0,2000,4000,..., 含最后一个 >= max_rows
        bins = list(range(0, max_rows + bin_width, bin_width))
        if bins[-1] < max_rows:
            bins.append(max_rows)

        # 使用 numpy 的 digitize 将每个 size 分到对应的 bin 索引(从1开始)
        arr = np.array(realistic_sizes)
        # 对于恰好为 max_rows 的值，让它归入最后一个 bin
        arr[arr > max_rows] = max_rows
        bin_indices = np.digitize(arr, bins)  # bin_indices 范围在 1..len(bins)

        # 统计每个 bin 中元素的数量
        bin_counts = {}
        for idx in bin_indices:
            # digitize 返回的 idx 可能是 len(bins)（当 value == bins[-1]），
            # 最终 bin 计数也包含在字典中
            bin_counts[idx] = bin_counts.get(idx, 0) + 1

        # 找到出现次数最多的 bin idx（多数情况下只需一个，若并列则取最小 idx）
        mode_bin_idx = min(
            [idx for idx, cnt in bin_counts.items() if cnt == max(bin_counts.values())]
        )
        # 计算该 bin 的左右边界
        # digitize 规则：如果 v 落在 (bins[i-1], bins[i]]，则 digitize 返回 i
        # 因此 bin i 的左边界是 bins[i-1]，右边界是 bins[i]
        left_edge = bins[mode_bin_idx - 1] if mode_bin_idx - 1 < len(bins) else bins[-2]
        right_edge = bins[mode_bin_idx] if mode_bin_idx < len(bins) else bins[-1]
        # 以中点为代表：也可以改为 left_edge 或 right_edge
        # rep_value = (left_edge + right_edge) / 2
        rep_value = right_edge

        print(
            f"样本量分段情况 -> bin 范围 [{left_edge}, {right_edge}] "
            f"包含 {bin_counts[mode_bin_idx]} 个样本量，"
            f"选用中点 {rep_value:.2f} 作为抽样数"
        )

        sample_size = math.ceil(rep_value)
        # 若计算后 sample_size 为 0，则至少取 1
        if sample_size < 1:
            sample_size = 1

        logger.info(f"最终采样方式 '{self.method}'，样本量: {sample_size}")
        return self._perform_sampling(clean_df, sample_size)

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        清理数据：
          - 将 "NULL" 字符串替换为 NaN
          - 删除全为空的列
          - 删除仅有单一值的列
        """
        data.replace("NULL", np.nan, inplace=True)
        total_cols = len(data.columns)
        data.dropna(axis=1, how='all', inplace=True)
        logger.info(f"移除全空列：{total_cols - len(data.columns)} / {total_cols}")
        single_cols = [col for col in data.columns if data[col].nunique(dropna=True) <= 1]
        data.drop(columns=single_cols, inplace=True)
        logger.info(f"移除单一值列：{len(single_cols)} 个")
        return data

    def _calculate_all_sizes(self, data: pd.DataFrame) -> List[float]:
        """
        遍历每对列并计算样本量，返回所有结果列表。
        """
        sizes: List[float] = []
        for c1, c2 in combinations(data.columns, 2):
            d1 = data[c1].nunique(dropna=True)
            d2 = data[c2].nunique(dropna=True)
            v = (d1 - 1) * (d2 - 1)
            d = min(d1, d2)
            try:
                size = calculate_full_formula(v, self.p, self.delta, d)
                sizes.append(size)
            except Exception as exc:
                logger.warning(f"计算样本量时出错: {exc}")
        return sizes

    def _perform_sampling(self, data: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """
        根据给定 sample_size 随机抽样。如果 sample_size >= 数据行数，则返回完整数据集。
        :param data: 清洗后的 DataFrame
        :param sample_size: 抽样行数
        :return: 抽样后的 DataFrame
        """
        if sample_size >= len(data):
            logger.warning(
                f"抽样数量 ({sample_size}) >= 数据行数 ({len(data)})，返回完整数据集"
            )
            return data
        seed = random.randint(0, 2**32 - 1)
        sampled = data.sample(n=sample_size, random_state=seed)
        logger.info(f"抽样 {sample_size} 行，随机种子：{seed}")
        self._save_sample(sampled)
        return sampled

    def _save_sample(self, data: pd.DataFrame) -> None:
        """
        将抽样结果保存为 CSV 文件。
        """
        output_path = "rule_mining/data/sample_data/sample_data.csv"
        data.to_csv(output_path, index=False)
        logger.info(f"抽样结果已保存至: {output_path}")
