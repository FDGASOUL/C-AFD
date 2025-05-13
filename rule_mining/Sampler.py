import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# 获取日志实例
logger = logging.getLogger(__name__)


class Sampler:
    """
    数据集抽样器：负责加载、清洗与随机抽样数据。

    Attributes:
        config: 配置对象，应包含 input_path, sample_size, input_separator, has_header 等。
        input_path (Path): 输入文件路径。
        sample_size (int): 抽样数量。
    """

    def __init__(self, config) -> None:
        """
        初始化 Sampler。

        :param config: 配置对象，包含以下属性：
            - input_path (Path): 输入 CSV 文件路径
            - sample_size (int): 抽样行数
            - input_separator (str): CSV 分隔符
            - has_header (bool): 是否包含表头
        """
        self.config = config
        self.input_path: Path = Path(config.input_path)
        self.sample_size: int = config.sample_size

    def random_sample(self) -> pd.DataFrame:
        """
        完整抽样流程：读取、清洗并随机抽样。

        :return: 抽样后的 pandas DataFrame
        :raises ValueError: 当输入文件读取失败时抛出
        """
        logger.info(f"Reading dataset from: {self.input_path}")

        try:
            df = pd.read_csv(
                self.input_path,
                sep=self.config.input_separator,
                header=0 if self.config.has_header else None,
                low_memory=False,
                encoding='utf-8'
            )
        except Exception as exc:
            logger.error(f"读取数据失败: {exc}")
            raise ValueError(f"输入数据集读取失败: {exc}") from exc

        # 数据清洗
        clean_df = self._clean_data(df)
        # 执行抽样
        sampled_df = self._perform_sampling(clean_df)
        return sampled_df

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗：
            - 替换 "NULL" 为 NaN
            - 删除全空列
            - 删除仅含单一值的列

        :param data: 原始 DataFrame
        :return: 清洗后的 DataFrame
        """
        # 替换特殊空值
        data.replace("NULL", np.nan, inplace=True)
        # 删除全空列
        total_cols = len(data.columns)
        data.dropna(axis=1, how='all', inplace=True)
        logger.info(f"移除全空列：{total_cols - len(data.columns)} / {total_cols}")
        # 删除单一值列
        single_cols = [col for col in data.columns if data[col].nunique(dropna=True) <= 1]
        data.drop(columns=single_cols, inplace=True)
        logger.info(f"移除单一值列：{len(single_cols)} 个")
        return data

    def _perform_sampling(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        随机抽样并保存结果。

        :param data: 清洗后的 DataFrame
        :return: 抽样后的 DataFrame
        """
        if self.sample_size >= len(data):
            logger.warning(
                f"抽样数量 ({self.sample_size}) >= 数据行数 ({len(data)})，返回完整数据集"
            )
            return data

        seed = random.randint(0, 2**32 - 1)
        sampled = data.sample(n=self.sample_size, random_state=seed)
        logger.info(f"抽样 {self.sample_size} 行，随机种子：{seed}")

        # 保存样本
        self._save_sample(sampled)
        return sampled

    def _save_sample(self, data: pd.DataFrame) -> None:
        """
        保存抽样结果到 CSV 文件。

        :param data: 抽样后的 DataFrame
        """
        output_path = "rule_mining/data/sample_data/sample_data.csv"
        data.to_csv(output_path, index=False)
        logger.info(f"抽样结果已保存至: {output_path}")
