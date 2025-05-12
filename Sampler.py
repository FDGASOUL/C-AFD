import math
import os
import random

import numpy as np
import pandas as pd
import logging

# 获取日志实例
logger = logging.getLogger(__name__)


class Sampler:
    def __init__(self, config):
        """
        初始化 Sampler 类，负责数据集的随机抽样。
        :param config: Config 配置对象，包含数据路径及其他配置。
        """
        self.config = config
        self.input_file_path = os.path.join(
            self.config.input_folder_path,
            self.config.input_dataset_name + self.config.input_file_ending,
        )

        self.sample_size = self.config.sample_size

    def random_sample(self):
        """
        对输入数据集进行简单随机抽样。
        """
        logger.info(f"Reading dataset from: {self.input_file_path}")

        # 使用 pandas 读取数据
        try:
            data = pd.read_csv(
                self.input_file_path,
                sep=self.config.input_file_separator,
                header=0 if self.config.input_file_has_header else None,
                low_memory=False
            )
        except Exception as e:
            raise ValueError(f"Failed to read the input dataset: {e}")

        data.replace("NULL", np.nan, inplace=True)
        # # 删除空值超过70%的列
        # # thresh 参数表示每一列需要至少有多少个非空值才能保留
        # required_non_null = math.ceil(len(data) * 0.7)
        # data.dropna(axis=1, thresh=required_non_null, inplace=True)
        # logger.info(f"Dropped columns with more than 70% missing values (required non-null: {required_non_null}).")

        # 删除全空的列
        # thresh=1 表示只要有一个非空值就保留
        data.dropna(axis=1, thresh=1, inplace=True)
        logger.info("Dropped completely empty columns.")

        # # 删除含有空值的行
        # data.dropna(inplace=True)
        # logger.info("Dropped rows containing any missing values.")

        # 预处理掉只有一种取值的列
        for col in data.columns:
            if data[col].nunique(dropna=True) == 1:
                data.drop(col, axis=1, inplace=True)
                logger.info(f"Dropped column '{col}' with only one unique value.")

        # # 计算抽样数量
        # cols = data.columns.tolist()
        # p = 0.000001  # 用户可根据需求调整
        # delta = 0.005  # 用户可根据需求调整
        # max_n = 0
        # # 遍历每一对列，计算样本量并取最大值
        # for i in range(len(cols)):
        #     for j in range(i + 1, len(cols)):
        #         d1 = data[cols[i]].nunique(dropna=True)
        #         d2 = data[cols[j]].nunique(dropna=True)
        #         d = min(d1, d2)
        #         v = (d1 - 1) * (d2 - 1)
        #         n_ij = self.calculate_full_formula(v, p, delta, d)
        #         if n_ij > max_n:
        #             max_n = n_ij
        # # 向上取整，并更新实例样本量
        # self.sample_size = math.ceil(max_n)
        logger.info(f"Calculated dynamic sample size: {self.sample_size} rows based on domain sizes.")

        # 检查抽样数量是否合理
        if self.sample_size > len(data):
            logger.info(
                f"Sample size ({self.sample_size}) exceeds dataset size ({len(data)}). Returning the entire dataset.")
            return data

        # 随机抽样
        sampled_data = data.sample(n=self.sample_size, random_state=random.randint(0, 10000))
        logger.info(f"Sampled {self.sample_size} rows from the dataset.")
        # 保存抽样后的数据
        sampled_data.to_csv("data/rwd/hospital_sample.csv", index=False)

        return sampled_data

    def save_sample(self, output_path="sampled_data.csv"):
        """
        保存抽样后的数据到文件。
        :param output_path: 保存路径，默认保存为 `sampled_data.csv`。
        """
        sampled_data = self.random_sample()
        sampled_data.to_csv(output_path, index=False)
        logger.info(f"Sampled data saved to: {output_path}")

    def calculate_full_formula(self, v, p, delta, d):
        # 计算 log(p * sqrt(2*pi)) 项
        log_term = math.log(p * math.sqrt(2 * math.pi))
        # 分子
        numerator = math.sqrt(-16 * v * log_term) - 8 * log_term
        # 分母
        denominator = 1.69 * delta * (d - 1) * (v ** (-0.071))
        return numerator / denominator
