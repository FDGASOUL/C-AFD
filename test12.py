import time

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import argparse


def build_and_test_crosstab(csv_file, column1, column2):
    # 读取 CSV 数据集
    df = pd.read_csv(csv_file)

    # 检查指定的列是否在数据集中
    if column1 not in df.columns or column2 not in df.columns:
        raise ValueError(f"数据集中不存在指定的列：{column1} 或 {column2}")

    # 过滤 column1 和 column2 中仅出现一次的行
    col1_counts = df[column1].value_counts()
    col2_counts = df[column2].value_counts()

    df = df[df[column1].isin(col1_counts[col1_counts > 1].index)]
    df = df[df[column2].isin(col2_counts[col2_counts > 1].index)]

    # 构建交叉表
    contingency_table = pd.crosstab(df[column1], df[column2])
    print("\n交叉表：")
    print(contingency_table)

    # # 计算卡方统计量
    # chi2, p, dof, expected = chi2_contingency(contingency_table)
    # print("\n卡方检验结果：")
    # print(f"卡方统计量: {chi2}")
    # print(f"p值: {p}")
    # print(f"自由度: {dof}")
    # print("期望频数：")
    # print(expected)

    # # 计算 φ² (Phi-squared)
    # total = contingency_table.values.sum()
    # min_dim = min(contingency_table.shape)
    # if total == 0:
    #     print("\n总观测数为零，设置 φ² 为 0。")
    #     phi_squared = 0
    # elif min_dim <= 1:
    #     print("\n自由度不足（最小维度小于等于1），设置 φ² 为 0。")
    #     phi_squared = 0
    # else:
    #     phi_squared = chi2 / (total * (min_dim - 1))
    # print("\nφ² (Phi-squared) 计算结果：")
    # print(np.sqrt(phi_squared))


if __name__ == "__main__":
    # 计算时间
    time_start = time.time()
    build_and_test_crosstab("data/data-n20.csv", "column1", "column2")
    time_end = time.time()
    print(f"\n运行时间：{time_end - time_start:.2f} 秒")

    # 计算时间
    time_start = time.time()
    build_and_test_crosstab("data/data-n20.csv", "column1", "column2")
    time_end = time.time()
    print(f"\n运行时间：{time_end - time_start:.2f} 秒")
