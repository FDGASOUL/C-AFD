import numpy as np
import pandas as pd


def compute_phi_squared(observed_table):
    """
    计算给定实际计数表的 φ² 值。

    :param observed_table: 实际计数表，二维列表或 pandas.DataFrame。
    :return: 计算得到的 φ² 值。
    """
    # 将输入数据转换为 NumPy 数组
    observed_table = np.array(observed_table) if not isinstance(observed_table, pd.DataFrame) else observed_table.values

    # 计算行总和和列总和
    row_totals = observed_table.sum(axis=1)
    col_totals = observed_table.sum(axis=0)
    total = observed_table.sum()

    # 检查总观测值是否为零
    if total == 0:
        raise ValueError("总观测值为 0，无法计算相关性。")

    # 计算期望计数表
    expected_table = np.outer(row_totals, col_totals) / total

    # 检查期望计数是否满足条件
    if np.any(expected_table <= 0):
        raise ValueError("期望计数表包含小于等于 0 的值，无法计算 χ²。")

    # 计算 χ² 值
    chi_squared = np.sum((observed_table - expected_table) ** 2 / expected_table)

    # 计算 φ² 值
    num_rows, num_cols = observed_table.shape
    d = min(num_rows, num_cols)  # 自由度的调整因子
    if d <= 1:
        raise ValueError("自由度小于等于 1，无法计算 φ² 值。")

    phi_squared = chi_squared / (total * (d - 1))

    return phi_squared


# 示例用法
if __name__ == "__main__":
    # 创建一个实际计数表示例
    observed = [
        [0, 71, 0, 7015],
        [0, 0, 2, 585],
        [0, 0, 16, 1581],
        [730, 0, 0, 0]
    ]

    # 计算 φ²
    try:
        phi_squared_value = compute_phi_squared(observed)
        print(f"计算得到的 φ² 值: {phi_squared_value}")
    except ValueError as e:
        print(f"错误: {e}")
