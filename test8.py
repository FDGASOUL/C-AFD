import numpy as np


def calculate_expected_frequencies(observed):
    # 计算期望频数
    row_totals = np.sum(observed, axis=1)
    col_totals = np.sum(observed, axis=0)
    total = np.sum(observed)
    expected = np.outer(row_totals, col_totals) / total
    return expected


def likelihood_ratio_chi_square_contributions(observed):
    # 计算每个格子的最大似然比卡方检验的贡献
    expected = calculate_expected_frequencies(observed)
    contributions = 2 * observed * np.log(observed / expected)
    contributions[np.isnan(contributions)] = 0  # 处理0/0的情况
    return contributions


def chi_square_contributions(observed):
    # 计算期望频数
    expected = calculate_expected_frequencies(observed)
    # 计算每个格子的卡方贡献
    contributions = (observed - expected) ** 2 / expected
    # 处理期望频数为0的情况，虽然这里可能不会有
    contributions[np.isnan(contributions)] = 0
    return contributions


# 设置numpy打印选项
np.set_printoptions(precision=2, suppress=True)

# 示例数据
observed = np.array([[7, 629, 10, 0], [631, 7, 0, 12], [668, 7, 1, 5], [10, 680, 5, 0], [0, 5, 532, 0], [9, 637, 8, 0], [624, 7, 0, 3], [0, 4, 496, 0], [0, 11, 566, 0], [665, 5, 0, 6], [0, 5, 574, 0], [0, 5, 579, 0], [9, 655, 3, 0], [8, 0, 0, 0], [636, 11, 0, 7], [6, 673, 3, 0], [0, 4, 546, 0], [6, 0, 0, 0], [6, 0, 0, 1], [6, 0, 0, 0], [7, 0, 0, 0]])

# 计算每个格子的最大似然比卡方检验的贡献
lr_contributions = likelihood_ratio_chi_square_contributions(observed)
print("每个格子的最大似然比卡方检验的贡献：")
print(lr_contributions)

# 计算每个格子的卡方检验的贡献
chi_contributions = chi_square_contributions(observed)
print("\n每个格子的卡方检验的贡献：")
print(chi_contributions)
