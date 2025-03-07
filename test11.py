def chi_square_test(observed):
    """
    输入一个二维列表 observed，返回期望计数交叉表、卡方统计量以及 φ²。
    """
    # 计算行总计和整体总和
    row_totals = [sum(row) for row in observed]
    total = sum(row_totals)

    # 计算列总计
    num_cols = len(observed[0])
    col_totals = [sum(observed[i][j] for i in range(len(observed))) for j in range(num_cols)]

    # 计算期望计数交叉表
    expected = [[(row_totals[i] * col_totals[j]) / total for j in range(num_cols)]
                for i in range(len(observed))]

    # 计算卡方统计量
    chi_squared = 0
    for i in range(len(observed)):
        for j in range(num_cols):
            chi_squared += (observed[i][j] - expected[i][j]) ** 2 / expected[i][j]

    # 计算 φ²，设 d 为行数和列数中较小的那个数
    d = min(len(observed), num_cols)
    phi_squared = chi_squared / (total * (d - 1))

    return expected, chi_squared, phi_squared


if __name__ == "__main__":
    # 示例实际计数交叉表
    observed = [
        [0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 2],
    ]

    expected, chi_squared, phi_squared = chi_square_test(observed)

    print("期望计数交叉表：")
    for row in expected:
        print(row)

    print("\n卡方统计量：", chi_squared)
    print("φ²：", phi_squared)
