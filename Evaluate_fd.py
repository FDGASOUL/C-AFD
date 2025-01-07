import os
import logging

# 获取日志实例
logger = logging.getLogger(__name__)


def parse_fd_file(file_path):
    """
    解析函数依赖文件，将文件内容转换为FD集合。

    :param file_path: 文件路径
    :return: FD集合，格式为 {(LHS1, RHS1), (LHS2, RHS2), ...}
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    fd_set = set()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 跳过空行和注释
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # 分割 LHS 和 RHS
            if "->" in line:
                lhs, rhs = line.split("->")
                lhs_attributes = tuple(attr.strip() for attr in lhs.split(","))
                rhs_attribute = rhs.strip()
                fd_set.add((lhs_attributes, rhs_attribute))
            else:
                raise ValueError(f"Invalid FD format: {line}")

    return fd_set


def evaluate_fd(all_discovered_dependencies, ground_truth_path):
    """
    根据发现的函数依赖和指定的真实FD文件计算精度、召回率和F1分数。

    :param all_discovered_dependencies: 发现的所有函数依赖，格式为 [(LHS1, RHS1), (LHS2, RHS2), ...]
    :param ground_truth_path: 指定的真实FD文件路径。
    :return: 精度、召回率、F1分数。
    """
    # 加载真实的FD集合
    groundtruth_set = parse_fd_file(ground_truth_path)

    # 格式化发现的FD为集合，便于比较
    discovered_set = set([(tuple(sorted(LHS)), RHS) for LHS, RHS in all_discovered_dependencies])
    groundtruth_set = set([(tuple(sorted(LHS)), RHS) for LHS, RHS in groundtruth_set])

    # 计算真阳性、假阳性、假阴性
    true_positive = len(discovered_set & groundtruth_set)
    false_positive = len(discovered_set - groundtruth_set)
    false_negative = len(groundtruth_set - discovered_set)

    # 计算精度、召回率和F1分数
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1
