import logging
import os
from typing import Set, Tuple, List

# 获取日志实例
logger = logging.getLogger(__name__)


def parse_fd_file(file_path: str) -> Set[Tuple[Tuple[str, ...], str]]:
    """
    解析函数依赖文件。

    从给定文件路径逐行读取，跳过空行和以 '#' 开头的注释行，
    每行应为 "A,B->C" 格式，返回标准化的函数依赖集合。

    :param file_path: 函数依赖文件路径
    :return: 标准化的函数依赖集合，每项为 (LHS 属性元组, RHS 属性)
    :raises FileNotFoundError: 当文件不存在时抛出
    :raises ValueError: 当文件行格式不符合 "X->Y" 时抛出
    """
    if not os.path.exists(file_path):
        logger.error(f"真值文件不存在: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    fd_collection: Set[Tuple[Tuple[str, ...], str]] = set()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            cleaned = line.strip()
            # 跳过空行和注释
            if not cleaned or cleaned.startswith('#'):
                continue
            if '->' not in cleaned:
                logger.error(f"第{line_num}行格式错误: {cleaned}")
                raise ValueError(f"Invalid FD format in line {line_num}: {cleaned}")
            lhs_str, rhs_str = cleaned.split('->', 1)
            lhs = tuple(attr.strip() for attr in lhs_str.split(','))
            rhs = rhs_str.strip()
            fd_collection.add((lhs, rhs))
    return fd_collection


def _log_diagnostic_details(
    tp: int,
    fp: int,
    fn: int,
    discovered: Set[Tuple[Tuple[str, ...], str]],
    truth: Set[Tuple[Tuple[str, ...], str]]
) -> None:
    """
    记录评估的诊断信息，包括误报和漏报详情。

    :param tp: 真阳性数量
    :param fp: 假阳性数量
    :param fn: 假阴性数量
    :param discovered: 算法发现的 FD 集合
    :param truth: 真值 FD 集合
    """
    logger.info(f"评估结果: TP={tp}, FP={fp}, FN={fn}")
    if fp > 0:
        error_list = [f"{','.join(lhs)}->{rhs}" for lhs, rhs in discovered - truth]
        logger.warning(f"误报 FD ({fp} 个):\n" + "\n".join(error_list))
    if fn > 0:
        missing_list = [f"{','.join(lhs)}->{rhs}" for lhs, rhs in truth - discovered]
        logger.warning(f"漏报 FD ({fn} 个):\n" + "\n".join(missing_list))


def _calculate_metrics(
    tp: int,
    fp: int,
    fn: int
) -> Tuple[float, float, float]:
    """
    计算评估指标：精度、召回率和 F1 分数。

    :param tp: 真阳性数量
    :param fp: 假阳性数量
    :param fn: 假阴性数量
    :return: (precision, recall, f1)
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


def evaluate_fd(
    discovered_fds: List[Tuple[List[int], int]],
    ground_truth_path: str
) -> Tuple[float, float, float]:
    """
    评估算法发现的函数依赖与真值的匹配程度。

    将发现的 FD 和真值 FD 标准化后计算精度、召回率及 F1。

    :param discovered_fds: 算法发现的 FD 列表，格式 [(LHS 属性列表, RHS 属性), ...]
    :param ground_truth_path: 真值文件路径
    :return: (precision, recall, f1)
    """
    try:
        truth = parse_fd_file(ground_truth_path)
        std_truth = {(tuple(sorted(lhs)), rhs) for lhs, rhs in truth}
        std_disc = {(tuple(sorted(map(str, lhs))), str(rhs)) for lhs, rhs in discovered_fds}

        tp = len(std_disc & std_truth)
        fp = len(std_disc - std_truth)
        fn = len(std_truth - std_disc)

        _log_diagnostic_details(tp, fp, fn, std_disc, std_truth)
        return _calculate_metrics(tp, fp, fn)
    except Exception:
        logger.exception("FD 评估过程中出现异常")
        return 0.0, 0.0, 0.0
