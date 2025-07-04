import logging
import time
from pathlib import Path
from typing import List, Tuple

from Cafd import CAFD
from Evaluate_fd import evaluate_fd


# 日志文件路径
LOG_FILE = Path("rule_mining/app.log")


def setup_logging(log_file: Path = LOG_FILE) -> None:
    """
    配置全局日志：同时输出到文件和控制台，使用 UTF-8 编码。

    :param log_file: 日志文件路径
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 文件处理器
    file_handler = logging.FileHandler(
        filename=str(log_file), mode="w", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 根日志器配置
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


class Config:
    """
    配置类：管理数据集路径、采样大小、空值比较策略等。

    Attributes:
        dataset_name (str): 数据集标识符
        sample_size (int): 采样行数上限
        null_equal_null (bool): 是否将空值视为相等
        input_separator (str): 输入文件分隔符
        data_dir (Path): 数据目录
        groundtruth_dir (Path): 真值文件目录
        input_path (Path): 输入文件完整路径（含扩展名）
        groundtruth_path (Path): 真值文件完整路径
    """

    # 支持的数据集及对应相对路径
    DATASET_MAP = {
        "DATA": "synthetic_data/tuples/data_t100k",
        "RWD-ADULT": "real_world_data/adult",
        "RWD-TAX": "real_world_data/tax",
        "RWD-HOSPITAL": "real_world_data/hospital.txt",
        "RWD-CLAIMS": "real_world_data/claims",
        "RWD-DBLP": "real_world_data/dblp10k",
        "RWD-C18": "real_world_data/t_biocase_gathering_agent_r72738_c18",
        "RWD-C11": "real_world_data/t_biocase_gathering_namedareas_r137711_c11",
        "RWD-C35": "real_world_data/t_biocase_gathering_r90992_c35",
        "RWD-C3": "real_world_data/t_biocase_identification_highertaxon_r562959_c3",
        "RWD-C38": "real_world_data/t_biocase_identification_r91800_c38",
    }

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.input_separator: str = ","
        self.sample_size: int = 4000
        self.null_equal_null: bool = True

        self.data_dir: Path = Path("rule_mining/data").resolve()
        self.groundtruth_dir: Path = Path("rule_mining/groundtruth").resolve()
        self.input_file_ending: str = ".csv"
        self.p = 0.01
        self.delta = 0.005

        self._initialize_dataset()

    def _initialize_dataset(self) -> None:
        """
        根据 dataset_name 设置输入文件与真值文件的完整路径。
        """
        if self.dataset_name not in self.DATASET_MAP:
            valid = ", ".join(self.DATASET_MAP)
            raise ValueError(f"不支持的数据集: {self.dataset_name}. 可选: {valid}")

        relative = self.DATASET_MAP[self.dataset_name]

        self.input_path = (self.data_dir / relative).with_suffix(self.input_file_ending)
        self.groundtruth_path = (self.groundtruth_dir / relative).with_suffix(".txt")
        self.has_header = True

    def __str__(self) -> str:
        return (
            f"Config: dataset={self.input_path.name}, "
            f"sample_size={self.sample_size}, null_equal_null={self.null_equal_null}"
        )


def main() -> None:
    """
    主程序入口：配置日志、初始化参数、执行 CAFD 并评估结果。
    """
    start = time.time()

    setup_logging()
    logging.info("日志配置完成。")

    config = Config("RWD-TAX")
    logging.info(config)

    logging.info("开始执行 CAFD...")
    cafd = CAFD(config)
    discovered: List[Tuple[List[int], int]] = cafd.execute()

    logging.info("发现的函数依赖如下：")
    for lhs, rhs in discovered:
        logging.info(f"{sorted(lhs)} -> {rhs}")

    precision, recall, f1 = evaluate_fd(discovered, str(config.groundtruth_path))
    elapsed_ms = int((time.time() - start) * 1000)
    logging.info(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
    logging.info(f"总运行时长: {elapsed_ms} ms")

    out_dir = Path("rule_mining/discovered_fd")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{config.input_path.stem}_discovered.txt"

    with out_file.open(mode="w", encoding="utf-8") as fw:
        for lhs, rhs in discovered:
            lhs_str = ",".join(map(str, sorted(lhs)))
            fw.write(f"{lhs_str}->{rhs}\n")
        fw.write(f"Precision: {precision:.4f}\n")
        fw.write(f"Recall: {recall:.4f}\n")
        fw.write(f"F1: {f1:.4f}\n")
        fw.write(f"Runtime: {elapsed_ms} ms\n")

    logging.info(f"结果已保存至: {out_file}")


if __name__ == "__main__":
    main()
