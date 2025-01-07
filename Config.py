import os
import logging
import time
from Cafd import CAFD
from Evaluate_fd import evaluate_fd


def setup_logging():
    """
    全局日志配置，输出到文件（UTF-8 编码）和控制台。
    """
    # 创建日志格式
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # 创建文件处理器，设置 UTF-8 编码
    file_handler = logging.FileHandler("app.log", mode="w", encoding="utf-8", delay=True)
    file_handler.setFormatter(formatter)

    # 创建控制台处理器（可选）
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 获取根日志实例并进行配置
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)  # 设置全局日志级别
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


class Config:
    DATASET_CONFIG = {
        "IRIS": {"path": "iris", "has_header": False},
        "DATA": {"path": "data", "has_header": True},
        "BEERS": {"path": "beers", "has_header": True},
        "FLIGHTS": {"path": "flights", "has_header": True},
        "HOSPITAL": {"path": "hospital", "has_header": True},
        "ALARM": {"path": "alarm", "has_header": True},
        "ASIA": {"path": "asia", "has_header": True},
    }

    def __init__(self, dataset):
        self.dataset = dataset
        self.input_file_separator = ","
        self.input_folder_path = os.path.join("data", "")
        self.input_file_ending = ".csv"
        self._set_dataset(dataset)
        self.sample_size = 5000
        self.is_null_equal_null = True

    def _set_dataset(self, dataset):
        config = self.DATASET_CONFIG.get(dataset)
        if not config:
            valid_datasets = ", ".join(self.DATASET_CONFIG.keys())
            raise ValueError(f"Unsupported dataset: {dataset}. Valid options are: {valid_datasets}")
        self.input_dataset_name = config["path"]
        self.input_file_has_header = config["has_header"]
        self.ground_truth_path = os.path.join("groundtruth", self.input_dataset_name + ".txt")

    def __str__(self):
        return f"Config:\n\tdataset: {self.input_dataset_name}{self.input_file_ending}"


if __name__ == "__main__":
    start_time = time.time()
    setup_logging()
    logging.info("全局日志配置已完成！")
    conf = Config("DATA")
    logging.info(conf)  # 使用日志代替打印
    logging.info("Starting CAFD execution...")

    # 执行 CAFD
    cafd = CAFD(conf)
    all_discovered_dependencies = cafd.execute()

    # 输出汇总结果
    logging.info("汇总发现的函数依赖:")
    for dependency in all_discovered_dependencies:
        logging.info(dependency)

    # 计算准确率、召回率和 F1 值
    precision, recall, f1 = evaluate_fd(all_discovered_dependencies, conf.ground_truth_path)
    runtime_ms = int((time.time() - start_time) * 1000)
    logging.info(f"精度: {precision:.2f}")
    logging.info(f"召回率: {recall:.2f}")
    logging.info(f"F1分数: {f1:.2f}")
    logging.info(f"Total runtime: {runtime_ms} ms")

    # 保存发现的函数依赖到文件
    discovered_file_path = os.path.join(
        "discovered_fd",
        os.path.basename(conf.ground_truth_path).replace(".txt", "_discovered.txt")
    )

    os.makedirs(os.path.dirname(discovered_file_path), exist_ok=True)

    with open(discovered_file_path, "w", encoding="utf-8") as f:
        for LHS, RHS in all_discovered_dependencies:
            # 格式化 LHS 和 RHS
            lhs_str = ",".join(sorted(map(str, LHS)))  # 左部属性按字母顺序排序并用逗号连接
            rhs_str = str(RHS)  # 将 RHS 转换为字符串
            f.write(f"{lhs_str}->{rhs_str}\n")
        # 写入精度、召回率、F1分数和运行时间
        f.write(f"精度: {precision:.2f}\n")
        f.write(f"召回率: {recall:.2f}\n")
        f.write(f"F1分数: {f1:.2f}\n")
        f.write(f"总运行时间: {runtime_ms} ms\n")

    logging.info(f"发现的函数依赖已保存到: {discovered_file_path}")
