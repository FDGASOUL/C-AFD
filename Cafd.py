import os
from Sampler import Sampler
from ColumnLayoutRelationData import ColumnLayoutRelationData
from SearchSpace import SearchSpace
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


def process_search_space(search_space):
    """
    处理单个搜索空间的逻辑。
    :param search_space: 单个搜索空间。
    """
    print(f"Processing search space: {search_space}")
    search_space.discover()
    # 在此执行其他相关操作
    return f"Finished {search_space}"


# TODO：每个搜索空间都是独立的，因为并行处理，但是如果需要共享数据，需要考虑线程安全性，A与B独立，不仅在A的搜索空间中剪枝，也可以在B的搜索空间中剪枝
# TODO：解决方法1：使用共享的全局结论表，存储已经得出的独立性结论。解决方法2：合并搜索空间，将所有搜索空间合并为一个，不再并行搜索空间，而是并行执行需要的任务
def run_worker(search_space_counters, use_threads=True):
    """
    并行运行工作器逻辑。
    :param search_space_counters: 搜索空间计数器。
    :param use_threads: 是否使用线程池（True）或进程池（False）。
    """
    executor_cls = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    with executor_cls() as executor:
        # 提交每个搜索空间的任务
        futures = {executor.submit(process_search_space, search_space): search_space
                   for search_space in search_space_counters.keys()}

        # 收集结果
        for future in futures:
            search_space = futures[future]
            try:
                result = future.result()
                print(f"Search space {search_space} completed: {result}")
            except Exception as e:
                print(f"Error processing {search_space}: {e}")


class CAFD:
    def __init__(self, config):
        """
        初始化 CAFD 类。
        """
        self.config = config
        self.input_file_path = os.path.join(
            self.config.input_folder_path,
            self.config.input_dataset_name + self.config.input_file_ending,
        )
        self.ground_truth_path = os.path.join(
            "groundtruth", self.config.input_dataset_name + ".txt"
        )

        # 检查文件是否存在
        if not os.path.exists(self.input_file_path):
            raise FileNotFoundError(f"Input file not found: {self.input_file_path}")
        if not os.path.exists(self.ground_truth_path):
            raise FileNotFoundError(f"Ground truth file not found: {self.ground_truth_path}")

    def execute(self):
        """
        执行 CAFD 方法的主逻辑。
        """
        print(f"Executing CAFD on dataset: {self.input_file_path}")
        print(f"Using ground truth: {self.ground_truth_path}")

        # 抽样数据
        sampler = Sampler(self.config)
        sampled_data = sampler.random_sample()

        # 构建列布局关系数据
        relation_data = ColumnLayoutRelationData(sampled_data)

        # 获取模式（列名列表）
        schema = relation_data.get_schema()
        # 初始化搜索空间计数器
        search_space_counters = {}

        # 为每个属性创建一个搜索空间并初始化计数器为 0
        for next_id in range(1, len(schema) + 1):
            search_space_counters[SearchSpace(next_id)] = 0

        # 为每个搜索空间设置上下文环境
        for search_space in search_space_counters.keys():
            search_space.set_context(relation_data)

        # 并行运行工作器
        run_worker(search_space_counters, use_threads=False)