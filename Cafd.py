import os

from Evaluate_fd import evaluate_fd
from Sampler import Sampler
from ColumnLayoutRelationData import ColumnLayoutRelationData
from SearchSpace import SearchSpace
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


# TODO：每个搜索空间都是独立的，因为并行处理，但是如果需要共享数据，需要考虑线程安全性，A与B独立，不仅在A的搜索空间中剪枝，也可以在B的搜索空间中剪枝
# TODO：解决方法1：使用共享的全局结论表，存储已经得出的独立性结论。解决方法2：合并搜索空间，将所有搜索空间合并为一个，不再并行搜索空间，而是并行执行需要的任务。解决方法3：使用线程池，线程之间共享内存，可以更快地传递数据
# TODO：为什么用线程池比进程池更快？线程池的优势在于线程之间共享内存，可以更快地传递数据，而进程池则需要通过序列化和反序列化来传递数据，效率较低
# TODO: 使用进程池感觉没有正确执行，还是顺序的，只有一两个在运行，其他的都在等待，目前看只有3个内核在用，或者是并行启动缓慢
# TODO: 三个指标的计算方法
def process_search_space(search_space):
    """
    处理单个搜索空间的逻辑。
    :param search_space: 单个搜索空间。
    """
    try:
        print(f"Processing search space: {search_space}")
        search_space.discover()
        dependencies = search_space.get_discovered_dependencies()
        return dependencies
    except Exception as e:
        print(f"Error processing search space {search_space}: {e}")
        return []  # 返回空依赖列表以避免影响后续汇总


def run_worker(search_space_counters, use_threads=True, max_workers=None):
    """
    并行运行工作器逻辑。
    :param search_space_counters: 搜索空间计数器。
    :param use_threads: 是否使用线程池（True）或进程池（False）。
    :param max_workers: 最大工作者数量。
    """
    executor_cls = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    discovered_dependencies = []  # 汇总所有搜索空间的依赖关系

    with executor_cls(max_workers=max_workers) as executor:
        # 提交每个搜索空间的任务
        future_to_space = {executor.submit(process_search_space, search_space): search_space
                           for search_space in search_space_counters.keys()}

        # 收集结果
        for future in as_completed(future_to_space):
            search_space = future_to_space[future]
            try:
                result = future.result()
                discovered_dependencies.extend(result)
            except Exception as e:
                print(f"Error processing search space {search_space}: {e}")

    return discovered_dependencies


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

        # 并行运行工作器并汇总发现的函数依赖
        all_discovered_dependencies = run_worker(search_space_counters, use_threads=True, max_workers=None)

        # 输出汇总结果
        print("汇总发现的函数依赖:")
        for dependency in all_discovered_dependencies:
            print(dependency)

        # 计算准确率、召回率和 F1 值
        precision, recall, f1 = evaluate_fd(all_discovered_dependencies, self.ground_truth_path)
        print(f"精度: {precision:.2f}")
        print(f"召回率: {recall:.2f}")
        print(f"F1分数: {f1:.2f}")

        # 保存发现的函数依赖到文件
        discovered_file_path = os.path.join(
            "discovered_fd",
            os.path.basename(self.ground_truth_path).replace(".txt", "_discovered.txt")
        )

        os.makedirs(os.path.dirname(discovered_file_path), exist_ok=True)

        with open(discovered_file_path, "w", encoding="utf-8") as f:
            for LHS, RHS in all_discovered_dependencies:
                # 格式化 LHS 和 RHS
                lhs_str = ",".join(sorted(map(str, LHS)))  # 左部属性按字母顺序排序并用逗号连接
                rhs_str = str(RHS)  # 将 RHS 转换为字符串
                f.write(f"{lhs_str}->{rhs_str}\n")

        print(f"发现的函数依赖已保存到: {discovered_file_path}")
