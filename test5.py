# import os
# from Sampler import Sampler
# from ColumnLayoutRelationData import ColumnLayoutRelationData
# from SearchSpace_bit import SearchSpace
# from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
# import logging
#
# # 获取日志实例
# logger = logging.getLogger(__name__)
#
#
# # TODO：使用共享的全局结论表，存储已经得出的独立性结论，目前只支持线程池
# # TODO: 顺序执行并不比并行执行慢，当运行较慢时，使用进程池，当运行较快时，使用线程池
# # TODO: 三个指标的计算方法
# # TODO: 对index列的处理
# def process_search_space(search_space):
#     """
#     处理单个搜索空间的逻辑。
#     :param search_space: 单个搜索空间。
#     """
#     try:
#         logger.info(f"Processing search space: {search_space}")
#         search_space.discover()
#         dependencies = search_space.get_discovered_dependencies()
#         return dependencies
#     except Exception as e:
#         logger.exception(f"Error processing search space {search_space}: {e}")
#         return []  # 返回空依赖列表以避免影响后续汇总
#
#
# def run_worker(search_space_counters, use_threads=True, max_workers=None):
#     """
#     并行运行工作器逻辑。
#     :param search_space_counters: 搜索空间计数器。
#     :param use_threads: 是否使用线程池（True）或进程池（False）。
#     :param max_workers: 最大工作者数量。
#     """
#     executor_cls = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
#     discovered_dependencies = []  # 汇总所有搜索空间的依赖关系
#
#     with executor_cls(max_workers=max_workers) as executor:
#         # 提交每个搜索空间的任务
#         future_to_space = {executor.submit(process_search_space, search_space): search_space
#                            for search_space in search_space_counters.keys()}
#
#         # 收集结果
#         for future in as_completed(future_to_space):
#             search_space = future_to_space[future]
#             try:
#                 result = future.result()
#                 discovered_dependencies.extend(result)
#             except Exception as e:
#                 logger.exception(f"Error processing search space {search_space}: {e}")
#
#     return discovered_dependencies
#
#
# class CAFD:
#     def __init__(self, config):
#         """
#         初始化 CAFD 类。
#         """
#         self.config = config
#         self.input_file_path = os.path.join(
#             self.config.input_folder_path,
#             self.config.input_dataset_name + self.config.input_file_ending,
#         )
#         self.ground_truth_path = os.path.join(
#             "groundtruth", self.config.input_dataset_name + ".txt"
#         )
#
#         # 检查文件是否存在
#         if not os.path.exists(self.input_file_path):
#             logger.error(f"Input file not found: {self.input_file_path}")
#             raise FileNotFoundError(f"Input file not found: {self.input_file_path}")
#         if not os.path.exists(self.ground_truth_path):
#             logger.error(f"Ground truth file not found: {self.ground_truth_path}")
#             raise FileNotFoundError(f"Ground truth file not found: {self.ground_truth_path}")
#
#     def execute(self):
#         """
#         执行 CAFD 方法的主逻辑。
#         """
#         logger.info(f"Executing CAFD on dataset: {self.input_file_path}")
#         logger.info(f"Using ground truth: {self.ground_truth_path}")
#
#         # 抽样数据
#         sampler = Sampler(self.config)
#         sampled_data = sampler.random_sample()
#
#         # 构建列布局关系数据
#         relation_data = ColumnLayoutRelationData(sampled_data)
#
#         # 获取模式（列名列表）
#         schema = relation_data.get_schema()
#         # 初始化搜索空间计数器
#         search_space_counters = {}
#
#         # 为每个属性创建一个搜索空间并初始化计数器为 0
#         for next_id in range(1, len(schema) + 1):
#             search_space_counters[SearchSpace(next_id)] = 0
#
#         # 为每个搜索空间设置上下文环境
#         for search_space in search_space_counters.keys():
#             search_space.set_context(relation_data)
#
#         # 并行运行工作器并汇总发现的函数依赖
#         all_discovered_dependencies = run_worker(search_space_counters, use_threads=True, max_workers=None)
#
#         return all_discovered_dependencies