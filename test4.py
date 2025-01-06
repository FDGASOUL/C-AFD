# import os
#
# from Evaluate_fd import evaluate_fd
# from Sampler import Sampler
# from ColumnLayoutRelationData import ColumnLayoutRelationData
# from SearchSpace import SearchSpace
#
#
# def process_search_space(search_space):
#     """
#     处理单个搜索空间的逻辑。
#     :param search_space: 单个搜索空间。
#     """
#     try:
#         print(f"Processing search space: {search_space}")
#         search_space.discover()
#         dependencies = search_space.get_discovered_dependencies()
#         return dependencies
#     except Exception as e:
#         print(f"Error processing search space {search_space}: {e}")
#         return []  # 返回空依赖列表以避免影响后续汇总
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
#             raise FileNotFoundError(f"Input file not found: {self.input_file_path}")
#         if not os.path.exists(self.ground_truth_path):
#             raise FileNotFoundError(f"Ground truth file not found: {self.ground_truth_path}")
#
#     def execute(self):
#         """
#         执行 CAFD 方法的主逻辑。
#         """
#         print(f"Executing CAFD on dataset: {self.input_file_path}")
#         print(f"Using ground truth: {self.ground_truth_path}")
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
#         # 顺序运行每个搜索空间并汇总发现的函数依赖
#         all_discovered_dependencies = []  # 汇总所有发现的函数依赖
#         for search_space in search_space_counters.keys():
#             print(f"Processing search space: {search_space}")
#             dependencies = process_search_space(search_space)
#             all_discovered_dependencies.extend(dependencies)
#
#         # 输出汇总结果
#         print("汇总发现的函数依赖:")
#         for dependency in all_discovered_dependencies:
#             print(dependency)
#
#         # 计算准确率、召回率和 F1 值
#         precision, recall, f1 = evaluate_fd(all_discovered_dependencies, self.ground_truth_path)
#         print(f"精度: {precision:.2f}")
#         print(f"召回率: {recall:.2f}")
#         print(f"F1分数: {f1:.2f}")
#
#         # 保存发现的函数依赖到文件
#         discovered_file_path = os.path.join(
#             "discovered_fd",
#             os.path.basename(self.ground_truth_path).replace(".txt", "_discovered.txt")
#         )
#
#         os.makedirs(os.path.dirname(discovered_file_path), exist_ok=True)
#
#         with open(discovered_file_path, "w", encoding="utf-8") as f:
#             for LHS, RHS in all_discovered_dependencies:
#                 # 格式化 LHS 和 RHS
#                 lhs_str = ",".join(sorted(map(str, LHS)))  # 左部属性按字母顺序排序并用逗号连接
#                 rhs_str = str(RHS)  # 将 RHS 转换为字符串
#                 f.write(f"{lhs_str}->{rhs_str}\n")
#
#         print(f"发现的函数依赖已保存到: {discovered_file_path}")
