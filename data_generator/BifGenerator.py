import numpy as np


class BIFGenerator:
    def __init__(self, num_attributes, fd_rules, output_path, error_rate, domain_sizes):
        self.fd_rules = fd_rules
        self.num_attributes = num_attributes
        self.bif_file_path = output_path
        self.error_rate = error_rate
        # 所有列的父节点映射
        self.parents = {f'column{i + 1}': [] for i in range(num_attributes)}
        # 记录列的域大小
        self.node_domains = domain_sizes
        # 记录哪些列是"弱决定"目标
        self.weak_children = set()

    def generate_bif(self):
        """生成同时包含强FD和弱决定关系的 BIF 文件"""
        # Step 1: 加入强函数依赖
        strong_rhs = set()
        for lhs, rhs in self.fd_rules:
            rhs_col = f'column{int(rhs) + 1}'
            strong_rhs.add(rhs_col)
            for p in lhs:
                parent_col = f'column{int(p) + 1}'
                self.parents[rhs_col].append(parent_col)

        # Step 1b: 为每个强FD RHS 分配两个弱决定目标
        all_cols = list(self.parents.keys())
        involved = set(strong_rhs)
        for lhs, rhs in self.fd_rules:
            for p in lhs:
                involved.add(f'column{int(p) + 1}')
        available = [c for c in all_cols if c not in involved]
        weak_iter = iter(available)
        for rhs_col in strong_rhs:
            for _ in range(2):
                try:
                    tgt = next(weak_iter)
                except StopIteration:
                    break
                self.parents[tgt].append(rhs_col)
                self.weak_children.add(tgt)

        # 写入 BIF 文件
        with open(self.bif_file_path, 'w') as bif_file:
            bif_file.write("network unknown {\n}\n")

            # Step 2: 定义变量及其离散域
            for col, dom in self.node_domains.items():
                vals = ', '.join(map(str, range(dom)))
                bif_file.write(f"variable {col} {{\n")
                bif_file.write(f"  type discrete[{dom}] {{ {vals} }};\n")
                bif_file.write("}\n")
            bif_file.write("\n")

            # Step 3: 写入条件概率表
            for i in range(1, self.num_attributes + 1):
                col = f'column{i}'
                dom = self.node_domains[col]
                parents = self.parents[col]

                # 无父节点：平滑 Dirichlet 分布
                if not parents:
                    bif_file.write(f"probability ({col}) {{\n")
                    probs = self.generate_random_probabilities(dom)
                    bif_file.write("  table " + ", ".join(f"{p:.2f}" for p in probs) + ";\n")
                    bif_file.write("}\n\n")
                    continue

                # 有父节点：逐行写出每个父值组合的 CPT
                parent_domains = [self.node_domains[p] for p in parents]
                bif_file.write(f"probability ({col} | {', '.join(parents)}) {{\n")

                # 主值循环池，保证均匀分配
                main_pool = list(range(dom))
                np.random.shuffle(main_pool)

                for combo in np.ndindex(*parent_domains):
                    if not main_pool:
                        main_pool = list(range(dom))
                        np.random.shuffle(main_pool)
                    mv = main_pool.pop()

                    # 计算主值概率，根据是否弱决定列设置上限
                    if col in self.weak_children:
                        raw_main = 1.0 - self.error_rate
                        main_prob = round(min(raw_main, 0.5), 2)
                    else:
                        main_prob = round(1.0 - self.error_rate, 2)

                    # 剩余总概率分配
                    rem_mass = round(1.0 - main_prob, 2)
                    others = self.generate_remaining_probabilities(dom - 1, rem_mass)

                    # 构造该组合下的概率列表
                    row_probs = [main_prob if idx == mv else others.pop(0) for idx in range(dom)]

                    # 格式化父取值组合
                    combo_str = f"({', '.join(map(str, combo))})"
                    prob_str = ", ".join(f"{p:.2f}" for p in row_probs)
                    bif_file.write(f"  {combo_str} {prob_str};\n")

                bif_file.write("}\n\n")

        print(f"BIF 文件已生成: {self.bif_file_path}")

    def generate_random_probabilities(self, domain_size, concentration=5.0):
        """使用 Dirichlet 分布生成平滑概率，四舍五入并修正总和为1.00。"""
        raw = np.random.dirichlet([concentration] * domain_size)
        probs = np.round(raw, 2)
        diff = 1.0 - probs.sum()
        idx = int(np.argmax(probs))
        probs[idx] = round(probs[idx] + diff, 2)
        return probs.tolist()

    def generate_remaining_probabilities(self, num_values, remaining_prob):
        """将 remaining_prob 分配给 num_values 个概率，保留两位小数，保证和不变。"""
        probs = []
        for _ in range(num_values - 1):
            p = round(np.random.uniform(0, remaining_prob), 2)
            probs.append(p)
            remaining_prob = round(remaining_prob - p, 2)
        probs.append(round(remaining_prob, 2))
        return probs
