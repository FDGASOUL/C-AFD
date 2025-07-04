import logging
from typing import List, Tuple, Any, Set, Dict, Optional

import numpy as np
from scipy.sparse import coo_matrix
from rule_mining.Cluster_merging import Incorporate
from rule_mining.ColumnLayoutRelationData import ColumnLayoutRelationData
# 获取日志实例
logger = logging.getLogger(__name__)


class FDAnalyzer:
    """
    FD 分析类，负责：
      1. 根据 PLI 构建列联表；
      2. 计算期望频数并检查条件；
      3. 调用列归并工具获取可合并的列对；
      4. 对所有不满足条件的 LHS 计算的可合并列对求交集；
      5. 根据最终交集更新 RHS 列的 PLI 索引。

    """
    def __init__(self, layout_data: ColumnLayoutRelationData):
        """
        :param layout_data: ColumnLayoutRelationData 对象引用，内部包含：
            - layout_data.column_vectors: List[List[int]] 形式的列向量
            - layout_data.column_data: List[{"PLI": ...}, ...] 形式的 PLI 信息
            - layout_data.null_value_id: 空值标记
        """
        self.layout_data = layout_data

        # 1. 将原来 List[List[int]] 的 column_vectors 转为 List[np.ndarray]
        raw_vecs = layout_data.get_column_vectors()         # List[List[int]]
        np_vecs = [np.array(v, dtype=int) for v in raw_vecs]  # List[np.ndarray]
        layout_data.column_vectors = np_vecs
        self.column_vectors = layout_data.column_vectors

        # 2. 保存 column_data 引用
        self.column_data = layout_data.get_column_data()  # List[{"PLI": ...}, ...]
        self.null_value_id = layout_data.get_null_value_id()
        self.n_cols = len(self.column_vectors)

        setattr(layout_data, 'rhs_expected_phi_records', {})

    def build_linked_table(
        self,
        lhs: List[int],
        rhs: int
    ) -> Tuple[np.ndarray, List[Any]]:
        """
        构建列联表并返回对应的非空值列表。
        """
        plis = self.column_data[lhs[0]]["PLI"]
        rhs_vec = self.column_vectors[rhs]
        unique_vals = sorted({rhs_vec[pos] for cluster in plis for pos in cluster
                              if rhs_vec[pos] != self.null_value_id})
        val2col = {val: idx for idx, val in enumerate(unique_vals)}
        rows: List[int] = []
        cols: List[int] = []
        for cid, cluster in enumerate(plis):
            for pos in cluster:
                val = rhs_vec[pos]
                if val != self.null_value_id:
                    rows.append(cid)
                    cols.append(val2col[val])
        data = np.ones(len(rows), dtype=int)
        mat = coo_matrix((data, (rows, cols)), shape=(len(plis), len(unique_vals)))
        dense = mat.toarray()
        mask = dense.sum(axis=1) > 0
        return dense[mask], unique_vals

    def compute_expected_frequencies(
            self,
            table: Any
    ) -> np.ndarray:
        """
        计算期望频数。

        E[i,j] = row_sum[i] * col_sum[j] / total_sum

        :param table: 实际频数表
        :return: 期望频数矩阵 (float32)
        """
        arr = np.asarray(table, dtype=np.float32)
        rt = arr.sum(axis=1, keepdims=True)
        ct = arr.sum(axis=0, keepdims=True)
        total = rt.sum() or 1.0
        return (rt * ct) / total

    def compute_expected_value(
            self,
            table: np.ndarray
    ) -> Optional[float]:
        """
        计算期望值：((C-1)*(R-1)) / ((T-1)*(d-1))，其中 d=min(R,C), T=总和
        若 T<=1 或 d<=1，则返回 None
        """
        arr = np.asarray(table, dtype=float)
        if arr.size == 0 or arr.sum() <= 1:
            return None
        R, C = arr.shape
        d = min(R, C)
        if d <= 1:
            return None
        T = arr.sum()
        return ((C - 1) * (R - 1)) / ((T - 1) * (d - 1))

    def _check_expected_frequencies(self, expected: np.ndarray) -> bool:
        """
        检查期望频数表中 >=5 的格子比例是否 >=80%。

        :param expected: 期望频数表
        :return: True 表示大部分格子满足 >=5 条件
        """
        return np.mean(expected > 5) >= 0.8

    def _check_linked_table(
            self,
            linked: np.ndarray,
            lhs: List[int]
    ) -> str:
        """
        检查列联表是否疑似 FD 或需要归并。

        逻辑：
        1. 随机抽样最多 10 行，看每行的最大值占该行总和的比例是否 >= 0.9，
           若任意一行满足，则返回 "suspected_fd"。
        2. 随机抽样最多 10 列，看每列的最大值占该列总和的比例是否 >= 0.9，
           若任意一列满足，则返回 "suspected_fd"。
        3. 否则返回 "incorporated"，表示需要进入期望频数检查或进一步归并。

        :param linked: 列联表数组，形状 (R, C)
        :param lhs: LHS 属性索引列表，用于日志或后续扩展（目前暂未用到）
        :return: "suspected_fd" 或 "incorporated"
        """
        R, C = linked.shape

        # # 行抽样检查
        # rows = linked if R <= 10 else linked[np.random.choice(R, 10, replace=False), :]
        # rs = rows.sum(axis=1)
        # # 对于 rs=0 的行，用分母 1 来避免除以 0
        # rr = rows.max(axis=1) / np.where(rs == 0, 1, rs)
        # if np.any(rr >= 0.9):
        #     return "suspected_fd"

        # 列抽样检查
        cols = linked if C <= 10 else linked[:, np.random.choice(C, 10, replace=False)]
        cs = cols.sum(axis=0)
        cr = cols.max(axis=0) / np.where(cs == 0, 1, cs)
        if np.any(cr >= 0.9):
            return "suspected_fd"

        return "incorporated"

    def rebuild_pli_from_vector(self, rhs: int) -> List[List[int]]:
        """
        给定已修改过值的 column_vectors[rhs]，重新构建该列的 PLI。
        与 ColumnLayoutRelationData 的规则保持一致：
          - 跳过 null_value_id
          - 只保留簇大小 >= 2 的聚簇

        :param rhs: 列索引
        :return: PLI 列表 List[List[int]]，每个子列表都是行号列表，且长度 >= 2
        """
        vec = self.column_vectors[rhs]  # 已经是 np.ndarray
        pli_map: Dict[Any, List[int]] = {}
        for row_idx, val in enumerate(vec):
            if val == self.null_value_id:
                # 跳过空值
                continue
            pli_map.setdefault(val, []).append(row_idx)

        new_pli: List[List[int]] = []
        for val, positions in pli_map.items():
            # 只保留那些出现次数 >= 2 的
            if len(positions) >= 2:
                new_pli.append(sorted(positions))
        return new_pli

    def refine_column_plis(self) -> None:
        """
        对每一列（视为 RHS），依次与其他列（LHS）计算列联表并检查期望频数：
          1. 若检查通过，则跳过该对。
          2. 若检查不通过，则调用列归并工具获取“可合并的列对”（值索引对）。
          3. 对所有不满足要求的 LHS，收集可合并值索引对并求交集。
          4. 根据最终交集，直接在该列的 column_vectors[rhs] 上做值替换，然后重建 PLI。
        """
        for rhs in range(self.n_cols):
            records: Dict[int, Optional[float]] = getattr(self.layout_data, 'rhs_expected_phi_records')
            records.setdefault(rhs, {})
            intersect_pairs: Set[Tuple[int, int]] = None
            rhs_unique_vals: List[Any] = []
            no_more_merge = False

            # 先遍历所有 lhs，收集每次不满足条件时返回的可合并对
            for lhs in range(rhs + 1, self.n_cols):

                # 1. 构建列联表并取出 unique_vals
                linked, unique_vals = self.build_linked_table([lhs], rhs)
                if linked.size == 0 or len(unique_vals) <= 1:
                    records[rhs][lhs] = None
                    continue

                # 记录期望值
                expected_val = self.compute_expected_value(linked)
                records[rhs][lhs] = expected_val

                if no_more_merge:
                    continue

                # 2. 初步调用 _check_linked_table
                status = self._check_linked_table(linked, [lhs])
                if status != "incorporated":
                    # 如果检测到疑似 FD（"suspected_fd"），则不做后续合并逻辑，直接跳到下一个 lhs
                    continue

                # 3. 计算期望频数并检查
                expected = self.compute_expected_frequencies(linked)
                if self._check_expected_frequencies(expected):
                    # 满足期望频数要求，则不需要归并该 lhs
                    continue

                # 3. 不满足条件时，调用列归并工具获取可合并的“值索引对”
                pairs = Incorporate().get_mergeable_column_pairs(linked)
                if not pairs:
                    # 没有可合并对，则同样跳过
                    continue

                # 4. 处理第一次与后续的交集逻辑
                if intersect_pairs is None:
                    intersect_pairs = set(pairs)
                    rhs_unique_vals = unique_vals.copy()
                else:
                    intersect_pairs &= set(pairs)

                if not intersect_pairs:
                    no_more_merge = True

            if intersect_pairs:
                vec = self.column_vectors[rhs]
                raw_map: Dict[Any, Any] = {rhs_unique_vals[j]: rhs_unique_vals[i] for i, j in intersect_pairs}
                final_map: Dict[Any, Any] = {}
                for k, v in raw_map.items():
                    tgt = v
                    while tgt in raw_map:
                        tgt = raw_map[tgt]
                    final_map[k] = tgt
                for idx in range(len(vec)):
                    if vec[idx] in final_map:
                        vec[idx] = final_map[vec[idx]]
                new_pli = self.rebuild_pli_from_vector(rhs)
                self.column_data[rhs]["PLI"] = new_pli
                logger.info(f"列 {rhs} 的值被合并替换，替换对：{intersect_pairs}，新 PLI 簇数：{len(new_pli)}")

        logger.info("所有列的 PLI 更新完成。")
