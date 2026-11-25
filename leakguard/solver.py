"""有限域整数求解内核（纯标准库）。

机制
----
1. 每个格子是闭区间整数域 [lo, hi]；
2. 对每条线性和等式做区间传播（HC4/盒一致性），把其它格子的极值
   折算成目标格子的新上下界，队列为空时到达不动点；
3. 传播无法继续时用 MRV（最小剩余域）选格二分回溯；
4. 所有运算只涉及 int 的加减、整除与比较，传播过程外部可逐格手算。

计数器 ``Counters`` 记录格子/约束/搜索节点的实际访问次数，
供增量评估演示“只重新访问被触及的格子和约束”。
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

from .model import Constraint, Counters


class BacktrackLimit(RuntimeError):
    """回溯节点预算耗尽 —— 判定器按 fail-closed 处理（不予批准）。"""


def _floor_div(a: int, b: int) -> int:
    """数学意义上的向下取整除法（b 可能为负）。"""
    assert b != 0
    # Python // 本身就是 floor 除法，这里显式封装以便阅读。
    return a // b


def _ceil_div(a: int, b: int) -> int:
    """数学意义上的向上取整除法（b 可能为负）。"""
    assert b != 0
    return -((-a) // b)


class DomainSolver:
    """对单张表上的一组约束做可满足性与定值判定。"""

    def __init__(self,
                 table,
                 constraints: Sequence[Constraint],
                 counters: Optional[Counters] = None,
                 node_limit: int = 200_000):
        self.table = table
        self.constraints: List[Constraint] = []
        seen = set()
        for con in constraints:
            if con.table_id != table.table_id:
                raise ValueError(
                    f"约束 {con.name} 属于表 {con.table_id}，"
                    f"不能加入表 {table.table_id} 的求解器")
            key = (con.terms, con.rhs)
            if key not in seen:  # 重复约束只保留一份，计数才公平
                seen.add(key)
                self.constraints.append(con)
        self.counters = counters or Counters()
        self.node_limit = node_limit
        # 每个格子出现在哪些约束里 —— 传播队列的唤醒依据
        self._constraints_of: Dict[str, List[int]] = {c: [] for c in table.cells}
        for idx, con in enumerate(self.constraints):
            for cell in con.cells:
                if cell not in self._constraints_of:
                    raise KeyError(
                        f"约束 {con.name} 引用了表 {table.table_id} 上不存在的格子 {cell}")
                self._constraints_of[cell].append(idx)
        self.base_domains: Dict[str, Tuple[int, int]] = {
            c: table.domain_of(c) for c in table.cells
        }

    # ------------------------------------------------------------------
    # 区间传播
    # ------------------------------------------------------------------
    def _initial_domains(self,
                         assumptions: Optional[Dict[str, int]] = None
                         ) -> Dict[str, Tuple[int, int]]:
        domains = dict(self.base_domains)
        for cell, value in (assumptions or {}).items():
            lo, hi = domains[cell]
            if not (lo <= value <= hi):
                return {}  # 假设直接越界，不可满足
            domains[cell] = (value, value)
        return domains

    def propagate(self,
                  domains: Dict[str, Tuple[int, int]],
                  queue: Optional[List[int]] = None
                  ) -> Optional[Dict[str, Tuple[int, int]]]:
        """不动点传播。返回收紧后的新域字典；不可满足返回 None。"""
        domains = dict(domains)
        if queue is None:
            queue = list(range(len(self.constraints)))
        in_queue = set(queue)

        while queue:
            cidx = queue.pop()
            in_queue.discard(cidx)
            con = self.constraints[cidx]
            self.counters.constraint_visits += 1

            # 除目标格以外所有项的极值
            others_min = 0
            others_max = 0
            for cell, coef in con.terms:
                if coef >= 0:
                    others_min += coef * domains[cell][0]
                    others_max += coef * domains[cell][1]
                else:
                    others_min += coef * domains[cell][1]
                    others_max += coef * domains[cell][0]

            for cell, coef in con.terms:
                self.counters.cell_visits += 1
                lo, hi = domains[cell]
                # coef * x ∈ [rhs - others_max, rhs - others_min]（含目标自身，扣除）
                if coef >= 0:
                    cur_min = coef * lo
                    cur_max = coef * hi
                    allow_lo = con.rhs - others_max + cur_min
                    allow_hi = con.rhs - others_min + cur_max
                    new_lo = max(lo, _ceil_div(allow_lo, coef))
                    new_hi = min(hi, _floor_div(allow_hi, coef))
                else:
                    cur_min = coef * hi   # 系数为负时，x=hi 给出最小项值
                    cur_max = coef * lo
                    allow_lo = con.rhs - others_max + cur_min
                    allow_hi = con.rhs - others_min + cur_max
                    # allow <= coef*x <= ...，coef<0，除以负数翻转不等号
                    new_lo = max(lo, _ceil_div(allow_hi, coef))
                    new_hi = min(hi, _floor_div(allow_lo, coef))

                if new_lo > new_hi:
                    return None
                if (new_lo, new_hi) != (lo, hi):
                    domains[cell] = (new_lo, new_hi)
                    for wake in self._constraints_of[cell]:
                        if wake not in in_queue:
                            in_queue.add(wake)
                            queue.append(wake)
        return domains

    # ------------------------------------------------------------------
    # 回溯搜索
    # ------------------------------------------------------------------
    def solve(self, assumptions: Optional[Dict[str, int]] = None
              ) -> Optional[Dict[str, int]]:
        """求一个完整可行整数赋值；不可满足返回 None；预算耗尽抛异常。"""
        self.counters.solver_runs += 1
        domains = self._initial_domains(assumptions)
        if not domains and assumptions:
            return None
        domains = self.propagate(domains)
        if domains is None:
            return None
        return self._dfs(domains)

    def _dfs(self, domains: Dict[str, Tuple[int, int]]) -> Optional[Dict[str, int]]:
        self.counters.search_nodes += 1
        if self.counters.search_nodes > self.node_limit:
            raise BacktrackLimit(
                f"表 {self.table.table_id} 回溯节点超过 {self.node_limit}")

        # MRV：选区间最窄且尚未定值的格子
        pick: Optional[str] = None
        pick_width: Optional[int] = None
        for cell, (lo, hi) in domains.items():
            width = hi - lo
            if width > 0 and (pick_width is None or width < pick_width):
                pick, pick_width = cell, width
        if pick is None:
            assignment = {c: lo for c, (lo, _) in domains.items()}
            return assignment if self._check(assignment) else None

        lo, hi = domains[pick]
        mid = (lo + hi) // 2
        # 区间 [lo, hi] 被完整划分为 [lo,mid-1] / {mid} / [mid+1,hi]，无重叠无遗漏
        partitions = [(mid, mid)]
        if lo <= mid - 1:
            partitions.append((lo, mid - 1))
        if mid + 1 <= hi:
            partitions.append((mid + 1, hi))
        for part_lo, part_hi in partitions:
            child = dict(domains)
            child[pick] = (part_lo, part_hi)
            child = self.propagate(child, list(self._constraints_of[pick]))
            if child is not None:
                found = self._dfs(child)
                if found is not None:
                    return found
        return None

    def _check(self, assignment: Dict[str, int]) -> bool:
        """叶子节点逐条约束做精确整数校验。"""
        for con in self.constraints:
            self.counters.constraint_visits += 1
            total = 0
            for cell, coef in con.terms:
                self.counters.cell_visits += 1
                total += coef * assignment[cell]
            if total != con.rhs:
                return False
        return True

    def feasible(self, assumptions: Optional[Dict[str, int]] = None) -> bool:
        return self.solve(assumptions) is not None

    # ------------------------------------------------------------------
    # 定值判定：保护格在所有可行赋值中是否恒为同一值
    # ------------------------------------------------------------------
    # 域组合数不超过该值时，一次穷举即可同时拿到 min/max 与双见证，
    # 避免二分法反复完整求解。
    ENUM_FASTPATH_CAP = 20_000

    def value_range(self, target: str) -> Dict[str, object]:
        """求目标格在可行解集合中的最小值/最大值及各自见证。

        返回::

            {"status": "feasible"|"infeasible"|"inconclusive",
             "min": int|None, "max": int|None,
             "witness_min": {...}|None, "witness_max": {...}|None}

        判定原理：谓词“存在 target ≤ mid 的可行解”对 mid 单调
        （mid 放宽只会扩大可行域），因此可二分最小/最大可行值；
        不要求可行值在区间上连续。min == max 即该格被唯一确定，
        min < max 则至少存在两个合法值并各自给出完整见证表。
        小表优先走一次穷举的快速路径，结论与二分路径完全一致。
        """
        combos = 1
        for lo, hi in self.base_domains.values():
            combos *= (hi - lo + 1)
            if combos > self.ENUM_FASTPATH_CAP:
                break
        if combos <= self.ENUM_FASTPATH_CAP:
            return self._enum_value_range(target)
        return self._binary_value_range(target)

    def _enum_value_range(self, target: str) -> Dict[str, object]:
        self.counters.solver_runs += 1
        try:
            solutions = self.enumerate_solutions(cap=self.ENUM_FASTPATH_CAP)
        except BacktrackLimit:
            return {"status": "inconclusive", "min": None, "max": None,
                    "witness_min": None, "witness_max": None}
        except OverflowError:
            return self._binary_value_range(target)
        if not solutions:
            return {"status": "infeasible", "min": None, "max": None,
                    "witness_min": None, "witness_max": None}
        wmin = min(solutions, key=lambda a: a[target])
        wmax = max(solutions, key=lambda a: a[target])
        return {"status": "feasible", "min": wmin[target], "max": wmax[target],
                "witness_min": wmin, "witness_max": wmax}

    def _binary_value_range(self, target: str) -> Dict[str, object]:
        base = self.solve()
        if base is None:
            return {"status": "infeasible", "min": None, "max": None,
                    "witness_min": None, "witness_max": None}
        lo0, hi0 = self.base_domains[target]
        try:
            vmin = self._binary_extreme(target, lo0, base[target], find_min=True)
            vmax = self._binary_extreme(target, base[target], hi0, find_min=False)
            wmin = self.solve({target: vmin})
            wmax = self.solve({target: vmax})
        except BacktrackLimit:
            return {"status": "inconclusive", "min": None, "max": None,
                    "witness_min": None, "witness_max": None}
        return {"status": "feasible", "min": vmin, "max": vmax,
                "witness_min": wmin, "witness_max": wmax}

    def _binary_extreme(self, target: str, low: int, high: int,
                        find_min: bool) -> int:
        """在已知边界可行的前提下二分最小/最大可行值。"""
        known_feasible_low, known_feasible_high = low, high
        while known_feasible_low < known_feasible_high:
            mid = (known_feasible_low + known_feasible_high) // 2
            if find_min:
                # 问：是否存在 x <= mid 的可行解
                dom = dict(self.base_domains)
                dom[target] = (dom[target][0], mid)
                saved = self.base_domains
                self.base_domains = dom
                try:
                    ok = self.solve() is not None
                finally:
                    self.base_domains = saved
                if ok:
                    known_feasible_high = mid
                else:
                    known_feasible_low = mid + 1
            else:
                dom = dict(self.base_domains)
                dom[target] = (mid, dom[target][1])
                saved = self.base_domains
                self.base_domains = dom
                try:
                    ok = self.solve() is not None
                finally:
                    self.base_domains = saved
                if ok:
                    known_feasible_low = mid
                else:
                    known_feasible_high = mid - 1
        return known_feasible_low

    # ------------------------------------------------------------------
    # 穷举（小表）与独立暴力参照
    # ------------------------------------------------------------------
    def enumerate_solutions(self, cap: int = 10_000) -> List[Dict[str, int]]:
        """枚举至多 cap 个完整可行解；超过 cap 抛出 OverflowError。"""
        solutions: List[Dict[str, int]] = []
        domains = self._initial_domains()
        domains = self.propagate(domains)
        if domains is None:
            return solutions
        self._enumerate(domains, solutions, cap)
        return solutions

    def _enumerate(self, domains, solutions, cap) -> None:
        if len(solutions) > cap:
            raise OverflowError(f"可行解数量超过穷举上限 {cap}")
        pick = None
        for cell, (lo, hi) in domains.items():
            if hi > lo and (pick is None or (hi - lo) < domains[pick][1] - domains[pick][0]):
                pick = cell
        if pick is None:
            assignment = {c: lo for c, (lo, _) in domains.items()}
            if self._check(assignment):
                solutions.append(assignment)
            return
        lo, hi = domains[pick]
        for value in range(lo, hi + 1):
            child = dict(domains)
            child[pick] = (value, value)
            child = self.propagate(child, list(self._constraints_of[pick]))
            if child is not None:
                self._enumerate(child, solutions, cap)

    def brute_force_feasible(self,
                             assumptions: Optional[Dict[str, int]] = None,
                             combo_cap: int = 200_000
                             ) -> bool:
        """独立参照实现：笛卡尔积暴力枚举，不共享传播/回溯代码路径。

        域组合数超过 combo_cap 时放弃并抛出 OverflowError。
        """
        domains = self._initial_domains(assumptions)
        if not domains and assumptions:
            return False
        cells = list(self.table.cells)
        ranges = [range(domains[c][0], domains[c][1] + 1) for c in cells]
        combos = 1
        for rng in ranges:
            combos *= len(rng)
            if combos > combo_cap:
                raise OverflowError("暴力枚举组合数超预算，该方法仅适用于小表")
        for values in itertools.product(*ranges):
            assignment = dict(zip(cells, values))
            if all(sum(coef * assignment[cell] for cell, coef in con.terms)
                   == con.rhs for con in self.constraints):
                return True
        return False
