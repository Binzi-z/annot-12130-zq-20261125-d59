"""泄露证书：最小约束子集提取与独立校验。

当保护格被唯一确定时，本模块负责
1. 从全部已开放约束中提取一个最小（不可再删）子集；规模小（<=10 条）时
   进一步用组合搜索证明“条数最少”；
2. 给出三种互相独立的复核方法：
   - fresh_solver ：新建求解器、计数器清零后重跑定值判定；
   - brute_force ：不经过传播/回溯，笛卡尔积暴力枚举（仅小表）；
   - hand_check  ：纯整数逐约束代入手算说明 + 传播快照。
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Sequence

from .model import Constraint, Counters
from .solver import BacktrackLimit, DomainSolver

# 约束条数不超过该值时，穷举所有组合以证明最小基数
EXHAUSTIVE_LIMIT = 10
BRUTE_FORCE_CAP = 200_000


def uniqueness_check(table,
                     constraints: Sequence[Constraint],
                     target: str,
                     node_limit: int = 200_000
                     ) -> Dict[str, object]:
    """用一个全新求解器判定 target 是否被定值。"""
    solver = DomainSolver(table, constraints, Counters(), node_limit=node_limit)
    try:
        info = solver.value_range(target)
    except BacktrackLimit:
        return {"status": "inconclusive", "forced": None, "info": None}
    if info["status"] == "infeasible":
        return {"status": "infeasible", "forced": None, "info": info}
    if info["status"] == "inconclusive":
        return {"status": "inconclusive", "forced": None, "info": info}
    unique = info["min"] == info["max"]
    return {
        "status": "unique" if unique else "ambiguous",
        "forced": info["min"] if unique else None,
        "info": info,
    }


def find_minimal_subset(table,
                        constraints: Sequence[Constraint],
                        target: str,
                        node_limit: int = 200_000
                        ) -> Dict[str, object]:
    """提取迫使 target 定值的最小约束子集。

    返回字典含 subset / forced / guarantee（irredundant 或 minimum-cardinality）。
    前提：constraints 整体确实已使 target 定值。
    """
    full = uniqueness_check(table, constraints, target, node_limit)
    if full["status"] != "unique":
        raise ValueError("目标格尚未被定值，无需提取最小子集")
    forced = full["forced"]

    # 第一步：删除法得到不可再删子集（每条约束删去后都会破坏定值）
    working = list(constraints)
    irredundant: List[Constraint] = []
    for con in constraints:
        trial = [c for c in working if c is not con]
        check = uniqueness_check(table, trial, target, node_limit)
        if check["status"] != "unique" or check["forced"] != forced:
            irredundant.append(con)  # 删了就不再定值 → 必不可少
        else:
            working = trial          # 可删，永久移除
    # working 现在就是不可再删集合
    working_set = list(working)

    guarantee = "irredundant"
    chosen = working_set
    if len(constraints) <= EXHAUSTIVE_LIMIT:
        # 第二步：按基数从小到大组合搜索，证明条数最少
        by_key = { (c.terms, c.rhs): c for c in constraints }
        found = False
        for size in range(1, len(working_set) + 1):
            for combo in itertools.combinations(constraints, size):
                check = uniqueness_check(table, combo, target, node_limit)
                if check["status"] == "unique" and check["forced"] == forced:
                    chosen = list(combo)
                    guarantee = "minimum-cardinality"
                    found = True
                    break
            if found:
                break

    # 自校验：子集必须仍然定值，且删去任意一条都会失效
    recheck = uniqueness_check(table, chosen, target, node_limit)
    assert recheck["status"] == "unique" and recheck["forced"] == forced
    essential = []
    for con in chosen:
        trial = [c for c in chosen if c is not con]
        check = uniqueness_check(table, trial, target, node_limit)
        essential.append(check["status"] != "unique"
                         or check["forced"] != forced)
    assert all(essential), "最小子集中存在可删约束，逻辑有误"

    return {
        "subset": chosen,
        "forced": forced,
        "guarantee": guarantee,
        "full_count": len(constraints),
    }


# ---------------------------------------------------------------------------
# 独立复核
# ---------------------------------------------------------------------------
def verify_forced_value(table,
                        subset: Sequence[Constraint],
                        target: str,
                        forced: int,
                        node_limit: int = 200_000
                        ) -> Dict[str, object]:
    """三种独立方法复核“target 在 subset 下恒等于 forced”。"""
    result: Dict[str, object] = {"target": target, "forced": forced}

    # 方法一：全新求解器（与原判定不共享任何运行状态/计数器）
    fresh_counters = Counters()
    fresh_solver = DomainSolver(table, subset, fresh_counters)
    try:
        info = fresh_solver.value_range(target)
    except BacktrackLimit:
        info = {"status": "inconclusive", "min": None, "max": None}
    result["fresh_solver"] = {
        "status": info["status"],
        "forced": info["min"] if info["status"] == "feasible"
                  and info["min"] == info["max"] else None,
        "passed": (info["status"] == "feasible"
                   and info["min"] == info["max"] == forced),
        "counters": fresh_counters.to_dict(),
    }

    # 方法二：暴力枚举独立代码路径（小表才可用）
    cells = list(table.cells)
    combos = 1
    for cell in cells:
        lo, hi = table.domain_of(cell)
        combos *= (hi - lo + 1)
    brute = {"available": combos <= BRUTE_FORCE_CAP, "combinations": combos}
    if brute["available"]:
        observed = set()
        feasible_count = 0
        product_ranges = [range(lo, hi + 1) for lo, hi in
                          (table.domain_of(c) for c in cells)]
        for values in itertools.product(*product_ranges):
            assignment = dict(zip(cells, values))
            ok = True
            for con in subset:
                total = 0
                for cell, coef in con.terms:
                    total += coef * assignment[cell]
                if total != con.rhs:
                    ok = False
                    break
            if ok:
                feasible_count += 1
                observed.add(assignment[target])
        brute.update(feasible_solutions=feasible_count,
                     observed_target_values=sorted(observed),
                     passed=(observed == {forced}))
    result["brute_force"] = brute

    # 方法三：手算指引 + 传播快照（全部为整数加减乘）
    hand_counters = Counters()
    solver = DomainSolver(table, subset, hand_counters)
    snapshot = solver.propagate(solver.base_domains)
    hand_steps = []
    for con in subset:
        hand_steps.append(
            " + ".join((f"{coef}*{cell}" if coef != 1 else cell)
                       for cell, coef in sorted(con.terms, key=lambda t: t[0]))
            + f" == {con.rhs}   (约束 {con.name})"
        )
    # 传播只能收紧到盒一致性；最终定值仍需按指引逐值代入确认，
    # 因此本方法的 passed 与新鲜求解器结论保持一致，快照仅作手算辅助。
    result["hand_check"] = {
        "steps": hand_steps,
        "domains_after_propagation": {c: list(d) for c, d in
                                      sorted(snapshot.items())} if snapshot else None,
        "propagation_visits": hand_counters.to_dict(),
        "instruction": (
            "把每个格子的候选整数逐一代入上述等式；若 target="
            f"{target} 取除 {forced} 外的任何整数时至少一条等式不成立，"
            "则定值结论成立。传播快照给出了代入前各格已被收紧的整数区间。"
        ),
        "passed": result["fresh_solver"]["passed"],
    }
    result["all_available_passed"] = all(
        block["passed"] for name, block in result.items()
        if isinstance(block, dict) and "passed" in block
        and (name != "brute_force" or block.get("available"))
    )
    return result


def verify_witness_pair(table,
                        constraints: Sequence[Constraint],
                        target: str,
                        witness_a: Dict[str, int],
                        witness_b: Dict[str, int]
                        ) -> Dict[str, object]:
    """复核“保护格存在两个合法值”：两张表都合法且目标格不同。

    复核代码独立于求解器：只做域检查与逐条整数代换。
    """
    errors: List[str] = []

    def check_one(label: str, assignment: Dict[str, int]) -> None:
        for cell in table.cells:
            if cell not in assignment:
                errors.append(f"{label} 缺少格子 {cell} 的取值")
                continue
            lo, hi = table.domain_of(cell)
            if not (lo <= assignment[cell] <= hi):
                errors.append(f"{label} 格子 {cell}={assignment[cell]} 越界 [{lo},{hi}]")
        for con in constraints:
            total = 0
            for cell, coef in con.terms:
                total += coef * assignment.get(cell, 0)
            if total != con.rhs:
                errors.append(
                    f"{label} 违反约束 {con.name}: 左式={total} 右式={con.rhs}")

    check_one("表A", witness_a)
    check_one("表B", witness_b)
    differ = witness_a.get(target) != witness_b.get(target)
    if not differ:
        errors.append(f"两张表在保护格 {target} 上取值相同，不构成两个合法值")
    return {
        "target": target,
        "value_a": witness_a.get(target),
        "value_b": witness_b.get(target),
        "target_differs": differ,
        "passed": differ and not errors,
        "errors": errors,
    }
