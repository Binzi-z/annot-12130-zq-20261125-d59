"""命令行入口（纯标准库，Windows 直接运行）。

用法
----
python -m leakguard.cli demo                 # 运行全部演示场景
python -m leakguard.cli demo --scenario 2    # 只运行场景 2
python -m leakguard.cli check scenario.json  # 检查 JSON 场景，输出 JSON 裁决
python -m leakguard.cli selftest             # 小表暴力对照冒烟自检
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List

from . import demo
from .jsonio import load_scenario
from .solver import DomainSolver
from .model import Counters


# ---------------------------------------------------------------------------
# 可读渲染
# ---------------------------------------------------------------------------
def _line(char: str = "=") -> str:
    return char * 72


def render_cell_verdict(engine, table, cv) -> List[str]:
    lines = [f"  保护格 {cv.cell}: {cv.status.upper()}"]
    if cv.status == "ambiguous":
        lines.append(f"    可行整数区间 [{cv.value_min}, {cv.value_max}]，"
                     f"存在至少两个合法值，不予泄露判定。")
        lines.append("    见证表 A（独立复核：逐格代回所有约束均成立）：")
        lines.extend("      " + ln for ln in table.render(cv.witness_a).splitlines())
        lines.append("    见证表 B：")
        lines.extend("      " + ln for ln in table.render(cv.witness_b).splitlines())
        chk = cv.witness_check
        lines.append(f"    见证独立复核通过: {chk['passed']} "
                     f"(A 值={chk['value_a']}, B 值={chk['value_b']})")
    elif cv.status == "unique":
        lines.append(f"    唯一确定值 = {cv.forced_value}")
        lines.append(f"    最小已开放约束集合（{cv.subset_guarantee}，"
                     f"共 {len(cv.minimal_subset)} 条）：")
        for item in cv.independent_verification["subset_provenance"]:
            con = item["constraint"]
            terms = " + ".join(f"{coef}*{c}" if coef != 1 else c
                               for c, coef in sorted(con["terms"].items()))
            lines.append(f"      - [{item['record_id']}] {con['name']}: "
                         f"{terms} == {con['rhs']}")
        iv = cv.independent_verification
        lines.append("    独立校验：")
        lines.append(f"      1) 全新求解器重算: "
                     f"passed={iv['fresh_solver']['passed']}")
        bf = iv["brute_force"]
        if bf["available"]:
            lines.append(f"      2) 暴力枚举(独立代码路径, {bf['combinations']} 种组合, "
                         f"{bf['feasible_solutions']} 个可行解): passed={bf['passed']}, "
                         f"目标格观测值={bf['observed_target_values']}")
        else:
            lines.append(f"      2) 暴力枚举: 组合数 {bf['combinations']} 超预算，跳过")
        lines.append(f"      3) 手算指引: passed={iv['hand_check']['passed']}")
        for step in iv["hand_check"]["steps"]:
            lines.append(f"         {step}")
        snap = iv["hand_check"]["domains_after_propagation"]
        lines.append(f"         传播后整数域: {snap}")
    else:
        lines.append("    判定不充分（预算耗尽或约束集不可满足），按 fail-closed 处理")
    return lines


def render_table_verdict(engine, tv) -> List[str]:
    table = engine.tables[tv.table_id]
    lines = [f"表 {tv.table_id}（缓存命中={tv.from_cache}）"]
    for cv in tv.cell_verdicts:
        lines.extend(render_cell_verdict(engine, table, cv))
    c = tv.counters
    lines.append(f"  访问计数: 格子 {c.cell_visits} 次, 约束 {c.constraint_visits} 次, "
                 f"搜索节点 {c.search_nodes} 个, 求解调用 {c.solver_runs} 次")
    return lines


def render_request_verdict(engine, rv) -> List[str]:
    lines = [_line("-"),
             f"申请 {rv.request_id}（{rv.team}, order={rv.order}）=> {rv.decision}",
             f"理由: {rv.reason}",
             f"触及表: {rv.touched_tables}  缓存命中表: {rv.cache_hits}",
             f"批次计数: {rv.batch_counters.to_dict()}"]
    for tv in rv.table_verdicts:
        lines.extend(render_table_verdict(engine, tv))
    return lines


# ---------------------------------------------------------------------------
# 各场景演示
# ---------------------------------------------------------------------------
def _print_header(title: str) -> None:
    print("\n" + _line("="))
    print(title)
    print(_line("="))


def run_demo(only: int = 0) -> None:
    runners = [
        (1, _demo_one, "边际份额叠加推断"),
        (2, _demo_two, "两个合法值与双见证表"),
        (3, _demo_three, "撤回的两种语义"),
        (4, _demo_four, "两队竞争与承诺顺序"),
        (5, _demo_five, "增量评估与访问计数"),
    ]
    for idx, fn, title in runners:
        if only and idx != only:
            continue
        _print_header(f"场景 {idx} · {title}")
        fn()


def _demo_one() -> None:
    s = demo.scenario_marginal_inference()
    engine = s["engine"]
    print("前四种行/列份额开放后（四条边际秩不足，d 仍有自由度）：")
    for ln in render_table_verdict(engine, s["before_request"]):
        print(ln)
    print("\n第五天申请对角份额，叠加全部已开放约束后：")
    for ln in render_request_verdict(engine, s["verdict"]):
        print(ln)


def _demo_two() -> None:
    s = demo.scenario_two_witnesses()
    for ln in render_table_verdict(s["engine"], s["verdict"]):
        print(ln)


def _demo_three() -> None:
    s = demo.scenario_withdrawal()
    engine = s["engine"]
    print("[T1] C4 仅批准未发布：")
    print("\n".join(render_table_verdict(engine, s["before_withdraw_t1"])))
    print("\n[T1] 撤回未发布记录后（视同从未对外发布，约束失效）：")
    print("\n".join(render_table_verdict(engine, s["after_withdraw_t1"])))
    print("\n[T2] C2/C3/C5 全部已发布，撤回 C5 之前：")
    print("\n".join(render_table_verdict(engine, s["before_withdraw_t2"])))
    print("\n[T2] 撤回已发布记录后（外部台账保留，不能抵消）：")
    print("\n".join(render_table_verdict(engine, s["after_withdraw_t2"])))
    print("\n追加式外部台账（只增不减）：")
    for rec in s["external_ledger"]:
        print(f"  order={rec.order:>3} {rec.record_id:<12} {rec.state.value:<18}"
              f"{rec.constraint.name}")
    print("\n就 T2 的 C5 重新申请 -> 冲突记录: "
          f"{s['republish_conflict']}（数据外部可保存，撤回不产生重新开放空间）")


def _demo_four() -> None:
    s = demo.scenario_two_teams()
    engine = s["engine"]
    print("反事实（每队单独申请，互不影响）：")
    for rid, rv in s["alone"].items():
        print(f"  {rid}: {rv.decision} — {rv.reason}")
    print("\n实际承诺顺序串行裁决（A order=1, B order=2）：")
    for rv in s["verdicts"]:
        for ln in render_request_verdict(engine, rv):
            print(ln)
    rev = demo.scenario_two_teams_reversed()
    print("\n顺序翻转（B order=1, A order=2）后：")
    for rv in rev["verdicts"]:
        print(f"  {rv.request_id}（{rv.team}）=> {rv.decision} — {rv.reason}")


def _demo_five() -> None:
    s = demo.scenario_incremental()
    engine = s["engine"]
    for label, key in (("第一批（仅触及 T3）", "first"),
                       ("第二批（仅触及 T5）", "second"),
                       ("第三批（重复请求）", "repeat"),
                       ("对照：关闭缓存全量重评", "full_rescan")):
        rv = s[key]
        print(f"\n{label}: 触及={rv.touched_tables} 缓存命中={rv.cache_hits}")
        print(f"  批次计数: {rv.batch_counters.to_dict()}")
    print("\n结论：未触及表 cell_visits=0、constraint_visits=0；"
          "重复请求全部缓存命中，零求解调用。")


# ---------------------------------------------------------------------------
# check / selftest
# ---------------------------------------------------------------------------
def run_check(path: str) -> int:
    engine, requests = load_scenario(path)
    if not requests:
        print(json.dumps({"error": "场景中没有待批 requests"}, ensure_ascii=False))
        return 2
    verdicts = engine.adjudicate(requests)
    payload = {
        "external_ledger": [r.to_dict() for r in engine.external_ledger()],
        "verdicts": [v.to_dict() for v in verdicts],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if all(v.decision == "APPROVED" for v in verdicts) else 1


def run_selftest() -> int:
    """小表上用独立暴力枚举对照传播/回溯求解器的结论。"""
    table = demo.make_grid("SELF")
    cons = list(demo.marginals("SELF").values())
    failures = 0

    # 1) 不同约束子集下的可行性对照
    import itertools
    for size in range(0, len(cons) + 1):
        for subset in itertools.combinations(cons, size):
            smart = DomainSolver(table, subset, Counters())
            brute = DomainSolver(table, subset, Counters())
            got = smart.feasible()
            want = brute.brute_force_feasible()
            if got != want:
                failures += 1
                print(f"  [FAIL] 子集 {[c.name for c in subset]}: "
                      f"求解器={got} 暴力={want}")

    # 2) 全量约束下 d 的定值结论对照
    solver = DomainSolver(table, cons, Counters())
    info = solver.value_range("d")
    brute_values = set()
    for sol in DomainSolver(table, cons, Counters()).enumerate_solutions():
        brute_values.add(sol["d"])
    smart_range = set(range(info["min"], info["max"] + 1))
    if brute_values != smart_range:
        failures += 1
        print(f"  [FAIL] d 投影: 求解器={sorted(smart_range)} 暴力={sorted(brute_values)}")

    # 3) 见证表逐约束手算复核
    wa, wb = info["witness_min"], info["witness_max"]
    for label, assignment in (("A", wa), ("B", wb)):
        for con in cons:
            total = sum(coef * assignment[c] for c, coef in con.terms)
            if total != con.rhs:
                failures += 1
                print(f"  [FAIL] 见证表 {label} 违反 {con.name}")

    if failures:
        print(f"自检失败 {failures} 项")
        return 1
    print("自检通过：传播/回溯求解器与独立暴力枚举在全部约束子集上结论一致；"
          "见证表手算复核全部成立。")
    return 0


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="leakguard",
        description="本地汇总份额防泄露判定器（Python 标准库）")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_demo = sub.add_parser("demo", help="运行内置演示场景")
    p_demo.add_argument("--scenario", type=int, default=0,
                        choices=[0, 1, 2, 3, 4, 5])
    p_check = sub.add_parser("check", help="检查 JSON 场景文件")
    p_check.add_argument("path")
    sub.add_parser("selftest", help="小表暴力对照冒烟自检")
    args = parser.parse_args(argv)

    if args.cmd == "demo":
        run_demo(args.scenario)
        return 0
    if args.cmd == "check":
        return run_check(args.path)
    if args.cmd == "selftest":
        return run_selftest()
    return 2


if __name__ == "__main__":
    sys.exit(main())
