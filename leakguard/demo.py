"""五个可复现演示场景（全部使用本地合成整数数据）。

运行：python -m leakguard.cli demo
每个场景同时返回结构化结果，测试与 CLI 共用同一份构造逻辑。
"""

from __future__ import annotations

from typing import Dict, List

from .engine import LeakGuardEngine
from .model import (
    Constraint,
    DisclosureRecord,
    RecordState,
    Request,
    Table,
)

UPPER = 9


# ---------------------------------------------------------------------------
# 公共构造：2x2 表 [a b; c d]，保护格 d，整数上限 U=9
# ---------------------------------------------------------------------------
def make_grid(table_id: str = "T1", upper: int = UPPER) -> Table:
    return Table(
        table_id=table_id,
        upper=upper,
        cells=["a", "b", "c", "d"],
        protected=["d"],
        rows=2,
        cols=2,
    )


def marginals(table_id: str = "T1") -> Dict[str, Constraint]:
    """五条汇总约束；任意单独一条都不暴露保护格 d。

    前四条是行/列份额（边际），第五条是对角份额。2x2 表的四条边际
    秩只有 3，单独或全部叠加都保留一个自由度；叠加对角份额后才把
    d 钉到唯一整数（解：a=3,b=4,c=2,d=2）。
    """
    return {
        "C1": Constraint.make(table_id, {"a": 1, "b": 1}, 7, "C1:行和R1=7"),
        "C2": Constraint.make(table_id, {"a": 1, "c": 1}, 5, "C2:列和K1=5"),
        "C3": Constraint.make(table_id, {"c": 1, "d": 1}, 4, "C3:行和R2=4"),
        "C4": Constraint.make(table_id, {"b": 1, "d": 1}, 6, "C4:列和K2=6"),
        "C5": Constraint.make(table_id, {"a": 1, "d": 1}, 5, "C5:对角和=5"),
    }


# ---------------------------------------------------------------------------
# 场景 1：单表合规的份额多次开放，叠加后唯一确定保护格
# ---------------------------------------------------------------------------
def scenario_marginal_inference() -> Dict[str, object]:
    table = make_grid()
    engine = LeakGuardEngine([table])
    cons = marginals()
    # 前四次开放：四种行/列份额单独看都合规，叠加后 d 仍有自由度
    for idx, key in enumerate(("C1", "C2", "C3", "C4"), start=1):
        engine.ingest(DisclosureRecord(
            f"PUB-{key}", "外部复核方", cons[key],
            RecordState.PUBLISHED, order=idx,
            note="每日开放的一种汇总格式"))
    before = engine.evaluate_table("T1")          # 此时 d 仍有多个合法值
    request = Request("REQ-5", "业务方B", 5, [cons["C5"]],
                      note="第五天申请开放对角份额")
    verdict = engine.evaluate_request(request)    # 叠加后 d 被唯一确定
    return {
        "name": "场景1·边际份额叠加推断",
        "engine": engine,
        "before_request": before,
        "request": request,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# 场景 2：保护格存在两个合法值 —— 给出两张可复核整数表
# ---------------------------------------------------------------------------
def scenario_two_witnesses() -> Dict[str, object]:
    table = make_grid()
    engine = LeakGuardEngine([table])
    cons = marginals()
    engine.ingest(DisclosureRecord(
        "PUB-C1", "外部复核方", cons["C1"],
        RecordState.PUBLISHED, order=1))
    verdict = engine.evaluate_table("T1")
    return {
        "name": "场景2·两个合法值与双见证表",
        "engine": engine,
        "table": table,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# 场景 3：撤回的两种语义，不能互相抵消
# ---------------------------------------------------------------------------
def scenario_withdrawal() -> Dict[str, object]:
    t1 = make_grid("T1")
    t2 = make_grid("T2")
    engine = LeakGuardEngine([t1, t2])
    m1, m2 = marginals("T1"), marginals("T2")

    # T1：C5 只批准未发布；已发布的 C2/C3 与它组合即可钉死 d
    draft = DisclosureRecord("DRAFT-C5", "业务方B", m1["C5"],
                             RecordState.APPROVED_DRAFT, order=1)
    engine.ingest(draft)
    for idx, key in enumerate(("C2", "C3"), start=2):
        engine.ingest(DisclosureRecord(
            f"T1-PUB-{key}", "外部复核方", m1[key],
            RecordState.PUBLISHED, order=idx))
    before_withdraw_t1 = engine.evaluate_table("T1")
    engine.withdraw("DRAFT-C5")
    after_withdraw_t1 = engine.evaluate_table("T1")  # 应恢复歧义

    # T2：同样的 C2/C3/C5 全部已发布，再撤回 C5 ——
    # 外部台账保留，d 仍被确定，撤回不能抵消
    for idx, key in enumerate(("C2", "C3", "C5"), start=1):
        engine.ingest(DisclosureRecord(
            f"T2-PUB-{key}", "外部复核方", m2[key],
            RecordState.PUBLISHED, order=10 + idx))
    before_withdraw_t2 = engine.evaluate_table("T2")
    engine.withdraw("T2-PUB-C5")
    after_withdraw_t2 = engine.evaluate_table("T2")   # 仍然 unique
    ledger_after = engine.external_ledger()
    republish_conflict = engine.republish_request(m2["C5"])
    return {
        "name": "场景3·撤回语义：未发布 vs 已可被外部保存",
        "engine": engine,
        "before_withdraw_t1": before_withdraw_t1,
        "after_withdraw_t1": after_withdraw_t1,
        "before_withdraw_t2": before_withdraw_t2,
        "after_withdraw_t2": after_withdraw_t2,
        "external_ledger": ledger_after,
        "republish_conflict": republish_conflict,
    }


# ---------------------------------------------------------------------------
# 场景 4：两队同时申请，各自安全、组合不安全，按承诺顺序裁决
# ---------------------------------------------------------------------------
def scenario_two_teams() -> Dict[str, object]:
    table = make_grid()
    cons = marginals()
    engine = LeakGuardEngine([table])
    # 最小定值集是 {C2,C3,C5}（3 条）。拆给两队后各自都不足 3 条，
    # 单独申请都安全；组合起来恰好构成定值集。
    r1 = Request("REQ-TEAM-A", "A队", order=1,
                 constraints=[cons["C2"]],
                 note="A队申请：列和K1")
    r2 = Request("REQ-TEAM-B", "B队", order=2,
                 constraints=[cons["C3"], cons["C5"]],
                 note="B队申请：行和R2 + 对角和")

    # 反事实：各自单独申请时都安全（任何单队约束都钉不死 d）
    alone = engine.counterfactual_order([r1, r2])
    # 实际：按 order 串行裁决，先到者占用安全余量，组合后 d 被定值
    verdicts = engine.adjudicate([r1, r2], publish=False)
    return {
        "name": "场景4·两队竞争与承诺顺序裁决",
        "engine": engine,
        "request_a": r1,
        "request_b": r2,
        "alone": alone,
        "verdicts": verdicts,
    }


def scenario_two_teams_reversed() -> Dict[str, object]:
    """同一批申请，承诺顺序翻转后裁决结果也翻转。"""
    table = make_grid()
    cons = marginals()
    engine = LeakGuardEngine([table])
    r1 = Request("REQ-TEAM-A", "A队", order=2,
                 constraints=[cons["C2"]])
    r2 = Request("REQ-TEAM-B", "B队", order=1,
                 constraints=[cons["C3"], cons["C5"]])
    verdicts = engine.adjudicate([r1, r2], publish=False)
    return {"name": "场景4b·顺序翻转", "engine": engine, "verdicts": verdicts}


# ---------------------------------------------------------------------------
# 场景 5：多批互不相关的表，增量评估只访问被触及的格子和约束
# ---------------------------------------------------------------------------
def scenario_incremental() -> Dict[str, object]:
    tables = [make_grid(f"T{i}") for i in range(1, 6)]  # 五张互不相关的表
    engine = LeakGuardEngine(tables)
    cons_by_table = {t.table_id: marginals(t.table_id) for t in tables}

    # 每张表先开放 C2/C3 两条份额（与 C5 组合才会钉死 d）
    for t in tables:
        cons = cons_by_table[t.table_id]
        for idx, key in enumerate(("C2", "C3"), start=1):
            engine.ingest(DisclosureRecord(
                f"{t.table_id}-PUB-{key}", "外部复核方", cons[key],
                RecordState.PUBLISHED, order=idx))

    # 夜间基线复评：无待批约束，把五张表的提交态全部预热进缓存
    baseline = engine.evaluate_request(
        Request("BASELINE", "审计", 99, constraints=[]))

    # 第一批：只触及 T3；申请 C5 会钉死 d，被否决且不落账，
    # 因此 T3 的基线缓存不被污染，其余四张表全部缓存命中、零访问
    req_t3 = Request("BATCH-1", "业务方C", 100,
                     [cons_by_table["T3"]["C5"]])
    first = engine.evaluate_request(req_t3)
    assert first.decision == "DENIED"

    # 第二批：只触及 T5 —— T1..T4（含上一批被否决的 T3）全部缓存命中
    req_t5 = Request("BATCH-2", "业务方D", 101,
                     [cons_by_table["T5"]["C5"]])
    second = engine.evaluate_request(req_t5)

    # 第三批：重复提交与第二批完全相同的请求 —— 全缓存、零求解
    repeat = engine.evaluate_request(
        Request("BATCH-2-DUP", "业务方D", 102,
                [cons_by_table["T5"]["C5"]]))

    # 对照：关闭缓存做全量重评
    full = engine.evaluate_request(
        Request("BATCH-FULL", "审计", 103,
                [cons_by_table["T5"]["C5"]]), use_cache=False)
    return {
        "name": "场景5·增量评估与访问计数",
        "engine": engine,
        "first": first,
        "second": second,
        "repeat": repeat,
        "full_rescan": full,
    }


ALL_SCENARIOS = [
    scenario_marginal_inference,
    scenario_two_witnesses,
    scenario_withdrawal,
    scenario_two_teams,
    scenario_two_teams_reversed,
    scenario_incremental,
]
