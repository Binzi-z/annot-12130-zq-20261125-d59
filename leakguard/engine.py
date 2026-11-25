"""防泄露判定引擎。

职责
----
1. 维护披露记录生命周期与追加式外部台账（撤回已发布数据不会抵消外部已知）；
2. 对每张表的每个保护格做定值/歧义判定，输出双见证整数表或最小约束子集；
3. 按实际承诺顺序（order）对多队申请做裁决，并支持反事实翻转演示；
4. 批次级增量评估：按约束宇宙签名缓存判定结果，只重新求解被触及的表，
   并用量化计数器证明未触及表零访问。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from . import certificate
from .model import (
    ACTIVE_STATES,
    EXTERNAL_LEDGER_STATES,
    Constraint,
    Counters,
    DisclosureRecord,
    RecordState,
    Request,
    Table,
)
from .solver import BacktrackLimit, DomainSolver


# ---------------------------------------------------------------------------
# 判定结果数据结构
# ---------------------------------------------------------------------------
@dataclass
class CellVerdict:
    table_id: str
    cell: str
    status: str                       # ambiguous / unique / infeasible / inconclusive
    forced_value: Optional[int] = None
    value_min: Optional[int] = None
    value_max: Optional[int] = None
    witness_a: Optional[Dict[str, int]] = None
    witness_b: Optional[Dict[str, int]] = None
    witness_check: Optional[dict] = None
    minimal_subset: Optional[List[Constraint]] = None
    subset_guarantee: Optional[str] = None
    independent_verification: Optional[dict] = None

    def to_dict(self) -> dict:
        out = {
            "table_id": self.table_id,
            "cell": self.cell,
            "status": self.status,
            "forced_value": self.forced_value,
            "value_min": self.value_min,
            "value_max": self.value_max,
        }
        if self.status == "ambiguous":
            out["witness_table_a"] = self.witness_a
            out["witness_table_b"] = self.witness_b
            out["witness_check"] = self.witness_check
        if self.status == "unique":
            out["minimal_subset"] = [c.to_dict() for c in self.minimal_subset]
            out["subset_guarantee"] = self.subset_guarantee
            out["independent_verification"] = self.independent_verification
        return out


@dataclass
class TableVerdict:
    table_id: str
    cell_verdicts: List[CellVerdict]
    from_cache: bool
    counters: Counters
    signature: str
    # 本批次内该表实际新增的访问量；缓存命中时全为 0，
    # 与 counters（首次求解的历史计数）严格区分。
    batch_delta: Optional[Counters] = None

    @property
    def safe(self) -> bool:
        return all(cv.status == "ambiguous" for cv in self.cell_verdicts)

    @property
    def inconclusive(self) -> bool:
        return any(cv.status in ("inconclusive", "infeasible")
                   for cv in self.cell_verdicts)

    def to_dict(self) -> dict:
        return {
            "table_id": self.table_id,
            "safe": self.safe,
            "from_cache": self.from_cache,
            "signature": self.signature,
            "counters": self.counters.to_dict(),
            "batch_delta": (self.batch_delta.to_dict()
                            if self.batch_delta is not None else None),
            "cells": [cv.to_dict() for cv in self.cell_verdicts],
        }


@dataclass
class RequestVerdict:
    request_id: str
    team: str
    order: int
    decision: str                     # APPROVED / DENIED
    reason: str
    table_verdicts: List[TableVerdict]
    batch_counters: Counters
    touched_tables: List[str]
    cache_hits: List[str]

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "team": self.team,
            "order": self.order,
            "decision": self.decision,
            "reason": self.reason,
            "touched_tables": self.touched_tables,
            "cache_hits": self.cache_hits,
            "batch_counters": self.batch_counters.to_dict(),
            "tables": [tv.to_dict() for tv in self.table_verdicts],
        }


# ---------------------------------------------------------------------------
# 引擎主体
# ---------------------------------------------------------------------------
class LeakGuardEngine:
    def __init__(self, tables: Sequence[Table], node_limit: int = 200_000):
        self.tables: Dict[str, Table] = {t.table_id: t for t in tables}
        if len(self.tables) != len(tables):
            raise ValueError("表 ID 重复")
        self.records: Dict[str, DisclosureRecord] = {}
        self.node_limit = node_limit
        self._cache: Dict[str, TableVerdict] = {}
        self._counter = Counters()

    # ------------------------------------------------------------------
    # 记录与生命周期
    # ------------------------------------------------------------------
    def ingest(self, record: DisclosureRecord) -> None:
        if record.record_id in self.records:
            raise ValueError(f"记录 ID 重复: {record.record_id}")
        if record.constraint.table_id not in self.tables:
            raise KeyError(f"未知表 {record.constraint.table_id}")
        self.records[record.record_id] = record
        # 只失效受影响表的缓存；其它表的约束宇宙未变，签名仍有效
        self._invalidate(record.constraint.table_id)

    def _invalidate(self, table_id: str) -> None:
        prefix = table_id + "::"
        for key in [k for k in self._cache if k.startswith(prefix)]:
            del self._cache[key]

    def _records_by_state(self) -> Dict[RecordState, List[DisclosureRecord]]:
        buckets: Dict[RecordState, List[DisclosureRecord]] = {}
        for rec in self.records.values():
            buckets.setdefault(rec.state, []).append(rec)
        return buckets

    def active_constraints(self, table_id: str,
                           extra: Sequence[Constraint] = ()) -> List[Constraint]:
        """当前约束宇宙：生效记录 + 待批临时约束。

        WITHDRAWN_PUBLISHED 仍计入 —— 外部可能已保存，撤回不能抵消。
        """
        result = [rec.constraint for rec in self.records.values()
                  if rec.state in ACTIVE_STATES
                  and rec.constraint.table_id == table_id]
        result.extend(con for con in extra if con.table_id == table_id)
        # 按 (项集合, 右端) 去重，保留首次出现并给出稳定顺序
        seen, unique = set(), []
        for con in result:
            key = (con.terms, con.rhs)
            if key not in seen:
                seen.add(key)
                unique.append(con)
        unique.sort(key=lambda c: (c.name, c.rhs, sorted(c.terms)))
        return unique

    def external_ledger(self, table_id: Optional[str] = None
                        ) -> List[DisclosureRecord]:
        """追加式外部台账：所有曾经 PUBLISHED 的记录，只增不减。"""
        out = [rec for rec in self.records.values()
               if rec.state in EXTERNAL_LEDGER_STATES]
        if table_id is not None:
            out = [rec for rec in out if rec.constraint.table_id == table_id]
        return sorted(out, key=lambda r: r.order)

    def withdraw(self, record_id: str) -> DisclosureRecord:
        """撤回一条记录，两种撤回严格区分、互不抵消。"""
        rec = self.records[record_id]
        if rec.state == RecordState.PUBLISHED:
            rec.state = RecordState.WITHDRAWN_PUBLISHED
        elif rec.state == RecordState.APPROVED_DRAFT:
            rec.state = RecordState.WITHDRAWN_DRAFT
        else:
            raise ValueError(
                f"记录 {record_id} 状态为 {rec.state.value}，不可撤回")
        self._invalidate(rec.constraint.table_id)
        return rec

    def republish_request(self, constraint: Constraint) -> Optional[str]:
        """重新申请一条约束时，检查它是否早已被外部保存。

        返回冲突记录 ID（说明撤回无法真正收回），否则 None。
        """
        for rec in self.external_ledger(constraint.table_id):
            if (rec.constraint.terms, rec.constraint.rhs) == \
               (constraint.terms, constraint.rhs):
                return rec.record_id
        return None

    # ------------------------------------------------------------------
    # 签名与增量缓存
    # ------------------------------------------------------------------
    def _signature(self, table_id: str,
                   constraints: Sequence[Constraint]) -> str:
        parts = [f"bounds:{sorted(self.tables[table_id].cell_bounds.items())}",
                 f"protected:{tuple(self.tables[table_id].protected)}"]
        for con in constraints:
            parts.append(f"{con.name}|{con.rhs}|{sorted(con.terms)}")
        return table_id + "::" + ";;".join(parts)

    # ------------------------------------------------------------------
    # 单表判定
    # ------------------------------------------------------------------
    def evaluate_table(self, table_id: str,
                       extra: Sequence[Constraint] = (),
                       use_cache: bool = True) -> TableVerdict:
        table = self.tables[table_id]
        constraints = self.active_constraints(table_id, extra)
        signature = self._signature(table_id, constraints)
        if use_cache and signature in self._cache:
            cached = self._cache[signature]
            self._counter.cache_hits += 1
            # 命中缓存：本批次对该表零访问，返回带零增量的副本，不污染缓存对象
            return TableVerdict(
                cached.table_id, cached.cell_verdicts, True,
                cached.counters, cached.signature, batch_delta=Counters())

        counters = Counters()
        solver = DomainSolver(table, constraints, counters, self.node_limit)
        cell_verdicts: List[CellVerdict] = []
        for target in table.protected:
            cell_verdicts.append(
                self._judge_cell(table, constraints, target, solver, counters))
        verdict = TableVerdict(table_id, cell_verdicts, False,
                               counters, signature,
                               batch_delta=Counters(**counters.to_dict()))
        self._cache[signature] = verdict
        self._counter.table_evaluations += 1
        self._counter.add(counters)
        return verdict

    def _judge_cell(self, table: Table, constraints: List[Constraint],
                    target: str, solver: DomainSolver,
                    counters: Counters) -> CellVerdict:
        try:
            info = solver.value_range(target)
        except BacktrackLimit:
            return CellVerdict(table.table_id, target, "inconclusive")

        if info["status"] == "infeasible":
            return CellVerdict(table.table_id, target, "infeasible")
        if info["status"] == "inconclusive":
            return CellVerdict(table.table_id, target, "inconclusive")

        vmin, vmax = info["min"], info["max"]
        if vmin < vmax:
            # 存在两个合法值 —— 两张可复核整数表
            wa, wb = info["witness_min"], info["witness_max"]
            check = certificate.verify_witness_pair(
                table, constraints, target, wa, wb)
            return CellVerdict(table.table_id, target, "ambiguous",
                               value_min=vmin, value_max=vmax,
                               witness_a=wa, witness_b=wb, witness_check=check)

        # 只剩一个值 —— 最小已开放约束集合 + 独立校验
        minimal = certificate.find_minimal_subset(
            table, constraints, target, self.node_limit)
        subset = minimal["subset"]
        verification = certificate.verify_forced_value(
            table, subset, target, vmin, self.node_limit)
        provenance = []
        for con in subset:
            origin = next(
                (rec.record_id for rec in self.records.values()
                 if rec.constraint.table_id == table.table_id
                 and (rec.constraint.terms, rec.constraint.rhs)
                 == (con.terms, con.rhs)),
                "pending-request")
            provenance.append({"record_id": origin, "constraint": con.to_dict()})
        verification["subset_provenance"] = provenance
        return CellVerdict(table.table_id, target, "unique",
                           forced_value=vmin, value_min=vmin, value_max=vmax,
                           minimal_subset=subset,
                           subset_guarantee=minimal["guarantee"],
                           independent_verification=verification)

    # ------------------------------------------------------------------
    # 批次/申请裁决
    # ------------------------------------------------------------------
    def begin_batch(self) -> Counters:
        self._counter = Counters()
        return self._counter

    def evaluate_request(self, request: Request,
                         use_cache: bool = True) -> RequestVerdict:
        """在当前已生效记录之上，临时加入申请约束做判定（不修改台账）。"""
        batch = self.begin_batch()
        touched = sorted({con.table_id for con in request.constraints})
        table_verdicts: List[TableVerdict] = []
        cache_hits: List[str] = []

        # 未触及的表也纳入同一批次测量，以展示缓存命中与零访问
        for table_id in sorted(self.tables):
            extra = request.constraints if table_id in touched else ()
            before = (batch.cell_visits, batch.constraint_visits,
                      batch.search_nodes, batch.solver_runs)
            tv = self.evaluate_table(table_id, extra, use_cache=use_cache)
            after = (batch.cell_visits, batch.constraint_visits,
                     batch.search_nodes, batch.solver_runs)
            # 该表在本批次的真实增量（缓存命中时四个差值全为 0）
            tv.batch_delta = Counters(
                cell_visits=after[0] - before[0],
                constraint_visits=after[1] - before[1],
                search_nodes=after[2] - before[2],
                solver_runs=after[3] - before[3])
            table_verdicts.append(tv)
            if tv.from_cache and after == before:
                cache_hits.append(table_id)

        unsafe = [tv for tv in table_verdicts
                  if any(cv.status == "unique" for cv in tv.cell_verdicts)]
        inconclusive = [tv for tv in table_verdicts if tv.inconclusive]
        if inconclusive:
            decision, reason = "DENIED", (
                "求解预算内无法完成定值判定，按 fail-closed 否决")
        elif unsafe:
            names = [f"{tv.table_id}:{cv.cell}={cv.forced_value}"
                     for tv in unsafe for cv in tv.cell_verdicts
                     if cv.status == "unique"]
            decision, reason = "DENIED", "保护格被唯一确定：" + "；".join(names)
        else:
            decision, reason = "APPROVED", "所有保护格均保持至少两个合法值"

        return RequestVerdict(request.request_id, request.team, request.order,
                              decision, reason, table_verdicts,
                              batch, touched, cache_hits)

    def commit(self, request: Request, publish: bool = False) -> List[DisclosureRecord]:
        """裁决通过后落账：默认成为 approved_draft；publish=True 直接发布。"""
        state = RecordState.PUBLISHED if publish else RecordState.APPROVED_DRAFT
        created = []
        for idx, con in enumerate(request.constraints):
            rid = f"{request.request_id}:c{idx}"
            rec = DisclosureRecord(rid, request.team, con, state,
                                   request.order, request.note)
            self.ingest(rec)
            created.append(rec)
        return created

    def adjudicate(self, requests: Sequence[Request],
                   publish: bool = False) -> List[RequestVerdict]:
        """按实际承诺顺序（order）串行裁决，通过者立即落账。

        两队各自安全、组合不安全时，order 较小者先获批并进入约束宇宙，
        后者因此被否决 —— 顺序决定谁能用掉这份“安全余量”。
        """
        verdicts = []
        for request in sorted(requests, key=lambda r: r.order):
            verdict = self.evaluate_request(request)
            verdicts.append(verdict)
            if verdict.decision == "APPROVED":
                self.commit(request, publish=publish)
            else:
                rid = request.request_id
                self.ingest(DisclosureRecord(
                    rid, request.team, request.constraints[0],
                    RecordState.DENIED, request.order,
                    f"已否决：{verdict.reason}"))
        return verdicts

    def counterfactual_order(self, requests: Sequence[Request]
                             ) -> Dict[str, RequestVerdict]:
        """反事实：分别假设每个申请先到（其余都不存在），用于展示顺序翻转。

        不修改引擎当前台账；每次都在同一已落账快照上做独立模拟。
        """
        out: Dict[str, RequestVerdict] = {}
        for request in requests:
            sim = LeakGuardEngine(list(self.tables.values()), self.node_limit)
            for rec in self.records.values():
                sim.ingest(DisclosureRecord(
                    rec.record_id, rec.team, rec.constraint,
                    rec.state, rec.order, rec.note))
            out[request.request_id] = sim.evaluate_request(request)
        return out
