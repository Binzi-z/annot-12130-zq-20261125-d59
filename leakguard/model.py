"""核心数据模型：整数表、线性和约束、披露记录、访问计数器。

所有运算都在整数域内完成，不引入浮点，保证外部可逐格手算复核。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 约束：sum(coef_i * x_i) == rhs
# 最常见的“份额/边际”是 coef 全为 1 的子集和；保留系数以支持对角和等形式。
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Constraint:
    """一条线性整数等式约束。"""

    table_id: str
    terms: FrozenSet[Tuple[str, int]]  # (cell_id, coefficient)，冻结集合可哈希
    rhs: int
    name: str

    @staticmethod
    def make(table_id: str,
             terms: Dict[str, int],
             rhs: int,
             name: str = "") -> "Constraint":
        if not terms:
            raise ValueError("约束至少要包含一个格子")
        clean = frozenset((cell, int(coef)) for cell, coef in terms.items()
                          if coef != 0)
        if not clean:
            raise ValueError("约束系数不能全为 0")
        return Constraint(table_id, clean, int(rhs),
                          name or f"{table_id}:sum={rhs}")

    @property
    def cells(self) -> List[str]:
        return [cell for cell, _ in self.terms]

    def to_dict(self) -> dict:
        return {
            "table_id": self.table_id,
            "name": self.name,
            "rhs": self.rhs,
            "terms": {cell: coef for cell, coef in sorted(self.terms)},
        }

    @staticmethod
    def from_dict(d: dict) -> "Constraint":
        return Constraint.make(
            d["table_id"],
            {str(cell): int(coef) for cell, coef in d["terms"].items()},
            int(d["rhs"]),
            str(d.get("name", "")),
        )


# ---------------------------------------------------------------------------
# 披露记录的生命周期状态
# ---------------------------------------------------------------------------
class RecordState(str, Enum):
    """披露记录状态。

    APPROVED_DRAFT  已批准但从未对外发布（撤回后直接消失，不进入外部台账）
    PUBLISHED       已经对外发布，外部可能已保存（永远留在追加式台账）
    WITHDRAWN_DRAFT     撤回“从未发布”的记录：约束失效，且可重新申请
    WITHDRAWN_PUBLISHED 撤回“已发布”的记录：内部停用，但外部台账保留，
                        判定时该约束仍计入外部已知信息，不能被撤回抵消
    DENIED          申请被否决，不产生任何约束
    """

    APPROVED_DRAFT = "approved_draft"
    PUBLISHED = "published"
    WITHDRAWN_DRAFT = "withdrawn_draft"
    WITHDRAWN_PUBLISHED = "withdrawn_published"
    DENIED = "denied"


# 判定“当前外部/内部约束宇宙”时计入的状态。
# 注意 WITHDRAWN_PUBLISHED 仍然计入 —— 已可被外部保存的数据无法真正收回。
ACTIVE_STATES = frozenset({
    RecordState.APPROVED_DRAFT,
    RecordState.PUBLISHED,
    RecordState.WITHDRAWN_PUBLISHED,
})

# 只代表“外部确实拿到过”的状态（追加式、只增不减）。
EXTERNAL_LEDGER_STATES = frozenset({
    RecordState.PUBLISHED,
    RecordState.WITHDRAWN_PUBLISHED,
})


@dataclass
class DisclosureRecord:
    """一条披露申请/记录及其生命周期。"""

    record_id: str
    team: str
    constraint: Constraint
    state: RecordState
    order: int                      # 实际承诺顺序（时间戳/序号），越小越早
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "team": self.team,
            "order": self.order,
            "state": self.state.value,
            "note": self.note,
            "constraint": self.constraint.to_dict(),
        }


@dataclass
class Table:
    """一张整数表：每个格子有闭区间整数域 [lo, hi]。"""

    table_id: str
    upper: int                                  # 合成的整数上限 U
    lower: int = 0
    cells: List[str] = field(default_factory=list)
    protected: List[str] = field(default_factory=list)
    rows: int = 0                               # 可选：二维展示布局
    cols: int = 0
    cell_bounds: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    def domain_of(self, cell: str) -> Tuple[int, int]:
        if cell not in self.cells:
            raise KeyError(f"表 {self.table_id} 中不存在格子 {cell}")
        return self.cell_bounds.get(cell, (self.lower, self.upper))

    def render(self, assignment: Dict[str, int]) -> str:
        """把一张完整整数赋值渲染成可复核文本表。"""
        if self.rows and self.cols and len(self.cells) == self.rows * self.cols:
            width = max(len(str(c)) for c in self.cells) + 2
            blocks = []
            for r in range(self.rows):
                row = self.cells[r * self.cols:(r + 1) * self.cols]
                blocks.append("".join(f"{c:>{width}}" for c in row))
                blocks.append("".join(f"{assignment[c]:>{width}}" for c in row))
            return "\n".join(blocks)
        width = max(len(str(c)) for c in self.cells)
        return "\n".join(f"{c:>{width}} = {assignment[c]}" for c in self.cells)

    def to_dict(self) -> dict:
        return {
            "table_id": self.table_id,
            "lower": self.lower,
            "upper": self.upper,
            "cells": list(self.cells),
            "protected": list(self.protected),
            "rows": self.rows,
            "cols": self.cols,
            "cell_bounds": {c: list(b) for c, b in self.cell_bounds.items()},
        }

    @staticmethod
    def from_dict(d: dict) -> "Table":
        lower = int(d.get("lower", 0))
        upper = int(d["upper"])
        cells = [str(c) for c in d["cells"]]
        bounds = {
            str(c): (int(b[0]), int(b[1]))
            for c, b in d.get("cell_bounds", {}).items()
        }
        for cell, (lo, hi) in bounds.items():
            if cell not in cells:
                raise ValueError(f"cell_bounds 引用了未知格子 {cell}")
            if lo > hi:
                raise ValueError(f"格子 {cell} 的下界大于上界")
        protected = [str(c) for c in d.get("protected", [])]
        for cell in protected:
            if cell not in cells:
                raise ValueError(f"保护格 {cell} 不在表 {d['table_id']} 中")
        return Table(
            table_id=str(d["table_id"]),
            upper=upper,
            lower=lower,
            cells=cells,
            protected=protected,
            rows=int(d.get("rows", 0)),
            cols=int(d.get("cols", 0)),
            cell_bounds=bounds,
        )


@dataclass
class Request:
    """待批申请：一个团队一次可申请多条约束（可跨表）。"""

    request_id: str
    team: str
    order: int
    constraints: List[Constraint]
    note: str = ""


# ---------------------------------------------------------------------------
# 访问计数器：用于证明“增量评估只重新访问被触及的格子和约束”
# ---------------------------------------------------------------------------
@dataclass
class Counters:
    """一次批次内求解器实际访问量的测量结果。"""

    cell_visits: int = 0       # 域被读取/收紧的格子次数
    constraint_visits: int = 0  # 约束被传播/校验的次数
    search_nodes: int = 0      # 回溯搜索展开的节点数
    solver_runs: int = 0       # 求解/可行性调用次数
    table_evaluations: int = 0  # 真正重新判定的表数
    cache_hits: int = 0        # 因表未被触及而直接复用缓存的表数

    def reset(self) -> None:
        self.cell_visits = 0
        self.constraint_visits = 0
        self.search_nodes = 0
        self.solver_runs = 0
        self.table_evaluations = 0
        self.cache_hits = 0

    def add(self, other: "Counters") -> None:
        self.cell_visits += other.cell_visits
        self.constraint_visits += other.constraint_visits
        self.search_nodes += other.search_nodes
        self.solver_runs += other.solver_runs
        self.table_evaluations += other.table_evaluations
        self.cache_hits += other.cache_hits

    def to_dict(self) -> dict:
        return asdict(self)
