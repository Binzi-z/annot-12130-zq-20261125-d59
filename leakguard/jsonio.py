"""JSON 场景文件的载入与结果序列化（仅用标准库 json）。

场景文件结构见 examples/scenario_sample.json 与 README。
"""

from __future__ import annotations

import json
from typing import List, Tuple

from .engine import LeakGuardEngine
from .model import Constraint, DisclosureRecord, RecordState, Request, Table


def load_scenario(path: str) -> Tuple[LeakGuardEngine, List[Request]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return build_scenario(data)


def build_scenario(data: dict) -> Tuple[LeakGuardEngine, List[Request]]:
    tables = [Table.from_dict(d) for d in data["tables"]]
    engine = LeakGuardEngine(tables, node_limit=int(data.get("node_limit", 200_000)))

    for d in data.get("records", []):
        state = RecordState(d["state"])
        rec = DisclosureRecord(
            record_id=str(d["record_id"]),
            team=str(d.get("team", "legacy")),
            constraint=Constraint.from_dict(d["constraint"]),
            state=state,
            order=int(d["order"]),
            note=str(d.get("note", "")),
        )
        engine.ingest(rec)

    requests: List[Request] = []
    for d in data.get("requests", []):
        req = Request(
            request_id=str(d["request_id"]),
            team=str(d["team"]),
            order=int(d["order"]),
            constraints=[Constraint.from_dict(c) for c in d["constraints"]],
            note=str(d.get("note", "")),
        )
        requests.append(req)
    return engine, requests


def dumps_verdict(obj, indent: int = 2) -> str:
    """把 RequestVerdict/TableVerdict 等 dataclass 结果序列化为 JSON。"""
    return json.dumps(_to_jsonable(obj), ensure_ascii=False, indent=indent)


def _to_jsonable(obj):
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, RecordState):
        return obj.value
    return obj
