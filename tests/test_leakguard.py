"""测试套件：纯标准库 unittest，运行 python -m unittest discover -v。

覆盖
----
1. 求解器：所有约束子集上与独立暴力枚举对照、min/max 与见证表；
2. 证书：最小子集的不可再删性、最小基数保证、三种独立复核；
3. 引擎：双见证、定值+最小集、撤回两态不可抵消、顺序裁决与翻转、
   增量缓存零访问、fail-closed；
4. JSON 往返与 CLI 冒烟。
"""

import itertools
import json
import os
import unittest

from leakguard import demo
from leakguard.certificate import (
    find_minimal_subset,
    verify_forced_value,
    verify_witness_pair,
)
from leakguard.engine import LeakGuardEngine
from leakguard.model import (
    Constraint,
    Counters,
    DisclosureRecord,
    RecordState,
    Request,
    Table,
)
from leakguard.solver import BacktrackLimit, DomainSolver


def grid(table_id="T1", upper=9):
    return demo.make_grid(table_id, upper)


def cons(table_id="T1"):
    return demo.marginals(table_id)


class SolverTests(unittest.TestCase):
    def test_feasibility_matches_brute_force_all_subsets(self):
        table = grid()
        all_cons = list(cons().values())
        for size in range(len(all_cons) + 1):
            for subset in itertools.combinations(all_cons, size):
                smart = DomainSolver(table, subset, Counters())
                brute = DomainSolver(table, subset, Counters())
                self.assertEqual(
                    smart.feasible(),
                    brute.brute_force_feasible(),
                    msg=f"可行性分歧: {[c.name for c in subset]}")

    def test_extremes_match_brute_force_enumeration(self):
        table = grid()
        all_cons = list(cons().values())
        # 全量约束下 d 的投影与穷举一致
        solver = DomainSolver(table, all_cons, Counters())
        info = solver.value_range("d")
        self.assertEqual(info["status"], "feasible")
        brute_values = {
            sol["d"] for sol in
            DomainSolver(table, all_cons, Counters()).enumerate_solutions()
        }
        self.assertEqual(set(range(info["min"], info["max"] + 1)),
                         brute_values)

    def test_witness_assignments_are_integral_and_satisfy_constraints(self):
        table = grid()
        solver = DomainSolver(table, [cons()["C1"]], Counters())
        info = solver.value_range("d")
        self.assertLess(info["min"], info["max"])
        for witness in (info["witness_min"], info["witness_max"]):
            self.assertTrue(all(isinstance(v, int) for v in witness.values()))
            for cell, value in witness.items():
                lo, hi = table.domain_of(cell)
                self.assertTrue(lo <= value <= hi)
            self.assertEqual(witness["a"] + witness["b"], 7)

    def test_propagation_detects_unsatisfiable_equation(self):
        table = grid("U")
        bad = Constraint.make("U", {"a": 1, "b": 1}, 99, "超界和")
        solver = DomainSolver(table, [bad], Counters())
        self.assertIsNone(solver.solve())

    def test_negative_coefficient_propagation(self):
        # a - d == 1，配合 a+d == 5（C5）可在整数域解出 a=3,d=2
        table = grid("N")
        diff = Constraint.make("N", {"a": 1, "d": -1}, 1, "a-d=1")
        solver = DomainSolver(table, [cons("N")["C5"], diff], Counters())
        info = solver.value_range("d")
        self.assertEqual((info["min"], info["max"]), (2, 2))

    def test_backtrack_limit_is_fail_closed(self):
        table = Table("BIG", upper=999, cells=["a", "b", "c", "d"],
                      protected=["d"])
        solver = DomainSolver(table, [], Counters(), node_limit=2)
        with self.assertRaises(BacktrackLimit):
            solver.solve()


class CertificateTests(unittest.TestCase):
    def test_minimal_subset_is_irredundant_and_minimum_cardinality(self):
        table = grid()
        all_cons = list(cons().values())
        result = find_minimal_subset(table, all_cons, "d")
        subset = result["subset"]
        self.assertEqual(result["guarantee"], "minimum-cardinality")
        # 已知最小定值集为 3 条
        self.assertEqual(len(subset), 3)
        # 删去任意一条都不再定值
        for con in subset:
            trial = [c for c in subset if c is not con]
            solver = DomainSolver(table, trial, Counters())
            info = solver.value_range("d")
            self.assertTrue(info["status"] != "feasible"
                            or info["min"] != info["max"])

    def test_independent_verification_three_methods(self):
        table = grid()
        all_cons = list(cons().values())
        result = find_minimal_subset(table, all_cons, "d")
        report = verify_forced_value(table, result["subset"], "d", 2)
        self.assertTrue(report["fresh_solver"]["passed"])
        self.assertTrue(report["brute_force"]["available"])
        self.assertTrue(report["brute_force"]["passed"])
        self.assertEqual(report["brute_force"]["observed_target_values"], [2])
        self.assertTrue(report["hand_check"]["passed"])
        self.assertTrue(report["all_available_passed"])

    def test_witness_pair_check_rejects_same_value_and_bad_tables(self):
        table = grid()
        constraints = [cons()["C1"]]
        ok = verify_witness_pair(
            table, constraints, "d",
            {"a": 1, "b": 6, "c": 0, "d": 0},
            {"a": 2, "b": 5, "c": 0, "d": 9})
        self.assertTrue(ok["passed"])
        bad = verify_witness_pair(
            table, constraints, "d",
            {"a": 1, "b": 6, "c": 0, "d": 0},
            {"a": 1, "b": 6, "c": 0, "d": 0})
        self.assertFalse(bad["passed"])
        bad2 = verify_witness_pair(
            table, constraints, "d",
            {"a": 9, "b": 9, "c": 0, "d": 0},  # a+b=18 != 7
            {"a": 2, "b": 5, "c": 0, "d": 9})
        self.assertFalse(bad2["passed"])


class EngineTests(unittest.TestCase):
    def test_ambiguous_cell_emits_two_witness_tables(self):
        s = demo.scenario_two_witnesses()
        cv = s["verdict"].cell_verdicts[0]
        self.assertEqual(cv.status, "ambiguous")
        self.assertTrue(cv.witness_check["passed"])
        self.assertNotEqual(cv.witness_a["d"], cv.witness_b["d"])

    def test_unique_cell_emits_minimal_subset_and_verification(self):
        s = demo.scenario_marginal_inference()
        # 申请前仍歧义
        pre = s["before_request"].cell_verdicts[0]
        self.assertEqual(pre.status, "ambiguous")
        tv = next(t for t in s["verdict"].table_verdicts
                  if t.table_id == "T1")
        cv = next(c for c in tv.cell_verdicts if c.cell == "d")
        self.assertEqual(cv.status, "unique")
        self.assertEqual(cv.forced_value, 2)
        self.assertEqual(cv.subset_guarantee, "minimum-cardinality")
        self.assertTrue(cv.independent_verification["all_available_passed"])

    def test_withdraw_draft_removes_constraint_but_published_remains(self):
        s = demo.scenario_withdrawal()
        eng = s["engine"]
        self.assertEqual(s["before_withdraw_t1"].cell_verdicts[0].status,
                         "unique")
        self.assertEqual(s["after_withdraw_t1"].cell_verdicts[0].status,
                         "ambiguous")
        self.assertEqual(s["after_withdraw_t2"].cell_verdicts[0].status,
                         "unique")
        # 撤回的已发布记录仍在外部台账
        ledger_ids = {r.record_id for r in eng.external_ledger()}
        self.assertIn("T2-PUB-C5", ledger_ids)
        # 且仍参与约束宇宙
        t2_active = eng.active_constraints("T2")
        self.assertTrue(
            any(c.name == "C5:对角和=5" for c in t2_active))
        # 重新申请同一约束会命中台账冲突
        self.assertEqual(s["republish_conflict"], "T2-PUB-C5")
        # 两类撤回状态互不抵消：计数各自独立存在
        states = {r.record_id: r.state for r in eng.records.values()}
        self.assertEqual(states["DRAFT-C5"],
                         RecordState.WITHDRAWN_DRAFT)
        self.assertEqual(states["T2-PUB-C5"],
                         RecordState.WITHDRAWN_PUBLISHED)
        # 未发布撤回记录不再产生约束
        t1_active = eng.active_constraints("T1")
        self.assertFalse(
            any(c.name == "C5:对角和=5" for c in t1_active))

    def test_two_teams_order_adjudication_and_reversal(self):
        s = demo.scenario_two_teams()
        for rv in s["alone"].values():
            self.assertEqual(rv.decision, "APPROVED")
        first, second = s["verdicts"]
        self.assertEqual(first.request_id, "REQ-TEAM-A")
        self.assertEqual(first.decision, "APPROVED")
        self.assertEqual(second.request_id, "REQ-TEAM-B")
        self.assertEqual(second.decision, "DENIED")
        self.assertIn("d=2", second.reason)
        # 翻转顺序后 B 获批、A 被否
        rev = demo.scenario_two_teams_reversed()
        ids = {rv.request_id: rv.decision for rv in rev["verdicts"]}
        self.assertEqual(ids["REQ-TEAM-B"], "APPROVED")
        self.assertEqual(ids["REQ-TEAM-A"], "DENIED")

    def test_incremental_only_touches_touched_table(self):
        s = demo.scenario_incremental()
        second = s["second"]
        self.assertEqual(second.touched_tables, ["T5"])
        self.assertEqual(sorted(second.cache_hits),
                         ["T1", "T2", "T3", "T4"])
        # 未触及表在本批次的增量访问全部为 0（历史计数保留在 counters 中）
        for tv in second.table_verdicts:
            if tv.table_id != "T5":
                self.assertTrue(tv.from_cache)
                self.assertEqual(tv.batch_delta.cell_visits, 0)
                self.assertEqual(tv.batch_delta.constraint_visits, 0)
                self.assertEqual(tv.batch_delta.solver_runs, 0)
        # 被触及表确有实际访问
        t5 = next(tv for tv in second.table_verdicts if tv.table_id == "T5")
        self.assertGreater(t5.batch_delta.cell_visits, 0)
        self.assertGreater(t5.batch_delta.solver_runs, 0)
        # 重复请求：所有表都命中缓存，批次零求解调用
        repeat = s["repeat"]
        self.assertEqual(repeat.batch_counters.solver_runs, 0)
        self.assertEqual(len(repeat.cache_hits), 5)
        # 关闭缓存的全量重评访问量严格更大
        full = s["full_rescan"]
        self.assertGreater(
            full.batch_counters.cell_visits,
            second.batch_counters.cell_visits)
        self.assertEqual(full.cache_hits, [])

    def test_inconclusive_request_is_denied_fail_closed(self):
        table = Table("BIG", upper=999, cells=["a", "b", "c", "d"],
                      protected=["d"])
        engine = LeakGuardEngine([table], node_limit=1)
        req = Request("R1", "X", 1,
                      [Constraint.make("BIG", {"a": 1, "b": 1}, 1000)])
        verdict = engine.evaluate_request(req)
        self.assertEqual(verdict.decision, "DENIED")
        self.assertIn("fail-closed", verdict.reason)


class JsonAndCliTests(unittest.TestCase):
    def test_json_roundtrip_via_example(self):
        from leakguard.jsonio import dumps_verdict, load_scenario
        path = os.path.join(os.path.dirname(__file__), "..", "examples",
                            "scenario_sample.json")
        engine, requests = load_scenario(path)
        verdicts = engine.adjudicate(requests)
        payload = json.loads(dumps_verdict(verdicts[0]))
        self.assertEqual(payload["decision"], "DENIED")
        cell = payload["tables"][0]["cells"][0]
        self.assertEqual(cell["status"], "unique")
        self.assertEqual(cell["forced_value"], 2)

    def test_cli_selftest_command(self):
        from leakguard.cli import run_selftest
        self.assertEqual(run_selftest(), 0)

    def test_cli_check_command_exit_code_on_denial(self):
        from leakguard.cli import run_check
        path = os.path.join(os.path.dirname(__file__), "..", "examples",
                            "scenario_sample.json")
        # 示例场景的待批申请会钉死保护格，退出码为 1
        self.assertEqual(run_check(path), 1)

    def test_cli_demo_scenario_index_is_complete(self):
        # 轻量校验：演示场景登记完整；重计算已由上面的引擎测试覆盖
        from leakguard.demo import ALL_SCENARIOS
        self.assertEqual(len(ALL_SCENARIOS), 6)
        names = [fn.__name__ for fn in ALL_SCENARIOS]
        self.assertIn("scenario_two_witnesses", names)
        self.assertIn("scenario_incremental", names)


if __name__ == "__main__":
    unittest.main()
