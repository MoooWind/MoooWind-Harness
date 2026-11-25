import unittest

import task_orchestrator
import tool_registry


class TaskOrchestratorTests(unittest.TestCase):
    def test_parses_ordered_plan_with_declared_dependencies(self):
        response = (
            '{"tasks":['
            '{"id":"derive","goal":"求导","tool":"math.differentiate",'
            '"arguments":{"expression":"x^2"},"depends_on":[]},'
            '{"id":"explain","goal":"解释结果","tool":"respond",'
            '"arguments":{},"depends_on":["derive"]}'
            "]}"
        )
        steps, warning = task_orchestrator.parse_plan(
            response,
            "求导并解释",
            {"respond", "math.differentiate"},
        )
        self.assertIsNone(warning)
        self.assertEqual([step["id"] for step in steps], ["derive", "explain"])

    def test_parses_simple_text_plan_without_nested_json(self):
        response = (
            "[book_search] 检索概念资料 | query=人工智能的定义\n"
            "[math.differentiate] 求导 | expression=x**2; variable=x; order=2"
        )
        steps, warning = task_orchestrator.parse_text_plan(
            response,
            "检索资料并求导",
            {"book_search", "math.differentiate", "respond"},
        )
        self.assertIsNone(warning)
        self.assertEqual([step["tool"] for step in steps], [
            "book_search",
            "math.differentiate",
        ])
        self.assertEqual(steps[1]["arguments"]["order"], 2)
        self.assertEqual(steps[1]["depends_on"], ["task1"])

    def test_text_plan_rejects_unavailable_tools(self):
        steps, warning = task_orchestrator.parse_text_plan(
            "[ppt] 设计结构 | title=AI概述",
            "输出文字大纲",
            {"respond"},
        )
        self.assertIsNotNone(warning)
        self.assertEqual(steps[0]["tool"], "respond")

    def test_invalid_or_unavailable_tool_falls_back_to_safe_response(self):
        response = (
            '{"tasks":[{"id":"read","goal":"读取文件","tool":"read_file",'
            '"arguments":{"path":"C:/private.txt"},"depends_on":[]}]}'
        )
        steps, warning = task_orchestrator.parse_plan(
            response,
            "回答用户",
            {"respond"},
        )
        self.assertIsNotNone(warning)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["tool"], "respond")

    def test_failed_dependency_skips_dependent_step(self):
        steps = [
            {
                "id": "first",
                "goal": "步骤一",
                "tool": "respond",
                "arguments": {},
                "depends_on": [],
            },
            {
                "id": "second",
                "goal": "步骤二",
                "tool": "respond",
                "arguments": {},
                "depends_on": ["first"],
            },
        ]

        def execute(step, _dependencies):
            if step["id"] == "first":
                raise RuntimeError("expected test failure")
            return "must not run"

        results = task_orchestrator.execute_plan(steps, execute)
        self.assertEqual(results[0]["status"], "error")
        self.assertEqual(results[1]["status"], "skipped")

    def test_independent_branches_execute_sequentially(self):
        events = []
        steps = [
            {
                "id": "first",
                "goal": "第一项",
                "tool": "respond",
                "arguments": {},
                "depends_on": [],
            },
            {
                "id": "second",
                "goal": "第二项",
                "tool": "respond",
                "arguments": {},
                "depends_on": [],
            },
        ]

        def execute(step, _dependencies):
            events.append(f"start:{step['id']}")
            events.append(f"finish:{step['id']}")
            return step["id"]

        task_orchestrator.execute_plan(steps, execute)
        self.assertEqual(
            events,
            ["start:first", "finish:first", "start:second", "finish:second"],
        )

    def test_document_exports_are_appended_after_content_steps(self):
        steps = [
            {
                "id": "ppt_outline",
                "goal": "设计PPT的结构",
                "tool": "ppt",
                "arguments": {"content": "AI概述"},
                "depends_on": [],
            },
            {
                "id": "ppt_content",
                "goal": "添加具体内容到PPT",
                "tool": "ppt",
                "arguments": {"content": "AI概述"},
                "depends_on": ["ppt_outline"],
            },
            {
                "id": "word_outline",
                "goal": "设计Word文档的结构",
                "tool": "ppt",
                "arguments": {"content": "AI概述"},
                "depends_on": [],
            },
            {
                "id": "word_content",
                "goal": "添加具体内容到Word文档",
                "tool": "ppt",
                "arguments": {"content": "AI概述"},
                "depends_on": ["word_outline"],
            },
            {
                "id": "word_export",
                "goal": "生成Word文档",
                "tool": "word",
                "arguments": {"content": "AI概述"},
                "depends_on": ["word_outline"],
            },
        ]
        results = [
            {"id": "ppt_outline", "status": "error", "output": "failed"},
            {"id": "ppt_content", "status": "skipped", "output": "skipped"},
            {"id": "word_outline", "status": "done", "output": "wrong PPT created"},
            {"id": "word_content", "status": "error", "output": "failed"},
            {"id": "word_export", "status": "done", "output": "Word created"},
        ]

        finalized, reset_ids, changed = task_orchestrator.finalize_output_steps(
            steps,
            {"ppt", "word"},
            results,
        )

        self.assertTrue(changed)
        self.assertEqual(
            [step["tool"] for step in finalized],
            ["respond", "respond", "respond", "respond", "respond", "ppt", "word"],
        )
        self.assertEqual(reset_ids, {"word_outline", "word_export"})
        self.assertEqual(finalized[-2]["depends_on"], [
            "ppt_outline",
            "ppt_content",
            "word_outline",
            "word_content",
            "word_export",
        ])
        self.assertEqual(finalized[-1]["depends_on"], finalized[-2]["depends_on"])

    def test_resume_reuses_completed_steps_and_retries_incomplete_steps(self):
        steps = [
            {
                "id": "first",
                "goal": "已完成步骤",
                "tool": "respond",
                "arguments": {},
                "depends_on": [],
            },
            {
                "id": "second",
                "goal": "未完成步骤",
                "tool": "respond",
                "arguments": {},
                "depends_on": ["first"],
            },
        ]
        calls = []

        def execute(step, dependencies):
            calls.append((step["id"], dependencies))
            return "second result"

        results = task_orchestrator.execute_plan(
            steps,
            execute,
            initial_results={
                "first": {"status": "done", "output": "first result"},
                "second": {"status": "interrupted", "output": "cancelled"},
            },
        )

        self.assertEqual(calls, [("second", {"first": "first result"})])
        self.assertEqual([result["status"] for result in results], ["done", "done"])
        self.assertEqual(results[1]["output"], "second result")

    def test_interrupted_step_is_checkpointed_as_interrupted(self):
        started = []
        saved = []
        cancelled = False

        def execute(_step, _dependencies):
            nonlocal cancelled
            cancelled = True
            raise RuntimeError("cancelled")

        results = task_orchestrator.execute_plan(
            [
                {
                    "id": "step",
                    "goal": "正在执行",
                    "tool": "respond",
                    "arguments": {},
                    "depends_on": [],
                }
            ],
            execute,
            is_cancelled=lambda: cancelled,
            on_step_start=lambda step: started.append(step["id"]),
            on_step_result=lambda _step, result: saved.append(result["status"]),
        )

        self.assertEqual(started, ["step"])
        self.assertEqual(saved, ["interrupted"])
        self.assertEqual(results[0]["status"], "interrupted")

    def test_file_and_document_tools_require_user_scope(self):
        no_path_tools = tool_registry.tools_for_task("解释这个概念", set())
        self.assertNotIn("read_file", no_path_tools)
        self.assertNotIn("word", no_path_tools)
        self.assertNotIn("ppt", no_path_tools)

        explicit_path = r"D:\Books\calculus.txt"
        requested_tools = tool_registry.tools_for_task(
            f"读取文件 {explicit_path}",
            {"word"},
        )
        self.assertIn("read_file", requested_tools)
        self.assertIn("word", requested_tools)
        self.assertNotIn("ppt", requested_tools)
        self.assertTrue(tool_registry.is_explicit_path(explicit_path, f"读取 {explicit_path}"))


if __name__ == "__main__":
    unittest.main()
