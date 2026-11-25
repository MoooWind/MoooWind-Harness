import unittest
from unittest.mock import patch

import Mooowind_App
import task_checkpoint


class FakeTokenizer:
    def encode(self, text, **_kwargs):
        return list(str(text))

    def decode(self, tokens, **_kwargs):
        return "".join(tokens)


class WorkModeFlowTests(unittest.TestCase):
    def test_model_reload_releases_before_loading_target(self):
        events = []
        target_model = object()
        target_tokenizer = object()
        worker = Mooowind_App.AIWorker(
            [{"role": "user", "content": "draft"}],
            True,
            Mooowind_App.PROMPT_WORK,
            loaded_model_name="DeepSeek-7B",
        )
        worker._generate_with_continuation = lambda *_args, **_kwargs: ("draft", True)

        def release_model():
            events.append("release")

        def load_model(model_name):
            events.append(("load", model_name))
            return target_model, target_tokenizer

        with (
            patch("Mooowind_App.model", object()),
            patch("Mooowind_App.tokenizer", object()),
            patch("Mooowind_App._release_text_components", side_effect=release_model),
            patch("Mooowind_App._load_text_components", side_effect=load_model),
        ):
            result = worker.call_model_with_reload(
                "Qwen-7B",
                [{"role": "user", "content": "draft"}],
                max_new_tokens=20,
            )
            self.assertIs(Mooowind_App.model, target_model)
            self.assertIs(Mooowind_App.tokenizer, target_tokenizer)

        self.assertEqual(result, ("draft", True))
        self.assertEqual(events, ["release", ("load", "Qwen-7B")])
        self.assertEqual(worker.loaded_model_name, "Qwen-7B")

    def test_planned_math_tool_is_run_in_a_fresh_step_context(self):
        original_tokenizer = Mooowind_App.tokenizer
        Mooowind_App.tokenizer = FakeTokenizer()
        try:
            checkpoint = task_checkpoint.create_checkpoint(
                "work",
                "求 x 平方的导数",
            )
            worker = Mooowind_App.AIWorker(
                [
                    {"role": "system", "content": Mooowind_App.PROMPT_WORK},
                    {"role": "user", "content": "求 x 平方的导数"},
                ],
                True,
                Mooowind_App.PROMPT_WORK,
                task_checkpoint=checkpoint,
            )
            planned_tasks = (
                "[math.differentiate] 求导 | "
                "expression=x^2; variable=x; order=1"
            )
            responses = iter(
                [
                    (planned_tasks, True),
                    ("工具结果对应的导数是 2x。", True),
                    ("最终结果：导数为 2x。", True),
                ]
            )
            calls = []

            def fake_generate(messages, *_args, **_kwargs):
                calls.append(messages)
                return next(responses)

            worker._generate_with_continuation = fake_generate
            with patch("Mooowind_App.task_checkpoint_store.save_checkpoint") as save:
                ui_response, updated_messages = worker._run_work_loop()

            self.assertEqual(len(calls), 3)
            self.assertIn('"result": "2*x"', calls[1][-1]["content"])
            self.assertEqual(len(calls[1]), 2)
            self.assertEqual(len(calls[2]), 2)
            self.assertIn("最终结果", ui_response)
            self.assertIn("math.differentiate", updated_messages[-1]["content"])
            self.assertEqual(checkpoint["results"][0]["status"], "done")
            self.assertIsNone(checkpoint["active_step_id"])
            self.assertGreaterEqual(save.call_count, 3)
        finally:
            Mooowind_App.tokenizer = original_tokenizer

    def test_legacy_plan_moves_document_exports_after_content_steps(self):
        original_tokenizer = Mooowind_App.tokenizer
        Mooowind_App.tokenizer = FakeTokenizer()
        try:
            checkpoint = task_checkpoint.create_checkpoint(
                "work",
                "做一个关于AI概述的PPT和Word",
            )
            checkpoint["task"] = checkpoint["user_task"]
            checkpoint["allowed_outputs"] = ["ppt", "word"]
            checkpoint["steps"] = [
                {
                    "id": "t1",
                    "goal": "设计PPT的结构",
                    "tool": "ppt",
                    "arguments": {"content": "AI概述"},
                    "depends_on": [],
                },
                {
                    "id": "t2",
                    "goal": "添加具体内容到PPT",
                    "tool": "ppt",
                    "arguments": {"content": "AI概述"},
                    "depends_on": ["t1"],
                },
                {
                    "id": "t3",
                    "goal": "设计Word文档的结构",
                    "tool": "ppt",
                    "arguments": {"content": "AI概述"},
                    "depends_on": [],
                },
                {
                    "id": "t4",
                    "goal": "添加具体内容到Word文档",
                    "tool": "ppt",
                    "arguments": {"content": "AI概述"},
                    "depends_on": ["t3"],
                },
                {
                    "id": "t5",
                    "goal": "生成Word文档",
                    "tool": "word",
                    "arguments": {"content": "AI概述"},
                    "depends_on": ["t3"],
                },
            ]
            checkpoint["results"] = [
                {"id": "t1", "status": "error", "output": "failed"},
                {"id": "t2", "status": "skipped", "output": "skipped"},
                {"id": "t3", "status": "done", "output": "wrong PPT created"},
                {"id": "t4", "status": "error", "output": "failed"},
                {"id": "t5", "status": "done", "output": "Word created"},
            ]
            checkpoint["active_step_id"] = "t4"
            worker = Mooowind_App.AIWorker(
                [
                    {"role": "system", "content": Mooowind_App.PROMPT_WORK},
                    {"role": "user", "content": checkpoint["user_task"]},
                ],
                True,
                Mooowind_App.PROMPT_WORK,
                task_checkpoint=checkpoint,
            )
            generated_markdown = "# AI 概述\n## 定义\n- 人工智能的核心概念\n"
            model_calls = []

            def fake_model_call(model_name, messages, **_kwargs):
                model_calls.append((model_name, messages))
                if len(model_calls) == 1:
                    return "# AI 概述\n## 定义\n- 核心概念\n", True
                return generated_markdown, True

            worker.call_model_with_reload = fake_model_call
            worker._ensure_model_loaded = lambda _model_name: None

            def fake_export(tool_name, arguments):
                if tool_name == "ppt":
                    self.assertEqual(arguments["title"], "AI 概述")
                    self.assertEqual(
                        arguments["outlines"],
                        ["定义 | 人工智能的核心概念"],
                    )
                else:
                    self.assertEqual(arguments["content"], generated_markdown.strip())
                return f"{tool_name}-file"

            with (
                patch("Mooowind_App.task_checkpoint_store.save_checkpoint"),
                patch("Mooowind_App.tool_registry.execute_tool", side_effect=fake_export),
            ):
                response, _messages = worker._run_work_loop()

            self.assertEqual(
                [(step["id"], step["tool"]) for step in checkpoint["steps"]],
                [
                    ("office_outline", "respond"),
                    ("office_content", "respond"),
                    ("office_ppt", "ppt"),
                    ("office_word", "word"),
                ],
            )
            self.assertIn("旧任务计划", checkpoint["plan_warning"])
            self.assertEqual([call[0] for call in model_calls], ["Qwen-7B", "Qwen-7B"])
            self.assertIn("## 定义", model_calls[1][1][-1]["content"])
            self.assertTrue(
                all(result["status"] == "done" for result in checkpoint["results"]),
                checkpoint["results"],
            )
            self.assertIn("PPT 已生成", response)
            self.assertIn("WORD 已生成", response)
        finally:
            Mooowind_App.tokenizer = original_tokenizer


if __name__ == "__main__":
    unittest.main()
