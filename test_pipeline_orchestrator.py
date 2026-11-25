import unittest

import pipeline_orchestrator


class PipelineOrchestratorTests(unittest.TestCase):
    def test_router_prefers_office_outputs_and_respects_mode(self):
        self.assertEqual(
            pipeline_orchestrator.ModelRouter.route(
                "做一份 AI 概述 PPT",
                mode="work",
                allowed_outputs={"ppt"},
            ),
            "office",
        )
        self.assertEqual(
            pipeline_orchestrator.ModelRouter.route(
                "写一个 C 语言程序",
                mode="code",
            ),
            "code",
        )
        self.assertEqual(
            pipeline_orchestrator.ModelRouter.route("求 x 的导数", mode="work"),
            "math",
        )
        self.assertEqual(
            pipeline_orchestrator.ModelRouter.route("今天怎么样", mode="chat"),
            "chat",
        )

    def test_office_pipeline_runs_content_then_serial_exports(self):
        blackboard = pipeline_orchestrator.Blackboard(
            "生成 AI 概述 PPT 和 Word",
            allowed_outputs=["ppt", "word"],
        )
        pipeline = pipeline_orchestrator.OfficePipeline(blackboard)
        events = []

        def execute_node(board, step, dependencies):
            events.append(("start", step["id"]))
            if step["id"] == "office_outline":
                output = "# AI 概述\n## 定义\n- 人工智能\n"
            elif step["id"] == "office_content":
                output = dependencies["office_outline"] + "\n- 详细说明"
            else:
                output = f"{step['tool']} uses {dependencies['office_content']}"
            events.append(("finish", step["id"]))
            return output

        results = pipeline.run(execute_node)
        self.assertEqual(
            [step["id"] for step in pipeline.blackboard.plan],
            ["office_outline", "office_content", "office_ppt", "office_word"],
        )
        self.assertEqual(
            events,
            [
                ("start", "office_outline"),
                ("finish", "office_outline"),
                ("start", "office_content"),
                ("finish", "office_content"),
                ("start", "office_ppt"),
                ("finish", "office_ppt"),
                ("start", "office_word"),
                ("finish", "office_word"),
            ],
        )
        self.assertTrue(all(result["status"] == "done" for result in results))
        self.assertIn("## 定义", blackboard.outline)
        self.assertIn("ppt uses", blackboard.final_output)
        self.assertIn("word uses", blackboard.final_output)

    def test_pipeline_records_node_failures_on_blackboard(self):
        blackboard = pipeline_orchestrator.Blackboard("test")
        blackboard.plan = [
            {
                "id": "node",
                "goal": "failing node",
                "tool": "respond",
                "arguments": {},
                "depends_on": [],
            }
        ]
        pipeline = pipeline_orchestrator.PipelineOrchestrator(blackboard)
        results = pipeline.run(
            lambda *_args: (_ for _ in ()).throw(RuntimeError("expected failure"))
        )
        self.assertEqual(results[0]["status"], "error")
        self.assertIn("expected failure", blackboard.errors[0])

    def test_markdown_is_converted_to_slide_text_without_json(self):
        title, slides = pipeline_orchestrator.markdown_to_ppt_outlines(
            "# AI 概述\n"
            "## 发展历程\n"
            "- 早期研究\n"
            "- 深度学习\n"
            "## 应用\n"
            "- 医疗\n",
        )
        self.assertEqual(title, "AI 概述")
        self.assertEqual(
            slides,
            ["发展历程 | 早期研究；深度学习", "应用 | 医疗"],
        )


if __name__ == "__main__":
    unittest.main()
