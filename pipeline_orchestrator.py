from dataclasses import dataclass, field
import re

import task_orchestrator


@dataclass
class Blackboard:
    task: str
    allowed_outputs: list = field(default_factory=list)
    route: str = "work"
    plan: list = field(default_factory=list)
    outline: str = ""
    content: str = ""
    script_code: str = ""
    errors: list = field(default_factory=list)
    final_output: str = ""
    results: list = field(default_factory=list)


class ModelRouter:
    OFFICE_PATTERN = re.compile(
        r"(?<![A-Za-z])(?:word|docx|pptx?|powerpoint)(?![A-Za-z])"
        r"|文档|演示文稿|幻灯片",
        re.IGNORECASE,
    )
    CODE_PATTERN = re.compile(
        r"python|c语言|编程|代码|debug|compile",
        re.IGNORECASE,
    )
    MATH_PATTERN = re.compile(
        r"求导|积分|极限|解方程|化简|微分|导数|高等数学"
        r"|\b(?:differentiate|integrate|derivative|limit|equation)\b",
        re.IGNORECASE,
    )

    @classmethod
    def route(cls, task, mode="work", allowed_outputs=None):
        if allowed_outputs or cls.OFFICE_PATTERN.search(task):
            return "office"
        if mode == "code" or cls.CODE_PATTERN.search(task):
            return "code"
        if cls.MATH_PATTERN.search(task):
            return "math"
        if mode == "chat":
            return "chat"
        return "work"


class PipelineOrchestrator:
    def __init__(
        self,
        blackboard,
        progress_callback=None,
        is_cancelled=None,
    ):
        self.blackboard = blackboard
        self.progress_callback = progress_callback or (lambda _message: None)
        self.is_cancelled = is_cancelled or (lambda: False)

    def run(
        self,
        execute_node,
        initial_results=None,
        on_step_start=None,
        on_step_result=None,
    ):
        def execute(task, dependencies):
            output = execute_node(self.blackboard, task, dependencies)
            if task["id"] == "office_outline":
                self.blackboard.outline = str(output)
            elif task["id"] == "office_content":
                self.blackboard.content = str(output)
            return output

        def record_result(task, result):
            if result["status"] != "done":
                self.blackboard.errors.append(
                    f"{task['id']} ({result['status']}): {result['output']}"
                )
            if on_step_result is not None:
                on_step_result(task, result)

        self.blackboard.results = task_orchestrator.execute_plan(
            self.blackboard.plan,
            execute,
            is_cancelled=self.is_cancelled,
            initial_results=initial_results,
            on_step_start=on_step_start,
            on_step_result=record_result,
        )
        successful_outputs = [
            result["output"]
            for result in self.blackboard.results
            if result["status"] == "done"
            and result["task"]["tool"] in ("word", "ppt")
        ]
        self.blackboard.final_output = "\n".join(successful_outputs)
        return self.blackboard.results


class OfficePipeline(PipelineOrchestrator):
    OUTLINE_STEP_ID = "office_outline"
    CONTENT_STEP_ID = "office_content"

    def __init__(self, blackboard, progress_callback=None, is_cancelled=None):
        super().__init__(blackboard, progress_callback, is_cancelled)
        self.blackboard.route = "office"
        self.blackboard.plan = self.build_plan(blackboard.allowed_outputs)

    @classmethod
    def build_plan(cls, allowed_outputs):
        outputs = [
            output_type
            for output_type in ("ppt", "word")
            if output_type in set(allowed_outputs)
        ]
        if not outputs:
            raise ValueError("Office 流水线需要用户明确要求 Word 或 PPT。")

        tasks = [
            {
                "id": cls.OUTLINE_STEP_ID,
                "goal": "策划 Office 文档或演示文稿结构",
                "tool": "respond",
                "arguments": {},
                "depends_on": [],
            },
            {
                "id": cls.CONTENT_STEP_ID,
                "goal": "按结构撰写完整 Markdown 内容",
                "tool": "respond",
                "arguments": {},
                "depends_on": [cls.OUTLINE_STEP_ID],
            }
        ]
        for output_type in outputs:
            tasks.append(
                {
                    "id": f"office_{output_type}",
                    "goal": f"将已完成内容导出为 {output_type.upper()} 文件",
                    "tool": output_type,
                    "arguments": {},
                    "depends_on": [cls.CONTENT_STEP_ID],
                }
            )
        return tasks

    def run(
        self,
        execute_node,
        initial_results=None,
        on_step_start=None,
        on_step_result=None,
    ):
        existing_content = (initial_results or {}).get(self.CONTENT_STEP_ID, {})
        existing_outline = (initial_results or {}).get(self.OUTLINE_STEP_ID, {})
        if (
            existing_outline.get("status") == "done"
            and isinstance(existing_outline.get("output"), str)
        ):
            self.blackboard.outline = existing_outline["output"]
        if (
            existing_content.get("status") == "done"
            and isinstance(existing_content.get("output"), str)
        ):
            self.blackboard.content = existing_content["output"]
        self.progress_callback("📋 Office 流水线已按纯文本内容与最终文件导出分解。")
        results = super().run(
            execute_node,
            initial_results,
            on_step_start,
            on_step_result,
        )
        return results


def markdown_to_ppt_outlines(markdown, fallback_title="MoooWind 报告"):
    lines = markdown.splitlines()
    title = fallback_title
    sections = []
    current_title = None
    current_body = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# ") and current_title is None:
            title = stripped[2:].strip() or fallback_title
        elif stripped.startswith("## "):
            if current_title is not None:
                sections.append((current_title, current_body))
            current_title = stripped[3:].strip()
            current_body = []
        elif stripped.startswith("### "):
            current_body.append(stripped[4:].strip())
        elif stripped:
            current_body.append(stripped.lstrip("-* ").strip())

    if current_title is not None:
        sections.append((current_title, current_body))

    if not sections:
        body = [
            line.strip().lstrip("-* ").strip()
            for line in lines
            if line.strip() and not line.strip().startswith("# ")
        ]
        if body:
            sections = [("内容", body)]
        else:
            raise ValueError("Markdown 没有可用于生成 PPT 的内容。")

    outlines = []
    for section_title, body in sections:
        points = [point for point in body if point]
        outlines.append(f"{section_title} | {'；'.join(points)}")
    return title, outlines
