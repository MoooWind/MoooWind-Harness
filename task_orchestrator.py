import json
import re


MAX_TASKS = 5
MAX_PLAN_CHARS = 12000
DOCUMENT_OUTPUT_TOOLS = {"word", "ppt"}


def _is_final_output_step(step, output_type):
    goal = step.get("goal", "").casefold()
    planning_terms = (
        "结构",
        "大纲",
        "设计",
        "添加",
        "具体内容",
        "内容",
        "撰写",
        "编写",
        "完善",
        "修改",
        "outline",
        "structure",
        "design",
        "content",
        "draft",
    )
    if any(term in goal for term in planning_terms):
        return False

    action_terms = ("生成", "创建", "保存", "导出", "制作", "generate", "create", "save", "export")
    if not any(term in goal for term in action_terms):
        return False

    type_terms = (
        ("ppt", "powerpoint", "演示文稿")
        if output_type == "ppt"
        else ("word", "docx", "文档")
    )
    return any(term in goal for term in type_terms)


def finalize_output_steps(tasks, allowed_outputs, initial_results=None):
    result_by_id = {
        result["id"]: result
        for result in (initial_results or [])
        if isinstance(result, dict) and isinstance(result.get("id"), str)
    }
    has_misassigned_output = any(
        task.get("tool") in DOCUMENT_OUTPUT_TOOLS
        and not _is_final_output_step(task, task.get("tool"))
        for task in tasks
    )

    finalized = []
    completed_output_types = set()
    reset_ids = set()
    changed = False
    for task in tasks:
        normalized = dict(task)
        tool = normalized.get("tool")
        if tool not in DOCUMENT_OUTPUT_TOOLS:
            finalized.append(normalized)
            continue

        result = result_by_id.get(normalized["id"], {})
        is_completed_artifact = (
            tool in allowed_outputs
            and _is_final_output_step(normalized, tool)
            and result.get("status") == "done"
            and tool not in completed_output_types
            and not has_misassigned_output
        )
        if is_completed_artifact:
            completed_output_types.add(tool)
            finalized.append(normalized)
            continue

        normalized["tool"] = "respond"
        normalized["arguments"] = {}
        finalized.append(normalized)
        changed = True
        if result.get("status") == "done":
            reset_ids.add(normalized["id"])

    existing_ids = {task["id"] for task in finalized}
    for output_type in sorted(set(allowed_outputs) & DOCUMENT_OUTPUT_TOOLS):
        if output_type in completed_output_types:
            continue
        task_id = f"output_{output_type}"
        suffix = 2
        while task_id in existing_ids:
            task_id = f"output_{output_type}_{suffix}"
            suffix += 1
        dependencies = [
            task["id"]
            for task in finalized
            if task.get("tool") not in DOCUMENT_OUTPUT_TOOLS
        ]
        finalized.append(
            {
                "id": task_id,
                "goal": f"生成并保存最终 {output_type.upper()} 文件",
                "tool": output_type,
                "arguments": {},
                "depends_on": dependencies,
            }
        )
        existing_ids.add(task_id)
        changed = True

    return finalized, reset_ids, changed


def parse_plan(response, original_task, allowed_tools, max_tasks=MAX_TASKS):
    fallback = [
        {
            "id": "task1",
            "goal": original_task,
            "tool": "respond",
            "arguments": {},
            "depends_on": [],
        }
    ]
    if not isinstance(response, str) or len(response) > MAX_PLAN_CHARS:
        return fallback, "规划输出为空或超过限制，已退回单步骤回答。"

    text = response.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    start = text.find("{")
    if start < 0:
        return fallback, "规划结果不是有效 JSON，已退回单步骤回答。"

    try:
        plan, _end = json.JSONDecoder().raw_decode(text[start:])
    except (json.JSONDecodeError, TypeError):
        return fallback, "规划 JSON 格式无效，已退回单步骤回答。"
    if not isinstance(plan, dict) or not isinstance(plan.get("tasks"), list):
        return fallback, "规划 JSON 缺少 tasks 列表，已退回单步骤回答。"

    raw_tasks = plan["tasks"]
    if not raw_tasks or len(raw_tasks) > max_tasks:
        return fallback, f"规划步骤必须为 1 到 {max_tasks} 步，已退回单步骤回答。"

    tasks = []
    known_ids = set()
    for raw_task in raw_tasks:
        if not isinstance(raw_task, dict):
            return fallback, "规划步骤格式无效，已退回单步骤回答。"
        task_id = raw_task.get("id")
        goal = raw_task.get("goal")
        tool = raw_task.get("tool")
        arguments = raw_task.get("arguments", {})
        dependencies = raw_task.get("depends_on", [])
        if (
            not isinstance(task_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", task_id)
            or task_id in known_ids
            or not isinstance(goal, str)
            or not goal.strip()
            or not isinstance(tool, str)
            or tool not in allowed_tools
            or not isinstance(arguments, dict)
            or not isinstance(dependencies, list)
            or any(
                not isinstance(dependency, str) or dependency not in known_ids
                for dependency in dependencies
            )
        ):
            return fallback, "规划包含无效字段、未授权工具或错误依赖，已退回单步骤回答。"
        if len(json.dumps(raw_task, ensure_ascii=False)) > 4000:
            return fallback, "单个规划步骤输入过长，已退回单步骤回答。"

        task = {
            "id": task_id,
            "goal": goal.strip(),
            "tool": tool,
            "arguments": arguments,
            "depends_on": dependencies,
        }
        tasks.append(task)
        known_ids.add(task_id)

    return tasks, None


def parse_text_plan(response, original_task, allowed_tools, max_tasks=MAX_TASKS):
    fallback = [
        {
            "id": "task1",
            "goal": original_task,
            "tool": "respond",
            "arguments": {},
            "depends_on": [],
        }
    ]
    if not isinstance(response, str) or len(response) > MAX_PLAN_CHARS:
        return fallback, "规划文本为空或超过限制，已退回单步骤回答。"

    text = re.sub(r"^```(?:markdown|text)?\s*|\s*```$", "", response.strip(), flags=re.IGNORECASE)
    tasks = []
    previous_id = None
    valid_lines = re.compile(
        r"^\s*(?:\d+[.)]\s*)?\[([A-Za-z0-9_.-]+)\]\s*(.+?)\s*$"
    )
    for line in text.splitlines():
        if not line.strip():
            continue
        match = valid_lines.fullmatch(line)
        if match is None:
            continue
        tool, task_text = match.groups()
        if tool not in allowed_tools:
            return fallback, f"规划使用了未授权工具 {tool}，已退回单步骤回答。"
        parts = task_text.split("|", 1)
        goal = parts[0].strip()
        if not goal:
            return fallback, "规划步骤缺少目标，已退回单步骤回答。"

        arguments = {}
        if len(parts) == 2:
            for item in parts[1].split(";"):
                if not item.strip():
                    continue
                if "=" not in item:
                    return fallback, "规划参数必须使用 key=value 格式，已退回单步骤回答。"
                name, value = item.split("=", 1)
                name, value = name.strip(), value.strip()
                if not name or not value:
                    return fallback, "规划参数名和值不能为空，已退回单步骤回答。"
                if name in arguments:
                    return fallback, f"规划重复定义参数 {name}，已退回单步骤回答。"
                arguments[name] = value

        if tool == "book_search":
            arguments.setdefault("query", goal)
        elif tool == "read_file":
            if "path" not in arguments:
                path_match = re.search(
                    r"(?:[A-Za-z]:[\\/][^\s;|]+|\\\\[^\s;|]+)",
                    original_task,
                )
                if path_match is None:
                    return fallback, "读取文件步骤未提供原任务中的明确路径，已退回单步骤回答。"
                arguments["path"] = path_match.group(0).rstrip(".,，。")
        elif tool.startswith("math."):
            if "expression" not in arguments:
                return fallback, "数学工具步骤缺少 expression 参数，已退回单步骤回答。"
            arguments.setdefault("variable", "x")
            for integer_parameter in ("order",):
                if integer_parameter in arguments:
                    try:
                        arguments[integer_parameter] = int(arguments[integer_parameter])
                    except ValueError:
                        return fallback, f"数学参数 {integer_parameter} 必须是整数，已退回单步骤回答。"

        task_id = f"task{len(tasks) + 1}"
        tasks.append(
            {
                "id": task_id,
                "goal": goal,
                "tool": tool,
                "arguments": arguments,
                "depends_on": [previous_id] if previous_id else [],
            }
        )
        previous_id = task_id
        if len(tasks) > max_tasks:
            return fallback, f"规划步骤不能超过 {max_tasks} 步，已退回单步骤回答。"

    if not tasks:
        return fallback, "规划文本未包含有效步骤，已退回单步骤回答。"
    return tasks, None


def execute_plan(
    tasks,
    execute_task,
    is_cancelled=lambda: False,
    initial_results=None,
    on_step_start=None,
    on_step_result=None,
):
    results = {}
    for task_id, result in (initial_results or {}).items():
        if (
            isinstance(result, dict)
            and result.get("status") == "done"
            and isinstance(result.get("output"), str)
        ):
            results[task_id] = {
                "status": "done",
                "output": result["output"],
            }
    ordered_results = []
    for task in tasks:
        if task["id"] in results:
            ordered_results.append(
                {"task": task, **results[task["id"]]}
            )
            continue
        if is_cancelled():
            break
        failed_dependencies = [
            dependency
            for dependency in task["depends_on"]
            if results.get(dependency, {}).get("status") != "done"
        ]
        if failed_dependencies:
            result = {
                "status": "skipped",
                "output": (
                    "未执行：依赖步骤未成功完成："
                    + ", ".join(failed_dependencies)
                ),
            }
        else:
            if on_step_start is not None:
                on_step_start(task)
            dependency_outputs = {
                dependency: results[dependency]["output"]
                for dependency in task["depends_on"]
            }
            try:
                output = execute_task(task, dependency_outputs)
                result = {"status": "done", "output": str(output)}
            except Exception as error:
                status = "interrupted" if is_cancelled() else "error"
                result = {"status": status, "output": str(error)}
        results[task["id"]] = result
        ordered_results.append({"task": task, **result})
        if on_step_result is not None:
            on_step_result(task, result)
    return ordered_results
