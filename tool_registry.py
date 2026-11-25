import re
import json

import book_library
import blender_tools
import office_tools
import ppt_tools


TOOL_DESCRIPTIONS = {
    "respond": "用本地语言模型完成推理、解释或写作，不调用外部服务。",
    "book_search": "按问题检索本地导入的 PDF、DOCX、TXT 资料。",
    "read_file": "读取用户明确提供路径的本地文本文件。",
    "math.simplify": "本地化简。表达式使用普通文本算术，乘方写 **，函数写 sin(x)、sqrt(x) 等。",
    "math.differentiate": "本地求导；参数 expression、variable、order。表达式乘方写 **，不用 LaTeX。",
    "math.integrate": "本地积分；参数 expression、variable、lower、upper。表达式乘方写 **，不用 LaTeX。",
    "math.solve_equation": "本地解方程；参数 expression、variable。可写 x**2=1，不用 LaTeX。",
    "math.limit": "本地求极限；参数 expression、variable、point、direction。表达式乘方写 **，不用 LaTeX。",
    "word": "将前序步骤已完成的内容最终导出为 Word 文件；仅在用户明确要求 Word 时可用。只负责文件导出，不负责设计结构或撰写内容。",
    "ppt": "将前序步骤已完成的内容最终导出为 PPT 文件；仅在用户明确要求 PPT 时可用。只负责文件导出，不负责设计结构或撰写内容。",
    "blender.scene": "根据受限的基础几何物体规格，由本机 Blender 生成 .blend 项目和渲染图。",
}
MATH_TOOLS = {
    "math.simplify",
    "math.differentiate",
    "math.integrate",
    "math.solve_equation",
    "math.limit",
}


def available_tools(allowed_outputs, include_file_reader=False):
    names = ["respond", "book_search", *sorted(MATH_TOOLS)]
    if include_file_reader:
        names.append("read_file")
    if "word" in allowed_outputs:
        names.append("word")
    if "ppt" in allowed_outputs:
        names.append("ppt")
    return {name: TOOL_DESCRIPTIONS[name] for name in names}


def is_explicit_path(path, user_task):
    if not isinstance(path, str) or not path.strip():
        return False
    normalized_path = path.strip().strip("\"'").replace("\\", "/").lower()
    normalized_task = user_task.replace("\\", "/").lower()
    return normalized_path in normalized_task


def _require_text(arguments, name, maximum=50000):
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"工具参数 {name} 必须是非空字符串")
    if len(value) > maximum:
        raise ValueError(f"工具参数 {name} 超过 {maximum} 个字符")
    return value.strip()


def execute_tool(name, arguments):
    if not isinstance(arguments, dict):
        raise ValueError("工具参数必须是 JSON 对象")

    if name == "book_search":
        query = _require_text(arguments, "query", maximum=4000)
        try:
            results = book_library.search(query, top_k=4)
        except Exception as error:
            raise RuntimeError(f"本地书籍检索失败：{error}") from error
        if not results:
            return "本地知识库没有检索到匹配片段。"
        return book_library.format_context(results)

    if name == "read_file":
        path = _require_text(arguments, "path", maximum=2000)
        return office_tools.read_local_file(path)

    if name == "word":
        content = _require_text(arguments, "content")
        return office_tools.create_word(content)

    if name == "ppt":
        title = _require_text(arguments, "title", maximum=500)
        outlines = arguments.get("outlines")
        if (
            not isinstance(outlines, list)
            or not outlines
            or len(outlines) > 40
            or any(not isinstance(item, str) or not item.strip() for item in outlines)
        ):
            raise ValueError("PPT 的 outlines 必须是 1 到 40 条非空文字")
        return ppt_tools.create_apple_style_ppt(title, outlines)

    if name == "blender.scene":
        try:
            result = blender_tools.create_scene(arguments.get("spec"))
        except Exception as error:
            raise RuntimeError(f"本地 Blender 场景生成失败：{error}") from error
        return json.dumps(result, ensure_ascii=False)

    if name in MATH_TOOLS:
        try:
            import math_tools
        except ImportError as error:
            raise RuntimeError(
                "本地数学工具依赖未安装，请运行："
                "python -m pip install -r requirements-feature-modules.txt"
            ) from error

        expression = _require_text(arguments, "expression", maximum=500)
        variable = arguments.get("variable", "x")
        if not isinstance(variable, str):
            raise ValueError("variable 必须是字符串")

        if name == "math.simplify":
            result = math_tools.simplify(expression)
        elif name == "math.differentiate":
            order = arguments.get("order", 1)
            result = math_tools.differentiate(expression, variable, order)
        elif name == "math.integrate":
            result = math_tools.integrate(
                expression,
                variable,
                arguments.get("lower"),
                arguments.get("upper"),
            )
        elif name == "math.solve_equation":
            result = math_tools.solve_equation(expression, variable)
        else:
            result = math_tools.calculate_limit(
                expression,
                variable,
                arguments.get("point", "0"),
                arguments.get("direction", "+-"),
            )
        return json.dumps(result, ensure_ascii=False)

    raise ValueError(f"未注册或未授权的工具：{name}")


def tools_for_task(user_task, allowed_outputs):
    needs_file_read = bool(
        re.search(r"读取|读一下|打开|查看文件|分析文件", user_task)
        and re.search(r"(?:[A-Za-z]:[\\/]|\\\\|(?:\.{1,2})[\\/])", user_task)
    )
    tools = available_tools(allowed_outputs, include_file_reader=needs_file_read)
    needs_blender = bool(
        re.search(r"blender|三维|3d|建模|场景建模", user_task, re.IGNORECASE)
    )
    if needs_blender and blender_tools.available():
        tools["blender.scene"] = TOOL_DESCRIPTIONS["blender.scene"]
    return tools
