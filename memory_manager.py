import json
import os
from pathlib import Path
import tempfile

import torch


BASE_DIR = Path(__file__).resolve().parent
MEMORY_DIR = BASE_DIR / "data"
MEMORY_FILES = {
    "life": MEMORY_DIR / "memory_life.json",
    "work": MEMORY_DIR / "memory_work.json",
    "code": MEMORY_DIR / "memory_code.json",
}
LEGACY_MEMORY_FILES = {
    "life": (Path(r"D:\Moowind AI\memory_life.json"), BASE_DIR / "memory_life.json"),
    "work": (Path(r"D:\Moowind AI\memory_work.json"), BASE_DIR / "memory_work.json"),
    "code": (BASE_DIR / "memory_code.json",),
}
MEMORY_TOKEN_LIMIT = 6000
SUMMARY_INPUT_TOKEN_LIMIT = 3000


def _memory_path(mode):
    if mode not in MEMORY_FILES:
        raise ValueError(f"不支持的记忆模式：{mode}")
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    return MEMORY_FILES[mode]


def load(mode="life"):
    path = _memory_path(mode)
    if not path.exists():
        legacy_path = next(
            (candidate for candidate in LEGACY_MEMORY_FILES[mode] if candidate.exists()),
            None,
        )
        if legacy_path is None:
            return []
        path = legacy_path

    try:
        with path.open("r", encoding="utf-8") as file:
            messages = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"读取记忆文件失败：{path}：{error}") from error

    if not isinstance(messages, list) or any(
        not isinstance(message, dict)
        or not isinstance(message.get("role"), str)
        or not isinstance(message.get("content"), str)
        for message in messages
    ):
        raise ValueError(f"记忆文件格式错误：{path}")
    return messages


def save(messages, mode="life"):
    path = _memory_path(mode)
    if not isinstance(messages, list) or any(
        not isinstance(message, dict)
        or not isinstance(message.get("role"), str)
        or not isinstance(message.get("content"), str)
        for message in messages
    ):
        raise ValueError("记忆内容必须是包含 role 和 content 字符串的消息列表")

    file_descriptor, temp_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=f"{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            json.dump(messages, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, path)
    except Exception as error:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError as cleanup_error:
                raise RuntimeError(
                    f"保存记忆文件失败：{path}：{error}；清理临时文件也失败：{cleanup_error}"
                ) from error
        if isinstance(error, OSError):
            raise RuntimeError(f"保存记忆文件失败：{path}：{error}") from error
        raise


def count_tokens(messages, tokenizer):
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    return token_ids.shape[-1] if hasattr(token_ids, "shape") else len(token_ids)


def fit_context(messages, tokenizer, max_tokens):
    if max_tokens <= 0:
        raise ValueError("上下文 token 预算必须大于 0")

    fitted_messages = [dict(message) for message in messages]
    while count_tokens(fitted_messages, tokenizer) > max_tokens:
        current_user_index = next(
            (
                index
                for index in range(len(fitted_messages) - 1, -1, -1)
                if fitted_messages[index].get("role") == "user"
            ),
            None,
        )
        older_user_indexes = [
            index
            for index, message in enumerate(fitted_messages)
            if message.get("role") == "user"
            and index != current_user_index
        ]

        if older_user_indexes:
            first_old_user = older_user_indexes[0]
            next_user = next(
                (
                    index
                    for index in range(first_old_user + 1, len(fitted_messages))
                    if fitted_messages[index].get("role") == "user"
                ),
                len(fitted_messages),
            )
            del fitted_messages[first_old_user:next_user]
            continue

        removable_index = next(
            (
                index
                for index, message in enumerate(fitted_messages)
                if message.get("role") != "system"
                and index != current_user_index
            ),
            None,
        )
        if removable_index is None:
            raise ValueError(
                "当前问题或系统提示过长，无法放入模型上下文，请缩短后重试。"
            )
        del fitted_messages[removable_index]

    return fitted_messages


def compress_if_needed(messages, model=None, tokenizer=None, max_tokens=MEMORY_TOKEN_LIMIT):
    if tokenizer is None:
        raise ValueError("压缩记忆需要 tokenizer")
    if count_tokens(messages, tokenizer) <= max_tokens:
        return messages

    system_messages = [message for message in messages if message.get("role") == "system"]
    conversation = [message for message in messages if message.get("role") != "system"]
    if len(conversation) <= 2:
        raise RuntimeError("最近一轮对话已超过记忆 token 预算，原对话已保留")

    print("记忆管理：正在按 token 预算压缩较早的对话...")
    recent_messages = conversation[-2:]
    old_messages = conversation[:-2]
    if not old_messages:
        raise RuntimeError("没有可压缩的较早对话，原对话已保留")
    history_text = "\n".join(
        f"{message['role']}: {message['content']}" for message in old_messages
    )
    history_ids = tokenizer.encode(
        history_text,
        truncation=True,
        max_length=SUMMARY_INPUT_TOKEN_LIMIT,
    )
    history_text = tokenizer.decode(history_ids, skip_special_tokens=True)
    prompt = (
        "请用简洁要点总结这段对话，只保留长期有用的事实、偏好、约定和未完成事项；"
        "不要编造信息：\n" + history_text
    )
    inputs = tokenizer([prompt], return_tensors="pt")

    try:
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
            )
        summary = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()
    except Exception as error:
        raise RuntimeError(f"生成长期记忆摘要失败：{error}") from error

    if not summary:
        raise RuntimeError("模型返回了空的长期记忆摘要，未压缩记忆")

    base_system = system_messages[0] if system_messages else {
        "role": "system",
        "content": "",
    }
    new_system = dict(base_system)
    new_system["content"] = (
        base_system["content"].rstrip()
        + "\n\n【长期记忆摘要】\n"
        + summary
    )
    compressed_messages = [new_system] + system_messages[1:] + recent_messages
    if count_tokens(compressed_messages, tokenizer) > max_tokens:
        raise RuntimeError("摘要和最近一轮对话仍超过记忆 token 预算，原对话已保留")
    return compressed_messages
