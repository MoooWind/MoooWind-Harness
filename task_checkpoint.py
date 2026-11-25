from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile


CHECKPOINT_PATH = Path(__file__).resolve().parent / "data" / "task_checkpoint.json"
SUPPORTED_MODES = {"work", "code"}


def create_checkpoint(mode, user_task):
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"不支持任务恢复的模式：{mode}")
    if not isinstance(user_task, str) or not user_task.strip():
        raise ValueError("待恢复任务不能为空")
    now = datetime.now(timezone.utc).isoformat()
    return {
        "version": 1,
        "mode": mode,
        "user_task": user_task,
        "task": None,
        "allowed_outputs": [],
        "steps": [],
        "results": [],
        "active_step_id": None,
        "state": "planning",
        "updated_at": now,
    }


def _validate_checkpoint(checkpoint):
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("version") != 1
        or not isinstance(checkpoint.get("mode"), str)
        or checkpoint["mode"] not in SUPPORTED_MODES
        or not isinstance(checkpoint.get("user_task"), str)
        or not checkpoint["user_task"].strip()
        or not isinstance(checkpoint.get("allowed_outputs"), list)
        or any(
            not isinstance(output_type, str)
            or output_type not in {"word", "ppt"}
            for output_type in checkpoint["allowed_outputs"]
        )
        or not isinstance(checkpoint.get("steps"), list)
        or any(not isinstance(step, dict) for step in checkpoint["steps"])
        or not isinstance(checkpoint.get("results"), list)
        or any(
            not isinstance(result, dict)
            or not isinstance(result.get("id"), str)
            or not isinstance(result.get("status"), str)
            or result.get("status")
            not in {"done", "error", "interrupted", "skipped"}
            or not isinstance(result.get("output"), str)
            for result in checkpoint["results"]
        )
        or not isinstance(checkpoint.get("state"), str)
        or checkpoint["state"] not in {"planning", "running"}
        or (
            checkpoint.get("task") is not None
            and not isinstance(checkpoint["task"], str)
        )
    ):
        raise ValueError(f"任务检查点格式错误：{CHECKPOINT_PATH}")
    return checkpoint


def save_checkpoint(checkpoint):
    _validate_checkpoint(checkpoint)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint["updated_at"] = datetime.now(timezone.utc).isoformat()
    file_descriptor, temp_path = tempfile.mkstemp(
        dir=str(CHECKPOINT_PATH.parent),
        prefix=f"{CHECKPOINT_PATH.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            json.dump(checkpoint, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, CHECKPOINT_PATH)
    except Exception as error:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError as cleanup_error:
                raise RuntimeError(
                    f"保存任务检查点失败：{error}；清理临时文件也失败：{cleanup_error}"
                ) from error
        if isinstance(error, OSError):
            raise RuntimeError(f"保存任务检查点失败：{error}") from error
        raise


def load_checkpoint():
    if not CHECKPOINT_PATH.exists():
        return None
    try:
        with CHECKPOINT_PATH.open("r", encoding="utf-8") as file:
            checkpoint = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"读取任务检查点失败：{error}") from error
    return _validate_checkpoint(checkpoint)


def clear_checkpoint():
    try:
        CHECKPOINT_PATH.unlink(missing_ok=True)
    except OSError as error:
        raise RuntimeError(f"清除任务检查点失败：{error}") from error
