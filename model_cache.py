import re
from pathlib import Path


MODEL_CACHE_DIR = Path(__file__).resolve().parent / "models"
MODEL_ID_PATTERN = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


def local_model_path(model_id, required_file="config.json"):
    if not isinstance(model_id, str) or not MODEL_ID_PATTERN.fullmatch(model_id):
        raise ValueError(f"无效的模型 ID：{model_id!r}")

    model_path = (
        MODEL_CACHE_DIR
        / "models"
        / model_id.replace("/", "--")
        / "snapshots"
        / "master"
    )
    if not (model_path / required_file).is_file():
        raise FileNotFoundError(
            f"本地模型文件不存在：{model_path / required_file}。"
            "请先运行 download_all_models.py 完成下载。"
        )
    return model_path
