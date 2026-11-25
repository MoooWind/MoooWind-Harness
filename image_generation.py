from datetime import datetime
from pathlib import Path

import torch

from model_cache import local_model_path

MODEL_ID = "AI-ModelScope/sd-turbo"
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "Output" / "GeneratedImages"
ALLOWED_SIZES = {(512, 512), (512, 768), (768, 512)}


def load_pipeline():
    try:
        from diffusers import DiffusionPipeline
    except ImportError as error:
        raise RuntimeError(
            "本地绘图需要安装 diffusers 和 accelerate；"
            "请运行 python -m pip install -r requirements-feature-modules.txt"
        ) from error

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    model_dir = local_model_path(MODEL_ID, required_file="model_index.json")
    pipeline = DiffusionPipeline.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        variant="fp16" if use_cuda else None,
        local_files_only=True,
    )
    if use_cuda:
        gpu_index = torch.cuda.current_device()
        pipeline.enable_model_cpu_offload(
            gpu_id=gpu_index,
            device=f"cuda:{gpu_index}",
        )
        pipeline.enable_vae_slicing()
        pipeline.enable_attention_slicing()
    else:
        pipeline.to("cpu")
    return pipeline


def generate_image(
    pipeline,
    prompt,
    width=512,
    height=512,
    seed=None,
    progress_callback=lambda _message: None,
):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("绘图提示词不能为空")
    if len(prompt) > 2000:
        raise ValueError("绘图提示词不能超过 2000 个字符")
    if (width, height) not in ALLOWED_SIZES:
        raise ValueError("当前本地绘图只支持 512×512、512×768、768×512")
    if seed is not None and (
        not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
    ):
        raise ValueError("seed 必须是非负整数")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
    progress_callback("正在本地生成图片...")
    result = pipeline(
        prompt=prompt.strip(),
        width=width,
        height=height,
        num_inference_steps=1,
        guidance_scale=0.0,
        generator=generator,
    )
    images = getattr(result, "images", None)
    if not images:
        raise RuntimeError("绘图模型没有返回图片")

    output_path = OUTPUT_DIR / f"sd_turbo_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
    images[0].save(output_path)
    return output_path
