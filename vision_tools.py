from pathlib import Path

import torch
from PIL import Image
from transformers import BitsAndBytesConfig

from model_cache import local_model_path


VISION_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv"}
MAX_VIDEO_FRAMES = 8
MAX_IMAGE_SIDE = 768
HANDWRITING_IMAGE_SIDE = 1024
HANDWRITING_PROMPT = (
    "请只转写图片中可见的手写/印刷文字与数学表达式，不要解题或补全缺失内容。"
    "尽量保留分数、根号、上下标、括号、积分号、极限符号和等号；"
    "数学式另给一份 LaTeX 和一份便于计算器输入的纯文本式（乘方用 **，根号写 sqrt(...)）。"
    "无法确定的字符用 [?] 标出，不要猜。"
)


def load_vision_model():
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError(
            "视觉模型需要 CUDA GPU；当前 Python 环境检测不到 CUDA。"
            "请确认使用 CUDA 版 PyTorch 和正确的 NVIDIA 驱动。"
        )
    device_map = {"": torch.cuda.current_device()}
    model_dir = local_model_path(VISION_MODEL_ID)
    processor = AutoProcessor.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        quantization_config=quantization,
        torch_dtype=torch.float16,
        device_map=device_map,
        local_files_only=True,
    )
    model.eval()
    return model, processor


def _resize_image(image, max_side=MAX_IMAGE_SIDE):
    image = image.convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def _read_image(path, max_side=MAX_IMAGE_SIDE):
    with Image.open(path) as image:
        return _resize_image(image.copy(), max_side)


def _read_video_frames(path, progress_callback):
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "读取视频需要安装 opencv-python："
            "python -m pip install -r requirements-feature-modules.txt"
        ) from error

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"无法打开视频文件：{path}")

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if frame_count <= 0:
            raise ValueError("视频没有可读取的帧")

        sample_count = min(MAX_VIDEO_FRAMES, frame_count)
        if sample_count == 1:
            frame_indexes = [0]
        else:
            frame_indexes = sorted(
                {
                    round(index * (frame_count - 1) / (sample_count - 1))
                    for index in range(sample_count)
                }
            )

        frames = []
        for frame_number in frame_indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = capture.read()
            if not success:
                continue
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = _resize_image(Image.fromarray(rgb_frame))
            timestamp = frame_number / fps if fps > 0 else float(frame_number)
            frames.append((image, timestamp))
            progress_callback(
                f"已提取视频帧 {len(frames)}/{len(frame_indexes)}"
            )

        if not frames:
            raise ValueError("未能从视频中提取可识别的画面")
        return frames
    finally:
        capture.release()


def analyze_media(
    path,
    prompt,
    model,
    processor,
    progress_callback=lambda _message: None,
    image_max_side=MAX_IMAGE_SIDE,
):
    media_path = Path(path)
    extension = media_path.suffix.lower()
    if not media_path.is_file():
        raise FileNotFoundError(f"找不到媒体文件：{media_path}")

    if extension in IMAGE_EXTENSIONS:
        images = [_read_image(media_path, image_max_side)]
        content = [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]
    elif extension in VIDEO_EXTENSIONS:
        frames = _read_video_frames(media_path, progress_callback)
        images = [image for image, _timestamp in frames]
        content = []
        for frame_index, (_image, timestamp) in enumerate(frames, start=1):
            content.append(
                {
                    "type": "text",
                    "text": f"视频时间点 {timestamp:.1f} 秒，第 {frame_index} 个抽样画面。",
                }
            )
            content.append({"type": "image"})
        content.append(
            {
                "type": "text",
                "text": (
                    "请按时间顺序分析这些视频关键帧，说明可见场景、动作和变化；"
                    "不要推断画面中无法确认的信息。"
                    "本次只分析画面，不包含音频。\n"
                    + prompt
                ),
            }
        )
    else:
        raise ValueError("支持的图片格式为 JPG、PNG、BMP、WEBP；视频为 MP4、MOV、AVI、MKV、WMV")

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text],
        images=images,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    progress_callback("视觉模型正在分析媒体内容...")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=600,
            do_sample=False,
        )
    generated = output[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
