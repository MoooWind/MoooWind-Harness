
# Optional prediction and media features

Install the extra feature dependencies in the same Python environment used to
run MoooWind (the current development environment has these installed):

```powershell
python -m pip install -r requirements-feature-modules.txt
```

## A-share forecasts

Choose **A股预测** in the app, then use AkShare online daily data or a local
CSV. A CSV must contain date, open, high, low, close, and volume columns. The
module accepts English column names (`date`, `open`, `high`, `low`, `close`,
`volume`) and common Chinese equivalents (`日期`, `开盘`, `最高`, `最低`,
`收盘`, `成交量`). At least roughly 1.5 calendar years of daily rows are needed;
longer history is preferable.

The module reports model estimates and a held-out historical validation result.
These are experimental statistics, not reliable return guarantees or investment
advice. It does not connect to a broker and cannot place orders. AkShare requires
an internet connection.

## Local image and video recognition

Choose **识别图片/视频** to analyze JPG, PNG, BMP, WEBP, MP4, MOV, AVI, MKV, or
WMV files. The Qwen2.5-VL 3B model is loaded from the local `models` cache
downloaded by `download_all_models.py`; the app does not download it at runtime.
Media contents are processed locally; video recognition samples at most eight
frames and does not analyze audio. The app temporarily unloads its text model to
make room on the GPU, then attempts to restore the previous text model.

## Handwritten homework photos and local math tools

Choose **上传照片解题** to have the local vision model transcribe a handwritten
or printed problem. You can review and correct the transcription before it is
sent to the local language model. The resulting question is also searched
against the imported book library, so you do not need to specify a chapter.
Handwriting and dense formulas can still be misread; review the transcription.

The work-mode planner can use local SymPy tools for expression simplification,
differentiation, integration, limits, and equation solving. These tools run in
Python without a hosted calculation API. They support common arithmetic and
functions, not every textbook notation or proof.

## Local image generation

Choose **本地AI绘图** to generate a 512-pixel image with SD-Turbo. The model is
loaded from the local `AI-ModelScope/sd-turbo` snapshot in the `models` cache;
generation runs locally and does not download weights at runtime.
The app temporarily unloads the active text model and uses CPU offload on CUDA
to fit the image model into limited VRAM. Output PNGs are saved under
`Output/GeneratedImages`. This model is for prompt-based image generation, not
an image editor or a Blender replacement.

## Blender and code model

Choose **Blender场景** to describe a simple 3D scene. The app can find Blender
from `PATH`, common Windows installation folders, or `BLENDER_EXE`; otherwise
select `blender.exe` once in the file picker. A local language model turns the
description into a validated, restricted list of basic meshes, materials,
camera, and lighting. Blender runs in background mode and saves a `.blend`
project and PNG preview under `Output/BlenderScenes`. It does not execute
model-written Python scripts.

The mode selector now includes **代码模式 (Qwen Coder)**, which loads the
locally cached `Qwen/Qwen2.5-Coder-3B-Instruct` model. Text models are read from
the `models` cache created by `download_all_models.py`; missing files produce an
explicit error instead of triggering a network download. Generated code is not
automatically run or applied to workspace files.

## Work-mode task execution

Work and code tasks plan up to five sequential steps using one plain-text line
per step (`[tool] goal | key=value; key=value`) instead of model-generated
nested JSON. The parser validates every tool against the task's allowlist;
steps run sequentially with a fresh context and only their declared dependency
results.

Word/PPT requests use `pipeline_orchestrator.py`: a `Blackboard` carries the
task, Markdown outline/content, results, errors, and output paths; `ModelRouter`
selects the Office pipeline. Office work is split into outline, content, and
one final export node per requested file. The outline and content are separate
plain-Markdown conversations generated with Qwen2.5 7B. Existing deterministic
Word/PPT renderers create the files; model-generated Python is not executed.
If another text model was active, the worker unloads it before loading Qwen and
restores it after the Office pipeline, so two text models are not kept resident
together.

The local tool registry includes book search, explicit-path text-file reading,
SymPy calculations, and user-authorized Word/PPT generation. Blender scene
creation is offered only when Blender is detected and the task mentions
Blender/3D. Stock prediction, image generation, and media recognition remain
separate manual actions. Checkpoints remain local JSON written by the app; the
language model does not generate that checkpoint JSON.

Work and code tasks save an atomic checkpoint under `data/task_checkpoint.json`
before planning and after every step. On the next launch, choose **继续未完成
任务** to retry the interrupted step and reuse successful step results. Older
Office checkpoints are migrated to the Markdown outline/content/export plan;
old results are not reused when their step IDs do not match. If an interrupted
step may have created a file, resuming it can create a duplicate; the app asks
for confirmation when such a step was active. **丢弃任务进度** removes the
saved checkpoint. Companion-mode chat and standalone media, image, stock, and
book-import operations are not resumable task plans.

## Local model inventory

The current `models` cache contains all six models referenced by the app:

- `Qwen/Qwen2.5-7B-Instruct` — companion chat and Office outline/content.
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` — work mode.
- `Qwen/Qwen2.5-Coder-3B-Instruct` — code mode.
- `Qwen/Qwen2.5-1.5B-Instruct` — small-model/game mode.
- `Qwen/Qwen2.5-VL-3B-Instruct` — image, video, and handwriting recognition.
- `AI-ModelScope/sd-turbo` — local image generation.

No additional model download is required for the current task pipeline. Model
loading is local-only; missing files are reported rather than fetched at
runtime.
