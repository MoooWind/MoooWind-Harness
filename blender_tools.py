import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "data" / "blender_config.json"
OUTPUT_DIR = BASE_DIR / "Output" / "BlenderScenes"
MAX_OBJECTS = 20
PRIMITIVES = {
    "cube": "primitive_cube_add",
    "sphere": "primitive_uv_sphere_add",
    "cylinder": "primitive_cylinder_add",
    "cone": "primitive_cone_add",
    "torus": "primitive_torus_add",
    "plane": "primitive_plane_add",
}


def set_blender_executable(path):
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file() or candidate.name.lower() != "blender.exe":
        raise ValueError("请选择 Blender 安装目录中的 blender.exe")
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps({"executable": str(candidate)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return candidate


def find_blender_executable():
    configured = os.environ.get("BLENDER_EXE")
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return candidate

    if CONFIG_PATH.is_file():
        try:
            config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            candidate = Path(config.get("executable", ""))
            if candidate.is_file():
                return candidate
        except (OSError, json.JSONDecodeError, TypeError):
            pass

    command = shutil.which("blender")
    if command:
        return Path(command)

    roots = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        / "Blender Foundation",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Blender Foundation",
        Path(r"D:\Program Files\Blender Foundation"),
    ]
    for root in roots:
        if root.is_dir():
            candidates = sorted(
                root.glob("Blender*/blender.exe"),
                key=lambda item: item.parent.name,
                reverse=True,
            )
            if candidates:
                return candidates[0]
    return None


def available():
    return find_blender_executable() is not None


def _vector(value, name, minimum, maximum):
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(
            not isinstance(component, (int, float))
            or isinstance(component, bool)
            or not minimum <= component <= maximum
            for component in value
        )
    ):
        raise ValueError(
            f"{name} 必须是三个介于 {minimum} 和 {maximum} 之间的数字"
        )
    return [float(component) for component in value]


def validate_scene_spec(spec):
    if not isinstance(spec, dict):
        raise ValueError("Blender 场景规格必须是 JSON 对象")
    title = spec.get("title", "MoooWind Scene")
    objects = spec.get("objects")
    if not isinstance(title, str) or not title.strip() or len(title) > 100:
        raise ValueError("场景标题必须是 1 到 100 个字符")
    if not isinstance(objects, list) or not 1 <= len(objects) <= MAX_OBJECTS:
        raise ValueError(f"objects 必须包含 1 到 {MAX_OBJECTS} 个物体")

    validated = []
    for item in objects:
        if not isinstance(item, dict):
            raise ValueError("每个物体规格必须是 JSON 对象")
        primitive = item.get("type")
        if primitive not in PRIMITIVES:
            raise ValueError(f"不支持的 Blender 基础物体类型：{primitive}")
        name = item.get("name", primitive)
        if not isinstance(name, str) or not name.strip() or len(name) > 80:
            raise ValueError("物体名称必须是 1 到 80 个字符")
        location = _vector(item.get("location", [0, 0, 0]), "location", -100, 100)
        scale = _vector(item.get("scale", [1, 1, 1]), "scale", 0.05, 20)
        rotation = _vector(item.get("rotation", [0, 0, 0]), "rotation", -360, 360)
        color = item.get("color", [0.35, 0.55, 0.8])
        if (
            not isinstance(color, (list, tuple))
            or len(color) != 3
            or any(
                not isinstance(component, (int, float))
                or isinstance(component, bool)
                or not 0 <= component <= 1
                for component in color
            )
        ):
            raise ValueError("color 必须是三个介于 0 和 1 之间的 RGB 数值")
        roughness = item.get("roughness", 0.45)
        if (
            not isinstance(roughness, (int, float))
            or isinstance(roughness, bool)
            or not math.isfinite(roughness)
        ):
            raise ValueError("roughness 必须是有限数值")
        validated.append(
            {
                "type": primitive,
                "name": name.strip(),
                "location": location,
                "scale": scale,
                "rotation": rotation,
                "color": [float(component) for component in color],
                "roughness": min(1.0, max(0.0, float(roughness))),
            }
        )
    return {"title": title.strip(), "objects": validated}


def _script_for_scene(spec, project_path, render_path):
    scene_json = json.dumps(spec, ensure_ascii=True)
    project_literal = repr(str(project_path))
    render_literal = repr(str(render_path))
    return f'''import bpy
import json
import math
from mathutils import Vector

spec = json.loads({scene_json!r})
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
for item in spec["objects"]:
    primitive = item["type"]
    if primitive == "cube":
        bpy.ops.mesh.primitive_cube_add(size=2, location=item["location"])
    elif primitive == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, location=item["location"])
    elif primitive == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(vertices=32, location=item["location"])
    elif primitive == "cone":
        bpy.ops.mesh.primitive_cone_add(vertices=32, location=item["location"])
    elif primitive == "torus":
        bpy.ops.mesh.primitive_torus_add(major_segments=48, minor_segments=12, location=item["location"])
    elif primitive == "plane":
        bpy.ops.mesh.primitive_plane_add(size=2, location=item["location"])
    obj = bpy.context.object
    obj.name = item["name"]
    obj.scale = item["scale"]
    obj.rotation_euler = [math.radians(value) for value in item["rotation"]]
    material = bpy.data.materials.new(name=item["name"] + "_Material")
    material.diffuse_color = (*item["color"], 1.0)
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled:
        principled.inputs["Base Color"].default_value = (*item["color"], 1.0)
        principled.inputs["Roughness"].default_value = item["roughness"]
    obj.data.materials.append(material)

scene = bpy.context.scene
try:
    scene.render.engine = "BLENDER_EEVEE_NEXT"
except (TypeError, ValueError):
    scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = 768
scene.render.resolution_y = 768
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = {render_literal}
scene.world.color = (0.08, 0.08, 0.08)

bpy.ops.object.camera_add(location=(7, -9, 7))
camera = bpy.context.object
camera.name = "MoooWind_Camera"
target = Vector((0, 0, 0))
camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
camera.data.lens = 50
scene.camera = camera

bpy.ops.object.light_add(type="AREA", location=(4, -4, 8))
light = bpy.context.object
light.data.energy = 1400
light.data.shape = "DISK"
light.data.size = 6

scene.render.image_settings.color_mode = "RGBA"
bpy.ops.wm.save_as_mainfile(filepath={project_literal})
bpy.ops.render.render(write_still=True)
'''


def create_scene(spec, executable=None, timeout_seconds=240):
    blender = Path(executable) if executable else find_blender_executable()
    if blender is None or not blender.is_file():
        raise FileNotFoundError(
            "没有找到 Blender。请在界面选择 blender.exe，或设置 BLENDER_EXE。"
        )
    safe_spec = validate_scene_spec(spec)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    project_path = OUTPUT_DIR / f"scene_{timestamp}.blend"
    render_path = OUTPUT_DIR / f"scene_{timestamp}.png"
    script_path = OUTPUT_DIR / f"scene_{timestamp}.py"
    script_path.write_text(
        _script_for_scene(safe_spec, project_path, render_path),
        encoding="utf-8",
    )
    try:
        result = subprocess.run(
            [str(blender), "--background", "--python", str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(
            f"Blender 超过 {timeout_seconds} 秒仍未完成；生成脚本保留在 {script_path}"
        ) from error
    if result.returncode != 0:
        details = (result.stderr or result.stdout)[-3000:]
        raise RuntimeError(
            f"Blender 执行失败（退出码 {result.returncode}）：{details}"
        )
    if not project_path.is_file() or not render_path.is_file():
        raise RuntimeError(
            f"Blender 返回成功但没有生成预期的 .blend 和预览图。日志："
            f"{(result.stdout or '')[-2000:]}"
        )
    return {"project": str(project_path), "render": str(render_path)}
