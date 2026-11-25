import unittest
from unittest.mock import patch

import blender_tools
import tool_registry


class BlenderToolsTests(unittest.TestCase):
    def test_scene_spec_is_limited_to_validated_primitives(self):
        spec = blender_tools.validate_scene_spec(
            {
                "title": "Simple scene",
                "objects": [
                    {
                        "type": "cube",
                        "location": [1, 2, 3],
                        "color": [0.2, 0.4, 0.6],
                    }
                ],
            }
        )
        self.assertEqual(spec["objects"][0]["type"], "cube")

        with self.assertRaises(ValueError):
            blender_tools.validate_scene_spec(
                {
                    "objects": [
                        {
                            "type": "python",
                            "code": "__import__('os').system('whoami')",
                        }
                    ]
                }
            )

    def test_generated_script_uses_validated_spec_not_model_code(self):
        spec = blender_tools.validate_scene_spec(
            {"objects": [{"type": "sphere", "name": "Ball"}]}
        )
        script = blender_tools._script_for_scene(
            spec,
            "scene.blend",
            "scene.png",
        )
        self.assertIn("bpy.ops.mesh.primitive_uv_sphere_add", script)
        self.assertNotIn("__import__", script)
        self.assertIn("scene.blend", script)

    @patch("blender_tools.find_blender_executable")
    def test_blender_tool_is_only_available_for_blender_tasks(self, find_blender):
        find_blender.return_value = object()
        ordinary_tools = tool_registry.tools_for_task("解释一个概念", set())
        scene_tools = tool_registry.tools_for_task("用 Blender 创建一个三维场景", set())
        self.assertNotIn("blender.scene", ordinary_tools)
        self.assertIn("blender.scene", scene_tools)


if __name__ == "__main__":
    unittest.main()
