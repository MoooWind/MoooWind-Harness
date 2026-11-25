import os
import sys
import unittest
from unittest.mock import patch


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import Mooowind_App
from PyQt5.QtWidgets import QApplication


class AppUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def test_feature_buttons_and_busy_state(self):
        original_change_model = Mooowind_App.ChatApp.change_model
        Mooowind_App.ChatApp.change_model = lambda _self, _name: None
        window = None
        try:
            window = Mooowind_App.ChatApp()
            self.assertIsNotNone(window.homework_photo_btn)
            self.assertIsNotNone(window.quant_btn)
            self.assertIsNotNone(window.media_btn)
            self.assertIsNotNone(window.draw_image_btn)
            self.assertIsNotNone(window.blender_btn)
            self.assertIsNotNone(window.resume_task_btn)
            self.assertIsNotNone(window.discard_task_btn)
            self.assertEqual(window.combo_mode.count(), 3)

            window._set_feature_busy(True)
            self.assertFalse(window.homework_photo_btn.isEnabled())
            self.assertFalse(window.quant_btn.isEnabled())
            self.assertFalse(window.draw_image_btn.isEnabled())
            window._set_feature_busy(False)
            self.assertTrue(window.homework_photo_btn.isEnabled())
            self.assertTrue(window.quant_btn.isEnabled())
            self.assertTrue(window.draw_image_btn.isEnabled())
            self.assertEqual(
                Mooowind_App._text_model_id("Qwen-Coder-3B"),
                "Qwen/Qwen2.5-Coder-3B-Instruct",
            )
        finally:
            if window is not None:
                window.close()
            Mooowind_App.ChatApp.change_model = original_change_model

    def test_task_tree_shows_planner_subtasks_tools_and_live_status(self):
        original_change_model = Mooowind_App.ChatApp.change_model
        Mooowind_App.ChatApp.change_model = lambda _self, _name: None
        window = None
        try:
            window = Mooowind_App.ChatApp()
            window.on_task_tree_update(
                {
                    "event": "plan",
                    "root": "求导并解释",
                    "steps": [
                        {
                            "id": "differentiate",
                            "goal": "计算导数",
                            "tool": "math.differentiate",
                        },
                        {
                            "id": "explain",
                            "goal": "解释结果",
                            "tool": "respond",
                            "depends_on": ["differentiate"],
                        },
                    ],
                    "results": {},
                }
            )

            root = window.task_tree.topLevelItem(0)
            self.assertEqual(root.text(0), "求导并解释")
            self.assertEqual(root.child(0).text(0), "任务规划对话")
            self.assertEqual(root.childCount(), 3)
            self.assertEqual(root.child(1).child(0).text(0), "工具：math.differentiate")

            window.on_task_tree_update(
                {"event": "step_started", "step_id": "differentiate"}
            )
            self.assertEqual(root.child(1).text(1), "执行中")
            self.assertEqual(root.child(1).child(0).text(1), "执行中")

            window.on_task_tree_update(
                {
                    "event": "step_finished",
                    "step_id": "differentiate",
                    "status": "done",
                    "output": "2*x",
                }
            )
            self.assertEqual(root.child(1).text(1), "已完成")
            self.assertEqual(root.child(1).child(0).toolTip(0), "2*x")
        finally:
            if window is not None:
                window.close()
            Mooowind_App.ChatApp.change_model = original_change_model

    @patch("Mooowind_App.torch.cuda.current_device", return_value=0)
    @patch("Mooowind_App.torch.cuda.is_available", return_value=True)
    def test_text_model_device_map_targets_current_cuda_device(
        self,
        _cuda_available,
        _current_device,
    ):
        self.assertEqual(Mooowind_App._cuda_device_map(), {"": 0})

    @patch("Mooowind_App.torch.cuda.is_available", return_value=False)
    def test_text_model_loader_reports_missing_cuda(self, _cuda_available):
        with self.assertRaisesRegex(RuntimeError, "检测不到 CUDA GPU"):
            Mooowind_App._cuda_device_map()

    def test_text_model_loader_uses_local_snapshot_only(self):
        with (
            patch(
                "Mooowind_App.local_model_path",
                return_value="D:/models/local-qwen",
            ) as local_model_path,
            patch("Mooowind_App._cuda_device_map", return_value={"": 0}),
            patch("Mooowind_App.AutoTokenizer.from_pretrained") as load_tokenizer,
            patch("Mooowind_App.AutoModelForCausalLM.from_pretrained") as load_model,
        ):
            model, tokenizer = Mooowind_App._load_text_components("Qwen-7B")

        self.assertIs(model, load_model.return_value)
        self.assertIs(tokenizer, load_tokenizer.return_value)
        local_model_path.assert_called_once_with("Qwen/Qwen2.5-7B-Instruct")
        load_tokenizer.assert_called_once_with(
            "D:/models/local-qwen",
            local_files_only=True,
        )
        self.assertEqual(load_model.call_args.kwargs["local_files_only"], True)

    @classmethod
    def tearDownClass(cls):
        cls.app.quit()


if __name__ == "__main__":
    unittest.main()
