import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import startup_animation
from PyQt5.QtWidgets import QApplication


class StartupAnimationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def test_default_config_uses_brand_logo(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(
                startup_animation,
                "CONFIG_PATH",
                Path(temp_dir) / "startup_animation.json",
            ):
                config = startup_animation.load_startup_animation_config()
        self.assertTrue(config["enabled"])
        self.assertEqual(config["type"], "brand_logo")
        self.assertEqual(config["duration_seconds"], 5)
        self.assertEqual(config["style"], "mirror")

    def test_config_duration_is_bounded_and_style_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "startup_animation.json"
            path.write_text(
                json.dumps(
                    {
                        "enabled": True,
                        "type": "particles",
                        "duration_seconds": 40,
                        "style": "stack",
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(startup_animation, "CONFIG_PATH", path):
                config = startup_animation.load_startup_animation_config()
        self.assertEqual(config["type"], "brand_logo")
        self.assertEqual(config["duration_seconds"], 10)
        self.assertEqual(config["style"], "stack")

    def test_splash_uses_loader_progress_and_waits_for_loader_completion(self):
        splash = startup_animation.LogoSplash(duration_seconds=3)
        splash.set_progress(42, "loading model")
        self.assertEqual(splash.percent, 42)
        self.assertEqual(splash.status, "LOADING MODEL")

        splash.elapsed = 3
        splash.loader_done = True
        splash.loader_success = True
        splash.done_at = 9
        splash._clock = 9
        with (
            patch("startup_animation.time.monotonic", side_effect=[10, 10.5]),
            patch("startup_animation.QTimer.singleShot") as single_shot,
        ):
            splash._tick()
            self.assertEqual(splash.outro_at, 10)
            splash._tick()
        single_shot.assert_called_once()

    @classmethod
    def tearDownClass(cls):
        cls.app.quit()


if __name__ == "__main__":
    unittest.main()
