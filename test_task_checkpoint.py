import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import task_checkpoint


class TaskCheckpointTests(unittest.TestCase):
    def test_checkpoint_round_trip_and_clear(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "task_checkpoint.json"
            checkpoint = task_checkpoint.create_checkpoint(
                "work",
                "求 x^2 的导数并解释",
            )
            checkpoint["steps"] = [
                {
                    "id": "differentiate",
                    "goal": "求导",
                    "tool": "math.differentiate",
                    "arguments": {"expression": "x**2"},
                    "depends_on": [],
                }
            ]
            checkpoint["results"] = [
                {
                    "id": "differentiate",
                    "status": "done",
                    "output": '{"result": "2*x"}',
                }
            ]
            with patch.object(task_checkpoint, "CHECKPOINT_PATH", checkpoint_path):
                task_checkpoint.save_checkpoint(checkpoint)
                self.assertEqual(task_checkpoint.load_checkpoint()["results"], checkpoint["results"])
                task_checkpoint.clear_checkpoint()
                self.assertIsNone(task_checkpoint.load_checkpoint())

    def test_invalid_checkpoint_mode_is_rejected(self):
        checkpoint = task_checkpoint.create_checkpoint("work", "task")
        checkpoint["mode"] = "life"
        with self.assertRaises(ValueError):
            task_checkpoint._validate_checkpoint(checkpoint)


if __name__ == "__main__":
    unittest.main()
