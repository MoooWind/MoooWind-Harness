import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import model_cache


class ModelCacheTests(unittest.TestCase):
    def test_resolves_modelscope_snapshot_under_configured_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir)
            snapshot = (
                cache_root
                / "models"
                / "Qwen--Qwen2.5-7B-Instruct"
                / "snapshots"
                / "master"
            )
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")

            with patch.object(model_cache, "MODEL_CACHE_DIR", cache_root):
                self.assertEqual(
                    model_cache.local_model_path("Qwen/Qwen2.5-7B-Instruct"),
                    snapshot,
                )

    def test_missing_local_model_fails_without_falling_back_to_network(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(model_cache, "MODEL_CACHE_DIR", Path(temp_dir)):
                with self.assertRaisesRegex(FileNotFoundError, "download_all_models.py"):
                    model_cache.local_model_path("Qwen/Qwen2.5-7B-Instruct")

    def test_rejects_model_ids_that_could_escape_cache_directory(self):
        with self.assertRaises(ValueError):
            model_cache.local_model_path("../outside/model")


if __name__ == "__main__":
    unittest.main()
