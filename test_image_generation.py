import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import image_generation


class FakePipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return type(
            "Result",
            (),
            {
                "images": [
                    Image.new("RGB", (kwargs["width"], kwargs["height"]))
                ]
            },
        )()


class ImageGenerationTests(unittest.TestCase):
    @patch("image_generation.local_model_path", return_value=Path("D:/models/sd-turbo"))
    @patch("torch.cuda.is_available", return_value=False)
    @patch("diffusers.DiffusionPipeline.from_pretrained")
    def test_pipeline_loads_local_snapshot_without_network(
        self,
        load_pipeline,
        _cuda_available,
        local_model_path,
    ):
        load_pipeline.return_value.to.return_value = load_pipeline.return_value

        image_generation.load_pipeline()

        local_model_path.assert_called_once_with(
            image_generation.MODEL_ID,
            required_file="model_index.json",
        )
        load_pipeline.assert_called_once_with(
            "D:\\models\\sd-turbo",
            torch_dtype=image_generation.torch.float32,
            variant=None,
            local_files_only=True,
        )

    def test_diffusers_pipeline_is_importable_without_downloading_weights(self):
        from diffusers import DiffusionPipeline

        self.assertTrue(callable(DiffusionPipeline.from_pretrained))

    def test_image_is_saved_and_seed_is_forwarded(self):
        pipeline = FakePipeline()
        original_dir = image_generation.OUTPUT_DIR
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                image_generation.OUTPUT_DIR = Path(temp_dir)
                path = image_generation.generate_image(
                    pipeline,
                    "a small house",
                    seed=7,
                )
                self.assertTrue(Path(path).is_file())
                self.assertEqual(pipeline.calls[0]["num_inference_steps"], 1)
                self.assertEqual(
                    pipeline.calls[0]["generator"].initial_seed(),
                    7,
                )
        finally:
            image_generation.OUTPUT_DIR = original_dir

    def test_unsupported_resolution_is_rejected(self):
        with self.assertRaises(ValueError):
            image_generation.generate_image(FakePipeline(), "test", 1024, 1024)


if __name__ == "__main__":
    unittest.main()
