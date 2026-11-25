import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

import vision_tools


class MockBatch(dict):
    def __init__(self):
        super().__init__(input_ids=torch.tensor([[1, 2]]))
        self.input_ids = self["input_ids"]

    def to(self, _device):
        return self


class MockProcessor:
    def apply_chat_template(self, messages, **_kwargs):
        self.prompt = messages[0]["content"][-1]["text"]
        return "mock prompt"

    def __call__(self, text, images, **_kwargs):
        self.image_size = images[0].size
        return MockBatch()

    def batch_decode(self, _output, **_kwargs):
        return ["x^2 + 1"]


class MockModel:
    def generate(self, **_kwargs):
        return torch.tensor([[1, 2, 3]])


class VisionToolsTests(unittest.TestCase):
    @patch("torch.cuda.current_device", return_value=0)
    @patch("torch.cuda.is_available", return_value=True)
    @patch("vision_tools.local_model_path", return_value="D:/models/local-vision")
    @patch("transformers.AutoProcessor.from_pretrained")
    @patch("transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained")
    def test_vision_loader_uses_local_snapshot_only(
        self,
        load_model,
        load_processor,
        local_model_path,
        _cuda_available,
        _current_device,
    ):
        model, processor = vision_tools.load_vision_model()

        self.assertIs(model, load_model.return_value)
        self.assertIs(processor, load_processor.return_value)
        local_model_path.assert_called_once_with(vision_tools.VISION_MODEL_ID)
        load_processor.assert_called_once_with(
            "D:/models/local-vision",
            local_files_only=True,
        )
        self.assertEqual(load_model.call_args.kwargs["local_files_only"], True)

    def test_handwriting_image_is_resized_and_transcription_prompted(self):
        processor = MockProcessor()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "handwritten.png"
            Image.new("RGB", (1600, 1200), color="white").save(path)
            result = vision_tools.analyze_media(
                path,
                vision_tools.HANDWRITING_PROMPT,
                MockModel(),
                processor,
                image_max_side=vision_tools.HANDWRITING_IMAGE_SIDE,
            )

        self.assertEqual(result, "x^2 + 1")
        self.assertEqual(max(processor.image_size), vision_tools.HANDWRITING_IMAGE_SIDE)
        self.assertIn("LaTeX", processor.prompt)
        self.assertIn("[?]", processor.prompt)


if __name__ == "__main__":
    unittest.main()
