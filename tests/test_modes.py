"""Request validation for the simple Studio modes (no GPU required)."""

import tempfile
import unittest
from pathlib import Path

from core_worker import validate_request


class ModeValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "image.png"
        self.audio = self.root / "audio.wav"
        self.image.write_bytes(b"image placeholder")
        self.audio.write_bytes(b"audio placeholder")
        self.base = {
            "width": 768, "height": 512, "frames": 49, "fps": 24,
            "seed": 1, "output_dir": str(self.root),
        }

    def test_existing_image_audio_mode_still_requires_image(self):
        with self.assertRaisesRegex(ValueError, "first-frame image"):
            validate_request(self.base)
        result = validate_request({**self.base, "image_path": str(self.image)})
        self.assertEqual(result["mode"], "image_audio")
        self.assertEqual(result["image_path"], str(self.image))

    def test_text_to_video_uses_no_image_or_uploaded_audio(self):
        result = validate_request({**self.base, "mode": "text_video", "prompt": "A river at sunrise"})
        self.assertEqual(result["image_path"], "")
        self.assertEqual(result["audio_path"], "")

    def test_audio_to_video_requires_audio_and_prompt(self):
        with self.assertRaisesRegex(ValueError, "Describe"):
            validate_request({**self.base, "mode": "audio_video", "audio_path": str(self.audio)})
        with self.assertRaisesRegex(ValueError, "audio track"):
            validate_request({**self.base, "mode": "audio_video", "prompt": "A singer on stage"})
        result = validate_request({**self.base, "mode": "audio_video",
                                   "audio_path": str(self.audio), "prompt": "A singer on stage"})
        self.assertEqual(result["audio_path"], str(self.audio))

    def test_unknown_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown generation mode"):
            validate_request({**self.base, "mode": "unknown"})


if __name__ == "__main__":
    unittest.main()
