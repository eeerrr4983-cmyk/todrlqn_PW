"""Unit tests for the Korean OCR preprocessing pipeline."""

import unittest

import cv2
import numpy as np

from ocr_service.preprocessing import KoreanOCRPreprocessor, PreprocessingReport


class KoreanOCRPreprocessorTest(unittest.TestCase):
    """Validates the adaptive preprocessing behaviour."""

    def setUp(self) -> None:  # noqa: D401
        self.preprocessor = KoreanOCRPreprocessor()

    @staticmethod
    def _create_sample_image(width: int = 256, height: int = 128) -> bytes:
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        cv2.putText(canvas, '생기부 AI', (10, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
        success, buffer = cv2.imencode('.jpg', canvas)
        if not success:
            raise RuntimeError('샘플 이미지를 인코딩하지 못했습니다.')
        return buffer.tobytes()

    def test_preprocess_returns_report(self) -> None:
        """Ensure preprocessing returns both bytes and a detailed report."""
        raw_bytes = self._create_sample_image()
        processed_bytes, report = self.preprocessor.preprocess(raw_bytes)

        self.assertIsInstance(report, PreprocessingReport)
        self.assertGreater(len(processed_bytes), 0)
        self.assertGreater(report.sharpness, 0)
        self.assertGreaterEqual(report.brightness, 0)
        self.assertLessEqual(report.brightness, 1.0)

    def test_upscales_small_images(self) -> None:
        """Very small inputs should trigger scaling to protect tiny glyphs."""
        small_canvas = np.full((64, 64, 3), 200, dtype=np.uint8)
        success, buffer = cv2.imencode('.jpg', small_canvas)
        self.assertTrue(success)
        processed_bytes, report = self.preprocessor.preprocess(buffer.tobytes())

        self.assertGreaterEqual(report.scale_factor, 1.0)
        self.assertGreater(report.processed_shape[0] * report.processed_shape[1], 64 * 64)
        self.assertGreater(len(processed_bytes), len(buffer))

    def test_invalid_bytes_raise_error(self) -> None:
        """Clearly invalid byte streams should raise a ValueError."""
        with self.assertRaises(ValueError):
            self.preprocessor.preprocess(b'not-an-image')


if __name__ == '__main__':
    unittest.main()
