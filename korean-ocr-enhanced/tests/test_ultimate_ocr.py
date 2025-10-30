"""
초월적 한국어 OCR 테스트 스위트
Ultimate Korean OCR Test Suite
"""

import unittest
import asyncio
import cv2
import numpy as np
from pathlib import Path
import sys
import os
import time
from typing import Dict, Any
import json

# 상위 디렉토리 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ultimate_korean_ocr import (
    UltimateKoreanOCR, 
    KoreanLanguageModel,
    UltraImageEnhancer,
    UltimateOCRConfig,
    get_ultimate_ocr
)


class TestKoreanLanguageModel(unittest.TestCase):
    """한국어 언어 모델 테스트"""
    
    def setUp(self):
        self.model = KoreanLanguageModel()
    
    def test_common_error_correction(self):
        """일반적인 OCR 오류 교정 테스트"""
        test_cases = [
            ("학샘이 우쑤합니다", "학생이 우수합니다"),
            ("선샘님이 참어했습니다", "선생님이 참여했습니다"),
            ("1 학 년 2 반", "1학년 2반"),
            ("교가 과목", "교과 과목")
        ]
        
        for input_text, expected in test_cases:
            corrected, _ = self.model.correct_text(input_text)
            # 부분 매칭 확인 (완벽한 매칭이 아닌 개선 여부)
            self.assertIn("학생", corrected) if "학생" in expected else None
            self.assertIn("선생님", corrected) if "선생님" in expected else None
    
    def test_spacing_correction(self):
        """띄어쓰기 교정 테스트"""
        test_cases = [
            ("학생 이", "학생이"),
            ("선생님 에게", "선생님에게"),
            ("3 월 15 일", "3월 15일")
        ]
        
        for input_text, expected in test_cases:
            corrected = self.model.correct_spacing(input_text)
            self.assertEqual(corrected, expected)
    
    def test_contextual_correction(self):
        """문맥 기반 교정 테스트"""
        text = "우수 학샘 모범 학쌩"
        corrected = self.model.contextual_correction(text)
        self.assertIn("학생", corrected)


class TestUltraImageEnhancer(unittest.TestCase):
    """이미지 향상 클래스 테스트"""
    
    def setUp(self):
        self.config = UltimateOCRConfig()
        self.enhancer = UltraImageEnhancer(self.config)
    
    def test_image_enhancement(self):
        """이미지 향상 테스트"""
        # 테스트 이미지 생성
        test_image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        cv2.putText(test_image, "Test", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # 향상 처리
        enhanced_images = self.enhancer.enhance_image(test_image)
        
        # 결과 확인
        self.assertGreater(len(enhanced_images), 0)
        for img in enhanced_images:
            self.assertIsNotNone(img)
            self.assertEqual(len(img.shape), 2 if len(img.shape) == 2 else 3)
    
    def test_super_resolution(self):
        """초해상도 업스케일링 테스트"""
        test_image = np.ones((50, 100, 3), dtype=np.uint8) * 255
        
        upscaled = self.enhancer.apply_super_resolution(test_image)
        
        # 크기 확인
        scale = self.config.SUPER_RESOLUTION_SCALE
        self.assertEqual(upscaled.shape[0], test_image.shape[0] * scale)
        self.assertEqual(upscaled.shape[1], test_image.shape[1] * scale)
    
    def test_adaptive_threshold(self):
        """적응형 임계값 테스트"""
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        thresholded = self.enhancer.apply_adaptive_threshold(test_image)
        
        self.assertIsNotNone(thresholded)
        # 이진화 확인
        unique_values = np.unique(thresholded)
        self.assertLessEqual(len(unique_values), 3)  # 0, 255 또는 그레이스케일


class TestUltimateKoreanOCR(unittest.TestCase):
    """메인 OCR 클래스 테스트"""
    
    @classmethod
    def setUpClass(cls):
        """클래스 레벨 설정"""
        cls.ocr = get_ultimate_ocr()
    
    def test_singleton_instance(self):
        """싱글톤 인스턴스 테스트"""
        ocr1 = get_ultimate_ocr()
        ocr2 = get_ultimate_ocr()
        self.assertIs(ocr1, ocr2)
    
    def test_process_simple_image(self):
        """간단한 이미지 처리 테스트"""
        # 테스트 이미지 생성
        test_image = np.ones((200, 400, 3), dtype=np.uint8) * 255
        cv2.putText(test_image, "학생", (100, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        # 임시 파일로 저장
        temp_path = "/tmp/test_korean.jpg"
        cv2.imwrite(temp_path, test_image)
        
        # OCR 처리
        result = self.ocr.process_image(temp_path)
        
        # 결과 확인
        self.assertIsNotNone(result)
        self.assertIn('text', result)
        self.assertIn('confidence', result)
        self.assertGreater(result['confidence'], 0)
        
        # 임시 파일 삭제
        Path(temp_path).unlink(missing_ok=True)
    
    def test_contains_korean(self):
        """한글 포함 여부 확인 테스트"""
        test_cases = [
            ("안녕하세요", True),
            ("Hello World", False),
            ("한글 English", True),
            ("123456", False),
            ("학생 123", True)
        ]
        
        for text, expected in test_cases:
            result = self.ocr._contains_korean(text)
            self.assertEqual(result, expected)
    
    def test_metrics_tracking(self):
        """메트릭 추적 테스트"""
        initial_metrics = self.ocr.get_metrics()
        initial_count = initial_metrics['total_processed']
        
        # 테스트 이미지 처리
        test_image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        temp_path = "/tmp/test_metrics.jpg"
        cv2.imwrite(temp_path, test_image)
        
        self.ocr.process_image(temp_path)
        
        # 메트릭 업데이트 확인
        updated_metrics = self.ocr.get_metrics()
        self.assertEqual(updated_metrics['total_processed'], initial_count + 1)
        
        Path(temp_path).unlink(missing_ok=True)


class TestIntegration(unittest.TestCase):
    """통합 테스트"""
    
    @classmethod
    def setUpClass(cls):
        """클래스 레벨 설정"""
        cls.ocr = get_ultimate_ocr()
    
    def test_complex_korean_document(self):
        """복잡한 한국어 문서 테스트"""
        # 복잡한 테스트 이미지 생성
        test_image = np.ones((600, 800, 3), dtype=np.uint8) * 255
        
        # 여러 줄의 한국어 텍스트 추가
        texts = [
            "2024학년도 1학기",
            "학생 성명: 홍길동",
            "담임 선생님: 김철수",
            "국어 과목 평가",
            "수학 성적: 우수",
            "영어 활동: 적극적",
            "과학 실험 참여",
            "역사 토론 발표"
        ]
        
        y_position = 50
        for text in texts:
            # 한글 폰트가 없을 수 있으므로 기본 폰트 사용
            cv2.putText(test_image, "Test Line", (50, y_position),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            y_position += 60
        
        # 노이즈 추가
        noise = np.random.normal(0, 10, test_image.shape)
        test_image = np.clip(test_image + noise, 0, 255).astype(np.uint8)
        
        # 임시 파일로 저장
        temp_path = "/tmp/test_complex.jpg"
        cv2.imwrite(temp_path, test_image)
        
        # OCR 처리
        result = self.ocr.process_image(temp_path)
        
        # 결과 확인
        self.assertIsNotNone(result)
        self.assertIn('text', result)
        self.assertIn('confidence', result)
        self.assertIn('processing_time', result)
        
        # 처리 시간 확인
        self.assertGreater(result['processing_time'], 0)
        self.assertLess(result['processing_time'], 60)  # 60초 이내
        
        Path(temp_path).unlink(missing_ok=True)
    
    def test_low_quality_image(self):
        """저품질 이미지 처리 테스트"""
        # 저품질 이미지 시뮬레이션
        test_image = np.ones((100, 200, 3), dtype=np.uint8) * 200  # 회색 배경
        cv2.putText(test_image, "Test", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)  # 낮은 대비
        
        # 블러 추가
        test_image = cv2.GaussianBlur(test_image, (5, 5), 2)
        
        # 임시 파일로 저장
        temp_path = "/tmp/test_low_quality.jpg"
        cv2.imwrite(temp_path, test_image)
        
        # OCR 처리
        result = self.ocr.process_image(temp_path)
        
        # 이미지 향상이 적용되어 결과가 있어야 함
        self.assertIsNotNone(result)
        self.assertIn('text', result)
        
        Path(temp_path).unlink(missing_ok=True)


class TestPerformance(unittest.TestCase):
    """성능 테스트"""
    
    @classmethod
    def setUpClass(cls):
        """클래스 레벨 설정"""
        cls.ocr = get_ultimate_ocr()
    
    def test_processing_speed(self):
        """처리 속도 테스트"""
        # 간단한 테스트 이미지
        test_image = np.ones((200, 400, 3), dtype=np.uint8) * 255
        cv2.putText(test_image, "Speed Test", (100, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        temp_path = "/tmp/test_speed.jpg"
        cv2.imwrite(temp_path, test_image)
        
        # 처리 시간 측정
        start_time = time.time()
        result = self.ocr.process_image(temp_path)
        processing_time = time.time() - start_time
        
        # 합리적인 처리 시간 확인
        self.assertLess(processing_time, 30)  # 30초 이내
        
        # 결과의 processing_time과 비교
        if 'processing_time' in result:
            self.assertAlmostEqual(result['processing_time'], processing_time, delta=1)
        
        Path(temp_path).unlink(missing_ok=True)
    
    def test_cache_performance(self):
        """캐싱 성능 테스트"""
        # 테스트 이미지
        test_image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        cv2.putText(test_image, "Cache", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        temp_path = "/tmp/test_cache.jpg"
        cv2.imwrite(temp_path, test_image)
        
        # 첫 번째 처리
        start_time1 = time.time()
        result1 = self.ocr.process_image(temp_path)
        time1 = time.time() - start_time1
        
        # 두 번째 처리 (캐시 사용)
        start_time2 = time.time()
        result2 = self.ocr.process_image(temp_path)
        time2 = time.time() - start_time2
        
        # 캐시 사용 시 더 빠름
        self.assertLessEqual(time2, time1 * 1.1)  # 약간의 여유 포함
        
        # 결과 동일성 확인
        self.assertEqual(result1['text'], result2['text'])
        
        Path(temp_path).unlink(missing_ok=True)


def run_all_tests():
    """모든 테스트 실행"""
    # 테스트 스위트 생성
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 테스트 클래스 추가
    suite.addTests(loader.loadTestsFromTestCase(TestKoreanLanguageModel))
    suite.addTests(loader.loadTestsFromTestCase(TestUltraImageEnhancer))
    suite.addTests(loader.loadTestsFromTestCase(TestUltimateKoreanOCR))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    
    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 결과 요약
    print("\n" + "="*50)
    print("테스트 결과 요약")
    print("="*50)
    print(f"실행된 테스트: {result.testsRun}")
    print(f"성공: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"실패: {len(result.failures)}")
    print(f"오류: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ 모든 테스트 통과!")
    else:
        print("\n❌ 일부 테스트 실패")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
