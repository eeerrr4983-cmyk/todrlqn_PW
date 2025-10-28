"""
한국어 OCR 성능 벤치마크 및 테스트 시스템
정확도 100% 달성을 위한 종합 테스트
"""

import unittest
import numpy as np
import cv2
import time
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from Levenshtein import distance as levenshtein_distance
import logging

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.korean_ocr_engine import UltraKoreanOCR, OCRConfig
from src.deep_learning_enhancer import DeepLearningOCREnhancer
from src.layout_analyzer import SchoolRecordLayoutAnalyzer

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCRBenchmark:
    """OCR 성능 벤치마크 클래스"""
    
    def __init__(self):
        self.ocr_engine = UltraKoreanOCR(OCRConfig())
        self.deep_enhancer = DeepLearningOCREnhancer()
        self.layout_analyzer = SchoolRecordLayoutAnalyzer()
        self.results = []
        
    def run_comprehensive_benchmark(self, test_dataset_path: str) -> Dict[str, Any]:
        """
        종합 벤치마크 실행
        
        Args:
            test_dataset_path: 테스트 데이터셋 경로
            
        Returns:
            벤치마크 결과
        """
        logger.info("Starting comprehensive benchmark...")
        
        # 테스트 데이터 로드
        test_data = self.load_test_dataset(test_dataset_path)
        
        # 각 테스트 이미지에 대해 OCR 수행
        for item in test_data:
            image_path = item["image_path"]
            ground_truth = item["ground_truth"]
            
            # OCR 실행 및 시간 측정
            start_time = time.time()
            
            # 기본 OCR
            basic_result = self.ocr_engine.process_image(
                image_path, 
                enable_enhancement=False,
                enable_multi_pass=False
            )
            basic_time = time.time() - start_time
            
            # 향상된 OCR
            enhanced_start = time.time()
            enhanced_result = self.ocr_engine.process_image(
                image_path,
                enable_enhancement=True,
                enable_multi_pass=True
            )
            enhanced_time = time.time() - enhanced_start
            
            # 딥러닝 향상 적용
            dl_start = time.time()
            image = cv2.imread(image_path)
            dl_result = self.deep_enhancer.enhance_ocr_result(image, enhanced_result)
            dl_time = time.time() - dl_start
            
            # 결과 평가
            evaluation = self.evaluate_result(
                ground_truth,
                basic_result.get("full_text", ""),
                enhanced_result.get("full_text", ""),
                dl_result.get("full_text", "")
            )
            
            # 결과 저장
            self.results.append({
                "image": Path(image_path).name,
                "basic_accuracy": evaluation["basic_accuracy"],
                "enhanced_accuracy": evaluation["enhanced_accuracy"],
                "dl_accuracy": evaluation["dl_accuracy"],
                "basic_time": basic_time,
                "enhanced_time": enhanced_time,
                "dl_time": dl_time,
                "character_accuracy": evaluation["character_accuracy"],
                "word_accuracy": evaluation["word_accuracy"],
                "levenshtein_score": evaluation["levenshtein_score"]
            })
            
            logger.info(f"Processed {Path(image_path).name}: "
                       f"DL Accuracy={evaluation['dl_accuracy']:.2%}")
        
        # 종합 통계 계산
        summary = self.calculate_summary_statistics()
        
        # 리포트 생성
        self.generate_benchmark_report(summary)
        
        return summary
    
    def load_test_dataset(self, dataset_path: str) -> List[Dict]:
        """테스트 데이터셋 로드"""
        # 실제 구현에서는 JSON 파일이나 데이터베이스에서 로드
        # 여기서는 샘플 데이터 생성
        test_data = []
        
        # 샘플 테스트 케이스
        sample_cases = [
            {
                "image_path": "./data/test1.jpg",
                "ground_truth": "학생은 성실하고 책임감이 강하며 학업에 대한 열정이 뛰어남"
            },
            {
                "image_path": "./data/test2.jpg",
                "ground_truth": "2학년 1학기 국어 과목에서 우수한 성적을 거두었으며"
            },
            {
                "image_path": "./data/test3.jpg",
                "ground_truth": "창의적체험활동에서 리더십을 발휘하여 동아리 활동을 주도함"
            }
        ]
        
        for case in sample_cases:
            # 실제 이미지가 없는 경우 더미 이미지 생성
            if not os.path.exists(case["image_path"]):
                self.create_dummy_image(case["image_path"], case["ground_truth"])
            
            test_data.append(case)
        
        return test_data
    
    def create_dummy_image(self, image_path: str, text: str):
        """테스트용 더미 이미지 생성"""
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        
        # 흰 배경 이미지 생성
        img = np.ones((200, 800, 3), dtype=np.uint8) * 255
        
        # 텍스트 추가
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text[:50], (10, 100), font, 0.8, (0, 0, 0), 2)
        
        # 노이즈 추가 (실제 스캔 문서 시뮬레이션)
        noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        
        cv2.imwrite(image_path, img)
    
    def evaluate_result(self, ground_truth: str, basic: str, 
                       enhanced: str, dl: str) -> Dict[str, float]:
        """OCR 결과 평가"""
        def calculate_accuracy(pred: str, truth: str) -> float:
            if not truth:
                return 0.0
            
            # 문자 단위 정확도
            correct = sum(1 for p, t in zip(pred, truth) if p == t)
            total = max(len(pred), len(truth))
            return correct / total if total > 0 else 0.0
        
        def word_accuracy(pred: str, truth: str) -> float:
            pred_words = pred.split()
            truth_words = truth.split()
            
            if not truth_words:
                return 0.0
            
            correct = sum(1 for p in pred_words if p in truth_words)
            return correct / len(truth_words)
        
        def normalized_levenshtein(s1: str, s2: str) -> float:
            max_len = max(len(s1), len(s2))
            if max_len == 0:
                return 1.0
            return 1 - (levenshtein_distance(s1, s2) / max_len)
        
        return {
            "basic_accuracy": calculate_accuracy(basic, ground_truth),
            "enhanced_accuracy": calculate_accuracy(enhanced, ground_truth),
            "dl_accuracy": calculate_accuracy(dl, ground_truth),
            "character_accuracy": calculate_accuracy(dl, ground_truth),
            "word_accuracy": word_accuracy(dl, ground_truth),
            "levenshtein_score": normalized_levenshtein(dl, ground_truth)
        }
    
    def calculate_summary_statistics(self) -> Dict[str, Any]:
        """종합 통계 계산"""
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results)
        
        summary = {
            "total_tests": len(self.results),
            "average_dl_accuracy": df["dl_accuracy"].mean(),
            "min_dl_accuracy": df["dl_accuracy"].min(),
            "max_dl_accuracy": df["dl_accuracy"].max(),
            "std_dl_accuracy": df["dl_accuracy"].std(),
            "average_character_accuracy": df["character_accuracy"].mean(),
            "average_word_accuracy": df["word_accuracy"].mean(),
            "average_levenshtein_score": df["levenshtein_score"].mean(),
            "average_processing_time": df["dl_time"].mean(),
            "improvement_over_basic": (df["dl_accuracy"] - df["basic_accuracy"]).mean(),
            "improvement_over_enhanced": (df["dl_accuracy"] - df["enhanced_accuracy"]).mean(),
            "perfect_accuracy_rate": (df["dl_accuracy"] >= 0.99).mean(),
            "high_accuracy_rate": (df["dl_accuracy"] >= 0.95).mean(),
            "results_dataframe": df
        }
        
        return summary
    
    def generate_benchmark_report(self, summary: Dict[str, Any]):
        """벤치마크 리포트 생성"""
        logger.info("\n" + "="*80)
        logger.info("한국어 OCR 벤치마크 리포트")
        logger.info("="*80)
        
        logger.info(f"\n📊 종합 성능 지표:")
        logger.info(f"  • 총 테스트 수: {summary['total_tests']}")
        logger.info(f"  • 평균 정확도: {summary['average_dl_accuracy']:.2%}")
        logger.info(f"  • 최소 정확도: {summary['min_dl_accuracy']:.2%}")
        logger.info(f"  • 최대 정확도: {summary['max_dl_accuracy']:.2%}")
        logger.info(f"  • 표준편차: {summary['std_dl_accuracy']:.4f}")
        
        logger.info(f"\n🎯 세부 정확도:")
        logger.info(f"  • 문자 단위 정확도: {summary['average_character_accuracy']:.2%}")
        logger.info(f"  • 단어 단위 정확도: {summary['average_word_accuracy']:.2%}")
        logger.info(f"  • Levenshtein 점수: {summary['average_levenshtein_score']:.2%}")
        
        logger.info(f"\n⚡ 성능 지표:")
        logger.info(f"  • 평균 처리 시간: {summary['average_processing_time']:.3f}초")
        logger.info(f"  • 99% 이상 정확도 달성률: {summary['perfect_accuracy_rate']:.1%}")
        logger.info(f"  • 95% 이상 정확도 달성률: {summary['high_accuracy_rate']:.1%}")
        
        logger.info(f"\n📈 개선 효과:")
        logger.info(f"  • 기본 대비 개선: +{summary['improvement_over_basic']:.2%}")
        logger.info(f"  • 향상 대비 개선: +{summary['improvement_over_enhanced']:.2%}")
        
        # 시각화 생성
        self.create_visualizations(summary)
    
    def create_visualizations(self, summary: Dict[str, Any]):
        """성능 시각화 생성"""
        df = summary["results_dataframe"]
        
        # 그래프 스타일 설정
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 정확도 비교
        accuracy_cols = ["basic_accuracy", "enhanced_accuracy", "dl_accuracy"]
        df[accuracy_cols].plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title("OCR 방식별 정확도 비교")
        axes[0, 0].set_ylabel("정확도")
        axes[0, 0].set_xlabel("테스트 이미지")
        axes[0, 0].legend(["기본", "향상", "딥러닝"])
        
        # 2. 처리 시간 비교
        time_cols = ["basic_time", "enhanced_time", "dl_time"]
        df[time_cols].plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title("OCR 방식별 처리 시간")
        axes[0, 1].set_ylabel("시간 (초)")
        axes[0, 1].set_xlabel("테스트 이미지")
        axes[0, 1].legend(["기본", "향상", "딥러닝"])
        
        # 3. 정확도 분포
        axes[1, 0].hist(df["dl_accuracy"], bins=20, edgecolor='black')
        axes[1, 0].axvline(x=0.99, color='r', linestyle='--', label='목표 (99%)')
        axes[1, 0].set_title("딥러닝 OCR 정확도 분포")
        axes[1, 0].set_xlabel("정확도")
        axes[1, 0].set_ylabel("빈도")
        axes[1, 0].legend()
        
        # 4. 정확도 vs 처리 시간
        axes[1, 1].scatter(df["dl_time"], df["dl_accuracy"])
        axes[1, 1].set_title("정확도 vs 처리 시간")
        axes[1, 1].set_xlabel("처리 시간 (초)")
        axes[1, 1].set_ylabel("정확도")
        
        plt.tight_layout()
        
        # 그래프 저장
        output_path = "./reports/benchmark_visualization.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        logger.info(f"시각화 저장 완료: {output_path}")
        
        # CSV로 결과 저장
        csv_path = "./reports/benchmark_results.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"결과 CSV 저장 완료: {csv_path}")


class TestKoreanOCR(unittest.TestCase):
    """한국어 OCR 유닛 테스트"""
    
    @classmethod
    def setUpClass(cls):
        """테스트 시작 전 설정"""
        cls.ocr_engine = UltraKoreanOCR(OCRConfig())
        cls.deep_enhancer = DeepLearningOCREnhancer()
        cls.layout_analyzer = SchoolRecordLayoutAnalyzer()
    
    def test_korean_text_recognition(self):
        """한국어 텍스트 인식 테스트"""
        # 테스트 이미지 생성
        test_text = "안녕하세요 한국어 OCR 테스트입니다"
        image = self._create_test_image(test_text)
        
        # OCR 수행
        result = self.ocr_engine.process_image("./temp_test.jpg")
        
        # 결과 확인
        self.assertIsNotNone(result)
        self.assertIn("full_text", result)
        
        # 정확도 확인
        accuracy = self._calculate_accuracy(result["full_text"], test_text)
        self.assertGreater(accuracy, 0.8)  # 80% 이상 정확도
    
    def test_image_preprocessing(self):
        """이미지 전처리 테스트"""
        # 노이즈가 있는 이미지 생성
        image = np.random.randint(0, 255, (100, 300, 3), dtype=np.uint8)
        
        # 전처리 수행
        preprocessor = self.ocr_engine.preprocessor
        enhanced = preprocessor.enhance_image(image)
        
        # 결과 확인
        self.assertIsNotNone(enhanced)
        self.assertEqual(enhanced.shape[:2], image.shape[:2])
    
    def test_layout_analysis(self):
        """레이아웃 분석 테스트"""
        # 테스트 이미지 생성 (테이블 포함)
        image = np.ones((500, 500, 3), dtype=np.uint8) * 255
        cv2.rectangle(image, (50, 50), (450, 450), (0, 0, 0), 2)
        cv2.line(image, (50, 150), (450, 150), (0, 0, 0), 2)
        cv2.line(image, (200, 50), (200, 450), (0, 0, 0), 2)
        
        # 레이아웃 분석
        result = self.layout_analyzer.analyze_layout(image)
        
        # 결과 확인
        self.assertIsNotNone(result)
        self.assertIn("layout_elements", result)
        self.assertIn("statistics", result)
    
    def test_field_extraction(self):
        """필드 추출 테스트"""
        # 테스트 텍스트
        test_text = "학년: 2학년 반: 3반 번호: 15번 이름: 홍길동"
        
        # OCR 결과 시뮬레이션
        ocr_result = {"full_text": test_text}
        
        # 필드 추출
        fields = self.ocr_engine.extract_school_record_fields(ocr_result)
        
        # 결과 확인
        self.assertEqual(fields["학년"], "2")
        self.assertEqual(fields["반"], "3")
        self.assertEqual(fields["번호"], "15")
        self.assertEqual(fields["이름"], "홍길동")
    
    def test_spell_checking(self):
        """맞춤법 검사 테스트"""
        # 오류가 있는 텍스트
        text_with_errors = "학생은 성싣하고 책임감이 강함니다"
        
        # 맞춤법 검사
        checker = self.deep_enhancer.spell_checker
        corrected, corrections = checker.check_and_correct(text_with_errors)
        
        # 결과 확인
        self.assertNotEqual(corrected, text_with_errors)
        self.assertGreater(len(corrections), 0)
    
    def test_performance(self):
        """성능 테스트"""
        # 테스트 이미지 생성
        image = self._create_test_image("성능 테스트용 한글 텍스트입니다")
        
        # 시간 측정
        start_time = time.time()
        result = self.ocr_engine.process_image("./temp_test.jpg")
        elapsed_time = time.time() - start_time
        
        # 성능 확인
        self.assertLess(elapsed_time, 5.0)  # 5초 이내 처리
        self.assertGreater(result.get("average_confidence", 0), 0.8)  # 80% 이상 신뢰도
    
    def _create_test_image(self, text: str) -> np.ndarray:
        """테스트용 이미지 생성"""
        image = np.ones((100, 600, 3), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, text, (10, 50), font, 0.7, (0, 0, 0), 2)
        cv2.imwrite("./temp_test.jpg", image)
        return image
    
    def _calculate_accuracy(self, predicted: str, ground_truth: str) -> float:
        """정확도 계산"""
        if not ground_truth:
            return 0.0
        
        correct = sum(1 for p, g in zip(predicted, ground_truth) if p == g)
        return correct / len(ground_truth)
    
    @classmethod
    def tearDownClass(cls):
        """테스트 종료 후 정리"""
        # 임시 파일 삭제
        if os.path.exists("./temp_test.jpg"):
            os.remove("./temp_test.jpg")


if __name__ == "__main__":
    # 벤치마크 실행
    logger.info("Starting Korean OCR Benchmark System...")
    
    benchmark = OCRBenchmark()
    results = benchmark.run_comprehensive_benchmark("./test_data")
    
    # 유닛 테스트 실행
    logger.info("\nRunning unit tests...")
    unittest.main(argv=[''], exit=False)