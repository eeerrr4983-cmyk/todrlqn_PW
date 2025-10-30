#!/usr/bin/env python3
"""
초월적 한국어 OCR 벤치마크 스크립트
Ultimate Korean OCR Benchmark Script
"""

import sys
import os
import time
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import argparse
import logging

# 상위 디렉토리 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ultimate_korean_ocr import get_ultimate_ocr

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCRBenchmark:
    """OCR 벤치마크 클래스"""
    
    def __init__(self):
        self.ocr = get_ultimate_ocr()
        self.results = []
        self.test_images = []
        
    def create_test_images(self) -> List[Tuple[str, np.ndarray, str]]:
        """다양한 테스트 이미지 생성"""
        test_cases = []
        
        # 1. 선명한 텍스트
        img1 = np.ones((200, 600, 3), dtype=np.uint8) * 255
        text1 = "2024학년도 1학기 학생 평가"
        cv2.putText(img1, "2024 1st Semester Student", (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        test_cases.append(("clear_text", img1, text1))
        
        # 2. 작은 글씨
        img2 = np.ones((150, 400, 3), dtype=np.uint8) * 255
        text2 = "작은 글씨 테스트"
        cv2.putText(img2, "Small Text Test", (50, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        test_cases.append(("small_text", img2, text2))
        
        # 3. 노이즈가 있는 이미지
        img3 = np.ones((200, 500, 3), dtype=np.uint8) * 255
        text3 = "노이즈 테스트"
        cv2.putText(img3, "Noise Test", (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        noise = np.random.normal(0, 25, img3.shape)
        img3 = np.clip(img3 + noise, 0, 255).astype(np.uint8)
        test_cases.append(("noisy_text", img3, text3))
        
        # 4. 블러 처리된 이미지
        img4 = np.ones((200, 500, 3), dtype=np.uint8) * 255
        text4 = "블러 테스트"
        cv2.putText(img4, "Blur Test", (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        img4 = cv2.GaussianBlur(img4, (7, 7), 3)
        test_cases.append(("blurred_text", img4, text4))
        
        # 5. 낮은 대비
        img5 = np.ones((200, 500, 3), dtype=np.uint8) * 200
        text5 = "낮은 대비"
        cv2.putText(img5, "Low Contrast", (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 180, 180), 2)
        test_cases.append(("low_contrast", img5, text5))
        
        # 6. 회전된 텍스트
        img6 = np.ones((300, 300, 3), dtype=np.uint8) * 255
        text6 = "회전 텍스트"
        cv2.putText(img6, "Rotated", (100, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        center = (150, 150)
        M = cv2.getRotationMatrix2D(center, 15, 1.0)
        img6 = cv2.warpAffine(img6, M, (300, 300))
        test_cases.append(("rotated_text", img6, text6))
        
        # 7. 다중 라인
        img7 = np.ones((400, 600, 3), dtype=np.uint8) * 255
        lines = ["Line 1: Student Name", "Line 2: Grade Info", "Line 3: Subject Score"]
        text7 = " ".join(lines)
        y_pos = 100
        for line in lines:
            cv2.putText(img7, line, (50, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            y_pos += 80
        test_cases.append(("multi_line", img7, text7))
        
        # 8. 복잡한 배경
        img8 = np.random.randint(100, 200, (200, 500, 3), dtype=np.uint8)
        text8 = "복잡한 배경"
        cv2.putText(img8, "Complex BG", (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        test_cases.append(("complex_bg", img8, text8))
        
        # 9. 매우 큰 글씨
        img9 = np.ones((300, 800, 3), dtype=np.uint8) * 255
        text9 = "큰 글씨"
        cv2.putText(img9, "LARGE", (50, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 5)
        test_cases.append(("large_text", img9, text9))
        
        # 10. 왜곡된 이미지
        img10 = np.ones((200, 500, 3), dtype=np.uint8) * 255
        text10 = "왜곡 테스트"
        cv2.putText(img10, "Distorted", (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        # 원근 변환
        pts1 = np.float32([[0, 0], [500, 0], [0, 200], [500, 200]])
        pts2 = np.float32([[10, 10], [490, 20], [0, 200], [500, 190]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img10 = cv2.warpPerspective(img10, M, (500, 200))
        test_cases.append(("distorted_text", img10, text10))
        
        return test_cases
    
    def run_benchmark(self, save_results: bool = True) -> Dict[str, Any]:
        """벤치마크 실행"""
        logger.info("벤치마크 시작...")
        
        # 테스트 이미지 생성
        test_cases = self.create_test_images()
        
        results = []
        total_time = 0
        
        # 각 테스트 케이스 처리
        for test_name, image, expected_text in tqdm(test_cases, desc="Processing"):
            # 임시 파일로 저장
            temp_path = f"/tmp/benchmark_{test_name}.jpg"
            cv2.imwrite(temp_path, image)
            
            # OCR 처리
            start_time = time.time()
            try:
                result = self.ocr.process_image(temp_path)
                processing_time = time.time() - start_time
                
                results.append({
                    'test_name': test_name,
                    'expected_text': expected_text,
                    'recognized_text': result.get('text', ''),
                    'confidence': result.get('confidence', 0),
                    'processing_time': processing_time,
                    'success': True,
                    'details': result.get('details', [])
                })
                
                total_time += processing_time
                
            except Exception as e:
                logger.error(f"Error processing {test_name}: {e}")
                results.append({
                    'test_name': test_name,
                    'expected_text': expected_text,
                    'recognized_text': '',
                    'confidence': 0,
                    'processing_time': time.time() - start_time,
                    'success': False,
                    'error': str(e)
                })
            
            # 임시 파일 삭제
            Path(temp_path).unlink(missing_ok=True)
        
        # 메트릭 계산
        successful_results = [r for r in results if r['success']]
        
        metrics = {
            'total_tests': len(results),
            'successful': len(successful_results),
            'failed': len(results) - len(successful_results),
            'average_confidence': np.mean([r['confidence'] for r in successful_results]) if successful_results else 0,
            'min_confidence': min([r['confidence'] for r in successful_results]) if successful_results else 0,
            'max_confidence': max([r['confidence'] for r in successful_results]) if successful_results else 0,
            'average_time': np.mean([r['processing_time'] for r in results]),
            'total_time': total_time,
            'perfect_recognitions': len([r for r in successful_results if r['confidence'] >= 0.99]),
            'high_confidence': len([r for r in successful_results if r['confidence'] >= 0.90])
        }
        
        # OCR 엔진 메트릭 추가
        ocr_metrics = self.ocr.get_metrics()
        metrics['ocr_metrics'] = ocr_metrics
        
        # 결과 저장
        if save_results:
            self.save_results(results, metrics)
        
        # 결과 출력
        self.print_results(results, metrics)
        
        # 그래프 생성
        self.create_visualizations(results, metrics)
        
        return {
            'results': results,
            'metrics': metrics
        }
    
    def save_results(self, results: List[Dict], metrics: Dict):
        """결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON 파일로 저장
        output_dir = Path("./benchmark_results")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"benchmark_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': timestamp,
                'results': results,
                'metrics': metrics
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        
        # CSV로도 저장
        df = pd.DataFrame(results)
        csv_file = output_dir / f"benchmark_{timestamp}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        logger.info(f"CSV saved to {csv_file}")
    
    def print_results(self, results: List[Dict], metrics: Dict):
        """결과 출력"""
        print("\n" + "="*60)
        print("벤치마크 결과 요약")
        print("="*60)
        
        print(f"\n총 테스트: {metrics['total_tests']}")
        print(f"성공: {metrics['successful']} ({metrics['successful']/metrics['total_tests']*100:.1f}%)")
        print(f"실패: {metrics['failed']}")
        
        print(f"\n평균 신뢰도: {metrics['average_confidence']:.2%}")
        print(f"최소 신뢰도: {metrics['min_confidence']:.2%}")
        print(f"최대 신뢰도: {metrics['max_confidence']:.2%}")
        
        print(f"\n완벽한 인식 (≥99%): {metrics['perfect_recognitions']}")
        print(f"높은 신뢰도 (≥90%): {metrics['high_confidence']}")
        
        print(f"\n평균 처리 시간: {metrics['average_time']:.3f}초")
        print(f"총 처리 시간: {metrics['total_time']:.3f}초")
        
        print("\n개별 테스트 결과:")
        print("-"*60)
        
        for result in results:
            status = "✅" if result['success'] else "❌"
            confidence = result['confidence'] * 100
            time_ms = result['processing_time'] * 1000
            
            print(f"{status} {result['test_name']:15} | "
                  f"신뢰도: {confidence:5.1f}% | "
                  f"시간: {time_ms:6.1f}ms")
            
            if result['recognized_text']:
                print(f"   인식: {result['recognized_text'][:50]}")
    
    def create_visualizations(self, results: List[Dict], metrics: Dict):
        """시각화 생성"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # GUI 없이 그래프 생성
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. 신뢰도 분포
            confidences = [r['confidence'] for r in results if r['success']]
            axes[0, 0].hist(confidences, bins=20, edgecolor='black', alpha=0.7)
            axes[0, 0].set_xlabel('Confidence')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].set_title('Confidence Distribution')
            axes[0, 0].axvline(0.99, color='r', linestyle='--', label='99% threshold')
            axes[0, 0].legend()
            
            # 2. 처리 시간 분포
            times = [r['processing_time'] for r in results]
            axes[0, 1].bar(range(len(times)), times, color='skyblue')
            axes[0, 1].set_xlabel('Test Case')
            axes[0, 1].set_ylabel('Processing Time (s)')
            axes[0, 1].set_title('Processing Time per Test')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 3. 테스트별 신뢰도
            test_names = [r['test_name'] for r in results]
            test_confidences = [r['confidence'] for r in results]
            
            axes[1, 0].barh(test_names, test_confidences, color='green')
            axes[1, 0].set_xlabel('Confidence')
            axes[1, 0].set_title('Confidence by Test Type')
            axes[1, 0].axvline(0.99, color='r', linestyle='--', alpha=0.5)
            
            # 4. 성공/실패 파이 차트
            success_counts = [metrics['successful'], metrics['failed']]
            labels = ['Success', 'Failed']
            colors = ['#90EE90', '#FFB6C1']
            
            axes[1, 1].pie(success_counts, labels=labels, colors=colors,
                          autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('Success Rate')
            
            plt.tight_layout()
            
            # 그래프 저장
            output_dir = Path("./benchmark_results")
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = output_dir / f"benchmark_plot_{timestamp}.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualizations saved to {plot_file}")
            
        except ImportError:
            logger.warning("Matplotlib not available. Skipping visualizations.")
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Ultimate Korean OCR Benchmark")
    parser.add_argument('--save', action='store_true', default=True,
                       help='Save benchmark results')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    print("="*60)
    print("초월적 한국어 OCR 벤치마크")
    print("Ultimate Korean OCR Benchmark")
    print("="*60)
    
    benchmark = OCRBenchmark()
    results = benchmark.run_benchmark(save_results=args.save)
    
    print("\n벤치마크 완료!")
    
    # 최종 점수 계산
    score = (
        results['metrics']['average_confidence'] * 50 +  # 신뢰도 50%
        (results['metrics']['successful'] / results['metrics']['total_tests']) * 30 +  # 성공률 30%
        (1 - min(results['metrics']['average_time'] / 2, 1)) * 20  # 속도 20%
    )
    
    print(f"\n최종 점수: {score:.1f}/100")
    
    if score >= 95:
        print("🏆 완벽한 성능!")
    elif score >= 90:
        print("⭐ 우수한 성능!")
    elif score >= 80:
        print("✅ 좋은 성능")
    else:
        print("⚠️ 개선 필요")
    
    return results


if __name__ == "__main__":
    main()
