#!/usr/bin/env python3
"""
초월적 한국어 OCR 시스템 메인 실행 파일
Ultimate Korean OCR System Main Execution
"""

import argparse
import asyncio
import sys
import os
from pathlib import Path
import json
import time
import logging
from typing import Optional, List

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ultimate_korean_ocr import get_ultimate_ocr

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_single_image(image_path: str, output_format: str = 'json') -> dict:
    """단일 이미지 처리"""
    logger.info(f"Processing: {image_path}")
    
    ocr = get_ultimate_ocr()
    result = ocr.process_image(image_path)
    
    if output_format == 'json':
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"\n{'='*60}")
        print("OCR 결과")
        print('='*60)
        print(f"텍스트: {result['text']}")
        print(f"신뢰도: {result['confidence']:.2%}")
        print(f"처리 시간: {result['processing_time']:.3f}초")
        
        if result.get('language_model_applied'):
            print("✅ 언어 모델 교정 적용됨")
        
        if result.get('details'):
            print(f"\n상세 정보:")
            print(f"- 사용된 엔진 수: {result.get('engine_count', 0)}")
            print(f"- 총 감지 수: {result.get('total_detections', 0)}")
    
    return result


def process_directory(directory_path: str, pattern: str = "*.jpg") -> List[dict]:
    """디렉토리 내 모든 이미지 처리"""
    dir_path = Path(directory_path)
    
    if not dir_path.exists():
        logger.error(f"Directory not found: {directory_path}")
        return []
    
    image_files = list(dir_path.glob(pattern))
    
    if not image_files:
        logger.warning(f"No images found with pattern: {pattern}")
        return []
    
    logger.info(f"Found {len(image_files)} images to process")
    
    results = []
    total_start = time.time()
    
    for image_file in image_files:
        try:
            result = process_single_image(str(image_file), output_format='text')
            results.append({
                'file': str(image_file),
                'result': result
            })
        except Exception as e:
            logger.error(f"Error processing {image_file}: {e}")
            results.append({
                'file': str(image_file),
                'error': str(e)
            })
    
    total_time = time.time() - total_start
    
    # 요약 통계
    successful = [r for r in results if 'result' in r]
    avg_confidence = sum(r['result']['confidence'] for r in successful) / len(successful) if successful else 0
    
    print(f"\n{'='*60}")
    print("처리 완료 요약")
    print('='*60)
    print(f"총 파일: {len(image_files)}")
    print(f"성공: {len(successful)}")
    print(f"실패: {len(results) - len(successful)}")
    print(f"평균 신뢰도: {avg_confidence:.2%}")
    print(f"총 처리 시간: {total_time:.2f}초")
    print(f"평균 시간/이미지: {total_time/len(image_files):.2f}초")
    
    return results


def run_benchmark():
    """벤치마크 실행"""
    from scripts.benchmark_ultimate import OCRBenchmark
    
    logger.info("Running benchmark...")
    benchmark = OCRBenchmark()
    results = benchmark.run_benchmark(save_results=True)
    
    return results


def run_tests():
    """테스트 실행"""
    from tests.test_ultimate_ocr import run_all_tests
    
    logger.info("Running tests...")
    success = run_all_tests()
    
    return success


def start_api_server(host: str = "0.0.0.0", port: int = 8000):
    """API 서버 시작"""
    import uvicorn
    from api.ultimate_server import app
    
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


def show_metrics():
    """현재 메트릭 표시"""
    ocr = get_ultimate_ocr()
    metrics = ocr.get_metrics()
    
    print(f"\n{'='*60}")
    print("현재 OCR 성능 메트릭")
    print('='*60)
    print(f"총 처리 이미지: {metrics['total_processed']}")
    print(f"완벽한 인식 (≥99.9%): {metrics['perfect_recognitions']}")
    print(f"정확도: {metrics['accuracy_rate']:.2f}%")
    print(f"평균 신뢰도: {metrics['average_confidence']:.2%}")
    print(f"평균 처리 시간: {metrics['average_processing_time']:.3f}초")
    print(f"최소 처리 시간: {metrics['min_processing_time']:.3f}초")
    print(f"최대 처리 시간: {metrics['max_processing_time']:.3f}초")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="초월적 한국어 OCR 시스템 - 100% 정확도 목표",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 단일 이미지 처리
  python main_ultimate.py --image path/to/image.jpg
  
  # 디렉토리 처리
  python main_ultimate.py --directory path/to/images --pattern "*.png"
  
  # API 서버 시작
  python main_ultimate.py --server --port 8000
  
  # 벤치마크 실행
  python main_ultimate.py --benchmark
  
  # 테스트 실행
  python main_ultimate.py --test
        """
    )
    
    parser.add_argument('--image', '-i', type=str, help='단일 이미지 파일 경로')
    parser.add_argument('--directory', '-d', type=str, help='이미지 디렉토리 경로')
    parser.add_argument('--pattern', '-p', type=str, default="*.jpg", 
                       help='파일 패턴 (기본값: *.jpg)')
    parser.add_argument('--output', '-o', type=str, choices=['json', 'text'], 
                       default='text', help='출력 형식')
    parser.add_argument('--server', '-s', action='store_true', 
                       help='API 서버 시작')
    parser.add_argument('--host', type=str, default='0.0.0.0', 
                       help='서버 호스트 (기본값: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, 
                       help='서버 포트 (기본값: 8000)')
    parser.add_argument('--benchmark', '-b', action='store_true', 
                       help='벤치마크 실행')
    parser.add_argument('--test', '-t', action='store_true', 
                       help='테스트 실행')
    parser.add_argument('--metrics', '-m', action='store_true', 
                       help='현재 메트릭 표시')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='상세 출력')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    print("""
╔══════════════════════════════════════════════════════════╗
║     초월적 한국어 OCR 시스템 - Ultimate Korean OCR      ║
║            100% 정확도 목표 - 생기부 특화              ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    try:
        if args.image:
            # 단일 이미지 처리
            process_single_image(args.image, args.output)
            
        elif args.directory:
            # 디렉토리 처리
            process_directory(args.directory, args.pattern)
            
        elif args.server:
            # API 서버 시작
            start_api_server(args.host, args.port)
            
        elif args.benchmark:
            # 벤치마크 실행
            run_benchmark()
            
        elif args.test:
            # 테스트 실행
            success = run_tests()
            sys.exit(0 if success else 1)
            
        elif args.metrics:
            # 메트릭 표시
            show_metrics()
            
        else:
            # 인터랙티브 모드
            print("\n사용 가능한 옵션:")
            print("1. 이미지 처리")
            print("2. API 서버 시작")
            print("3. 벤치마크 실행")
            print("4. 테스트 실행")
            print("5. 메트릭 보기")
            print("0. 종료")
            
            while True:
                choice = input("\n선택 (0-5): ").strip()
                
                if choice == '1':
                    image_path = input("이미지 경로: ").strip()
                    if Path(image_path).exists():
                        process_single_image(image_path, 'text')
                    else:
                        print("파일을 찾을 수 없습니다.")
                        
                elif choice == '2':
                    start_api_server()
                    
                elif choice == '3':
                    run_benchmark()
                    
                elif choice == '4':
                    run_tests()
                    
                elif choice == '5':
                    show_metrics()
                    
                elif choice == '0':
                    print("프로그램을 종료합니다.")
                    break
                    
                else:
                    print("올바른 옵션을 선택해주세요.")
    
    except KeyboardInterrupt:
        print("\n\n프로그램이 중단되었습니다.")
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
