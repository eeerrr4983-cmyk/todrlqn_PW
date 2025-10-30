#!/usr/bin/env python3
"""
Ultra Korean OCR Benchmark Script
실제 성능 측정 및 리포트 생성
"""

import asyncio
import sys
import os
import time
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ultra_precision_ocr import get_ultra_ocr, process_image_ultra


async def run_comprehensive_benchmark():
    """종합 벤치마크 실행"""
    print("🚀 Ultra Korean OCR Benchmark Starting...")
    print("="*60)
    
    # 테스트 이미지 준비
    test_images = [
        "data/test/easy_text.jpg",
        "data/test/medium_text.jpg", 
        "data/test/hard_text.jpg",
        "data/test/handwritten.jpg",
        "data/test/low_quality.jpg",
        "data/test/school_record.jpg"
    ]
    
    results = []
    ocr = get_ultra_ocr()
    
    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"⚠️  Skipping {image_path} (not found)")
            continue
        
        print(f"\n📄 Processing: {Path(image_path).name}")
        print("-" * 40)
        
        # 3회 반복 측정
        times = []
        confidences = []
        
        for i in range(3):
            start = time.time()
            result = await process_image_ultra(image_path)
            elapsed = time.time() - start
            
            times.append(elapsed)
            confidences.append(result['confidence'])
            
            print(f"  Run {i+1}: {elapsed:.2f}s, Confidence: {result['confidence']:.3%}")
        
        avg_time = np.mean(times)
        avg_confidence = np.mean(confidences)
        
        results.append({
            'image': Path(image_path).name,
            'avg_time': avg_time,
            'avg_confidence': avg_confidence,
            'is_perfect': avg_confidence >= 0.99,
            'text_length': len(result.get('text', ''))
        })
    
    # 결과 요약
    print("\n" + "="*60)
    print("📊 BENCHMARK RESULTS")
    print("="*60)
    
    total_images = len(results)
    perfect_count = sum(1 for r in results if r['is_perfect'])
    avg_overall_confidence = np.mean([r['avg_confidence'] for r in results])
    avg_overall_time = np.mean([r['avg_time'] for r in results])
    
    print(f"\n✅ Total Images Processed: {total_images}")
    print(f"🎯 Perfect Recognition (≥99%): {perfect_count}/{total_images} ({perfect_count/total_images*100:.1f}%)")
    print(f"📈 Average Confidence: {avg_overall_confidence:.3%}")
    print(f"⏱️  Average Processing Time: {avg_overall_time:.2f}s")
    
    # 개별 결과
    print("\n📋 Detailed Results:")
    print("-" * 60)
    for r in results:
        status = "✅" if r['is_perfect'] else "⚠️"
        print(f"{status} {r['image']:20s} | Conf: {r['avg_confidence']:.3%} | Time: {r['avg_time']:.2f}s")
    
    # 성능 리포트 저장
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'summary': {
            'total_images': total_images,
            'perfect_recognition_rate': perfect_count / total_images,
            'average_confidence': avg_overall_confidence,
            'average_time': avg_overall_time
        },
        'details': results,
        'performance_grade': get_performance_grade(avg_overall_confidence)
    }
    
    # JSON 저장
    report_path = Path('reports') / f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Report saved to: {report_path}")
    
    # 그래프 생성
    create_visualization(results)
    
    # 최종 평가
    print("\n" + "="*60)
    print(f"🏆 PERFORMANCE GRADE: {report['performance_grade']}")
    print("="*60)
    
    return report


def get_performance_grade(confidence):
    """성능 등급 판정"""
    if confidence >= 0.99:
        return "S+ (Perfect)"
    elif confidence >= 0.97:
        return "S (Excellent)"
    elif confidence >= 0.95:
        return "A+ (Very Good)"
    elif confidence >= 0.93:
        return "A (Good)"
    elif confidence >= 0.90:
        return "B+ (Above Average)"
    else:
        return "B (Average)"


def create_visualization(results):
    """결과 시각화"""
    if not results:
        return
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Confidence Chart
    images = [r['image'].replace('.jpg', '') for r in results]
    confidences = [r['avg_confidence'] * 100 for r in results]
    
    ax1 = axes[0]
    bars = ax1.bar(images, confidences, color=['green' if c >= 99 else 'orange' for c in confidences])
    ax1.axhline(y=99, color='r', linestyle='--', alpha=0.7, label='Target (99%)')
    ax1.set_title('OCR Confidence by Image', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Confidence (%)')
    ax1.set_ylim([90, 100])
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # Processing Time Chart
    times = [r['avg_time'] for r in results]
    
    ax2 = axes[1]
    ax2.bar(images, times, color='steelblue')
    ax2.axhline(y=2.0, color='r', linestyle='--', alpha=0.7, label='Target (2s)')
    ax2.set_title('Processing Time by Image', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Time (seconds)')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = Path('reports') / f"benchmark_chart_{time.strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"📊 Chart saved to: {fig_path}")
    
    plt.close()


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║        Ultra Korean OCR Benchmark Tool v2.0                 ║
║        초월적 정밀도 한국어 OCR 성능 측정기                   ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Run benchmark
    loop = asyncio.get_event_loop()
    report = loop.run_until_complete(run_comprehensive_benchmark())
    
    # Check if target achieved
    if report['summary']['average_confidence'] >= 0.99:
        print("\n🎊 CONGRATULATIONS! Target accuracy of 99% achieved! 🎊")
    else:
        print(f"\n📈 Current accuracy: {report['summary']['average_confidence']:.3%}")
        print(f"   Target accuracy: 99.0%")
        print(f"   Gap: {0.99 - report['summary']['average_confidence']:.3%}")
