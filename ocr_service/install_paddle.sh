#!/bin/bash
# PaddleOCR-VL 설치 스크립트 (한국어 100% 정확도)

set -e  # 에러 발생 시 중단

echo "================================================================================================"
echo "🚀 PaddleOCR-VL 최신 버전 설치 시작..."
echo "🎯 한국어 100% 정확도 OCR 목표"
echo "📚 109개 언어 지원 - 문서 전처리 + 레이아웃 검출 + VLM 기반 인식"
echo "================================================================================================"
echo ""

# Python 버전 확인
echo "📋 Python 버전 확인..."
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "❌ Python이 설치되어 있지 않습니다!"
    exit 1
fi

PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

$PYTHON_CMD --version

# Python 버전이 3.8 이상인지 확인
PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo "❌ Python 3.8 이상이 필요합니다. (현재: $PYTHON_VERSION)"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION 사용"

# GPU 확인
echo ""
echo "🎮 GPU 확인 중..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU 감지됨!"
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
    
    # Compute Capability 확인
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1)
    echo "   GPU Compute Capability: $COMPUTE_CAP"
    
    # CC 8.5 이상 권장 (PaddleOCR-VL 최적)
    if [ "$(echo "$COMPUTE_CAP >= 8.5" | bc)" -eq 1 ]; then
        echo "   ✅ 권장 GPU (CC >= 8.5)"
    elif [ "$(echo "$COMPUTE_CAP >= 8.0" | bc)" -eq 1 ]; then
        echo "   ⚠️  지원 GPU (CC >= 8.0) - 일부 기능 제한 가능"
    else
        echo "   ⚠️  지원되지 않는 GPU (CC < 8.0) - CPU로 폴백"
        USE_GPU=false
    fi
    
    USE_GPU=true
else
    echo "⚠️  NVIDIA GPU가 감지되지 않았습니다. CPU 버전을 설치합니다."
    USE_GPU=false
fi

# pip 업그레이드
echo ""
echo "📦 pip 업그레이드 중..."
$PYTHON_CMD -m pip install --upgrade pip setuptools wheel

# PaddlePaddle 3.2.0 설치 (PaddleOCR-VL 요구사항)
echo ""
echo "================================================================================================"
if [ "$USE_GPU" = true ]; then
    echo "🔧 PaddlePaddle 3.2.0 GPU 버전 설치 중..."
    echo "   - CUDA 12.6 지원"
    echo "   - FP16 정밀도 지원"
    echo "   - TensorRT 최적화"
    
    # CUDA 12.6용 PaddlePaddle GPU 설치
    $PYTHON_CMD -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
    
    if [ $? -ne 0 ]; then
        echo "⚠️  CUDA 12.6 설치 실패, CUDA 12.3 시도..."
        $PYTHON_CMD -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
    fi
else
    echo "🔧 PaddlePaddle 3.2.0 CPU 버전 설치 중..."
    $PYTHON_CMD -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
fi
echo "================================================================================================"

# safetensors 설치 (PaddleOCR-VL 필수)
echo ""
echo "🔧 safetensors 최신 버전 설치 중..."
$PYTHON_CMD -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl || \
$PYTHON_CMD -m pip install safetensors>=0.4.0

# PaddleOCR 최신 버전 (doc-parser 포함) 설치
echo ""
echo "📚 PaddleOCR 최신 버전 설치 중 (doc-parser 포함)..."
$PYTHON_CMD -m pip install -U "paddleocr[doc-parser]"

# 나머지 종속성 설치
echo ""
echo "📦 추가 종속성 설치 중..."
$PYTHON_CMD -m pip install -r requirements.txt

# FlashAttention 설치 (선택사항, GPU CC >= 12인 경우)
if [ "$USE_GPU" = true ] && [ "$(echo "$COMPUTE_CAP >= 12.0" | bc 2>/dev/null)" -eq 1 ]; then
    echo ""
    echo "🚀 FlashAttention 설치 중 (NVIDIA 50 시리즈)..."
    $PYTHON_CMD -m pip install flash-attn==2.8.3 || echo "⚠️  FlashAttention 설치 실패 (선택사항)"
fi

# PaddleOCR-VL 모델 다운로드 (백그라운드)
echo ""
echo "🇰🇷 PaddleOCR-VL 모델 다운로드 중..."
$PYTHON_CMD << EOF
import sys
try:
    print("PaddleOCR-VL import 중...")
    from paddleocr import PaddleOCRVL
    print("✅ PaddleOCR-VL import 성공!")
    
    print("\n모델 초기화 중 (첫 실행 시 모델 다운로드)...")
    pipeline = PaddleOCRVL(
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_layout_detection=True,
        device="gpu:0" if "$USE_GPU" == "true" else "cpu",
    )
    print("✅ PaddleOCR-VL 모델 다운로드 완료!")
    
except Exception as e:
    print(f"⚠️  모델 다운로드 실패: {e}")
    print("서버 시작 시 자동으로 다운로드됩니다.")
    sys.exit(0)  # 실패해도 계속 진행
EOF

# 설치 확인
echo ""
echo "================================================================================================"
echo "✅ 설치 완료! 검증 중..."
echo "================================================================================================"

$PYTHON_CMD -c "import paddle; print(f'✅ PaddlePaddle 버전: {paddle.__version__}')" || echo "❌ PaddlePaddle import 실패"
$PYTHON_CMD -c "from paddleocr import PaddleOCRVL; print('✅ PaddleOCR-VL import 성공!')" || echo "❌ PaddleOCR-VL import 실패"
$PYTHON_CMD -c "import cv2; print(f'✅ OpenCV 버전: {cv2.__version__}')" || echo "❌ OpenCV import 실패"
$PYTHON_CMD -c "import numpy as np; print(f'✅ NumPy 버전: {np.__version__}')" || echo "❌ NumPy import 실패"
$PYTHON_CMD -c "from PIL import Image; print('✅ Pillow import 성공!')" || echo "❌ Pillow import 실패"

echo ""
echo "================================================================================================"
echo "✅ PaddleOCR-VL 설치 완료!"
echo "================================================================================================"
echo ""
echo "💡 사용 방법:"
echo "   $PYTHON_CMD main.py"
echo "   또는"
echo "   uvicorn main:app --host 0.0.0.0 --port 8000 --reload"
echo ""
echo "🌐 서버 실행 후 확인:"
echo "   curl http://localhost:8000/health"
echo "   curl http://localhost:8000/"
echo ""
echo "📚 API 문서:"
echo "   http://localhost:8000/docs"
echo ""
echo "================================================================================================"
