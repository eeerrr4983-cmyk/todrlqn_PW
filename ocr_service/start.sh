#!/bin/bash
# PaddleOCR 백엔드 서버 시작 스크립트

set -e

cd "$(dirname "$0")"

echo "🚀 PaddleOCR 백엔드 서버 시작 중..."
echo ""

# Python 확인
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "❌ Python이 설치되어 있지 않습니다!"
    exit 1
fi

# Python 실행 파일 결정
PYTHON_CMD="python"
if ! command -v python &> /dev/null; then
    PYTHON_CMD="python3"
fi

echo "✅ Python: $($PYTHON_CMD --version)"

# PaddleOCR 설치 확인
echo "📦 PaddleOCR 설치 확인 중..."
if ! $PYTHON_CMD -c "from paddleocr import PaddleOCR" 2>/dev/null; then
    echo "⚠️ PaddleOCR이 설치되어 있지 않습니다."
    echo "❓ 지금 설치하시겠습니까? (y/n)"
    read -r response
    if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
        echo "📥 PaddleOCR 설치 중..."
        ./install_paddle.sh
    else
        echo "❌ PaddleOCR 설치가 필요합니다. ./install_paddle.sh를 실행하세요."
        exit 1
    fi
fi

echo "✅ PaddleOCR 설치 확인 완료"
echo ""

# 포트 설정
PORT="${OCR_PORT:-8000}"

# 포트 사용 확인
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️ 포트 $PORT이 이미 사용 중입니다."
    echo "❓ 기존 프로세스를 종료하시겠습니까? (y/n)"
    read -r response
    if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
        echo "🛑 기존 프로세스 종료 중..."
        lsof -ti:$PORT | xargs kill -9 2>/dev/null || true
        sleep 1
    else
        echo "❌ 포트 $PORT을 사용할 수 없습니다."
        exit 1
    fi
fi

# 서버 시작
echo "🚀 서버 시작: http://localhost:$PORT"
echo ""
echo "💡 서버를 중지하려면 Ctrl+C를 누르세요."
echo ""
echo "=========================================="
echo ""

exec $PYTHON_CMD main.py
