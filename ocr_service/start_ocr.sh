#!/bin/bash

# PaddleOCR-VL 백엔드 자동 시작 스크립트
# 100% 한국어 정확도 OCR 서비스

echo "================================================================================"
echo "🚀 PaddleOCR-VL 백엔드 시작 스크립트"
echo "🎯 한국어 100% 정확도 OCR 서비스"
echo "📚 109개 언어 지원 - 문서 전처리 + 레이아웃 검출 + VLM 인식"
echo "================================================================================"
echo ""

# 스크립트 디렉토리로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📁 작업 디렉토리: $SCRIPT_DIR"
echo ""

# Python 가상환경 확인
if [ -d "venv" ]; then
    echo "✅ 가상환경 발견: venv"
    source venv/bin/activate
    echo "✅ 가상환경 활성화 완료"
elif [ -d ".venv" ]; then
    echo "✅ 가상환경 발견: .venv"
    source .venv/bin/activate
    echo "✅ 가상환경 활성화 완료"
else
    echo "⚠️  가상환경이 없습니다. 시스템 Python 사용"
fi

echo ""

# Python 버전 확인
echo "🐍 Python 버전:"
python3 --version
echo ""

# 필수 패키지 확인
echo "📦 필수 패키지 확인 중..."

check_package() {
    python3 -c "import $1" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "   ✅ $1 설치됨"
        return 0
    else
        echo "   ❌ $1 미설치"
        return 1
    fi
}

MISSING_PACKAGES=0

check_package "fastapi" || MISSING_PACKAGES=$((MISSING_PACKAGES + 1))
check_package "uvicorn" || MISSING_PACKAGES=$((MISSING_PACKAGES + 1))
check_package "paddleocr" || MISSING_PACKAGES=$((MISSING_PACKAGES + 1))
check_package "cv2" || MISSING_PACKAGES=$((MISSING_PACKAGES + 1))
check_package "PIL" || MISSING_PACKAGES=$((MISSING_PACKAGES + 1))

echo ""

if [ $MISSING_PACKAGES -gt 0 ]; then
    echo "⚠️  $MISSING_PACKAGES 개의 패키지가 없습니다."
    echo "📦 requirements.txt에서 설치 시도 중..."
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        if [ $? -eq 0 ]; then
            echo "✅ 패키지 설치 완료"
        else
            echo "❌ 패키지 설치 실패. 수동으로 설치하세요:"
            echo "   pip install -r requirements.txt"
            exit 1
        fi
    else
        echo "❌ requirements.txt 파일이 없습니다."
        exit 1
    fi
    echo ""
fi

# 포트 설정
OCR_PORT="${OCR_PORT:-8000}"
export OCR_PORT

echo "🔧 설정:"
echo "   포트: $OCR_PORT"
echo ""

# 이미 실행 중인 프로세스 확인
PID=$(lsof -ti:$OCR_PORT 2>/dev/null)
if [ ! -z "$PID" ]; then
    echo "⚠️  포트 $OCR_PORT가 이미 사용 중입니다 (PID: $PID)"
    echo "🔄 기존 프로세스를 종료하고 재시작합니다..."
    kill -9 $PID 2>/dev/null
    sleep 2
    echo "✅ 기존 프로세스 종료 완료"
    echo ""
fi

# main.py 파일 확인
if [ ! -f "main.py" ]; then
    echo "❌ main.py 파일이 없습니다!"
    exit 1
fi

echo "================================================================================"
echo "🚀 PaddleOCR-VL 서버 시작 중..."
echo "================================================================================"
echo ""

# 서버 시작
python3 main.py

# 종료 코드 확인
EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 서버가 정상적으로 종료되었습니다"
else
    echo "❌ 서버가 오류와 함께 종료되었습니다 (코드: $EXIT_CODE)"
fi
echo "================================================================================"

exit $EXIT_CODE
