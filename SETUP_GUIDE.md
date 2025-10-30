# 생기부AI 설치 및 실행 가이드

## 🚀 빠른 시작

### 필수 요구사항

1. **Node.js**: 18.17 이상
2. **Python**: 3.8 이상 (OCR 서비스용)
3. **GPU (선택사항)**: NVIDIA GPU + CUDA 12.6 (더 빠른 OCR 처리)

### 1단계: 프로젝트 클론

\`\`\`bash
git clone <repository-url>
cd webapp
\`\`\`

### 2단계: Next.js 프론트엔드 설정

\`\`\`bash
# 의존성 설치
npm install --legacy-peer-deps

# 환경 변수 설정
cp .env.local.example .env.local

# .env.local 파일 편집
# GEMINI_API_KEY=your_actual_api_key_here
# PADDLE_OCR_URL=http://localhost:8000
\`\`\`

### 3단계: PaddleOCR-VL 백엔드 설정

#### 옵션 A: 스크립트로 자동 설치 (Linux/WSL/Mac)

\`\`\`bash
cd ocr_service
./install_paddle.sh
\`\`\`

#### 옵션 B: 수동 설치

\`\`\`bash
cd ocr_service

# 기본 종속성 설치
pip install -r requirements.txt

# PaddlePaddle GPU 버전 (CUDA 12.6)
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# 또는 PaddlePaddle CPU 버전 (GPU가 없는 경우)
# python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# PaddleOCR 설치
python -m pip install -U "paddleocr[doc-parser]"
\`\`\`

#### 옵션 C: Docker 사용

\`\`\`bash
cd ocr_service

# 이미지 빌드
docker build -t paddleocr-vl-service .

# 컨테이너 실행
docker run -d -p 8000:8000 --name ocr-service paddleocr-vl-service

# 로그 확인
docker logs -f ocr-service
\`\`\`

### 4단계: 서비스 실행

**터미널 1 - OCR 백엔드 서버**
\`\`\`bash
cd ocr_service
python main.py

# 또는 uvicorn 직접 사용
uvicorn main:app --host 0.0.0.0 --port 8000
\`\`\`

**터미널 2 - Next.js 프론트엔드**
\`\`\`bash
# 개발 모드
npm run dev

# 프로덕션 모드
npm run build
npm start
\`\`\`

### 5단계: 브라우저에서 접속

\`\`\`
http://localhost:3000
\`\`\`

## 🔍 서비스 확인

### OCR 백엔드 헬스 체크
\`\`\`bash
# 기본 상태 확인
curl http://localhost:8000

# 상세 헬스 체크
curl http://localhost:8000/health
\`\`\`

예상 응답:
\`\`\`json
{
  "status": "healthy",
  "paddle_ocr": "initialized",
  "service": "PaddleOCR-VL API"
}
\`\`\`

### OCR 테스트
\`\`\`bash
curl -X POST "http://localhost:8000/ocr" \
  -F "files=@test_image.jpg"
\`\`\`

## 🐛 문제 해결

### OCR 백엔드가 시작되지 않을 때

1. **Python 버전 확인**
   \`\`\`bash
   python --version  # 3.8 이상이어야 함
   \`\`\`

2. **PaddlePaddle 설치 확인**
   \`\`\`bash
   python -c "import paddle; print(paddle.__version__)"
   \`\`\`

3. **PaddleOCR 설치 확인**
   \`\`\`bash
   python -c "from paddleocr import PaddleOCRVL; print('OK')"
   \`\`\`

### Next.js에서 OCR API 연결 실패

1. **OCR 백엔드 실행 확인**
   \`\`\`bash
   curl http://localhost:8000/health
   \`\`\`

2. **환경 변수 확인**
   - `.env.local` 파일에 `PADDLE_OCR_URL=http://localhost:8000` 설정

3. **포트 충돌 확인**
   \`\`\`bash
   lsof -i :8000  # 포트 8000이 사용 중인지 확인
   \`\`\`

### CUDA/GPU 오류

GPU가 없거나 CUDA 버전이 다른 경우, CPU 버전으로 전환:

\`\`\`bash
pip uninstall paddlepaddle-gpu
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
\`\`\`

### Windows 사용자

Windows에서는 WSL2 또는 Docker 사용을 권장합니다:

\`\`\`bash
# WSL2 설치 후
wsl
cd /path/to/project
./ocr_service/install_paddle.sh
\`\`\`

## 📊 성능 최적화

### GPU 사용 (권장)

- NVIDIA GPU + CUDA 12.6 환경에서 최적 성능
- OCR 처리 속도 3-5배 향상

### CPU만 사용하는 경우

- 처리 시간이 더 길 수 있음 (이미지당 5-10초)
- 메모리 사용량 주의

## 🔐 프로덕션 배포

### 환경 변수 설정

\`\`\`bash
# .env.local (Next.js)
GEMINI_API_KEY=your_production_api_key
PADDLE_OCR_URL=http://your-ocr-server:8000

# OCR 서비스 (환경 변수)
OCR_PORT=8000
\`\`\`

### Docker Compose 사용 (권장)

프로젝트 루트에 `docker-compose.yml` 생성:

\`\`\`yaml
version: '3.8'

services:
  ocr-service:
    build: ./ocr_service
    ports:
      - "8000:8000"
    environment:
      - OCR_PORT=8000
    restart: unless-stopped

  nextjs-app:
    build: .
    ports:
      - "3000:3000"
    environment:
      - PADDLE_OCR_URL=http://ocr-service:8000
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    depends_on:
      - ocr-service
    restart: unless-stopped
\`\`\`

실행:
\`\`\`bash
docker-compose up -d
\`\`\`

## 📚 추가 리소스

- [PaddleOCR 공식 문서](https://www.paddleocr.ai/)
- [FastAPI 문서](https://fastapi.tiangolo.com/)
- [Next.js 문서](https://nextjs.org/docs)

## 💡 팁

1. **첫 실행 시**: OCR 모델 다운로드로 인해 초기화에 시간이 걸릴 수 있습니다 (10-30초)
2. **이미지 품질**: 고해상도 이미지일수록 OCR 정확도가 높습니다
3. **메모리 관리**: 대용량 이미지 처리 시 메모리 사용량 모니터링 권장
