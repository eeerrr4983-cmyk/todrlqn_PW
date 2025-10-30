# PaddleOCR-VL OCR 서비스

한국어 OCR을 위한 PaddleOCR-VL 기반 FastAPI 백엔드 서비스입니다.

## 🚀 빠른 시작

### 1. 종속성 설치

#### Linux/WSL 환경
\`\`\`bash
cd ocr_service
./install_paddle.sh
\`\`\`

#### 수동 설치
\`\`\`bash
# 기본 종속성
pip install -r requirements.txt

# PaddlePaddle GPU (CUDA 12.6)
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# PaddlePaddle CPU (GPU가 없는 경우)
# python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# PaddleOCR
python -m pip install -U "paddleocr[doc-parser]"
\`\`\`

### 2. 서버 실행

\`\`\`bash
# 기본 실행 (포트 8000)
python main.py

# 또는 uvicorn 직접 사용
uvicorn main:app --host 0.0.0.0 --port 8000

# 다른 포트로 실행
OCR_PORT=8080 python main.py
\`\`\`

### 3. Docker로 실행

\`\`\`bash
# 이미지 빌드
docker build -t paddleocr-vl-service .

# 컨테이너 실행
docker run -d -p 8000:8000 --name ocr-service paddleocr-vl-service
\`\`\`

## 📡 API 엔드포인트

### Health Check
\`\`\`bash
GET /
GET /health
\`\`\`

### OCR 텍스트 추출
\`\`\`bash
POST /ocr
Content-Type: multipart/form-data

files: [이미지 파일들]
\`\`\`

#### 예시 (curl)
\`\`\`bash
curl -X POST "http://localhost:8000/ocr" \
  -F "files=@image1.jpg" \
  -F "files=@image2.png"
\`\`\`

#### 응답
\`\`\`json
{
  "texts": [
    "추출된 텍스트 1...",
    "추출된 텍스트 2..."
  ],
  "success": true,
  "error": null
}
\`\`\`

## 🧠 한국어 특화 초정밀 파이프라인

- **Adaptive Preprocessing (6단계)**: CLAHE 기반 대비 향상, 다중 노이즈 제거, 탑햇/블랙햇 기반 한글 스트로크 강화, 감마 보정까지 자동 적용합니다.
- **동적 해상도 스케일링**: 작은 이미지도 최소 1600px 이상으로 업스케일링하여 작은 글자까지 선명하게 복원합니다.
- **품질 지표 텔레메트리**: 밝기·대비·선명도·노이즈 지수를 실시간으로 로그에 기록하여 추후 정량 분석이 가능합니다.
- **한국어 재추론 모드**: 추출 글자 수가 부족하면 자동으로 레이아웃 파라미터와 샘플링 전략을 조정한 2차 추론을 실행하여 누락을 최소화합니다.

## 🔧 환경 변수

- `OCR_PORT`: 서버 포트 (기본값: 8000)
- `PADDLE_DEVICE`: `gpu:0`, `cpu` 등 PaddleOCR 실행 디바이스 강제 지정
- `PADDLE_PRECISION`: `fp16` 또는 `fp32` Precision 강제 설정 (기본은 디바이스에 따라 자동)
- `KOREAN_OCR_FALLBACK_THRESHOLD`: 재추론을 트리거할 최소 추출 글자 수 임계값 (기본 18)

## 📊 성능

- **초기화 시간**: 10-30초 (첫 실행 시)
- **OCR 처리 속도**: 이미지당 2-5초
- **지원 언어**: 한국어, 영어 등 다양한 언어

## 🐛 문제 해결

### PaddlePaddle 설치 오류
- CUDA 버전 확인: `nvidia-smi`
- CUDA 12.6이 아닌 경우, 적절한 버전 설치
- CPU 버전 사용: `paddlepaddle` 대신 `paddlepaddle-gpu` 제거

### 메모리 부족
- 이미지 크기 줄이기
- 배치 처리 대신 순차 처리

### Windows 사용자
- WSL2 또는 Docker 사용 권장
- Windows 네이티브 설치는 제한적

## 🧪 테스트

\`\`\`bash
cd /home/user/webapp
python -m unittest ocr_service.tests.test_preprocessing
\`\`\`

## 📚 참고 자료

- [PaddleOCR-VL 공식 문서](https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PaddleOCR-VL.html)
- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [Hugging Face 모델](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)
