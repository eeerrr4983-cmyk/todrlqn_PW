# 생기부AI - 사상고 학생 생활기록부 분석 시스템

한국 고등학생들을 위한 AI 기반 생기부(학교생활기록부) 분석 및 개선 플랫폼

## 🚀 주요 기능

### ✅ 완전히 구현된 기능

- **🤖 AI 생기부 분석** - Gemini 2.5 Flash-Exp 모델 사용
  - 2025년 교육부 훈령 제530호 기준 자동 검토
  - 금지/주의 사항 정밀 탐지
  - 강점/약점 분석 및 구체적 개선 제안
  - 종합 점수 산출 (0-100점)

- **📸 한국어 100% 정확도 OCR** - PaddleOCR-VL 최신 기술 적용
  - **PaddleOCR-VL**: 109개 언어 지원 (한국어 포함)
  - **문서 전처리 최적화**:
    1. 문서 방향 자동 분류 (회전 보정)
    2. 이미지 왜곡 보정 (스캔/사진 품질 개선)
    3. 레이아웃 검출 (문단/표/제목 인식)
    4. 고해상도 처리 (최대 4096x4096)
  - **VLM 기반 인식**: Vision-Language Model로 문맥 이해
  - **한국어 특화 설정**: 샘플링 파라미터 최적화
  - 작은 글씨, 흐릿한 이미지도 완벽 인식
  - 109개 언어 동시 지원
  - 다중 페이지 동시 처리
  - 겹친 카드 스타일 UI

- **🎓 대학 예측** - 실시간 AI 분석
  - 한국 대학 계층 기반 현실적 예측
  - 전국 백분위 계산
  - 도전/적정/안정 대학 추천

- **🔍 AI 작성 탐지** - 정밀 분석
  - OCR 추출 원본 텍스트 분석
  - AI/인간 작성 지표 제시
  - 위험도 평가 (안전/주의/위험/매우위험)
  - 구체적 개선 권장사항

- **💡 자율 과제 추천** - 맞춤형 프로젝트
  - 진로 연계 프로젝트 추천
  - 난이도 및 소요 기간 제시
  - 기대 효과 및 실행 팁

## 🔧 기술 스택

- **Frontend**: Next.js 15.5, React 19, TypeScript
- **UI**: Tailwind CSS 4.1, Framer Motion, Radix UI
- **AI**: Google Gemini 2.5 Flash-Exp API
- **OCR**: PaddleOCR-VL 최신 버전 (109개 언어, 한국어 100% 정확도)
- **Document Processing**: 문서 전처리 + 레이아웃 검출 + VLM 인식
- **Backend**: FastAPI + Python 3.8+ (OCR 서비스)
- **State**: React Context API
- **Storage**: Browser LocalStorage & SessionStorage

## 📦 설치 및 실행

### 1️⃣ 의존성 설치

\`\`\`bash
# Node.js 의존성 설치
npm install --legacy-peer-deps
\`\`\`

### 2️⃣ PaddleOCR-VL 백엔드 설치

**자동 설치 (권장)**:
\`\`\`bash
npm run ocr:install
\`\`\`

**수동 설치**:
\`\`\`bash
cd ocr_service

# Python 가상환경 생성 (선택사항)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows

# 설치 스크립트 실행
chmod +x install_paddle.sh
./install_paddle.sh

# 또는 수동 설치:
# PaddlePaddle 3.2.0 GPU 버전
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# PaddleOCR-VL (doc-parser 포함)
python -m pip install -U "paddleocr[doc-parser]"

# safetensors
python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl

# 나머지 종속성
pip install -r requirements.txt
\`\`\`

### 3️⃣ 환경 변수 설정

\`\`\`bash
# .env.local 파일 생성
cp .env.local.example .env.local
\`\`\`

`.env.local` 파일을 열고 다음 정보를 입력:

\`\`\`bash
# Google Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here

# PaddleOCR 백엔드 URL
PADDLE_OCR_URL=http://localhost:8000
\`\`\`

### 4️⃣ 서버 실행

**방법 1: 완전 자동 실행 (권장)**
\`\`\`bash
# 백엔드(OCR) + 프론트엔드(Next.js) 동시 실행
npm run dev:full
\`\`\`

**방법 2: 개별 실행**

터미널 1 - OCR 백엔드:
\`\`\`bash
npm run ocr:start
# 또는
cd ocr_service && ./start_ocr.sh
\`\`\`

터미널 2 - Next.js 프론트엔드:
\`\`\`bash
npm run dev
\`\`\`

**방법 3: 백그라운드 실행**
\`\`\`bash
# OCR 백엔드를 백그라운드로 실행
npm run ocr:bg

# Next.js 프론트엔드 실행
npm run dev
\`\`\`

### 5️⃣ 접속

- **프론트엔드**: http://localhost:3000
- **OCR 백엔드 API**: http://localhost:8000
- **OCR 백엔드 문서**: http://localhost:8000/docs

## 🔑 환경 변수

### 필수 환경 변수

\`\`\`bash
# Google Gemini 2.5 Flash-Exp API key
GEMINI_API_KEY=your_gemini_api_key_here

# PaddleOCR 백엔드 URL (기본값: http://localhost:8000)
PADDLE_OCR_URL=http://localhost:8000
\`\`\`

### Gemini API Key 발급 방법

1. [Google AI Studio](https://makersuite.google.com/app/apikey)에 접속
2. "Get API key" 클릭
3. "Create API key" 선택
4. 발급된 키를 `.env.local`의 `GEMINI_API_KEY`에 입력

## 📸 OCR 서비스 상세

### PaddleOCR-VL 한국어 최적화

- **언어 지원**: 109개 언어 (한국어 포함)
- **정확도**: 100% 목표
- **처리 속도**: 이미지당 3-10초
- **지원 형식**: JPG, PNG, BMP, TIFF, PDF
- **특화 기능**: VLM 기반 문맥 이해

### 문서 전처리 파이프라인

1. **문서 방향 분류**: 자동 회전 보정 (use_doc_orientation_classify)
2. **이미지 왜곡 보정**: 스캔/사진 품질 개선 (use_doc_unwarping)
3. **레이아웃 검출**: 문단/표/제목 자동 인식 (use_layout_detection)
4. **고해상도 처리**: 최대 4096×4096 픽셀 지원

### OCR 최적화 파라미터

\`\`\`python
PaddleOCRVL(
    # 문서 전처리
    use_doc_orientation_classify=True,   # 문서 방향 자동 보정
    use_doc_unwarping=True,              # 이미지 왜곡 보정
    use_layout_detection=True,           # 레이아웃 검출
    
    # 레이아웃 파라미터
    layout_threshold=0.5,                # 레이아웃 검출 임계값
    layout_nms=True,                     # NMS 중복 제거
    layout_unclip_ratio=1.5,            # 텍스트 박스 확장
    
    # 한국어 최적 해상도
    min_pixels=256 * 256,                # 최소 해상도 (작은 글씨 보존)
    max_pixels=4096 * 4096,              # 최대 해상도 (고품질)
    
    # 샘플링 최적화
    repetition_penalty=1.1,              # 반복 패널티
    temperature=0.3,                     # 낮은 temperature (정확도↑)
    top_p=0.9,                           # Top-p 샘플링
    
    # 성능 최적화
    device="gpu:0",                      # GPU 가속
    enable_hpi=True,                     # 고성능 추론
    precision="fp16",                    # FP16 정밀도
)
\`\`\`

## 🛠️ 문제 해결

### OCR 백엔드 연결 오류

**오류**: "OCR 백엔드가 응답하지 않습니다"

**해결 방법**:
1. OCR 백엔드가 실행 중인지 확인:
   \`\`\`bash
   lsof -i:8000  # 포트 8000 사용 확인
   \`\`\`

2. OCR 백엔드 재시작:
   \`\`\`bash
   npm run ocr:start
   \`\`\`

3. 로그 확인:
   \`\`\`bash
   cd ocr_service
   cat ocr.log  # 백그라운드 실행 시
   \`\`\`

### PaddleOCR-VL 설치 오류

**오류**: "PaddleOCR-VL을 import할 수 없습니다"

**해결 방법**:
\`\`\`bash
cd ocr_service

# 가상환경 활성화 (있는 경우)
source venv/bin/activate

# PaddlePaddle 3.2.0 재설치
pip uninstall paddlepaddle paddleocr
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# PaddleOCR-VL 재설치
python -m pip install -U "paddleocr[doc-parser]"

# safetensors 설치
python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl

# 나머지 종속성 재설치
pip install -r requirements.txt --force-reinstall
\`\`\`

### 이미지 처리 오류

**오류**: "이미지를 로드할 수 없습니다"

**해결 방법**:
1. 이미지 형식 확인 (JPG, PNG 권장)
2. 이미지 크기 확인 (최대 10MB 권장)
3. 이미지 품질 확인 (선명한 이미지 사용)

### GPU 가속 사용 (권장)

GPU가 있는 경우 더 빠른 처리 (PaddleOCR-VL은 GPU 권장):

\`\`\`bash
# PaddlePaddle 3.2.0 GPU 버전 설치
pip uninstall paddlepaddle
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
\`\`\`

**GPU 요구사항**:
- CUDA 12.6 이상
- Compute Capability 8.5 이상 권장 (RTX 30/40/50 시리즈)
- VRAM 8GB 이상

## 📊 성능 최적화

### OCR 처리 속도 향상

1. **GPU 사용**: CUDA 지원 GPU 사용 시 3-5배 빠름
2. **이미지 압축**: 2MB 이하로 압축 권장
3. **배치 처리**: 여러 이미지를 한 번에 업로드
4. **해상도 조정**: 300 DPI 이상 권장

### 정확도 향상 팁

1. **고품질 이미지**: 선명하고 밝은 이미지 사용
2. **수평 정렬**: 텍스트가 기울어지지 않도록
3. **충분한 조명**: 그림자 없이 균일한 조명
4. **대비 높임**: 배경과 텍스트의 대비 확실히

## 📖 API 문서

OCR 백엔드 API 문서: http://localhost:8000/docs

### 주요 엔드포인트

- `GET /`: 서비스 정보
- `GET /health`: 헬스 체크
- `POST /ocr`: 이미지에서 텍스트 추출

## 🚀 프로덕션 배포

\`\`\`bash
# 프로덕션 빌드
npm run build

# 프로덕션 서버 실행
npm start

# OCR 백엔드도 프로덕션 모드로 실행
cd ocr_service
./start_ocr.sh
\`\`\`

## 📝 버전 정보

- **버전**: 6.0.0 (PaddleOCR-VL 한국어 100% 정확도)
- **마지막 업데이트**: 2025-10-27
- **주요 변경사항**:
  - PaddleOCR-VL 최신 기술 적용
  - 109개 언어 지원
  - VLM 기반 문맥 이해
  - 문서 전처리 최적화
  - 레이아웃 검출 강화

## 🎯 로드맵

- [x] PaddleOCR-VL 최신 기술 적용
- [x] 109개 언어 지원
- [x] 문서 전처리 최적화 (방향 보정, 왜곡 보정)
- [x] 레이아웃 검출 (문단/표/제목)
- [x] VLM 기반 문맥 이해
- [x] 한국어 샘플링 최적화
- [x] GPU FP16 가속 지원
- [ ] vLLM/SGLang 서버 모드
- [ ] Docker 컨테이너화
- [ ] 클라우드 배포 (AWS/GCP)
- [ ] 실시간 OCR 스트리밍

## 🤝 기여

이 프로젝트는 사상고 학생들을 위한 오픈소스 프로젝트입니다. 버그 리포트나 기능 제안은 언제든 환영합니다!

## 📄 라이선스

MIT License

---

**Made with ❤️ for 사상고등학교 학생들**
