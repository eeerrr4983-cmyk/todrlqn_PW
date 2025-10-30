# 🚀 생기부AI 빠른 시작 가이드

## ⚡ 5분 안에 시작하기

### 1단계: OCR 서비스 설치 (최초 1회)

\`\`\`bash
npm run ocr:install
\`\`\`

이 명령어는 자동으로:
- PaddlePaddle 설치 (GPU/CPU 자동 감지)
- PaddleOCR 설치
- 필요한 Python 패키지 설치

**⏱️ 소요 시간**: 5-10분 (인터넷 속도에 따라)

### 2단계: OCR 서비스 시작

**터미널 1**에서:
\`\`\`bash
npm run ocr:start
\`\`\`

다음 메시지를 확인하세요:
\`\`\`
✅ PaddleOCR 초기화 완료!
✅ 서버가 정상적으로 시작되었습니다!
📡 엔드포인트: http://0.0.0.0:8000
\`\`\`

### 3단계: Next.js 앱 시작

**터미널 2 (새 터미널)**에서:
\`\`\`bash
npm run dev
\`\`\`

### 4단계: 브라우저에서 열기

\`\`\`
http://localhost:3000
\`\`\`

---

## 🔧 환경 변수 설정

`.env.local` 파일 생성:
\`\`\`bash
cp .env.local.example .env.local
\`\`\`

`.env.local` 파일 편집:
\`\`\`bash
# Google Gemini API Key (필수)
GEMINI_API_KEY=your_gemini_api_key_here

# PaddleOCR 백엔드 URL (기본값 사용 가능)
PADDLE_OCR_URL=http://localhost:8000
\`\`\`

### Gemini API 키 발급

1. https://aistudio.google.com/app/apikey 방문
2. Google 계정 로그인
3. "Create API Key" 클릭
4. API 키 복사 후 `.env.local`에 붙여넣기

---

## ❓ 문제 해결

### ❌ OCR 서비스 시작 실패

**증상**: `npm run ocr:start` 실행 시 에러

**해결책**:
\`\`\`bash
# Python 설치 확인
python --version  # 또는 python3 --version

# PaddleOCR 재설치
npm run ocr:install
\`\`\`

### ❌ "Failed to fetch" 에러

**증상**: 이미지 업로드 시 "Failed to fetch" 에러

**원인**: OCR 백엔드가 실행되지 않음

**해결책**:
1. 터미널 1에서 `npm run ocr:start` 실행 확인
2. http://localhost:8000/health 접속하여 확인
3. 응답이 없으면 OCR 서비스 재시작

### ❌ 포트 8000이 이미 사용 중

**해결책**:
\`\`\`bash
# 포트 사용 중인 프로세스 찾기
lsof -i :8000

# 프로세스 종료 (PID는 위 명령어 결과에서 확인)
kill -9 <PID>

# 또는 다른 포트 사용
OCR_PORT=8001 npm run ocr:start
\`\`\`

### ❌ GPU 메모리 부족

**증상**: CUDA out of memory 에러

**해결책**:
\`\`\`bash
# CPU 버전으로 재설치
cd ocr_service
pip uninstall paddlepaddle-gpu
pip install paddlepaddle==3.0.0 -i https://mirror.baidu.com/pypi/simple
\`\`\`

---

## 📊 성능 팁

### 🚀 빠른 OCR 처리

1. **이미지 품질 최적화**
   - 해상도: 1000-2000px (가로)
   - 형식: JPG (압축) 또는 PNG
   - 파일 크기: 2MB 이하

2. **GPU 사용** (권장)
   - NVIDIA GPU + CUDA 11.8 또는 12.0
   - 처리 속도 3-5배 향상

3. **배치 처리**
   - 한 번에 여러 이미지 업로드 가능
   - 최대 10장 권장

---

## 🎯 다음 단계

### 프로덕션 배포

1. **Docker 사용** (권장)
   \`\`\`bash
   cd ocr_service
   docker build -t paddleocr-service .
   docker run -d -p 8000:8000 paddleocr-service
   \`\`\`

2. **Next.js 빌드**
   \`\`\`bash
   npm run build
   npm start
   \`\`\`

### 모니터링

- OCR 백엔드 상태: http://localhost:8000/health
- Next.js 앱: http://localhost:3000

---

## 📚 추가 문서

- 상세 설치 가이드: [SETUP_GUIDE.md](./SETUP_GUIDE.md)
- OCR 서비스 문서: [ocr_service/README.md](./ocr_service/README.md)
- 프로젝트 README: [README.md](./README.md)

---

## 💡 도움이 필요하세요?

1. 로그 확인:
   - OCR 백엔드: 터미널 1 출력
   - Next.js: 터미널 2 출력
   - 브라우저: F12 > Console

2. GitHub Issues:
   - https://github.com/eeerrr4983-cmyk/todrlqn_PW/issues

---

**🎉 축하합니다! 이제 생기부AI를 사용할 준비가 되었습니다!**
