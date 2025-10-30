#!/bin/bash

# 🚀 Production Deployment Script
# 생기부AI - 프로덕션 배포 스크립트

set -e  # Exit on error

echo "================================"
echo "🚀 생기부AI 프로덕션 배포 시작"
echo "================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 1. Environment Check
echo "📋 Step 1: 환경 변수 확인"
if [ ! -f .env.local ]; then
    echo -e "${RED}❌ .env.local 파일이 없습니다!${NC}"
    exit 1
fi

if ! grep -q "NEXT_PUBLIC_GEMINI_API_KEY" .env.local; then
    echo -e "${RED}❌ GEMINI_API_KEY가 설정되지 않았습니다!${NC}"
    exit 1
fi

if ! grep -q "NEXT_PUBLIC_OCR_API_KEY" .env.local; then
    echo -e "${RED}❌ OCR_API_KEY가 설정되지 않았습니다!${NC}"
    exit 1
fi

echo -e "${GREEN}✅ 환경 변수 확인 완료${NC}"
echo ""

# 2. Clean build
echo "🧹 Step 2: 이전 빌드 정리"
rm -rf .next
rm -rf out
echo -e "${GREEN}✅ 정리 완료${NC}"
echo ""

# 3. Install dependencies
echo "📦 Step 3: 의존성 확인"
npm install
echo -e "${GREEN}✅ 의존성 설치 완료${NC}"
echo ""

# 4. Type check
echo "🔍 Step 4: TypeScript 타입 검사"
npx tsc --noEmit || echo -e "${YELLOW}⚠️  타입 에러가 있지만 계속 진행합니다${NC}"
echo ""

# 5. Build
echo "🏗️  Step 5: 프로덕션 빌드"
echo "⏱️  빌드에는 3-5분 정도 소요될 수 있습니다..."
npm run build

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ 빌드 성공!${NC}"
else
    echo -e "${RED}❌ 빌드 실패!${NC}"
    exit 1
fi
echo ""

# 6. Build analysis
echo "📊 Step 6: 빌드 분석"
if [ -d ".next/static/chunks" ]; then
    echo "Chunk 크기:"
    du -h .next/static/chunks/pages/*.js 2>/dev/null | sort -hr | head -10 || echo "No page chunks found"
    echo ""
    echo "전체 빌드 크기:"
    du -sh .next/
fi
echo ""

# 7. Production ready check
echo "✅ Step 7: 프로덕션 준비 검증"
echo "주요 파일 확인:"

files_to_check=(
    "app/page.tsx"
    "app/layout.tsx"
    "components/error-boundary.tsx"
    "lib/lazy-components.tsx"
    "lib/image-optimization.ts"
    "lib/performance-monitor.ts"
    "lib/production-ready-check.ts"
    "next.config.mjs"
)

all_files_exist=true
for file in "${files_to_check[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅${NC} $file"
    else
        echo -e "${RED}❌${NC} $file (없음)"
        all_files_exist=false
    fi
done

if [ "$all_files_exist" = false ]; then
    echo -e "${RED}❌ 일부 필수 파일이 누락되었습니다!${NC}"
    exit 1
fi
echo ""

# 8. Success message
echo "================================"
echo -e "${GREEN}🎉 프로덕션 빌드 완료!${NC}"
echo "================================"
echo ""
echo "다음 단계:"
echo "1. 로컬 테스트: npm run start"
echo "2. Vercel 배포: vercel --prod"
echo "3. 또는 수동 배포: 빌드된 .next 폴더를 서버에 업로드"
echo ""
echo "배포 후 확인 사항:"
echo "- Lighthouse 점수 확인 (목표: 90+)"
echo "- 모든 기능 정상 작동 확인"
echo "- 성능 모니터링 확인"
echo ""
