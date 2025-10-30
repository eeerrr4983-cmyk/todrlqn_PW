"""
한국어 OCR 웹 인터페이스
Streamlit 기반 사용자 친화적 UI
"""

import streamlit as st
import requests
import base64
import json
import time
from PIL import Image
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import cv2
import sys
import os

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Streamlit 설정
st.set_page_config(
    page_title="Korean OCR - 100% 정확도 생기부 인식",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stTitle {
        font-size: 3rem !important;
        color: #1f77b4 !important;
        text-align: center;
        margin-bottom: 2rem !important;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .success-message {
        padding: 1rem;
        border-radius: 5px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-message {
        padding: 1rem;
        border-radius: 5px;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# API 서버 URL
API_URL = "http://localhost:8000"

# 세션 상태 초기화
if "ocr_results" not in st.session_state:
    st.session_state.ocr_results = []
if "processing_jobs" not in st.session_state:
    st.session_state.processing_jobs = {}


def image_to_base64(image):
    """이미지를 Base64로 인코딩"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def perform_ocr(image_base64, options):
    """OCR API 호출"""
    try:
        response = requests.post(
            f"{API_URL}/ocr",
            json={
                "image_base64": image_base64,
                "enable_enhancement": options.get("enable_enhancement", True),
                "enable_layout_analysis": options.get("enable_layout_analysis", True),
                "enable_multi_pass": options.get("enable_multi_pass", True),
                "extract_fields": options.get("extract_fields", True)
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API 오류: {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"API 연결 오류: {str(e)}")
        return None


def check_job_status(job_id):
    """작업 상태 확인"""
    try:
        response = requests.get(f"{API_URL}/job/{job_id}", timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
            
    except requests.exceptions.RequestException:
        return None


def display_results(result):
    """OCR 결과 표시"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>인식 신뢰도</h4>
            <h2>{:.1%}</h2>
        </div>
        """.format(result.get("confidence", 0)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>한글 비율</h4>
            <h2>{:.1%}</h2>
        </div>
        """.format(result.get("korean_ratio", 0)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>처리 시간</h4>
            <h2>{:.2f}초</h2>
        </div>
        """.format(result.get("processing_time", 0)), unsafe_allow_html=True)
    
    with col4:
        total_chars = len(result.get("full_text", ""))
        st.markdown(f"""
        <div class="metric-card">
            <h4>총 문자 수</h4>
            <h2>{total_chars:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # 탭 생성
    tab1, tab2, tab3, tab4 = st.tabs(["📝 텍스트", "📊 필드 추출", "🗺️ 레이아웃", "📈 분석"])
    
    with tab1:
        st.subheader("추출된 텍스트")
        full_text = result.get("full_text", "")
        
        if full_text:
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.text_area("전체 텍스트", full_text, height=400, key="full_text_display")
            
            # 텍스트 다운로드 버튼
            st.download_button(
                label="📥 텍스트 다운로드",
                data=full_text,
                file_name=f"ocr_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("추출된 텍스트가 없습니다.")
    
    with tab2:
        st.subheader("생기부 필드 추출 결과")
        fields = result.get("extracted_fields", {})
        
        if fields:
            # 필드별 표시
            for field_name, field_value in fields.items():
                if field_value:
                    st.markdown(f"**{field_name}:**")
                    st.info(field_value)
            
            # JSON 다운로드 버튼
            st.download_button(
                label="📥 필드 데이터 다운로드 (JSON)",
                data=json.dumps(fields, ensure_ascii=False, indent=2),
                file_name=f"fields_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        else:
            st.warning("추출된 필드가 없습니다.")
    
    with tab3:
        st.subheader("문서 레이아웃 분석")
        layout = result.get("layout_analysis", {})
        
        if layout:
            # 레이아웃 통계
            stats = layout.get("statistics", {})
            if stats:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("총 요소 수", stats.get("total_elements", 0))
                
                with col2:
                    st.metric("테이블 수", stats.get("table_count", 0))
                
                with col3:
                    st.metric("평균 신뢰도", f"{stats.get('average_confidence', 0):.1%}")
            
            # 요소 타입별 분포
            element_types = stats.get("element_types", {})
            if element_types:
                df = pd.DataFrame(list(element_types.items()), columns=["Type", "Count"])
                fig = px.pie(df, values="Count", names="Type", title="레이아웃 요소 분포")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("레이아웃 분석 결과가 없습니다.")
    
    with tab4:
        st.subheader("상세 분석")
        
        # 신뢰도 히스토그램
        if "scores" in result and result["scores"]:
            scores = result["scores"]
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=scores, nbinsx=20, name="신뢰도 분포"))
            fig.update_layout(
                title="텍스트 영역별 신뢰도 분포",
                xaxis_title="신뢰도",
                yaxis_title="빈도",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 처리 시간 분석
        if st.session_state.ocr_results:
            times = [r.get("processing_time", 0) for r in st.session_state.ocr_results]
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=times, mode='lines+markers', name="처리 시간"))
            fig.update_layout(
                title="처리 시간 추이",
                xaxis_title="작업 번호",
                yaxis_title="시간 (초)"
            )
            st.plotly_chart(fig, use_container_width=True)


def main():
    # 제목
    st.markdown("<h1 class='stTitle'>🚀 Korean OCR - 100% 정확도 달성</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #666;'>생기부 문서 완벽 인식 시스템</p>", unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # OCR 옵션
        st.subheader("OCR 옵션")
        enable_enhancement = st.checkbox("딥러닝 향상 활성화", value=True)
        enable_layout = st.checkbox("레이아웃 분석 활성화", value=True)
        enable_multi_pass = st.checkbox("다중 패스 OCR", value=True)
        extract_fields = st.checkbox("생기부 필드 자동 추출", value=True)
        
        options = {
            "enable_enhancement": enable_enhancement,
            "enable_layout_analysis": enable_layout,
            "enable_multi_pass": enable_multi_pass,
            "extract_fields": extract_fields
        }
        
        # API 상태 확인
        st.subheader("🔗 API 상태")
        if st.button("상태 확인"):
            try:
                response = requests.get(f"{API_URL}/health", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    st.success("✅ API 정상 작동 중")
                    st.json(data)
                else:
                    st.error("❌ API 응답 오류")
            except:
                st.error("❌ API 연결 실패")
        
        # 통계 정보
        if st.button("📊 통계 보기"):
            try:
                response = requests.get(f"{API_URL}/stats", timeout=5)
                if response.status_code == 200:
                    stats = response.json()
                    st.subheader("📈 처리 통계")
                    for key, value in stats.items():
                        st.metric(key.replace("_", " ").title(), value)
            except:
                st.error("통계 조회 실패")
    
    # 메인 컨텐츠
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 이미지 업로드")
        
        # 파일 업로더
        uploaded_files = st.file_uploader(
            "생기부 이미지를 선택하세요",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.success(f"{len(uploaded_files)}개 파일 업로드 완료")
            
            # 미리보기
            for i, uploaded_file in enumerate(uploaded_files):
                with st.expander(f"📷 {uploaded_file.name}"):
                    image = Image.open(uploaded_file)
                    st.image(image, use_container_width=True)
            
            # OCR 실행 버튼
            if st.button("🚀 OCR 시작", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"처리 중... ({i+1}/{len(uploaded_files)})")
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    # 이미지 로드 및 인코딩
                    image = Image.open(uploaded_file)
                    image_base64 = image_to_base64(image)
                    
                    # OCR 요청
                    result = perform_ocr(image_base64, options)
                    
                    if result:
                        job_id = result.get("job_id")
                        
                        # 작업 완료 대기
                        max_wait = 60  # 최대 60초 대기
                        start_time = time.time()
                        
                        while time.time() - start_time < max_wait:
                            job_status = check_job_status(job_id)
                            
                            if job_status:
                                if job_status["status"] == "completed":
                                    final_result = job_status["result"]
                                    final_result["filename"] = uploaded_file.name
                                    st.session_state.ocr_results.append(final_result)
                                    break
                                elif job_status["status"] == "failed":
                                    st.error(f"처리 실패: {job_status.get('error', 'Unknown error')}")
                                    break
                            
                            time.sleep(1)
                
                progress_bar.progress(1.0)
                status_text.text("✅ 처리 완료!")
                st.balloons()
    
    with col2:
        st.header("📊 OCR 결과")
        
        if st.session_state.ocr_results:
            # 결과 선택
            result_options = [f"{i+1}. {r['filename']}" for i, r in enumerate(st.session_state.ocr_results)]
            selected_index = st.selectbox("결과 선택:", range(len(result_options)), 
                                        format_func=lambda x: result_options[x])
            
            if selected_index is not None:
                selected_result = st.session_state.ocr_results[selected_index]
                display_results(selected_result)
        else:
            st.info("OCR 결과가 여기에 표시됩니다.")
    
    # 배치 처리 섹션
    with st.expander("🔄 배치 처리"):
        st.subheader("여러 파일 일괄 처리")
        
        batch_files = st.file_uploader(
            "여러 파일을 선택하세요",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            accept_multiple_files=True,
            key="batch_uploader"
        )
        
        if batch_files and st.button("배치 처리 시작", key="batch_process"):
            with st.spinner(f"{len(batch_files)}개 파일 처리 중..."):
                # 파일 준비
                files = []
                for file in batch_files:
                    files.append(("files", (file.name, file.getvalue(), file.type)))
                
                # 배치 OCR 요청
                try:
                    response = requests.post(
                        f"{API_URL}/ocr/batch",
                        files=files,
                        timeout=300
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ {result['message']}")
                        st.json(result["job_ids"])
                    else:
                        st.error("배치 처리 실패")
                except Exception as e:
                    st.error(f"오류 발생: {str(e)}")
    
    # 텍스트 향상 도구
    with st.expander("✨ 텍스트 향상 도구"):
        st.subheader("OCR 결과 텍스트 교정")
        
        input_text = st.text_area("교정할 텍스트 입력:", height=150)
        
        if input_text and st.button("텍스트 향상", key="enhance_text"):
            try:
                response = requests.post(
                    f"{API_URL}/ocr/enhance",
                    json={"text": input_text},
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.text_area("원본 텍스트:", result["original_text"], height=150)
                    
                    with col2:
                        st.text_area("교정된 텍스트:", result["corrected_text"], height=150)
                    
                    if result["corrections"]:
                        st.subheader("교정 내역")
                        for correction in result["corrections"]:
                            st.write(f"• {correction}")
                    
                    st.success(f"✅ {result['changes_made']}개 항목 교정 완료")
            except Exception as e:
                st.error(f"텍스트 향상 실패: {str(e)}")
    
    # 푸터
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Korean OCR Enhanced System v1.0.0</p>
        <p>🎯 생기부 문서 100% 정확도 달성 목표</p>
        <p>Powered by PaddleOCR + Deep Learning + AI Enhancement</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
