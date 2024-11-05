import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import cv2

# 페이지 설정
st.set_page_config(
    page_title="이미지 분류기",
    page_icon="🖼️",
    layout="wide"
)

# 캐시된 모델 로드 함수
@st.cache_resource
def load_model():
    return ResNet50(weights='imagenet')

def preprocess_image(image):
    # PIL 이미지를 numpy 배열로 변환
    img_array = np.array(image)
    
    # RGB로 변환 (이미지가 RGBA인 경우를 대비)
    if img_array.shape[-1] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # 크기 조정
    img_resized = cv2.resize(img_array, (224, 224))
    
    # 배치 차원 추가 및 전처리
    img_batch = np.expand_dims(img_resized, 0)
    processed_img = preprocess_input(img_batch)
    
    return processed_img

def main():
    # 제목과 설명
    st.title("📸 이미지 분류기")
    st.write("""
    이미지를 업로드하면 AI가 이미지 속 물체나 동물을 식별해드립니다.
    이 앱은 ImageNet 데이터셋으로 학습된 ResNet50 모델을 사용합니다.
    """)
    
    # 사이드바에 설정 옵션 추가
    st.sidebar.title("⚙️ 설정")
    confidence_threshold = st.sidebar.slider(
        "신뢰도 임계값",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        help="이 값보다 높은 신뢰도를 가진 예측만 표시됩니다."
    )
    
    top_k = st.sidebar.number_input(
        "표시할 예측 개수",
        min_value=1,
        max_value=10,
        value=3,
        help="상위 몇 개의 예측을 표시할지 선택하세요."
    )
    
    # 파일 업로더 배치
    uploaded_file = st.file_uploader(
        "이미지를 업로드하세요 (JPG, PNG 파일)",
        type=['jpg', 'jpeg', 'png']
    )
    
    # 이미지가 업로드되면 처리 시작
    if uploaded_file is not None:
        try:
            # 이미지 열기
            image = Image.open(uploaded_file)
            
            # 이미지와 예측 결과를 나란히 표시하기 위한 컬럼 생성
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("업로드된 이미지")
                st.image(image, caption="입력 이미지", use_column_width=True)
            
            # 모델 로드
            with st.spinner('모델 로딩 중...'):
                model = load_model()
            
            # 이미지 전처리 및 예측
            with st.spinner('이미지 분석 중...'):
                processed_img = preprocess_image(image)
                predictions = model.predict(processed_img)
                decoded_predictions = decode_predictions(predictions, top=int(top_k))[0]
            
            with col2:
                st.subheader("분류 결과")
                
                # 예측 결과를 표로 표시
                results_data = []
                for _, label, confidence in decoded_predictions:
                    if confidence >= confidence_threshold:
                        results_data.append({
                            "분류": label.replace('_', ' ').title(),
                            "신뢰도": f"{confidence*100:.2f}%"
                        })
                
                if results_data:
                    st.table(results_data)
                    
                    # 가장 높은 신뢰도를 가진 예측 강조 표시
                    st.success(f"🎯 가장 유력한 예측: {results_data[0]['분류']} ({results_data[0]['신뢰도']})")
                else:
                    st.warning("설정된 신뢰도 임계값을 넘는 예측 결과가 없습니다.")
                
        except Exception as e:
            st.error(f"에러가 발생했습니다: {str(e)}")
            st.error("이미지를 처리하는 도중 문제가 발생했습니다. 다른 이미지를 시도해보세요.")
    
    # 앱 사용 방법 및 정보
    with st.expander("ℹ️ 앱 사용 방법 및 정보"):
        st.markdown("""
        ### 사용 방법
        1. 왼쪽 사이드바에서 신뢰도 임계값과 표시할 예측 개수를 설정합니다.
        2. '이미지를 업로드하세요' 버튼을 클릭하여 분류하고 싶은 이미지를 선택합니다.
        3. 잠시 기다리면 AI가 이미지를 분석하고 결과를 표시합니다.
        
        ### 모델 정보
        - 이 앱은 ImageNet 데이터셋으로 사전 학습된 ResNet50 모델을 사용합니다.
        - 1000개의 다양한 클래스를 인식할 수 있습니다.
        - 결과의 신뢰도는 0%에서 100% 사이의 값으로 표시됩니다.
        
        ### 주의사항
        - 지원되는 이미지 형식: JPG, JPEG, PNG
        - 이미지 파일의 크기가 클 경우 처리에 시간이 걸릴 수 있습니다.
        - 결과의 정확도는 이미지의 품질과 선명도에 따라 달라질 수 있습니다.
        """)

if __name__ == '__main__':
    main()
