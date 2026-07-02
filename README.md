# geoai_landcover (CNN Beginner Version)

위성사진을 5개 클래스로 분류하는 CNN 프로젝트입니다.

"전처리 -> 학습 -> 앱 예측" 흐름이 보이도록 단순하게 구성되어 있습니다.

## 1. 분류 클래스
- 도시
- 농지
- 산림
- 바다
- 황무지

## 2. 사용 모델
- `SimpleLandCoverCNN` (직접 만든 기본 CNN)
- ResNet 같은 복잡한 모델은 사용하지 않음

## 3. 프로젝트 흐름
1. EuroSAT 데이터 로드 및 train/val/test 분할 (8:1:1)
2. CNN 학습 후 최고 성능 모델 저장
3. Streamlit 앱에서 이미지 업로드 후 예측

## 4. 실행 방법 (Windows PowerShell)

### (1) 프로젝트 폴더 이동
```powershell
cd D:\geoai_landcover
```

### (2) 가상환경 활성화
```powershell
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& .venv\Scripts\Activate.ps1)
```

### (3) 라이브러리 설치
```powershell
pip install -r requirements.txt
```

### (4) 전처리 실행
```powershell
python -m src.data.preprocess
```

### (5) CNN 학습 실행
```powershell
python -m src.training.train
```

학습이 끝나면 아래 파일이 생성됩니다.
- `outputs/checkpoints/best_cnn_eurosat.pt`

### (6) Streamlit 앱 실행
```powershell
python -m streamlit run app/streamlit_app.py
```

브라우저에서 `http://localhost:8501` 로 접속합니다.

## 5. 자주 나는 오류

### 오류: `File does not exist: streamlit_app.py`
원인: 루트 폴더에서 `streamlit_app.py`를 직접 실행했기 때문입니다.

해결: 아래처럼 실행하세요.
```powershell
python -m streamlit run app/streamlit_app.py
```

## 6. 핵심 코드 위치
- 전처리: `src/data/preprocess.py`
- 모델: `src/models/landcover_model.py`
- 학습: `src/training/train.py`
- 추론: `src/inference/predict.py`
- 앱: `app/streamlit_app.py`
