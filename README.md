# SKN32-2nd-5Team
## 💭 프로젝트명 : Churn-Zero AI (가입 고객 이탈 예측 시스템)  

![Python](https://img.shields.io/badge/Python-3.10-3776AB)
![Pandas](https://img.shields.io/badge/Pandas-3.0.3-150458)
![NumPy](https://img.shields.io/badge/NumPy-2.5.0-013243)
![matplotlib](https://img.shields.io/badge/matplotlib-3.11.0-FF7F0E)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12.1-EE4C2C)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.9.0-F7931E)
![tqdm](https://img.shields.io/badge/tqdm-4.68.3-FFC107)

---

## 😃 구성원 소개
### 팀명 : 딥콰이어트🤫

| 이름 | 역할 |
| :---- | :--- |
| 하정원 | Text-CNN 모델 학습 |
| 권소라 | 데이터 정제 및 중복 제거, KLUE-BERT Fine-tuning |
| 김지혜 | 데이터 정제 및 중복 제거, LSTM 모델 학습 |
| 신누리 | 데이터 정제 및 중복 제거, 데이터 분할 및 분포 조정, 언어 모델 입력을 위한 변환, Transformer Encoder (Scratch) 모델 학습 |

## 📜 Git Convention
1\. main은 직접 수정 금지    
2\. 각자 브랜치 생성 후 작업  
3\. 작업 시작 전에는 항상 main에서 pull 받기  
4\. 프로젝트 시작 전 개발환경 동일하게 설정(. gitignore, requests.txt 등)  
5\. 브랜치 통합은 모두 있을 때 진행

## 🧑‍💻 개발환경 설정
#### 1. 로컬 환경에 Git 복사하기
```bash
# 프로젝트를 생성할 위치에서 실행  

git clone https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN32-2nd-5Team.git
```
#### 2. 가상 환경 만들기 & 활성화
```bash
# 가상 환경 생성
python -m venv .venv

# 자동 활성화가 되지 않았을 경우
.venv\Scripts\activate.bat
```
#### 3. 패키지 설치
```bash
pip install -r requirements.txt
```
#### 4. main 브랜치를 복사한 개인용 브랜치 생성
```bash
# 브랜치 생성 후 이동
git checkout -b 개인용브랜치이름  
```
#### 5. 생성된 개인 브랜치에서 작업 후 원격 브랜치에 업로드
```bash
# 작업 내용 모두 스테이징
git add .   

# 커밋 메세지 작성
git commit -m "커밋 메세지"  

# 개인 원격 브랜치에 푸시
git push origin 개인용브랜치이름   
```

---

## 📒 프로젝트 구조
```text
SKN32-2nd-5Team
├── data/
│   ├── Training.json
│   └── Validation.json
│
├── model/
│   ├── predict.py
│   ├── train_bert.py
│   ├── train_cnn.py
│   ├── train_lstm.py
│   ├── train_transformer.py
│   └── utils.py
│
├── output/
│
├── preprocessing
│   ├── data_analysis.py
│   ├── generate_report.py
│   ├── labeling.py
│   ├── prepare_dataset.py
│   └── sklearn_validate_labels.py
│
├── reports/
├── temp/
├── web/
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🧾 데이터셋 불러오기
#### 1. 'AI Hub' 홈페이지 로그인  [링크🔗](https://www.google.com)
#### 2. 민원(콜센터) 질의-응답 데이터 다운로드
- https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=98 링크
- 다운로드 클릭
  - 다운로드 할 데이터 체크
    - 022.민원(콜센터) 질의-응답 데이터/01.데이터  
        - 1.Training/라벨링데이터_220121_add/금융보험/민원(콜센터) 질의응답_금융보험_상품 가입 및 해지_Training.zip
        - 2.Validation/라벨링데이터_220121_add/금융보험/민원(콜센터) 질의응답_금융보험_상품 가입 및 해지_Validation.zip
- 선택 다운로드 클릭
- AI 허브 인공지능 학습용 데이터 다운로드 프로그램(INNORIX) 설치
- 다운로드 받은 zip파일 압축풀고 이름 변경 (Training.json, Validation.json)

#### 3. 프로젝트에 데이터셋 추가
- 프로젝트 루트에 data 폴더 생성 후 Training.json, Validation.json 추가

---

## 🧾 데이터셋 전처리

실행 경로 : preprocessing/  
터미널에서 경로 이동 :  
```bash
cd preprocessing
```
   
#### 1. 데이터셋 라벨링
- labeling.py 실행

#### 2. 분석용 데이터와 검수용 데이터 생성
- prepare_dataset.py 실행
- output 폴더에 analysis_dataset.json, review_dataset.csv 생성 되었는지 확인

####  3. review_dataset.csv에 정답라벨(worker_1, worker_2, worker_3) 작성
- r과 c로 입력(r: retain인 경우, c: churn_signal인 경우)

#### 4. 라벨 품질 검증
- sklearn_validate_labels.py 실행
- 검증 결과 확인 및 에러케이스 확인(output 폴더에 생성된 error_cases.csv 확인)

#### 5. 데이터 분석
- data_analysis.py 실행
- 실행하면, 입력 JSON을 정제·분석하고(중복, 라벨 분포, 길이, 토큰 길이 등) 보고서(PNG)와 CSV/JSON 결과를 생성
    - reports/analysis_result.json: 데이터 분석 결과와 생성된 csv 파일 경로 등을 json으로 정리한 파일
    - output/clean_train.csv: 최종 학습용 데이터셋
    - output/clean_valid.csv: 최종 검증용 데이터셋
    - output/duplicate_train.csv: 원본 train 내부에서 label+text 기준으로 중복된 모든 행(중복 그룹의 모든 항목 포함, keep=False)을 모아둔 파일
    - output/duplicate_valid.csv: 원본 valid 내부에서 label+text 기준으로 중복된 모든 행을 모아둔 파일
    - output/duplicate_cross.csv: train과 valid 양쪽에 동일한 (label, text) 쌍이 존재한 행들을 모은 파일

---

## 📝 모델 학습
#### 1. model/{모델명}.py 실행하여 모델을 학습
- preprocessing/output/ 경로에 생성된 실험 결과(.json)와 최적 모델 가중치(.pt) 파일 확인

---

## 데이터 전처리 결과 보고서
- [바로가기](reports/data-preprocessing/preprocessing_report.md)

---

## 인공지능 학습 결과서
- [바로가기](<reports/model-training-results/(Churn-Zero_AI)인공지능_학습_결과_보고서.pdf>)
