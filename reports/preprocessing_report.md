
# 데이터 전처리 결과서

## 1. 데이터 개요

원천 데이터 보호 정책에 따라 원문 데이터는 공개하지 않음.

|항목|값|
|---|---|
|전체 데이터|98177|
|Train|87265|
|Validation|10912|
|Unique Dialog|4905|

---

## 2. Label 분포

| label        |   count |   ratio(%) |
|:-------------|--------:|-----------:|
| retain       |   81675 |      83.19 |
| churn_signal |   16502 |      16.81 |

---

## 3. 결측치 분석

| column    |   missing_count |
|:----------|----------------:|
| dataset   |               0 |
| dialog_id |               0 |
| seq       |               0 |
| label     |               0 |
| text      |               0 |

---

## 4. 중복 데이터

|항목|값|
|---|---|
|중복 행 수|0|

---

## 5. Dialog 분석

|항목|값|
|---|---|
|평균 Turn|20.02|
|중앙값 Turn|20.0|
|최소 Turn|1|
|최대 Turn|49|

---

## 6. 문자 길이 분석

|항목|값|
|---|---|
|평균|21.93|
|중앙값|19.0|
|최소|0|
|최대|356|
|P90|41.0|
|P95|50.0|
|P99|76.0|

---

## 7. 토큰 길이 분석

Tokenizer : KLUE/BERT SentencePiece

|항목|값|
|---|---|
|평균|12.43|
|중앙값|11.0|
|최소|0|
|최대|176|
|P90|22.0|
|P95|28.0|
|P99|41.0|


![image](token_length_distribution.png)

---

## 8. Train / Validation 분포 비교

|항목|PSI|
|---|---|
|Token Length|0.0046|

PSI 기준

- PSI < 0.1 : 안정
- 0.1 ~ 0.25 : 주의
- PSI > 0.25 : 분포 차이 큼

---

## 9. Sequence Length 결정

|항목|값|
|---|---|
|추천 Max Length|28|
|초과 샘플 수|4451|

선정 기준 : P95

---

## 10. Vocabulary 정보

|항목|값|
|---|---|
|Tokenizer|KLUE/BERT|
|Tokenization|SentencePiece|
|Vocabulary Size|32000|

---

## 11. 데이터 품질 점검

|항목|값|
|---|---|
|Empty Text|58|
|Duplicate Rows|0|
|Imbalance Ratio|4.95|

---

## 12. 모델 입력 사양

공통 Tokenizer

KLUE SentencePiece

사용 모델

1. LSTM
2. Text CNN
3. Transformer Encoder (Scratch)
4. KLUE-BERT Fine-tuning

---

## 13. 생성 산출물

- preprocessing_report.md
- token_length_distribution.png

