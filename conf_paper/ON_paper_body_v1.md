
# 본문 초안 — 비지도 예지보전(SMPS: FFT/초음파·전압·전류)

> **목적**: 라벨 희소 환경의 SMPS에서 FFT/초음파·전압·전류를 이용해 **특징 기반 이상탐지(경로 A)**와 **예측-잔차 기반 이상탐지(경로 B)**를 **병렬 융합**하고, **EVT 임계치**와 **Health Index(HI)/준-비지도 RUL**로 운영 가능성을 제시한다.

---

## 1. 서론
산업 현장 SMPS/LED 드라이버는 오랜 수명과 다양한 운전조건(부하·온도·입력 전원)으로 **라벨 수집 비용**이 매우 높다. 이 연구는 **비지도**로 조기 이상 경보를 제공하고, 점수를 지표화해 유지보수 결정을 지원하는 프레임워크를 제안한다.

기여: (i) 도메인지식 기반 **주파수 특징 + 통계/격리 모델**과 (ii) **TCN/1D-CNN 예측-잔차**를 이중화하여 강건성을 확보. (iii) **EVT**로 목표 FPR을 보장하고, **EWMA-HI**와 **등위회귀 기반 준-RUL** 산출 절차를 제시.

---

## 2. 데이터와 전처리
- **채널**: FFT/초음파, 전압, 전류(동일 윈도 기준 정렬).  
- **윈도**: 전원 주파수/스위칭 주파수의 정수배 길이.  
- **조건 태깅**: 부하율·온도·입력전압. 조건별 표준화(z-score).

> *Figure 1.* 파이프라인 개요(`fig_pdm_pipeline.png`).

---

## 3. 경로 A — 도메인 특징 기반 이상탐지
### 3.1 특징
- **기본파/고조파** \(A_1, A_2, A_3\), **THD** \(=\sqrt{\sum_{n\ge2}A_n^2}/A_1\)  
- **대역에너지 비율**(스위칭 대역/전체), **센트로이드·커토시스**, **사이드밴드 비율**  
- 시간영역: **RMS, 크레스트 팩터**, 전압 **sag** 깊이/지속, **샘플 엔트로피**

> *Figure 2.* FFT 스펙트럼 예시와 특징 지점(`fig_feature_spectrum.png`).  
> *Table 1.* 특징 목록과 정의(`tab_features.csv`).

### 3.2 모델
- **PCA + Hotelling T²/SPE**, **Isolation Forest**, **LOF/One-Class SVM** 중 하나 또는 앙상블.  
- 상태 다양성 보완: **조건별 모델 분리** 또는 **HDBSCAN 클러스터-가딩**.

---

## 4. 경로 B — 예측-잔차 기반 이상탐지
- **모델**: **TCN** 또는 **1D-CNN**으로 단기 예측 \(\hat{x}_{t+1:t+H}\).  
- **점수**: \(S_{\text{pred}}=\|\mathbf{x}-\hat{\mathbf{x}}\|_1\) 또는 Huber.  
- **불확실성(옵션)**: MC Dropout 분산으로 잔차 보정.

---

## 5. 점수 융합과 EVT 임계치
총점수 \(S=\alpha S_{\text{feat}} + (1-\alpha) S_{\text{pred}}\). 정상 로그의 상위 \(q\%\) 초과치를 GPD에 피팅하여, 목표 거짓양성률 \(\alpha_{\text{FPR}}\)에 대한 **임계치** \(T\)를 얻는다.

> *Figure 3.* EVT 임계치 개념(`fig_evt_threshold.png`).

---

## 6. Health Index(HI)와 준-RUL
- **EWMA**: \(HI_t=\lambda S_t+(1-\lambda)HI_{t-1}\) (초기 \(HI_0=S_0\)).  
- **스케일링**: 0–1로 정규화(상한=임계치 부근).  
- **준-RUL**: HI의 장기 상승을 등위회귀로 근사, 목표 HI\* 도달까지 잔여시간 추정.

> *Figure 4.* 점수→EWMA(HI) 평활화 예시(`fig_hi_ewma.png`).

---

## 7. 평가 계획(라벨 제한 가정)
- **지표**: 시간당 경보수(FPR), 평균 **리드타임**, 점수 **분산**, PR-AUC(부분 라벨), **지연**.  
- **프로토콜**: 합성 이상(2f/3f 주입, sag 삽입), 정상 로그 기반 EVT 피팅, 민감도 스윕.

> *Table 2.* PdM 평가 지표 템플릿(`tab_pdm_metrics.csv`).

---

## 8. 배포 고려
- **엣지**: RPi/Hailo에서 특징 계산 + IF/PCA 또는 소형 TCN, 정수 양자화.  
- **MCU**: 특징 + IF 조합(고정소수점). 로그: 윈도 점수/HI/조건 태그.

---

## 9. 한계와 향후 과제
- 도메인 시프트 대응(현장별 보정), 악조건 노이즈, TTC류 사건 연동 필요.  
- 후속: **변화점 탐지**·**대비학습 임베딩(TS2Vec)** 병행, 준지도 라벨링.

---

## 10. 의사코드
```python
for window in stream.signals():
    feats = extract_features(window)          # A1,A2,A3,THD,band ratios, ...
    s_feat = anomaly_score_IF_PCA(feats)
    pred = tcn.predict(window)
    s_pred = l1(window - pred)
    S = alpha*s_feat + (1-alpha)*s_pred
    alarm = evt_threshold(S)                   # fitted on normal logs
    HI = ewma(S)                               # health index
    log.save(S, HI, alarm)

