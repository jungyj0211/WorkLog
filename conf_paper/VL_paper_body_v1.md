
# 본문 초안 — 버추얼라인(LED 라인) + 영상처리 AI 안전 보조 시스템

> **목적**: 객체 검출·세그멘테이션·포즈추정·추적을 결합해 **위험 인지 → LED 라인 시각 경고**까지의 **엣지 실시간 시스템**을 설계·제안한다. 실험이 제한적인 상황을 고려해 **설계 근거, 수식, 평가계획, 배포 고려**를 충실히 기술한다.

---

## 1. 서론(Introduction)

스마트 산업/교통 현장에서는 보행자-차량(지게차 포함) 간 시야 교차, 야간/역광, 혼잡 등으로 인해 **사고 위험**이 상존한다. 기존의 페인트형 안전라인이나 정적 표시는 **가시성·가변성 한계**로 인해 동적 상황 반영이 어렵다. 본 연구는 **엣지 AI**로 실시간 위험을 인지하고, **LED 기반 버추얼라인**으로 즉각적인 시각 경고를 제공하는 **감지–판단–표시 일체형** 시스템을 제안한다.

본 연구의 기여는 다음과 같다. (i) YOLO(det/seg/pose)+ByteTrack 기반 **경량 지각 파이프라인**과 ROI/라인크로싱/TTC 규칙을 결합한 **시나리오 로직**을 정식화한다. (ii) **이벤트 점수 융합**과 **EVT(극값이론) 임계치**로 오경보율을 목표 수준으로 통제한다. (iii) RPi5 + Hailo-8 등 엣지 플랫폼 배포 지침과 **E2E 지연 ≤100 ms** 목표를 포함한 운영 KPI를 제시한다.

---

## 2. 시스템 개요(System Overview)

**하드웨어**: RGB 카메라(30–60 FPS, 1080p), 엣지 컴퓨팅(Raspberry Pi 5 + Hailo-8/8L 또는 Jetson Orin Nano), LED 라인 프로젝터/바(광도/패턴 제어), 선택적 경광등·버저. 옥외 설치를 고려해 **IP65 이상** 방진/방수와 방열 설계를 권장한다.

**소프트웨어**: (1) YOLO 계열 검출·세그·포즈(경량 n/s급), (2) ByteTrack 추적, (3) ROI/라인크로싱/TTC 기반 시나리오 로직, (4) EVT 임계치 기반 경보 레벨러, (5) LED 제어기(길이/밝기/패턴), (6) 로그/관제(선택).

**E2E 지연 목표**: 카메라 캡처 → AI 추론 → 로직 → LED 구동까지 **≤100 ms**. 프레임 드롭 없이 **≥30 FPS**를 목표로 한다.

> *Figure 1.* 시스템 블록도(제공 파일: `fig_vline_system.png`).  
> *Figure 2.* 골목 교차로 ROI/버추얼라인 개념도(제공 파일: `fig_vline_scenarios.png`).

---

## 3. 지각(Perception) 모듈

### 3.1 객체 검출·세그멘테이션·포즈
- **검출**: YOLOv11-n/s. 클래스: 사람, 자전거, 이륜차, 차량, 지게차(확장 가능).  
- **세그멘테이션**: 경량 YOLO-Seg(옵션) — 바닥/차선·보행자 공간 분리 향상.  
- **포즈추정**: YOLOv11-pose 또는 RTM-Pose — **넘어짐/주시 방향** 등 상태 추정에 활용.

### 3.2 추적(ByteTrack)
탐지 박스를 입력, 프레임 간 ID를 유지하면서 속도·방향을 추정한다. 오클루전 환경에서도 **ID 안정성**을 확보하기 위해 iou + score 기반 결합을 사용한다.

### 3.3 경량화·성능
- **모델 경량화**: INT8 양자화(엣지 NPU), 입력 축소(640→512), 멀티스케일 제한.  
- **추적 빈도 조절**: 고정 30 FPS 추정, 상황에 따라 2프레임/스텝 업데이트 등으로 지연 감소.

---

## 4. 시나리오 로직(Scenario Logic)

### 4.1 ROI/라인 정의
현장별로 **폴리곤 ROI**(안전구역·차량 동선·교차 시야 구역)를 사전 정의하고, **버추얼라인 위치**를 지도 좌표(픽셀)로 설정한다. 학습 없이도 규칙 기반 이벤트를 결합할 수 있다.

### 4.2 라인 크로싱 이벤트
트랙의 대표점(예: 바운딩 박스 하단 중앙)을 \(p_t\)라 하자. 프레임 \(t-1 \to t\) 사이에 가상 라인 \(\ell\)을 교차하면 라인-크로싱 이벤트 \(E_{\text{lc}}=1\)로 기록한다.

\[
E_{\text{lc}} = \mathbb{1}\big(\text{sign}(d_{\ell}(p_{t-1})) \neq \text{sign}(d_{\ell}(p_t))\big)
\]
여기서 \(d_{\ell}(\cdot)\)은 점과 라인의 부호 있는 거리이다.

### 4.3 근접/충돌 위험(TTC)
카메라 내부보정과 평면 가정하에 픽셀 위치를 거리로 근사하면, 상대 속도 \(v_{\text{rel}}\)와 현재 거리 \(d\)로 **충돌까지 시간(Time-to-Collision)**을
\[
\text{TTC} = \frac{d}{\max(\epsilon, v_{\text{rel}}^{+})}
\]
로 정의한다(\(v_{\text{rel}}^{+}\): 접근 성분, \(\epsilon\): 0-division 방지). 작은 TTC일수록 위험이 크다.

### 4.4 포즈 기반 위험(옵션)
사람의 기울기/넘어짐·주시각 등에서 위험 가중치 \(r_{\text{pose}}\in[0,1]\)를 산출한다.

---

## 5. 경보 점수 융합 및 EVT 임계치

### 5.1 위험 점수 융합
객체 \(i\)에 대해 다음과 같이 위험 점수를 정의한다.
\[
S_i = w_1(1-\hat{c}_i) + w_2(1-\text{trackstab}_i) + w_3\ \phi(\text{TTC}_i) + w_4\ r_{\text{pose},i} + w_5\ \mathbb{1}(E_{\text{lc},i})
\]
- \(\hat{c}_i\): 검출 신뢰도(높을수록 감지 확실 → 위험 가중은 반대로 \(1-\hat{c}\)).  
- \(\text{trackstab}\): ID 유지·속도 안정 지표(0–1).  
- \(\phi(\text{TTC})\): 예: \(\phi(x)=\min(1, \frac{\tau}{x})\) 형태(임계 \(\tau\)).  
- \(r_{\text{pose}}\): 넘어짐/주시각 위험(옵션).  
- \(\mathbb{1}(E_{\text{lc}})\): 라인-크로싱 이벤트.

### 5.2 EVT(극값이론) 임계치
운영기간 정상 구간에서 \(S\)의 상위 \(q\%\) 초과치를 집합 \(Y=\{S-\mu\ |\ S>\mu\}\)로 만들고, **GPD(Generalized Pareto Distribution)**에 피팅한다.
임계치는 목표 거짓양성률 \(\alpha\)에 맞춰
\[
T = \mu + \frac{\sigma}{\xi}\Big(\alpha^{-\xi}-1\Big)
\]
로 산출한다(\(\sigma,\xi\): GPD 파라미터). \(S\ge T\)이면 경보.  
오경보 제어의 **통계적 근거**를 제공하며, 환경 변화에 따라 주기적으로 업데이트한다.

> *Figure 3.* EVT 임계치 개념도(제공 파일: `fig_evt_threshold.png`).

---

## 6. 평가 계획(Evaluation Plan)

### 6.1 데이터셋/프로토콜
- **데이터**: COCO(사람/자전거/차량) 기반 프리트레인 + 내부 산업/골목 영상 일부(익명화/마스킹).  
- **프로토콜**: 주·야간/역광/우천·혼잡 등 시나리오 균형 샘플링. ROI/라인 사전 정의.

### 6.2 지표(KPI)
- **지각**: mAP@50/95(검출·세그), Keypoint AP(포즈), MOTA/MOTP(추적).  
- **이벤트**: 라인-크로싱/근접 경보 **Precision/Recall**, **경보 리드타임**(사건 전), **시간당 경보수(FPR)**.  
- **시스템**: **E2E 지연(ms)**, **FPS**, **가동률(availability)**.

> *Table 1.* 버추얼라인 평가 지표 템플릿(제공 파일: `tab_vline_eval.csv`).

### 6.3 실험이 제한된 경우
- 합성/리플레이 영상으로 **리드타임·지연**을 측정.  
- EVT 임계치는 정상 로그로 사전 피팅, \(\alpha\)를 변경하며 민감도 스윕.

---

## 7. 배포·안전·윤리(Deployment, Safety, Ethics)

- **프라이버시**: 얼굴/번호판 엣지 모자이크, 원본 외부 반출 금지.  
- **기본 안전 상태(Fail-safe)**: 감지 실패/모델 다운 시 **경계색 상시 점등**.  
- **자체 진단**: 프레임 드롭·추론 오류·LED 구동 오류 감지→재시작/격리.  
- **하드웨어**: IP65 이상, 난반사 억제 위해 투사각·광도·패턴 최적화.  
- **규제**: 교통 신호 대체/보완 용도는 관할 규정 준수 및 인증 계획 별도 기술.

---

## 8. 한계와 향후 과제(Limitations & Future Work)

- 도메인 시프트(현장 교체) 시 성능 저하 → **소규모 현장별 리튜닝** 또는 DA 도입.  
- 악천후/오염/강한 난반사로 감지 성능 하락 가능 → **IR 보조/HDR** 옵션.  
- TTC 근사 정확도는 카메라 보정·기하 가정에 의존 → **멀티뷰/깊이 센서** 확장.

---

## 9. 의사코드(Pseudocode)

```python
for frame in camera.stream():
    dets = yolo.detect(frame)                 # boxes, scores, classes
    tracks = bytetrack.update(dets)           # ID, velocity, direction
    pose = yolo_pose.keypoints(frame)         # optional
    events = rules.evaluate(tracks, roi_polys, vlines)   # line-crossing, proximity
    scores = fuse(dets, tracks, pose, events)            # S_i
    alarm = evt_threshold(scores)                         
    vline.render(alarm.level)                  # LED control
    log.save(scores, alarm, events, latency())


