# 🏥 난임 환자 임신 성공 여부 예측 AI

> DACON · 오즈코딩스쿨 AI 해커톤  
> **팀명**: 6아 하고싶조 | **팀원**: 정정화 · 권순현 · 김윤기  
> **기간**: 2025.04.23 ~ 2025.05.04 (12일)  
> **최종 Public LB ROC-AUC**: `0.7414`

---

## 📌 프로젝트 개요

난임 환자의 임상 기록 데이터를 기반으로 **임신 성공 여부를 예측하는 이진 분류 모델**을 구축했습니다.  
단순 통계 기반 접근이 아닌, **의학 논문 5건을 사전에 검토하고 임상 가설을 먼저 세운 뒤 데이터로 검증**하는 방식으로 파생변수를 설계했습니다.

| 항목 | 내용 |
|------|------|
| 대회 | DACON · 오즈코딩스쿨 AI 해커톤 |
| 문제 유형 | 이진 분류 (임신 성공: 1 / 실패: 0) |
| 평가 지표 | ROC-AUC |
| 학습 데이터 | 256,351 rows / 69 features |
| 평가 데이터 | 90,067 rows / 68 features |
| 타깃 분포 | 성공 25.83% / 실패 74.17% (불균형) |
| 최종 모델 | CatBoost · 5-Fold · Seed 5개 평균 |

---

## 👤 담당 역할 (권순현)

- **프로젝트 개요 설계**: 데이터 사양 분석 및 평가 전략 수립
- **전처리**: 결측치 처리, 횟수형 문자열 수치화, 나이 구간 수치 변환, 데이터 누수 방지
- **피처 엔지니어링**: 시술 유형 파생변수, 조합 피처 설계
- **의학 도메인 인사이트**: 학술 자료 5건 분석 → 임상 가설 수립 → 파생변수 근거 도출
- **모델 검증 및 선택**: CatBoost vs XGBoost vs LightGBM 비교 실험, Seed Averaging 전략 수립
- **코드 취합 및 최종 제출 파일 생성**

---

## 🔬 의학 도메인 기반 접근 (핵심 차별점)

논문을 먼저 읽고 가설을 세운 뒤 데이터로 검증하는 **"근거 기반 피처 엔지니어링"** 을 적용했습니다.

| REF | 근거 | 적용한 파생변수 |
|-----|------|----------------|
| REF 01 | 39세를 기점으로 임신 성공 오즈비 3.363배 격차 (연세대 석사논문) | 연령 3단계 분류 변수 |
| REF 02 | AFC + 배아 이식일 결합 효과 (43편 systematic review) | `feat_embryo_transfer_rate` |
| REF 03 | 자궁내막증 유무가 착상률에 영향 (분당서울대병원) | 동반 질환 변수 |
| REF 04 | 인공수정 3~4회 이후 누적 성공률 정체 (서울아산병원) | 시술 횟수 임계 기반 변수 |
| REF 05 | 배반포(5~6일) vs 분할기(2~3일) 이식 성공률 차이 (ASRM 2023) | `is_배반포이식` / `is_초기배아이식` |

> **결과**: 18개 파생변수 중 8개가 최종 모델 상위 17위 이내 진입 → **44% 적중률**

---

## ⚙️ 전처리 파이프라인

### 1. 결측치 처리 전략

```python
# 이진 플래그 컬럼 → 0으로 대체 (해당 없음을 의미)
fill_zero_cols = [
    '단일 배아 이식 여부', '착상 전 유전 검사 사용 여부',
    '동결 배아 사용 여부', '신선 배아 사용 여부', ...
]
pregnancy_data[fill_zero_cols] = pregnancy_data[fill_zero_cols].fillna(0)

# 수치형 컬럼 → Train 기준 평균/중앙값으로 대체 (데이터 누수 방지)
train_means = X_train[mean_cols].mean()
train_medians = X_train[median_cols].median()

X_val[mean_cols] = X_val[mean_cols].fillna(train_means)   # Train 기준값 사용
X_val[median_cols] = X_val[median_cols].fillna(train_medians)
```

### 2. 횟수형 문자열 → 수치 변환

```python
# "6회 이상" → 6, "3회" → 3 형태로 변환
def count_to_num(x):
    if pd.isna(x): return np.nan
    s = str(x)
    if '이상' in s: return 6
    digits = ''.join(ch for ch in s if ch.isdigit())
    return int(digits) if digits else np.nan
```

### 3. 나이 구간 → 중간값 수치화

```python
age_map = {
    '만18-34세': 26, '만35-37세': 36, '만38-39세': 38.5,
    '만40-42세': 41, '만43-44세': 43.5, '만45-50세': 47.5,
    '알 수 없음': -1
}
```

---

## 🧬 피처 엔지니어링

### 핵심 파생변수

```python
# 조합 피처 (나이 × 임상 수치)
df['나이x이식'] = df['시술 당시 나이'] * df['이식된 배아 수']
df['나이x배아'] = df['시술 당시 나이'] * df['총 생성 배아 수']
df['남은_배아'] = df['총 생성 배아 수'] - df['이식된 배아 수']
df['배아_생성효율'] = df['총 생성 배아 수'] / (df['수집된 신선 난자 수'] + 1e-5)

# 시술 유형 파생 (텍스트 → 이진 플래그)
for word in ['ICSI', 'IVF', 'BLASTOCYST', 'AH', 'Unknown']:
    df[f'시술_{word}'] = df['특정 시술 유형'].str.contains(word, na=False).astype(int)
```

---

## 🤖 모델링

### CatBoost 선택 근거

| 모델 | CV ROC-AUC | 선택 여부 |
|------|-----------|---------|
| **CatBoost** | **0.74009** | ✅ 최종 선택 |
| XGBoost | 0.73967 | — |
| LightGBM | 0.73960 | — |

**CatBoost 채택 이유:**
- 범주형 변수 자동 처리 → LabelEncoder 불필요, **데이터 누수 방지**
- GPU 학습 + Early Stopping으로 학습 효율 우수
- `auto_class_weights='Balanced'`로 클래스 불균형 자동 처리

### Seed Averaging 전략

```python
seeds = [42, 0, 1, 2, 3]
all_oof = np.zeros(len(X))
all_test = np.zeros(len(X_test))

for seed in seeds:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    # ... 5-Fold 학습
    all_oof += oof / len(seeds)
    all_test += test_pred / len(seeds)

# 단일 시드: 0.74008 → 시드 5개 평균: 0.74041
```

---

## 📊 실험 결과

| 실험 | 피처 수 | CV ROC-AUC |
|------|---------|-----------|
| v3 · 피처 엔지니어링 기본 | 78 | 0.7398 |
| v6 · 논문 기반 18개 추가 | 96 | 0.7399 |
| v7 · 상호작용 피처 추가 | 104 | 0.7397 |
| v8 · Target Encoding | 104 | 0.7398 |
| v9 · 통계 피처 | 108 | 0.7396 |
| CatBoost 단독 (노이즈 제거) | 120 | 0.74009 |
| 3모델 가중 앙상블 | 120 | 0.74030 |
| **CatBoost Seed 5개 평균** | **120** | **0.74041** |
| **최종 Public LB** | — | **0.74157** |

---

## 🏆 최종 모델 Top 8 피처 중요도

| 순위 | 피처명 | 중요도 | 설명 |
|------|--------|--------|------|
| 1 | `feat_embryo_transfer_rate` | 24.95 | 배아 이식 비율 · **자체 설계 파생변수** |
| 2 | `이식된 배아 수` | 23.29 | 이식한 배아 개수 · 원본 변수 |
| 3 | `ivf_transfer_ratio` | 12.01 | IVF 이식 비율 · **자체 설계 파생변수** |
| 4 | `난자 출처` | 4.36 | 본인/기증 난자 구분 |
| 5 | `수집된 신선 난자 수` | 4.03 | 신선 난자 채취 개수 |
| 6 | `age_num` | 2.69 | 시술 당시 나이 (수치 변환) |
| 7 | `배아 이식 경과일` | 2.46 | 배반포(5일+) vs 초기 이식 |
| 8 | `ivf_freeze_ratio` | 2.33 | IVF 동결 비율 · **자체 설계 파생변수** |

> Top 5 중 **4개가 자체 설계 파생변수** → 의학 도메인 기반 피처 엔지니어링의 효과

---

## 💡 핵심 인사이트

1. **도메인 지식 + 데이터** — 학술 자료 5건 → 의학적 가설 → 데이터 검증. 단순 통계가 아닌 근거 기반 분석으로 변수의 해석력과 일반화력을 동시에 확보
2. **44% 적중률** — 18개 파생변수 중 8개가 모델 상위 17위 이내 진입. 의학적 직관이 모델 학습에 효과적이라는 정량적 근거
3. **정보 재분배 vs 진짜 새 정보** — Target Encoding, 통계 피처 등 같은 정보를 다르게 표현하는 피처는 효과 미미. 진짜 새 정보를 만드는 것이 핵심
4. **데이터 천장 돌파** — CV 0.7396~0.7399 범위의 단일 모델 한계를 Seed Averaging으로 돌파 (+0.0006)

---

## 📁 폴더 구조

```
├── README.md
├── notebooks/
│   ├── 01_EDA_전처리.ipynb       # EDA 및 결측치 처리
│   └── 02_모델링_앙상블.ipynb    # 모델 학습 및 Seed Averaging
└── presentation/
    └── 6조_난임임신예측AI_발표자료.pptx
```

---

## 🛠️ 기술 스택

![Python](https://img.shields.io/badge/Python-3.10-blue)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2-yellow)
![Pandas](https://img.shields.io/badge/Pandas-2.0-lightgrey)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.3-orange)

- **언어**: Python 3.10
- **모델**: CatBoost, XGBoost, LightGBM
- **분석**: Pandas, NumPy, Scikit-learn
- **환경**: Google Colab (GPU)
