# 12. 케라스(Keras)와 Sequential 모델, 그리고 텐서 Shape

[![Keras](https://img.shields.io/badge/Keras-3.x-D00000?logo=keras&logoColor=white)](https://keras.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![JAX](https://img.shields.io/badge/JAX-latest-2196F3)](https://jax.readthedocs.io)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org)

> 머신러닝 기초 학습 시리즈 — **12장**.
> Keras 고수준 API, Sequential 모델, 텐서 Shape에 대한 학습 메모를 **사실 검증(Fact-check)** 과 **보충 설명** 과 함께 정리합니다.

---

## 📑 목차

- [표기 규칙](#표기-규칙)
- [12.1 Keras란 무엇인가](#121-keras란-무엇인가)
- [12.2 백엔드 — 누가 실제 계산을 수행하는가](#122-백엔드--누가-실제-계산을-수행하는가)
- [12.3 핵심 데이터 구조 — Model과 Layer](#123-핵심-데이터-구조--model과-layer)
- [12.4 Sequential 모델 — 가장 단순한 선형 스택](#124-sequential-모델--가장-단순한-선형-스택)
- [12.5 학습 가중치는 어디에 존재하는가](#125-학습-가중치는-어디에-존재하는가)
- [12.6 텐서 Shape — `(4, 28, 28, 3)`의 의미](#126-텐서-shape--4-28-28-3의-의미)
- [✅ 검증 요약 (Self-Validation)](#-검증-요약-self-validation)
- [📚 참고 자료](#-참고-자료)

---

## 표기 규칙

| 표시 | 의미 |
| :---: | :--- |
| 🔴 | **오류 정정** — 원문이 사실과 다른 부분을 바로잡음 |
| 🟡 | **참고/모호** — 표현이 모호하거나 맥락에 따라 해석이 갈리는 부분 |
| 🟢 | **보충 설명** — 원문에는 없지만 이해를 돕기 위한 추가 |

---

## 12.1 Keras란 무엇인가

> **원문 메모** — *"파이썬으로 작성되었고, 고수준의 딥러닝 API"*

> [!NOTE]
> **검증 결과 : ✅ 정확함.**
> Keras는 순수 Python으로 작성된 **고수준(High-level) 딥러닝 API**입니다. 저수준 텐서 연산은 백엔드 프레임워크에 위임하고, 개발자는 **모델 구조 정의** 와 **학습 루프 작성** 에 집중할 수 있도록 설계되었습니다.

<details>
<summary>🟢 <b>보충 — Keras의 4가지 설계 철학</b></summary>

Keras의 슬로건은 **"Deep Learning for humans"** 이며, 다음 4가지 목표를 추구합니다.

1. **단순하고 일관된 인터페이스** 제공
2. 일반적인 사용 사례에서 **요구되는 행동 수를 최소화**
3. **점진적 복잡성 공개(Progressive disclosure of complexity)** — 시작은 쉽고, 필요할 때만 복잡한 기능 학습
4. **간결하고 가독성 높은 코드** 작성 지원

</details>

---

## 12.2 백엔드 — 누가 실제 계산을 수행하는가

> **원문 메모** — *"주로 사용되는 백엔드 → 텐서플로우"*

> [!WARNING]
> **검증 결과 : 🔴 시점에 따라 사실이 달라짐. 현재(2026년) 기준으로는 부정확.**
> *"주 백엔드 = TensorFlow"* 는 **Keras 2.x (`tf.keras`) 시절의 설명** 이며, **Keras 3 (2023년 11월 출시)** 부터는 multi-backend 체제로 회귀했습니다.

### 시기별 백엔드 지원 현황

| 시점 | 실제 상황 |
| :--- | :--- |
| **Keras 1.x ~ 2.x 초기** (2015 ~ 2017) | TensorFlow / Theano / CNTK 중 선택 가능 (multi-backend) |
| **`tf.keras` 시대** (2017 ~ 2023) | **TensorFlow 단일 체제** 로 통합. 이 시기 한정으로 원문이 맞음 |
| **Keras 3 시대** (2023년 11월 ~ 현재) | **JAX · TensorFlow · PyTorch** 모두 지원. 추론 전용으로 OpenVINO도 지원 |

### 정정된 표현

> ✅ **정정** : *"Keras 3는 JAX, TensorFlow, PyTorch를 모두 백엔드로 지원하며, 환경 변수 `KERAS_BACKEND`로 선택할 수 있다. 한국 강의 자료에서는 역사적 이유로 TensorFlow 백엔드를 기준으로 설명하는 경우가 많다."*

### 백엔드 전환 방법

```python
import os
os.environ["KERAS_BACKEND"] = "jax"   # "tensorflow" | "torch" | "jax"

import keras   # ← 환경 변수 설정 후에 import 해야 함
```

> [!IMPORTANT]
> `import keras` **이전에** 환경 변수를 설정해야 적용됩니다. 순서가 바뀌면 기본 백엔드로 로드됩니다.

### 백엔드별 특성 (실무 선택 기준)

| 백엔드 | 강점 | 약점 |
| :--- | :--- | :--- |
| **TensorFlow** | 프로덕션 생태계가 가장 풍부 (TF-Serving, TFLite, TF.js) | 일부 워크로드에서 JAX보다 느림 |
| **JAX** | XLA 컴파일 기반, GPU/TPU에서 최고 수준의 속도. 대규모 분산 학습에 유리 | 디버깅 난이도 ↑, 생태계가 상대적으로 작음 |
| **PyTorch** | 연구 커뮤니티 표준, 동적 그래프(Define-by-Run)로 디버깅 용이 | TPU 지원이 TF/JAX보다 약함 |

---

## 12.3 핵심 데이터 구조 — Model과 Layer

> **원문 메모** — *"핵심 데이터 구조 : 모델 레이어를 구성하는 방법을 나타냄"*

> [!CAUTION]
> **검증 결과 : 🟡 의미는 통하지만 표현이 모호함.** Keras의 핵심 추상화는 정확히 **두 가지** 입니다.

### 두 가지 핵심 추상화

#### 1️⃣ `Layer` — 가중치(State)와 연산(Computation)의 캡슐화 단위

- 가중치(weights)를 보유하며, `call()` 메서드에 연산 로직 정의
- 가중치는 **학습 가능(trainable)** 또는 **학습 불가(non-trainable)** 로 구분
- **재귀적 합성(Recursive composition)** 가능 — 레이어가 다른 레이어를 속성으로 가지면, 바깥 레이어가 안쪽 레이어의 가중치까지 자동 추적

#### 2️⃣ `Model` — 여러 `Layer`를 묶어 학습·추론 가능한 단위로 만든 객체

- `fit()`, `evaluate()`, `predict()`, `save()` 등의 학습/배포 메서드 제공

### 정정된 표현

> ✅ **정정** : *"Keras의 핵심 데이터 구조는 **`Layer`**(가중치+연산 캡슐화 단위)와 **`Model`**(레이어 묶음을 학습·추론 단위로 만든 객체) 두 가지이며, 모델은 이 레이어들을 어떻게 조합할지를 정의하는 컨테이너 역할을 한다."*

### 🟢 보충 — 모델을 정의하는 3가지 방식

| 방식 | 특징 | 적합한 경우 |
| :--- | :--- | :--- |
| **Sequential API** | 레이어를 순차적으로 쌓는 가장 단순한 방식 | 입력→출력 흐름이 한 갈래인 단순 MLP, 기본 CNN |
| **Functional API** | 그래프 형태로 레이어 연결. 다중 입출력, 분기, 잔차 연결 가능 | ResNet, U-Net, Multi-modal 모델 |
| **Model Subclassing** | `keras.Model`을 상속하여 `call()` 직접 구현. 가장 자유로움 | 커스텀 학습 로직, 동적 그래프 연구용 모델 |

---

## 12.4 Sequential 모델 — 가장 단순한 선형 스택

> **원문 메모** — *"가장 간단한 모델 유형 : Sequential 선형 스택 모델 / 레이어를 선형으로 쌓을 수 있는 신경망 모델"*

> [!NOTE]
> **검증 결과 : ✅ 정확함.**
> 공식 문서 정의 그대로입니다 — *"Sequential groups a linear stack of layers into a Model."*
> (Sequential은 레이어들의 선형 스택을 하나의 Model로 묶는다.)

### 🟢 보충 — Sequential 모델의 제약 (언제 쓰면 안 되는가)

다음 중 **하나라도 해당되면** Sequential을 쓸 수 없으며, **Functional API** 나 **Subclassing** 으로 가야 합니다.

- 레이어가 **다중 입력 또는 다중 출력** 을 가질 때
- **레이어 공유(layer sharing)** 가 필요할 때 (같은 레이어 인스턴스를 여러 위치에서 사용)
- **비선형 연결 그래프** 가 필요할 때 (ResNet의 skip connection, U-Net의 encoder-decoder concat 등)

### Sequential 작성 예시 (Keras 3 표준)

#### 방법 ① — 생성자에 레이어 리스트 전달

```python
import keras
from keras import layers

model = keras.Sequential([
    keras.Input(shape=(784,), name="input_layer"),         # 입력 shape 명시 (권장)
    layers.Dense(64, activation="relu", name="hidden_1"),  # 은닉층
    layers.Dense(10, activation="softmax", name="output"), # 출력층 (다중 분류)
], name="my_simple_mlp")

model.summary()
```

#### 방법 ② — `add()` 메서드로 점진적으로 쌓기

```python
import keras
from keras import layers

model = keras.Sequential(name="my_simple_mlp_added")
model.add(keras.Input(shape=(784,)))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

model.summary()
```

> [!TIP]
> `keras.Input(shape=...)` 을 **첫 줄에 두는 것이 권장 패턴** 입니다. 입력 shape을 명시하지 않으면 모델은 첫 데이터를 받기 전까지 *"빌드되지 않은(unbuilt)"* 상태로 남아 `model.summary()` 도 호출할 수 없습니다.

---

## 12.5 학습 가중치는 어디에 존재하는가

> **원문 메모** — *"학습 : 출력, 은닉층에 해당"*

> [!WARNING]
> **검증 결과 : 🔴 단어 선택이 부정확하여 의미가 성립하지 않음. 재해석 필요.**

원문 그대로 읽으면 *"학습(Training)이 출력층·은닉층에 해당한다"* 가 되어 의미가 통하지 않습니다. 메모의 흐름과 Sequential 모델의 구조를 고려하면 다음 둘 중 하나의 의미였을 것으로 추정됩니다.

| 추정 의도 | 정정된 표현 |
| :--- | :--- |
| ① Sequential에 추가하는 레이어가 무엇인가? | *"Sequential 모델에서 `add()` 로 추가되는 레이어는 **은닉층(Hidden Layer)** 과 **출력층(Output Layer)** 의 역할을 수행한다. 입력층은 별도의 학습 가중치를 갖지 않으며 `keras.Input()` 으로 shape만 선언한다."* |
| ② 학습되는 가중치는 어디에 있는가? | *"학습 가능한 가중치(trainable weights)는 **은닉층과 출력층** 에 존재한다. 입력층은 데이터 진입점일 뿐 학습 대상 파라미터가 없다."* |

> ✅ **정정** : 어느 쪽이든 핵심은 **"입력층은 학습 대상이 아니며, 학습 가중치는 은닉층과 출력층에만 존재한다"** 입니다.

### 🟢 보충 — 출력층 활성화 함수 매핑 (이전 7장과 연결)

| 문제 유형 | 출력층 활성화 함수 | 출력층 노드 수 |
| :--- | :--- | :--- |
| 회귀 (실수 예측) | Linear (항등) | 1 (또는 타깃 차원) |
| 이진 분류 | Sigmoid | 1 |
| 다중 분류 | Softmax | 클래스 개수 |

---

## 12.6 텐서 Shape — `(4, 28, 28, 3)`의 의미

> **원문 메모** — *"shape = (4, 28, 28, 3)  28×28 사이즈의 컬러(3)가 4장 있다"*

> [!NOTE]
> **검증 결과 : ✅ 정확함.**

### 각 축의 의미

| 축 인덱스 | 값 | 의미 |
| :---: | :---: | :--- |
| 0 (Batch) | 4 | **이미지 4장** (배치 크기) |
| 1 (Height) | 28 | 세로 픽셀 수 |
| 2 (Width) | 28 | 가로 픽셀 수 |
| 3 (Channels) | 3 | RGB **3채널 컬러** |

### 🟢 보충 — NHWC vs NCHW 포맷 (실무에서 자주 발생하는 함정)

원문의 `(4, 28, 28, 3)` 은 **NHWC** 포맷이며, TensorFlow / Keras의 **기본값** 입니다.

| 포맷 | 축 순서 | 동일 데이터의 shape | 사용처 |
| :---: | :--- | :---: | :--- |
| **NHWC** | (Batch, Height, Width, Channels) | `(4, 28, 28, 3)` | **TensorFlow / Keras 기본**, CPU에서 일반적으로 빠름 |
| **NCHW** | (Batch, Channels, Height, Width) | `(4, 3, 28, 28)` | **PyTorch 기본**, GPU(cuDNN)에서 일반적으로 빠름 |

> [!CAUTION]
> **실무 주의점**
> 같은 이미지 데이터라도 프레임워크마다 축 순서가 다르므로, **PyTorch ↔ TensorFlow 간 모델·가중치를 이식할 때 `permute()` 또는 `transpose()` 로 축을 바꿔주어야 합니다.**

### Keras `Conv2D` 입력에서 shape이 어떻게 흐르는가

```python
import keras
from keras import layers

# 입력 : (batch=4, height=28, width=28, channels=3)
inputs  = keras.Input(shape=(28, 28, 3))                                  # 배치 차원 제외하고 선언

x       = layers.Conv2D(32, kernel_size=3, activation="relu")(inputs)     # → (4, 26, 26, 32)
x       = layers.MaxPooling2D(pool_size=2)(x)                             # → (4, 13, 13, 32)
x       = layers.Flatten()(x)                                             # → (4, 5408)
outputs = layers.Dense(10, activation="softmax")(x)                       # → (4, 10)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()
```

> [!TIP]
> `keras.Input(shape=...)` 에는 **배치 차원을 제외** 한 `(28, 28, 3)` 만 적습니다. 배치 차원은 `fit()` 호출 시 동적으로 결정됩니다.

---

## ✅ 검증 요약 (Self-Validation)

| 항목 | 원문 표현 | 검증 결과 | 조치 |
| :--- | :--- | :---: | :--- |
| Keras 정의 | 파이썬으로 작성된 고수준 딥러닝 API | ✅ 정확 | 설계 철학 4가지 보충 |
| 주 백엔드 | 주로 사용되는 백엔드 → 텐서플로우 | 🔴 시점 의존 | Keras 3 multi-backend 체제 명시 |
| 핵심 데이터 구조 | 모델 레이어를 구성하는 방법 | 🟡 모호 | `Layer` + `Model` 두 가지로 명확화 |
| Sequential 정의 | 선형 스택 모델 | ✅ 정확 | 공식 문서 정의 인용 + 제약사항 보충 |
| "학습:출력,은닉층" | 단어 부정확 | 🔴 의미 불성립 | "학습 가중치는 은닉층·출력층에 존재"로 재정의 |
| Shape `(4,28,28,3)` | 28×28 컬러 4장 | ✅ 정확 | NHWC vs NCHW 포맷 차이 보충 |

### 검증자 코멘트

원문 메모는 **Keras 2.x 시절 한국어 강의 자료** 를 기반으로 한 것으로 추정됩니다. 따라서 *"주 백엔드 = TensorFlow"* 같은 부분은 **2023년 11월 이전 기준에서는 옳았던** 설명입니다. 강의가 틀린 것이 아니라, **현재(2026년) 시점에서는 업데이트가 필요한 정보** 라는 점을 인지하면 됩니다. Sequential의 정의나 텐서 Shape 해석은 시점에 무관하게 **여전히 모두 정확** 합니다.

---

## 📚 참고 자료

- [Keras 공식 문서 — The Sequential model](https://keras.io/guides/sequential_model/)
- [TensorFlow 공식 가이드 — Keras: The high-level API for TensorFlow](https://www.tensorflow.org/guide/keras)
- [Keras 3 공식 발표문 — Introducing Keras 3.0](https://keras.io/keras_3/)
- [Keras GitHub Repository](https://github.com/keras-team/keras)
- [Keras 2 → 3 마이그레이션 가이드](https://keras.io/guides/migrating_to_keras_3/)

---

<div align="center">

**[⬆ 맨 위로](#12-케라스keras와-sequential-모델-그리고-텐서-shape)**

</div>
