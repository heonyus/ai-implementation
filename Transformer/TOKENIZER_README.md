# BPETokenizer 사용 가이드

## 개요
BPE(Byte Pair Encoding) 기반의 토크나이저 구현입니다. 텍스트를 토큰으로 변환하거나, 토큰을 다시 텍스트로 복원할 수 있습니다.

## 주요 기능

### 1. `train(corpus, num_merges=1000)`
BPE 토크나이저를 학습합니다.

**매개변수:**
- `corpus`: 학습할 텍스트 리스트 (예: `["안녕하세요", "반갑습니다"]`)
- `num_merges`: 병합 횟수 (기본값: 1000)

**예시:**
```python
tokenizer = BPETokenizer()
corpus = ["안녕하세요", "안녕하세요 반갑습니다"]
tokenizer.train(corpus, num_merges=50)
```

### 2. `encode(text)`
텍스트를 토큰 ID 리스트로 변환합니다.

**매개변수:**
- `text`: 변환할 텍스트

**반환값:** 토큰 ID 리스트

**예시:**
```python
encoded = tokenizer.encode("안녕하세요")
print(encoded)  # [30]
```

### 3. `decode(token_ids)`
토큰 ID 리스트를 텍스트로 복원합니다.

**매개변수:**
- `token_ids`: 토큰 ID 리스트

**반환값:** 복원된 텍스트

**예시:**
```python
decoded = tokenizer.decode([30])
print(decoded)  # "안녕하세요"
```

## 완전한 사용 예시

```python
from src.tokenizer import BPETokenizer

# 1. 토크나이저 생성
tokenizer = BPETokenizer()

# 2. 학습 데이터 준비
corpus = [
    "안녕하세요",
    "안녕하세요 반갑습니다",
    "코드 작성 중입니다",
    "트랜스포머 모델 구현"
]

# 3. 학습
tokenizer.train(corpus, num_merges=50)

# 4. 인코딩
text = "안녕하세요 반갑습니다"
encoded = tokenizer.encode(text)
print(f"인코딩: {encoded}")

# 5. 디코딩
decoded = tokenizer.decode(encoded)
print(f"디코딩: {decoded}")
```

## 특수 토큰

- `<PAD>`: 패딩 토큰 (모델 입력 길이를 맞추기 위해 사용)
- `<UNK>`: 알 수 없는 토큰 (학습 시 본 적 없는 단어가 나올 때 사용)

## 내부 데이터 구조

- `self.vocab`: 토큰 → ID 매핑
- `self.merges`: BPE 병합 규칙 리스트
- `self.token_to_id`: 토큰 → ID 사전
- `self.id_to_token`: ID → 토큰 사전

## 테스트 방법

```bash
cd Transformer
python test/test_tokenizer.py
```

## 주의사항

1. `train()` 메서드를 먼저 호출해야 `encode()`와 `decode()`를 사용할 수 있습니다.
2. `num_merges` 값이 클수록 더 많은 병합이 일어나 vocab 크기가 커집니다.
3. 한글의 경우 문자 단위로 처리되므로 한국어에 특화된 후처리가 필요할 수 있습니다.
