# Transformer Implementation with PyTorch

PyTorch로 구현한 Transformer 모델입니다.

## 프로젝트 구조

```
Transformer/
├── src/
│   ├── __init__.py              # 패키지 초기화
│   ├── attention.py              # Scaled Dot-Product Attention & Multi-Head Attention
│   ├── positional_encoding.py   # Positional Encoding
│   ├── tokenizer.py             # BPE Tokenizer
│   └── transformer.py           # Transformer Model (Encoder, Decoder, Full Model)
├── test/
│   ├── test_tokenizer.py        # 토크나이저 테스트
│   └── test_transformer.py      # Transformer 모델 테스트
├── venv/                        # 가상환경
└── README.md                    # 이 파일

```

## 주요 기능

### 1. Attention Mechanism
- **Scaled Dot-Product Attention**: 기본 attention 메커니즘
- **Multi-Head Attention**: 여러 head를 사용한 attention

### 2. Positional Encoding
- 사인/코사인 기반 위치 인코딩
- 시퀀스의 순서 정보 제공

### 3. BPE Tokenizer
- Byte Pair Encoding 기반 토크나이저
- 텍스트를 토큰으로 변환/복원

### 4. Transformer Model
- **Encoder**: Self-attention + Feed Forward
- **Decoder**: Self-attention + Cross-attention + Feed Forward
- Masking 지원

## 사용 방법

### 1. 설치

```bash
# 가상환경 활성화 (Windows)
venv\Scripts\activate

# 또는 Linux/Mac
source venv/bin/activate
```

### 2. 테스트 실행

#### 토크나이저 테스트
```bash
python test/test_tokenizer.py
```

#### Transformer 모델 테스트
```bash
python test/test_transformer.py
```

## 코드 예시

### Transformer 모델 사용

```python
from src.transformer import Transformer

# 모델 생성
model = Transformer(
    src_vocab_size=1000,      # Source vocabulary 크기
    tgt_vocab_size=1000,      # Target vocabulary 크기
    d_model=512,              # 모델 차원
    num_heads=8,              # Attention head 수
    num_encoder_layers=6,     # Encoder 레이어 수
    num_decoder_layers=6,     # Decoder 레이어 수
    d_ff=2048,                # Feed Forward 차원
    max_len=5000,             # 최대 시퀀스 길이
    dropout=0.1               # Dropout 비율
)

# Forward pass
src = torch.randint(1, 1000, (batch_size, src_len))
tgt = torch.randint(1, 1000, (batch_size, tgt_len))
output = model(src, tgt)  # (batch_size, tgt_len, tgt_vocab_size)
```

### BPE Tokenizer 사용

```python
from src.tokenizer import BPETokenizer

# 토크나이저 생성 및 학습
tokenizer = BPETokenizer()
corpus = ["안녕하세요", "반갑습니다"]
tokenizer.train(corpus, num_merges=50)

# 인코딩
encoded = tokenizer.encode("안녕하세요 반갑습니다")

# 디코딩
decoded = tokenizer.decode(encoded)
```

## 하이퍼파라미터

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `d_model` | 모델 차원 | 512 |
| `num_heads` | Attention head 수 | 8 |
| `num_encoder_layers` | Encoder 레이어 수 | 6 |
| `num_decoder_layers` | Decoder 레이어 수 | 6 |
| `d_ff` | Feed Forward 차원 | 2048 |
| `dropout` | Dropout 비율 | 0.1 |

## 참고 자료

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer Paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual Explanation

## 라이선스

MIT License
