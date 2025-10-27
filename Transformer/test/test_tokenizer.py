import sys
import os

# 상위 디렉토리 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer import BPETokenizer

def test_tokenizer():
    # 테스트 데이터
    corpus = [
        "안녕하세요",
        "안녕하세요 반갑습니다",
        "코드 작성 중입니다",
        "트랜스포머 모델 구현"
    ]
    
    print("=== BPE 토크나이저 테스트 ===\n")
    
    # 토크나이저 생성 및 학습
    tokenizer = BPETokenizer()
    print("데이터 학습 중...")
    tokenizer.train(corpus, num_merges=50)
    
    print(f"\n병합 규칙 수: {len(tokenizer.merges)}")
    print(f"Vocab 크기: {len(tokenizer.vocab)}")
    
    # 인코딩 테스트
    test_text = "안녕하세요 반갑습니다"
    print(f"\n원본 텍스트: '{test_text}'")
    
    encoded = tokenizer.encode(test_text)
    print(f"인코딩 결과: {encoded}")
    
    # 디코딩 테스트
    decoded = tokenizer.decode(encoded)
    print(f"디코딩 결과: '{decoded}'")
    
    # 추가 테스트
    print("\n=== 추가 테스트 ===")
    test_cases = [
        "코드 작성",
        "모델 구현",
        "안녕하세요"
    ]
    
    for text in test_cases:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"'{text}' -> {encoded} -> '{decoded}'")

if __name__ == "__main__":
    test_tokenizer()
