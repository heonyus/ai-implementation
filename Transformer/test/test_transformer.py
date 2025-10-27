import sys
import os
import torch

# 상위 디렉토리 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transformer import Transformer

def test_transformer():
    print("=== Transformer 모델 테스트 ===\n")
    
    # 하이퍼파라미터 설정
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 512
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    d_ff = 2048
    max_len = 100
    dropout = 0.1
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 12
    
    # 모델 생성
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout=dropout
    )
    
    print(f"모델 생성 완료!")
    print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # 더미 데이터 생성
    src = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))
    
    print(f"\n입력 shape:")
    print(f"Source: {src.shape}")
    print(f"Target: {tgt.shape}")
    
    # Forward pass
    output = model(src, tgt)
    
    print(f"\n출력 shape: {output.shape}")
    print(f"예상 shape: (batch_size={batch_size}, tgt_seq_len={tgt_seq_len}, tgt_vocab_size={tgt_vocab_size})")
    
    # Backward pass
    loss = output.sum()
    loss.backward()
    
    print("\nBackward pass 성공!")
    print(f"Loss: {loss.item():.4f}")
    
    print("\n=== 테스트 성공! ===")

if __name__ == "__main__":
    test_transformer()
