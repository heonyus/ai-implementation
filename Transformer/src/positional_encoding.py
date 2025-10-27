import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Positional Encoding
        
        Args:
            d_model: 모델의 임베딩 차원
            max_len: 최대 시퀀스 길이
        """
        super(PositionalEncoding, self).__init__()
        
        # 사인 함수에 사용할 각도 계산
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 짝수 위치에 sin 적용
        pe[:, 0::2] = torch.sin(position * div_term)
        # 홀수 위치에 cos 적용
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (seq_len, batch_size, d_model)
        Returns:
            (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0), :]
        return x 