import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    """
    def __init__(self, dropout=0.1):
        super().__init__()
        # self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1) # query의 shape: (batch_size, seq_len, d_k)

        # query, key의 행렬곱으로 attention score 계산
        scores = torch.matmul(query, key.transpose(-1, -2)) 
        # matmul은 input이 3차원 이상일 때 앞쪽 차원은 batch로 간주함
        # transpose는 input의 마지막(-1)과 마지막 전(-2) 차원을 서로 교환

        # scaling 적용 (√d_k로 나누기)
        scale = torch.tensor(d_k, device = scores.device, dtype = scores.dtype).sqrt()
        scores /= scale

        # 마스킹을 해야한다면 즉, 트랜스포머의 디코더 부분에서 마스킹 할때 외부에서 torch.tril() 선언해서 사용하면 됨
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, value)

        return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # 각 head의 차원

        # linear 변환 레이어들
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # ScaledDotProductAttention 인스턴스
        self.attention = ScaledDotProductAttention()

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear 변환 적용
        Q = self.W_q(query) # (batch_size, seq_len, d_model)



        



