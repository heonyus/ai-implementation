import torch
import torch.nn as nn
import math
from .attention import MultiHeadAttention
from .positional_encoding import PositionalEncoding


class FeedForward(nn.Module):
    """Feed Forward Network"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Transformer Encoder Layer"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """Transformer Decoder Layer"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention
        attn_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class Transformer(nn.Module):
    """Transformer Model"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, 
                 max_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        
        # 임베딩 층
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder & Decoder 층
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_encoder_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_decoder_layers)
        ])
        
        # 출력 층
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def generate_mask(self, src, tgt):
        # Source padding mask
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        # Target padding mask
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        
        # Causal mask for decoder
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        nopeak_mask = nopeak_mask.to(tgt.device)
        tgt_mask = tgt_mask & nopeak_mask
        
        return src_mask, tgt_mask
    
    def forward(self, src, tgt):
        # Mask 생성
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        # Encoder
        src_embedded = self.dropout(self.pos_encoding(self.encoder_embedding(src)))
        enc_output = src_embedded
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        
        # Decoder
        tgt_embedded = self.dropout(self.pos_encoding(self.decoder_embedding(tgt)))
        dec_output = tgt_embedded
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
        
        # Output
        output = self.fc_out(dec_output)
        return output
