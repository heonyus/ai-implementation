import torch
import torch.nn as nn
import math
from collections import Counter
import re

class BPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = []
        self.token_to_id = {}
        self.id_to_token = {}
    
    def get_word_freqs(self, corpus):
        """텍스트에서 단어와 빈도를 추출"""
        words = []
        for text in corpus:
            # 텍스트를 단어로 분리
            for word in text.split():
                words.append(word)
        return Counter(words)
    
    def get_pairs(self, word):
        """단어를 문자 쌍으로 분리"""
        pairs = set()
        if len(word) <= 1:
            return pairs
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def train(self, corpus, num_merges=1000):
        """
        BPE 학습
        corpus: 텍스트 리스트 (예: ["이것은 나의 코드", "안녕하세요"])
        num_merges: 몇 번 병합할지
        """
        # 1. 초기 vocab 구성 (각 문자와 특수 토큰 추가)
        vocab = set()
        for word in self.get_word_freqs(corpus).keys():
            vocab.update(list(word))

        # 특수 토큰 추가
        vocab.add('<PAD>')
        vocab.add('<UNK>')

        # vocab을 리스트로 변환
        vocab = sorted(list(vocab))

        # 2. 각 단어를 문자 단위로 분리하여 저장
        word_freqs = self.get_word_freqs(corpus)
        word_vocab = {word: list(word) for word in word_freqs.keys()}

        # 3. num_merges 번 반복하면서 병합
        for i in range(num_merges):
            pairs = Counter()

            # 모든 단어에서 쌍의 빈도 계산
            for word, freq in word_freqs.items():
                splits = word_vocab[word]
                if len(splits) == 1:
                    continue
                
                # 단어 내의 모든 쌍 추출
                word_pairs = self.get_pairs(word_vocab[word])
                for pair in word_pairs:
                    pairs[pair] += freq
            
            if len(pairs) == 0:
                break
            
            # 가장 빈번한 쌍 선택
            most_frequent_pair = pairs.most_common(1)[0][0]
            
            # 병합 수행
            for word in word_vocab.keys():
                splits = word_vocab[word]
                if len(splits) == 1:
                    continue
                
                i = 0
                new_word = []
                while i < len(splits):
                    if (i < len(splits) - 1 and 
                        splits[i] == most_frequent_pair[0] and 
                        splits[i+1] == most_frequent_pair[1]):
                        new_word.append(most_frequent_pair[0] + most_frequent_pair[1])
                        i += 2
                    else:
                        new_word.append(splits[i])
                        i += 1
                word_vocab[word] = new_word
            
            # 병합 결과를 vocab에 추가
            merge_result = most_frequent_pair[0] + most_frequent_pair[1]
            if merge_result not in vocab:
                vocab.append(merge_result)
            
            # merges에 기록
            self.merges.append(most_frequent_pair)
        
        # vocab을 딕셔너리로 변환
        self.vocab = {token: idx for idx, token in enumerate(vocab)}
        self.token_to_id = self.vocab
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        
        print(f"학습 완료! Vocab 크기: {len(self.vocab)}")
    
    def apply_merges(self, word):
        """단어에 병합 규칙 적용"""
        if word in self.token_to_id:
            return [word]
        
        word = list(word)
        
        for pair in self.merges:
            i = 0
            new_word = []
            while i < len(word):
                if (i < len(word) - 1 and 
                    word[i] == pair[0] and 
                    word[i+1] == pair[1]):
                    new_word.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        
        return word
    
    def encode(self, text):
        """
        텍스트를 토큰 ID로 변환
        """
        tokens = []
        for word in text.split():
            word_tokens = self.apply_merges(word)
            for token in word_tokens:
                if token in self.token_to_id:
                    tokens.append(self.token_to_id[token])
                else:
                    tokens.append(self.token_to_id.get('<UNK>', 0))
        return tokens
    
    def decode(self, token_ids):
        """
        토큰 ID를 텍스트로 복원
        """
        tokens = [self.id_to_token.get(token_id, '<UNK>') for token_id in token_ids]
        return ' '.join(tokens)