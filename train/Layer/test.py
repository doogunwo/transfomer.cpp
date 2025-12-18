import InputEmbedding as IE
import PositionalEncoding as PE

import torch # type: ignore
import torch.nn as nn # type: ignore
import math

def Test1():
    # 1. 설정값 정의
    d_model = 512       # 논문 기준 벡터 차원
    vocab_size = 1000   # 단어장 크기 (임의 설정)

    # 2. 모델 생성
    emb_model = IE.InputEmbedding(d_model, vocab_size)
    print(f"모델 생성 완료: d_model={d_model}, vocab_size={vocab_size}")

    # 3. 가상의 입력 데이터 만들기
    # (배치 크기: 2개 문장, 문장 길이: 4개 단어)
    input_data = torch.LongTensor([
        [1, 2, 3, 4],  # 첫 번째 문장
        [5, 6, 7, 8]   # 두 번째 문장
    ])
    
    # 4. 모델 실행 (Forward)
    output = emb_model(input_data)

    # 5. 결과 검증
    print("-" * 30)
    print(f"입력 데이터 형태(Shape): {input_data.shape}") 
    # 예상: [2, 4]
    
    print(f"출력 데이터 형태(Shape): {output.shape}")      
    # 예상: [2, 4, 512] -> (배치, 문장길이, 임베딩차원)
    
    print("-" * 30)
    
    # 6. 값 확인 (스케일링이 적용되었는지 확인)
    print(f"출력 값 샘플 (첫번째 단어의 앞 5개 값):\n{output[0, 0, :5]}")

    max_len = 100   # 최대 문장 길이 설정
    dropout = 0.1   # 드롭아웃 비율
    
    # 2. Positional Encoding 모델 생성
    pe_model = PE.PositionalEncoding(d_model, max_len, dropout)
    
    # 3. 연결: Input Embedding 결과(output)를 여기에 넣음
    # output shape: (2, 4, 512)
    pe_output = pe_model(output)
    
    print("-" * 30)
    print(f"PE 적용 전: {output.shape}")
    print(f"PE 적용 후: {pe_output.shape}") # 모양은 그대로여야 함
    print("PE 적용 완료! 값이 변했는지 확인:", not torch.equal(output, pe_output))

if __name__ == '__main__':
    # 1. 설정값 정의
    Test1()