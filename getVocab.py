from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
import torch
import pandas as pd
from tqdm import tqdm
import pickle

# GPU 설정
gpu_device = 'cuda:3'  # 예를 들어 GPU 1을 사용
torch.cuda.set_device(gpu_device)

tokenizer=AutoTokenizer.from_pretrained("KoichiYasuoka/roberta-base-korean-morph-upos")
model=AutoModelForTokenClassification.from_pretrained("KoichiYasuoka/roberta-base-korean-morph-upos")
pipeline=TokenClassificationPipeline(tokenizer=tokenizer,model=model,aggregation_strategy="simple")


# 데이터 로드
file_path = './sorted_paragraphs.csv'
df_sorted = pd.read_csv(file_path)

# 토큰 분류 파이프라인 설정
pipeline = TokenClassificationPipeline(tokenizer=tokenizer, model=model, aggregation_strategy="simple", device=3)

# 품사 태깅 함수 정의
nlp=lambda x:[(x[t["start"]:t["end"]],t["entity_group"]) for t in pipeline(x)]

# 품사 태깅 실행 및 진행률 표시
all_results = []
for text in tqdm(df_sorted['paragraphs'], desc="Processing texts"):
    text_result = nlp(text)
    all_results.append(text_result)

# 결과 저장
output_file = './vocab_by_nlp.pkl'
with open(output_file, 'wb') as file:
    pickle.dump(all_results, file)
