import os
import sys
import re
import pandas as pd
import numpy as np
import torch
import pickle
import json
import datetime

from transformers import AutoTokenizer, AutoModel

print("Start time: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print(f"Current PID: {os.getpid()}")

# 加载BioBERT模型和分词器
DIS_MODEL = "/home/worker/users/ZC/KnowledgeGraph/TarKG_reason/data/huggingface/dmis-lab/biobert-base-cased-v1.2"
tokenizer = AutoTokenizer.from_pretrained(DIS_MODEL)
model = AutoModel.from_pretrained(DIS_MODEL)

if len(sys.argv) > 1 and sys.argv[1] == "demo":
    source_dir = "./data_demo"
    target_dir = "./dataset_demo"
    print("Start process dataset_demo...")
else:
    source_dir = "./data"
    target_dir = "./dataset"
    print("Start process dataset...")
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

def encode_disease_features(name, definition):
    # 将疾病名称和定义拼接为一个完整的输入句子
    input_text = f"Disease's name: {name}. Definition: {definition}"
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # 获取BioBERT输出
    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
    
    # 进行池化操作（平均池化）
    disease_embedding = torch.mean(last_hidden_state, dim=1).squeeze()
    return disease_embedding

# 读取疾病数据 disease_file = os.path.join(source_dir, "disease_feature.csv")
# suffix = ""
# suffix = "_update"
suffix = "_latest"
disease_file = os.path.join(source_dir, f"disease_feature{suffix}.csv")
print(f"Disease feature file: {disease_file}")
disease_data = pd.read_csv(disease_file)
disease_data['index'] = disease_data['index'].astype(int) - 1

# 对每条疾病记录进行编码
print("\nExtracting disease embeddings...")
print(f"Total diseases: {len(disease_data)}")

disease_embeddings = {}
disease_embeddings_list = []
count = 0
for _, row in disease_data.iterrows():
    name = row['name']
    definition = row['def'] if pd.notna(row['def']) else "This disease currently lacks a specific definition."
    embedding = encode_disease_features(name, definition)
    disease_embeddings[row['index']] = embedding.detach().numpy()  # 转为numpy格式方便保存
    disease_embeddings_list.append(embedding.detach().numpy())

    count += 1
    if count % 100 == 0:
        print(f"  Processed {count} diseases.")

# 保存字典格式
PATH = os.path.join(target_dir, f"disease_embedding{suffix}.pkl")
with open(PATH, 'wb') as f:
    pickle.dump(disease_embeddings, f)

# 保存numpy数组格式
disease_embeddings_array = np.array(disease_embeddings_list)
np.save(PATH.replace('pkl', 'npy'), disease_embeddings_array)

print()
print("End time: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))