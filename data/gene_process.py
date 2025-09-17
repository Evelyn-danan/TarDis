import os
import sys
import re
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import pickle
import json

# import pudb
import argparse

from transformers import BertModel, BertTokenizer
from transformers import AutoTokenizer, EsmModel

# parser = argparse.ArgumentParser(description='Extract BERT embeddings for proteins')
# parser.add_argument('--model', type=int, default=0, help='0 for ProtBERT, 1 for DistilProtBERT, 2 for ESM-2')
# args = parser.parse_args()
# model_choice = args.model

prot_models = ["Rostlab/prot_bert", "yarongef/DistilProtBert", "facebook/esm2_t33_650M_UR50D"]
model_choice = 2
PROT_MODEL = prot_models[model_choice]
PROT_MODEL = "/home/worker/users/ZC/KnowledgeGraph/TarKG_reason/data/huggingface/facebook/esm2_t33_650M_UR50D"

model_names = ["ProtBERT", "DistilProtBERT", "ESM-2"]
model_name = model_names[model_choice]

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

if model_choice == 2:
    prot_tokenizer = AutoTokenizer.from_pretrained(PROT_MODEL, do_lower_case=False) #, local_files_only=True
    prot_model = EsmModel.from_pretrained(PROT_MODEL)
else:
    prot_tokenizer = BertTokenizer.from_pretrained(PROT_MODEL, do_lower_case=False) #, local_files_only=True
    prot_model = BertModel.from_pretrained(PROT_MODEL)
prot_model.to(device)

BATCH_SIZE = 32 # 256
MAX_PROT_LEN = 1800

if len(sys.argv) > 1 and sys.argv[1] == "demo":
    source_dir = "./data_demo"
    target_dir = "./dataset_demo"
else:
    source_dir = "./data"
    target_dir = "./dataset"
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

def preprocess_protein(sequence):
    processProtein = [aa for aa in sequence] # aa is a single amino acid
    processProtein = " ".join(processProtein)
    processProtein = re.sub(r"[UZOB]", "X", processProtein)
    return processProtein

gene_data = pd.read_csv(os.path.join(source_dir, "gene_feature.csv"))
gene_data = gene_data[['index', 'sequence']]
gene_data['index'] = gene_data['index'].astype(int) - 1
gene_data['sequence'] = gene_data['sequence'].apply(preprocess_protein)

print("\nExtracting protein embeddings...")
train_batches = [(gene_data['index'].tolist()[i:i+BATCH_SIZE], gene_data['sequence'].tolist()[i:i+BATCH_SIZE])
                 for i in range(0, len(gene_data), BATCH_SIZE)]
print(f"  Total proteins: {len(gene_data)}, Batch size: {BATCH_SIZE}, Total batches: {len(train_batches)}")

protein_embeddings = {}
protein_embeddings_list = []
for batch_indices, train_batch in tqdm(train_batches):
    encoded_proteins = prot_tokenizer(train_batch, 
                                  return_tensors='pt', 
                                  max_length=MAX_PROT_LEN, 
                                  truncation=True, 
                                  padding=True)
    
    encoded_proteins = encoded_proteins.to(device)
    with torch.no_grad():
        train_target_embeddings = prot_model(**encoded_proteins).last_hidden_state[:, 0, :]
    for index, embedding in zip(batch_indices, train_target_embeddings.cpu().detach().numpy()):
        protein_embeddings[index] = embedding
        protein_embeddings_list.append(embedding)
    torch.cuda.empty_cache()

# 保存蛋白质嵌入字典格式
PATH = os.path.join(target_dir, "gene_embedding.pkl")
with open(PATH, 'wb') as f:
    pickle.dump(protein_embeddings, f)

# 保存numpy数组格式
protein_embeddings_array = np.array(protein_embeddings_list)
np.save(PATH.replace("pkl", "npy"), protein_embeddings_array)