import os
import argparse
import numpy as np
import pandas as pd
import pickle
import datetime
from tqdm import tqdm
import torch

from mlp_model import MLPScoringModel
from utils import load_IDMapping, load_gene_disease_ids, load_train_data, read_triplets
from utils import roc_auc, pr_auc
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder
from sklearn.decomposition import PCA

def parse_args():
    parser = argparse.ArgumentParser(description='KGE-MLP fused model prediction')
    
    parser.add_argument("-data", "--data", default="../data/dataset", help="data directory")
    parser.add_argument("--gpu", type=int, default=-1)
    
    # below are parameters for KGE model
    parser.add_argument("--model", type=str, default='CompGCN')

    # below are parameters for MLP model
    parser.add_argument("--hpo_mode", type=str, default='merged')
    parser.add_argument("--mlp_mode", type=str, choices=["merged", "mergedAtt", "onlyKGE", "onlyFeat", "direct", "empty"], 
                        default="merged", help="MLP input mode: 'merged', 'mergedAtt', 'direct', 'empty', 'onlyKGE', or 'onlyFeat'")
    args = parser.parse_args()

    return args

print("Start time: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print(f"Current PID: {os.getpid()}")

args = parse_args()
print(args)
data_dir = args.data
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

if args.hpo_mode == 'merged':
    if args.mlp_mode == 'merged':
        mlp_model_res = "/home/worker/users/ZC/KnowledgeGraph/TarKG_reason/pykeen_model/train_results/mlp_ablation_merged0215/MLP_merged_20250215-144803"
    elif args.mlp_mode == 'onlyKGE':
        mlp_model_res = "/home/worker/users/ZC/KnowledgeGraph/TarKG_reason/pykeen_model/train_results/mlp_ablation_merged0215/MLP_onlyKGE_20250215-144803"
    elif args.mlp_mode == 'onlyFeat':
        mlp_model_res = "/home/worker/users/ZC/KnowledgeGraph/TarKG_reason/pykeen_model/train_results/mlp_ablation_merged0215/MLP_onlyFeat_20250215-144804"

# 获取嵌入的两种方式
## 111111
from utils import load_entity_feature, get_merged_embeddings
def merging_embeddings_byghand(data_dir, mlp_model_res, entity2id, relation2id, gene_ids, disease_ids, target_dim=200):
    gene_feat_emb, disease_feat_emb = load_entity_feature(data_dir)
    kge_ent_emb = np.load(f'{mlp_model_res}/entity_embedding.npy')
    kge_rel_emb = np.load(f'{mlp_model_res}/relation_embedding.npy')
    print(gene_feat_emb.shape, disease_feat_emb.shape)
    print(kge_ent_emb.shape, kge_rel_emb.shape)

    entity_embeddings, relation_embeddings = get_merged_embeddings(kge_ent_emb, kge_rel_emb, entity2id, relation2id,
                                                                    gene_feat_emb, disease_feat_emb, gene_ids, disease_ids, target_dim)
    return entity_embeddings, relation_embeddings

## 222222
def get_mlp_embeddings(mlp_model_res):
    # with open(os.path.join(mlp_model_res, 'entity_embedding.pkl'), 'rb') as f:
    #     entity_embeddings = pickle.load(f)
    # with open(os.path.join(mlp_model_res, 'relation_embedding.pkl'), 'rb') as f:
    #     relation_embeddings = pickle.load(f)

    entity_embeddings = np.load(os.path.join(mlp_model_res, 'entity_embedding.npy'))
    relation_embeddings = np.load(os.path.join(mlp_model_res, 'relation_embedding.npy'))

    return entity_embeddings, relation_embeddings

# 打分排名
def score_triplet(model, triplet, entity_embeddings, relation_embeddings, device):
    """
    给定模型和三元组，生成评分。
    triplet: 包含 head, relation, tail 的三元组
    entity_embeddings: 实体嵌入 tensor
    relation_embeddings: 关系嵌入 tensor
    """
    # 获取 head, relation 和 tail 的嵌入
    head_emb = entity_embeddings[triplet[0]]
    relation_emb = relation_embeddings[triplet[1]]
    tail_emb = entity_embeddings[triplet[2]]
    
    # 将它们沿着最后一个维度拼接
    input_emb = torch.cat([head_emb, relation_emb, tail_emb], dim=-1)
    
    # 将输入传递到模型并计算分数
    score = model(input_emb)#.squeeze()#.item()

    return score

def score_batch_triplets(model, fix_entity, relation, perturb_entity_index, perturb_type, entity_embeddings, relation_embeddings, device):
    """
    给定模型、三元组和扰动实体索引，批量生成评分。
    fix_entity: 固定实体
    relation: 关系
    perturb_entity_index: 扰动实体索引列表
    perturb_type: 'head' 或 'tail'
    entity_embeddings: 实体嵌入 tensor, 已在 device 上
    relation_embeddings: 关系嵌入 tensor, 已在 device 上
    """

    # 获取固定实体和关系的嵌入
    fix_emb = entity_embeddings[fix_entity]  # (embedding_dim,)
    relation_emb = relation_embeddings[relation]  # (embedding_dim,)
    
    # 获取扰动实体的嵌入
    perturb_emb = entity_embeddings[perturb_entity_index]  # (len(perturb_entity_index), embedding_dim)
    
    # 扩展固定实体和关系嵌入
    fix_emb = fix_emb.unsqueeze(0).expand(len(perturb_entity_index), -1)  # (len(perturb_entity_index), embedding_dim)
    relation_emb = relation_emb.unsqueeze(0).expand(len(perturb_entity_index), -1)  # (len(perturb_entity_index), embedding_dim)
    
    # 根据扰动类型拼接输入嵌入
    if perturb_type == 'tail':
        input_emb = torch.cat([fix_emb, relation_emb, perturb_emb], dim=-1)  # (len(perturb_entity_index), 3 * embedding_dim)
    elif perturb_type == 'head':
        input_emb = torch.cat([perturb_emb, relation_emb, fix_emb], dim=-1)  # (len(perturb_entity_index), 3 * embedding_dim)

    # 计算分数并返回结果
    scores = model(input_emb).squeeze()  # (len(perturb_entity_index),)
    return scores#.cpu().detach().numpy()  # 直接返回结果

def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=0, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices

def calc_mrr(model, test_triplets, entity_embeddings, relation_embeddings, known_triplets, device, hits=[1, 3, 10, 100, 300, 500, 1000]):
    """
    计算测试集上的 MR, MRR, Hit@K。
    test_triplets: 测试集 test_triplets
    model: 训练好的评分模型
    entity_embeddings: 实体嵌入 tensor, 已在 device 上
    relation_embeddings: 关系嵌入 tensor, 已在 device 上
    known_triplets: 所有已知的三元组集合（例如，训练集+测试集的三元组）
    device: 设备 (CPU 或 GPU)
    k_values: Hit@K 的 K 值列表，例如 [1, 3, 10]
    """
    model.eval()  # 切换到评估模式
    with torch.no_grad():
        ranks_s = []
        ranks_o = []

        head_relation_triplets = known_triplets[:, :2]
        tail_relation_triplets = torch.stack((known_triplets[:, 2], known_triplets[:, 1])).transpose(0, 1)

        subject_relation_map = {}
        object_relation_map = {}

        all_entities = torch.arange(len(entity_embeddings), device=entity_embeddings.device)
        for i in range(len(test_triplets)):
            test_triplet = test_triplets[i]
            # if i % 1000 == 0:
            #     print(f" Processing triplet {i}/{len(test_triplets)}")
            
            # target = score_triplet(model, test_triplet, entity_embeddings, relation_embeddings, device)
            subject, relation, object_ = test_triplet[0], test_triplet[1], test_triplet[2]

            # Perturb object (head is fixed)
            subject_relation = test_triplet[:2]  # (subject, relation)
            subject_relation_key = (subject_relation[0].item(), subject_relation[1].item())

            if subject_relation_key not in subject_relation_map:
                delete_index = torch.sum(head_relation_triplets == subject_relation, dim=1)
                delete_index = torch.nonzero(delete_index == 2).squeeze()
                delete_entity_index = known_triplets[delete_index, 2] #columns 3, is tail entity 
                
                mask = torch.isin(all_entities, delete_entity_index)
                perturb_entity_index = all_entities[~mask]
                subject_relation_map[subject_relation_key] = perturb_entity_index

            perturb_entity_index = subject_relation_map[subject_relation_key]
            perturb_entity_index = torch.cat((perturb_entity_index, object_.view(-1)))
            scores = score_batch_triplets(model, subject, relation, perturb_entity_index, 'tail', entity_embeddings, relation_embeddings, device)
            target = torch.tensor(len(perturb_entity_index) - 1).to(entity_embeddings.device)

            ranks_s.append(sort_and_rank(scores, target))

            # Perturb subject (tail is fixed)
            object_relation  = torch.cat([object_.unsqueeze(0), relation.unsqueeze(0)], dim=0)  # (subject, relation)
            object_relation_key = (object_relation[0].item(), object_relation[1].item())

            if object_relation_key  not in object_relation_map:
                delete_index = torch.sum(tail_relation_triplets == object_relation, dim=1)
                delete_index = torch.nonzero(delete_index == 2).squeeze()
                delete_entity_index = known_triplets[delete_index, 0] #columns 1, is head entity
                
                mask = torch.isin(all_entities, delete_entity_index)
                perturb_entity_index = all_entities[~mask]
                object_relation_map[object_relation_key] = perturb_entity_index

            perturb_entity_index = object_relation_map[object_relation_key]
            perturb_entity_index = torch.cat((perturb_entity_index, subject.view(-1)))
            scores = score_batch_triplets(model, object_, relation, perturb_entity_index, 'head', entity_embeddings, relation_embeddings, device)

            target = torch.tensor(len(perturb_entity_index) - 1).to(entity_embeddings.device)
            ranks_o.append(sort_and_rank(scores, target))
        
        ranks_s = torch.cat(ranks_s)
        ranks_o = torch.cat(ranks_o)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1
        # Calculate MRR (Mean Reciprocal Rank)
        mrr = torch.mean(1.0 / ranks.float()).item()

        # Calculate MR (Mean Rank)
        mr = torch.mean(ranks.float()).item()

        # Calculate Hits@k
        hits_result = {}
        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float()).item()
            hits_result[hit] = avg_count
        
        evaluate_res = {'MRR': mrr, 'MR': mr, 'Hits@1': hits_result[1], 'Hits@3': hits_result[3], 'Hits@10': hits_result[10], 'Hits@100': hits_result[100],
                        'Hits@300': hits_result[300], 'Hits@500': hits_result[500], 'Hits@1000': hits_result[1000]}
        return evaluate_res

def calc_mrr_simple(model, test_triplets, entity_embeddings, relation_embeddings, known_triplets, device, hits=[1, 3, 10]):
    """
    计算测试集上的 MR, MRR, Hit@K。
    test_triplets: 测试集 test_triplets
    model: 训练好的评分模型
    entity_embeddings: 实体嵌入字典
    relation_embeddings: 关系嵌入字典
    known_triplets: 所有已知的三元组集合（例如，训练集+测试集的三元组）
    device: 设备 (CPU 或 GPU)
    k_values: Hit@K 的 K 值列表，例如 [1, 3, 10]
    """
    model.eval()  # 切换到评估模式
    with torch.no_grad():
        ranks_s = []
        ranks_o = []

        all_entities = torch.arange(len(entity_embeddings), device=entity_embeddings.device)
        for i in range(len(test_triplets)):
            test_triplet = test_triplets[i]
            # if i % 100 == 0:
            #     print(f" Processing triplet {i}/{len(test_triplets)}")
            
            target = score_triplet(model, test_triplet, entity_embeddings, relation_embeddings, device).item()
            subject, relation, object_ = test_triplet[0], test_triplet[1], test_triplet[2]

            # Perturb object (head is fixed)
            scores = score_batch_triplets(model, subject, relation, all_entities, 'tail', entity_embeddings, relation_embeddings, device)
            target = torch.tensor(object_.item()).to(entity_embeddings.device)
            ranks_s.append(sort_and_rank(scores, target))

            # Perturb subject (tail is fixed)
            scores = score_batch_triplets(model, object_, relation, all_entities, 'head', entity_embeddings, relation_embeddings, device)
            target = torch.tensor(subject.item()).to(entity_embeddings.device)
            ranks_o.append(sort_and_rank(scores, target))
        
        ranks_s = torch.cat(ranks_s)
        ranks_o = torch.cat(ranks_o)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1
        # Calculate MRR (Mean Reciprocal Rank)
        mrr = torch.mean(1.0 / ranks.float()).item()

        # Calculate MR (Mean Rank)
        mr = torch.mean(ranks.float()).item()

        # Calculate Hits@k
        hits_result = {}
        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float()).item()
            hits_result[hit] = avg_count
        
        evaluate_res = {'MRR': mrr, 'MR': mr, 'Hits@1': hits_result[1], 'Hits@3': hits_result[3], 'Hits@10': hits_result[10]}
        return evaluate_res

def evaluate_Rankbased(model, test_triplets, entity_embeddings, relation_embeddings, known_triplets, device, batch_size:int=10000):
    # test_triplets = test_triplets.to(device)
    # entity_embeddings = entity_embeddings.to(device)
    # relation_embeddings = relation_embeddings.to(device)
    # known_triplets = known_triplets.to(device)
    
    total_res = {'MRR': 0, 'MR': 0, 'Hits@1': 0, 'Hits@3': 0, 'Hits@10': 0, 'Hits@100': 0, 'Hits@300': 0, 'Hits@500': 0, 'Hits@1000': 0}
    for i in tqdm(range(0, len(test_triplets), batch_size), desc="Evaluating Batches", unit="batch"):
        batch_valid_triplets = test_triplets[i:i+batch_size]
        batch_length = len(batch_valid_triplets)
        # print(f"now evaluating {i+1} batch, its length is {batch_length}")

        batch_res = calc_mrr(model, batch_valid_triplets, entity_embeddings, relation_embeddings, known_triplets, device)
        total_res['MRR'] += batch_res['MRR'] * batch_length
        total_res['MR'] += batch_res['MR'] * batch_length
        total_res['Hits@1'] += batch_res['Hits@1'] * batch_length
        total_res['Hits@3'] += batch_res['Hits@3'] * batch_length
        total_res['Hits@10'] += batch_res['Hits@10'] * batch_length
        total_res['Hits@100'] += batch_res['Hits@100'] * batch_length
        total_res['Hits@300'] += batch_res['Hits@300'] * batch_length
        total_res['Hits@500'] += batch_res['Hits@500'] * batch_length
        total_res['Hits@1000'] += batch_res['Hits@1000'] * batch_length

        # Update progress bar with current batch Hits@1, Hits@3, Hits@10
        tqdm.write(f"Batch {i//batch_size + 1}, Hits@1: {batch_res['Hits@1']:.4f}, Hits@3: {batch_res['Hits@3']:.4f}, Hits@10: {batch_res['Hits@10']:.4f}")

    total_res['MRR'] /= len(test_triplets)
    total_res['MR'] /= len(test_triplets)
    total_res['Hits@1'] /= len(test_triplets)
    total_res['Hits@3'] /= len(test_triplets)
    total_res['Hits@10'] /= len(test_triplets)
    total_res['Hits@100'] /= len(test_triplets)
    total_res['Hits@300'] /= len(test_triplets)
    total_res['Hits@500'] /= len(test_triplets)
    total_res['Hits@1000'] /= len(test_triplets)

    return total_res

# 预测
def predict_hrt(model, entity_embeddings, relation_embeddings, entity2id, relation2id, gene_ids, disease_ids,
                 query_entity, query_entity_location, query_relation, known_triplets, device):
    # known_triplets = read_triplets(os.path.join(data_dir, "gene_disease_triplet.tsv"), entity2id, relation2id)
    head_relation_triplets = known_triplets[:, :2]
    tail_relation_triplets = torch.stack((known_triplets[:, 2], known_triplets[:, 1])).transpose(0, 1)

    # gene_ids, disease_ids = load_gene_disease_ids(data_dir, entity2id)
    entity_index = entity2id[query_entity]
    relation_index = relation2id[query_relation]
    query_entity_type = query_entity.split("::")[0]

    if query_entity_type == "Gene":
        target_ids = torch.tensor(disease_ids, dtype=torch.long, device=entity_embeddings.device)
    elif query_entity_type == "Disease":
        target_ids =  torch.tensor(gene_ids, dtype=torch.long, device=entity_embeddings.device)

    # 剔除已存在的三元组
    if query_entity_location == "head":
        target_entity_location = "tail"

        subject_relation = torch.tensor([entity_index, relation_index], dtype=torch.long, device=entity_embeddings.device)
        delete_index = torch.sum(head_relation_triplets == subject_relation, dim=1)
        delete_index = torch.nonzero(delete_index == 2).squeeze()
        delete_entity_index = known_triplets[delete_index, 2]
        mask = torch.isin(target_ids, delete_entity_index)
        target_entity_index = target_ids[~mask]

    elif query_entity_location == "tail":
        target_entity_location = "head"

        object_relation = torch.tensor([entity_index, relation_index], dtype=torch.long, device=entity_embeddings.device)
        delete_index = torch.sum(tail_relation_triplets == object_relation, dim=1)
        delete_index = torch.nonzero(delete_index == 2).squeeze()
        delete_entity_index = known_triplets[delete_index, 0]
        mask = torch.isin(target_ids, delete_entity_index)
        target_entity_index = target_ids[~mask]

    model.eval()  # 切换到评估模式
    with torch.no_grad():
        scores = score_batch_triplets(model, entity_index, relation_index, target_entity_index, target_entity_location, entity_embeddings, relation_embeddings, device)
        index_scores = torch.cat([target_entity_index.unsqueeze(1), scores.unsqueeze(1)], dim=1)
    scores_df = pd.DataFrame(index_scores.cpu().numpy(), columns=['target_index', 'score'])
    scores_df = scores_df.sort_values(by='score', ascending=False, ignore_index=True)
    scores_df['target_index'] = scores_df['target_index'].astype('int64')

    id2entity = {v: k for k, v in entity2id.items()}
    scores_df.insert(1, 'target_id', scores_df['target_index'].map(id2entity))
    scores_df['rank'] = scores_df.index + 1

    return scores_df

def predict_triplets_scores(model, entity_embeddings, relation_embeddings, triplets: pd.DataFrame, device):
    """
    给定模型和三元组，批量预测三元组的分数。
    triplets: 包含 head, relation, tail 的三元组, 为 DataFrame结构
    entity_embeddings: 实体嵌入 tensor, 已在 device 上
    relation_embeddings: 关系嵌入 tensor, 已在 device 上
    """
    # 从 triplets 中提取 head, relation, tail 的索引
    head_entity_index = triplets.iloc[:, 0]#.to(device)
    relation_index = triplets.iloc[:, 1]#.to(device)
    tail_entity_index = triplets.iloc[:, 2]#.to(device)
    
    # 获取头实体、关系和尾实体的嵌入
    head_emb = entity_embeddings[head_entity_index]  # (len(triplets), embedding_dim)
    relation_emb = relation_embeddings[relation_index]  # (len(triplets), embedding_dim)
    tail_emb = entity_embeddings[tail_entity_index]  # (len(triplets), embedding_dim)
    
    # 拼接 head, relation, tail 的嵌入
    input_emb = torch.cat([head_emb, relation_emb, tail_emb], dim=-1)  # (len(triplets), 3 * embedding_dim)
    
    # 计算分数
    model.eval()
    with torch.no_grad():
        scores = model(input_emb).squeeze()  # (len(triplets),)
    
    return scores.cpu().detach().numpy()  # 转换为 NumPy 数组并返回

def evaluate_Classification(model, test_data_id, entity_embeddings, relation_embeddings, device):
    test_data_triplets = test_data_id[columns]

    labels = test_data_id['label'].values
    scores = predict_triplets_scores(model, entity_embeddings, relation_embeddings, test_data_triplets, device)

    auc_roc = roc_auc(labels, scores)
    auc_pr = pr_auc(labels, scores)
    preds = np.where(scores > 0.5, 1, 0)
    acc = accuracy_score(labels, preds)

    print(f"AUC-ROC: {auc_roc}")
    print(f"AUC-PR: {auc_pr}")
    print(f"ACC: {acc}")

columns = ['head', 'relation', 'tail']
entity2id, relation2id = load_IDMapping(data_dir)
gene_ids, disease_ids = load_gene_disease_ids(data_dir, entity2id)
train_data, test_data, train_data_id, test_data_id = load_train_data(data_dir, entity2id, relation2id, 0)
known_triplets = read_triplets(os.path.join(data_dir, "gene_disease_triplet.tsv"), entity2id, relation2id)
known_triplets = torch.tensor(known_triplets, dtype=torch.long, device=device)
print(f" Gene_Disease triplets: {len(known_triplets)}")
print(f" Gene ids: {len(gene_ids)}, Disease ids: {len(disease_ids)}")

entity_embeddings, relation_embeddings = get_mlp_embeddings(mlp_model_res)
entity_embeddings = torch.tensor(entity_embeddings, dtype=torch.float32).to(device)
relation_embeddings = torch.tensor(relation_embeddings, dtype=torch.float32).to(device)
print(f"entity_embeddings shape: {entity_embeddings.shape}, relation_embeddings shape: {relation_embeddings.shape}")

test_data_id_pos = test_data_id[test_data_id['label'] == 1]
test_triplets = test_data_id_pos[columns].values
test_triplets = torch.tensor(test_triplets, dtype=torch.long, device=device)
print(f"known triplets shape: {known_triplets.shape}, test triplets shape: {test_triplets.shape}\n")

checkpoint = torch.load(os.path.join(mlp_model_res, "best_MLPmodel.pth"))
model = checkpoint['model'].to(device)

# 计算排名
print("Current time: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
batch_res = evaluate_Rankbased(model, test_triplets, entity_embeddings, relation_embeddings, known_triplets, device, batch_size=10000)

print('MRR: ', batch_res['MRR'])
print('MR: ', batch_res['MR'])
print('Hits@1: ', batch_res['Hits@1'])
print('Hits@3: ', batch_res['Hits@3'])
print('Hits@10: ', batch_res['Hits@10'])
print('Hits@100: ', batch_res['Hits@100'])
print('Hits@300: ', batch_res['Hits@300'])
print('Hits@500: ', batch_res['Hits@500'])
print('Hits@1000: ', batch_res['Hits@1000'])
print()

# 应用
query_entity = 'Gene::O95630' #AMSH
query_entity_location = 'head'
query_relation = 'Gene:Disease::drug targets'

print("Current time: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
predict_scores_df = predict_hrt(model, entity_embeddings, relation_embeddings, entity2id, relation2id, gene_ids, disease_ids,
                                 query_entity, query_entity_location, query_relation, known_triplets, device)
print("chronic kidney disease:", predict_scores_df[predict_scores_df['target_id'] == 'Disease::DOID:784']['rank'].values[0])
print("ulcerative colitis", predict_scores_df[predict_scores_df['target_id'] == 'Disease::DOID:8577']['rank'].values[0])

test_data_triplets = test_data_id[columns]
labels = test_data_id['label'].values
scores = predict_triplets_scores(model, entity_embeddings, relation_embeddings, test_data_triplets, device)

auc_roc = roc_auc(labels, scores)
auc_pr = pr_auc(labels, scores)
preds = np.where(scores > 0.5, 1, 0)
acc = accuracy_score(labels, preds)
print(f"AUC-ROC: {auc_roc}")
print(f"AUC-PR: {auc_pr}")
print(f"ACC: {acc}")

print()
print("End time: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))