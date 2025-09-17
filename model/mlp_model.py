import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score
from utils import roc_auc, pr_auc

class GeneDiseaseDataset(Dataset):
    def __init__(self, dataframe, entity_embeddings, relation_embeddings):
        """
        dataframe: pd.DataFrame, 包含4列: head, relation, tail, label
        entity_embeddings: dict, 实体ID到嵌入的映射 (torch.Tensor 类型)
        relation_embeddings: dict, 关系ID到嵌入的映射 (torch.Tensor 类型)
        """
        self.dataframe = dataframe
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        head_emb = self.entity_embeddings[row['head']]
        relation_emb = self.relation_embeddings[row['relation']]
        tail_emb = self.entity_embeddings[row['tail']]
        label = row['label']
        
        # 拼接三元组嵌入向量
        input_emb = np.concatenate([head_emb, relation_emb, tail_emb], axis=-1)
        return torch.tensor(input_emb, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
    
class MLPScoringModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[1024, 1024, 1024], dropout_rate=0.2):
        """
        多层感知机模型
        input_dim: 输入嵌入的维度
        hidden_dims: 隐藏层维度列表
        dropout_rate: Dropout 概率
        """
        super(MLPScoringModel, self).__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))  # 输出层
        self.mlp = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.mlp(x)
        return self.sigmoid(x)  # 将输出映射到 [0, 1]

def train_mlp(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test_mlp(model, test_loader, device):
    model.eval()
    all_labels = []
    all_scores = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(outputs.cpu().numpy())

    # 计算AUC和准确率
    auc_roc = roc_auc(all_labels, all_scores)
    auc_pr = pr_auc(all_labels, all_scores)
    preds = [1 if score > 0.5 else 0 for score in all_scores]
    acc = accuracy_score(all_labels, preds)
    return auc_roc, auc_pr, acc

def score_triplet(model, triplet, entity_embeddings, relation_embeddings, device):
    """
    给定模型和三元组，生成评分。
    triplet: 包含 head, relation, tail 的三元组
    entity_embeddings: 实体嵌入字典
    relation_embeddings: 关系嵌入字典
    """
    head_emb = entity_embeddings[triplet[0]]
    relation_emb = relation_embeddings[triplet[1]]
    tail_emb = entity_embeddings[triplet[2]]
    input_emb = np.concatenate([head_emb, relation_emb, tail_emb], axis=-1)
    input_tensor = torch.tensor(input_emb, dtype=torch.float32).to(device).unsqueeze(0)
    score = model(input_tensor).item()

    return score

def score_batch_triplets(model, fix_entity, relation, perturb_entity_index, perturb_type, entity_embeddings, relation_embeddings, device):
    fix_emb = torch.tensor(entity_embeddings[fix_entity], dtype=torch.float32).to(device)  # (embedding_dim,)
    relation_emb = torch.tensor(relation_embeddings[relation], dtype=torch.float32).to(device)  # (embedding_dim,)
    perturb_emb = torch.stack([torch.tensor(entity_embeddings[entity_id], dtype=torch.float32).to(device) 
                            for entity_id in perturb_entity_index])  # (len(perturb_entity_index), embedding_dim)
    
    fix_emb = fix_emb.unsqueeze(0).expand(len(perturb_entity_index), -1)  # (len(perturb_entity_index), embedding_dim)
    relation_emb = relation_emb.unsqueeze(0).expand(len(perturb_entity_index), -1)  # (len(perturb_entity_index), embedding_dim)

    if perturb_type == 'tail':
        input_emb = torch.cat([fix_emb, relation_emb, perturb_emb], dim=-1)  # (len(perturb_entity_index), 3 * embedding_dim)
    elif perturb_type == 'head':
        input_emb = torch.cat([perturb_emb, relation_emb, fix_emb], dim=-1)  # (len(perturb_entity_index), 3 * embedding_dim)
    
    # calculate scores
    scores = model(input_emb).squeeze()  # (len(perturb_entity_index),)
    
    return scores.cpu().numpy()

def get_rank(score_list, target):
    sorted_indices = np.argsort(score_list)[::-1]
    rank = np.where(sorted_indices == np.where(score_list == target)[0][0])[0][0] + 1
    return rank

def evaluate_metrics(model, test_triplets, entity_embeddings, relation_embeddings, all_triplets, device, hits=[1, 3, 10]):
    """
    计算测试集上的 MR, MRR, Hit@K。
    test_loader: 测试集 DataLoader
    model: 训练好的评分模型
    entity_embeddings: 实体嵌入字典
    relation_embeddings: 关系嵌入字典
    all_triplets: 所有已知的三元组集合（例如，训练集+测试集的三元组）
    device: 设备 (CPU 或 GPU)
    k_values: Hit@K 的 K 值列表，例如 [1, 3, 10]
    """
    model.eval()  # 切换到评估模式
    with torch.no_grad():
        ranks_s = []
        ranks_o = []

        head_relation_triplets = all_triplets[:, :2]
        tail_relation_triplets = np.stack((all_triplets[:, 2], all_triplets[:, 1]), axis=1)

        subject_relation_map = {}
        object_relation_map = {}

        for i in range(len(test_triplets)):
            test_triplet = test_triplets[i]
            if i % 1 == 0:
                print(f"Processing triplet {i}/{len(test_triplets)}")
            
            target = score_triplet(model, test_triplet, entity_embeddings, relation_embeddings, device)
            subject, relation, object_ = test_triplet[0], test_triplet[1], test_triplet[2]

            # Perturb object (head is fixed)
            subject_relation = test_triplet[:2]  # (subject, relation)
            subject_relation_key = (subject_relation[0], subject_relation[1])

            if subject_relation_key not in subject_relation_map:
                delete_index = np.sum(head_relation_triplets == subject_relation, axis=1)
                delete_index = np.where(delete_index == 2)[0]
                delete_entity_index = all_triplets[delete_index, 2] #columns 3, is tail entity 
                perturb_entity_index = np.array(list(set(entity_embeddings.keys()) - set(delete_entity_index)))
                subject_relation_map[subject_relation_key] = perturb_entity_index

            perturb_entity_index = subject_relation_map[subject_relation_key]
            perturb_entity_index = np.concatenate((perturb_entity_index, np.array([object_])), axis=0)
            scores = score_batch_triplets(model, subject, relation, perturb_entity_index, 'tail', entity_embeddings, relation_embeddings, device)
            ranks_s.append(get_rank(scores, target))

            # Perturb subject (tail is fixed)
            object_relation  = np.array([object_, relation])  # (subject, relation)
            object_relation_key  = (object_relation[0], object_relation[1])

            if object_relation_key  not in object_relation_map:
                delete_index = np.sum(tail_relation_triplets == object_relation, axis=1)
                delete_index = np.where(delete_index == 2)[0]
                delete_entity_index = all_triplets[delete_index, 0] #columns 1, is head entity
                perturb_entity_index = np.array(list(set(entity_embeddings.keys()) - set(delete_entity_index)))
                object_relation_map[object_relation_key] = perturb_entity_index

            perturb_entity_index = object_relation_map[object_relation_key]
            perturb_entity_index = np.concatenate([perturb_entity_index, np.array([subject])], axis=0)
            scores = score_batch_triplets(model, object_, relation, perturb_entity_index, 'head', entity_embeddings, relation_embeddings, device)
            ranks_o.append(get_rank(scores, target))
        
        ranks_s = np.array(ranks_s)
        ranks_o = np.array(ranks_o)

        ranks = np.concatenate([ranks_s, ranks_o])
        ranks += 1
        # Calculate MRR (Mean Reciprocal Rank)
        mrr = np.mean(1.0 / ranks.astype(np.float32))

        # Calculate MR (Mean Rank)
        mr = np.mean(ranks.astype(np.float32))

        # Calculate Hits@k
        hits_result = {}
        for hit in hits:
            avg_count = np.mean(ranks <= hit)
            hits_result[hit] = avg_count
        
        evaluate_res = {'MRR': mrr, 'MR': mr, 'Hits@1': hits_result[1], 'Hits@3': hits_result[3], 'Hits@10': hits_result[10]}
        return evaluate_res
    
def evaluate_metrics_simple(model, test_triplets, entity_embeddings, relation_embeddings, all_triplets, device, hits=[1, 3, 10]):
    """
    计算测试集上的 MR, MRR, Hit@K。
    test_loader: 测试集 DataLoader
    model: 训练好的评分模型
    entity_embeddings: 实体嵌入字典
    relation_embeddings: 关系嵌入字典
    all_triplets: 所有已知的三元组集合（例如，训练集+测试集的三元组）
    device: 设备 (CPU 或 GPU)
    k_values: Hit@K 的 K 值列表，例如 [1, 3, 10]
    """
    model.eval()  # 切换到评估模式
    with torch.no_grad():
        ranks_s = []
        ranks_o = []

        for i in range(len(test_triplets)):
            test_triplet = test_triplets[i]
            if i % 1 == 0:
                print(f"Processing triplet {i}/{len(test_triplets)}")
            
            target = score_triplet(model, test_triplet, entity_embeddings, relation_embeddings, device)
            subject, relation, object_ = test_triplet[0], test_triplet[1], test_triplet[2]

            # Perturb object (head is fixed)
            perturb_entity_index = np.array(list(entity_embeddings.keys()))
            scores = score_batch_triplets(model, subject, relation, perturb_entity_index, 'tail', entity_embeddings, relation_embeddings, device)
            ranks_s.append(get_rank(scores, target))

            # Perturb subject (tail is fixed)
            scores = score_batch_triplets(model, object_, relation, perturb_entity_index, 'head', entity_embeddings, relation_embeddings, device)
            ranks_o.append(get_rank(scores, target))
        
        ranks_s = np.array(ranks_s)
        ranks_o = np.array(ranks_o)

        ranks = np.concatenate([ranks_s, ranks_o])
        ranks += 1
        # Calculate MRR (Mean Reciprocal Rank)
        mrr = np.mean(1.0 / ranks.astype(np.float32))

        # Calculate MR (Mean Rank)
        mr = np.mean(ranks.astype(np.float32))

        # Calculate Hits@k
        hits_result = {}
        for hit in hits:
            avg_count = np.mean(ranks <= hit)
            hits_result[hit] = avg_count
        
        evaluate_res = {'MRR': mrr, 'MR': mr, 'Hits@1': hits_result[1], 'Hits@3': hits_result[3], 'Hits@10': hits_result[10]}
        return evaluate_res


# 更新添加注意力机制
# 分块处理
class AttentionWeightedFusion(nn.Module):
    def __init__(self, kge_dim, attr_dim):
        """
        分块处理注意力机制（拼接输出）
        kge_dim: KGE 嵌入维度
        attr_dim: 属性嵌入维度
        """
        super(AttentionWeightedFusion, self).__init__()
        self.kge_attention = nn.Sequential(
            nn.Linear(kge_dim, 1),
            nn.Softmax(dim=-1)  # 对每个样本的KGE部分归一化
        )
        self.attr_attention = nn.Sequential(
            nn.Linear(attr_dim, 1),
            nn.Softmax(dim=-1)  # 对每个样本的属性部分归一化
        )
    
    def forward(self, kge_emb, attr_emb):
        """
        kge_emb: [batch_size, kge_dim]
        attr_emb: [batch_size, attr_dim]
        """
        kge_weight = self.kge_attention(kge_emb)  # [batch_size, 1]
        attr_weight = self.attr_attention(attr_emb)  # [batch_size, 1]

        # 加权后的嵌入
        kge_weighted = kge_weight * kge_emb
        attr_weighted = attr_weight * attr_emb

        # 拼接加权后的嵌入
        fused_emb = torch.cat([kge_weighted, attr_weighted], dim=-1)  # [batch_size, kge_dim + attr_dim]
        return fused_emb

# 多头注意力机制
class MultiHeadAttentionFusion(nn.Module):
    def __init__(self, kge_dim, attr_dim, num_heads=2):
        """
        多头注意力机制
        kge_dim: KGE 嵌入维度
        attr_dim: 属性嵌入维度
        num_heads: 注意力头的数量
        """
        super(MultiHeadAttentionFusion, self).__init__()
        self.num_heads = num_heads
        
        # 独立注意力头
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(kge_dim + attr_dim, kge_dim + attr_dim),
                nn.ReLU(),
                nn.Linear(kge_dim + attr_dim, 1),  # 输出单个权重
                nn.Softmax(dim=-1)
            ) for _ in range(num_heads)
        ])
        self.output_layer = nn.Linear(num_heads, 1)  # 多头整合
    
    def forward(self, kge_emb, attr_emb):
        """
        kge_emb: [batch_size, kge_dim]
        attr_emb: [batch_size, attr_dim]
        """
        concat_emb = torch.cat([kge_emb, attr_emb], dim=-1)  # [batch_size, kge_dim + attr_dim]
        attention_outputs = []

        for head in self.attention_heads:
            weight = head(concat_emb)  # [batch_size, 1]
            attention_outputs.append(weight)

        multi_head_weights = torch.cat(attention_outputs, dim=-1)  # [batch_size, num_heads]
        fused_weight = self.output_layer(multi_head_weights)  # [batch_size, 1]
        
        kge_weighted = fused_weight * kge_emb
        attr_weighted = (1 - fused_weight) * attr_emb
        fused_emb = torch.cat([kge_weighted, attr_weighted], dim=-1)  # [batch_size, kge_dim + attr_dim]
        return fused_emb

# 优化的多头注意力机制
class EnhancedMultiHeadAttentionFusion(nn.Module):
    def __init__(self, kge_dim, attr_dim, num_heads=2, num_layers=2):
        """
        优化的多头注意力机制，支持残差连接和堆叠多层交互
        kge_dim: KGE 嵌入维度
        attr_dim: 属性嵌入维度
        num_heads: 注意力头的数量
        num_layers: 注意力层堆叠数量
        """
        super(EnhancedMultiHeadAttentionFusion, self).__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.input_dim = kge_dim + attr_dim

        # 初始化注意力层堆叠
        self.attention_layers = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(self.input_dim, self.input_dim),
                nn.ReLU(),
                nn.Linear(self.input_dim, 1),
                nn.Softmax(dim=-1),
                nn.LayerNorm(self.input_dim)
            ]) for _ in range(num_layers)
        ])
        # 每个头的独立输出层
        self.output_layers = nn.ModuleList([
            nn.Linear(self.input_dim, 1) for _ in range(num_heads)
        ])
        
        # 最终的压缩层
        self.final_layer = nn.Linear(num_heads, 1)

    def forward(self, kge_emb, attr_emb):
        """
        kge_emb: [batch_size, kge_dim]
        attr_emb: [batch_size, attr_dim]
        """
        concat_emb = torch.cat([kge_emb, attr_emb], dim=-1)  # [batch_size, kge_dim + attr_dim]
        residual = concat_emb.clone()

        attention_outputs = []
        for layer in self.attention_layers:
            linear1, activation, linear2, softmax, norm = layer
            weight = softmax(linear2(activation(linear1(concat_emb))))  # [batch_size, 1]
            concat_emb = norm(residual + weight * concat_emb)  # Add & Norm (残差连接)
            attention_outputs.append(concat_emb)  # 收集每个头的输出

        # 将每头的注意力输出映射到标量
        head_outputs = [self.output_layers[i](attention_outputs[i]) for i in range(len(self.attention_layers))]
        fused_weight = torch.cat(head_outputs, dim=-1)  # [batch_size, num_heads]

        # 压缩到最终输出
        fused_weight = self.final_layer(fused_weight)  # [batch_size, 1]
        kge_weighted = fused_weight * kge_emb
        attr_weighted = (1 - fused_weight) * attr_emb
        fused_emb = torch.cat([kge_weighted, attr_weighted], dim=-1)  # [batch_size, kge_dim + attr_dim]
        return fused_emb

# 数据加载类
class GeneDiseaseDataset_Att(Dataset):
    def __init__(self, dataframe, entity_kge_embeddings, entity_attr_embeddings, relation_embeddings):
        """
        dataframe: pd.DataFrame, 包含4列: head, relation, tail, label
        entity_kge_embeddings: dict, 实体ID到KGE嵌入的映射 (torch.Tensor 类型)
        entity_attr_embeddings: dict, 实体ID到属性嵌入的映射 (torch.Tensor 类型)
        relation_embeddings: dict, 关系ID到嵌入的映射 (torch.Tensor 类型)
        """
        self.dataframe = dataframe
        self.entity_kge_embeddings = entity_kge_embeddings
        self.entity_attr_embeddings = entity_attr_embeddings
        self.relation_embeddings = relation_embeddings

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        head_kge = torch.tensor(self.entity_kge_embeddings[row['head']], dtype=torch.float32)
        head_attr = torch.tensor(self.entity_attr_embeddings[row['head']], dtype=torch.float32)
        tail_kge = torch.tensor(self.entity_kge_embeddings[row['tail']], dtype=torch.float32)
        tail_attr = torch.tensor(self.entity_attr_embeddings[row['tail']], dtype=torch.float32)
        relation_emb = torch.tensor(self.relation_embeddings[row['relation']], dtype=torch.float32)
        label = torch.tensor(row['label'], dtype=torch.float32)

        return {
            'head_kge': head_kge,
            'head_attr': head_attr,
            'tail_kge': tail_kge,
            'tail_attr': tail_attr,
            'relation_emb': relation_emb,
            'label': label
        }

# 多层感知机模型
class MLPScoringModel_Att(nn.Module):
    def __init__(self, kge_dim, attr_dim, hidden_dims=[1024, 1024, 1024], dropout_rate=0.2, num_heads=2, num_layers=2):
        """
        MLP 模型，带注意力机制
        kge_dim: KGE 嵌入维度
        attr_dim: 属性嵌入维度
        rel_dim: 关系嵌入维度
        hidden_dims: 隐藏层维度列表
        dropout_rate: Dropout 概率
        num_heads: 注意力头数量
        """
        super(MLPScoringModel_Att, self).__init__()

        # self.fusion_layer = MultiHeadAttentionFusion(kge_dim, attr_dim, num_heads=num_heads)
        self.fusion_layer = EnhancedMultiHeadAttentionFusion(kge_dim, attr_dim, num_heads=num_heads, num_layers=num_layers)
        input_dim = (kge_dim + attr_dim) * 2 + kge_dim  # 头 + 尾 + 关系

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))  # 输出层
        self.mlp = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, head_kge, head_attr, tail_kge, tail_attr, relation_emb):
        head_fused = self.fusion_layer(head_kge, head_attr)
        tail_fused = self.fusion_layer(tail_kge, tail_attr)
        input_emb = torch.cat([head_fused, relation_emb, tail_fused], dim=-1)  # 拼接所有嵌入
        return self.sigmoid(self.mlp(input_emb))

# 训练函数
def train_mlp_att(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        head_kge = batch['head_kge'].to(device)
        head_attr = batch['head_attr'].to(device)
        tail_kge = batch['tail_kge'].to(device)
        tail_attr = batch['tail_attr'].to(device)
        relation_emb = batch['relation_emb'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(head_kge, head_attr, tail_kge, tail_attr, relation_emb).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test_mlp_att(model, test_loader, device):
    model.eval()
    all_labels = []
    all_scores = []
    with torch.no_grad():
        for batch in test_loader:
            head_kge = batch['head_kge'].to(device)
            head_attr = batch['head_attr'].to(device)
            tail_kge = batch['tail_kge'].to(device)
            tail_attr = batch['tail_attr'].to(device)
            relation_emb = batch['relation_emb'].to(device)
            labels = batch['label'].to(device)

            outputs = model(head_kge, head_attr, tail_kge, tail_attr, relation_emb).squeeze()
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(outputs.cpu().numpy())

    # 计算AUC和准确率
    auc_roc = roc_auc(all_labels, all_scores)
    auc_pr = pr_auc(all_labels, all_scores)
    preds = [1 if score > 0.5 else 0 for score in all_scores]
    acc = accuracy_score(all_labels, preds)
    return auc_roc, auc_pr, acc