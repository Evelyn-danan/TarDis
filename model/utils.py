import os
import numpy as np
import pandas as pd
import json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder
from sklearn.decomposition import PCA
from sklearn import metrics

# Load data and related information
def load_IDMapping(file_path):
    with open(os.path.join(file_path, 'entity_map.dict')) as f:
        entity2id = dict()

        for line in f:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(file_path, 'relation_map.dict')) as f:
        relation2id = dict()

        for line in f:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    
    print(' num_entity: {}'.format(len(entity2id)))
    print(' num_relation: {}'.format(len(relation2id)))

    return entity2id, relation2id

def read_triplets(file_path, entity2id, relation2id, label: bool=False):
    triplets = []

    if label:
        with open(file_path) as f:
            for line in f:
                if line.startswith("head"):
                    continue
                head, relation, tail, label = line.strip().split('\t')
                triplets.append((entity2id[head], relation2id[relation], entity2id[tail], int(label)))

        return pd.DataFrame(triplets, columns=['head', 'relation', 'tail', 'label'])

    with open(file_path) as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            triplets.append((entity2id[head], relation2id[relation], entity2id[tail]))

    return np.array(triplets)

def load_support_data(file_path, entity2id, relation2id):
    support_triplet = read_triplets(os.path.join(file_path, 'support_triplet.tsv'), entity2id, relation2id)
    gene_triplet = read_triplets(os.path.join(file_path, 'gene_triplet.tsv'), entity2id, relation2id)
    disease_triplet = read_triplets(os.path.join(file_path, 'disease_triplet.tsv'), entity2id, relation2id)

    print(' num_support_triples: {}'.format(len(support_triplet)))
    print(' num_gene_triples: {}'.format(len(gene_triplet)))
    print(' num_disease_triples: {}'.format(len(disease_triplet)))

    return support_triplet, gene_triplet, disease_triplet

def load_train_data(file_path, entity2id, relation2id, i):

    print("Load data from {} and {} fold:".format(file_path, i+1))
    train_file = os.path.join(file_path, f'data_folds/train_fold_{i+1}.tsv')
    test_file = os.path.join(file_path, f'data_folds/test_fold_{i+1}.tsv')

    train_data_str = pd.read_csv(train_file, sep='\t')
    test_data_str = pd.read_csv(test_file, sep='\t')
    train_data_str_pos = train_data_str[train_data_str['label'] == 1]
    test_data_str_pos = test_data_str[test_data_str['label'] == 1]

    train_data_id = read_triplets(train_file, entity2id, relation2id, label=True)
    test_data_id = read_triplets(test_file, entity2id, relation2id, label=True)

    # train_triplet = train_data[train_data['label'] == 1][['head', 'relation', 'tail']].copy(deep=True)
    # test_triplet = test_data[test_data['label'] == 1][['head', 'relation', 'tail']].copy(deep=True)
    # train_triplet = train_triplet.to_numpy()
    # test_triplet = test_triplet.to_numpy()
    print(' num_train_triples: {}'.format(len(train_data_str_pos)))
    print(' num_test_triples: {}'.format(len(test_data_str_pos)))

    return train_data_str, test_data_str, train_data_id, test_data_id

def load_entity_feature(file_path, file_type='numpy'):
    # read gene and disease feature embedding from preprocessing file
    if file_type == 'numpy':
        gene_feat_emb = np.load(file_path + '/gene_embedding.npy')
        disease_feat_emb = np.load(file_path + '/disease_embedding.npy')
    elif file_type == 'pickle':
        with open(os.path.join(file_path, "gene_embedding.pkl"), 'rb') as f:
            gene_feat_emb = pickle.load(f)
        with open(os.path.join(file_path, "disease_embedding.pkl"), 'rb') as f:
            disease_feat_emb = pickle.load(f)

        # # convert embedding into numpy array from list
        # gene_feat_emb = {int(index): np.array(embedding) for index, embedding in gene_feat_emb.items()}
        # disease_feat_emb = {int(index): np.array(embedding) for index, embedding in disease_feat_emb.items()}

        # feat_emb = {**gene_feat_emb, **disease_feat_emb}

    return gene_feat_emb, disease_feat_emb

def load_gene_disease_ids(file_path, entity2id, isTotal=True):
    if isTotal:
        dataset = pd.read_csv(os.path.join(file_path, 'whole_triplet.tsv'), sep='\t', header=None, names=['head', 'relation', 'tail'])
    else:
        dataset = pd.read_csv(os.path.join(file_path, 'gene_disease_triplet.tsv'), sep='\t', header=None, names=['head', 'relation', 'tail'])
    entities = pd.concat([dataset['head'], dataset['tail']], ignore_index=True)

    disease_entities = entities[entities.str.contains('Disease')].unique()
    gene_entities = entities[entities.str.contains('Gene')].unique()

    disease_entities_ids = sorted([entity2id[disease] for disease in disease_entities])
    gene_entities_ids = sorted([entity2id[gene] for gene in gene_entities])

    return gene_entities_ids, disease_entities_ids


# Merge gene and disease embedding from 3 parts
class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.q = nn.Linear(input_dim, input_dim)
        self.k = nn.Linear(input_dim, input_dim)
        self.v = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn_weights = self.softmax(torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5))
        return torch.matmul(attn_weights, v)

## 1.fill empty with zero
def get_full_embeddings(entity_ids, emb_dict, dim):
    """
    将嵌入字典补齐到完整实体集合，缺失的实体补零向量。
    """
    full_emb = {}
    zero_vector = np.zeros(dim)
    for entity_id in entity_ids:
        full_emb[entity_id] = emb_dict.get(entity_id, zero_vector)
    return full_emb

## 2.convert to matrix
def dict_to_matrix(emb_dict, entity_ids):
    """
    将嵌入字典转换为矩阵，行对应实体，列为嵌入向量。
    """
    return np.array([emb_dict[eid] for eid in entity_ids])

## 3.principal component dimension reduction
def reduce_dim(embedding, entity_ids, target_dim):
    """
    使用PCA对嵌入矩阵降维。
    """
    # emb_dim = next(iter(embedding.values())).shape[0]
    # full_emb = get_full_embeddings(entity_ids, embedding, emb_dim)
    # emb_matrix = dict_to_matrix(full_emb, entity_ids)
    emb_matrix = embedding
    emb_dim = emb_matrix.shape[1]

    mms = MinMaxScaler(feature_range=(0,1))
    emb_matrix_scaled = mms.fit_transform(emb_matrix)
    if emb_dim > target_dim:
        pca = PCA(n_components=target_dim, random_state=42)
        reduced_matrix_scaled = pca.fit_transform(emb_matrix_scaled)
        reduced_matrix_scaled1 = mms.fit_transform(reduced_matrix_scaled)
        return reduced_matrix_scaled1
    else:
        return emb_matrix_scaled
   
def get_merged_embeddings(kge_ent_emb, kge_rel_emb, entity2id, relation2id,
                           gene_feat_emb, disease_feat_emb, gene_ids, disease_ids, target_dim=300):
    mms = MinMaxScaler(feature_range=(0,1))
    
    entity_ids = disease_ids + gene_ids # the entity ids order is [disease,gene]
    # kge_ent_emb = {id: kge_ent_emb[id] for name, id in entity2id.items()}
    kge_ent_emb = kge_ent_emb[:len(entity_ids)]
    kge_ent_reduced = reduce_dim(kge_ent_emb, entity_ids, target_dim)

    disease_feat_reduced = reduce_dim(disease_feat_emb, disease_ids, target_dim)
    gene_feat_reduced = reduce_dim(gene_feat_emb, gene_ids, target_dim)
    feat_reduced = np.concatenate([disease_feat_reduced, gene_feat_reduced])

    combined_ent_emb = np.concatenate([kge_ent_reduced, feat_reduced], axis=1)
    combined_ent_emb_scaled = mms.fit_transform(combined_ent_emb)
    # combined_ent_emb_scaled_dict = {eid: combined_ent_emb_scaled[idx] for idx, eid in enumerate(entity_ids)}

    relation_ids = sorted(list(relation2id.values()))
    # kge_rel_emb = {id: kge_rel_emb[id] for name, id in relation2id.items()}
    kge_rel_emb = kge_rel_emb[:len(relation_ids)]
    kge_rel_reduced = reduce_dim(kge_rel_emb, relation_ids, target_dim)
    # rel_emb_scaled_dict = {eid: kge_rel_reduced[idx] for idx, eid in enumerate(relation_ids)}

    return combined_ent_emb_scaled, kge_rel_reduced

def get_merged_embeddings_direct(kge_ent_emb, kge_rel_emb, entity2id, relation2id,
                           gene_feat_emb, disease_feat_emb, gene_ids, disease_ids, target_dim=1500):
    mms = MinMaxScaler(feature_range=(0,1))
    
    entity_ids = disease_ids + gene_ids # the entity ids order is [disease,gene]
    # kge_ent_emb = {id: kge_ent_emb[id] for name, id in entity2id.items()}
    kge_ent_emb = kge_ent_emb[:len(entity_ids)]
    kge_ent_reduced = reduce_dim(kge_ent_emb, entity_ids, target_dim)

    disease_feat_reduced = reduce_dim(disease_feat_emb, disease_ids, target_dim)
    gene_feat_reduced = reduce_dim(gene_feat_emb, gene_ids, target_dim)
    # feat_reduced = np.concatenate([disease_feat_reduced, gene_feat_reduced])

    combined_disease_emb = np.concatenate([kge_ent_reduced[:len(disease_ids)], disease_feat_reduced], axis=1)
    combined_disease_emb_scaled = mms.fit_transform(combined_disease_emb)
    combined_disease_emb_scaled_dict = {eid: combined_disease_emb_scaled[idx] for idx, eid in enumerate(disease_ids)}
    combined_gene_emb = np.concatenate([kge_ent_reduced[len(disease_ids):], gene_feat_reduced], axis=1)
    combined_gene_emb_scaled = mms.fit_transform(combined_gene_emb)
    combined_gene_emb_scaled_dict = {eid: combined_gene_emb_scaled[idx] for idx, eid in enumerate(gene_ids)}

    combined_ent_emb_scaled_dict = {**combined_disease_emb_scaled_dict, **combined_gene_emb_scaled_dict}

    relation_ids = sorted(list(relation2id.values()))
    # kge_rel_emb = {id: kge_rel_emb[id] for name, id in relation2id.items()}
    kge_rel_emb = kge_rel_emb[:len(relation_ids)]
    kge_rel_reduced = reduce_dim(kge_rel_emb, relation_ids, target_dim)
    rel_emb_scaled_dict = {eid: kge_rel_reduced[idx] for idx, eid in enumerate(relation_ids)}

    return combined_ent_emb_scaled_dict, rel_emb_scaled_dict

def get_merged_embeddings_noDiseaseFeature(kge_ent_emb, kge_rel_emb, entity2id, relation2id, gene_feat_emb, gene_ids, disease_ids, target_dim=300):
    mms = MinMaxScaler(feature_range=(0,1))
    
    entity_ids = disease_ids + gene_ids # the entity ids order is [disease,gene]
    # kge_ent_emb = {id: kge_ent_emb[id] for name, id in entity2id.items()}
    kge_ent_emb = kge_ent_emb[:len(entity_ids)]
    kge_ent_reduced = reduce_dim(kge_ent_emb, entity_ids, target_dim)

    #disease_feat_reduced = reduce_dim(disease_feat_emb, disease_ids, target_dim)
    gene_feat_reduced = reduce_dim(gene_feat_emb, gene_ids, target_dim)
    # feat_reduced = np.concatenate([disease_feat_reduced, gene_feat_reduced])

    #combined_disease_emb = np.concatenate([kge_ent_reduced[:len(disease_ids)], disease_feat_reduced], axis=1)
    #combined_disease_emb_scaled = mms.fit_transform(combined_disease_emb)
    nocombined_disease_emb_scaled_dict = {eid: kge_ent_reduced[idx] for idx, eid in enumerate(disease_ids)}
    combined_gene_emb = np.concatenate([kge_ent_reduced[len(disease_ids):], gene_feat_reduced], axis=1)
    combined_gene_emb_scaled = mms.fit_transform(combined_gene_emb)
    combined_gene_emb_scaled_dict = {eid: combined_gene_emb_scaled[idx] for idx, eid in enumerate(gene_ids)}

    combined_ent_emb_scaled_dict = {**nocombined_disease_emb_scaled_dict, **combined_gene_emb_scaled_dict}

    relation_ids = sorted(list(relation2id.values()))
    # kge_rel_emb = {id: kge_rel_emb[id] for name, id in relation2id.items()}
    kge_rel_emb = kge_rel_emb[:len(relation_ids)]
    kge_rel_reduced = reduce_dim(kge_rel_emb, relation_ids, target_dim)
    rel_emb_scaled_dict = {eid: kge_rel_reduced[idx] for idx, eid in enumerate(relation_ids)}

    return combined_ent_emb_scaled_dict, rel_emb_scaled_dict

def get_merged_embeddings_noTargetFeature(kge_ent_emb, kge_rel_emb, entity2id, relation2id, disease_feat_emb, gene_ids, disease_ids, target_dim=300):
    mms = MinMaxScaler(feature_range=(0,1))
    
    entity_ids = disease_ids + gene_ids # the entity ids order is [disease,gene]
    # kge_ent_emb = {id: kge_ent_emb[id] for name, id in entity2id.items()}
    kge_ent_emb = kge_ent_emb[:len(entity_ids)]
    kge_ent_reduced = reduce_dim(kge_ent_emb, entity_ids, target_dim)

    disease_feat_reduced = reduce_dim(disease_feat_emb, disease_ids, target_dim)
    #gene_feat_reduced = reduce_dim(gene_feat_emb, gene_ids, target_dim)
    # feat_reduced = np.concatenate([disease_feat_reduced, gene_feat_reduced])

    combined_disease_emb = np.concatenate([kge_ent_reduced[:len(disease_ids)], disease_feat_reduced], axis=1)
    combined_disease_emb_scaled = mms.fit_transform(combined_disease_emb)
    nocombined_disease_emb_scaled_dict = {eid: combined_disease_emb_scaled for idx, eid in enumerate(disease_ids)}
    #combined_gene_emb = np.concatenate([kge_ent_reduced[len(disease_ids):], gene_feat_reduced], axis=1)
    #combined_gene_emb_scaled = mms.fit_transform(combined_gene_emb)
    nocombined_gene_emb_scaled_dict = {eid: kge_ent_reduced[idx] for idx, eid in enumerate(gene_ids)}

    combined_ent_emb_scaled_dict = {**nocombined_disease_emb_scaled_dict, **nocombined_gene_emb_scaled_dict}

    relation_ids = sorted(list(relation2id.values()))
    # kge_rel_emb = {id: kge_rel_emb[id] for name, id in relation2id.items()}
    kge_rel_emb = kge_rel_emb[:len(relation_ids)]
    kge_rel_reduced = reduce_dim(kge_rel_emb, relation_ids, target_dim)
    rel_emb_scaled_dict = {eid: kge_rel_reduced[idx] for idx, eid in enumerate(relation_ids)}
    
    return combined_ent_emb_scaled_dict, rel_emb_scaled_dict

def get_merged_embeddings_att(kge_ent_emb, kge_rel_emb, entity2id, relation2id,
                           gene_feat_emb, disease_feat_emb, gene_ids, disease_ids, target_dim=300):
    mms = MinMaxScaler(feature_range=(0,1))
    
    entity_ids = disease_ids + gene_ids # the entity ids order is [disease,gene]
    # kge_ent_emb = {id: kge_ent_emb[id] for name, id in entity2id.items()}
    kge_ent_emb = kge_ent_emb[:len(entity_ids)]
    kge_ent_reduced = reduce_dim(kge_ent_emb, entity_ids, target_dim)
    # kge_ent_emb_dict = {eid: kge_ent_reduced[idx] for idx, eid in enumerate(entity_ids)}

    disease_feat_reduced = reduce_dim(disease_feat_emb, disease_ids, target_dim)
    gene_feat_reduced = reduce_dim(gene_feat_emb, gene_ids, target_dim)
    feat_reduced = np.concatenate([disease_feat_reduced, gene_feat_reduced])
    feat_reduced_scaled = mms.fit_transform(feat_reduced)
    # feat_reduced_scaled_dict = {eid: feat_reduced_scaled[idx] for idx, eid in enumerate(entity_ids)}

    relation_ids = sorted(list(relation2id.values()))
    # kge_rel_emb = {id: kge_rel_emb[id] for name, id in relation2id.items()}
    kge_rel_emb = kge_rel_emb[:len(relation_ids)]
    kge_rel_reduced = reduce_dim(kge_rel_emb, relation_ids, target_dim)
    # rel_emb_scaled_dict = {eid: kge_rel_reduced[idx] for idx, eid in enumerate(relation_ids)}

    return kge_ent_reduced, feat_reduced_scaled, kge_rel_reduced

def get_merged_embeddings_onlyKGE(kge_ent_emb, kge_rel_emb, entity2id, relation2id, gene_ids, disease_ids, target_dim=300):
    mms = MinMaxScaler(feature_range=(0,1))
    
    entity_ids = disease_ids + gene_ids # the entity ids order is [disease,gene]
    # kge_ent_emb = {id: kge_ent_emb[id] for name, id in entity2id.items()}
    kge_ent_emb = kge_ent_emb[:len(entity_ids)]
    kge_ent_reduced = reduce_dim(kge_ent_emb, entity_ids, target_dim)
    # ent_emb_scaled_dict = {eid: kge_ent_reduced[idx] for idx, eid in enumerate(entity_ids)}

    relation_ids = sorted(list(relation2id.values()))
    # kge_rel_emb = {id: kge_rel_emb[id] for name, id in relation2id.items()}
    kge_rel_emb = kge_rel_emb[:len(relation_ids)]
    kge_rel_reduced = reduce_dim(kge_rel_emb, relation_ids, target_dim)
    # rel_emb_scaled_dict = {eid: kge_rel_reduced[idx] for idx, eid in enumerate(relation_ids)}
    
    return kge_ent_reduced, kge_rel_reduced

def get_merged_embeddings_onlyFeat(kge_rel_emb, relation2id, gene_feat_emb, disease_feat_emb, gene_ids, disease_ids, target_dim=300):
    mms = MinMaxScaler(feature_range=(0,1))
    
    entity_ids = disease_ids + gene_ids # the entity ids order is [disease,gene]

    disease_feat_reduced = reduce_dim(disease_feat_emb, disease_ids, target_dim)
    gene_feat_reduced = reduce_dim(gene_feat_emb, gene_ids, target_dim)
    feat_reduced = np.concatenate([disease_feat_reduced, gene_feat_reduced])
    feat_reduced_scaled = mms.fit_transform(feat_reduced)
    # ent_emb_scaled_dict = {eid: feat_reduced_scaled[idx] for idx, eid in enumerate(entity_ids)}

    relation_ids = sorted(list(relation2id.values()))
    rel_emb_dim = kge_rel_emb.shape[1]
    random_rel_emb = np.random.randn(len(relation_ids), rel_emb_dim)
    random_rel_emb_scaled = mms.fit_transform(random_rel_emb)
    # rel_emb_scaled_dict = {eid: random_rel_emb_scaled[idx] for idx, eid in enumerate(relation_ids)}

    return feat_reduced_scaled, random_rel_emb_scaled

def get_merged_embeddings_empty(kge_ent_emb, kge_rel_emb, entity2id, relation2id, gene_ids, disease_ids, target_dim=300):
    mms = MinMaxScaler(feature_range=(0,1))
    
    entity_ids = disease_ids + gene_ids # the entity ids order is [disease,gene]
    ent_emb_dim = kge_ent_emb.shape[1]
    random_ent_emb = np.random.rand(len(entity_ids), ent_emb_dim)
    random_ent_emb_scaled = mms.fit_transform(random_ent_emb)
    # random_ent_emb_scaled_dict = {eid: random_ent_emb_scaled[idx] for idx, eid in enumerate(entity_ids)}

    relation_ids = sorted(list(relation2id.values()))
    rel_emb_dim = kge_rel_emb.shape[1]
    random_rel_emb = np.random.randn(len(relation_ids), rel_emb_dim)
    random_rel_emb_scaled = mms.fit_transform(random_rel_emb)
    # random_rel_emb_scaled_dict = {eid: random_rel_emb_scaled[idx] for idx, eid in enumerate(relation_ids)}

    return random_ent_emb_scaled, random_rel_emb_scaled



def save_mlp_embeddings(mlp_model_res_dir, entity_embeddings, relation_embeddings):
    if isinstance(entity_embeddings, dict):
        with open(os.path.join(mlp_model_res_dir, f"entity_embedding.pkl"), 'wb') as f:
            pickle.dump(entity_embeddings, f)
        with open(os.path.join(mlp_model_res_dir, f"relation_embedding.pkl"), 'wb') as f:
            pickle.dump(relation_embeddings, f)
    else:
        np.save(os.path.join(mlp_model_res_dir, f"entity_embedding.npy"), entity_embeddings)
        np.save(os.path.join(mlp_model_res_dir, f"relation_embedding.npy"), relation_embeddings)

# Calculate evalution metrics
## classification metrics
def roc_auc(y, pred):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc

def pr_auc(y, pred):
    precision, recall, thresholds = metrics.precision_recall_curve(y, pred)
    pr_auc = metrics.auc(recall, precision)
    return pr_auc

def calc_classi_metrics(model, test_data_tf, test_data):
    test_data_triplet =test_data_tf.triples
    score = model.predict_hrt(test_data_tf.mapped_triples).sigmoid().detach().cpu().numpy() #

    test_data_score = pd.DataFrame(np.concatenate([test_data_triplet, score.reshape(-1, 1)], axis=1), columns=['head', 'relation', 'tail', 'score'])
    test_data_label_score = pd.merge(test_data, test_data_score, on=['head', 'relation', 'tail'], how='inner')

    y = test_data_label_score['label'].astype(float).values
    pred = test_data_label_score['score'].astype(float).values

    roc = roc_auc(y, pred)
    pr = pr_auc(y, pred)

    # 计算 Accuracy
    preds = (pred > 0.5).astype(int)  # 预测标签，阈值设为 0.5
    acc = metrics.accuracy_score(y, preds)

    return roc, pr, acc

from pykeen.evaluation import RankBasedEvaluator
def print_pykeen_metrics(pipeline_result, test_data_tf, test_data, emb_trainning, emb_validation, emb_testing):
    final_model = pipeline_result.model

    # Calculate Ranking metrics
    evaluator = RankBasedEvaluator()
    test_results = evaluator.evaluate(
        model=final_model,
        mapped_triples=test_data_tf.mapped_triples,
        additional_filter_triples=[
            emb_trainning.mapped_triples,
            emb_validation.mapped_triples,
            emb_testing.mapped_triples,
        ],
        use_tqdm=False,
    )

    # Calculate Classification metrics
    auc_roc, auc_pr, acc = calc_classi_metrics(final_model, test_data_tf, test_data)

    print(f"Hits@1={pipeline_result.get_metric('hits@1'):.4f}")
    print(f"Hits@3={pipeline_result.get_metric('hits@3'):.4f}")
    print(f"Hits@10={pipeline_result.get_metric('hits@10'):.4f}")
    print(f"MRR={pipeline_result.get_metric('mean_reciprocal_rank'):.4f}")
    print(f"MR={pipeline_result.get_metric('mean_rank'):.4f}\n")

    print(f"Hits@1-subGD={test_results.get_metric('hits@1'):.4f}")
    print(f"Hits@3-subGD={test_results.get_metric('hits@3'):.4f}")
    print(f"Hits@10-subGD={test_results.get_metric('hits@10'):.4f}")
    print(f"MRR-subGD={test_results.get_metric('mean_reciprocal_rank'):.4f}")
    print(f"MR-subGD={test_results.get_metric('mean_rank'):.4f}\n")

    print(f"AUC_ROC={auc_roc:.4f}")
    print(f"AUC_PR={auc_pr:.4f}")
    print(f"ACC={acc:.4f}")

# store pykeen embeddings
def save_embeddings(model, model_res_dir):
    entity_representation_modules = model.entity_representations
    relation_representation_modules = model.relation_representations
    entity_embeddings = entity_representation_modules[0]
    relation_embeddings = relation_representation_modules[0]
    entity_embedding_tensor = entity_embeddings()
    relation_embedding_tensor = relation_embeddings()
    entity_embedding_numpy = entity_embedding_tensor.detach().cpu().numpy()
    relation_embedding_numpy = relation_embedding_tensor.detach().cpu().numpy()

    np.save(f'{model_res_dir}/entity_embedding.npy', entity_embedding_numpy)
    np.save(f'{model_res_dir}/relation_embedding.npy', relation_embedding_numpy)
