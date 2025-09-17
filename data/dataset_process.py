import os
import sys
import time
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold

if len(sys.argv) > 1 and sys.argv[1] == "demo":
    source_dir = "./data_demo"
    target_dir = "./dataset_demo"
else:
    source_dir = "./data"
    target_dir = "./dataset"

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# create a ID dictionary for entity and relation
def assign_priority(rgroup):
    entity_pair = rgroup.split('::')[0]
    if entity_pair == "Disease:Gene":
        return 0
    elif entity_pair == "Gene:Disease":
        return 1
    else:
        return 2

def generate_id_dict(relation_df: pd.DataFrame, ent_df:pd.DataFrame):
    ent_map = ent_df[['index', 'id', 'kind']].copy(deep=True)
    ent_map.loc[:, 'index'] = ent_map['index'].astype(int) - 1
    ent_map.loc[:, 'map_id'] = ent_map['kind'] + '::' + ent_map['id']
    ent_map[['index', 'map_id']].to_csv(os.path.join(target_dir, "entity_map.dict"), sep='\t', index=False, header=False)

    relation_map = relation_df[['rgroup']].drop_duplicates().copy(deep=True)
    relation_map['priority'] = relation_map['rgroup'].apply(assign_priority)
    relation_map = relation_map.sort_values(by=['priority', 'rgroup'], ascending=[True, True], ignore_index=True)
    relation_map.insert(0, 'rel_index', relation_map.index)
    relation_map[['rel_index', 'rgroup']].to_csv(os.path.join(target_dir, "relation_map.dict"), sep='\t', index=False, header=False)

    print(" generate entity_map.dict and relation_map.dict successfully!")
    print(" the number of entity_map and relation_map is", len(ent_map), len(relation_map))

# generate triplet set
def generate_GeneDisease_set(triplet_set, dis_feat, dis_filter=False):
    gene_disease_triplet = triplet_set[triplet_set['rgroup'].str.contains('^Gene:Disease|^Disease:Gene')].copy(deep=True)
    
    if dis_filter:
        dis_feat = dis_feat[dis_feat['def'].notnull()].drop_duplicates('name')
        dis_ids = dis_feat['id'].tolist()
        gene_disease_triplet = gene_disease_triplet[((gene_disease_triplet['node1_type'] == 'Disease') & (gene_disease_triplet['node1'].isin(dis_ids))) |
                                ((gene_disease_triplet['node2_type'] == 'Disease') & (gene_disease_triplet['node2'].isin(dis_ids)))]

    return gene_disease_triplet

def generate_triplet_set(total_kg: pd.DataFrame, dis_feat, dis_filter=False):
    triplet_set = total_kg.copy(deep=True)
    triplet_set_dict = {}

    triplet_set.loc[:, 'head'] = triplet_set['node1_type'] + '::' + triplet_set['node1']
    triplet_set.loc[:, 'relation'] = triplet_set['rgroup']
    triplet_set.loc[:, 'tail'] = triplet_set['node2_type'] + '::' + triplet_set['node2']
    triplet_set.loc[:, 'label'] = 1

    triplet_set_dict['whole_triplet'] = triplet_set.copy(deep=True)
    triplet_set_dict['support_triplet'] = triplet_set[~triplet_set['rgroup'].str.contains('^Gene:Disease|^Disease:Gene')].copy(deep=True)
    triplet_set_dict['gene_disease_triplet'] = generate_GeneDisease_set(triplet_set, dis_feat, dis_filter)
    # triplet_set_dict['gene_triplet'] = triplet_set[triplet_set['rgroup'].str.startswith('Gene:Gene')].copy(deep=True)
    # triplet_set_dict['disease_triplet'] = triplet_set[triplet_set['rgroup'].str.startswith('Disease:Disease')].copy(deep=True)

    store_info = ['head', 'relation', 'tail'] #, 'direction', 'label'

    print(' ---------------------------------------------------')
    for key in triplet_set_dict.keys():
        triplet_set_dict[key][store_info].to_csv(os.path.join(target_dir, f'{key}.tsv'), sep='\t', index=False, header=False)
        print(' The number of triplets in %s: %s' % (key, len(triplet_set_dict[key])))
    print(' ---------------------------------------------------')

# generate train set and test set
def generate_train_test_set(fold_num: int=5):
    datafold_path = os.path.join(target_dir, 'data_folds')
    if not os.path.exists(datafold_path):
        os.makedirs(datafold_path)

    dataset = pd.read_csv(os.path.join(target_dir, 'gene_disease_triplet.tsv'), sep='\t', header=None, names=['head', 'relation', 'tail'])

    # 提取关系列作为分层依据
    relations = dataset['relation']

    # 初始化StratifiedKFold
    skf = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=42)

    # 进行分层抽样并保存结果
    for fold_idx, (train_index, test_index) in enumerate(skf.split(dataset, relations)):
        train_triples, test_triples = dataset.iloc[train_index], dataset.iloc[test_index]
        
        # 保存训练集和测试集到TSV文件
        train_file = f'train_fold_{fold_idx+1}.tsv'
        test_file = f'test_fold_{fold_idx+1}.tsv'
        
        train_triples.to_csv(os.path.join(datafold_path, train_file), sep='\t', index=False, header=False)
        test_triples.to_csv(os.path.join(datafold_path, test_file), sep='\t', index=False, header=False)
        
        print(f"    Fold {fold_idx + 1} saved:")
        print(f"    Train file: {train_file}")
        print(f"    Test file: {test_file}")
        print("="*50)

# generate train set and test set with negative samples
def calculate_relation_specific_ratios_bern(dataset):
    """Calculate head/tail entity replacement probabilities (tph/hpt) for each relation."""
    # 初始化数据结构
    relation_head = {}
    relation_tail = {}
    relation_tph = {}
    relation_hpt = {}

    # 构建relation_head和relation_tail字典
    for _, row in dataset.iterrows():
        h_ = row['head'] # h_ = entity2id[row['head']]
        t_ = row['tail'] # t_ = entity2id[row['tail']]
        r_ = row['relation'] # r_ = relation2id[row['relation']]
        
        # 统计每个关系的头实体数量
        if r_ in relation_head:
            relation_head[r_][h_] = relation_head[r_].get(h_, 0) + 1
        else:
            relation_head[r_] = {h_: 1}

        # 统计每个关系的尾实体数量
        if r_ in relation_tail:
            relation_tail[r_][t_] = relation_tail[r_].get(t_, 0) + 1
        else:
            relation_tail[r_] = {t_: 1}

    # 计算每个关系的 tph 和 hpt
    for r_ in relation_head:
        head_count = len(relation_head[r_])  # 不同头实体的数量
        total_head_instances = sum(relation_head[r_].values())  # 所有头实体的出现次数
        tph = total_head_instances / head_count
        relation_tph[r_] = tph

    for r_ in relation_tail:
        tail_count = len(relation_tail[r_])  # 不同尾实体的数量
        total_tail_instances = sum(relation_tail[r_].values())  # 所有尾实体的出现次数
        hpt = total_tail_instances / tail_count
        relation_hpt[r_] = hpt

    # 根据 tph 和 hpt 确定替换概率
    relation_probs = {}
    for r_ in relation_tph:
        tph = relation_tph[r_]
        hpt = relation_hpt.get(r_, 1)  # 默认值为1，避免除以0
        replace_head_prob = tph / (tph + hpt)
        relation_probs[r_] = replace_head_prob

    return relation_probs

def generate_negative_samples(positive_samples, known_triplets, relation_probs, disease_entities, gene_entities, num_negatives: int=1):
    """Generate negative samples based on head/tail entity ratios per relation, avoiding conflicts with positive samples."""
    neg_samples = []
    
    for head, relation, tail in positive_samples.values:
        if head.startswith('Disease'):
            head_set = disease_entities
            tail_set = gene_entities
        else:
            head_set = gene_entities
            tail_set = disease_entities

        replace_head_prob = relation_probs.get(relation, 0.5)  # Default to 0.5 if relation is missing
        for _ in range(num_negatives):
            if np.random.rand() < replace_head_prob:  # Use head_ratio to decide whether to replace head or tail
                # Replace head
                new_head = np.random.choice(head_set)
                while (new_head, relation, tail) in known_triplets or [new_head, relation, tail] in neg_samples:
                    new_head = np.random.choice(head_set)
                neg_samples.append([new_head, relation, tail])
            else:
                # Replace tail
                new_tail = np.random.choice(tail_set)
                while (head, relation, new_tail) in known_triplets or [head, relation, new_tail] in neg_samples:
                    new_tail = np.random.choice(tail_set)
                neg_samples.append([head, relation, new_tail])
    
    return pd.DataFrame(neg_samples, columns=['head', 'relation', 'tail'])

def generate_negative_samples_twoTuple(positive_samples, known_twoTuples, relation_probs, disease_entities, gene_entities, num_negatives: int=1):
    """Generate negative samples based on head/tail entity ratios per relation, avoiding conflicts with positive samples."""
    neg_samples = []
    
    for head, relation, tail in positive_samples.values:
        if head.startswith('Disease'):
            head_set = disease_entities
            tail_set = gene_entities
        else:
            head_set = gene_entities
            tail_set = disease_entities

        replace_head_prob = relation_probs.get(relation, 0.5)  # Default to 0.5 if relation is missing
        for _ in range(num_negatives):
            if np.random.rand() < replace_head_prob:  # Use head_ratio to decide whether to replace head or tail
                # Replace head
                new_head = np.random.choice(head_set)
                while (new_head, tail) in known_twoTuples or [new_head, relation, tail] in neg_samples:
                    new_head = np.random.choice(head_set)
                neg_samples.append([new_head, relation, tail])
            else:
                # Replace tail
                new_tail = np.random.choice(tail_set)
                while (head, new_tail) in known_twoTuples or [head, relation, new_tail] in neg_samples:
                    new_tail = np.random.choice(tail_set)
                neg_samples.append([head, relation, new_tail])
    
    return pd.DataFrame(neg_samples, columns=['head', 'relation', 'tail'])

def generate_train_test_set_with_negatives(fold_num: int=5, num_negatives: int=1):
    datafold_path = os.path.join(target_dir, 'data_folds')
    if not os.path.exists(datafold_path):
        os.makedirs(datafold_path)

    dataset = pd.read_csv(os.path.join(target_dir, 'gene_disease_triplet.tsv'), sep='\t', header=None, names=['head', 'relation', 'tail'])
    
    # 创建正样本集合，用于检测负样本冲突
    # known_triplets = set(map(tuple, dataset.values))
    known_twoTuples = set(zip(dataset['head'], dataset['tail']))

    # 筛选出Disease和Gene类型的实体
    entities = pd.concat([dataset['head'], dataset['tail']], ignore_index=True)
    disease_entities = entities[entities.str.contains('Disease')].unique()
    gene_entities = entities[entities.str.contains('Gene')].unique()
    
    # 进行分层抽样并保存结果，以关系列作为分层依据
    relations = dataset['relation']
    relation_probs = calculate_relation_specific_ratios_bern(dataset)
    skf = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=42)
    for fold_idx, (train_index, test_index) in enumerate(skf.split(dataset, relations)):
        train_triples = dataset.iloc[train_index]
        test_triples = dataset.iloc[test_index]
        
        # 生成负样本并添加到训练集和测试集
        # train_negatives = generate_negative_samples(train_triples, known_triplets, relation_probs, disease_entities, gene_entities, num_negatives)
        # test_negatives = generate_negative_samples(test_triples, known_triplets, relation_probs, disease_entities, gene_entities, num_negatives)
        train_negatives = generate_negative_samples_twoTuple(train_triples, known_twoTuples, relation_probs, disease_entities, gene_entities, num_negatives)
        test_negatives = generate_negative_samples_twoTuple(test_triples, known_twoTuples, relation_probs, disease_entities, gene_entities, num_negatives)
        
        # 合并正负样本
        train_triples.loc[:, 'label'] = 1
        test_triples.loc[:, 'label'] = 1
        train_negatives.loc[:, 'label'] = 0
        test_negatives.loc[:, 'label'] = 0
        train_with_neg = pd.concat([train_triples, train_negatives], ignore_index=True)
        test_with_neg = pd.concat([test_triples, test_negatives], ignore_index=True)
        
        # 保存训练集和测试集到TSV文件
        train_file = f'train_fold_{fold_idx+1}.tsv'
        test_file = f'test_fold_{fold_idx+1}.tsv'
        
        train_with_neg.to_csv(os.path.join(datafold_path, train_file), sep='\t', index=False) #, header=False
        test_with_neg.to_csv(os.path.join(datafold_path, test_file), sep='\t', index=False) #, header=False
        
        print(f"    Fold {fold_idx + 1} saved:")
        print(f"    Train file with negatives: {train_file}  positve samples: {len(train_triples)}, negative samples: {len(train_negatives)}")
        print(f"    Test file with negatives: {test_file}  positve samples: {len(test_triples)}, negative samples: {len(test_negatives)}")
        print("="*100)

if __name__ == '__main__':
    whole_ent = pd.read_csv(os.path.join(source_dir, 'whole_ent.csv'), dtype=object)
    whole_kg = pd.read_csv(os.path.join(source_dir, 'whole_kg.csv'), dtype=object)
    if 'rgroup' not in whole_kg.columns:
        whole_kg.insert(1, 'rgroup', whole_kg['node1_type'] + ':' + whole_kg['node2_type'] + '::' + whole_kg['relation'])

    dis_feat = pd.read_csv(os.path.join(source_dir, 'disease_feature.csv'), dtype=object)
    
    print("Start Time:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print("Start generating ID dict of entity and relation...")
    generate_id_dict(whole_kg, whole_ent)

    print("Start generating triplet set...")
    generate_triplet_set(whole_kg, dis_feat, dis_filter=False)

    print("Start generating train/test set...")
    generate_train_test_set_with_negatives(fold_num=5, num_negatives=1)

    print("Preprocessing finished.")
    print("End Time:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))