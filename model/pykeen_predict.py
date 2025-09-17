import os
import argparse
import numpy as np
import pandas as pd
import torch
from datetime import datetime
import time

from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pathlib import Path
import pystow

from utils import load_IDMapping, load_train_data, calc_classi_metrics

# # 设置 PYSTOW_HOME 为自定义目录
# os.environ['PYSTOW_HOME'] = '/home/worker/users/ZC/KnowledgeGraph/TarKG_reason/pykeen_logs'
# pykeen_directory = pystow.join('pykeen')

def parse_args():
    parser = argparse.ArgumentParser(description='KGE-MLP fused model')
    
    parser.add_argument("-data", "--data", default="../data/dataset", help="data directory")
    parser.add_argument("--gpu", type=int, default=-1)
    
    # below are parameters for KGE model
    parser.add_argument("--model", type=str, default='CompGCN')

    args = parser.parse_args()

    return args

args = parse_args()
print(args)

print("Start time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print(f"Current PID: {os.getpid()}")

columns = ['head', 'relation', 'tail']
entity2id, relation2id = load_IDMapping(args.data)
support_triplet = pd.read_csv(args.data + "/support_triplet.tsv", sep='\t', names=columns)
print(f" num_support_triplets: {len(support_triplet)}")

train_data, test_data, train_data_id, test_data_id = load_train_data(args.data, entity2id, relation2id, 0)
train_data_pos = train_data[train_data['label'] == 1]
train_data_pos = train_data_pos[columns]

#Create a TriplesFactory from the DataFrame
emb_graph = pd.concat([train_data_pos, support_triplet], ignore_index=True)
create_inverse_triples = False
if args.model in ['CompGCN', 'ConvE']:
    create_inverse_triples = True
emb_triplet = TriplesFactory.from_labeled_triples(emb_graph[columns].values, entity_to_id=entity2id, relation_to_id=relation2id, create_inverse_triples=create_inverse_triples)
emb_trainning, emb_testing, emb_validation = emb_triplet.split([0.8, 0.1, 0.1])
print(f" num_emb_triplets: {emb_triplet.num_triples}")
print(f" emb_train_triplets: {emb_trainning.num_triples}, emb_test_triplets: {emb_testing.num_triples}, emb_valid_triplets: {emb_validation.num_triples}")

test_data_tf = TriplesFactory.from_labeled_triples(test_data[columns].values, entity_to_id=entity2id, relation_to_id=relation2id, create_inverse_triples=False)

from pykeen.evaluation import RankBasedEvaluator
def calc_rank_metrics_GD(model, test_data_tf, entity2id):
    gd_ent = []
    gd_rel = []
    for key, value in entity2id.items():
        if key.startswith("Disease") or key.startswith("Gene"):
            gd_ent.append(value)
    for key, value in relation2id.items():
        if key.startswith("Disease:Gene") or key.startswith("Gene:Disease"):
            gd_rel.append(value)
    print(f"gd_ent: {len(gd_ent)}, gd_rel: {len(gd_rel)}")

    evaluator = RankBasedEvaluator()
    test_results = evaluator.evaluate(
        model=model,
        mapped_triples=test_data_tf.mapped_triples,
        restrict_entities_to=gd_ent,
        restrict_relations_to=gd_rel,
        additional_filter_triples=[
            emb_triplet.mapped_triples,
            # emb_trainning.mapped_triples,
            # emb_validation.mapped_triples,
            # emb_testing.mapped_triples,
        ],
        use_tqdm=True,
    )

    print(f"Hits@1-subGD={test_results.get_metric('hits@1'):.4f}")
    print(f"Hits@3-subGD={test_results.get_metric('hits@3'):.4f}")
    print(f"Hits@10-subGD={test_results.get_metric('hits@10'):.4f}")
    print(f"MRR-subGD={test_results.get_metric('mean_reciprocal_rank'):.4f}")
    print(f"MR-subGD={test_results.get_metric('mean_rank'):.4f}\n")

def calc_classi_metrics_GD(model, test_data_tf, test_data):
    auc_roc, auc_pr, acc = calc_classi_metrics(model, test_data_tf, test_data)

    print(f"AUC_ROC={auc_roc:.4f}")
    print(f"AUC_PR={auc_pr:.4f}")
    print(f"ACC={acc:.4f}")

if args.model == "CompGCN":
    model_res_dir = "/home/worker/users/ZC/KnowledgeGraph/TarKG_reason/pykeen_model/train_results/baseline/CompGCN_20250109-215102"
elif args.model == "TransE":
    model_res_dir = "/home/worker/users/ZC/KnowledgeGraph/TarKG_reason/pykeen_model/train_results/baseline/TransE_20250109-215115"
elif args.model == "TransR":
    model_res_dir = "/home/worker/users/ZC/KnowledgeGraph/TarKG_reason/pykeen_model/train_results/baseline/TransR_20250109-215206"
elif args.model == "RotatE":
    model_res_dir = "/home/worker/users/ZC/KnowledgeGraph/TarKG_reason/pykeen_model/train_results/baseline/RotatE_20250109-215430"
elif args.model == "RESCAL":
    model_res_dir = "/home/worker/users/ZC/KnowledgeGraph/TarKG_reason/pykeen_model/train_results/baseline/RESCAL_20250117-214900"
elif args.model == "DistMult":
    model_res_dir = "/home/worker/users/ZC/KnowledgeGraph/TarKG_reason/pykeen_model/train_results/baseline/DistMult_20250111-102406"
elif args.model == "ComplEx":
    model_res_dir = "/home/worker/users/ZC/KnowledgeGraph/TarKG_reason/pykeen_model/train_results/baseline/ComplEx_20250111-102451"
elif args.model == "ConvE":
    model_res_dir = "/home/worker/users/ZC/KnowledgeGraph/TarKG_reason/pykeen_model/train_results/baseline/ConvE_20250124-172107"

model = torch.load(os.path.join(model_res_dir, "trained_model.pkl"))
# calc_rank_metrics_GD(model, test_data_tf, entity2id)

# if args.model not in ['RESCAL', 'ConvE']:
#     calc_classi_metrics_GD(model, test_data_tf, test_data)

calc_classi_metrics_GD(model, test_data_tf, test_data)

print("End time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))