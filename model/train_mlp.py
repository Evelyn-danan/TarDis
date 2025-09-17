import os
import argparse
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pathlib import Path
import pystow

from torch.utils.tensorboard import SummaryWriter
import logging
from tqdm import tqdm, trange
from datetime import datetime
import time

from utils import load_IDMapping, load_train_data, load_entity_feature, load_gene_disease_ids, print_pykeen_metrics, save_embeddings
from utils import get_merged_embeddings, get_merged_embeddings_att, get_merged_embeddings_onlyKGE, get_merged_embeddings_onlyFeat
from utils import get_merged_embeddings_direct, get_merged_embeddings_empty, save_mlp_embeddings
from mlp_model import GeneDiseaseDataset, MLPScoringModel, train_mlp, test_mlp
from mlp_model import GeneDiseaseDataset_Att, MLPScoringModel_Att, train_mlp_att, test_mlp_att

# # 设置 PYSTOW_HOME 为自定义目录
# os.environ['PYSTOW_HOME'] = '/home/worker/users/ZC/KnowledgeGraph/TarKG_reason/pykeen_logs'
# pykeen_directory = pystow.join('pykeen')

def parse_args():
    parser = argparse.ArgumentParser(description='KGE-MLP fused model')
    
    parser.add_argument("-data", "--data", default="../data/dataset", help="data directory")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--save_emb", type=int, default=1)
    parser.add_argument("--dir_name", type=str, default="")
    
    # below are parameters for KGE model
    parser.add_argument("--model", type=str, default='CompGCN')
    parser.add_argument("--emb_dim_kge", type=int, default=20)
    parser.add_argument("--epochs_kge", type=int, default=500)
    parser.add_argument("--batch_size_kge", type=int, default=10000)
    parser.add_argument("--lr_kge", type=float, default=0.001)
    parser.add_argument("--checkpoint_freq", type=int, default=60)

    # below are parameters for MLP model
    parser.add_argument("--emb_dim_feat", type=int, default=200)
    parser.add_argument("--epochs_mlp", type=int, default=1000)
    parser.add_argument("--batch_size_mlp", type=int, default=10000)
    parser.add_argument("--lr_mlp", type=float, default=0.00005)
    parser.add_argument("--dropout_mlp", type=float, default=0.4)
    parser.add_argument("--hidden_dims_mlp", type=str, default="1024-1024-1024")
    parser.add_argument("--num_heads_layers", type=int, default=2)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--mlp_mode", type=str, choices=["merged", "mergedAtt", "onlyKGE", "onlyFeat", "direct", "empty", "noDiseaseFeature", "noTargetFeature"], 
                        default="merged", help="MLP input mode: 'merged', 'mergedAtt', 'direct', 'empty', 'onlyKGE', 'onlyFeat', "noDiseaseFeature", or"noTargetFeature")
    args = parser.parse_args()

    return args

args = parse_args()
print(args)

train_res_dir = "./train_results"
tb_dir = "./tb_logs"
ckp_dir = "./checkpoints"
if args.dir_name != "":
    train_res_dir = f"./train_results/{args.dir_name}"
    tb_dir = f"./tb_logs/{args.dir_name}"
    ckp_dir = f"./checkpoints/{args.dir_name}"
    os.makedirs(train_res_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(ckp_dir, exist_ok=True)

current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
print(f"Current time: {current_time}")
print(f"Current PID: {os.getpid()}")
print("Start training time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print()

# KGE model store path
# proj_name = f"{args.model}_{current_time}"
proj_name = "CompGCN_20250109-215102"
model_res = os.path.join("./train_results", proj_name) #dir: store training results
tb_log = os.path.join(tb_dir, proj_name) #dir: store tensorboard logs
best_model_path = Path(os.path.join(ckp_dir, f"{proj_name}_best-model-weights.pt")) #file: record best model of early stopping

# MLP model store path
mlp_proj_name = f"MLP_{args.mlp_mode}_{current_time}"
mlp_model_res = os.path.join(train_res_dir, mlp_proj_name)
os.makedirs(mlp_model_res, exist_ok=True)
mlp_tb_log = os.path.join(tb_dir, mlp_proj_name)
mlp_best_model_path = os.path.join(mlp_model_res, f"best_MLPmodel.pth")
mlp_final_model_path = os.path.join(mlp_model_res, f"final_MLPmodel.pth")
# mlp_Entemb_path = os.path.join(mlp_model_res, f"entity_embedding.pkl")
# mlp_Relemb_path = os.path.join(mlp_model_res, f"relation_embedding.pkl")

columns = ['head', 'relation', 'tail']
entity2id, relation2id = load_IDMapping(args.data)
support_triplet = pd.read_csv(args.data + "/support_triplet.tsv", sep='\t', names=columns)
print(f" num_support_triplets: {len(support_triplet)}")

def train_pykeen(train_data, test_data):
    train_data_pos = train_data[train_data['label'] == 1]
    train_data_pos = train_data_pos[columns]

    emb_graph = pd.concat([train_data_pos, support_triplet], ignore_index=True)

    # Create a TriplesFactory from the DataFrame
    create_inverse_triples = False
    if args.model in ['CompGCN', 'ConvE']:
        create_inverse_triples = True
    emb_triplet = TriplesFactory.from_labeled_triples(emb_graph[columns].values, entity_to_id=entity2id, relation_to_id=relation2id, create_inverse_triples=create_inverse_triples)
    emb_trainning, emb_testing, emb_validation = emb_triplet.split([0.8, 0.1, 0.1])
    print(f" num_emb_triplets: {emb_triplet.num_triples}")
    print(f" emb_train_triplets: {emb_trainning.num_triples}, emb_test_triplets: {emb_testing.num_triples}, emb_valid_triplets: {emb_validation.num_triples}")

    test_data_tf = TriplesFactory.from_labeled_triples(test_data[columns].values, entity_to_id=entity2id, relation_to_id=relation2id, create_inverse_triples=False)

    # Create a model
    pipeline_result = pipeline(
        random_seed=42,

        model=args.model,
        model_kwargs=dict(embedding_dim=args.emb_dim_kge,),

        training=emb_trainning,
        testing=emb_testing,
        validation=emb_validation,
        # dataset_kwargs=dict(create_inverse_triples=True),

        training_loop='sLCWA',
        training_kwargs=dict(
            num_epochs=args.epochs_kge,
            batch_size=args.batch_size_kge,
            use_tqdm_batch=False,
            checkpoint_name=f'{proj_name}_checkpoint.pt',
            checkpoint_directory=ckp_dir,
            checkpoint_frequency=args.checkpoint_freq,
            checkpoint_on_failure=True,
            # sampler='schlichtkrull',
        ),
        
        optimizer='Adam',
        optimizer_kwargs=dict(lr=args.lr_kge,),

        # loss='marginranking',
        # loss_kwargs=dict(margin=1),

        negative_sampler='bernoulli',
        negative_sampler_kwargs=dict(num_negs_per_pos=1,),

        evaluator='RankBasedEvaluator',
        evaluator_kwargs=dict(filtered=True,),

        stopper='early',
        stopper_kwargs=dict(
            frequency=20,
            patience=2,
            relative_delta=0.005,
            best_model_path=best_model_path,
        ),
        
        device=torch.device(f"cuda:{args.gpu}"),

        result_tracker='tensorboard',
        result_tracker_kwargs=dict(
            experiment_path=tb_log,
        ),
    )
    pipeline_result.save_to_directory(model_res)
    final_model = pipeline_result.model

    # Print evaluation metrics
    print_pykeen_metrics(pipeline_result, test_data_tf, test_data, emb_trainning, emb_validation, emb_testing)

    # Save result embedding
    if args.save_emb:
        save_embeddings(final_model, model_res_dir=model_res)

    return final_model

def train_MLP(train_data_id, test_data_id, entity_embeddings, relation_embeddings, input_dim):
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter(log_dir=mlp_tb_log)

    # 数据加载器
    train_dataset = GeneDiseaseDataset(train_data_id, entity_embeddings, relation_embeddings)
    test_dataset = GeneDiseaseDataset(test_data_id, entity_embeddings, relation_embeddings)
    train_loader = DataLoader(train_dataset, args.batch_size_mlp, shuffle=True)
    test_loader = DataLoader(test_dataset, args.batch_size_mlp, shuffle=False)

    # 模型初始化
    hidden_dims = [int(dim) for dim in args.hidden_dims_mlp.split('-')]
    model = MLPScoringModel(input_dim=input_dim, hidden_dims=hidden_dims, dropout_rate=args.dropout_mlp).to(device)
    criterion = nn.BCELoss()  # 二分类交叉熵损失
    optimizer = optim.Adam(model.parameters(), args.lr_mlp)

    # 早停初始化
    best_auc_roc = -np.inf  # 初始化最好的AUC_ROC为负无穷
    best_auc_pr = -np.inf  # 初始化最好的AUC_PR为负无穷
    epochs_no_improve = 0  # 没有改善的轮次
    early_stop_patience = args.early_stop_patience  # 设定耐心轮次
    best_epoch = 0

    # 训练与测试
    # for epoch in trange(1, (args.epochs_mlp+1), desc="Epoch", position=0):
    for epoch in range(1, (args.epochs_mlp+1)):
        train_loss = train_mlp(model, train_loader, optimizer, criterion, device)
        writer.add_scalar("MLP/Loss", train_loss, epoch)
        print(f"Epoch {epoch}/{args.epochs_mlp}: Loss={train_loss:.4f}")

        if epoch % 10 == 0:
            auc_roc, auc_pr, acc = test_mlp(model, test_loader, device)
            writer.add_scalar("MLP/ROC_AUC", auc_roc, epoch)
            writer.add_scalar("MLP/PR_AUC", auc_pr, epoch)
            writer.add_scalar("MLP/ACC", acc, epoch)
            print(f"Epoch {epoch}/{args.epochs_mlp}: AUC_ROC={auc_roc:.4f}, AUC_PR={auc_pr:.4f}, ACC={acc:.4f}")

            if auc_roc > best_auc_roc:
                best_auc_roc = auc_roc
                best_auc_pr = auc_pr
                epochs_no_improve = 0
                best_epoch = epoch
                # 保存当前最好的模型
                print(f"Current best model saved at epoch {epoch}: AUC_ROC={best_auc_roc:.4f}, AUC_PR={best_auc_pr:.4f}, ACC={acc:.4f}")
                torch.save({'model': model, 'state_dict': model.state_dict(), 'epoch': best_epoch}, mlp_best_model_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    print(f"Early stopping triggered at epoch {epoch}.")
                    break
    
    print(f"Best model saved at epoch {best_epoch}: AUC_ROC={best_auc_roc:.4f}, AUC_PR={best_auc_pr:.4f}, ACC={acc:.4f}")
    model = model.to('cpu')
    torch.save({'model': model, 'state_dict': model.state_dict(), 'epoch': epoch}, mlp_final_model_path)

    if args.save_emb:
        save_mlp_embeddings(mlp_model_res, entity_embeddings, relation_embeddings)

def train_MLP_Att(train_data_id, test_data_id, entity_kge_embeddings, entity_feat_embeddings, relation_embeddings):
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter(log_dir=mlp_tb_log)

    # 数据加载器
    train_dataset = GeneDiseaseDataset_Att(train_data_id, entity_kge_embeddings, entity_feat_embeddings, relation_embeddings)
    test_dataset = GeneDiseaseDataset_Att(test_data_id, entity_kge_embeddings, entity_feat_embeddings, relation_embeddings)
    train_loader = DataLoader(train_dataset, args.batch_size_mlp, shuffle=True)
    test_loader = DataLoader(test_dataset, args.batch_size_mlp, shuffle=False)

    # 模型初始化
    hidden_dims = [int(dim) for dim in args.hidden_dims_mlp.split('-')]
    model = MLPScoringModel_Att(kge_dim=args.emb_dim_kge, attr_dim=args.emb_dim_feat, hidden_dims=hidden_dims,
                                 dropout_rate=args.dropout_mlp, num_heads=args.num_heads_layers, num_layers=args.num_heads_layers).to(device)
    criterion = nn.BCELoss()  # 二分类交叉熵损失
    optimizer = optim.Adam(model.parameters(), args.lr_mlp)

    # 早停初始化
    best_auc_roc = -np.inf  # 初始化最好的AUC_ROC为负无穷
    best_auc_pr = -np.inf  # 初始化最好的AUC_PR为负无穷
    epochs_no_improve = 0  # 没有改善的轮次
    early_stop_patience = args.early_stop_patience  # 设定耐心轮次
    best_epoch = 0

    # 训练与测试
    # for epoch in trange(1, (args.epochs_mlp+1), desc="Epoch", position=0):
    for epoch in range(1, (args.epochs_mlp+1)):
        train_loss = train_mlp_att(model, train_loader, optimizer, criterion, device)
        writer.add_scalar("MLP/Loss", train_loss, epoch)
        print(f"Epoch {epoch}/{args.epochs_mlp}: Loss={train_loss:.4f}")

        if epoch % 10 == 0:
            auc_roc, auc_pr, acc = test_mlp_att(model, test_loader, device)
            writer.add_scalar("MLP/ROC_AUC", auc_roc, epoch)
            writer.add_scalar("MLP/PR_AUC", auc_pr, epoch)
            writer.add_scalar("MLP/ACC", acc, epoch)
            print(f"Epoch {epoch}/{args.epochs_mlp}: AUC_ROC={auc_roc:.4f}, AUC_PR={auc_pr:.4f}, ACC={acc:.4f}")

            if auc_roc > best_auc_roc:
                best_auc_roc = auc_roc
                best_auc_pr = auc_pr
                epochs_no_improve = 0
                best_epoch = epoch
                # 保存当前最好的模型
                print(f"Current best model saved at epoch {epoch}: AUC_ROC={best_auc_roc:.4f}, AUC_PR={best_auc_pr:.4f}, ACC={acc:.4f}")
                torch.save({'model': model, 'state_dict': model.state_dict(), 'epoch': best_epoch}, mlp_best_model_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    print(f"Early stopping triggered at epoch {epoch}.")
                    break

    print(f"Best model saved at epoch {best_epoch}: AUC_ROC={best_auc_roc:.4f}, AUC_PR={best_auc_pr:.4f}, ACC={acc:.4f}")
    model = model.to('cpu')
    torch.save({'model': model, 'state_dict': model.state_dict(), 'epoch': epoch}, mlp_final_model_path)

    if args.save_emb:
        # with open(os.path.join(mlp_model_res, 'entity_kge_embeddings.pkl'), 'wb') as f:
        #     pickle.dump(entity_kge_embeddings, f)
        # with open(os.path.join(mlp_model_res, 'entity_feat_embeddings.pkl'), 'wb') as f:
        #     pickle.dump(entity_feat_embeddings, f)
        # with open(os.path.join(mlp_model_res, 'relation_embeddings.pkl'), 'wb') as f:
        #     pickle.dump(relation_embeddings, f)
        np.save(os.path.join(mlp_model_res, 'entity_kge_embedding.npy'), entity_kge_embeddings)
        np.save(os.path.join(mlp_model_res, 'entity_feat_embedding.npy'), entity_feat_embeddings)
        np.save(os.path.join(mlp_model_res, 'relation_embedding.npy'), relation_embeddings)

def train(i):
    train_data, test_data, train_data_id, test_data_id = load_train_data(args.data, entity2id, relation2id, i)

    print()
    print("Start training KGE model embedding...")
    start_time = time.time()
    # emb_model = train_pykeen(train_data, test_data)
    kge_ent_emb = np.load(f'{model_res}/entity_embedding.npy')
    kge_rel_emb = np.load(f'{model_res}/relation_embedding.npy')
    end_time = time.time()
    print(f"Finishing training KGE model embedding, time cost: {end_time - start_time:.4f} seconds\n")

    print("Loading gene and disease feature embeddings...\n")
    gene_feat_emb, disease_feat_emb = load_entity_feature(args.data)
    gene_ids, disease_ids = load_gene_disease_ids(args.data, entity2id)

    print("Merging embeddings...\n")
    start_time = time.time()
    if args.mlp_mode == "merged":
        print("Start training MLP scoring model with merged embedding...")
        entity_embeddings, relation_embeddings = get_merged_embeddings(kge_ent_emb, kge_rel_emb, entity2id, relation2id,
                                                                    gene_feat_emb, disease_feat_emb, gene_ids, disease_ids, args.emb_dim_feat)
        input_dim = args.emb_dim_kge*3 + args.emb_dim_feat*2
        print(f"MLP input dimension: {input_dim}")
        train_MLP(train_data_id, test_data_id, entity_embeddings, relation_embeddings, input_dim)
    elif args.mlp_mode == "onlyKGE":
        print("Start training MLP scoring model with only KGE embedding...")
        entity_embeddings, relation_embeddings = get_merged_embeddings_onlyKGE(kge_ent_emb, kge_rel_emb, entity2id, relation2id,
                                                                            gene_ids, disease_ids, args.emb_dim_feat)
        input_dim = args.emb_dim_kge*3
        print(f"MLP input dimension: {input_dim}")
        train_MLP(train_data_id, test_data_id, entity_embeddings, relation_embeddings, input_dim)
    elif args.mlp_mode == "onlyFeat":
        print("Start training MLP scoring model with only Feat embedding...")
        entity_embeddings, relation_embeddings = get_merged_embeddings_onlyFeat(kge_rel_emb, relation2id, gene_feat_emb,
                                                                            disease_feat_emb, gene_ids, disease_ids, args.emb_dim_feat)
        input_dim = args.emb_dim_feat*2 + args.emb_dim_kge
        print(f"MLP input dimension: {input_dim}")
        train_MLP(train_data_id, test_data_id, entity_embeddings, relation_embeddings, input_dim)
    elif args.mlp_mode == "mergedAtt":
        print("Start training MLP scoring model with attention-merged embedding...")
        entity_kge_embeddings, entity_feat_embeddings, relation_embeddings = get_merged_embeddings_att(kge_ent_emb, kge_rel_emb, entity2id, relation2id,
                                                                        gene_feat_emb, disease_feat_emb, gene_ids, disease_ids, args.emb_dim_feat)
        train_MLP_Att(train_data_id, test_data_id, entity_kge_embeddings, entity_feat_embeddings, relation_embeddings)
    elif args.mlp_mode == "direct":
        print("Start training MLP scoring model with direct merged embedding...")
        entity_embeddings, relation_embeddings = get_merged_embeddings_direct(kge_ent_emb, kge_rel_emb, entity2id, relation2id,
                                                                            gene_feat_emb, disease_feat_emb, gene_ids, disease_ids, 1500)
        input_dim = args.emb_dim_kge*3 + 768 + 1280
        print(f"MLP input dimension: {input_dim}")
        train_MLP(train_data_id, test_data_id, entity_embeddings, relation_embeddings, input_dim)
    elif args.mlp_mode == "empty":
        print("Start training MLP scoring model with random embedding...")
        entity_embeddings, relation_embeddings = get_merged_embeddings_empty(kge_ent_emb, kge_rel_emb, entity2id, relation2id,
                                                                            gene_ids, disease_ids, args.emb_dim_feat)
        input_dim = args.emb_dim_kge*3
        print(f"MLP input dimension: {input_dim}")
        train_MLP(train_data_id, test_data_id, entity_embeddings, relation_embeddings, input_dim)
    elif args.mlp_mode == 'noTargetFeature':
        print("Start training MLP scoring model without target feature...")
        logging.info("Start training MLP scoring model without target feature...")
        entity_embeddings, relation_embeddings = get_merged_embeddings_noTargetFeature(kge_ent_emb, kge_rel_emb, entity2id, relation2id, disease_feat_emb, gene_ids, disease_ids, args.emb_dim_feat)
        input_dim = args.emb_dim_kge*3 + args.emb_dim_feat
        print(f"MLP input dimension: {input_dim}")
        logging.info(f"MLP input dimension: {input_dim}")
        train_MLP(train_data_id, test_data_id, entity_embeddings, relation_embeddings, input_dim)
    elif args.mlp_mode == 'noDiseaseFeature':
        print("Start training MLP scoring model without disease feature...")
        logging.info("Start training MLP scoring model without disease feature...")
        entity_embeddings, relation_embeddings = get_merged_embeddings_noDiseaseFeature(kge_ent_emb, kge_rel_emb, entity2id, relation2id, gene_feat_emb, gene_ids, disease_ids, args.emb_dim_feat)
        input_dim = args.emb_dim_kge*3 + args.emb_dim_feat
        print(f"MLP input dimension: {input_dim}")
        logging.info(f"MLP input dimension: {input_dim}")
        train_MLP(train_data_id, test_data_id, entity_embeddings, relation_embeddings, input_dim)
    end_time = time.time()
    print(f"Finishing training MLP scoring model, time cost: {end_time - start_time:.4f} seconds\n")

if __name__ == '__main__':
    start_time = time.time()

    for i in range(1):
        train(i)
    
    end_time = time.time()
    print("Finish training time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print(f"Total time cost: {end_time - start_time:.4f} seconds")
