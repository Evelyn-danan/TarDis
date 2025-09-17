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

from utils import load_IDMapping, load_train_data, print_pykeen_metrics, save_embeddings

# # 设置 PYSTOW_HOME 为自定义目录
# os.environ['PYSTOW_HOME'] = '/home/worker/users/ZC/KnowledgeGraph/TarKG_reason/pykeen_logs'
# pykeen_directory = pystow.join('pykeen')

def parse_args():
    parser = argparse.ArgumentParser(description='KGE-MLP fused model')
    
    parser.add_argument("-data", "--data", default="../data/dataset", help="data directory")
    parser.add_argument("--gpu", type=int, default=-1)
    
    # below are parameters for KGE model
    parser.add_argument("--model", type=str, default='CompGCN')
    parser.add_argument("--emb_dim_kge", type=int, default=100)
    parser.add_argument("--epochs_kge", type=int, default=500)
    parser.add_argument("--batch_size_kge", type=int, default=10000)
    parser.add_argument("--lr_kge", type=float, default=0.01)
    parser.add_argument("--checkpoint_freq", type=int, default=60)
    parser.add_argument("--save_emb", type=int, default=1)
    parser.add_argument("--dir_name", type=str, default="")

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

columns = ['head', 'relation', 'tail']
entity2id, relation2id = load_IDMapping(args.data)
support_triplet = pd.read_csv(args.data + "/support_triplet.tsv", sep='\t', names=columns)
print(f" num_support_triplets: {len(support_triplet)}")

proj_name = f"{args.model}_{current_time}"
model_res = os.path.join(train_res_dir, proj_name) #dir: store training results
tb_log = os.path.join(tb_dir, proj_name) #dir: store tensorboard logs
best_model_path = Path(os.path.join(ckp_dir, f"{proj_name}_best-model-weights.pt")) #file: record best model of early stopping

start_time = time.time()
for i in range(1):
    train_data, test_data, train_data_id, test_data_id = load_train_data(args.data, entity2id, relation2id, i)
    train_data_pos = train_data[train_data['label'] == 1]
    train_data_pos = train_data_pos[columns]

    emb_graph = pd.concat([train_data_pos, support_triplet], ignore_index=True)

    # Create a TriplesFactory from the DataFrame    
    create_inverse_triples = False
    if args.model == 'CompGCN':
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
            # checkpoint_name=f'{proj_name}_checkpoint.pt',
            # checkpoint_directory=ckp_dir,
            # checkpoint_frequency=args.checkpoint_freq,
            # checkpoint_on_failure=True,
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
            patience=1,
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

end_time = time.time()
print("Finish training time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print(f"Total time cost: {end_time - start_time:.4f} seconds")
