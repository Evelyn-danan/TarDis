<h1 align="center"> TarDis </h1>


<img src="model.png" alt="model"  width="90%" align="center" />


TarDis generates protein sequence embeddings (G<sub>f</sub>,)using ESM-2 (650Mparameters), graph structural embeddings (G<sub>k</sub>, R and D<sub>k</sub>) for disease nodetarget-disease relationship, and target node using CompGCN, and diseasesemantic embeddings (D<sub>f</sub>,) using BiBERT, Before training the MLP model.TarDis fuses these embeddings for diseases and targets into G and Drespectively, which are then fed into the MLP network to infer target-diseaseassociations.


## Installation
### Dependency

The codes have been tested in the following environment:
Package  | Version
--- | ---
Python | 3.9.21
PyTorch | 2.1.2
CUDA | 12.1
Pykeen | 1.11.0 
Pandas | 2.2.3
NumPy | 1.26.3
### Install via conda yaml file
```bash
conda env create -f pykeen.yml
conda activate pykeen
```

## Dataset

Please refer to [`README.md`](./data/README.md) in the `data` folder.

## How to run

### **Training Stage**

The training process of the TarDis model is divided into two stages: knowledge graph embedding and multilayer perceptron.

In the knowledge graph embedding training stage, we use PyKEEN to train the CompGCN model. Users can follow the cmd in run_kge_hpo.sh.
```bash
python -u train_pykeen_hpo.py --model CompGCN --gpu 0 --epochs_kge 500 --batch_size_kge 10000 
--emb_dim_kge 20 --lr_kge 0.001 --save_emb 1 --checkpoint_freq 60 --dir_name kge_hpo
```
In the multilayer perceptron training stage, users can follow the cmd in run_mlp_hpo.sh.
```bash
python -u train_mlp_run.py --gpu 0 --epochs_mlp 1000 --batch_size_mlp 10000 --emb_dim_kge 20
--emb_dim_feat 100 --lr_mlp 0.0001 --dropout_mlp 0.3 --hidden_dims_mlp '1024-1024-1024'
--num_heads_layers 2 --mlp_mode merged --save_emb 1 --dir_name 'mlp_ablation_merged'
--early_stop_patience 10
```

You can find the training results of the TarDis model at 
[Zenodo](https://zenodo.org/records/17156565).


### **Inference Stage**

During the inference stage, Users can follow the cmd in mlp_predict.ipynb or mlp_predict.py.

## **Baseline**

To establish a baseline comparison for the performance of the TarDis model, this study employs knowledge graph embedding methods provided by PyKEEN as baseline models. You can run the following command on GPUs to train baseline models. You can find the training results of baseline models at 
[Zenodo](https://zenodo.org/records/17156565).
```bash
#Training the CompGCN model using GPU 0.
bash run_kge_baseline_rescal_conve.sh CompGCN 0

#Training the RESCAL model using GPU 0.
bash run_kge_baseline_rescal_conve.sh RESCAL 0

#Training the ConvE model using GPU 0.
bash run_kge_baseline_rescal_conve.sh ConvE 0

#Training the TransE model using GPU 0.
bash run_kge_baseline_rescal_conve.sh TransE 0

#Training the TransR model using GPU 0.
bash run_kge_baseline_rescal_conve.sh TransR 0

#Training the RotatE model using GPU 0.
bash run_kge_baseline_rescal_conve.sh RotatE 0

#Training the DistMult model using GPU 0.
bash run_kge_baseline_rescal_conve.sh DistMult 0

#Training the ComplEX model using GPU 0.
bash run_kge_baseline_rescal_conve.sh ComplEX 0
```
