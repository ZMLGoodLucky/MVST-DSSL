# MVST-DSSL: A multi-view spatio-temporal dual self-supervised learning model for traffic prediction

**MVST-DSSL** is an advanced traffic forecasting model built upon the **BasicTS** framework. It extends the vanilla Transformer architecture by incorporating a **Multi-View Spatial Attention (MVSA)** mechanism, designed to capture complex node interactions from geographic, semantic, and pivotal perspectives simultaneously.



## 🌟 Key Innovations: Multi-View Spatial Attention (MVSA)

The model moves beyond static physical adjacency by processing spatial features through three parallel attention branches:

1.  **Local Geometric View:** Uses the physical adjacency matrix $A_{geo}$ (1-hop neighbors with self-loops) to capture direct road connectivity.
2.  **Global Semantic View:** Dynamically calculates $A_{sem}$ based on the cosine similarity of node time-series embeddings, identifying distant nodes with similar traffic patterns.
3.  **Pivotal Node View:** Identifies "hub" nodes via degree centrality to build $A_{piv}$, modeling traffic pressure propagation through critical infrastructure.

These views are fused using a **1x1 Convolution** layer (Fusion Conv) and integrated into a Transformer block with **Residual Connections** and **Layer Normalization**.

---

## 📂 Project Structure

* `DSTRformer.py`: Main model architecture and `MultiViewSpatialAttention` implementation.
* `mlp.py`: Utility classes for `MultiLayerPerceptron` and `GraphMLP`.
* `baselines/DSTRformer/`: Dataset-specific configuration files (PEMS, METR-LA, etc.).
* `scripts/data_preparation/`: Data preprocessing scripts for standard benchmarks.

---

## ⚙️ Model Hyperparameters

The following parameters define the model's capacity and spatio-temporal embedding logic:

| Parameter | Description |
| :--- | :--- |
| `num_nodes` | Number of sensors/nodes in the traffic network. |
| `input_dim` | Dimension of the input features (e.g., flow, occupancy). |
| `output_dim` | Dimension of the prediction output. |
| `input_embedding_dim` | Dimension for the initial linear projection of raw data. |
| `tod_embedding_dim` | Embedding size for Time-of-Day (daily periodicity). |
| `dow_embedding_dim` | Embedding size for Day-of-Week (weekly periodicity). |
| `ts_embedding_dim` | Dimension for the Conv2d-based time-series embedding. |
| `time_embedding_dim` | Embedding size for combined (ToD + DoW * 7) features. |
| `adaptive_embedding_dim`| Dimension for learnable node-specific adaptive embeddings. |
| `model_dim` | Total hidden dimension (Sum of all embeddings). |
| `node_dim` | Hidden dimension for the adaptive graph/node encoding. |
| `feed_forward_dim` | Hidden dimension of the FFN within Transformer layers. |
| `num_heads` | Number of multi-head attention heads. |
| `num_layers` | Number of stacked Spatio-Temporal Transformer layers. |
| `num_layers_m` | Number of layers for the auto-regressive (AR) attention. |
| `mlp_num_layers` | Number of layers in the graph fusion MLP. |
| `dropout` | Dropout rate for regularization. |
| `k_similar` | Number of Top-K nodes for Global Semantic Adjacency. |
| `k_ratio` | Ratio of nodes identified as hubs in Pivotal Node Module. |
| `use_mixed_proj` | Boolean: Use Mixed Projection vs. Temporal Projection for output. |

---

## 🛠️ Environment Setup

### 1. Requirements (`requirements.txt`)
```text
easy-torch==1.3.2
easydict==1.10
pandas==1.3.5
packaging==23.1
setuptools==59.5.0
scipy==1.7.3
tables==3.7.0
sympy==1.10.1
setproctitle==1.3.2
scikit-learn==1.0.2

## 2. Installation

```bash
pip install -r requirements.txt
pip install timm
```

## 📊 Data Preparation

Run the generation scripts for the respective datasets before starting training:

```bash
# PEMS08
python scripts/data_preparation/PEMS08/generate_training_data.py

# PEMS04
python scripts/data_preparation/PEMS04/generate_training_data.py

# PEMS03
python scripts/data_preparation/PEMS03/generate_training_data.py

# PEMS07
python scripts/data_preparation/PEMS07/generate_training_data.py

# PEMS-BAY
python scripts/data_preparation/PEMS-BAY/generate_training_data.py

# METR-LA
python scripts/data_preparation/METR-LA/generate_training_data.py
```

## 🚀 Training & Execution

### PEMS08
```bash
python experiments/train.py -c baselines/DSTRformer/PEMS08.py --gpus '0' && shutdown
python experiments/train.py --cfg baselines/DSTRformer/PEMS08.py --gpus "0,1"
```

### PEMS04
```bash
python experiments/train.py -c baselines/DSTRformer/PEMS04.py --gpus '0' && shutdown
```

### PEMS03
```bash
python experiments/train.py -c baselines/DSTRformer/PEMS03.py --gpus '0'
```

### PEMS07
```bash
python experiments/train.py -c baselines/DSTRformer/PEMS07.py --gpus '0'
python experiments/train.py --cfg baselines/DSTRformer/PEMS07.py --gpus "0,1"
```

### PEMS-BAY
```bash
python experiments/train.py -c baselines/DSTRformer/PEMS-BAY.py --gpus '0'
python experiments/train.py --cfg baselines/DSTRformer/PEMS-BAY.py --gpus "0,1"
```

### METR-LA
```bash
python experiments/train.py -c baselines/DSTRformer/METR-LA.py --gpus '0' && shutdown
```

## 📈 Evaluation & Analysis

To generate data for comparative plots or ablation studies:

```bash
python generate_plot_data_ablated.py
```

