import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from .mlp import MultiLayerPerceptron, GraphMLP
import torch.nn.functional as F
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value, ):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
                             query @ key
                     ) / self.head_dim ** 0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)

        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)
        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
            self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False,num_nodes=170,global_rank =8,node_rank =12,alpha=1
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        # self.feed_forward = nn.Sequential(
        #     nn.Linear(model_dim, feed_forward_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(feed_forward_dim, model_dim),
        # )
        self.fc1 = NALL(
            in_features=model_dim,
            out_features=model_dim,
            num_nodes=170,
            global_rank=8,
            node_rank =12,
            alpha=1.0
        )
        self.argumented_linear = nn.Linear(model_dim, model_dim)
        self.act1 = nn.GELU()
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, y=None, dim=-2, c=None, augment=False):
        x = x.transpose(dim, -2)
        augmented = None
        # x: (batch_size, ..., length, model_dim)
        if c is not None:
            residual = c
        else:
            residual = x
        if y is None:
            out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
            if augment is True:
                augmented = self.act1(self.argumented_linear(residual))
        else:
            y = y.transpose(dim, -2)
            out = self.attn(y, x, x)

        out = self.dropout1(out)

        if augmented is not None and augment is not False:
            out = self.ln1(residual + out + augmented)
        else:
            out = self.ln1(residual + out)

        residual = out
        out = self.fc1(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


############################   多视角时空图注意力  ###################################

# ==============================================================================
# 1. 因果传播视角模块 (Causal Propagation View)
# ==============================================================================

class CausalPropagationAdjacency(nn.Module):
    """
    因果传播视角邻接矩阵生成器。
    核心创新：建模交通流的上下游因果链，而非简单的相关性。

    技术实现：
    1. 使用神经格兰杰因果发现识别有向依赖
    2. 引入时间延迟建模（考虑拥堵传播需要时间）
    3. 生成有向因果邻接矩阵
    """

    def __init__(self, model_dim, num_nodes, max_lag=3, threshold=0.1):
        """
        :param model_dim: 特征维度
        :param num_nodes: 节点数量
        :param max_lag: 最大时间滞后（考虑因果传播延迟）
        :param threshold: 因果强度阈值（用于稀疏化）
        """
        super().__init__()
        self.model_dim = model_dim
        self.num_nodes = num_nodes
        self.max_lag = max_lag
        self.threshold = threshold

        # ============ 模块1: 时间滞后编码器 ============
        # 对于每个时间滞后，学习不同的变换权重
        self.lag_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(model_dim, model_dim // 2),
                nn.ReLU(),
                nn.Linear(model_dim // 2, model_dim)
            ) for _ in range(max_lag + 1)  # 包括lag=0
        ])

        # ============ 模块2: 因果强度评估器 ============
        # 评估节点i对节点j的因果影响强度
        self.causal_scorer = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(model_dim, 1),
            nn.Sigmoid()  # 输出 [0, 1] 的因果强度
        )

        # ============ 模块3: 传播路径建模器 ============
        # 建模多跳传播（如 S1 → S3 → S5）
        self.path_attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        # ============ 模块4: 有向图正则化 ============
        # 确保生成的图是有向无环图（DAG）的软约束
        self.register_buffer('eye', torch.eye(num_nodes))

    def compute_granger_causality(self, x):
        """
        神经格兰杰因果检验。
        核心思想：如果加入节点i的历史信息能显著提升节点j的预测，
                 则认为i对j存在因果影响。

        :param x: 输入特征 (B, T, N, D)
        :return: 因果邻接矩阵 (B, N, N) - 有向图
        """
        B, T, N, D = x.shape

        # ========== 步骤1: 提取多时间滞后特征 ==========
        lag_features = []
        for lag_idx, encoder in enumerate(self.lag_encoders):
            if lag_idx < T:
                # 提取第 t-lag 时刻的特征
                lagged_x = x[:, max(0, T - lag_idx - 1), :, :]  # (B, N, D)
                encoded = encoder(lagged_x)  # (B, N, D)
                lag_features.append(encoded)

        # 将所有滞后特征堆叠并聚合
        lag_stack = torch.stack(lag_features, dim=1)  # (B, max_lag+1, N, D)
        # 使用平均池化聚合不同滞后
        aggregated_lag = lag_stack.mean(dim=1)  # (B, N, D)

        # ========== 步骤2: 计算成对因果强度 ==========
        # 对于每对节点 (i, j)，评估 i → j 的因果强度
        causal_adj = torch.zeros(B, N, N, device=x.device)

        for i in range(N):
            # 源节点 i 的特征
            source_feat = aggregated_lag[:, i:i + 1, :].expand(-1, N, -1)  # (B, N, D)
            # 目标节点 j 的特征
            target_feat = aggregated_lag  # (B, N, D)

            # 拼接源和目标特征
            pair_feat = torch.cat([source_feat, target_feat], dim=-1)  # (B, N, 2D)

            # 计算因果强度
            causal_strength = self.causal_scorer(pair_feat).squeeze(-1)  # (B, N)
            causal_adj[:, i, :] = causal_strength

        # ========== 步骤3: 稀疏化（保留强因果关系）==========
        # 应用阈值
        causal_adj = torch.where(
            causal_adj > self.threshold,
            causal_adj,
            torch.zeros_like(causal_adj)
        )

        # ========== 步骤4: 去除自环（节点不对自己产生因果）==========
        eye_expanded = self.eye.unsqueeze(0).expand(B, -1, -1)
        causal_adj = causal_adj * (1 - eye_expanded)

        return causal_adj

    def enhance_multi_hop_propagation(self, causal_adj, x):
        """
        增强多跳因果传播路径建模（如 S1 → S3 → S5）。

        :param causal_adj: 1跳因果邻接矩阵 (B, N, N)
        :param x: 节点特征 (B, T, N, D)
        :return: 增强的因果邻接矩阵 (B, N, N)
        """
        B, T, N, D = x.shape

        # 使用最后一个时间步的特征
        node_feat = x[:, -1, :, :]  # (B, N, D)

        # ========== 多跳传播建模 ==========
        # 计算 2-hop 和 3-hop 的因果传播
        causal_adj_2hop = torch.bmm(causal_adj, causal_adj)  # (B, N, N)
        causal_adj_3hop = torch.bmm(causal_adj_2hop, causal_adj)  # (B, N, N)

        # 加权融合不同跳数的传播
        # 越远的传播，权重越小（因果衰减）
        enhanced_adj = (
                1.0 * causal_adj +
                0.5 * causal_adj_2hop +
                0.25 * causal_adj_3hop
        )

        # 归一化到 [0, 1]
        max_val = enhanced_adj.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        enhanced_adj = enhanced_adj / (max_val + 1e-8)

        return enhanced_adj

    def forward(self, x):
        """
        :param x: 输入特征 (B, T, N, D)
        :return: 因果传播邻接矩阵 (B, N, N) - 有向图
        """
        # 计算基础因果邻接矩阵
        causal_adj = self.compute_granger_causality(x)

        # 增强多跳传播
        enhanced_causal_adj = self.enhance_multi_hop_propagation(causal_adj, x)

        return enhanced_causal_adj


# ==============================================================================
# 2. 有向图注意力层 (Directed Graph Attention Layer)
# ==============================================================================

class DirectedGraphAttentionLayer(nn.Module):
    """
    专为有向图设计的注意力层。
    与标准GAT不同，它区分"入边"和"出边"的影响。
    """

    def __init__(self, model_dim, num_heads, dropout):
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.model_dim = model_dim

        # 分别为入边和出边设计变换
        self.q_proj = nn.Linear(model_dim, model_dim)  # Query (接收方)
        self.k_proj_in = nn.Linear(model_dim, model_dim)  # Key (发送方 → 我)
        self.v_proj_in = nn.Linear(model_dim, model_dim)  # Value (发送方 → 我)

        self.out_proj = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, directed_adj):
        """
        :param x: 输入张量 (BT, N, D)
        :param directed_adj: 有向邻接矩阵 (BT, N, N) 或 (B, N, N)
                            directed_adj[i, j] 表示 j → i 的因果强度
        :return: 输出张量 (BT, N, D)
        """
        BT, N, D = x.shape

        # 1. 计算 Q, K, V
        q = self.q_proj(x).reshape(BT, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_in = self.k_proj_in(x).reshape(BT, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_in = self.v_proj_in(x).reshape(BT, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 2. 计算注意力分数（入边：谁影响我）
        scores = torch.matmul(q, k_in.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (BT, H, N, N)

        # 3. 应用有向邻接矩阵作为掩码
        # directed_adj[i, j] > 0 表示存在 j → i 的因果边
        if directed_adj.dim() == 2:
            adj_mask = (directed_adj == 0).unsqueeze(0).unsqueeze(0)
        elif directed_adj.dim() == 3:
            adj_mask = (directed_adj == 0).unsqueeze(1)
        else:
            raise ValueError("邻接矩阵维度错误")

        # 掩码处理
        scores = scores.masked_fill(adj_mask, float('-inf'))

        # 4. 计算注意力权重（使用有向图权重增强）
        attn_weights = F.softmax(scores, dim=-1)

        # *** 核心创新：用因果强度加权注意力 ***
        if directed_adj.dim() == 2:
            causal_weights = directed_adj.unsqueeze(0).unsqueeze(0)
        else:
            causal_weights = directed_adj.unsqueeze(1)
        attn_weights = attn_weights * causal_weights

        # 重新归一化
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)
        attn_weights = self.dropout(attn_weights)

        # 5. 计算上下文向量
        context = torch.matmul(attn_weights, v_in)
        context = context.permute(0, 2, 1, 3).reshape(BT, N, self.model_dim)
        output = self.out_proj(context)

        return output

# ==============================================================================
# 1. 视角邻接矩阵生成器 (View Adjacency Generators)
# ==============================================================================

class LocalGeometricAdjacency(nn.Module):
    """
    生成局部地理视角邻接矩阵 (A_geo)。
    当前实现：直接使用原始邻接矩阵 (代表1跳邻居) 并添加自环。
    """

    def __init__(self, k_hops=1):
        super().__init__()
        # 如果 k_hops > 1, 则需要实现图遍历算法 (如 BFS 或矩阵幂)
        self.k_hops = k_hops
        if self.k_hops != 1:
            print("警告: LocalGeometricAdjacency 当前仅实现 K=1 的情况。")

    def forward(self, adj):
        """
        :param adj: 原始邻接矩阵 (N, N)
        :return: 局部地理视角邻接矩阵 A_geo (N, N)
        """
        # 添加自环
        adj_geo = adj + torch.eye(adj.size(0), device=adj.device)
        # 确保是二值的 (或者保留权重，如果需要的话)
        adj_geo = (adj_geo > 0).float()
        return adj_geo


class GlobalSemanticAdjacency(nn.Module):
    """
    生成全局语义视角邻接矩阵 (A_sem)。
    基于节点时间序列的相似性构建。
    """

    def __init__(self, k_similar=10):
        super().__init__()
        self.k_similar = k_similar

    def forward(self, x):
        """
        :param x: 输入特征张量 (B, T, N, D)
        :return: 全局语义视角邻接矩阵 A_sem (N, N)
        """
        B, T, N, D = x.shape

        # 对时间和批次维度求平均，得到节点嵌入 (N, D)
        # 注意：对时间求平均可能会丢失动态信息。
        # 这里我们先对时间求平均，再对批次求平均，得到一个平均的节点画像。
        x_node = x.mean(dim=1)  # (B, N, D)
        x_mean = x_node.mean(dim=0)  # (N, D)

        # 计算余弦相似度
        sim = F.cosine_similarity(x_mean.unsqueeze(1), x_mean.unsqueeze(0), dim=-1)  # (N, N)

        # 确保 K 不大于 N
        k = min(self.k_similar, N)

        # 选择 Top-K 相似的节点
        _, top_k_indices = torch.topk(sim, k, dim=1)

        # 创建 A_sem, 只保留 Top-K 连接 (权重为相似度)
        adj_sem = torch.zeros_like(sim)
        adj_sem.scatter_(1, top_k_indices, sim.gather(1, top_k_indices))

        # 可选：使其对称 (A + A.T) / 2
        # adj_sem = (adj_sem + adj_sem.T) / 2

        # 添加自环
        adj_sem.fill_diagonal_(1.0)

        return adj_sem


class PivotalNodeIdentificationModule(nn.Module):
    """
    识别关键节点并生成关键节点视角邻接矩阵 (A_piv)。
    采用基于度中心性的方法识别关键节点，并构建一个星型连接图。
    """

    def __init__(self, embed_dim, k_ratio=0.2):
        super(PivotalNodeIdentificationModule, self).__init__()
        self.k_ratio = k_ratio
        # 注意：这里我们使用了一个简化的基于度中心性的版本。
        # 如果你想使用你提供的带 W 权重的版本，请确保 W 的计算方式正确。
        # 你的版本存在 W 维度不匹配的问题，需要修正。

    def forward(self, H, adj):
        """
        :param H: 输入特征张量 (B, T, N, D)
        :param adj: 原始邻接矩阵 (N, N)
        :return: 关键节点视角邻接矩阵 A_piv (N, N)
        """
        B, T, N, D = H.shape

        # 使用度中心性作为评分标准
        Score = adj.sum(dim=1)

        # 计算 K 值
        K = int(N * self.k_ratio)
        if K == 0: K = 1  # 确保至少有一个关键节点

        # 识别关键节点
        _, pivotal_nodes = torch.topk(Score, K)

        # 构建 A_piv: 将所有节点连接到关键节点
        A_piv = torch.zeros_like(adj)
        A_piv[pivotal_nodes, :] = 1
        A_piv[:, pivotal_nodes] = 1

        # 添加自环
        A_piv = A_piv + torch.eye(N, device=adj.device)
        A_piv = (A_piv > 0).float()  # 确保是二值的

        return A_piv


# ==============================================================================
# 2. 图注意力层 (Graph Attention Layer)
# ==============================================================================

class GraphAttentionLayer(nn.Module):
    """
    实现一个多头自注意力层，它将图结构（邻接矩阵）作为注意力偏置 (bias) 引入。
    *** 已修改以支持批处理邻接矩阵 ***
    """

    def __init__(self, model_dim, num_heads, dropout):
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.model_dim = model_dim

        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        """
        :param x: 输入张量 (Batch_Size * Time_Steps, Num_Nodes, Model_Dim)
        :param adj: 对应的邻接矩阵 (Batch_Size * Time_Steps, Num_Nodes, Num_Nodes)
                     或 (Num_Nodes, Num_Nodes)
        :return: 输出张量 (Batch_Size * Time_Steps, Num_Nodes, Model_Dim)
        """
        BT, N, D = x.shape

        # 1. 计算 Q, K, V
        q = self.q_proj(x).reshape(BT, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (BT, H, N, D_h)
        k = self.k_proj(x).reshape(BT, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (BT, H, N, D_h)
        v = self.v_proj(x).reshape(BT, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (BT, H, N, D_h)

        # 2. 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (BT, H, N, N)

        # 3. 应用邻接矩阵作为注意力偏置
        attn_bias = torch.zeros_like(scores)

        # *** 新增: 处理批处理或非批处理的 adj ***
        if adj.dim() == 2:  # (N, N) - 静态图
            adj_mask = (adj == 0).unsqueeze(0).unsqueeze(0)  # (1, 1, N, N) 用于广播
        elif adj.dim() == 3:  # (BT, N, N) - 动态图
            adj_mask = (adj == 0).unsqueeze(1)  # (BT, 1, N, N) 用于广播
        else:
            raise ValueError("邻接矩阵必须是 2 维 (N, N) 或 3 维 (BT, N, N)")
        # *** 结束新增 ***

        attn_bias.masked_fill_(adj_mask, float('-inf'))

        # 将偏置加到分数上
        scores = scores + attn_bias

        # 4. 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # (BT, H, N, N)
        attn_weights = self.dropout(attn_weights)

        # 5. 计算上下文向量
        context = torch.matmul(attn_weights, v)  # (BT, H, N, D_h)

        # 6. 重塑并输出
        context = context.permute(0, 2, 1, 3).reshape(BT, N, self.model_dim)  # (BT, N, D)
        output = self.out_proj(context)

        return output


# ==============================================================================
# 3. 多视角空间注意力模块 (MultiViewSpatialAttention)
# ==============================================================================

class MultiViewSpatialAttention(nn.Module):
    """
    实现多视角空间注意力机制。
    *** 已修改为包含四个并行分支 (地理、语义、关键节点、动态图) ***
    并使用 1x1 卷积融合它们的输出。
    """

    def __init__(self, model_dim, feed_forward_dim, num_heads, dropout, adj, k_similar=10, k_ratio=0.2, num_nodes=None, max_lag=3):
        super().__init__()
        self.adj = adj  # 存储原始邻接矩阵 (N, N)
        self.model_dim = model_dim
        self.num_nodes = num_nodes
        # 初始化邻接矩阵生成器
        self.geo_adj_gen = LocalGeometricAdjacency()
        self.sem_adj_gen = GlobalSemanticAdjacency(k_similar=k_similar)
        self.piv_adj_gen = PivotalNodeIdentificationModule(model_dim, k_ratio=k_ratio)

        # *** 新增：因果传播邻接矩阵生成器 ***
        self.causal_adj_gen = CausalPropagationAdjacency(
            model_dim=model_dim,
            num_nodes=self.num_nodes,
            max_lag=max_lag,
            threshold=0.1
        )

        # 初始化每个视角的图注意力层
        self.attn_geo = GraphAttentionLayer(model_dim, num_heads, dropout)
        self.attn_sem = GraphAttentionLayer(model_dim, num_heads, dropout)
        self.attn_piv = GraphAttentionLayer(model_dim, num_heads, dropout)
        # *** 新增：有向图注意力层用于因果传播视角 ***
        self.attn_causal = DirectedGraphAttentionLayer(model_dim, num_heads, dropout)

        # *** 修改: 融合层 (1x1 卷积) 处理 4 个输入 ***
        self.fusion_conv = nn.Conv2d(model_dim * 4, model_dim, kernel_size=(1, 1))

        # FFN 和 LayerNorm 部分 (保持不变)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dynamic_adj): # *** 修改: 增加 dynamic_adj 参数 ***
        """
        :param x: 输入张量 spatial_x (B, T, N, D)
        :param dynamic_adj: 动态邻接矩阵 (B, N, N)
        :return: 经过多视角空间注意力处理后的张量 (B, T, N, D)
        """
        B, T, N, D = x.shape
        adj_dev = self.adj.to(x.device)  # 确保邻接矩阵在正确的设备上
        # dynamic_adj_dev = dynamic_adj.to(x.device) # 确保动态邻接矩阵在正确的设备上

        # ================== 1. 计算多视角注意力输出 ==================
        residual1 = x

        # 1. 生成三个静态/半静态视角的邻接矩阵
        A_geo = self.geo_adj_gen(adj_dev) # (N, N)
        A_sem = self.sem_adj_gen(x)      # (N, N)
        A_piv = self.piv_adj_gen(x, adj_dev) # (N, N)

        # *** 新增：生成因果传播邻接矩阵 ***
        A_causal = self.causal_adj_gen(x)  # (B, N, N) - 有向！
        # 扩展为 (B*T, N, N)
        A_causal_expanded = A_causal.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, N, N)

        # 2. 重塑输入以适应注意力层 (B * T, N, D)
        x_reshaped = x.reshape(B * T, N, D)

        # 3. 分别通过四个视角的注意力层
        H_geo = self.attn_geo(x_reshaped, A_geo)  # (B*T, N, D)
        H_sem = self.attn_sem(x_reshaped, A_sem)  # (B*T, N, D)
        H_piv = self.attn_piv(x_reshaped, A_piv)  # (B*T, N, D)
        # *** 使用有向GAT处理因果传播视角 ***
        H_causal = self.attn_causal(x_reshaped, A_causal_expanded)  # (B*T, N, D)

        # 4. 融合结果
        # *** 修改: 沿特征维度拼接 4 个视图 ***
        H_cat = torch.cat([H_geo, H_sem, H_piv, H_causal], dim=-1)  # (B*T, N, D*4)
        # 恢复原始形状
        H_cat_reshaped = H_cat.reshape(B, T, N, D * 4)  # (B, T, N, D*4)

        # 5. 使用 1x1 卷积进行融合
        H_cat_permuted = H_cat_reshaped.permute(0, 3, 1, 2)  # (B, D*4, T, N)
        H_fused_permuted = self.fusion_conv(H_cat_permuted)  # (B, D, T, N)
        H_fused = H_fused_permuted.permute(0, 2, 3, 1)  # (B, T, N, D)

        # ================== 2. Add & Norm 1 ==================
        x = residual1 + self.dropout1(H_fused)
        x = self.norm1(x)

        # ================== 3. FFN + Add & Norm 2 ==================
        residual2 = x
        ffn_output = self.ffn(x)
        x = residual2 + self.dropout2(ffn_output)
        x = self.norm2(x)
        return x

################################  结束   ##########################################

################################生成动态图##############################

class DynamicGraphGenerator(nn.Module):
    def __init__(self, num_nodes, input_dim, node_embedding_dim, alpha=3.0):
        """
        Args:
            num_nodes (int): 节点数量.
            input_dim (int): 输入特征维度 (例如，流量).
            node_embedding_dim (int): 用于计算相似度的节点嵌入维度.
            alpha (float): 用于 softmax 缩放的超参数.
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.node_embedding_dim = node_embedding_dim
        self.alpha = alpha

        # 学习每个节点的静态嵌入 (作为基础)
        self.static_node_emb = nn.Parameter(torch.randn(num_nodes, node_embedding_dim))

        # 线性层，将输入特征映射到动态调整向量
        self.dynamic_proj = nn.Linear(input_dim, node_embedding_dim)

    def forward(self, x_flow, tod_emb=None, dow_emb=None):
        """
        生成动态图.
        Args:
            x_flow (torch.Tensor): 流量数据 (B, L, N, D_flow).
            tod_emb (torch.Tensor, optional): 时间点嵌入 (B, L, N, D_tod).
            dow_emb (torch.Tensor, optional): 星期几嵌入 (B, L, N, D_dow).

        Returns:
            torch.Tensor: 动态邻接矩阵 (B, N, N) or (B, L, N, N).
                          这里我们生成一个基于序列平均的图 (B, N, N) 以简化.
        """
        batch_size, seq_len, _, _ = x_flow.shape

        # 1. 计算基于流量的动态嵌入 (取序列平均或最后一个时间步)
        dynamic_node_state = torch.mean(x_flow, dim=1)  # (B, N, D_flow)
        dynamic_adjustment = self.dynamic_proj(dynamic_node_state)  # (B, N, D_node_emb)

        # 2. 结合静态嵌入和动态调整 (可以加入周期性特征)
        # 简单起见，我们这里只用 静态 + 动态
        # 你可以实验加入 tod_emb 和 dow_emb 的均值或投影
        node_embeddings = self.static_node_emb.unsqueeze(0).expand(batch_size, -1, -1) + dynamic_adjustment

        # 3. 计算节点相似度 (例如，点积 + ReLU/Softmax)
        node_embeddings_t = node_embeddings.transpose(1, 2)  # (B, D_node_emb, N)
        adj = F.relu(torch.bmm(node_embeddings, node_embeddings_t))  # (B, N, N) - 使用 ReLU 保持非负性

        # 4. (可选) 应用 Softmax 进行归一化 (类似 GAT)
        # adj = F.softmax(adj * self.alpha, dim=-1)

        # 5. (可选) 设对角线为0，避免自环 (或根据需要保留)
        # mask = torch.eye(self.num_nodes, device=adj.device).bool().unsqueeze(0)
        # adj.masked_fill_(mask, 0)

        return adj

################################结束#################################



class RnnLayer(nn.Module):
    def __init__(self, model_dim, feed_forward_dim=2048, dropout=0.1):
        """
        使用RNN (LSTM) 替代自注意力机制的模块。

        参数:
            model_dim (int): 输入和输出的特征维度。
            feed_forward_dim (int): 前馈网络的中间层维度。
            dropout (float): Dropout的比率。
        """
        super().__init__()

        # 使用双向LSTM，隐藏层大小为 model_dim / 2，
        # 这样前向和后向拼接后维度依然是 model_dim。
        # 如果 model_dim 是奇数，需要做一些调整。这里假设是偶数。
        if model_dim % 2 != 0:
            raise ValueError("model_dim 必须是偶数才能方便地使用双向RNN。")

        self.rnn = nn.LSTM(
            input_size=model_dim,
            hidden_size=model_dim // 2,
            num_layers=1,  # 可以在这里增加层数来加深模型
            batch_first=True,  # 输入张量格式为 (batch, seq_len, features)
            bidirectional=True  # 使用双向RNN
        )

        # 保留原有的前馈网络、LayerNorm和残差连接结构
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )

        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        """
        前向传播。
        参数:
            x (torch.Tensor): 输入张量，形状通常为 (batch_size, ..., length, model_dim)
            dim (int): 序列长度所在的维度，默认为-2
        """
        # 如果序列长度维度不是倒数第二个，先进行转置
        if dim != -2:
            x = x.transpose(dim, -2)

        # 此时 x 的形状应该是 (batch_size, num_nodes, length, model_dim) -> 这是一个4D张量
        residual = x
        # 获取原始形状以便后续恢复
        batch_size, num_nodes, seq_len, model_dim = x.shape

        # 1. RNN层
        # ===================== 修改核心 =====================
        # 将4D张量重塑为LSTM可以处理的3D张量
        # (B, N, L, D) -> (B * N, L, D)
        x_reshaped = x.reshape(batch_size * num_nodes, seq_len, model_dim)

        # 将重塑后的3D张量送入RNN
        out, _ = self.rnn(x_reshaped)

        # 将RNN的输出从3D恢复为原始的4D形状
        # (B * N, L, D) -> (B, N, L, D)
        out = out.reshape(batch_size, num_nodes, seq_len, model_dim)
        # ====================================================

        out = self.dropout1(out)
        out = self.ln1(residual + out)  # 第一个残差连接和LayerNorm

        # 2. 前馈网络层
        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)  # 第二个残差连接和LayerNorm

        # 如果输入时转置过，再转换回去
        if dim != -2:
            out = out.transpose(dim, -2)

        return out


class NALL(nn.Module):
    """
    改进版 - 节点自适应低秩层 (Node-Adaptive Low-rank Layer).

    更忠实于论文描述，包含一个全局共享的LoRA更新 (∆W)
    和一个节点独有的LoRA更新 (W(i))。
    """

    def __init__(self, in_features, out_features, num_nodes,
                 global_rank, node_rank, alpha=1.0):
        """
        初始化 NALL.
        Args:
            in_features (int): 输入特征维度.
            out_features (int): 输出特征维度.
            num_nodes (int): 节点数量.
            global_rank (int): 全局共享LoRA的秩.
            node_rank (int): 节点独有LoRA的秩 (对应论文的 node embedding dimension).
            alpha (float): 缩放因子.
        """
        super().__init__()
        self.alpha = alpha
        self.global_rank = global_rank
        self.node_rank = node_rank

        # 核心的线性层 (对应 W_0)
        self.linear = nn.Linear(in_features, out_features)

        # 1. 全局共享的低秩矩阵 (对应 ∆W)
        if global_rank > 0:
            self.global_lora_A = nn.Parameter(torch.empty(in_features, global_rank))
            self.global_lora_B = nn.Parameter(torch.empty(global_rank, out_features))
            nn.init.kaiming_uniform_(self.global_lora_A, a=1.414)
            nn.init.zeros_(self.global_lora_B)

        # 2. 节点独有的低秩矩阵 (对应 W(i))
        if node_rank > 0:
            # 为每个节点创建一对 A, B 矩阵
            self.node_lora_A = nn.Parameter(torch.empty(num_nodes, in_features, node_rank))
            self.node_lora_B = nn.Parameter(torch.empty(num_nodes, node_rank, out_features))
            nn.init.kaiming_uniform_(self.node_lora_A, a=1.414)
            nn.init.zeros_(self.node_lora_B)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.linear.weight, a=1.414)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        """
        x 的期望形状: (batch_size, time_steps, num_nodes, in_features)
        """
        # 基础线性变换 (W_0 * x + b)
        base_output = self.linear(x)

        # 计算全局低秩更新 (∆W * x)
        global_update = 0
        if self.global_rank > 0:
            global_update = (x @ self.global_lora_A @ self.global_lora_B) * (self.alpha / self.global_rank)

        # 计算节点独有的低秩更新 (W(i) * x)
        node_update = 0
        if self.node_rank > 0:
            # 我们需要让每个节点的输入 x_i 只和它自己的 W(i) 作用
            # x shape:       (B, T, N, D_in)
            # node_lora_A:   (N, D_in, r_n)
            # node_lora_B:   (N, r_n, D_out)
            # 使用 einsum 实现高效的批处理矩阵乘法
            # 'btni, nir -> btnr' : 对每个batch, 每个time, 每个node，输入特征与对应的node_A矩阵相乘
            # 'btnr, nro -> btno' : 上一步结果与对应的node_B矩阵相乘
            node_lora_part1 = torch.einsum('btni,nir->btnr', x, self.node_lora_A)
            node_update = torch.einsum('btnr,nro->btno', node_lora_part1, self.node_lora_B) * (
                        self.alpha / self.node_rank)

        # 组合所有结果
        output = base_output + global_update + node_update
        return output

class DSTRformer(nn.Module):
    """
    Paper: STAEformer: Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting
    Link: https://arxiv.org/abs/2308.10425
    Official Code: https://github.com/XDZhelheim/STAEformer
    """

    def __init__(
            self,
            num_nodes,
            adj_mx,
            in_steps,
            out_steps,
            steps_per_day,
            input_dim,
            output_dim,
            input_embedding_dim,
            tod_embedding_dim,
            ts_embedding_dim,
            dow_embedding_dim,
            time_embedding_dim,
            adaptive_embedding_dim,
            node_dim,
            feed_forward_dim,
            out_feed_forward_dim,
            num_heads,
            num_layers,
            num_layers_m,
            mlp_num_layers,
            dropout,
            use_mixed_proj,
            # ================== 新增参数 ==================
            mask_ratio=0.15,  # 默认掩码比例 15%
            ssl_lambda=0.1,  # 默认 SSL 损失权重 0.1
            # ============================================

            # ================== PDG-CL 新增参数 ==================
            node_embedding_dim_dg = 64,  # 动态图生成器的节点嵌入维度
            cl_projection_dim = 128,  # 对比学习投影头维度
            cl_temperature = 0.1,  # 对比学习温度参数
            graph_perturb_ratio = 0.1,  # 图扰动比例 (例如，丢弃多少边)
            # ==================================================
            # ================== 新增 NALL 集成参数 ==================
            use_nall = True,  # 是否启用 NALL 模块
            global_rank = 8,  # 低秩分解的秩 (一个较小的值)
            node_rank=12,
            lora_alpha = 1.0  # 低秩更新的缩放因子
    # ====================================================
        ):
        super().__init__()
        self.num_nodes = num_nodes
        self.adj_mx = adj_mx
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.ts_embedding_dim = ts_embedding_dim
        self.time_embedding_dim = time_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.node_dim = node_dim
        self.feed_forward_dim = feed_forward_dim
        self.model_dim = (
                input_embedding_dim
                + tod_embedding_dim
                + dow_embedding_dim
                + adaptive_embedding_dim
                + ts_embedding_dim
                + time_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj
        self.num_layers_m = num_layers_m
        self.dropout = dropout
        # ================== 新增参数赋值 ==================
        self.mask_ratio = mask_ratio
        self.ssl_lambda = ssl_lambda
        # ===============================================

        # ================== PDG-CL 参数赋值 ==================
        self.node_embedding_dim_dg = node_embedding_dim_dg
        self.cl_projection_dim = cl_projection_dim
        self.cl_temperature = cl_temperature
        self.graph_perturb_ratio = graph_perturb_ratio
        # ===================================================
        # ================== 新增：定义可学习的 MASK Token ==================
        # 它的维度必须与我们想要掩码的输入特征维度 (self.input_dim) 相同。
        # 形状设为 (1, 1,  D) 是为了方便后续利用广播机制进行赋值。
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.input_dim))
        # ================================================================
        # ================== NALL 参数赋值 ==================
        self.use_nall = use_nall
        # ===============================================


        if self.input_embedding_dim > 0:
            self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if time_embedding_dim > 0:
            self.time_embedding = nn.Embedding(7 * steps_per_day, self.time_embedding_dim)

        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )
        self.adj_mx_forward_encoder = nn.Sequential(
            GraphMLP(input_dim=self.num_nodes, hidden_dim=self.node_dim)
        )

        self.adj_mx_backward_encoder = nn.Sequential(
            GraphMLP(input_dim=self.num_nodes, hidden_dim=self.node_dim)
        )


        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout,self.num_nodes,global_rank ,node_rank ,lora_alpha)
                for _ in range(num_layers)
            ]
        )
        ################ 时间注意力  ############
        self.attn_layers_t_01 = SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout,self.num_nodes,global_rank ,node_rank ,lora_alpha)
        ################   结束   ############
        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout,self.num_nodes,global_rank ,node_rank ,lora_alpha)
                for _ in range(num_layers)
            ]
        )

        ################ 多视角时空注意力 ############
        self.adj_mx_01 = adj_mx[0].clone().detach().float()
        self.attn_layers_s_01 = MultiViewSpatialAttention(self.model_dim, self.feed_forward_dim,self.num_heads, self.dropout, self.adj_mx_01,k_similar=10, k_ratio=0.2, num_nodes=self.num_nodes, max_lag=3)

        ################   结束   ############


        self.attn_layers_c = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout,self.num_nodes,global_rank ,node_rank ,lora_alpha)
            ]
        )
        self.ar_attn = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, out_feed_forward_dim, num_heads,
                                   dropout,self.num_nodes,global_rank ,node_rank ,lora_alpha)
                for _ in range(num_layers_m)
            ]
        )
        if self.ts_embedding_dim > 0:
            self.time_series_emb_layer = nn.Conv2d(
                in_channels=self.input_dim * self.in_steps, out_channels=self.ts_embedding_dim, kernel_size=(1, 1),
                bias=True)

        self.fusion_model = nn.Sequential(
            *[MultiLayerPerceptron(input_dim=self.adaptive_embedding_dim + 2 * self.node_dim,
                                   hidden_dim=self.adaptive_embedding_dim + 2 * self.node_dim,
                                   dropout=0.2)
              for _ in range(mlp_num_layers)],
            nn.Linear(in_features=self.adaptive_embedding_dim + 2 * self.node_dim, out_features=self.adaptive_embedding_dim, bias=True)
        )
        # ================== 实例化动态图生成器 ==================
        self.dynamic_graph_gen = DynamicGraphGenerator(
            num_nodes=num_nodes,
            input_dim=input_dim,  # 基于原始流量生成
            node_embedding_dim=node_embedding_dim_dg
        )
        # ====================================================

        # ================== 实现图卷积层 (GCN) 使用动态图 ==================
        # 你需要一个 GCN 层来处理动态图。这里只是一个占位符。
        # 你可以使用现有的 GCN 库或自己实现一个简单的层。
        # 它应该接收节点特征和动态邻接矩阵，并输出更新后的节点特征。
        # GCN 应该能够处理 (B, L, N, D) 的特征和 (B, N, N) 的图。
        self.dynamic_gcn = nn.Linear(self.model_dim, self.model_dim)  # 这是一个非常简化的替代方案，理想情况应使用 GCN
        # ================================================================

        # ================== 新增对比学习投影头 ==================
        self.cl_projection_head = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim // 2),
            nn.ReLU(),
            nn.Linear(self.model_dim // 2, self.cl_projection_dim)
        )
        # ====================================================

        # ================== 新增重建头 ==================
        # 这个头接收编码后的表示 (model_dim)，并尝试重建原始输入 (input_dim)
        self.ssl_head = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim // 2),
            nn.ReLU(),
            nn.Linear(self.model_dim // 2, self.input_dim)
        )
        # ============================================


        # 替换方案一：使用RNN模块
        self.ar_rnn = nn.ModuleList([RnnLayer(self.model_dim) for _ in range(num_layers_m)])

        # ================== 插入 NALL 层实例化 ==================
        # 我们将 NALL 作为一个处理层，对主干网络提取的特征进行自适应调整
        # 输入和输出维度保持 model_dim 不变
        if self.use_nall:
            self.nall_layer = NALL(
                in_features=self.model_dim,
                out_features=self.model_dim,
                num_nodes=self.num_nodes,
                global_rank=global_rank,
                node_rank=node_rank,
                alpha=lora_alpha
            )
        # ==========================================================
    def perturb_graph(self, adj, ratio):  # <--- 在这里添加 self
        """
        对邻接矩阵进行边丢弃扰动.
        Args:
            self: 类实例自身 (自动传入).
            adj (torch.Tensor): 邻接矩阵 (B, N, N).
            ratio (float): 丢弃边的比例.
        Returns:
            torch.Tensor: 扰动后的邻接矩阵 (B, N, N).
        """
        adj_perturbed = adj.clone()
        non_zero_indices = torch.nonzero(adj_perturbed)  # 获取所有存在的边

        # 检查是否有边可以丢弃
        if len(non_zero_indices) == 0:
            return adj_perturbed  # 如果没有边，直接返回

        num_edges_to_drop = int(len(non_zero_indices) * ratio)

        if num_edges_to_drop > 0:
            drop_indices_idx = torch.randperm(len(non_zero_indices), device=adj.device)[:num_edges_to_drop]  # 确保在同一设备
            drop_indices = non_zero_indices[drop_indices_idx]

            # 确保索引是正确的格式
            if drop_indices.numel() > 0:
                adj_perturbed[drop_indices[:, 0], drop_indices[:, 1], drop_indices[:, 2]] = 0

        return adj_perturbed

    def calculate_infonce_loss(self, z1, z2, temperature):
        """
        计算 InfoNCE 对比损失.
        Args:
            z1 (torch.Tensor): 视图1的投影表示 (N_total, D_proj).
            z2 (torch.Tensor): 视图2的投影表示 (N_total, D_proj).
            temperature (float): 温度参数.

        Returns:
            torch.Tensor: InfoNCE 损失值.
        """
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # 计算相似度矩阵
        sim_matrix = torch.mm(z1, z2.t()) / temperature  # (N_total, N_total)

        # 标签是对角线 (正样本对)
        labels = torch.arange(sim_matrix.size(0)).long().to(sim_matrix.device)

        # 计算交叉熵损失 (分子是正样本对，分母是所有样本对)
        loss = F.cross_entropy(sim_matrix, labels)

        return loss

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        device = history_data.device  # 获取当前设备
        x_full = history_data  # 保留完整输入，以备提取时间特征
        batch_size, in_steps, num_nodes, _ = x_full.shape
        # ================== 初始化 SSL 返回值 ==================
        ssl_pred = None
        ssl_true = None
        cl_loss_val = None  # 初始化对比损失
        # ====================================================
        # 提取时间特征
        tod = x_full[..., 1] if self.tod_embedding_dim > 0 else None
        dow = x_full[..., 2] if self.dow_embedding_dim > 0 else None

        # 提取原始输入数据（例如流量值）
        x = x_full[..., : self.input_dim].clone()  # 使用 .clone() 避免修改原始输入
        # ================== 1. 生成动态图 ==================
        dynamic_adj = self.dynamic_graph_gen(x)  # (B, N, N)
        # ==================================================

        # ================== 2. 创建图扰动视图 ==================
        if train:  # 只在训练时进行对比学习
            dynamic_adj_perturbed = self.perturb_graph(dynamic_adj, self.graph_perturb_ratio)
        # =====================================================
        # !!! 将时间序列嵌入的计算移到这里，在进行掩码操作之前
        # !!! 这样 time_series_emb 始终在需要时被赋值，与 train 模式无关
        if self.ts_embedding_dim > 0:
            # Note: time_series_emb should be calculated based on the ORIGINAL x_full before masking.
            # If you want it based on the masked x, move this block after masking.
            # For time series embedding, it's usually based on the full input features.
            input_data_for_ts_emb = x.transpose(1, 2).contiguous()  # Use the cloned x, which is initially full
            input_data_for_ts_emb = input_data_for_ts_emb.view(
                batch_size, self.num_nodes, -1).transpose(1, 2).unsqueeze(-1)
            time_series_emb = self.time_series_emb_layer(input_data_for_ts_emb)
            time_series_emb = time_series_emb.transpose(1, -1).expand(batch_size, self.in_steps, self.num_nodes,
                                                                      self.ts_embedding_dim)
        # !!!

        # ================== 输入掩码 (仅在训练时) ==================
        if train and self.mask_ratio > 0:
            # 1. 计算需要掩码的数量
            num_elements = in_steps * num_nodes
            num_mask = int(num_elements * self.mask_ratio)
            # 2. 生成随机索引
            # (B, L, N, D) -> (B, L*N, D)
            x_flat = x.view(batch_size, -1, self.input_dim)
            # 生成每个样本的随机索引
            mask_indices_flat = torch.stack([
                torch.randperm(num_elements, device=device)[:num_mask]
                for _ in range(batch_size)
            ])  # (B, num_mask)
            # 3. 保存原始值 (ssl_true)
            # 使用 gather 收集被掩码位置的原始值
            ssl_true = torch.gather(x_flat, 1,
                                    mask_indices_flat.unsqueeze(-1).expand(-1, -1, self.input_dim))  # (B, num_mask, D)
            # 4. 应用掩码 (使用可学习的 MASK token 替换)

            # 将我们定义的 MASK token 扩展到需要替换的形状 (B, num_mask, D)
            # .expand() 是一个高效的操作，不会实际复制数据
            mask_value = self.mask_token.expand(batch_size, num_mask, self.input_dim)

            # 使用 scatter_ 将可学习的 MASK token 的值写入被掩码的位置
            # mask_indices_flat 需要扩展维度以匹配 x_flat 和 mask_value 的形状
            x_flat.scatter_(1, mask_indices_flat.unsqueeze(-1).expand(-1, -1, self.input_dim), mask_value)
            # ================================================================
            # 5. 恢复形状
            x = x_flat.view(batch_size, in_steps, num_nodes, self.input_dim)
            # 6. 存储掩码位置以便后续提取编码表示
            # 我们需要 batch_indices, time_indices, node_indices
            batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_mask)  # (B, num_mask)
            mask_t = mask_indices_flat // num_nodes  # (B, num_mask)
            mask_n = mask_indices_flat % num_nodes  # (B, num_mask)
            mask_indices = (batch_idx, mask_t, mask_n)
        # ===========================================================

        # B ts_embedding_dim N 1
        # 原始输入数据的投影
        x_emb = self.input_proj(x)
        features = [x_emb]

        # 添加各种嵌入特征
        if self.ts_embedding_dim > 0:
            features.append(time_series_emb)
        # (8,12,170,28)
        if self.tod_embedding_dim > 0: #(8,12,170,24)
            tod_emb = self.tod_embedding((tod * self.steps_per_day).long())  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:#(8,12,170,24)
            dow_emb = self.dow_embedding(dow.long())  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
        if self.time_embedding_dim > 0:#(time_embedding_dim=0，不执行)
            time_emb = self.time_embedding(((tod + dow * 7) * self.steps_per_day).long())
            features.append(time_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(size=(batch_size, *self.adaptive_embedding.shape))
            features.append(adp_emb)#(8,12,170,100)

###################以上是进行所有不同类型的嵌入###################

        temporal_x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)
        #################### 时间序列延迟感知##############
        # temporal_x = self.SeriesAlignedGraphConvolution(temporal_x)+temporal_x
        #####################结束#####################
        spatial_x = temporal_x.clone()  # 空间编码器的输入是时序编码器的克隆
        # --- 编码器处理函数 (用于两个视图) ---
        def encode(input_x, adj):  # 'adj' 现在是动态图
            spatial_x = input_x.clone()
            temp_x = input_x.clone()  # 确保时序输入不受影响

            # --- 使用修改后的 MultiViewSpatialAttention ---
            # 现在它接收动态图作为第二个参数
            spatial_x = self.attn_layers_s_01(spatial_x, dynamic_adj=adj)

            # --- 原有注意力 ---
            temp_x = self.attn_layers_t_01(temp_x)

            # 交叉融合
            x_encoded = self.attn_layers_c[0](temp_x, spatial_x, dim=2)

            return x_encoded
        # ================== 3. 通过编码器得到两个视图的表示 ==================
        x_encoded_view1 = encode(temporal_x, dynamic_adj)
        if train:
            x_encoded_view2 = encode(temporal_x, dynamic_adj_perturbed)

            # ================== 4. 计算对比损失 ==================
            # (B, L, N, D_model) -> (B*L*N, D_model)
            h1 = x_encoded_view1.view(-1, self.model_dim)
            h2 = x_encoded_view2.view(-1, self.model_dim)

            # 通过投影头
            z1 = self.cl_projection_head(h1)
            z2 = self.cl_projection_head(h2)

            # 计算 InfoNCE Loss
            cl_loss_val = self.calculate_infonce_loss(z1, z2, self.cl_temperature)
            # =======================================================
        # --- 主流程继续使用 view1 (或你可以选择混合) ---
        x = x_encoded_view1

        # x = self.SeriesAlignedGraphConvolution(x) + x

###################以上是进行时空嵌入以及时空交叉融合嵌入，动态时空趋势 Transformer (DST2former)###################

        # ================== 在此处插入 NALL 进行节点自适应调整 ==================
        if self.use_nall:
            # x 的形状是 (B, T, N, model_dim)，正好符合 NALL 的输入要求
            x = self.nall_layer(x)
        # ====================================================================




        # ================== 提取被掩码位置的编码并进行重建 ==================
        if train and self.mask_ratio > 0:
            b_idx, t_idx, n_idx = mask_indices
            # 从 x_encoded 中提取出对应掩码位置的向量
            masked_encoded_vectors = x[b_idx, t_idx, n_idx]  # (B, num_mask, model_dim)
            # 通过重建头进行预测
            ssl_pred = self.ssl_head(masked_encoded_vectors)  # (B, num_mask, input_dim)
            # ===================================================================

        # --- 图融合和 AR 注意力 ---
        if self.node_dim > 0:
            # ... (原有图融合代码，可以考虑是否也使用动态图) ...
            adp_graph = x[..., -self.adaptive_embedding_dim:]
            x_main = x[..., :self.model_dim - self.adaptive_embedding_dim]  # 修正变量名

            node_forward = self.adj_mx[0].to(device)
            node_forward = self.adj_mx_forward_encoder(node_forward.unsqueeze(0)).expand(batch_size, self.in_steps,
                                                                                             -1,
                                                                                             -1)
            node_backward = self.adj_mx[1].to(device)
            node_backward = self.adj_mx_backward_encoder(node_backward.unsqueeze(0)).expand(batch_size,
                                                                                                self.in_steps,
                                                                                                -1, -1)

            graph = torch.cat([adp_graph, node_forward, node_backward], dim=-1)
            graph = self.fusion_model(graph)

            x = torch.cat([x_main, graph], dim=-1)  # 修正拼接

###################以上是进行时空嵌入以及时空交叉融合嵌入###################



        # for attn in self.ar_attn:
        #     x = attn(x, dim=2, augment=True)

        for layer in self.ar_rnn:
            x = layer(x)

###################以上是自回归注意力###################

        # --- 输出层 ---
        if self.use_mixed_proj:
            # ... (原有输出层代码) ...
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(batch_size, self.num_nodes, self.in_steps * self.model_dim)
            out = self.output_proj(out).view(batch_size, self.num_nodes, self.out_steps, self.output_dim)
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(out)  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(out.transpose(1, 3))  # (batch_size, out_steps, num_nodes, output_dim)

        # ================== 返回主预测、SSL 数据和对比损失 ==================
        # 你需要修改训练循环来处理这个新的返回值
        return out, (ssl_pred, ssl_true), cl_loss_val
        # ================================================================

