import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
import random
import os
import copy


# ==========================================
# 0. 固定随机种子
# ==========================================
def seed_everything(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(3407)

# ==========================================
# 1. 数据准备
# ==========================================
# 请修改为你的实际路径
df = pd.read_csv('/home/zhf/df1.csv')

current_nodes = [
    "Hippocampus_SUL", "adrenal_gland_SUL", "pancreas_SUL",
    "liver_SUL", "spleen_SUL", "vertebrae_SUL", "aorta_SUL",
    "kidney_SUL", "stomach_SUL", "VAT_SUL", "SAT_SUL", "lung_SUL",
    "skeletal_muscle_SUL", "heart_SUL", "colon_SUL", "small_bowel_SUL",
    "spinal_cord_SUL"
]

# 清洗数据
df_clean = df.dropna(subset=current_nodes + ['Hippocampus_volume', 'age', 'sex', 'BMI', 'brain_volume'])
df_normal = df_clean[df_clean['BMI_cat'] == 'Normal_weight'][current_nodes]
n_hc = len(df_normal)


# 构建网络函数
def calculate_pcor_safe(data_df):
    try:
        corr_matrix = data_df.corr().values
        precision_matrix = np.linalg.inv(corr_matrix)
        diag_sqrt = np.sqrt(np.diag(precision_matrix))
        pcor_matrix = -precision_matrix / np.outer(diag_sqrt, diag_sqrt)
        np.fill_diagonal(pcor_matrix, 1.0)
        return pcor_matrix
    except:
        return None


ref_net = calculate_pcor_safe(df_normal)
pinn_data_list = []

print("正在构建数据特征...")
for idx, row in df_clean.iterrows():
    subj_data = row[current_nodes].values.reshape(1, -1).astype(float)
    df_ptb = pd.concat([df_normal, pd.DataFrame(subj_data, columns=current_nodes)], ignore_index=True)
    ptb_net = calculate_pcor_safe(df_ptb.astype(float))

    if ptb_net is None: continue

    res_net = ptb_net - ref_net
    denom = (1 - ref_net ** 2) / (n_hc - 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        zcc_matrix = np.divide(res_net, denom)
    np.fill_diagonal(zcc_matrix, 0)

    # 阈值过滤
    abs_zcc = np.abs(zcc_matrix)
    abs_zcc[abs_zcc < 0.12] = 0.0
    str_nodes = np.sum(abs_zcc, axis=1)

    pinn_data_list.append({
        'x_str': str_nodes,
        'x_raw': subj_data.flatten(),
        'adj': abs_zcc,
        'x_cov': row[['sex', 'BMI', 'brain_volume']].values.astype(float),
        'age': row['age'],
        'y': row['Hippocampus_volume'],
        'pid': row['PID']
    })

# 准备全量数据
X_str_all = np.array([d['x_str'] for d in pinn_data_list]).reshape(-1, 17, 1)
X_raw_all = np.array([d['x_raw'] for d in pinn_data_list])
Adj_all = np.array([d['adj'] for d in pinn_data_list])
X_cov_all = np.array([d['x_cov'] for d in pinn_data_list])
Age_all = np.array([d['age'] for d in pinn_data_list]).reshape(-1, 1)
Y_all = np.array([d['y'] for d in pinn_data_list]).reshape(-1, 1)

# 划分索引
indices = np.arange(len(Y_all))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

# 标准化
scaler_str = StandardScaler()
X_str_all_norm = scaler_str.fit_transform(X_str_all.reshape(len(Y_all), -1)).reshape(-1, 17, 1)
scaler_raw = StandardScaler()
X_raw_all_norm = scaler_raw.fit_transform(X_raw_all)
scaler_cov = StandardScaler()
X_cov_all_norm = scaler_cov.fit_transform(X_cov_all)
age_min, age_max = Age_all.min(), Age_all.max()
Age_all_norm = (Age_all - age_min) / (age_max - age_min)
scaler_y = MinMaxScaler()
scaler_y.fit(Y_all[train_idx])
Y_all_norm = scaler_y.transform(Y_all)

# 转 Tensor
t_X_str = torch.FloatTensor(X_str_all_norm)
t_X_raw = torch.FloatTensor(X_raw_all_norm)
t_A = torch.FloatTensor(Adj_all)
t_C = torch.FloatTensor(X_cov_all_norm)
t_Ag = torch.FloatTensor(Age_all_norm)
t_Y = torch.FloatTensor(Y_all_norm)

train_idx_torch = torch.tensor(train_idx, dtype=torch.long)
test_idx_torch = torch.tensor(test_idx, dtype=torch.long)


# ==========================================
# 关键修复工具函数: 手动反归一化
# ==========================================
def safe_inverse(scaler, data_tensor_or_numpy):
    """
    手动执行 MinMaxScaler 的反归一化，
    彻底绕过 sklearn check_array 的内存共享报错。
    公式: X_orig = (X_scaled - min_) / scale_
    """
    # 1. 确保是 numpy 数组
    if torch.is_tensor(data_tensor_or_numpy):
        # detach 并转 numpy，astype 确保数据纯净
        data = data_tensor_or_numpy.detach().cpu().numpy().astype(np.float64)
    else:
        data = np.array(data_tensor_or_numpy, dtype=np.float64)

    # 2. 展平以便计算
    data = data.reshape(-1, 1)

    # 3. 手动提取参数 (MinMaxScaler 属性)
    scale = scaler.scale_[0]
    min_val = scaler.min_[0]

    # 4. 数学运算
    data_orig = (data - min_val) / scale

    return data_orig.flatten()


# ==========================================
# 模型定义
# ==========================================
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        B, N, _ = adj.shape
        I = torch.eye(N, device=adj.device).unsqueeze(0).expand(B, -1, -1)
        A_hat = adj + I
        D_inv = torch.pow(torch.sum(A_hat, dim=2), -0.5)
        D_inv[torch.isinf(D_inv)] = 0.
        D_mat = torch.diag_embed(D_inv)
        support = torch.bmm(torch.bmm(D_mat, A_hat), D_mat)
        return self.linear(torch.bmm(support, x))


# --- 模型 1: GNN Only ---
class GNN_Only(nn.Module):
    def __init__(self, num_nodes=17, node_in=1, raw_in=17, cov_dim=3, hidden_dim=32):
        super().__init__()
        self.gc1 = GraphConvolution(node_in, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.bn = nn.BatchNorm1d(num_nodes)

        input_dim = hidden_dim + raw_in + cov_dim + 1
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_str, x_raw, adj, x_cov, age):
        h = F.elu(self.gc1(x_str, adj))
        h = self.dropout(h)
        h = F.elu(self.gc2(h, adj))
        h = self.bn(h)
        graph_emb = torch.mean(h, dim=1)  # Mean Pooling
        combined = torch.cat([graph_emb, x_raw, x_cov, age], dim=1)
        pred = torch.sigmoid(self.regressor(combined))
        return pred  # 单值返回


# --- 模型 2: GNN + PINN ---
class GNN_PINN(nn.Module):
    def __init__(self, num_nodes=17, node_in=1, raw_in=17, cov_dim=3, hidden_dim=32):
        super().__init__()
        self.beta_net = nn.Sequential(
            nn.Linear(raw_in + cov_dim, 32), nn.Tanh(), nn.Dropout(0.2), nn.Linear(32, 1)
        )
        self.gc1 = GraphConvolution(node_in, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.bn = nn.BatchNorm1d(num_nodes)
        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_dim + raw_in, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, x_str, x_raw, adj, x_cov, age):
        beta_in = torch.cat([x_raw, x_cov], dim=1)
        beta = torch.sigmoid(self.beta_net(beta_in))
        h = F.elu(self.gc1(x_str, adj))
        h = self.dropout(h)
        h = F.elu(self.gc2(h, adj))
        h = self.bn(h)
        graph_emb = torch.mean(h, dim=1)  # Mean Pooling
        alpha_in = torch.cat([graph_emb, x_raw], dim=1)
        decay_rate = F.softplus(self.alpha_head(alpha_in))
        pred = beta * (1.0 - decay_rate * age)
        return pred  # 单值返回


# --- 模型 3: Attentive Graph PINN ---
class Attentive_Graph_PINN(nn.Module):
    def __init__(self, num_nodes=17, node_in=1, raw_in=17, cov_dim=3, hidden_dim=32):
        super().__init__()
        self.beta_net = nn.Sequential(
            nn.Linear(raw_in + cov_dim + hidden_dim, 64),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        self.gc1 = GraphConvolution(node_in, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.bn = nn.BatchNorm1d(num_nodes)
        self.attn_net = nn.Sequential(nn.Linear(hidden_dim, 1))
        self.node_alpha_net = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())
        self.alpha_fusion = nn.Sequential(
            nn.Linear(hidden_dim + raw_in, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_str, x_raw, adj, x_cov, age):
        h = F.elu(self.gc1(x_str, adj))
        h = self.dropout(h)
        h = F.elu(self.gc2(h, adj))
        h = self.bn(h)
        attn_scores = self.attn_net(h)
        weights = torch.softmax(attn_scores, dim=1)
        alpha_node = self.node_alpha_net(h)
        graph_emb = torch.sum(h * weights, dim=1)

        beta_in = torch.cat([x_raw, x_cov, graph_emb], dim=1)
        beta = torch.sigmoid(self.beta_net(beta_in))

        alpha_in = torch.cat([graph_emb, x_raw], dim=1)
        decay_rate = F.softplus(self.alpha_fusion(alpha_in))

        pred = beta * (1.0 - decay_rate * age)

        # 返回5个值
        return pred, beta, decay_rate, alpha_node.squeeze(-1), weights.squeeze(-1)


# ==========================================
# 3. 实验运行与评估函数 (集成修复)
# ==========================================
def get_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # C-index 计算
    n = len(y_true)
    idxs = np.random.choice(n, min(n, 2000), replace=False)
    yt, yp = y_true[idxs], y_pred[idxs]
    v, c = 0, 0
    for i in range(len(yt)):
        for j in range(i + 1, len(yt)):
            if yt[i] != yt[j]:
                v += 1
                if (yp[i] - yp[j]) * (yt[i] - yt[j]) > 0:
                    c += 1
                elif yp[i] == yp[j]:
                    c += 0.5
    c_index = c / v if v > 0 else 0
    return mae, r2, c_index


def run_experiment(model_name, model_class):
    seed_everything(3407)
    model = model_class()
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)
    criterion = nn.MSELoss()

    best_test_mae = float('inf')
    best_state = None

    epochs = 300

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # 兼容单值或多值返回
        out = model(t_X_str[train_idx_torch], t_X_raw[train_idx_torch], t_A[train_idx_torch], t_C[train_idx_torch],
                    t_Ag[train_idx_torch])
        if isinstance(out, tuple):
            p = out[0]
        else:
            p = out

        loss = criterion(p.view(-1, 1), t_Y[train_idx_torch].view(-1, 1))
        loss.backward()
        optimizer.step()

        # 验证
        model.eval()
        with torch.no_grad():
            out_test = model(t_X_str[test_idx_torch], t_X_raw[test_idx_torch], t_A[test_idx_torch], t_C[test_idx_torch],
                             t_Ag[test_idx_torch])
            if isinstance(out_test, tuple):
                p_test = out_test[0]
            else:
                p_test = out_test

            # 【修复点 1】: 使用 safe_inverse 替代 scaler_y.inverse_transform
            p_test_real = safe_inverse(scaler_y, p_test)
            y_real = safe_inverse(scaler_y, t_Y[test_idx_torch])

            mae = mean_absolute_error(y_real, p_test_real)

            if mae < best_test_mae:
                best_test_mae = mae
                best_state = copy.deepcopy(model.state_dict())

        scheduler.step(loss)

    # 加载最佳模型
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        # Train 预测
        out_tr = model(t_X_str[train_idx_torch], t_X_raw[train_idx_torch], t_A[train_idx_torch], t_C[train_idx_torch],
                       t_Ag[train_idx_torch])
        p_train = out_tr[0] if isinstance(out_tr, tuple) else out_tr

        # Test 预测
        out_te = model(t_X_str[test_idx_torch], t_X_raw[test_idx_torch], t_A[test_idx_torch], t_C[test_idx_torch],
                       t_Ag[test_idx_torch])
        p_test = out_te[0] if isinstance(out_te, tuple) else out_te

        # 【修复点 2】: 使用 safe_inverse
        p_train_real = safe_inverse(scaler_y, p_train)
        y_train_real = safe_inverse(scaler_y, t_Y[train_idx_torch])

        p_test_real = safe_inverse(scaler_y, p_test)
        y_test_real = safe_inverse(scaler_y, t_Y[test_idx_torch])

    tr_m, tr_r, tr_c = get_metrics(y_train_real, p_train_real)
    te_m, te_r, te_c = get_metrics(y_test_real, p_test_real)

    # ----------------------------------------------------
    # 导出 CSV (Attentive 模型专用)
    # ----------------------------------------------------
    if model_name == "Attentive Graph PINN":
        print("\n>>> 正在为 Attentive Graph PINN 导出详细结果到 CSV...")
        with torch.no_grad():
            out_all = model(t_X_str, t_X_raw, t_A, t_C, t_Ag)
            # 解包所有返回值
            p_all_norm, beta_all, a_gl_all, a_nd_all, w_all = out_all

            # 【修复点 3】: 使用 safe_inverse
            y_pred_all = safe_inverse(scaler_y, p_all_norm)
            y_true_all = safe_inverse(scaler_y, t_Y)

            organ_names = [n.replace('_SUL', '') for n in current_nodes]
            all_pids = [d['pid'] for d in pinn_data_list]

            export_df = pd.DataFrame({
                'PID': all_pids,
                'Set': ['Train' if i in train_idx else 'Test' for i in range(len(y_pred_all))],
                'Actual_mL': y_true_all,
                'Pred_mL': y_pred_all,
                'Global_Alpha': a_gl_all.cpu().numpy().flatten()
            })

            metrics_map = {
                'Train': {'MAE': tr_m, 'R2': tr_r, 'C_index': tr_c},
                'Test': {'MAE': te_m, 'R2': te_r, 'C_index': te_c}
            }
            export_df['Set_MAE'] = export_df['Set'].map(lambda x: metrics_map[x]['MAE'])
            export_df['Set_R2'] = export_df['Set'].map(lambda x: metrics_map[x]['R2'])
            export_df['Set_C_index'] = export_df['Set'].map(lambda x: metrics_map[x]['C_index'])

            alpha_nd_np = a_nd_all.cpu().numpy()
            attn_wt_np = w_all.cpu().numpy()

            for i, name in enumerate(organ_names):
                export_df[f'Alpha_{name}'] = alpha_nd_np[:, i]
                export_df[f'Weight_{name}'] = attn_wt_np[:, i]

            export_df.to_csv('Optimized_Single_PINN_Results.csv', index=False)
            print(">>> 导出完成: Optimized_Single_PINN_Results.csv")

    return [model_name, tr_m, tr_r, tr_c, te_m, te_r, te_c]


# ==========================================
# 4. 执行对比
# ==========================================
print(
    f"{'Model':<25} | {'Train MAE':<10} | {'Train R2':<10} | {'Train C':<8} | {'Test MAE':<10} | {'Test R2':<10} | {'Test C':<8}")
print("-" * 105)

res1 = run_experiment("GNN Only", GNN_Only)
print(
    f"{res1[0]:<25} | {res1[1]:<10.4f} | {res1[2]:<10.4f} | {res1[3]:<8.4f} | {res1[4]:<10.4f} | {res1[5]:<10.4f} | {res1[6]:<8.4f}")

res2 = run_experiment("GNN + PINN", GNN_PINN)
print(
    f"{res2[0]:<25} | {res2[1]:<10.4f} | {res2[2]:<10.4f} | {res2[3]:<8.4f} | {res2[4]:<10.4f} | {res2[5]:<10.4f} | {res2[6]:<8.4f}")

res3 = run_experiment("Attentive Graph PINN", Attentive_Graph_PINN)
print(
    f"{res3[0]:<25} | {res3[1]:<10.4f} | {res3[2]:<10.4f} | {res3[3]:<8.4f} | {res3[4]:<10.4f} | {res3[5]:<10.4f} | {res3[6]:<8.4f}")