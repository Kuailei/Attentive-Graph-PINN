import os
import copy
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.mixture import GaussianMixture

from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


# =========================================================
# 0. Reproducibility
# =========================================================
def seed_everything(seed=3407):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed_everything(3407)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# 1. User settings
# =========================================================
CSV_PATH = "/mnt/data/df1.csv"    
OUTPUT_DIR = "./ai_outputs_7_3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COL = "Hippocampus_volume"
ID_COL = "PID"
BMI_GROUP_COL = "BMI_cat"

NODE_COLS = [
    "Hippocampus_SUL", "adrenal_gland_SUL", "pancreas_SUL",
    "liver_SUL", "spleen_SUL", "vertebrae_SUL", "aorta_SUL",
    "kidney_SUL", "stomach_SUL", "VAT_SUL", "SAT_SUL", "lung_SUL",
    "skeletal_muscle_SUL", "heart_SUL", "colon_SUL", "small_bowel_SUL",
    "spinal_cord_SUL"
]

COV_COLS = ["sex", "BMI", "brain_volume"]
AGE_COL = "age"

NORMAL_LABEL = "Normal_weight"
EDGE_THRESHOLD = 0.12

TEST_SIZE = 0.30          # 7:3
EPOCHS = 300
PATIENCE = 40
LR = 0.005
WEIGHT_DECAY = 0.001
EARLYSTOP_MONITOR_RATIO = 0.15   # 从训练集内部再切一小块做早停监控


# =========================================================
# 2. Utilities
# =========================================================
def calculate_pcor_safe(data_df: pd.DataFrame):
    """
    Partial correlation estimated from inverse correlation matrix.
    """
    try:
        corr_matrix = data_df.corr().values.astype(float)
        precision_matrix = np.linalg.inv(corr_matrix)
        diag_sqrt = np.sqrt(np.diag(precision_matrix))
        pcor_matrix = -precision_matrix / np.outer(diag_sqrt, diag_sqrt)
        np.fill_diagonal(pcor_matrix, 1.0)
        return pcor_matrix
    except Exception:
        return None


def build_reference_network(train_df: pd.DataFrame, node_cols, bmi_group_col, normal_label):
    ref_df = train_df.loc[train_df[bmi_group_col] == normal_label, node_cols].astype(float)
    if ref_df.shape[0] < 10:
        raise ValueError("Too few normal-weight participants in the training set to build reference network.")
    ref_net = calculate_pcor_safe(ref_df)
    if ref_net is None:
        raise ValueError("Reference network construction failed.")
    return ref_df, ref_net


def build_features_from_reference(
    df_subset,
    ref_df,
    ref_net,
    node_cols,
    cov_cols,
    age_col,
    target_col,
    id_col,
    edge_threshold=0.12
):
    """
    用训练集参考网络，为每个样本构建 subject-level features
    """
    n_ref = len(ref_df)
    data_list = []

    for _, row in df_subset.iterrows():
        subj_data = row[node_cols].values.reshape(1, -1).astype(float)
        df_ptb = pd.concat([ref_df, pd.DataFrame(subj_data, columns=node_cols)], ignore_index=True)
        ptb_net = calculate_pcor_safe(df_ptb.astype(float))
        if ptb_net is None:
            continue

        res_net = ptb_net - ref_net
        denom = (1 - ref_net ** 2) / (n_ref - 1)

        with np.errstate(divide="ignore", invalid="ignore"):
            zcc_matrix = np.divide(res_net, denom)

        zcc_matrix = np.nan_to_num(zcc_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(zcc_matrix, 0.0)

        abs_zcc = np.abs(zcc_matrix)
        abs_zcc[abs_zcc < edge_threshold] = 0.0
        str_nodes = np.sum(abs_zcc, axis=1)

        data_list.append({
            "pid": row[id_col],
            "x_str": str_nodes.astype(np.float32),
            "x_raw": row[node_cols].values.astype(np.float32),
            "adj": abs_zcc.astype(np.float32),
            "x_cov": row[cov_cols].values.astype(np.float32),
            "age": float(row[age_col]),
            "y": float(row[target_col])
        })

    return data_list


def fit_feature_scalers(train_data):
    X_str = np.array([d["x_str"] for d in train_data]).reshape(len(train_data), -1)
    X_raw = np.array([d["x_raw"] for d in train_data])
    X_cov = np.array([d["x_cov"] for d in train_data])
    Age = np.array([d["age"] for d in train_data]).reshape(-1, 1)
    Y = np.array([d["y"] for d in train_data]).reshape(-1, 1)

    scaler_str = StandardScaler().fit(X_str)
    scaler_raw = StandardScaler().fit(X_raw)
    scaler_cov = StandardScaler().fit(X_cov)
    scaler_y = MinMaxScaler().fit(Y)

    age_min = float(Age.min())
    age_max = float(Age.max())
    age_range = max(age_max - age_min, 1e-8)

    return {
        "scaler_str": scaler_str,
        "scaler_raw": scaler_raw,
        "scaler_cov": scaler_cov,
        "scaler_y": scaler_y,
        "age_min": age_min,
        "age_max": age_max,
        "age_range": age_range
    }


def transform_data(data_list, scalers):
    X_str = np.array([d["x_str"] for d in data_list]).reshape(len(data_list), -1)
    X_raw = np.array([d["x_raw"] for d in data_list])
    Adj = np.array([d["adj"] for d in data_list])
    X_cov = np.array([d["x_cov"] for d in data_list])
    Age = np.array([d["age"] for d in data_list]).reshape(-1, 1)
    Y = np.array([d["y"] for d in data_list]).reshape(-1, 1)
    PIDs = np.array([d["pid"] for d in data_list])

    X_str = scalers["scaler_str"].transform(X_str).reshape(len(data_list), len(NODE_COLS), 1)
    X_raw = scalers["scaler_raw"].transform(X_raw)
    X_cov = scalers["scaler_cov"].transform(X_cov)
    Age = (Age - scalers["age_min"]) / scalers["age_range"]
    Y_norm = scalers["scaler_y"].transform(Y)

    return {
        "pid": PIDs,
        "x_str": torch.tensor(X_str, dtype=torch.float32, device=DEVICE),
        "x_raw": torch.tensor(X_raw, dtype=torch.float32, device=DEVICE),
        "adj": torch.tensor(Adj, dtype=torch.float32, device=DEVICE),
        "x_cov": torch.tensor(X_cov, dtype=torch.float32, device=DEVICE),
        "age": torch.tensor(Age, dtype=torch.float32, device=DEVICE),
        "y": torch.tensor(Y_norm, dtype=torch.float32, device=DEVICE),
        "y_real": Y.flatten()
    }


def split_tensor_dict(tensor_dict, train_ratio=0.85, random_state=3407):
    """
    从训练集内部切出一部分作为 early-stopping monitor
    """
    n = len(tensor_dict["pid"])
    idx = np.arange(n)

    fit_idx, monitor_idx = train_test_split(
        idx,
        test_size=(1 - train_ratio),
        random_state=random_state
    )

    def subset_dict(d, indices):
        out = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                out[k] = v[indices]
            elif torch.is_tensor(v):
                out[k] = v[indices]
            else:
                out[k] = v
        return out

    return subset_dict(tensor_dict, fit_idx), subset_dict(tensor_dict, monitor_idx)


def safe_inverse_minmax(scaler, tensor_or_numpy):
    if torch.is_tensor(tensor_or_numpy):
        x = tensor_or_numpy.detach().cpu().numpy().astype(np.float64)
    else:
        x = np.asarray(tensor_or_numpy, dtype=np.float64)

    x = x.reshape(-1, 1)
    scale = scaler.scale_[0]
    min_val = scaler.min_[0]
    x_orig = (x - min_val) / scale
    return x_orig.flatten()


def get_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    n = len(y_true)
    if n <= 1:
        c_index = np.nan
    else:
        v, c = 0, 0
        for i in range(n):
            for j in range(i + 1, n):
                if y_true[i] != y_true[j]:
                    v += 1
                    if (y_pred[i] - y_pred[j]) * (y_true[i] - y_true[j]) > 0:
                        c += 1
                    elif y_pred[i] == y_pred[j]:
                        c += 0.5
        c_index = c / v if v > 0 else np.nan

    return {"MAE": mae, "R2": r2, "C_index": c_index}


def flatten_baseline_features(tensor_dict):
    x_str = tensor_dict["x_str"].detach().cpu().numpy().reshape(len(tensor_dict["pid"]), -1)
    x_raw = tensor_dict["x_raw"].detach().cpu().numpy()
    x_cov = tensor_dict["x_cov"].detach().cpu().numpy()
    age = tensor_dict["age"].detach().cpu().numpy()
    X = np.concatenate([x_str, x_raw, x_cov, age], axis=1)
    y = tensor_dict["y_real"]
    return X, y


# =========================================================
# 3. Model definitions
# =========================================================
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        bsz, n_nodes, _ = adj.shape
        I = torch.eye(n_nodes, device=adj.device).unsqueeze(0).expand(bsz, -1, -1)
        A_hat = adj + I
        D_inv = torch.pow(torch.sum(A_hat, dim=2), -0.5)
        D_inv[torch.isinf(D_inv)] = 0.0
        D_mat = torch.diag_embed(D_inv)
        support = torch.bmm(torch.bmm(D_mat, A_hat), D_mat)
        return self.linear(torch.bmm(support, x))


class GNNOnly(nn.Module):
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
        graph_emb = torch.mean(h, dim=1)
        combined = torch.cat([graph_emb, x_raw, x_cov, age], dim=1)
        pred = torch.sigmoid(self.regressor(combined))
        return pred


class GNNPINN(nn.Module):
    def __init__(self, num_nodes=17, node_in=1, raw_in=17, cov_dim=3, hidden_dim=32):
        super().__init__()
        self.beta_net = nn.Sequential(
            nn.Linear(raw_in + cov_dim, 32),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        self.gc1 = GraphConvolution(node_in, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.bn = nn.BatchNorm1d(num_nodes)
        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_dim + raw_in, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x_str, x_raw, adj, x_cov, age):
        beta_in = torch.cat([x_raw, x_cov], dim=1)
        beta = torch.sigmoid(self.beta_net(beta_in))

        h = F.elu(self.gc1(x_str, adj))
        h = self.dropout(h)
        h = F.elu(self.gc2(h, adj))
        h = self.bn(h)
        graph_emb = torch.mean(h, dim=1)

        alpha_in = torch.cat([graph_emb, x_raw], dim=1)
        decay_rate = F.softplus(self.alpha_head(alpha_in))

        pred = beta * (1.0 - decay_rate * age)
        return pred


class AttentiveGraphPINN(nn.Module):
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

        pred = beta * torch.exp(-decay_rate * age)
        return pred, beta, decay_rate, alpha_node.squeeze(-1), weights.squeeze(-1)


# =========================================================
# 4. Training / evaluation
# =========================================================
def run_torch_model(
    model_name,
    model_class,
    train_t,
    test_t,
    scaler_y,
    epochs=300,
    patience=40,
    earlystop_monitor_ratio=0.15
):
    seed_everything(3407)

    # 从训练集内部切出 monitor set 做 early stopping
    fit_t, monitor_t = split_tensor_dict(
        train_t,
        train_ratio=(1 - earlystop_monitor_ratio),
        random_state=3407
    )

    model = model_class().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=15, factor=0.5)
    criterion = nn.MSELoss()

    best_monitor_mae = np.inf
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = model(fit_t["x_str"], fit_t["x_raw"], fit_t["adj"], fit_t["x_cov"], fit_t["age"])
        pred_fit = out[0] if isinstance(out, tuple) else out
        loss = criterion(pred_fit.view(-1, 1), fit_t["y"].view(-1, 1))
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_monitor = model(
                monitor_t["x_str"], monitor_t["x_raw"], monitor_t["adj"],
                monitor_t["x_cov"], monitor_t["age"]
            )
            pred_monitor = out_monitor[0] if isinstance(out_monitor, tuple) else out_monitor
            pred_monitor_real = safe_inverse_minmax(scaler_y, pred_monitor)
            y_monitor_real = monitor_t["y_real"]
            monitor_mae = mean_absolute_error(y_monitor_real, pred_monitor_real)

        scheduler.step(loss.item())
        history.append((epoch + 1, float(loss.item()), float(monitor_mae)))

        if monitor_mae < best_monitor_mae:
            best_monitor_mae = monitor_mae
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()

    def infer(split_t):
        with torch.no_grad():
            out = model(split_t["x_str"], split_t["x_raw"], split_t["adj"], split_t["x_cov"], split_t["age"])
            if isinstance(out, tuple):
                pred_norm = out[0]
                extras = out[1:]
            else:
                pred_norm = out
                extras = None

            pred_real = safe_inverse_minmax(scaler_y, pred_norm)
            y_real = split_t["y_real"]
            metrics = get_metrics(y_real, pred_real)
            return pred_real, y_real, metrics, extras

    pred_train, y_train, met_train, _ = infer(train_t)
    pred_test, y_test, met_test, extras_test = infer(test_t)

    result = {
        "model_name": model_name,
        "model": model,
        "history": history,
        "train_metrics": met_train,
        "test_metrics": met_test,
        "pred_train": pred_train,
        "pred_test": pred_test,
        "y_train": y_train,
        "y_test": y_test
    }

    if model_name == "Attentive Graph PINN":
        rows = []
        organ_names = [n.replace("_SUL", "") for n in NODE_COLS]

        for split_name, split_t in [("Train", train_t), ("Test", test_t)]:
            with torch.no_grad():
                out = model(split_t["x_str"], split_t["x_raw"], split_t["adj"], split_t["x_cov"], split_t["age"])
                pred_norm, beta_all, alpha_global, alpha_node, attn_w = out
                pred_real = safe_inverse_minmax(scaler_y, pred_norm)

                beta_all = beta_all.detach().cpu().numpy().flatten()
                alpha_global = alpha_global.detach().cpu().numpy().flatten()
                alpha_node = alpha_node.detach().cpu().numpy()
                attn_w = attn_w.detach().cpu().numpy()

                for i, pid in enumerate(split_t["pid"]):
                    rec = {
                        "PID": pid,
                        "Set": split_name,
                        "Actual_mL": split_t["y_real"][i],
                        "Pred_mL": pred_real[i],
                        "Beta": beta_all[i],
                        "Global_Alpha": alpha_global[i]
                    }
                    for j, organ in enumerate(organ_names):
                        rec[f"Alpha_{organ}"] = alpha_node[i, j]
                        rec[f"Weight_{organ}"] = attn_w[i, j]
                    rows.append(rec)

        export_df = pd.DataFrame(rows)
        export_path = os.path.join(OUTPUT_DIR, "Attentive_Graph_PINN_predictions_and_latents.csv")
        export_df.to_csv(export_path, index=False)
        result["attentive_export_path"] = export_path

    return result


def run_ml_baselines(train_t, test_t):
    X_train, y_train = flatten_baseline_features(train_t)
    X_test, y_test = flatten_baseline_features(test_t)

    baselines = {
        "Linear": LinearRegression(),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=3407),
        "SVR": SVR(C=1.0, epsilon=0.1, kernel="rbf"),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=None, random_state=3407, n_jobs=-1
        )
    }

    if HAS_XGB:
        baselines["XGBoost"] = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=3407,
            n_jobs=-1
        )

    if HAS_LGBM:
        baselines["LightGBM"] = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=3407
        )

    results = []
    for name, model in baselines.items():
        model.fit(X_train, y_train)

        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)

        results.append({
            "model_name": name,
            "train_metrics": get_metrics(y_train, pred_train),
            "test_metrics": get_metrics(y_test, pred_test)
        })

    return results


# =========================================================
# 5. Main pipeline
# =========================================================
def main():
    df = pd.read_csv(CSV_PATH)

    required_cols = NODE_COLS + [TARGET_COL, AGE_COL, ID_COL, BMI_GROUP_COL] + COV_COLS
    df = df.dropna(subset=required_cols).copy()

    # -----------------------------------------------------
    # 5.1 First split: 7:3
    # -----------------------------------------------------
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=3407,
        stratify=df[BMI_GROUP_COL] if BMI_GROUP_COL in df.columns else None
    )

    print(f"Train n = {len(train_df)}, Test n = {len(test_df)}")

    # -----------------------------------------------------
    # 5.2 Build reference network using TRAIN normal-weight only
    # -----------------------------------------------------
    ref_df_train, ref_net_train = build_reference_network(
        train_df, NODE_COLS, BMI_GROUP_COL, NORMAL_LABEL
    )

    # -----------------------------------------------------
    # 5.3 Build features using training-derived reference
    # -----------------------------------------------------
    train_data = build_features_from_reference(
        train_df, ref_df_train, ref_net_train,
        NODE_COLS, COV_COLS, AGE_COL, TARGET_COL, ID_COL, EDGE_THRESHOLD
    )
    test_data = build_features_from_reference(
        test_df, ref_df_train, ref_net_train,
        NODE_COLS, COV_COLS, AGE_COL, TARGET_COL, ID_COL, EDGE_THRESHOLD
    )

    print(f"Feature-built Train n = {len(train_data)}, Test n = {len(test_data)}")

    # -----------------------------------------------------
    # 5.4 Fit scalers on TRAIN only
    # -----------------------------------------------------
    scalers = fit_feature_scalers(train_data)

    # -----------------------------------------------------
    # 5.5 Transform train/test
    # -----------------------------------------------------
    train_t = transform_data(train_data, scalers)
    test_t = transform_data(test_data, scalers)

    # -----------------------------------------------------
    # 5.6 Traditional ML baselines
    # -----------------------------------------------------
    baseline_results = run_ml_baselines(train_t, test_t)

    # -----------------------------------------------------
    # 5.7 Graph models
    # -----------------------------------------------------
    graph_results = []
    for model_name, model_class in [
        ("GNN Only", GNNOnly),
        ("GNN + PINN", GNNPINN),
        ("Attentive Graph PINN", AttentiveGraphPINN)
    ]:
        res = run_torch_model(
            model_name=model_name,
            model_class=model_class,
            train_t=train_t,
            test_t=test_t,
            scaler_y=scalers["scaler_y"],
            epochs=EPOCHS,
            patience=PATIENCE,
            earlystop_monitor_ratio=EARLYSTOP_MONITOR_RATIO
        )
        graph_results.append(res)

    # -----------------------------------------------------
    # 5.8 Summarize performance
    # -----------------------------------------------------
    records = []
    for item in baseline_results:
        rec = {"Model": item["model_name"]}
        for split_key, prefix in [("train_metrics", "Train"), ("test_metrics", "Test")]:
            for m in ["MAE", "R2", "C_index"]:
                rec[f"{prefix}_{m}"] = item[split_key][m]
        records.append(rec)

    for item in graph_results:
        rec = {"Model": item["model_name"]}
        for split_key, prefix in [("train_metrics", "Train"), ("test_metrics", "Test")]:
            for m in ["MAE", "R2", "C_index"]:
                rec[f"{prefix}_{m}"] = item[split_key][m]
        records.append(rec)

    perf_df = pd.DataFrame(records).sort_values("Model")
    perf_path = os.path.join(OUTPUT_DIR, "model_performance_summary_7_3.csv")
    perf_df.to_csv(perf_path, index=False)
    print("\nSaved:", perf_path)
    print(perf_df)

if __name__ == "__main__":
    main()
