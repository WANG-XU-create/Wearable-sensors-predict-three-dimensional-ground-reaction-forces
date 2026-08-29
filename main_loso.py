import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import OneCycleLR


# =============================================================================
# 以下类与 main.py 完全一致：GRFDataset, LTCCell, SimpleSelfAttention, LTCGRFModel
# =============================================================================

class GRFDataset(Dataset):
    """Load and preprocess the biomechanical dataset"""

    def __init__(self, file_paths, sequence_length=200, step_size=5, feature_scaler=None, target_scaler=None):
        self.sequence_length = sequence_length
        self.step_size = step_size
        self.features = []
        self.targets = []
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler

        if isinstance(file_paths, str):
            file_paths = [file_paths]

        all_dfs = []
        for fp in file_paths:
            if not os.path.isfile(fp):
                print(f"  Warning: File not found, skipping: {fp}")
                continue
            df_part = pd.read_csv(fp)
            all_dfs.append(df_part)

        if not all_dfs:
            raise FileNotFoundError("No valid data files found")

        df = pd.concat(all_dfs, ignore_index=True)
        print(f"  Loaded {len(all_dfs)} files, combined shape: {df.shape}")

        joint_kinematics = [
            'pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
            'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'knee_adduction_r', 'knee_rotation_r',
            'ankle_angle_r', 'ankle_adduction_r', 'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l',
            'knee_adduction_l', 'knee_rotation_l', 'ankle_angle_l', 'ankle_adduction_l'
        ]
        left_pressure = [f'Left_pressure{i}' for i in range(1, 17)]
        left_cop = ['Left_totalForce', 'Left_cop_x', 'Left_cop_y']
        right_pressure = [f'Right_pressure{i}' for i in range(1, 17)]
        right_cop = ['Right_totalForce', 'Right_cop_x', 'Right_cop_y']

        self.input_cols = joint_kinematics + left_pressure + left_cop + right_pressure + right_cop
        self.target_cols = ['Left_GRF_ML', 'Left_GRF_V', 'Left_GRF_AP', 'Right_GRF_ML', 'Right_GRF_V', 'Right_GRF_AP']

        missing_cols = [col for col in self.input_cols + self.target_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {', '.join(missing_cols)}")

        features_df = df[self.input_cols].copy()
        targets_df = df[self.target_cols].copy()

        nan_before = features_df.isna().sum().sum() + targets_df.isna().sum().sum()
        inf_count = np.isinf(features_df.values).sum() + np.isinf(targets_df.values).sum()
        if nan_before > 0 or inf_count > 0:
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            targets_df = targets_df.replace([np.inf, -np.inf], np.nan)
            features_df = features_df.interpolate(method='linear', limit_direction='both')
            targets_df = targets_df.interpolate(method='linear', limit_direction='both')
            features_df = features_df.ffill().bfill()
            targets_df = targets_df.ffill().bfill()
            nan_after = features_df.isna().sum().sum() + targets_df.isna().sum().sum()
            print(f"  Data cleaning: {nan_before} NaN + {inf_count} Inf → {nan_after} remaining")

        features = features_df.values.astype(np.float32)
        targets = targets_df.values.astype(np.float32)

        num_sequences = (len(features) - self.sequence_length) // self.step_size + 1

        for i in range(num_sequences):
            start_idx = i * self.step_size
            end_idx = start_idx + self.sequence_length
            if end_idx <= len(features) and end_idx <= len(targets):
                seq_features = features[start_idx:end_idx]
                seq_targets = targets[start_idx:end_idx]
                if seq_features.shape == (self.sequence_length, len(self.input_cols)) and \
                   seq_targets.shape == (self.sequence_length, len(self.target_cols)):
                    self.features.append(seq_features)
                    self.targets.append(seq_targets)

        if self.features and self.targets:
            self.features = np.array(self.features, dtype=np.float32)
            self.targets = np.array(self.targets, dtype=np.float32)
            print(f"  Sequences created: {self.features.shape[0]}")
        else:
            raise ValueError("No valid sequences found")

        if self.feature_scaler is not None:
            orig_shape = self.features.shape
            self.features = self.feature_scaler.transform(
                self.features.reshape(-1, orig_shape[-1])
            ).reshape(orig_shape)

        if self.target_scaler is not None:
            orig_shape = self.targets.shape
            self.targets = self.target_scaler.transform(
                self.targets.reshape(-1, orig_shape[-1])
            ).reshape(orig_shape)

        if np.any(np.isnan(self.features)) or np.any(np.isinf(self.features)):
            raise ValueError("Features contain NaN or Inf values")
        if np.any(np.isnan(self.targets)) or np.any(np.isinf(self.targets)):
            raise ValueError("Targets contain NaN or Inf values")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        targets = torch.tensor(self.targets[idx], dtype=torch.float32)
        return features, targets


class LTCCell(nn.Module):
    """LTC cell — merged weight matrices, 4 matmuls → 2 matmuls"""

    def __init__(self, input_size, hidden_size):
        super(LTCCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ix = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hx = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.bias_all = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.tau = nn.Parameter(torch.ones(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        H = self.hidden_size
        for start in range(0, 4 * H, H):
            nn.init.orthogonal_(self.weight_ix[start:start + H])
            nn.init.orthogonal_(self.weight_hx[start:start + H])
        nn.init.constant_(self.bias_all[:H], 0)
        nn.init.constant_(self.bias_all[H:2 * H], 1)
        nn.init.constant_(self.bias_all[2 * H:3 * H], 0)
        nn.init.constant_(self.bias_all[3 * H:], 0)

    def forward(self, input_proj, hidden, cell_state):
        H = self.hidden_size
        hx = torch.mm(hidden, self.weight_hx.t())
        gates = input_proj + hx + self.bias_all

        i_gate = torch.sigmoid(gates[:, :H])
        f_gate = torch.sigmoid(gates[:, H:2 * H])
        g_gate = torch.tanh(gates[:, 2 * H:3 * H])
        o_gate = torch.sigmoid(gates[:, 3 * H:])

        new_cell = f_gate * cell_state + i_gate * g_gate
        dh = (o_gate * torch.tanh(new_cell) - hidden) / (self.tau.abs() + 1e-6)
        new_hidden = hidden + dh

        return new_hidden, new_cell


class SimpleSelfAttention(nn.Module):
    """Multi-Head Self-Attention + 相对位置偏置 + SDPA 加速"""

    def __init__(self, hidden_size, num_heads, dropout, max_len=200):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = dropout
        self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, 2 * max_len - 1))

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        pos = torch.arange(L, device=x.device)
        rel = pos.unsqueeze(1) - pos.unsqueeze(0) + (L - 1)
        bias = self.rel_pos_bias[:, rel]
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=self.attn_dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)


class CausalInputProj(nn.Module):
    def __init__(self, input_size=60, hidden_size=256):
        super().__init__()
        H4 = 4 * hidden_size
        in_ch = input_size * 2
        self.dw_s = nn.Conv1d(in_ch, in_ch, 3, dilation=1, groups=in_ch, bias=False)
        self.dw_m = nn.Conv1d(in_ch, in_ch, 7, dilation=2, groups=in_ch, bias=False)
        self.dw_l = nn.Conv1d(in_ch, in_ch, 15, dilation=3, groups=in_ch, bias=False)
        self.pad_s = (2, 0)
        self.pad_m = (12, 0)
        self.pad_l = (42, 0)
        self.pointwise = nn.Conv1d(in_ch * 3, H4, kernel_size=1)

    def forward(self, x):
        delta = torch.zeros_like(x)
        delta[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        x = torch.cat([x, delta], dim=-1)
        x = x.permute(0, 2, 1)
        xs = F.pad(x, self.pad_s)
        xm = F.pad(x, self.pad_m)
        xl = F.pad(x, self.pad_l)
        x = torch.cat([self.dw_s(xs), self.dw_m(xm), self.dw_l(xl)], dim=1)
        x = self.pointwise(x)
        x = x.permute(0, 2, 1)
        return x


class LTCGRFModel(nn.Module):
    """LTC network for GRF prediction with self-attention context aggregation"""

    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout_rate=0.5, num_heads=8):
        super(LTCGRFModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_proj = CausalInputProj(input_size, hidden_size)

        self.ltc_cells = nn.ModuleList()
        for i in range(num_layers):
            self.ltc_cells.append(LTCCell(hidden_size, hidden_size))

        self.ltc_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.self_attn = SimpleSelfAttention(hidden_size, num_heads, dropout_rate)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.skip_gate = nn.Parameter(torch.zeros(1))
        self.skip_proj = nn.Linear(4 * hidden_size, hidden_size)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        H = self.hidden_size

        input_proj_all = self.input_proj(x)

        hidden_states = [torch.zeros(batch_size, H, device=x.device)
                         for _ in range(self.num_layers)]
        cell_states = [torch.zeros(batch_size, H, device=x.device)
                       for _ in range(self.num_layers)]

        ltc_outputs = []
        for t in range(seq_len):
            h, c = self.ltc_cells[0](input_proj_all[:, t, :], hidden_states[0], cell_states[0])
            h = self.ltc_norms[0](h)
            new_hidden = [h]
            new_cell = [c]

            for i in range(1, self.num_layers):
                prev_h = new_hidden[i - 1]
                input_proj = torch.mm(prev_h, self.ltc_cells[i].weight_ix.t())
                h, c = self.ltc_cells[i](input_proj, hidden_states[i], cell_states[i])
                h = self.ltc_norms[i](h + prev_h)
                new_hidden.append(h)
                new_cell.append(c)

            hidden_states = new_hidden
            cell_states = new_cell
            ltc_outputs.append(hidden_states[-1])

        ltc_seq = torch.stack(ltc_outputs, dim=1)
        attn_out = self.self_attn(ltc_seq)
        enhanced = self.attn_norm(ltc_seq + attn_out)
        gate = torch.sigmoid(self.skip_gate)
        local_feat = self.skip_proj(input_proj_all)
        return self.fc(enhanced + gate * local_feat)


# =============================================================================
# 训练 & 评估函数
# =============================================================================

def train_model(model, train_loader, val_loader, epochs, lr, device, patience=7):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer, max_lr=lr * 2, epochs=epochs, steps_per_epoch=steps_per_epoch,
        pct_start=0.5, div_factor=5, final_div_factor=100, anneal_strategy='cos'
    )

    use_amp = device.type == 'cuda'
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16

    best_val_loss = float('inf')
    best_model = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                if use_amp:
                    with torch.amp.autocast('cuda', dtype=amp_dtype):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

        print(f'    Epoch {epoch + 1}/{epochs}: Train {train_loss:.4f}, Val {val_loss:.4f}')

    if best_model is not None:
        model.load_state_dict(best_model)
    return model


def calculate_metrics(y_true, y_pred):
    component_names = ['Left_GRF_ML', 'Left_GRF_V', 'Left_GRF_AP', 'Right_GRF_ML', 'Right_GRF_V', 'Right_GRF_AP']
    rmse_pcts, r_vals, r2_vals = [], [], []
    metrics = {}

    for i, name in enumerate(component_names):
        true_i = y_true[:, i]
        pred_i = y_pred[:, i]
        rmse = np.sqrt(mean_squared_error(true_i, pred_i))
        value_range = np.max(true_i) - np.min(true_i)
        rmse_pct = (rmse / value_range) * 100 if value_range > 0 else 0
        r_val, _ = pearsonr(true_i, pred_i)
        r2_val = r2_score(true_i, pred_i)
        rmse_pcts.append(rmse_pct)
        r_vals.append(r_val)
        r2_vals.append(r2_val)
        metrics[name] = {'RMSE%': rmse_pct, 'R': r_val, 'R2': r2_val}

    metrics['Overall'] = {
        'RMSE%': np.mean(rmse_pcts),
        'R': np.mean(r_vals),
        'R2': np.mean(r2_vals),
    }
    return metrics


def evaluate_model(model, loader, dataset, device):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    all_outputs, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    print(f'  Test Loss: {avg_loss:.6f}')

    if not all_outputs:
        return avg_loss, None

    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)

    target_scaler = getattr(dataset, 'target_scaler', None)
    if target_scaler is None and hasattr(dataset, 'dataset'):
        target_scaler = getattr(dataset.dataset, 'target_scaler', None)

    if target_scaler is not None:
        all_outputs = target_scaler.inverse_transform(
            all_outputs.reshape(-1, all_outputs.shape[-1])
        ).reshape(all_outputs.shape)
        all_targets = target_scaler.inverse_transform(
            all_targets.reshape(-1, all_targets.shape[-1])
        ).reshape(all_targets.shape)

    y_true = all_targets.reshape(-1, 6)
    y_pred = all_outputs.reshape(-1, 6)
    metrics = calculate_metrics(y_true, y_pred)
    return avg_loss, metrics


# =============================================================================
# LOSO-CV 主流程
# =============================================================================

def build_subject_file_map(subjects, run_speeds, walk_speeds):
    """构建每个受试者的文件路径映射，滤除不存在的文件"""
    subject_files = {}
    for subject in subjects:
        files = []
        for speed in run_speeds:
            speed_str = f"{int(speed * 10):02d}"
            fp = f"/root/autodl-tmp/data/rundata/{subject}_run_{speed_str}.csv"
            if os.path.isfile(fp):
                files.append(fp)
        for speed in walk_speeds:
            speed_str = f"{int(speed * 10):02d}"
            fp = f"/root/autodl-tmp/data/walkdata/{subject}_walk_{speed_str}.csv"
            if os.path.isfile(fp):
                files.append(fp)
        if files:
            subject_files[subject] = files
            print(f"{subject}: {len(files)} files found")
        else:
            print(f"{subject}: NO FILES FOUND — skipping")
    return subject_files


def print_loso_summary(all_fold_metrics):
    """打印 LOSO-CV 汇总结果"""
    component_names = ['Left_GRF_ML', 'Left_GRF_V', 'Left_GRF_AP',
                       'Right_GRF_ML', 'Right_GRF_V', 'Right_GRF_AP']

    print("\n" + "=" * 90)
    print("LOSO-CV FINAL RESULTS")
    print("=" * 90)

    # 每折结果表
    print(f"\n{'Fold':<8}", end="")
    for c in component_names + ['Overall']:
        print(f" | {c + '_R':<14} {'R2':<8}", end="")
    print()
    print("-" * 180)

    for fold_result in all_fold_metrics:
        subject = fold_result['subject']
        metrics = fold_result['metrics']
        print(f"{subject:<8}", end="")
        for key in component_names + ['Overall']:
            r = metrics[key]['R']
            r2 = metrics[key]['R2']
            print(f" | {r:<14.4f} {r2:<8.4f}", end="")
        print()

    # 均值 ± 标准差
    print("-" * 180)

    overall_r_list = [m['metrics']['Overall']['R'] for m in all_fold_metrics]
    overall_r2_list = [m['metrics']['Overall']['R2'] for m in all_fold_metrics]
    overall_rmse_list = [m['metrics']['Overall']['RMSE%'] for m in all_fold_metrics]

    print(f"\n{'Metric':<14} | {'Mean':<10} | {'Std':<10} | {'Min':<10} | {'Max':<10}")
    print("-" * 60)
    print(f"{'Overall R':<14} | {np.mean(overall_r_list):<10.4f} | {np.std(overall_r_list):<10.4f} | {np.min(overall_r_list):<10.4f} | {np.max(overall_r_list):<10.4f}")
    print(f"{'Overall R2':<14} | {np.mean(overall_r2_list):<10.4f} | {np.std(overall_r2_list):<10.4f} | {np.min(overall_r2_list):<10.4f} | {np.max(overall_r2_list):<10.4f}")
    print(f"{'Overall RMSE%':<14} | {np.mean(overall_rmse_list):<10.4f} | {np.std(overall_rmse_list):<10.4f} | {np.min(overall_rmse_list):<10.4f} | {np.max(overall_rmse_list):<10.4f}")
    print("-" * 60)

    # 各分量均值
    print(f"\n{'Component':<14} | {'R Mean':<10} | {'R Std':<10} | {'R2 Mean':<10} | {'R2 Std':<10} | {'RMSE% Mean':<10} | {'RMSE% Std':<10}")
    print("-" * 86)
    for comp in component_names:
        r_list = [m['metrics'][comp]['R'] for m in all_fold_metrics]
        r2_list = [m['metrics'][comp]['R2'] for m in all_fold_metrics]
        rmse_list = [m['metrics'][comp]['RMSE%'] for m in all_fold_metrics]
        print(f"{comp:<14} | {np.mean(r_list):<10.4f} | {np.std(r_list):<10.4f} | "
              f"{np.mean(r2_list):<10.4f} | {np.std(r2_list):<10.4f} | "
              f"{np.mean(rmse_list):<10.4f} | {np.std(rmse_list):<10.4f}")
    print("-" * 93)
    print()

    # R² 热力图：受试者 × 分量
    subjects_order = [m['subject'] for m in all_fold_metrics]
    heatmap_data = np.array([
        [m['metrics'][c]['R2'] for c in component_names]
        for m in all_fold_metrics
    ])
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(heatmap_data, cmap='RdYlGn', vmin=-0.2, vmax=1.0, aspect='auto')
    ax.set_xticks(range(6))
    ax.set_xticklabels(component_names, rotation=30, ha='right', fontsize=10)
    ax.set_yticks(range(len(subjects_order)))
    ax.set_yticklabels(subjects_order, fontsize=10)
    ax.set_title('LOSO-CV: R² per Subject × Component', fontsize=14)
    for i in range(len(subjects_order)):
        for j in range(6):
            val = heatmap_data[i, j]
            color = 'white' if val < 0.3 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=8, color=color)
    plt.colorbar(im, ax=ax, shrink=0.85, label='R²')
    plt.tight_layout()
    plt.savefig('loso_heatmap.png', dpi=300)
    print("Heatmap saved to: loso_heatmap.png\n")
    plt.close(fig)


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 参数
    subjects = [f"S{i:02d}" for i in range(4, 13)]
    run_speeds = [6.3, 8.1, 9.9]
    walk_speeds = [0.9, 1.8, 2.7, 3.6, 4.5, 5.4]
    sequence_length = 200
    step_size = 50
    batch_size = 64
    hidden_size = 256
    num_layers = 2
    dropout_rate = 0.5
    epochs = 30
    learning_rate = 0.001

    # 构建受试者-文件映射
    print("\nBuilding subject file map...")
    subject_files = build_subject_file_map(subjects, run_speeds, walk_speeds)
    valid_subjects = sorted(subject_files.keys())
    if len(valid_subjects) < 2:
        print("Need at least 2 subjects for LOSO-CV")
        return
    print(f"\nValid subjects: {valid_subjects} ({len(valid_subjects)} total)")

    all_fold_metrics = []

    for test_subject in valid_subjects:
        print(f"\n{'=' * 70}")
        print(f"FOLD: Test = {test_subject}, Train = {[s for s in valid_subjects if s != test_subject]}")
        print(f"{'=' * 70}")

        # 构建训练/测试文件列表
        train_files = []
        for s in valid_subjects:
            if s != test_subject:
                train_files.extend(subject_files[s])
        test_files = subject_files[test_subject]

        # 加载数据集
        print(f"\n[1] Loading TRAIN dataset ({len(train_files)} files)...")
        train_dataset = GRFDataset(train_files, sequence_length, step_size)

        print(f"\n[2] Loading TEST dataset ({len(test_files)} files)...")
        test_dataset = GRFDataset(test_files, sequence_length, step_size)

        # 用训练集拟合 scaler
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()

        train_feat = train_dataset.features.reshape(-1, train_dataset.features.shape[-1])
        train_targ = train_dataset.targets.reshape(-1, train_dataset.targets.shape[-1])
        feature_scaler.fit(train_feat)
        target_scaler.fit(train_targ)

        # 应用 scaler 到两个数据集
        for ds, name in [(train_dataset, 'train'), (test_dataset, 'test')]:
            ds.feature_scaler = feature_scaler
            ds.target_scaler = target_scaler
            orig = ds.features.shape
            ds.features = feature_scaler.transform(ds.features.reshape(-1, orig[-1])).reshape(orig)
            ds.targets = target_scaler.transform(ds.targets.reshape(-1, ds.targets.shape[-1])).reshape(
                ds.targets.shape[0], ds.targets.shape[1], -1)

        # 训练集中划分 10% 作为内部验证集（用于早停）
        num_train = int(0.9 * len(train_dataset))
        train_indices = np.arange(num_train)
        val_indices = np.arange(num_train, len(train_dataset))

        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(train_dataset, val_indices)

        def collate_fn(batch):
            features, targets = zip(*batch)
            return torch.stack(features), torch.stack(targets)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                  collate_fn=collate_fn, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                                collate_fn=collate_fn, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=collate_fn, num_workers=2, pin_memory=True)

        print(f"\n  Train seqs: {len(train_subset)}, Val seqs: {len(val_subset)}, Test seqs: {len(test_dataset)}")

        # 创建并训练模型
        print(f"\n[3] Training model...")
        input_size = len(train_dataset.input_cols)
        output_size = len(train_dataset.target_cols)
        model = LTCGRFModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                            num_layers=num_layers, dropout_rate=dropout_rate).to(device)
        model = train_model(model, train_loader, val_loader, epochs, learning_rate, device)

        # 评估
        print(f"\n[4] Evaluating on {test_subject}...")
        test_loss, metrics = evaluate_model(model, test_loader, test_dataset, device)
        if metrics is not None:
            all_fold_metrics.append({'subject': test_subject, 'metrics': metrics,
                                      'test_loss': test_loss})
            print(f"  {test_subject} Overall: R={metrics['Overall']['R']:.4f}, "
                  f"R2={metrics['Overall']['R2']:.4f}, RMSE%={metrics['Overall']['RMSE%']:.2f}")

        # 保存折模型
        torch.save(model.state_dict(), f"loso_model_{test_subject}.pth")

    # 最终汇总
    if all_fold_metrics:
        print_loso_summary(all_fold_metrics)

        # 保存到 JSON
        json_metrics = []
        for m in all_fold_metrics:
            fold_entry = {'subject': m['subject'], 'test_loss': m['test_loss']}
            for key, val in m['metrics'].items():
                fold_entry[key] = {k: float(v) for k, v in val.items()}
            json_metrics.append(fold_entry)

        with open('loso_results.json', 'w') as f:
            json.dump(json_metrics, f, indent=2)
        print("Results saved to loso_results.json")
    else:
        print("\nNo valid fold results to summarize.")


if __name__ == "__main__":
    main()



