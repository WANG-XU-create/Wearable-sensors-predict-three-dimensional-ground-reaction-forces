import os
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


class GRFDataset(Dataset):
    """Load and preprocess the new biomechanical dataset"""

    def __init__(self, file_paths, sequence_length=200, step_size=5, feature_scaler=None, target_scaler=None):
        self.sequence_length = sequence_length
        # 如果未指定步长，则默认使用序列长度（即不重叠的序列）
        self.step_size = step_size
        self.features = []
        self.targets = []
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler

        # 兼容单文件路径字符串
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        # 加载并合并所有CSV文件
        all_dfs = []
        for fp in file_paths:
            if not os.path.isfile(fp):
                print(f"Warning: File not found, skipping: {fp}")
                continue
            print(f"Loading: {fp}")
            df_part = pd.read_csv(fp)
            print(f"  Shape: {df_part.shape}")
            all_dfs.append(df_part)

        if not all_dfs:
            raise FileNotFoundError("No valid data files found")

        df = pd.concat(all_dfs, ignore_index=True)
        print(f"All {len(all_dfs)} files loaded. Combined shape: {df.shape}")

        # Define input feature columns
        # Joint kinematics (22 columns)
        joint_kinematics = [
            'pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
            'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'knee_adduction_r', 'knee_rotation_r',
            'ankle_angle_r', 'ankle_adduction_r', 'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l',
            'knee_adduction_l', 'knee_rotation_l', 'ankle_angle_l', 'ankle_adduction_l'
        ]

        # Left foot pressure and COP (19 columns)
        left_pressure = [f'Left_pressure{i}' for i in range(1, 17)]
        left_cop = ['Left_totalForce', 'Left_cop_x', 'Left_cop_y']

        # Right foot pressure and COP (19 columns)
        right_pressure = [f'Right_pressure{i}' for i in range(1, 17)]
        right_cop = ['Right_totalForce', 'Right_cop_x', 'Right_cop_y']

        # Combine all input features
        self.input_cols = joint_kinematics + left_pressure + left_cop + right_pressure + right_cop

        # Define GRF target columns (6 columns)
        self.target_cols = ['Left_GRF_ML', 'Left_GRF_V', 'Left_GRF_AP', 'Right_GRF_ML', 'Right_GRF_V', 'Right_GRF_AP']

        # Check for required columns (移除了对Cycle_ID的检查)
        missing_cols = [col for col in self.input_cols + self.target_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in data file: {', '.join(missing_cols)}")

        # Extract features and targets
        features_df = df[self.input_cols].copy()
        targets_df = df[self.target_cols].copy()

        # 修复 NaN/Inf：先替换 Inf → NaN，再线性插值 + 前后填充
        nan_count_before = features_df.isna().sum().sum() + targets_df.isna().sum().sum()
        inf_count = np.isinf(features_df.values).sum() + np.isinf(targets_df.values).sum()
        if nan_count_before > 0 or inf_count > 0:
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            targets_df = targets_df.replace([np.inf, -np.inf], np.nan)
            features_df = features_df.interpolate(method='linear', limit_direction='both')
            targets_df = targets_df.interpolate(method='linear', limit_direction='both')
            features_df = features_df.ffill().bfill()
            targets_df = targets_df.ffill().bfill()
            nan_count_after = features_df.isna().sum().sum() + targets_df.isna().sum().sum()
            print(f"Data cleaning: {nan_count_before} NaN + {inf_count} Inf values → "
                  f"{nan_count_after} remaining after interpolation")

        features = features_df.values.astype(np.float32)
        targets = targets_df.values.astype(np.float32)

        # Split long sequence into subsequences using sliding window
        num_sequences = (len(features) - self.sequence_length) // self.step_size + 1
        print(f"Total samples: {len(features)}, creating {num_sequences} sequences of length {sequence_length} with step size {self.step_size}")

        for i in range(num_sequences):
            start_idx = i * self.step_size
            end_idx = start_idx + self.sequence_length

            # 确保不会超出数据范围
            if end_idx <= len(features) and end_idx <= len(targets):
                seq_features = features[start_idx:end_idx]
                seq_targets = targets[start_idx:end_idx]
                # 已移除Cycle_ID相关处理

                # 确保所有序列具有正确的形状
                if seq_features.shape == (self.sequence_length, len(self.input_cols)) and \
                   seq_targets.shape == (self.sequence_length, len(self.target_cols)):
                    self.features.append(seq_features)
                    self.targets.append(seq_targets)
                    # 已移除Cycle_ID相关处理

        # Convert to numpy arrays
        if self.features and self.targets:
            self.features = np.array(self.features, dtype=np.float32)
            self.targets = np.array(self.targets, dtype=np.float32)
            print(
                f"Final dataset shape - features: {self.features.shape}, targets: {self.targets.shape}")
        else:
            raise ValueError("No valid sequences found with consistent shapes")

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

        # 验证数据无 NaN/Inf
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
    """LTC细胞 — 合并权重矩阵，4 次 matmul → 2 次 matmul"""

    def __init__(self, input_size, hidden_size):
        super(LTCCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 合并输入投影: (4 * hidden, input) — 一次性计算 4 个门的输入分量
        self.weight_ix = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        # 合并隐藏投影: (4 * hidden, hidden) — 一次性计算 4 个门的隐藏分量
        self.weight_hx = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        # 合并偏置: (4 * hidden,)
        self.bias_all = nn.Parameter(torch.Tensor(4 * hidden_size))

        # 时间常数参数
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

        # 隐藏投影：1 次 matmul 替代 4 次
        hx = torch.mm(hidden, self.weight_hx.t())

        # 合并所有门控输入
        gates = input_proj + hx + self.bias_all

        # 切片拆分为 4 个门
        i_gate = torch.sigmoid(gates[:, :H])
        f_gate = torch.sigmoid(gates[:, H:2 * H])
        g_gate = torch.tanh(gates[:, 2 * H:3 * H])
        o_gate = torch.sigmoid(gates[:, 3 * H:])

        # 细胞状态更新
        new_cell = f_gate * cell_state + i_gate * g_gate

        # ODE 时间步进
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

        # 可学习相对位置偏置: (heads, 2L-1)
        self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, 2 * max_len - 1))

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 相对位置偏置
        pos = torch.arange(L, device=x.device)
        rel = pos.unsqueeze(1) - pos.unsqueeze(0) + (L - 1)  # (L, L), 值域 [0, 2L-2]
        bias = self.rel_pos_bias[:, rel]  # (heads, L, L)

        # SDPA (Flash Attention on RTX 5080)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=self.attn_dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)


class CausalInputProj(nn.Module):
    """多尺度因果卷积输入投影 — delta 特征 + 三尺度深度可分离卷积"""

    def __init__(self, input_size=60, hidden_size=256):
        super().__init__()
        H4 = 4 * hidden_size
        in_ch = input_size * 2  # raw + delta = 120

        # 三尺度深度卷积
        self.dw_s = nn.Conv1d(in_ch, in_ch, 3, dilation=1, groups=in_ch, bias=False)
        self.dw_m = nn.Conv1d(in_ch, in_ch, 7, dilation=2, groups=in_ch, bias=False)
        self.dw_l = nn.Conv1d(in_ch, in_ch, 15, dilation=3, groups=in_ch, bias=False)

        # 各尺度因果填充量
        self.pad_s = (2, 0)    # (3-1)*1
        self.pad_m = (12, 0)   # (7-1)*2
        self.pad_l = (42, 0)   # (15-1)*3

        # 跨通道投影
        self.pointwise = nn.Conv1d(in_ch * 3, H4, kernel_size=1)

    def forward(self, x):
        delta = torch.zeros_like(x)
        delta[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        x = torch.cat([x, delta], dim=-1)               # (b, seq, 120)
        x = x.permute(0, 2, 1)                           # (b, 120, seq)

        xs = F.pad(x, self.pad_s)
        xm = F.pad(x, self.pad_m)
        xl = F.pad(x, self.pad_l)

        xs = self.dw_s(xs)
        xm = self.dw_m(xm)
        xl = self.dw_l(xl)

        x = torch.cat([xs, xm, xl], dim=1)               # (b, 360, seq)
        x = self.pointwise(x)                             # (b, 1024, seq)
        x = x.permute(0, 2, 1)                           # (b, seq, 1024)
        return x


class LTCGRFModel(nn.Module):
    """LTC network for GRF prediction with self-attention context aggregation"""

    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout_rate=0.5, num_heads=8):
        super(LTCGRFModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 替代逐帧 matmul 的因果卷输入投影
        self.input_proj = CausalInputProj(input_size, hidden_size)

        # Stacked LTC cells（全部接收 hidden_size 维输入）
        self.ltc_cells = nn.ModuleList()
        for i in range(num_layers):
            self.ltc_cells.append(LTCCell(hidden_size, hidden_size))

        # LayerNorm for each LTC layer
        self.ltc_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])

        # Self-attention context aggregation
        self.self_attn = SimpleSelfAttention(hidden_size, num_heads, dropout_rate)
        self.attn_norm = nn.LayerNorm(hidden_size)

        # 可学习 skip gate: 局部卷积特征直连
        self.skip_gate = nn.Parameter(torch.zeros(1))
        self.skip_proj = nn.Linear(4 * hidden_size, hidden_size)

        # Per-timestep output layer
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

        # 因果卷输入投影: (b, seq, 60) → (b, seq, 4*H) — 1 次 conv 替代逐帧 matmul
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

        # Self-attention over sequence with residual
        attn_out = self.self_attn(ltc_seq)
        enhanced = self.attn_norm(ltc_seq + attn_out)

        # 门控残差直连：局部卷积特征直接透传
        gate = torch.sigmoid(self.skip_gate)
        local_feat = self.skip_proj(input_proj_all)
        return self.fc(enhanced + gate * local_feat)


def train_model(model, train_loader, val_loader, epochs, lr, device, patience=10):
    """Train model with cosine annealing LR scheduler and return best model"""

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # 计算每个epoch的步数
    steps_per_epoch = len(train_loader)
    # 创建OneCycleLR调度器
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr * 2,  # 最大学习率
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.5,  # 学习率上升阶段占总训练的比例
        div_factor=5,  # 初始学习率 = max_lr/div_factor
        final_div_factor=100,  # 最终学习率 = max_lr/final_div_factor
        anneal_strategy='cos'  # 退火策略：余弦
    )

    # 混合精度训练 (BF16，避免 FP16 溢出导致 NaN)
    use_amp = device.type == 'cuda'
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }
    best_val_loss = float('inf')
    best_model = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        epoch_learning_rates = []
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
                print(f"Warning: NaN/Inf loss detected at epoch {epoch+1}, skipping batch")
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            epoch_learning_rates.append(current_lr)
            train_loss += loss.item() * inputs.size(0)

        avg_epoch_lr = np.mean(epoch_learning_rates)
        history['learning_rates'].append(avg_epoch_lr)

        # Validation phase
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

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            epochs_no_improve = 0
            print(f"New best model at epoch {epoch + 1} with val loss: {val_loss:.6f}, lr: {avg_epoch_lr:.6f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs!")
                break

        print(
            f'Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {avg_epoch_lr:.6f}')

    if best_model is not None:
        model.load_state_dict(best_model)
    return model, history


def calculate_metrics(y_true, y_pred):
    """Calculate RMSE, R (Pearson correlation), and R² for each component and overall"""
    metrics = {}
    component_names = ['Left_GRF_ML', 'Left_GRF_V', 'Left_GRF_AP', 'Right_GRF_ML', 'Right_GRF_V', 'Right_GRF_AP']

    # Lists to store R and R² values for each component
    rmse_percent_values = []
    r_values = []
    r2_values = []

    # Calculate for each component
    for i, name in enumerate(component_names):
        true_i = y_true[:, i]
        pred_i = y_pred[:, i]

        # RMSE
        rmse = np.sqrt(mean_squared_error(true_i, pred_i))
        # Calculate range of true values
        value_range = np.max(true_i) - np.min(true_i)

        # Calculate RMSE as percentage of range
        rmse_percent = (rmse / value_range) * 100
        rmse_percent_values.append(rmse_percent)

        # Pearson correlation
        r_value, _ = pearsonr(true_i, pred_i)
        r_values.append(r_value)

        # R-squared
        r2 = r2_score(true_i, pred_i)
        r2_values.append(r2)

        metrics[name] = {
            'RMSE%': rmse_percent,
            'R': r_value,
            'R2': r2
        }

    # Calculate overall metrics as the average of component metrics
    rmse_percent_overall = np.mean(rmse_percent_values)
    r_overall = np.mean(r_values)
    r2_overall = np.mean(r2_values)

    metrics['Overall'] = {
        'RMSE%': rmse_percent_overall,
        'R': r_overall,
        'R2': r2_overall
    }

    return metrics


def print_metrics(metrics):
    """Print evaluation metrics in a formatted table"""
    print("\nEvaluation Metrics:")
    print("-" * 80)
    print(f"{'Component':<12} | {'RMSE%':<12} | {'R':<12} | {'R2':<12}")
    print("-" * 80)

    for name, values in metrics.items():
        if name != 'Overall':
            print(f"{name:<12} | {values['RMSE%']:<12.4f} | {values['R']:<12.4f} | {values['R2']:<12.4f}")

    print("-" * 80)
    overall = metrics['Overall']
    print(f"{'Overall':<12} | {overall['RMSE%']:<12.4f} | {overall['R']:<12.4f} | {overall['R2']:<12.4f}")
    print("-" * 80)


# --- 修改后的 visualize_results 函数 ---
def visualize_results(predictions, targets, metrics, sequence_idx=0, save_path=None):
    """
    Visualize predictions vs targets for a single gait cycle sequence.

    Args:
        predictions (np.ndarray): Array of predicted GRF values (shape: [num_sequences, seq_len, 6]).
        targets (np.ndarray): Array of target GRF values (shape: [num_sequences, seq_len, 6]).
        metrics (dict): Dictionary of calculated metrics.
        sequence_idx (int): Index of the sequence to visualize.
        save_path (str, optional): If provided, save the figure to this path.
    """
    component_names = ['Left_GRF_ML', 'Left_GRF_V', 'Left_GRF_AP', 'Right_GRF_ML', 'Right_GRF_V', 'Right_GRF_AP']
    component_units = ['Force (N)'] * 6

    # Check if the sequence index is valid
    if sequence_idx >= len(predictions) or sequence_idx >= len(targets):
        print(f"Invalid sequence index {sequence_idx}. Available range: 0-{min(len(predictions), len(targets)) - 1}")
        return

    print(f"Plotting single gait cycle sequence {sequence_idx}")

    # Get the specific sequence to plot
    sequence_predictions = predictions[sequence_idx]
    sequence_targets = targets[sequence_idx]

    # Create time axis for the sequence
    total_timesteps = sequence_predictions.shape[0]
    time_axis = np.arange(total_timesteps)  # Simple timestep index

    # --- Create the plot ---
    fig, axs = plt.subplots(3, 2, figsize=(18, 12))  # Wider figure for better layout
    axs = axs.flatten()

    for i in range(6):
        ax = axs[i]
        # Plot target (ground truth)
        ax.plot(time_axis, sequence_targets[:, i], 'b-', linewidth=2, label='Target', alpha=0.8)
        # Plot prediction
        ax.plot(time_axis, sequence_predictions[:, i], 'r--', linewidth=2, label='Prediction', alpha=0.8)

        ax.set_title(f'{component_names[i]}', fontsize=14)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel(component_units[i], fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=10)

        # Add metrics text box if available for this component
        if metrics and component_names[i] in metrics:
            comp_metrics = metrics[component_names[i]]
            metrics_text = (f'RMSE%: {comp_metrics["RMSE%"]:.2f}\n'
                            f'R: {comp_metrics["R"]:.4f}\n'
                            f'R2: {comp_metrics["R2"]:.4f}')
            ax.text(0.02, 0.98, metrics_text,
                    transform=ax.transAxes,
                    verticalalignment='top',
                    horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Make room for suptitle
    plt.suptitle(f'Single Gait Cycle Sequence {sequence_idx} - GRF Predictions vs Targets', fontsize=16)
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()


# --- 修改后的 evaluate_model 函数 ---
def evaluate_model(model, loader, dataset, device):
    """Evaluate model performance, calculate metrics, and visualize results"""
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            # Collect results for visualization and metrics
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Calculate average loss
    avg_loss = total_loss / len(loader.dataset)
    print(f'Test Loss: {avg_loss:.6f}')

    # Concatenate all batches
    if all_outputs:
        all_outputs = np.concatenate(all_outputs)
        all_targets = np.concatenate(all_targets)

        # 获取数据集的target_scaler
        if hasattr(dataset, 'target_scaler'):
            target_scaler = dataset.target_scaler
        elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'target_scaler'):
            target_scaler = dataset.dataset.target_scaler
        else:
            raise AttributeError("Dataset does not have target_scaler attribute")

        # Denormalize GRF values
        outputs_denorm = target_scaler.inverse_transform(
            all_outputs.reshape(-1, all_outputs.shape[-1])
        ).reshape(all_outputs.shape)

        targets_denorm = target_scaler.inverse_transform(
            all_targets.reshape(-1, all_targets.shape[-1])
        ).reshape(all_targets.shape)

        # Calculate evaluation metrics
        y_true = targets_denorm.reshape(-1, 6)  # Reshape to (n_samples * sequence_length, 6)
        y_pred = outputs_denorm.reshape(-1, 6)
        metrics = calculate_metrics(y_true, y_pred)
        print_metrics(metrics)

        # 保存验证集序列的随机一段序列的真实值与预测值对比图
        import random
        random_seq_idx = random.randint(0, outputs_denorm.shape[0] - 1)
        save_path = f"grf_prediction_vs_target_val_seq{random_seq_idx}.png"
        visualize_results(outputs_denorm, targets_denorm, metrics, sequence_idx=random_seq_idx, save_path=save_path)
        return avg_loss, metrics
    else:
        print("No data for evaluation")
        return avg_loss, None


def plot_training_history(history):
    """Plot training history including loss and learning rate"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('MSE Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot learning rate
    ax2.plot(history['learning_rates'], label='Learning Rate', color='red')
    ax2.set_title('Learning Rate Schedule')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Learning Rate')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    print("Training history figure saved to: training_history.png")
    plt.close(fig)


def main():
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 构建全部受试者×全部速度的文件路径列表
    subjects = [f"S{i:02d}" for i in range(4, 13)]  # S04 ~ S12
    run_speeds = [6.3, 8.1, 9.9]       # 跑步速度 (km/h)
    walk_speeds = [0.9, 1.8, 2.7, 3.6, 4.5, 5.4]  # 步行速度 (km/h)

    file_paths = []
    for subject in subjects:
        for speed in run_speeds:
            speed_str = f"{int(speed * 10):02d}"
            file_paths.append(f"/root/autodl-tmp/data/rundata/{subject}_run_{speed_str}.csv")
        for speed in walk_speeds:
            speed_str = f"{int(speed * 10):02d}"
            file_paths.append(f"/root/autodl-tmp/data/walkdata/{subject}_walk_{speed_str}.csv")

    print(f"Prepared {len(file_paths)} file paths ({len(subjects)} subjects × "
          f"{len(run_speeds) + len(walk_speeds)} speeds)")

    # Create dataset with sliding window
    sequence_length = 200  
    step_size = 50
    try:
        dataset = GRFDataset(file_paths, sequence_length, step_size)
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return

    # 按照序列顺序划分训练集和验证集，前80%为训练集，后20%为验证集
    num_sequences = len(dataset)
    num_train = int(0.8 * num_sequences)
    train_indices = np.arange(0, num_train)
    val_indices = np.arange(num_train, num_sequences)

    # 复用已计算的scalers
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # 获取训练集数据并拟合scalers
    train_features_for_scaling = np.concatenate([dataset.features[i] for i in train_indices])
    train_targets_for_scaling = np.concatenate([dataset.targets[i] for i in train_indices])
    feature_scaler.fit(train_features_for_scaling.reshape(-1, train_features_for_scaling.shape[-1]))
    target_scaler.fit(train_targets_for_scaling.reshape(-1, train_targets_for_scaling.shape[-1]))

    # 应用scalers到完整数据集
    full_features_scaled = feature_scaler.transform(
        dataset.features.reshape(-1, dataset.features.shape[-1])
    ).reshape(dataset.features.shape)
    full_targets_scaled = target_scaler.transform(
        dataset.targets.reshape(-1, dataset.targets.shape[-1])
    ).reshape(dataset.targets.shape)

    # 直接创建训练集和验证集（使用Subset避免重复加载数据）
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    # 为Subset设置scalers属性
    train_dataset.dataset.feature_scaler = feature_scaler
    train_dataset.dataset.target_scaler = target_scaler
    val_dataset.dataset.feature_scaler = feature_scaler
    val_dataset.dataset.target_scaler = target_scaler
    
    # 更新归一化后的数据
    train_dataset.dataset.features = full_features_scaled
    train_dataset.dataset.targets = full_targets_scaled

    print(f"Total sequences: {len(dataset)}, Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create data loaders
    batch_size = 256
    # 使用自定义collate_fn来处理Subset的情况
    def collate_fn(batch):
        features, targets = zip(*batch)
        return torch.stack(features), torch.stack(targets)
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                             num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                           num_workers=2, pin_memory=True, persistent_workers=True)

    # Model parameters - updated for new dataset
    input_size = len(dataset.input_cols)  # 60 features (22 joint kinematics + 38 pressure/COP)
    output_size = len(dataset.target_cols)  # 6 outputs (3D GRF for both feet)
    hidden_size = 256
    num_layers = 2

    print(
        f"Model parameters: input_size={input_size}, output_size={output_size}, hidden_size={hidden_size}, num_layers={num_layers}")

    # Create model
    dropout_rate = 0.5
    model = LTCGRFModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                        num_layers=num_layers, dropout_rate=dropout_rate).to(device)
    print(f"Model architecture:\n{model}")

    # Train model
    epochs = 50
    learning_rate = 0.001
    print(f"Starting training: epochs={epochs}, lr={learning_rate}")
    trained_model, history = train_model(model, train_loader, val_loader, epochs, learning_rate, device)

    # Plot training history
    plot_training_history(history)

    # Evaluate model on validation set
    print("Evaluating model performance on validation set...")
    val_loss, metrics = evaluate_model(trained_model, val_loader, val_dataset, device)

    # Save model
    save_path = 'ltc_grf_model_both_feet_by_cycle.pth'
    torch.save(trained_model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")

if __name__ == "__main__":
    main()