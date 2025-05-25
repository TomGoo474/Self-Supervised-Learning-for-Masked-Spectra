import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 全局超参数
BATCH_SIZE = 32
NUM_WORKERS = 0
INPUT_DIM = 400
NUM_CLASSES = 2
SSL_EPOCHS = 100
CLASSIFIER_EPOCHS = 50
NUM_CLASSIFIER_RUNS = 10
SSL_LEARNING_RATE = 1e-3
CLASSIFIER_LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
DROPOUT = 0.1
MASK_RATIO = 0.9

# 确保可重复性
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# 自定义数据集
class SpectralDataset(Dataset):
    def __init__(self, spectra, labels=None):
        self.spectra = torch.tensor(spectra, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.spectra[idx], self.labels[idx]
        return self.spectra[idx]


# CBAM模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAMBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


# Mamba模块
class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super(MambaBlock, self).__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(dim * expand)

        self.in_proj = nn.Linear(dim, self.d_inner * 2)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv - 1, groups=self.d_inner)
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.d_inner + d_state)
        self.out_proj = nn.Linear(self.d_inner, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        batch, seq_len, dim = x.shape
        x, z = self.in_proj(x).chunk(2, dim=-1)
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)[:, :, :-self.d_conv]
        x = x.permute(0, 2, 1)
        x = self.act(x)

        weights, B = self.x_proj(x).split([self.d_inner, self.d_state], dim=-1)
        weights = torch.softmax(weights, dim=-1)
        x = x * weights

        x = self.out_proj(x)
        x = self.norm(x)
        return x


# MBConv模块
class MBConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, reduction=16):
        super(MBConv1d, self).__init__()
        self.stride = stride
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = int(in_channels * expand_ratio)

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Sequential(
                nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU()
            ))
        else:
            hidden_dim = in_channels

        layers.append(nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU()
        ))

        layers.append(CBAMBlock(hidden_dim, reduction=reduction))

        layers.append(nn.Sequential(
            nn.Conv1d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        ))

        self.block = nn.Sequential(*layers)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        if self.use_residual:
            residual = x
            x = self.block(x)
            x = self.dropout(x)
            x = x + residual
        else:
            x = self.block(x)
            x = self.dropout(x)
        return x


# EfficientNetEncoder
class EfficientNetEncoder(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, dropout=DROPOUT):
        super(EfficientNetEncoder, self).__init__()
        self.input_dim = input_dim
        config = [
            (32, 16, 3, 1, 1, 1),
            (16, 24, 3, 2, 6, 2),
            (24, 40, 5, 2, 6, 2),
            (40, 80, 3, 2, 6, 3),
            (80, 112, 5, 1, 6, 3),
            (112, 192, 5, 2, 6, 4),
            (192, 320, 3, 1, 6, 1)
        ]

        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.SiLU()
        )

        self.blocks = nn.ModuleList()
        in_channels = 32
        for (in_channels_config, out_channels, kernel_size, stride, expand_ratio, num_repeats) in config:
            for i in range(num_repeats):
                block_stride = stride if i == 0 else 1
                block_in_channels = in_channels if i == 0 else out_channels
                self.blocks.append(MBConv1d(
                    block_in_channels, out_channels, kernel_size, block_stride, expand_ratio, reduction=16
                ))
                in_channels = out_channels

        self.global_attention = MambaBlock(dim=320, d_state=16, d_conv=4, expand=2)

        self.head = nn.Sequential(
            nn.Conv1d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm1d(1280),
            nn.SiLU(),
            CBAMBlock(1280, reduction=16)
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 2, 1)
        x = self.global_attention(x)
        x = x.permute(0, 2, 1)
        x = self.head(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return x


# 解码器
class SpectralDecoder(nn.Module):
    def __init__(self, latent_dim=1280, output_dim=INPUT_DIM, dropout=DROPOUT):
        super(SpectralDecoder, self).__init__()
        self.fc_init = nn.Linear(latent_dim, 512)
        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512 if i == 0 else 512 * 2 ** i, 512 * 2 ** (i + 1)),
                nn.BatchNorm1d(512 * 2 ** (i + 1)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ) for i in range(2)
        ])
        self.fc_out = nn.Linear(512 * 2 ** 2, output_dim)
        self.dropout = nn.Dropout(dropout)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        x = self.fc_init(z)
        x = self.dropout(x)
        for block in self.decoder_blocks:
            x = block(x)
        recon = self.fc_out(x)
        return recon


# 自监督模型
class SSLModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(SSLModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon


# 分类器
class Classifier(nn.Module):
    def __init__(self, encoder, num_classes=NUM_CLASSES, importance_scores=None, dropout=DROPOUT):
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.input_dim = encoder.input_dim
        self.wavelength_weights = nn.Parameter(torch.empty(self.input_dim))
        if importance_scores is not None:
            importance_scores = (importance_scores - importance_scores.mean()) / (importance_scores.std() + 1e-8)
            with torch.no_grad():
                self.wavelength_weights.copy_(torch.tensor(importance_scores, dtype=torch.float32))
            print("Initialized wavelength weights based on importance scores.")
        else:
            nn.init.kaiming_uniform_(self.wavelength_weights, a=np.sqrt(5))
            print("Initialized wavelength weights with default method.")
        self.fc = nn.Linear(1280, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        z = self.encoder(x)
        sigmoid_weights = torch.sigmoid(self.wavelength_weights)
        weighted_x = x * sigmoid_weights.unsqueeze(0)
        logits = self.fc(z)
        return logits, z


# 计算重构误差
def calculate_reconstruction_errors(ssl_model, data_loader, device):
    ssl_model.eval()
    criterion = nn.MSELoss(reduction='none')
    total_errors = torch.zeros(ssl_model.encoder.input_dim, device=device)
    total_counts = torch.zeros(ssl_model.encoder.input_dim, device=device)
    with torch.no_grad():
        for x in data_loader:
            x = x.to(device)
            batch_size, seq_len = x.shape
            for pos in range(seq_len):
                x_masked = x.clone()
                x_masked[:, pos] = 0
                x_recon = ssl_model(x_masked)
                error = criterion(x_recon[:, pos], x[:, pos])
                total_errors[pos] += error.sum()
                total_counts[pos] += batch_size
    total_counts[total_counts == 0] = 1e-8
    average_errors_per_wavelength = total_errors / total_counts
    return average_errors_per_wavelength.cpu().numpy()


# 掩码函数
def apply_mask(x, mask_ratio=MASK_RATIO):
    batch_size, seq_len = x.shape
    # 计算每个样本要掩盖的特征数量
    num_masked = int(seq_len * mask_ratio)  # 精确的掩码数量，例如 400 * 0.6 = 240
    # 初始化掩码张量，全为 False
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=x.device)
    
    # 为每个样本生成随机掩码位置
    for i in range(batch_size):
        # 生成 0 到 seq_len-1 的随机排列
        indices = torch.randperm(seq_len, device=x.device)[:num_masked]
        # 将对应位置设为 True
        mask[i, indices] = True
    
    # 应用掩码
    x_masked = x.clone()
    x_masked[mask] = 0
    
   
    
    return x_masked, mask


# 自监督训练（修改为保存最佳重构损失）
def train_ssl(model, train_loader, val_loader, epochs=SSL_EPOCHS, device='cpu', mask_ratio=MASK_RATIO,
              learning_rate=SSL_LEARNING_RATE):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss(reduction='none')
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=False)
    best_val_loss = float('inf')
    best_model_path = 'best_ssl_model.pth'
    print(f"Starting SSL training on device: {device}")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_batches = 0
        for x in train_loader:
            x = x.to(device)
            x_masked, mask = apply_mask(x, mask_ratio)
            optimizer.zero_grad()
            x_recon = model(x_masked)
            loss_per_element = criterion(x_recon, x)
            loss = loss_per_element[mask].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train_loss += loss.item()
            train_batches += 1
        model.eval()
        total_val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                x_masked, mask = apply_mask(x, mask_ratio)
                x_recon = model(x_masked)
                loss_per_element = criterion(x_recon, x)
                loss = loss_per_element[mask].mean()
                total_val_loss += loss.item()
                val_batches += 1
        avg_train_loss = total_train_loss / train_batches if train_batches > 0 else 0
        avg_val_loss = total_val_loss / val_batches if val_batches > 0 else 0
        scheduler.step(avg_val_loss)
        print(f"SSL Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best SSL model with Val Loss: {best_val_loss:.4f}")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best SSL model state from {best_model_path}")
    # 保存最佳重构损失
    np.save('best_reconstruction_loss.npy', np.array([best_val_loss]))
    print(f"Saved best reconstruction loss ({best_val_loss:.4f}) as best_reconstruction_loss.npy")
    return model


# 分类器训练
def train_classifier(model, train_loader, val_loader, epochs=CLASSIFIER_EPOCHS, device='cpu',
                     learning_rate=CLASSIFIER_LEARNING_RATE, weight_decay=WEIGHT_DECAY):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5, verbose=False, threshold=1e-4)
    best_val_acc = 0.0
    best_model_path = 'best_classifier_model.pth'
    best_val_preds = None
    best_val_labels = None
    best_epoch = 0
    print(f"\nStarting Classifier training on device: {device}")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            train_batches += 1
        model.eval()
        val_preds, val_labels = [], []
        total_val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                loss = criterion(logits, y)
                total_val_loss += loss.item()
                val_batches += 1
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y.cpu().numpy())
        acc = accuracy_score(val_labels, val_preds)
        f1 = f1_score(val_labels, val_preds, average='weighted')
        avg_train_loss = total_loss / train_batches if train_batches > 0 else 0
        avg_val_loss = total_val_loss / val_batches if val_batches > 0 else 0
        scheduler.step(acc)
        print(f"Classifier Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Val Acc: {acc:.4f}, Val F1: {f1:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        if acc > best_val_acc:
            best_val_acc = acc
            best_val_preds = val_preds
            best_val_labels = val_labels
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best classifier model with Val Acc: {best_val_acc:.4f}")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best classifier model state from {best_model_path} (Epoch {best_epoch})")
    return best_val_preds, best_val_labels


# 保存潜在特征
def save_latent_features(classifier_model, data_loader, device, filename):
    classifier_model.eval()
    latent_features = []
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            _, z = classifier_model(x)
            latent_features.append(z.cpu().numpy())
    latent_features = np.concatenate(latent_features, axis=0)
    np.save(filename, latent_features)
    print(f"Saved latent features to {filename}")


# 可视化函数（修改为保存光谱数据）
def visualize_reconstruction(model, data_loader, scaler, norm_scaler, device, num_samples=3):
    model.eval()
    batch_x = next(iter(data_loader)).to(device)
    samples = batch_x[:num_samples]

    def inverse_transform(x_tensor):
        x_np = x_tensor.detach().cpu().numpy()
        x_np = norm_scaler.inverse_transform(x_np)
        x_np = scaler.inverse_transform(x_np)
        return x_np

    original_spectra = []
    masked_spectra = []
    reconstructed_spectra = []

    plt.figure(figsize=(15, 5 * num_samples))
    for i in range(num_samples):
        x = samples[i].unsqueeze(0)
        x_masked, mask = apply_mask(x, mask_ratio=MASK_RATIO)
        x_recon = model(x_masked)
        x_orig_unscaled = inverse_transform(x)
        x_masked_unscaled = inverse_transform(x_masked)
        x_recon_unscaled = inverse_transform(x_recon)

        # 保存光谱数据
        original_spectra.append(x_orig_unscaled[0])
        masked_spectra.append(x_masked_unscaled[0])
        reconstructed_spectra.append(x_recon_unscaled[0])

        plt.subplot(num_samples, 1, i + 1)
        plt.plot(x_orig_unscaled[0], label='Original', alpha=0.8)
        plt.plot(x_masked_unscaled[0], label='Masked', alpha=0.8)
        plt.plot(x_recon_unscaled[0], label='Reconstructed', alpha=0.8)
        plt.xlabel('Wavelength Point Index')
        plt.ylabel('Intensity')
        plt.title(f'Sample {i + 1} Reconstruction (Best SSL Model)')
        plt.legend()
        plt.grid(True)

    # 保存光谱数据为npy
    np.save('original_spectra.npy', np.array(original_spectra))
    np.save('masked_spectra.npy', np.array(masked_spectra))
    np.save('reconstructed_spectra.npy', np.array(reconstructed_spectra))
    print("Saved original spectra as original_spectra.npy")
    print("Saved masked spectra as masked_spectra.npy")
    print("Saved reconstructed spectra as reconstructed_spectra.npy")

    plt.tight_layout()
    plt.savefig('reconstruction.png')
    plt.close()


def visualize_confusion_matrix(val_preds, val_labels, title='Confusion Matrix', filename='confusion_matrix.png'):
    cm = confusion_matrix(val_labels, val_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non', 'AF'],
                yticklabels=['Non', 'AF'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    try:
        plt.savefig(filename)
        print(f"Saved confusion matrix as {filename}")
    except OSError as e:
        print(f"Error saving confusion matrix to {filename}: {e}")
    finally:
        plt.close()


def visualize_wavelength_weights(classifier_model, scaler):
    weights = torch.sigmoid(classifier_model.wavelength_weights.data).cpu().numpy()
    plt.figure(figsize=(12, 6))
    plt.plot(weights, label='Learned Wavelength Weight (Sigmoid)', color='purple', alpha=0.8)
    plt.xlabel('Wavelength Point Index')
    plt.ylabel('Learned Weight (0-1)')
    plt.title('Learned Wavelength Importance Weights (Classifier Model)')
    plt.legend()
    plt.grid(True)
    plt.savefig('learned_wavelength_weights.png')
    plt.close()


def plot_wavelength_errors(errors, avg_spectrum_unscaled, scaler, norm_scaler):
    avg_spectrum_unscaled = norm_scaler.inverse_transform(avg_spectrum_unscaled.reshape(1, -1))
    avg_spectrum_unscaled = scaler.inverse_transform(avg_spectrum_unscaled)[0]
    plt.figure(figsize=(12, 6))
    plt.plot(errors, label='Reconstruction Error (MSE)', color='blue', alpha=0.8)
    plt.plot(avg_spectrum_unscaled, label='Average Spectrum (Unscaled)', color='green', alpha=0.3)
    plt.xlabel('Wavelength Point Index')
    plt.ylabel('Error / Intensity')
    plt.title('Per-Wavelength Reconstruction Error (SSL Model)')
    plt.legend()
    plt.grid(True)
    plt.savefig(' reclamation_errors_importance.png')
    plt.close()


def main():
    file_path = "T_Spectr_Maize.xlsx"
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        return
    df = pd.read_excel(file_path)
    spectra = df.iloc[:, 2:].values
    labels = df['Class'].map({'Non': 0, 'AF': 1}).values
    global INPUT_DIM
    if spectra.shape[1] != INPUT_DIM:
        print(f"Adjusting input_dim to {spectra.shape[1]}")
        INPUT_DIM = spectra.shape[1]

    X_train, X_val, y_train, y_val = train_test_split(
        spectra, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}")

    scaler = StandardScaler()
    X_train_ssl = scaler.fit_transform(X_train)
    X_val_ssl = scaler.transform(X_val)
    spectra_scaled = scaler.transform(spectra)
    norm_scaler = MinMaxScaler()
    X_train_ssl = norm_scaler.fit_transform(X_train_ssl)
    X_val_ssl = norm_scaler.transform(X_val_ssl)
    spectra_normalized = norm_scaler.transform(spectra_scaled)

    scaler_clf = StandardScaler()
    X_train_clf = scaler_clf.fit_transform(X_train)
    X_val_clf = scaler_clf.transform(X_val)

    ssl_train_dataset = SpectralDataset(X_train_ssl)
    ssl_val_dataset = SpectralDataset(X_val_ssl)
    train_dataset = SpectralDataset(X_train_clf, y_train)
    val_dataset = SpectralDataset(X_val_clf, y_val)
    ssl_importance_dataset = SpectralDataset(X_train_ssl)
    ssl_recon_dataset = SpectralDataset(spectra_normalized)
    ssl_train_loader = DataLoader(ssl_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    ssl_val_loader = DataLoader(ssl_val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    ssl_importance_loader = DataLoader(ssl_importance_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                       num_workers=NUM_WORKERS)
    ssl_recon_loader = DataLoader(ssl_recon_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    encoder = EfficientNetEncoder(input_dim=INPUT_DIM, dropout=DROPOUT).to(device)
    decoder = SpectralDecoder(latent_dim=1280, output_dim=INPUT_DIM, dropout=DROPOUT).to(device)
    ssl_model = SSLModel(encoder, decoder).to(device)
    ssl_model = train_ssl(ssl_model, ssl_train_loader, ssl_val_loader, epochs=SSL_EPOCHS,
                          device=device, mask_ratio=MASK_RATIO, learning_rate=SSL_LEARNING_RATE)

    print("\nCalculating per-wavelength reconstruction errors...")
    wavelength_errors = calculate_reconstruction_errors(ssl_model, ssl_importance_loader, device)
    wavelength_errors[wavelength_errors < 0] = 0
    np.save('wavelength_errors.npy', wavelength_errors)
    print("Saved wavelength errors as wavelength_errors.npy")
    print("Finished calculating wavelength errors.")

    avg_spectrum_unscaled = spectra_normalized.mean(axis=0)
    plot_wavelength_errors(wavelength_errors, avg_spectrum_unscaled, scaler, norm_scaler)
    print("Saved per-wavelength reconstruction errors plot as reconstruction_errors_importance.png")

    visualize_reconstruction(ssl_model, ssl_recon_loader, scaler, norm_scaler, device, num_samples=3)
    print("Saved reconstruction examples as reconstruction.png")

    all_run_results = []
    classifier_accuracies = []
    best_acc_overall = 0.0
    best_f1_overall = 0.0
    best_val_preds_overall = None
    best_val_labels_overall = None
    os.makedirs('classifier_runs', exist_ok=True)
    initial_classifier = None
    for run in range(NUM_CLASSIFIER_RUNS):
        print(f"\nClassifier Run {run + 1}/{NUM_CLASSIFIER_RUNS}")
        classifier_model = Classifier(
            encoder=ssl_model.encoder,
            num_classes=NUM_CLASSES,
            importance_scores=wavelength_errors,
            dropout=DROPOUT
        ).to(device)

        # 保存初始分类器的潜在特征（仅第一次运行）
        if run == 0:
            initial_classifier = classifier_model
            save_latent_features(initial_classifier, val_loader, device, 'initial_latent_features.npy')

        val_preds, val_labels = train_classifier(
            classifier_model, train_loader, val_loader, epochs=CLASSIFIER_EPOCHS,
            device=device, learning_rate=CLASSIFIER_LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        acc = accuracy_score(val_labels, val_preds)
        f1 = f1_score(val_labels, val_preds, average='weighted')
        classifier_accuracies.append(acc)
        report = classification_report(val_labels, val_preds, target_names=['Non', 'AF'], output_dict=True)
        run_model_path = f'classifier_runs/best_classifier_model_run_{run + 1}.pth'
        torch.save(classifier_model.state_dict(), run_model_path)
        print(f"Saved best classifier model for run {run + 1} at {run_model_path}")
        cm_path = f'classifier_runs/confusion_matrix_run_{run + 1}.png'
        visualize_confusion_matrix(val_preds, val_labels,
                                   title=f'Confusion Matrix (Run {run + 1})',
                                   filename=cm_path)
        print(f"Saved confusion matrix for run {run + 1} at {cm_path}")
        all_run_results.append({
            'run': run + 1,
            'accuracy': acc,
            'f1_score': f1,
            'val_preds': val_preds,
            'val_labels': val_labels,
            'model_path': run_model_path,
            'cm_path': cm_path,
            'classification_report': report
        })
        if acc > best_acc_overall:
            best_acc_overall = acc
            best_f1_overall = f1
            best_val_preds_overall = val_preds
            best_val_labels_overall = val_labels
            torch.save(classifier_model.state_dict(), 'best_classifier_model_final.pth')
            print(f"Updated overall best classifier model with Val Acc: {best_acc_overall:.4f}")

    # 保存分类器准确率
    np.save('classifier_accuracies.npy', np.array(classifier_accuracies))
    print(f"Saved classifier accuracies as classifier_accuracies.npy")

    # 保存最佳分类器的潜在特征
    classifier_model.load_state_dict(torch.load('best_classifier_model_final.pth'))
    save_latent_features(classifier_model, val_loader, device, 'best_latent_features.npy')

    print("\n--- Summary of All Classifier Runs ---")
    for result in all_run_results:
        print(f"Run {result['run']}:")
        print(f"  Validation Accuracy: {result['accuracy']:.4f}")
        print(f"  Validation F1 Score (weighted): {result['f1_score']:.4f}")
        print(f"  Model Weights Saved: {result['model_path']}")
        print(f"  Confusion Matrix Saved: {result['cm_path']}")
        cm = confusion_matrix(result['val_labels'], result['val_preds'])
        print(f"  Confusion Matrix:\n{cm}")
        print("  Classification Report:")
        for cls in ['Non', 'AF']:
            print(f"    {cls}: Precision={result['classification_report'][cls]['precision']:.4f}, "
                  f"Recall={result['classification_report'][cls]['recall']:.4f}, "
                  f"F1={result['classification_report'][cls]['f1-score']:.4f}")
        print()

    print("\n--- Overall Best Results ---")
    print(f"Best Validation Accuracy: {best_acc_overall:.4f}")
    print(f"Best Validation F1 Score (weighted): {best_f1_overall:.4f}")
    print(f"Best Model Weights Saved: best_classifier_model_final.pth")
    cm_best = confusion_matrix(best_val_labels_overall, best_val_preds_overall)
    print(f"Best Confusion Matrix:\n{cm_best}")
    best_report = classification_report(best_val_labels_overall, best_val_preds_overall,
                                        target_names=['Non', 'AF'], output_dict=True)
    for cls in ['Non', 'AF']:
        print(f"  {cls}: Precision={best_report[cls]['precision']:.4f}, "
              f"Recall={best_report[cls]['recall']:.4f}, "
              f"F1={best_report[cls]['f1-score']:.4f}")

    visualize_confusion_matrix(best_val_preds_overall, best_val_labels_overall,
                               title='Confusion Matrix (Best Classifier Model)',
                               filename='confusion_matrix.png')
    print("Saved overall best confusion matrix as confusion_matrix.png")
    classifier_model.load_state_dict(torch.load('best_classifier_model_final.pth'))
    visualize_wavelength_weights(classifier_model, scaler)
    print("Saved learned wavelength weights plot as learned_wavelength_weights.png")
    print("\n--- Training Complete ---")


if __name__ == "__main__":
    main()