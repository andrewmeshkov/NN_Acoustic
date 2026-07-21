"""
Пайплайн каскадной классификации рефлектограмм:
WPD → PSNR-отбор каналов → CNN + предфильтр (PCA + LogReg).

Поддерживает синтетические данные и H5-датасет (64×64×3 спектрограммы).
"""

from __future__ import annotations

import argparse
import copy
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
  import h5py
except ImportError:
  h5py = None
import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Конфигурации
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
  data_source: str = "synthetic"          # synthetic | h5
  h5_path: str = "test_on_Polygon_I_direct_butt.h5"
  target_meta_class: str = "human__step"
  h5_max_samples: Optional[int] = None    # ограничение для быстрых прогонов

  num_samples: int = 500
  time_len: int = 6400
  num_sensors: int = 128
  target_ratio: float = 0.1
  seed: int = 42

  wavelet: str = "db12"
  wpd_level: int = 6
  wpd_mode: str = "full"                  # full | center | stride
  wpd_sensor_stride: int = 8              # для mode=stride

  top_k_channels: int = 3
  low_freq_only: bool = True              # отбор только из низкочастотных каналов

  model_name: str = "paper"
  use_prefilter: bool = True
  checkpoint_path: str = "best_cnn.pth"

  batch_size: int = 32
  epochs: int = 15
  lr: float = 1e-3
  weight_decay: float = 1e-4
  optimizer: str = "adam"                 # adam | adamw
  scheduler_patience: int = 3
  early_stopping_patience: int = 7
  pos_weight: Optional[float] = None      # для BCE; None = авто по train

  prefilter_k: int = 8
  prefilter_degree: int = 4
  prefilter_class_weight: float = 800.0
  prefilter_target_recall: float = 0.99

  device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ModelConfig:
  name: str
  arch: str                               # paper | compact | deep | wide | spectrogram
  encoder_blocks: int = 10
  base_channels: int = 64
  encoder_out_channels: int = 3
  pool_size: Tuple[int, int] = (16, 16)
  classifier_channels: Tuple[int, int] = (16, 32)
  fc_hidden: Tuple[int, int] = (64, 32)
  dropout: float = 0.0


MODEL_CONFIGS: Dict[str, ModelConfig] = {
  "paper": ModelConfig(
    name="paper",
    arch="paper",
    encoder_blocks=10,
    base_channels=64,
    encoder_out_channels=3,
    pool_size=(16, 16),
    classifier_channels=(16, 32),
    fc_hidden=(64, 32),
    dropout=0.0,
  ),
  "compact": ModelConfig(
    name="compact",
    arch="paper",
    encoder_blocks=4,
    base_channels=32,
    encoder_out_channels=3,
    pool_size=(8, 8),
    classifier_channels=(16, 32),
    fc_hidden=(32, 16),
    dropout=0.1,
  ),
  "deep": ModelConfig(
    name="deep",
    arch="paper",
    encoder_blocks=14,
    base_channels=64,
    encoder_out_channels=3,
    pool_size=(16, 16),
    classifier_channels=(32, 64),
    fc_hidden=(128, 64),
    dropout=0.15,
  ),
  "wide": ModelConfig(
    name="wide",
    arch="paper",
    encoder_blocks=8,
    base_channels=96,
    encoder_out_channels=3,
    pool_size=(16, 16),
    classifier_channels=(32, 64),
    fc_hidden=(128, 64),
    dropout=0.1,
  ),
  "spectrogram": ModelConfig(
    name="spectrogram",
    arch="spectrogram",
    encoder_blocks=0,
    base_channels=32,
    encoder_out_channels=0,
    pool_size=(1, 1),
    classifier_channels=(32, 64),
    fc_hidden=(128, 64),
    dropout=0.2,
  ),
  "spectrogram_deep": ModelConfig(
    name="spectrogram_deep",
    arch="spectrogram",
    encoder_blocks=0,
    base_channels=48,
    encoder_out_channels=0,
    pool_size=(1, 1),
    classifier_channels=(64, 128),
    fc_hidden=(256, 128),
    dropout=0.25,
  ),
}


# ---------------------------------------------------------------------------
# 1. Данные
# ---------------------------------------------------------------------------

def generate_synthetic_data(
  num_samples: int = 1000,
  time_len: int = 6400,
  num_sensors: int = 128,
  target_ratio: float = 0.1,
  seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
  """Генерирует синтетические рефлектограммы (target = импульсы копания)."""
  rng = np.random.default_rng(seed)
  X = rng.standard_normal((num_samples, time_len, num_sensors)) * 0.5
  y = np.zeros(num_samples, dtype=np.int64)

  n_target = int(num_samples * target_ratio)
  target_idx = rng.choice(num_samples, n_target, replace=False)
  y[target_idx] = 1

  sensor_center = num_sensors // 2
  t_grid = np.arange(time_len)
  s_grid = np.arange(num_sensors)

  for idx in target_idx:
    for t_center in (1000, 3000, 5000):
      duration = 100
      sigma_t = max(duration // 4, 1)
      sigma_s = 10.0
      t_mask = np.abs(t_grid - t_center) <= duration // 2
      amp_t = np.exp(-((t_grid[t_mask] - t_center) ** 2) / (2 * sigma_t ** 2))
      amp_s = np.exp(-((s_grid - sensor_center) ** 2) / (2 * sigma_s ** 2))
      X[idx, t_mask, :] += 3.0 * amp_t[:, None] * amp_s[None, :]

  X += rng.standard_normal(X.shape) * 0.1
  return X.astype(np.float32), y


def build_h5_class_mapping(class_list: List[str], target_meta_class: str) -> np.ndarray:
  """161 класс → 0 (target) / 1 (noise). Классы near__ считаются шумом."""
  target_classes = sorted({target_meta_class, "noise"})
  mapping = []
  for cl in class_list:
    if target_meta_class in cl and "near" not in cl:
      mapping.append(target_classes.index(target_meta_class))
    else:
      mapping.append(target_classes.index("noise"))
  return np.array(mapping, dtype=np.int64)


def load_h5_data(
  h5_path: str,
  target_meta_class: str = "human__step",
  max_samples: Optional[int] = None,
  seed: int = 42,
) -> Dict[str, np.ndarray]:
  """Загружает train/val/test из H5. y: 0=target, 1=noise."""
  if h5py is None:
    raise ImportError("Для H5-режима установите: pip install h5py")
  with h5py.File(h5_path, "r") as f:
    class_list = f.attrs["class_map"].split(",")
    class_mapping = build_h5_class_mapping(class_list, target_meta_class)

    result = {}
    for split in ("train", "val", "test"):
      x = f[f"{split}/x"][:].astype(np.float32)
      y_raw = f[f"{split}/y"][:, 0]
      y = class_mapping[y_raw]
      y = (y != 0).astype(np.int64)  # 0=target→0, noise→1; инвертируем: 1=target
      y = 1 - y  # теперь 1=target (human__step), 0=noise — как в синтетике

      if max_samples is not None and len(x) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x), max_samples, replace=False)
        x, y = x[idx], y[idx]

      result[f"x_{split}"] = x
      result[f"y_{split}"] = y

  return result


# ---------------------------------------------------------------------------
# 2. WPD
# ---------------------------------------------------------------------------

def _wpd_single_sensor(
  sensor_data: np.ndarray,
  wavelet: str,
  level: int,
) -> np.ndarray:
  wp = pywt.WaveletPacket(data=sensor_data, wavelet=wavelet, mode="symmetric", maxlevel=level)
  nodes = wp.get_level(level, "freq")
  return np.stack([node.data for node in nodes], axis=0)


def apply_wpd(
  reflectogram: np.ndarray,
  wavelet: str = "db12",
  level: int = 6,
  mode: str = "full",
  sensor_stride: int = 8,
) -> np.ndarray:
  """
  WPD по сенсорам.
  Возвращает (num_channels, channel_length, num_sensors_used).
  """
  _, num_sensors = reflectogram.shape

  if mode == "center":
    sensor_indices = [num_sensors // 2]
  elif mode == "stride":
    sensor_indices = list(range(0, num_sensors, max(sensor_stride, 1)))
  else:
    sensor_indices = list(range(num_sensors))

  first = _wpd_single_sensor(reflectogram[:, sensor_indices[0]], wavelet, level)
  num_channels, channel_length = first.shape
  channels = np.zeros((num_channels, channel_length, len(sensor_indices)), dtype=np.float32)
  channels[:, :, 0] = first

  for i, s in enumerate(sensor_indices[1:], start=1):
    channels[:, :, i] = _wpd_single_sensor(reflectogram[:, s], wavelet, level)

  return channels


def apply_wpd_batch(
  reflectograms: np.ndarray,
  wavelet: str = "db12",
  level: int = 6,
  mode: str = "full",
  sensor_stride: int = 8,
) -> np.ndarray:
  """(N, time, sensors) → (N, num_channels, channel_length, num_sensors_used)."""
  out = []
  for i in tqdm(range(len(reflectograms)), desc="WPD", leave=False):
    out.append(apply_wpd(reflectograms[i], wavelet, level, mode, sensor_stride))
  return np.stack(out, axis=0)


# ---------------------------------------------------------------------------
# 3. Признаки
# ---------------------------------------------------------------------------

def compute_psnr(image: np.ndarray) -> float:
  max_val = float(np.max(image))
  min_val = float(np.min(image))
  if max_val == min_val:
    return 0.0
  sigma = float(np.std(image))
  if sigma == 0.0:
    return 0.0
  return 20.0 * np.log10((max_val - min_val) / sigma)


def select_channels_by_psnr(
  train_channels: np.ndarray,
  top_k: int = 3,
  low_freq_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """
  train_channels: (N, C, H, W) или (N, C, H).
  Возвращает (indices, psnr_avg, psnr_std).
  """
  n_channels = train_channels.shape[1]
  psnr_avg = np.zeros(n_channels)
  psnr_std = np.zeros(n_channels)

  for c in range(n_channels):
    vals = []
    for i in range(len(train_channels)):
      img = train_channels[i, c]
      vals.append(compute_psnr(img))
    psnr_avg[c] = np.mean(vals)
    psnr_std[c] = np.std(vals)

  search_range = n_channels
  if low_freq_only:
    search_range = max(n_channels // 4, top_k)

  candidate_idx = np.arange(search_range)
  ranked = candidate_idx[np.argsort(psnr_avg[candidate_idx])]
  top_channels = ranked[-top_k:]
  return top_channels, psnr_avg, psnr_std


def extract_std_features(channels: np.ndarray) -> np.ndarray:
  if channels.ndim == 4:
    return np.std(channels, axis=(2, 3))
  if channels.ndim == 3:
    return np.std(channels, axis=(1, 2))
  return np.std(channels, axis=1)


def prepare_cnn_data(channels: np.ndarray, selected_indices: np.ndarray) -> np.ndarray:
  """
  (N, C, H, W) → (N, len(selected), H, W) — вход для Conv2d.
  Для 3D (N, C, H) добавляет W=1.
  """
  x = channels[:, selected_indices]
  if x.ndim == 3:
    x = x[:, :, :, np.newaxis]
  return x.astype(np.float32)


def prepare_spectrogram_cnn(x: np.ndarray) -> np.ndarray:
  """(N, H, W, C) → (N, C, H, W)."""
  return np.transpose(x, (0, 3, 1, 2)).astype(np.float32)


# ---------------------------------------------------------------------------
# 4. Модели
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
  def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)
    self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    self.shortcut = (
      nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1),
        nn.BatchNorm2d(out_channels),
      )
      if in_channels != out_channels
      else nn.Identity()
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = self.dropout(self.relu(self.bn1(self.conv1(x))))
    out = self.bn2(self.conv2(out))
    out = self.relu(out + self.shortcut(x))
    return out


class Encoder(nn.Module):
  def __init__(self, in_channels: int, cfg: ModelConfig):
    super().__init__()
    blocks = [ResidualBlock(in_channels, cfg.base_channels, cfg.dropout)]
    for _ in range(1, cfg.encoder_blocks):
      blocks.append(ResidualBlock(cfg.base_channels, cfg.base_channels, cfg.dropout))
    self.blocks = nn.Sequential(*blocks)
    self.adaptive_pool = nn.AdaptiveAvgPool2d(cfg.pool_size)
    self.conv_out = nn.Conv2d(cfg.base_channels, cfg.encoder_out_channels, 3, padding=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.blocks(x)
    x = self.adaptive_pool(x)
    return self.conv_out(x)


class ClassifierHead(nn.Module):
  def __init__(self, in_channels: int, cfg: ModelConfig):
    super().__init__()
    c1, c2 = cfg.classifier_channels
    h1, h2 = cfg.fc_hidden
    self.features = nn.Sequential(
      nn.Conv2d(in_channels, c1, 3, padding=1),
      nn.BatchNorm2d(c1),
      nn.ReLU(inplace=True),
      nn.Conv2d(c1, c2, 3, padding=1),
      nn.BatchNorm2d(c2),
      nn.ReLU(inplace=True),
      nn.AdaptiveAvgPool2d((1, 1)),
    )
    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(c2, h1),
      nn.ReLU(inplace=True),
      nn.Dropout(cfg.dropout),
      nn.Linear(h1, h2),
      nn.ReLU(inplace=True),
      nn.Dropout(cfg.dropout),
      nn.Linear(h2, 1),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.classifier(self.features(x))


class PaperCNN(nn.Module):
  def __init__(self, in_channels: int, cfg: ModelConfig):
    super().__init__()
    self.encoder = Encoder(in_channels, cfg)
    self.classifier = ClassifierHead(cfg.encoder_out_channels, cfg)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(self.classifier(self.encoder(x)))


class SpectrogramCNN(nn.Module):
  """Прямая CNN для H5-спектрограмм 64×64×3."""

  def __init__(self, in_channels: int, cfg: ModelConfig):
    super().__init__()
    ch = cfg.base_channels
    self.features = nn.Sequential(
      nn.Conv2d(in_channels, ch, 3, padding=1),
      nn.BatchNorm2d(ch),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2),
      ResidualBlock(ch, ch * 2, cfg.dropout),
      nn.MaxPool2d(2),
      ResidualBlock(ch * 2, ch * 4, cfg.dropout),
      nn.MaxPool2d(2),
      ResidualBlock(ch * 4, ch * 4, cfg.dropout),
      nn.AdaptiveAvgPool2d((1, 1)),
    )
    c2 = cfg.classifier_channels[1]
    h1, h2 = cfg.fc_hidden
    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(ch * 4, h1),
      nn.ReLU(inplace=True),
      nn.Dropout(cfg.dropout),
      nn.Linear(h1, h2),
      nn.ReLU(inplace=True),
      nn.Dropout(cfg.dropout),
      nn.Linear(h2, 1),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(self.classifier(self.features(x)))


def build_model(in_channels: int, model_name: str) -> nn.Module:
  if model_name not in MODEL_CONFIGS:
    raise ValueError(f"Неизвестная модель: {model_name}. Доступны: {list(MODEL_CONFIGS)}")
  cfg = MODEL_CONFIGS[model_name]
  if cfg.arch == "spectrogram":
    return SpectrogramCNN(in_channels, cfg)
  return PaperCNN(in_channels, cfg)


# ---------------------------------------------------------------------------
# 5. Обучение и оценка
# ---------------------------------------------------------------------------

def _make_optimizer(model: nn.Module, cfg: PipelineConfig) -> optim.Optimizer:
  if cfg.optimizer == "adamw":
    return optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
  return optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def _compute_pos_weight(y: np.ndarray) -> float:
  n_pos = max(int((y == 1).sum()), 1)
  n_neg = max(int((y == 0).sum()), 1)
  return n_neg / n_pos


@dataclass
class TrainHistory:
  best_val_acc: float = 0.0
  best_epoch: int = 0
  train_losses: List[float] = field(default_factory=list)
  val_accs: List[float] = field(default_factory=list)


def train_cnn(
  model: nn.Module,
  train_loader: DataLoader,
  val_loader: DataLoader,
  cfg: PipelineConfig,
  pos_weight: Optional[float] = None,
) -> TrainHistory:
  device = cfg.device
  model.to(device)
  optimizer = _make_optimizer(model, cfg)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=cfg.scheduler_patience, factor=0.5
  )

  if pos_weight is None:
    pos_weight = cfg.pos_weight
  pw = float(pos_weight) if pos_weight is not None else None

  history = TrainHistory()
  best_state = copy.deepcopy(model.state_dict())
  epochs_no_improve = 0

  for epoch in range(cfg.epochs):
    model.train()
    train_loss = 0.0
    for x_batch, y_batch in train_loader:
      x_batch = x_batch.to(device)
      y_batch = y_batch.to(device).float().view(-1, 1)
      optimizer.zero_grad()
      pred = model(x_batch)
      if pw is not None:
        sample_w = torch.where(y_batch > 0.5, pw, 1.0)
        loss = (sample_w * nn.functional.binary_cross_entropy(pred, y_batch, reduction="none")).mean()
      else:
        loss = nn.functional.binary_cross_entropy(pred, y_batch)
      loss.backward()
      optimizer.step()
      train_loss += loss.item()

    model.eval()
    val_probs, val_labels = [], []
    with torch.no_grad():
      for x_batch, y_batch in val_loader:
        x_batch = x_batch.to(device)
        pred = model(x_batch)
        val_probs.extend(pred.cpu().numpy().flatten())
        val_labels.extend(y_batch.numpy())

    val_probs_arr = np.array(val_probs)
    val_preds = val_probs_arr > 0.5
    val_acc = accuracy_score(val_labels, val_preds)
    scheduler.step(val_acc)

    history.train_losses.append(train_loss / max(len(train_loader), 1))
    history.val_accs.append(val_acc)

    if val_acc > history.best_val_acc:
      history.best_val_acc = val_acc
      history.best_epoch = epoch + 1
      best_state = copy.deepcopy(model.state_dict())
      torch.save(best_state, cfg.checkpoint_path)
      epochs_no_improve = 0
    else:
      epochs_no_improve += 1

    if (epoch + 1) % 5 == 0 or epoch == 0:
      print(
        f"  Epoch {epoch + 1}/{cfg.epochs} | "
        f"loss={history.train_losses[-1]:.4f} | val_acc={val_acc:.4f}"
      )

    if epochs_no_improve >= cfg.early_stopping_patience:
      print(f"  Early stopping на эпохе {epoch + 1}")
      break

  model.load_state_dict(best_state)
  return history


@torch.no_grad()
def predict_cnn(
  model: nn.Module,
  loader: DataLoader,
  device: str,
) -> Tuple[np.ndarray, np.ndarray]:
  model.eval()
  probs, labels = [], []
  for x_batch, y_batch in loader:
    x_batch = x_batch.to(device)
    out = model(x_batch)
    probs.extend(out.cpu().numpy().flatten())
    labels.extend(y_batch.numpy())
  return np.array(probs), np.array(labels)


def evaluate_predictions(
  y_true: np.ndarray,
  y_prob: np.ndarray,
  threshold: float = 0.5,
) -> Dict[str, float]:
  y_pred = y_prob > threshold
  metrics = {
    "accuracy": accuracy_score(y_true, y_pred),
    "recall": recall_score(y_true, y_pred, zero_division=0),
    "precision": precision_score(y_true, y_pred, zero_division=0),
  }
  if len(np.unique(y_true)) > 1:
    metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
  else:
    metrics["roc_auc"] = float("nan")
  return metrics


# ---------------------------------------------------------------------------
# 6. Предфильтр
# ---------------------------------------------------------------------------

def build_prefilter(
  X_train: np.ndarray,
  y_train: np.ndarray,
  k: int = 8,
  degree: int = 4,
  class_weight_target: float = 800.0,
) -> Tuple[PCA, PolynomialFeatures, LogisticRegression]:
  pca = PCA(n_components=k)
  X_pca = pca.fit_transform(X_train)
  poly = PolynomialFeatures(degree=degree, include_bias=False)
  X_poly = poly.fit_transform(X_pca)
  lr = LogisticRegression(
    class_weight={0: 1.0, 1: class_weight_target},
    max_iter=1000,
    random_state=42,
  )
  lr.fit(X_poly, y_train)
  return pca, poly, lr


def apply_prefilter(
  X: np.ndarray,
  pca: PCA,
  poly: PolynomialFeatures,
  lr_model: LogisticRegression,
  threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
  X_poly = poly.transform(pca.transform(X))
  probs = lr_model.predict_proba(X_poly)[:, 1]
  return probs > threshold, probs


def tune_prefilter_threshold(
  val_stds: np.ndarray,
  y_val: np.ndarray,
  pca: PCA,
  poly: PolynomialFeatures,
  lr_model: LogisticRegression,
  target_recall: float = 0.99,
) -> Tuple[float, float]:
  """
  Минимальный порог, при котором recall ≥ target_recall
  (максимизирует drop rate при сохранении recall).
  """
  thresholds = np.linspace(0.01, 0.99, 99)
  best_th, best_recall = 0.5, 0.0
  for th in thresholds:
    mask, _ = apply_prefilter(val_stds, pca, poly, lr_model, threshold=th)
    rec = recall_score(y_val, mask, zero_division=0)
    if rec >= target_recall:
      best_th, best_recall = th, rec
      break
    if rec > best_recall:
      best_recall, best_th = rec, th
  return best_th, best_recall


def evaluate_cascade(
  y_true: np.ndarray,
  prefilter_mask: np.ndarray,
  cnn_probs: np.ndarray,
  cnn_threshold: float = 0.5,
) -> Dict[str, float]:
  """
  End-to-end: target обнаружен, если прошёл предфильтр И CNN предсказал target.
  """
  cnn_positive = cnn_probs > cnn_threshold
  final_positive = prefilter_mask & cnn_positive
  return {
    "e2e_recall": recall_score(y_true, final_positive, zero_division=0),
    "e2e_precision": precision_score(y_true, final_positive, zero_division=0),
    "prefilter_recall": recall_score(y_true, prefilter_mask, zero_division=0),
    "prefilter_drop_rate": 1.0 - float(np.mean(prefilter_mask)),
    "cnn_recall_on_kept": recall_score(
      y_true[prefilter_mask], cnn_positive[prefilter_mask], zero_division=0
    ) if prefilter_mask.any() else 0.0,
  }


# ---------------------------------------------------------------------------
# 7. Основной пайплайн
# ---------------------------------------------------------------------------

def run_synthetic_pipeline(cfg: PipelineConfig) -> Dict[str, object]:
  print("=== Синтетический пайплайн ===")
  print(f"Модель: {cfg.model_name} | WPD: {cfg.wpd_mode} | device: {cfg.device}")

  X, y = generate_synthetic_data(
    cfg.num_samples, cfg.time_len, cfg.num_sensors, cfg.target_ratio, cfg.seed
  )
  X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=cfg.seed, stratify=y
  )
  X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=cfg.seed, stratify=y_temp
  )

  print("WPD...")
  train_ch = apply_wpd_batch(X_train, cfg.wavelet, cfg.wpd_level, cfg.wpd_mode, cfg.wpd_sensor_stride)
  val_ch = apply_wpd_batch(X_val, cfg.wavelet, cfg.wpd_level, cfg.wpd_mode, cfg.wpd_sensor_stride)
  test_ch = apply_wpd_batch(X_test, cfg.wavelet, cfg.wpd_level, cfg.wpd_mode, cfg.wpd_sensor_stride)

  top_channels, psnr_avg, psnr_std = select_channels_by_psnr(
    train_ch, cfg.top_k_channels, cfg.low_freq_only
  )
  print("PSNR (топ каналов):")
  for c in top_channels:
    print(f"  канал {c}: {psnr_avg[c]:.2f} ± {psnr_std[c]:.2f} dB")
  print(f"Выбранные каналы: {top_channels}")

  X_train_cnn = prepare_cnn_data(train_ch, top_channels)
  X_val_cnn = prepare_cnn_data(val_ch, top_channels)
  X_test_cnn = prepare_cnn_data(test_ch, top_channels)

  return _train_evaluate(
    cfg, X_train_cnn, y_train, X_val_cnn, y_val, X_test_cnn, y_test,
    train_ch, val_ch, test_ch,
  )


def run_h5_pipeline(cfg: PipelineConfig) -> Dict[str, object]:
  print("=== H5 пайплайн ===")
  if not os.path.exists(cfg.h5_path):
    raise FileNotFoundError(f"H5 не найден: {cfg.h5_path}")

  model_cfg = MODEL_CONFIGS[cfg.model_name]
  if model_cfg.arch != "spectrogram":
    print(f"Предупреждение: для H5 рекомендуется spectrogram*, сейчас: {cfg.model_name}")

  data = load_h5_data(cfg.h5_path, cfg.target_meta_class, cfg.h5_max_samples, cfg.seed)
  X_train_cnn = prepare_spectrogram_cnn(data["x_train"])
  X_val_cnn = prepare_spectrogram_cnn(data["x_val"])
  X_test_cnn = prepare_spectrogram_cnn(data["x_test"])
  y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]

  print(f"Train: {X_train_cnn.shape}, target={y_train.mean():.2%}")
  print(f"Val:   {X_val_cnn.shape}, target={y_val.mean():.2%}")
  print(f"Test:  {X_test_cnn.shape}, target={y_test.mean():.2%}")

  # Для предфильтра на H5 используем STD по пространственным осям каждого канала
  train_ch = np.transpose(data["x_train"], (0, 3, 1, 2))
  val_ch = np.transpose(data["x_val"], (0, 3, 1, 2))
  test_ch = np.transpose(data["x_test"], (0, 3, 1, 2))

  return _train_evaluate(
    cfg, X_train_cnn, y_train, X_val_cnn, y_val, X_test_cnn, y_test,
    train_ch, val_ch, test_ch,
  )


def _train_evaluate(
  cfg: PipelineConfig,
  X_train_cnn: np.ndarray,
  y_train: np.ndarray,
  X_val_cnn: np.ndarray,
  y_val: np.ndarray,
  X_test_cnn: np.ndarray,
  y_test: np.ndarray,
  train_channels: np.ndarray,
  val_channels: np.ndarray,
  test_channels: np.ndarray,
) -> Dict[str, object]:
  in_channels = X_train_cnn.shape[1]
  pos_weight = cfg.pos_weight or _compute_pos_weight(y_train)

  train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_train_cnn), torch.from_numpy(y_train.astype(np.float32))),
    batch_size=cfg.batch_size, shuffle=True,
  )
  val_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_val_cnn), torch.from_numpy(y_val.astype(np.float32))),
    batch_size=cfg.batch_size, shuffle=False,
  )
  test_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_test_cnn), torch.from_numpy(y_test.astype(np.float32))),
    batch_size=cfg.batch_size, shuffle=False,
  )

  print(f"\nОбучение CNN ({cfg.model_name}, in_channels={in_channels})...")
  model = build_model(in_channels, cfg.model_name)
  history = train_cnn(model, train_loader, val_loader, cfg, pos_weight=pos_weight)
  print(f"Лучшая val accuracy: {history.best_val_acc:.4f} (эпоха {history.best_epoch})")

  test_probs, test_labels = predict_cnn(model, test_loader, cfg.device)
  cnn_metrics = evaluate_predictions(test_labels, test_probs)
  print("\nCNN на тесте (без предфильтра):")
  for k, v in cnn_metrics.items():
    print(f"  {k}: {v:.4f}")

  results: Dict[str, object] = {
    "history": history,
    "cnn_metrics": cnn_metrics,
    "model": model,
  }

  if not cfg.use_prefilter:
    return results

  print("\nПостроение предфильтра...")
  train_stds = np.array([extract_std_features(ch) for ch in train_channels])
  val_stds = np.array([extract_std_features(ch) for ch in val_channels])
  test_stds = np.array([extract_std_features(ch) for ch in test_channels])

  pca, poly, lr_model = build_prefilter(
    train_stds, y_train,
    k=cfg.prefilter_k,
    degree=cfg.prefilter_degree,
    class_weight_target=cfg.prefilter_class_weight,
  )

  best_th, val_recall = tune_prefilter_threshold(
    val_stds, y_val, pca, poly, lr_model, cfg.prefilter_target_recall
  )
  print(f"Порог предфильтра: {best_th:.3f} | val recall: {val_recall:.4f}")

  test_mask, _ = apply_prefilter(test_stds, pca, poly, lr_model, best_th)
  cascade_metrics = evaluate_cascade(y_test, test_mask, test_probs)
  print("\nКаскад (end-to-end на тесте):")
  for k, v in cascade_metrics.items():
    print(f"  {k}: {v:.4f}")

  results["cascade_metrics"] = cascade_metrics
  results["prefilter_threshold"] = best_th
  return results


def parse_args() -> PipelineConfig:
  parser = argparse.ArgumentParser(description="Каскадный пайплайн классификации рефлектограмм")
  parser.add_argument("--data", choices=["synthetic", "h5"], default="synthetic")
  parser.add_argument("--h5-path", default="test_on_Polygon_I_direct_butt.h5")
  parser.add_argument("--model", default="paper", choices=list(MODEL_CONFIGS))
  parser.add_argument("--wpd-mode", default="stride", choices=["full", "center", "stride"])
  parser.add_argument("--samples", type=int, default=500)
  parser.add_argument("--h5-max-samples", type=int, default=None)
  parser.add_argument("--epochs", type=int, default=15)
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
  parser.add_argument("--no-prefilter", action="store_true")
  parser.add_argument("--list-models", action="store_true")
  args = parser.parse_args()

  if args.list_models:
    print("Доступные конфигурации моделей:")
    for name, mc in MODEL_CONFIGS.items():
      print(f"  {name:18s} arch={mc.arch:12s} blocks={mc.encoder_blocks} ch={mc.base_channels}")
    raise SystemExit(0)

  return PipelineConfig(
    data_source=args.data,
    h5_path=args.h5_path,
    model_name=args.model,
    wpd_mode=args.wpd_mode,
    num_samples=args.samples,
    h5_max_samples=args.h5_max_samples,
    epochs=args.epochs,
    batch_size=args.batch_size,
    lr=args.lr,
    optimizer=args.optimizer,
    use_prefilter=not args.no_prefilter,
  )


def main(cfg: Optional[PipelineConfig] = None) -> Dict[str, object]:
  cfg = cfg or parse_args()
  if cfg.data_source == "h5":
    return run_h5_pipeline(cfg)
  return run_synthetic_pipeline(cfg)


if __name__ == "__main__":
  main()
