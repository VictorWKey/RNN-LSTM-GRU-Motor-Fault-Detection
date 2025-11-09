"""
Utilidades para preparación de datos y creación de datasets.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


def make_windows_regression(y, window_size, horizon=1):
    """
    Crea ventanas deslizantes para regresión de series temporales.
    
    Args:
        y: array de series temporales (1D)
        window_size: tamaño de la ventana de entrada
        horizon: horizonte de predicción
    
    Returns:
        X: (n_samples, window_size, 1)
        Y: (n_samples, horizon)
    """
    X, Y = [], []
    for i in range(len(y) - window_size - horizon + 1):
        X.append(y[i:i + window_size].reshape(-1, 1))
        Y.append(y[i + window_size:i + window_size + horizon])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def make_windows_classification(signals, labels, window_size):
    """
    Crea ventanas deslizantes para clasificación de señales.
    
    Args:
        signals: array de señales (n_samples, n_features)
        labels: etiquetas correspondientes
        window_size: tamaño de la ventana
    
    Returns:
        X: (n_windows, window_size, n_features)
        Y: (n_windows,) - etiquetas
    """
    X, Y = [], []
    for i in range(signals.shape[0] - window_size + 1):
        X_window = signals[i:i + window_size]
        Y_label = int(labels[i + window_size - 1])  # Usar la última etiqueta
        X.append(X_window)
        Y.append(Y_label)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int64)


class TimeSeriesDataset(Dataset):
    """Dataset para regresión de series temporales."""
    
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class SignalClassificationDataset(Dataset):
    """Dataset para clasificación de señales."""
    
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.int64)
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def normalize_data(train_data, val_data, test_data):
    """
    Normaliza los datos usando estadísticas del conjunto de entrenamiento.
    
    Args:
        train_data: datos de entrenamiento
        val_data: datos de validación
        test_data: datos de prueba
    
    Returns:
        train_normalized, val_normalized, test_normalized, mean, std
    """
    mean = train_data.mean(axis=0, keepdims=True)
    std = train_data.std(axis=0, keepdims=True)
    
    # Evitar división por cero
    std = np.where(std == 0, 1.0, std)
    
    train_norm = (train_data - mean) / std
    val_norm = (val_data - mean) / std
    test_norm = (test_data - mean) / std
    
    return train_norm, val_norm, test_norm, mean, std
