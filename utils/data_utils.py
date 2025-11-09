"""
Utilidades para carga y preparación de datos de señales de motor.

Este módulo contiene funciones para:
- Cargar señales de motor desde archivos CSV
- Crear ventanas deslizantes para clasificación y regresión
- Normalizar datos usando estadísticas de entrenamiento
- Crear DataLoaders para PyTorch
"""
import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# ============================================================================
# FUNCIONES DE CARGA DE DATOS
# ============================================================================

def load_motor_signals(data_path, class_mapping, col_indices):
    """
    Carga señales de motor desde archivos CSV organizados por carpetas de clases.
    
    Args:
        data_path: ruta al directorio raíz con las carpetas de clases
        class_mapping: diccionario {nombre_carpeta: label_numérico}
        col_indices: lista de índices de columnas a usar (ej: [0, 1, 2])
    
    Returns:
        signals: array numpy (n_samples, n_features)
        labels: array numpy (n_samples,)
        class_info: diccionario con información de las clases cargadas
    """
    all_signals = []
    all_labels = []
    class_info = {}
    
    for class_name, class_label in class_mapping.items():
        class_path = os.path.join(data_path, class_name)
        
        # Verificar que existe el directorio
        if not os.path.exists(class_path):
            print(f"⚠️  Advertencia: No se encontró el directorio {class_path}")
            continue
        
        # Buscar archivos CSV
        csv_pattern = os.path.join(class_path, '*.csv')
        csv_files = glob.glob(csv_pattern)
        
        if not csv_files:
            print(f"⚠️  Advertencia: No se encontraron archivos CSV en {class_path}")
            continue
        
        # Cargar y concatenar todos los CSVs de esta clase
        class_dataframes = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, header=None)
                class_dataframes.append(df)
            except Exception as e:
                print(f"⚠️  Error al cargar {csv_file}: {e}")
                continue
        
        if class_dataframes:
            # Concatenar todos los DataFrames de esta clase
            class_df = pd.concat(class_dataframes, ignore_index=True)
            
            # Extraer las columnas especificadas
            class_signals = class_df.iloc[:, col_indices].values
            class_labels_array = np.full(class_signals.shape[0], class_label, dtype=np.int64)
            
            all_signals.append(class_signals)
            all_labels.append(class_labels_array)
            
            # Guardar información
            class_info[class_name] = {
                'label': class_label,
                'n_files': len(csv_files),
                'n_samples': class_signals.shape[0]
            }
            
            print(f"✅ Clase '{class_name}' (label={class_label}): {len(csv_files)} archivos, {class_signals.shape[0]} muestras")
    
    # Concatenar todas las señales y etiquetas
    if all_signals:
        signals = np.vstack(all_signals)
        labels = np.concatenate(all_labels)
        
        print(f"\n📊 Total cargado: {signals.shape[0]} muestras, {signals.shape[1]} features")
        print(f"   Distribución de clases: {np.unique(labels, return_counts=True)}")
        
        return signals, labels, class_info
    else:
        raise ValueError("No se pudieron cargar datos. Verifica la ruta y la estructura de directorios.")


# ============================================================================
# FUNCIONES PARA CREAR VENTANAS
# ============================================================================

def create_windows_classification(signals, labels, window_size, stride=1):
    """
    Crea ventanas deslizantes para clasificación de señales.
    
    Args:
        signals: array (n_samples, n_features)
        labels: array (n_samples,) - etiquetas
        window_size: tamaño de la ventana temporal
        stride: paso para ventanas deslizantes (default=1)
    
    Returns:
        X_windows: array (n_windows, window_size, n_features)
        y_windows: array (n_windows,) - etiquetas
    """
    n_samples, n_features = signals.shape
    
    # Calcular número de ventanas posibles
    n_windows = (n_samples - window_size) // stride + 1
    
    X_windows = []
    y_windows = []
    
    for i in range(0, n_samples - window_size + 1, stride):
        # Ventana de señal
        window = signals[i:i + window_size, :]
        
        # Usar la etiqueta del punto final de la ventana
        # (todas las muestras de la ventana deberían tener la misma etiqueta)
        label = labels[i + window_size - 1]
        
        X_windows.append(window)
        y_windows.append(label)
    
    X_windows = np.array(X_windows, dtype=np.float32)
    y_windows = np.array(y_windows, dtype=np.int64)
    
    print(f"   Ventanas creadas: {X_windows.shape[0]}")
    print(f"   Shape: {X_windows.shape}")
    
    return X_windows, y_windows


def create_windows_regression(timeseries, window_size, horizon=1, stride=1):
    """
    Crea ventanas deslizantes para regresión de series temporales.
    
    Args:
        timeseries: array 1D de serie temporal
        window_size: tamaño de la ventana de entrada
        horizon: horizonte de predicción
        stride: paso para ventanas deslizantes (default=1)
    
    Returns:
        X_windows: array (n_windows, window_size, 1)
        y_targets: array (n_windows, horizon)
    """
    n = len(timeseries)
    
    X_windows = []
    y_targets = []
    
    for i in range(0, n - window_size - horizon + 1, stride):
        # Ventana de entrada
        X_window = timeseries[i:i + window_size].reshape(-1, 1)
        
        # Target (horizonte de predicción)
        y_target = timeseries[i + window_size:i + window_size + horizon]
        
        X_windows.append(X_window)
        y_targets.append(y_target)
    
    X_windows = np.array(X_windows, dtype=np.float32)
    y_targets = np.array(y_targets, dtype=np.float32)
    
    print(f"   Ventanas creadas: {X_windows.shape[0]}")
    print(f"   X shape: {X_windows.shape}, y shape: {y_targets.shape}")
    
    return X_windows, y_targets


# ============================================================================
# NORMALIZACIÓN DE DATOS
# ============================================================================

def normalize_data(train_data, val_data=None, test_data=None, method='standard'):
    """
    Normaliza datos usando estadísticas del conjunto de entrenamiento.
    
    IMPORTANTE: La normalización se calcula SOLO con datos de entrenamiento
    para evitar data leakage.
    
    Args:
        train_data: datos de entrenamiento
        val_data: datos de validación (opcional)
        test_data: datos de test (opcional)
        method: 'standard' (z-score) o 'minmax'
    
    Returns:
        train_normalized: datos de entrenamiento normalizados
        val_normalized: datos de validación normalizados (si se proporcionó)
        test_normalized: datos de test normalizados (si se proporcionó)
        stats: diccionario con estadísticas de normalización
    """
    if method == 'standard':
        # StandardScaler: (x - mean) / std
        mean = train_data.mean(axis=0, keepdims=True)
        std = train_data.std(axis=0, keepdims=True)
        
        # Evitar división por cero
        std = np.where(std == 0, 1.0, std)
        
        train_norm = (train_data - mean) / std
        
        stats = {'mean': mean, 'std': std, 'method': 'standard'}
        
        print(f"   Normalización (StandardScaler):")
        print(f"   - Mean: {mean.flatten()}")
        print(f"   - Std: {std.flatten()}")
        
    elif method == 'minmax':
        # MinMaxScaler: (x - min) / (max - min)
        min_val = train_data.min(axis=0, keepdims=True)
        max_val = train_data.max(axis=0, keepdims=True)
        
        # Evitar división por cero
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1.0, range_val)
        
        train_norm = (train_data - min_val) / range_val
        
        stats = {'min': min_val, 'max': max_val, 'method': 'minmax'}
        
        print(f"   Normalización (MinMaxScaler):")
        print(f"   - Min: {min_val.flatten()}")
        print(f"   - Max: {max_val.flatten()}")
    
    else:
        raise ValueError(f"Método de normalización '{method}' no soportado. Use 'standard' o 'minmax'.")
    
    # Normalizar validación y test si se proporcionaron
    results = [train_norm]
    
    if val_data is not None:
        if method == 'standard':
            val_norm = (val_data - mean) / std
        else:
            val_norm = (val_data - min_val) / range_val
        results.append(val_norm)
    
    if test_data is not None:
        if method == 'standard':
            test_norm = (test_data - mean) / std
        else:
            test_norm = (test_data - min_val) / range_val
        results.append(test_norm)
    
    results.append(stats)
    
    return tuple(results)


# ============================================================================
# DATASETS DE PYTORCH
# ============================================================================

class SignalDataset(Dataset):
    """
    Dataset de PyTorch para clasificación de señales.
    """
    def __init__(self, X, y):
        """
        Args:
            X: array numpy (n_samples, window_size, n_features)
            y: array numpy (n_samples,) - etiquetas
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TimeSeriesDataset(Dataset):
    """
    Dataset de PyTorch para regresión de series temporales.
    """
    def __init__(self, X, y):
        """
        Args:
            X: array numpy (n_samples, window_size, n_features)
            y: array numpy (n_samples, horizon)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================================
# FUNCIONES PARA CREAR DATALOADERS
# ============================================================================

def get_data_loaders_classification(data_path, class_mapping, col_indices, window_size,
                                    batch_size=64, train_ratio=0.7, val_ratio=0.15,
                                    random_seed=42, num_workers=2):
    """
    Crea DataLoaders para entrenamiento, validación y test (clasificación).
    
    Args:
        data_path: ruta al directorio con datos
        class_mapping: diccionario {nombre_clase: label}
        col_indices: lista de índices de columnas a usar
        window_size: tamaño de ventana temporal
        batch_size: tamaño de batch
        train_ratio: proporción de datos para entrenamiento
        val_ratio: proporción de datos para validación
        random_seed: semilla aleatoria
        num_workers: número de workers para DataLoader
    
    Returns:
        train_loader, val_loader, test_loader, class_names, data_info
    """
    print("\n" + "="*70)
    print("CARGANDO Y PREPARANDO DATOS PARA CLASIFICACIÓN")
    print("="*70)
    
    # 1. Cargar datos
    print("\n1. Cargando señales de motor...")
    signals, labels, class_info = load_motor_signals(data_path, class_mapping, col_indices)
    
    # 2. Dividir en train/val/test ANTES de crear ventanas
    # Esto asegura que no haya mezcla de segmentos del mismo ciclo entre particiones
    print("\n2. Dividiendo datos (train/val/test)...")
    
    indices = np.arange(len(signals))
    test_ratio = 1.0 - train_ratio - val_ratio
    
    # Primero dividir en train+val y test
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_ratio, random_state=random_seed, stratify=labels
    )
    
    # Luego dividir train+val en train y val
    train_labels = labels[train_val_idx]
    relative_val_ratio = val_ratio / (train_ratio + val_ratio)
    
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=relative_val_ratio, random_state=random_seed, stratify=train_labels
    )
    
    print(f"   Train: {len(train_idx)} muestras ({len(train_idx)/len(signals)*100:.1f}%)")
    print(f"   Val:   {len(val_idx)} muestras ({len(val_idx)/len(signals)*100:.1f}%)")
    print(f"   Test:  {len(test_idx)} muestras ({len(test_idx)/len(signals)*100:.1f}%)")
    
    # 3. Separar los datos
    train_signals = signals[train_idx]
    val_signals = signals[val_idx]
    test_signals = signals[test_idx]
    
    train_labels_arr = labels[train_idx]
    val_labels_arr = labels[val_idx]
    test_labels_arr = labels[test_idx]
    
    # 4. Normalizar (usando SOLO estadísticas de train)
    print("\n3. Normalizando datos...")
    train_norm, val_norm, test_norm, norm_stats = normalize_data(
        train_signals, val_signals, test_signals, method='standard'
    )
    
    # 5. Crear ventanas
    print("\n4. Creando ventanas temporales...")
    print(f"   Tamaño de ventana: {window_size}")
    
    X_train, y_train = create_windows_classification(train_norm, train_labels_arr, window_size)
    X_val, y_val = create_windows_classification(val_norm, val_labels_arr, window_size)
    X_test, y_test = create_windows_classification(test_norm, test_labels_arr, window_size)
    
    # 6. Crear Datasets
    print("\n5. Creando Datasets de PyTorch...")
    train_dataset = SignalDataset(X_train, y_train)
    val_dataset = SignalDataset(X_val, y_val)
    test_dataset = SignalDataset(X_test, y_test)
    
    # 7. Crear DataLoaders
    print("\n6. Creando DataLoaders...")
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    # Información del dataset
    class_names = list(class_mapping.keys())
    data_info = {
        'n_features': X_train.shape[2],
        'n_classes': len(class_mapping),
        'window_size': window_size,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
        'class_info': class_info,
        'normalization_stats': norm_stats
    }
    
    print("\n" + "="*70)
    print("DATOS LISTOS PARA ENTRENAMIENTO")
    print("="*70)
    print(f"Features: {data_info['n_features']}")
    print(f"Clases: {data_info['n_classes']}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print("="*70 + "\n")
    
    return train_loader, val_loader, test_loader, class_names, data_info


def get_data_loaders_regression(timeseries, window_size, horizon=1, batch_size=64,
                                train_ratio=0.7, val_ratio=0.15, num_workers=2):
    """
    Crea DataLoaders para entrenamiento, validación y test (regresión).
    
    Args:
        timeseries: array 1D de serie temporal
        window_size: tamaño de ventana de entrada
        horizon: horizonte de predicción
        batch_size: tamaño de batch
        train_ratio: proporción de datos para entrenamiento
        val_ratio: proporción de datos para validación
        num_workers: número de workers para DataLoader
    
    Returns:
        train_loader, val_loader, test_loader, norm_stats
    """
    print("\n" + "="*70)
    print("PREPARANDO DATOS PARA REGRESIÓN")
    print("="*70)
    
    # 1. Dividir serie temporal
    n = len(timeseries)
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)
    
    train_series = timeseries[:train_end]
    val_series = timeseries[train_end:val_end]
    test_series = timeseries[val_end:]
    
    print(f"\n1. División temporal:")
    print(f"   Train: {len(train_series)} puntos")
    print(f"   Val:   {len(val_series)} puntos")
    print(f"   Test:  {len(test_series)} puntos")
    
    # 2. Normalizar
    print("\n2. Normalizando series temporales...")
    mean = train_series.mean()
    std = train_series.std()
    
    train_norm = (train_series - mean) / std
    val_norm = (val_series - mean) / std
    test_norm = (test_series - mean) / std
    
    norm_stats = {'mean': mean, 'std': std}
    
    print(f"   Mean: {mean:.4f}, Std: {std:.4f}")
    
    # 3. Crear ventanas
    print(f"\n3. Creando ventanas (window_size={window_size}, horizon={horizon})...")
    X_train, y_train = create_windows_regression(train_norm, window_size, horizon)
    X_val, y_val = create_windows_regression(val_norm, window_size, horizon)
    X_test, y_test = create_windows_regression(test_norm, window_size, horizon)
    
    # 4. Crear Datasets y DataLoaders
    print("\n4. Creando DataLoaders...")
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    print("\n" + "="*70)
    print("DATOS DE REGRESIÓN LISTOS")
    print("="*70)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print("="*70 + "\n")
    
    return train_loader, val_loader, test_loader, norm_stats


# ============================================================================
# GENERACIÓN DE SERIES TEMPORALES SINTÉTICAS
# ============================================================================

def generate_synthetic_timeseries(T=5000, seed=42):
    """
    Genera una serie temporal sintética para demostración.
    
    Args:
        T: número de puntos temporales
        seed: semilla aleatoria
    
    Returns:
        t: array de tiempo
        y: serie temporal
    """
    np.random.seed(seed)
    
    t = np.arange(T, dtype=np.float32)
    
    # Serie temporal con componentes de tendencia, estacionalidad y ruido
    trend = 0.0001 * t  # Tendencia ligera
    seasonal = 0.75 * np.sin(2 * np.pi * t / 60)  # Componente estacional
    noise = 0.01 * np.random.randn(T)  # Ruido blanco
    
    y = trend + seasonal + noise
    
    print(f"\n📊 Serie temporal generada:")
    print(f"   Puntos: {T}")
    print(f"   Rango: [{y.min():.4f}, {y.max():.4f}]")
    print(f"   Media: {y.mean():.4f}, Std: {y.std():.4f}")
    
    return t, y


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("TESTING DATA_UTILS")
    print("="*70)
    
    # Test de generación de serie temporal
    print("\nTest 1: Generación de serie temporal sintética")
    t, y = generate_synthetic_timeseries(T=1000, seed=42)
    
    # Test de creación de ventanas para regresión
    print("\nTest 2: Creación de ventanas para regresión")
    X, y_target = create_windows_regression(y, window_size=32, horizon=1)
    print(f"X shape: {X.shape}, y shape: {y_target.shape}")
    
    # Test de normalización
    print("\nTest 3: Normalización de datos")
    train_data = np.random.randn(100, 3)
    val_data = np.random.randn(20, 3)
    test_data = np.random.randn(20, 3)
    
    train_norm, val_norm, test_norm, stats = normalize_data(train_data, val_data, test_data)
    print(f"Train normalized: {train_norm.shape}")
    print(f"Val normalized: {val_norm.shape}")
    print(f"Test normalized: {test_norm.shape}")
    
    print("\n" + "="*70)
    print("TESTS COMPLETADOS")
    print("="*70)
