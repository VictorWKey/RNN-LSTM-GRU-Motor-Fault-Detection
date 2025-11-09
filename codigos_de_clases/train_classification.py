"""
Script para entrenamiento de modelos de clasificación de señales de motor.

Este script implementa el entrenamiento de RNN, LSTM y GRU para clasificación
de señales de motor con múltiples clases de fallas.
"""
import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from models import LSTMClassifier, GRUClassifier, BiRNNClassifier
from utils import (
    make_windows_classification,
    SignalClassificationDataset,
    normalize_data,
    train_model,
    compute_classification_metrics,
    plot_training_history,
    plot_confusion_matrix
)


def cargar_datos_motor(path, nombres_cls, col_index):
    """
    Carga datos de señales de motor desde archivos CSV.
    
    Args:
        path: ruta al directorio con los datos
        nombres_cls: diccionario {nombre_carpeta: label}
        col_index: índices de columnas a usar
    
    Returns:
        Signals: array de señales (n_samples, n_features)
        Labels: array de etiquetas
    """
    lst_signals = []
    lst_labels = []
    
    for fname_cls, label_cls in nombres_cls.items():
        path_cls = os.path.join(path, fname_cls)
        pattern_files = os.path.join(path_cls, '*.csv')
        files_csv = glob.glob(pattern_files)
        
        if not files_csv:
            print(f'Advertencia: No se encontraron archivos en {path_cls}')
            continue
        
        df_cls = pd.concat([pd.read_csv(f, header=None) for f in files_csv], ignore_index=True)
        signals = df_cls[col_index].values
        labels = np.full(signals.shape[0], label_cls, dtype=np.int64)
        
        lst_signals.append(signals)
        lst_labels.append(labels)
        
        print(f'Clase {fname_cls}: {signals.shape[0]} muestras.')
    
    Signals = np.concatenate(lst_signals, axis=0)
    Labels = np.concatenate(lst_labels, axis=0)
    
    return Signals, Labels


def main():
    # Configuración
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Parámetros
    WINDOW_SIZE = 32
    BATCH_SIZE = 64
    EPOCHS = 70
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    CLIP_NORM = 1.0
    RANDOM_SEED = 42
    
    # Establecer semillas para reproducibilidad
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Definir clases (ajustar según tus datos)
    # Ejemplo: 13 clases (1 sano + 4 niveles de falla × 3 fases)
    name_cls = {
        'SC_HLT': 0,      # Sano
        'SC_A0_B0_C1': 1,
        'SC_A0_B0_C2': 2,
        'SC_A0_B0_C3': 3,
        'SC_A0_B0_C4': 4,
        # Agregar más clases según sea necesario
    }
    
    col_index = [0, 1, 2]  # Índices de columnas a usar
    path_db = 'Dataset'     # Ajustar ruta
    
    # Verificar si existe el directorio
    if not os.path.exists(path_db):
        print(f"Error: El directorio '{path_db}' no existe.")
        print("Por favor, ajusta la ruta 'path_db' en el script.")
        return
    
    # Cargar datos
    print("\n=== Cargando datos ===")
    Signals, Labels = cargar_datos_motor(path_db, name_cls, col_index)
    print(f'Total muestras: {Signals.shape[0]}, Features: {Signals.shape[1]}')
    
    # Dividir en train/val/test
    print("\n=== Dividiendo datos ===")
    id_train, id_temp, lb_train, lb_temp = train_test_split(
        np.arange(Signals.shape[0]), Labels,
        test_size=0.3, random_state=RANDOM_SEED, stratify=Labels
    )
    
    id_val, id_test, lb_val, lb_test = train_test_split(
        id_temp, lb_temp,
        test_size=0.5, random_state=RANDOM_SEED, stratify=lb_temp
    )
    
    print(f'Train: {len(id_train)}, Val: {len(id_val)}, Test: {len(id_test)}')
    
    # Normalizar datos
    print("\n=== Normalizando datos ===")
    train_signals = Signals[id_train]
    val_signals = Signals[id_val]
    test_signals = Signals[id_test]
    
    train_norm, val_norm, test_norm, mu, sg = normalize_data(
        train_signals, val_signals, test_signals
    )
    print(f'Media: {mu.flatten()}')
    print(f'Desviación estándar: {sg.flatten()}')
    
    # Crear ventanas
    print("\n=== Creando ventanas ===")
    Xtr, ytr = make_windows_classification(train_norm, lb_train, WINDOW_SIZE)
    Xvl, yvl = make_windows_classification(val_norm, lb_val, WINDOW_SIZE)
    Xte, yte = make_windows_classification(test_norm, lb_test, WINDOW_SIZE)
    
    print(f'Xtr: {Xtr.shape}, ytr: {ytr.shape}')
    print(f'Xvl: {Xvl.shape}, yvl: {yvl.shape}')
    print(f'Xte: {Xte.shape}, yte: {yte.shape}')
    
    # Crear DataLoaders
    train_loader = DataLoader(
        SignalClassificationDataset(Xtr, ytr),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        SignalClassificationDataset(Xvl, yvl),
        batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        SignalClassificationDataset(Xte, yte),
        batch_size=BATCH_SIZE, shuffle=False
    )
    
    # Parámetros del modelo
    n_features = Xtr.shape[2]
    n_classes = len(name_cls)
    hidden_size = 64
    num_layers = 2
    
    print(f'\n=== Configuración del modelo ===')
    print(f'Features de entrada: {n_features}')
    print(f'Número de clases: {n_classes}')
    print(f'Tamaño oculto: {hidden_size}')
    print(f'Número de capas: {num_layers}')
    
    # Entrenar diferentes modelos
    models_to_train = {
        'LSTM': LSTMClassifier(n_features, hidden_size, num_layers, n_classes),
        'LSTM-Bidirectional': LSTMClassifier(n_features, hidden_size, num_layers, n_classes, bidirectional=True),
        'GRU': GRUClassifier(n_features, hidden_size, num_layers, n_classes),
        'BiRNN': BiRNNClassifier(n_features, hidden_size, num_layers, n_classes)
    }
    
    results = {}
    
    for model_name, model in models_to_train.items():
        print(f'\n{"="*60}')
        print(f'Entrenando: {model_name}')
        print(f'{"="*60}')
        
        model = model.to(device)
        
        # Configurar optimizador y loss
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        
        # Entrenar
        history, best_state = train_model(
            model, train_loader, val_loader,
            criterion, optimizer, EPOCHS, device,
            clip_norm=CLIP_NORM, verbose=True
        )
        
        # Cargar mejor modelo
        model.load_state_dict(best_state)
        
        # Evaluar en test
        print(f'\n--- Evaluación en conjunto de prueba ---')
        class_names = list(name_cls.keys())
        metrics = compute_classification_metrics(model, test_loader, device, class_names)
        
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Test F1-score (macro): {metrics['f1_score']:.4f}")
        print(f"\nReporte de Clasificación:\n{metrics['classification_report']}")
        
        # Guardar resultados
        results[model_name] = {
            'model': model,
            'history': history,
            'metrics': metrics
        }
        
        # Visualizar historial
        plot_training_history(history)
        
        # Visualizar matriz de confusión
        plot_confusion_matrix(metrics['confusion_matrix'], class_names)
        
        # Guardar modelo
        torch.save(best_state, f'{model_name.lower().replace("-", "_")}_best.pth')
        print(f'Modelo guardado: {model_name.lower().replace("-", "_")}_best.pth')
    
    # Resumen comparativo
    print(f'\n{"="*60}')
    print('RESUMEN COMPARATIVO')
    print(f'{"="*60}')
    print(f'{"Modelo":<25} {"Accuracy":<12} {"F1-Score":<12}')
    print('-' * 60)
    for model_name, result in results.items():
        acc = result['metrics']['accuracy']
        f1 = result['metrics']['f1_score']
        print(f'{model_name:<25} {acc:<12.4f} {f1:<12.4f}')


if __name__ == '__main__':
    main()
