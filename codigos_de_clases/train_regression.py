"""
Script para entrenamiento de modelos de regresión de series temporales.

Este script implementa el entrenamiento de RNN, LSTM y GRU para predicción
de series temporales (regresión).
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import RNNSimple, LSTMRegressor, GRURegressor
from utils import (
    make_windows_regression,
    TimeSeriesDataset,
    train_model,
    plot_training_history,
    plot_predictions_vs_actual,
    plot_time_series_prediction
)


def generar_serie_temporal(T=5000, seed=42):
    """
    Genera una serie temporal sintética para demostración.
    
    Args:
        T: número de puntos
        seed: semilla aleatoria
    
    Returns:
        t: array de tiempo
        y: serie temporal
    """
    np.random.seed(seed)
    t = np.arange(T, dtype=np.float32)
    y = 0.75 * np.sin(2 * np.pi * t / 60) + 0.01 * np.random.randn(T)
    return t, y


def main():
    # Configuración
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Parámetros
    T = 5000
    WINDOW_SIZE = 32
    HORIZON = 1
    BATCH_SIZE = 64
    EPOCHS = 70
    LEARNING_RATE = 0.001
    CLIP_NORM = 1.0
    RANDOM_SEED = 42
    
    # Establecer semillas
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Generar datos
    print("\n=== Generando serie temporal ===")
    t, y = generar_serie_temporal(T, RANDOM_SEED)
    print(f'Total de puntos: {T}')
    
    # Dividir datos
    print("\n=== Dividiendo datos ===")
    n_train = int(0.7 * T)
    n_val = int(0.2 * T)
    n_test = T - n_train - n_val
    
    y_train = y[:n_train]
    y_val = y[n_train:n_train + n_val]
    y_test = y[n_train + n_val:]
    
    print(f'Train: {n_train}, Val: {n_val}, Test: {n_test}')
    
    # Normalizar
    print("\n=== Normalizando datos ===")
    y_mu = y_train.mean(keepdims=True)
    y_sd = y_train.std(keepdims=True)
    
    y_train_z = (y_train - y_mu) / y_sd
    y_val_z = (y_val - y_mu) / y_sd
    y_test_z = (y_test - y_mu) / y_sd
    
    print(f'Media: {y_mu[0]:.4f}, Desv. estándar: {y_sd[0]:.4f}')
    
    # Crear ventanas
    print("\n=== Creando ventanas ===")
    Xtr, Ytr = make_windows_regression(y_train_z, WINDOW_SIZE, HORIZON)
    Xvl, Yvl = make_windows_regression(y_val_z, WINDOW_SIZE, HORIZON)
    Xts, Yts = make_windows_regression(y_test_z, WINDOW_SIZE, HORIZON)
    
    print(f'Xtr: {Xtr.shape}, Ytr: {Ytr.shape}')
    print(f'Xvl: {Xvl.shape}, Yvl: {Yvl.shape}')
    print(f'Xts: {Xts.shape}, Yts: {Yts.shape}')
    
    # Crear DataLoaders
    train_loader = DataLoader(
        TimeSeriesDataset(Xtr, Ytr),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TimeSeriesDataset(Xvl, Yvl),
        batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        TimeSeriesDataset(Xts, Yts),
        batch_size=BATCH_SIZE, shuffle=False
    )
    
    # Parámetros del modelo
    in_dim = 1
    hidden_size = 128
    num_layers = 2
    out_dim = HORIZON
    
    print(f'\n=== Configuración del modelo ===')
    print(f'Dimensión entrada: {in_dim}')
    print(f'Tamaño oculto: {hidden_size}')
    print(f'Número de capas: {num_layers}')
    print(f'Horizonte de predicción: {out_dim}')
    
    # Entrenar diferentes modelos
    models_to_train = {
        'RNN-tanh': RNNSimple(in_dim, hidden_size, num_layers, out_dim, 'tanh'),
        'RNN-relu': RNNSimple(in_dim, hidden_size, num_layers, out_dim, 'relu'),
        'LSTM': LSTMRegressor(in_dim, hidden_size, num_layers, out_dim),
        'GRU': GRURegressor(in_dim, hidden_size, num_layers, out_dim)
    }
    
    results = {}
    
    for model_name, model in models_to_train.items():
        print(f'\n{"="*60}')
        print(f'Entrenando: {model_name}')
        print(f'{"="*60}')
        
        model = model.to(device)
        
        # Configurar optimizador y loss
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
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
        model.eval()
        test_predictions = []
        test_targets = []
        
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                pred = model(xb)
                test_predictions.append(pred.cpu().numpy())
                test_targets.append(yb.numpy())
        
        y_pred = np.concatenate(test_predictions, axis=0)
        y_true = np.concatenate(test_targets, axis=0)
        
        # Calcular MSE y RMSE
        mse = np.mean((y_pred - y_true) ** 2)
        rmse = np.sqrt(mse)
        
        # Calcular R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        print(f"Test MSE: {mse:.6f}")
        print(f"Test RMSE: {rmse:.6f}")
        print(f"Test R²: {r2:.6f}")
        
        # Guardar resultados
        results[model_name] = {
            'model': model,
            'history': history,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred,
            'targets': y_true
        }
        
        # Visualizar historial
        plot_training_history(history)
        
        # Visualizar predicciones
        plot_predictions_vs_actual(y_true.flatten(), y_pred.flatten())
        plot_time_series_prediction(y_true.flatten(), y_pred.flatten(), n_samples=200)
        
        # Guardar modelo
        torch.save(best_state, f'{model_name.lower().replace("-", "_")}_regressor_best.pth')
        print(f'Modelo guardado: {model_name.lower().replace("-", "_")}_regressor_best.pth')
    
    # Resumen comparativo
    print(f'\n{"="*70}')
    print('RESUMEN COMPARATIVO')
    print(f'{"="*70}')
    print(f'{"Modelo":<20} {"MSE":<15} {"RMSE":<15} {"R²":<15}')
    print('-' * 70)
    for model_name, result in results.items():
        print(f'{model_name:<20} {result["mse"]:<15.6f} {result["rmse"]:<15.6f} {result["r2"]:<15.6f}')


if __name__ == '__main__':
    main()
