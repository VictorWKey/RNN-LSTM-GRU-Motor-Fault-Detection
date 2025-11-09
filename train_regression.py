"""
Script principal para entrenamiento de modelos de REGRESIÓN de series temporales.

Este script entrena RNN, LSTM y GRU (modelos base + variantes) para predecir
series temporales sintéticas.

Uso:
    python train_regression.py
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import time
from datetime import datetime

# Importar configuración
from config import (
    set_seed, RANDOM_SEED, DATA_CONFIG, TRAIN_CONFIG,
    MODEL_CONFIG, PATHS, HYPOTHESES
)

# Importar modelos
from models import (
    RNNRegressor, LSTMRegressor, GRURegressor
)

# Importar utilidades
from utils import (
    generate_synthetic_timeseries,
    get_data_loaders_regression,
    train_model, compute_regression_metrics,
    plot_training_history,
    plot_predictions_vs_actual,
    plot_time_series_prediction,
    plot_model_comparison,
    save_results_table
)
from utils.training_utils import count_parameters


def main():
    print("="*80)
    print("PRÁCTICA 02 - ENTRENAMIENTO DE MODELOS RECURRENTES PARA REGRESIÓN")
    print("="*80)
    print(f"Fecha y hora de inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # ========================================================================
    # 1. CONFIGURACIÓN INICIAL
    # ========================================================================
    
    # Establecer semillas para reproducibilidad
    set_seed(RANDOM_SEED)
    print(f"✅ Semilla aleatoria establecida: {RANDOM_SEED}\n")
    
    # Detectar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Dispositivo detectado: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memoria disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()
    
    # Crear directorios necesarios
    for path in PATHS.values():
        os.makedirs(path, exist_ok=True)
    
    # ========================================================================
    # 2. GENERAR Y PREPARAR DATOS
    # ========================================================================
    
    print("="*80)
    print("FASE 1: GENERACIÓN Y PREPARACIÓN DE DATOS")
    print("="*80)
    
    # Obtener configuración de datos
    reg_config = DATA_CONFIG['regression']
    
    # Generar serie temporal sintética
    print("\n📊 Generando serie temporal sintética...")
    t, y = generate_synthetic_timeseries(T=reg_config['T'], seed=RANDOM_SEED)
    
    # Crear DataLoaders
    train_loader, val_loader, test_loader, norm_stats = get_data_loaders_regression(
        timeseries=y,
        window_size=reg_config['window_size'],
        horizon=reg_config['horizon'],
        batch_size=TRAIN_CONFIG['batch_size'],
        train_ratio=reg_config['train_split'],
        val_ratio=reg_config['val_split'],
        num_workers=2
    )
    
    input_size = 1  # Serie univariada
    output_size = reg_config['horizon']
    
    print(f"\n📊 Dataset listo:")
    print(f"   - Input size: {input_size}")
    print(f"   - Output size: {output_size}")
    print(f"   - Window size: {reg_config['window_size']}")
    print()
    
    # ========================================================================
    # 3. DEFINIR MODELOS A ENTRENAR
    # ========================================================================
    
    print("="*80)
    print("FASE 2: DEFINICIÓN DE MODELOS")
    print("="*80 + "\n")
    
    models_to_train = {}
    
    # ------------------------------------------------------------------------
    # RNN MODELS
    # ------------------------------------------------------------------------
    
    # RNN Base
    rnn_base_config = MODEL_CONFIG['rnn']['base']
    models_to_train['RNN_Base'] = {
        'model': RNNRegressor(
            input_size=input_size,
            hidden_size=rnn_base_config['hidden_size'],
            num_layers=rnn_base_config['num_layers'],
            output_size=output_size,
            nonlinearity=rnn_base_config['nonlinearity'],
            bidirectional=rnn_base_config['bidirectional'],
            dropout=rnn_base_config['dropout']
        ),
        'hypothesis': HYPOTHESES['rnn']['base']
    }
    
    # RNN Variante 1: Profunda (2 capas)
    rnn_deep_config = MODEL_CONFIG['rnn']['variants']['deep']
    models_to_train['RNN_Deep'] = {
        'model': RNNRegressor(
            input_size=input_size,
            hidden_size=rnn_deep_config['hidden_size'],
            num_layers=rnn_deep_config['num_layers'],
            output_size=output_size,
            nonlinearity=rnn_deep_config['nonlinearity'],
            bidirectional=rnn_deep_config['bidirectional'],
            dropout=rnn_deep_config['dropout']
        ),
        'hypothesis': HYPOTHESES['rnn']['deep']
    }
    
    # RNN Variante 2: Con ReLU
    rnn_relu_config = MODEL_CONFIG['rnn']['variants']['relu']
    models_to_train['RNN_ReLU'] = {
        'model': RNNRegressor(
            input_size=input_size,
            hidden_size=rnn_relu_config['hidden_size'],
            num_layers=rnn_relu_config['num_layers'],
            output_size=output_size,
            nonlinearity=rnn_relu_config['nonlinearity'],
            bidirectional=rnn_relu_config['bidirectional'],
            dropout=rnn_relu_config['dropout']
        ),
        'hypothesis': HYPOTHESES['rnn']['relu']
    }
    
    # ------------------------------------------------------------------------
    # LSTM MODELS
    # ------------------------------------------------------------------------
    
    # LSTM Base
    lstm_base_config = MODEL_CONFIG['lstm']['base']
    models_to_train['LSTM_Base'] = {
        'model': LSTMRegressor(
            input_size=input_size,
            hidden_size=lstm_base_config['hidden_size'],
            num_layers=lstm_base_config['num_layers'],
            output_size=output_size,
            bidirectional=lstm_base_config['bidirectional'],
            dropout=lstm_base_config['dropout']
        ),
        'hypothesis': HYPOTHESES['lstm']['base']
    }
    
    # LSTM Variante 1: Bidireccional
    lstm_bi_config = MODEL_CONFIG['lstm']['variants']['bidirectional']
    models_to_train['LSTM_Bidirectional'] = {
        'model': LSTMRegressor(
            input_size=input_size,
            hidden_size=lstm_bi_config['hidden_size'],
            num_layers=lstm_bi_config['num_layers'],
            output_size=output_size,
            bidirectional=lstm_bi_config['bidirectional'],
            dropout=lstm_bi_config['dropout']
        ),
        'hypothesis': HYPOTHESES['lstm']['bidirectional']
    }
    
    # LSTM Variante 2: Apilada con dropout
    lstm_stacked_config = MODEL_CONFIG['lstm']['variants']['stacked']
    models_to_train['LSTM_Stacked'] = {
        'model': LSTMRegressor(
            input_size=input_size,
            hidden_size=lstm_stacked_config['hidden_size'],
            num_layers=lstm_stacked_config['num_layers'],
            output_size=output_size,
            bidirectional=lstm_stacked_config['bidirectional'],
            dropout=lstm_stacked_config['dropout']
        ),
        'hypothesis': HYPOTHESES['lstm']['stacked']
    }
    
    # ------------------------------------------------------------------------
    # GRU MODELS
    # ------------------------------------------------------------------------
    
    # GRU Base
    gru_base_config = MODEL_CONFIG['gru']['base']
    models_to_train['GRU_Base'] = {
        'model': GRURegressor(
            input_size=input_size,
            hidden_size=gru_base_config['hidden_size'],
            num_layers=gru_base_config['num_layers'],
            output_size=output_size,
            bidirectional=gru_base_config['bidirectional'],
            dropout=gru_base_config['dropout']
        ),
        'hypothesis': HYPOTHESES['gru']['base']
    }
    
    # GRU Variante 1: Bidireccional
    gru_bi_config = MODEL_CONFIG['gru']['variants']['bidirectional']
    models_to_train['GRU_Bidirectional'] = {
        'model': GRURegressor(
            input_size=input_size,
            hidden_size=gru_bi_config['hidden_size'],
            num_layers=gru_bi_config['num_layers'],
            output_size=output_size,
            bidirectional=gru_bi_config['bidirectional'],
            dropout=gru_bi_config['dropout']
        ),
        'hypothesis': HYPOTHESES['gru']['bidirectional']
    }
    
    # GRU Variante 2: Apilada
    gru_stacked_config = MODEL_CONFIG['gru']['variants']['stacked']
    models_to_train['GRU_Stacked'] = {
        'model': GRURegressor(
            input_size=input_size,
            hidden_size=gru_stacked_config['hidden_size'],
            num_layers=gru_stacked_config['num_layers'],
            output_size=output_size,
            bidirectional=gru_stacked_config['bidirectional'],
            dropout=gru_stacked_config['dropout']
        ),
        'hypothesis': HYPOTHESES['gru']['stacked']
    }
    
    # Mostrar resumen de modelos
    print("📋 Modelos a entrenar:")
    for idx, (model_name, model_info) in enumerate(models_to_train.items(), 1):
        total_params, trainable_params = count_parameters(model_info['model'])
        print(f"\n{idx}. {model_name}")
        print(f"   Parámetros: {total_params:,} ({total_params/1e6:.4f}M)")
        print(f"   Hipótesis: {model_info['hypothesis']}")
    print()
    
    # ========================================================================
    # 4. ENTRENAR MODELOS
    # ========================================================================
    
    print("="*80)
    print("FASE 3: ENTRENAMIENTO DE MODELOS")
    print("="*80 + "\n")
    
    # Criterio y configuración de entrenamiento
    criterion = nn.MSELoss()
    
    # Diccionario para almacenar resultados
    all_results = {}
    
    # Entrenar cada modelo
    for model_name, model_info in models_to_train.items():
        print("\n" + "="*80)
        print(f"ENTRENANDO: {model_name}")
        print("="*80)
        print(f"Hipótesis: {model_info['hypothesis']}")
        print("-"*80 + "\n")
        
        # Mover modelo al dispositivo
        model = model_info['model'].to(device)
        
        # Contar parámetros
        total_params, _ = count_parameters(model)
        params_millions = total_params / 1e6
        
        # Crear optimizador
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=TRAIN_CONFIG['learning_rate'],
            weight_decay=TRAIN_CONFIG['weight_decay']
        )
        
        # Ruta para guardar checkpoint
        checkpoint_path = os.path.join(PATHS['checkpoints'], f'{model_name}_regression_best.pth')
        
        # Entrenar
        history, best_state, training_time = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            epochs=TRAIN_CONFIG['epochs'],
            device=device,
            clip_norm=TRAIN_CONFIG['clip_norm'],
            task='regression',
            save_path=checkpoint_path,
            patience=TRAIN_CONFIG.get('patience', None),
            verbose=True
        )
        
        # Cargar mejor modelo
        model.load_state_dict(best_state)
        
        # Evaluar en conjunto de test
        print(f"\n{'='*80}")
        print(f"EVALUACIÓN EN TEST: {model_name}")
        print(f"{'='*80}")
        
        test_metrics = compute_regression_metrics(model, test_loader, device)
        
        print(f"\n📊 Resultados en Test:")
        print(f"   MSE:  {test_metrics['mse']:.6f}")
        print(f"   RMSE: {test_metrics['rmse']:.6f}")
        print(f"   MAE:  {test_metrics['mae']:.6f}")
        print(f"   R²:   {test_metrics['r2']:.6f}")
        
        # Guardar resultados
        all_results[model_name] = {
            'model': model,
            'history': history,
            'test_metrics': test_metrics,
            'mse': test_metrics['mse'],
            'rmse': test_metrics['rmse'],
            'mae': test_metrics['mae'],
            'r2': test_metrics['r2'],
            'params_millions': params_millions,
            'training_time': training_time,
            'time_per_epoch': training_time / TRAIN_CONFIG['epochs']
        }
        
        # Visualizar historial de entrenamiento
        fig_path = os.path.join(PATHS['figures'], f'{model_name}_regression_history.png')
        plot_training_history(history, task='regression', save_path=fig_path, model_name=model_name)
        
        # Visualizar predicciones vs reales
        pred_path = os.path.join(PATHS['figures'], f'{model_name}_regression_predictions.png')
        plot_predictions_vs_actual(
            test_metrics['targets'],
            test_metrics['predictions'],
            save_path=pred_path,
            model_name=model_name
        )
        
        # Visualizar serie temporal
        ts_path = os.path.join(PATHS['figures'], f'{model_name}_regression_timeseries.png')
        plot_time_series_prediction(
            test_metrics['targets'],
            test_metrics['predictions'],
            n_samples=200,
            save_path=ts_path,
            model_name=model_name
        )
    
    # ========================================================================
    # 5. COMPARAR MODELOS Y GUARDAR RESULTADOS
    # ========================================================================
    
    print("\n" + "="*80)
    print("FASE 4: COMPARACIÓN DE MODELOS Y RESULTADOS FINALES")
    print("="*80 + "\n")
    
    # Gráfica comparativa
    comp_path = os.path.join(PATHS['figures'], 'regression_model_comparison.png')
    plot_model_comparison(all_results, task='regression', save_path=comp_path)
    
    # Tabla de resultados
    table_path = os.path.join(PATHS['results'], 'regression_results')
    results_df = save_results_table(all_results, table_path, task='regression')
    
    # Identificar mejor modelo (menor RMSE)
    best_model_name = min(all_results.keys(), key=lambda k: all_results[k]['rmse'])
    best_rmse = all_results[best_model_name]['rmse']
    best_r2 = all_results[best_model_name]['r2']
    
    print(f"\n🏆 MEJOR MODELO: {best_model_name}")
    print(f"   RMSE: {best_rmse:.6f}")
    print(f"   R²: {best_r2:.6f}")
    
    # ========================================================================
    # 6. RESUMEN FINAL
    # ========================================================================
    
    print("\n" + "="*80)
    print("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("="*80)
    print(f"\n📁 Resultados guardados en:")
    print(f"   - Checkpoints: {PATHS['checkpoints']}")
    print(f"   - Figuras: {PATHS['figures']}")
    print(f"   - Resultados: {PATHS['results']}")
    print(f"\n⏰ Fecha y hora de finalización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Entrenamiento interrumpido por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
