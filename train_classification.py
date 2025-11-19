"""
Script principal para entrenamiento de modelos de CLASIFICACIÓN de señales de motor.

Este script entrena RNN, LSTM y GRU (modelos base + variantes) para clasificar
señales de motor en 13 clases (1 sano + 4 niveles de falla × 3 fases).

Uso:
    python train_classification.py
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
    SPLIT_CONFIG, MODEL_CONFIG, PATHS, HYPOTHESES
)

# Importar modelos
from models import (
    RNNClassifier, LSTMClassifier, GRUClassifier
)

# Importar utilidades
from utils import (
    get_data_loaders_classification,
    train_model, compute_classification_metrics,
    plot_training_history,
    plot_confusion_matrix, plot_model_comparison,
    save_results_table
)
from utils.training_utils import count_parameters


def main():
    print("="*80)
    print("PRÁCTICA 02 - ENTRENAMIENTO DE MODELOS RECURRENTES PARA CLASIFICACIÓN")
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
    # 2. CARGAR Y PREPARAR DATOS
    # ========================================================================
    
    print("="*80)
    print("FASE 1: CARGA Y PREPARACIÓN DE DATOS")
    print("="*80)
    
    # Obtener configuración de datos
    data_config = DATA_CONFIG['classification']
    
    # Crear DataLoaders
    train_loader, val_loader, test_loader, class_names, data_info = get_data_loaders_classification(
        data_path=data_config['path'],
        class_mapping=data_config['classes'],
        col_indices=data_config['col_index'],
        window_size=data_config['window_size'],
        batch_size=TRAIN_CONFIG['batch_size'],
        train_ratio=SPLIT_CONFIG['train_ratio'],
        val_ratio=SPLIT_CONFIG['val_ratio'],
        random_seed=RANDOM_SEED,
        num_workers=2
    )
    
    # Información del dataset
    n_features = data_info['n_features']
    n_classes = data_info['n_classes']
    
    print(f"📊 Dataset listo:")
    print(f"   - Features: {n_features}")
    print(f"   - Clases: {n_classes}")
    print(f"   - Clases: {class_names}")
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
        'model': RNNClassifier(
            input_size=n_features,
            hidden_size=rnn_base_config['hidden_size'],
            num_layers=rnn_base_config['num_layers'],
            num_classes=n_classes,
            nonlinearity=rnn_base_config['nonlinearity'],
            bidirectional=rnn_base_config['bidirectional'],
            dropout=rnn_base_config['dropout']
        ),
        'hypothesis': HYPOTHESES['rnn']['base']
    }
    
    # RNN Variante 1: Profunda (2 capas)
    rnn_deep_config = MODEL_CONFIG['rnn']['variants']['deep']
    models_to_train['RNN_Deep'] = {
        'model': RNNClassifier(
            input_size=n_features,
            hidden_size=rnn_deep_config['hidden_size'],
            num_layers=rnn_deep_config['num_layers'],
            num_classes=n_classes,
            nonlinearity=rnn_deep_config['nonlinearity'],
            bidirectional=rnn_deep_config['bidirectional'],
            dropout=rnn_deep_config['dropout']
        ),
        'hypothesis': HYPOTHESES['rnn']['deep']
    }
    
    # RNN Variante 2: Con ReLU
    rnn_relu_config = MODEL_CONFIG['rnn']['variants']['relu']
    models_to_train['RNN_ReLU'] = {
        'model': RNNClassifier(
            input_size=n_features,
            hidden_size=rnn_relu_config['hidden_size'],
            num_layers=rnn_relu_config['num_layers'],
            num_classes=n_classes,
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
        'model': LSTMClassifier(
            input_size=n_features,
            hidden_size=lstm_base_config['hidden_size'],
            num_layers=lstm_base_config['num_layers'],
            num_classes=n_classes,
            bidirectional=lstm_base_config['bidirectional'],
            dropout=lstm_base_config['dropout']
        ),
        'hypothesis': HYPOTHESES['lstm']['base']
    }
    
    # LSTM Variante 1: Bidireccional
    lstm_bi_config = MODEL_CONFIG['lstm']['variants']['bidirectional']
    models_to_train['LSTM_Bidirectional'] = {
        'model': LSTMClassifier(
            input_size=n_features,
            hidden_size=lstm_bi_config['hidden_size'],
            num_layers=lstm_bi_config['num_layers'],
            num_classes=n_classes,
            bidirectional=lstm_bi_config['bidirectional'],
            dropout=lstm_bi_config['dropout']
        ),
        'hypothesis': HYPOTHESES['lstm']['bidirectional']
    }
    
    # LSTM Variante 2: Apilada con dropout
    lstm_stacked_config = MODEL_CONFIG['lstm']['variants']['stacked']
    models_to_train['LSTM_Stacked'] = {
        'model': LSTMClassifier(
            input_size=n_features,
            hidden_size=lstm_stacked_config['hidden_size'],
            num_layers=lstm_stacked_config['num_layers'],
            num_classes=n_classes,
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
        'model': GRUClassifier(
            input_size=n_features,
            hidden_size=gru_base_config['hidden_size'],
            num_layers=gru_base_config['num_layers'],
            num_classes=n_classes,
            bidirectional=gru_base_config['bidirectional'],
            dropout=gru_base_config['dropout']
        ),
        'hypothesis': HYPOTHESES['gru']['base']
    }
    
    # GRU Variante 1: Bidireccional
    gru_bi_config = MODEL_CONFIG['gru']['variants']['bidirectional']
    models_to_train['GRU_Bidirectional'] = {
        'model': GRUClassifier(
            input_size=n_features,
            hidden_size=gru_bi_config['hidden_size'],
            num_layers=gru_bi_config['num_layers'],
            num_classes=n_classes,
            bidirectional=gru_bi_config['bidirectional'],
            dropout=gru_bi_config['dropout']
        ),
        'hypothesis': HYPOTHESES['gru']['bidirectional']
    }
    
    # GRU Variante 2: Apilada
    gru_stacked_config = MODEL_CONFIG['gru']['variants']['stacked']
    models_to_train['GRU_Stacked'] = {
        'model': GRUClassifier(
            input_size=n_features,
            hidden_size=gru_stacked_config['hidden_size'],
            num_layers=gru_stacked_config['num_layers'],
            num_classes=n_classes,
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
    criterion = nn.CrossEntropyLoss()
    
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
        checkpoint_path = os.path.join(PATHS['checkpoints'], f'{model_name}_classification_best.pth')
        
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
            task='classification',
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
        
        test_metrics = compute_classification_metrics(
            model, test_loader, device, class_names
        )
        
        print(f"\n📊 Resultados en Test:")
        print(f"   Accuracy: {test_metrics['accuracy']*100:.2f}%")
        print(f"   F1-Score (macro): {test_metrics['f1_macro']*100:.2f}%")
        print(f"\nReporte de Clasificación:")
        print(test_metrics['classification_report'])
        
        # Guardar resultados
        all_results[model_name] = {
            'model': model,
            'history': history,
            'test_metrics': test_metrics,
            'accuracy': test_metrics['accuracy'],
            'f1_macro': test_metrics['f1_macro'],
            'params_millions': params_millions,
            'training_time': training_time,
            'time_per_epoch': training_time / TRAIN_CONFIG['epochs']
        }
        
        # Visualizar historial de entrenamiento
        fig_path = os.path.join(PATHS['figures'], f'{model_name}_classification_history.png')
        plot_training_history(history, task='classification', save_path=fig_path, model_name=model_name)
        
        # Visualizar matriz de confusión
        cm_path = os.path.join(PATHS['figures'], f'{model_name}_classification_confusion_matrix.png')
        plot_confusion_matrix(
            test_metrics['confusion_matrix'],
            class_names,
            save_path=cm_path,
            model_name=model_name,
            normalize=True
        )
    
    # ========================================================================
    # 5. COMPARAR MODELOS Y GUARDAR RESULTADOS
    # ========================================================================
    
    print("\n" + "="*80)
    print("FASE 4: COMPARACIÓN DE MODELOS Y RESULTADOS FINALES")
    print("="*80 + "\n")
    
    # Gráfica comparativa
    comp_path = os.path.join(PATHS['figures'], 'classification_model_comparison.png')
    plot_model_comparison(all_results, task='classification', save_path=comp_path)
    
    # Tabla de resultados
    table_path = os.path.join(PATHS['results'], 'classification_results')
    results_df = save_results_table(all_results, table_path, task='classification')
    
    # Identificar mejor modelo
    best_model_name = max(all_results.keys(), key=lambda k: all_results[k]['f1_macro'])
    best_f1 = all_results[best_model_name]['f1_macro'] * 100
    
    print(f"\n🏆 MEJOR MODELO: {best_model_name}")
    print(f"   F1-Score (macro): {best_f1:.2f}%")
    print(f"   Accuracy: {all_results[best_model_name]['accuracy']*100:.2f}%")
    
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
