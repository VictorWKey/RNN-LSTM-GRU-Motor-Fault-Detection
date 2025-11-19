"""
Configuración principal para la Práctica 02 - Modelos Recurrentes para Señales
"""
import torch
import numpy as np

# ============================================================================
# CONFIGURACIÓN GLOBAL DE SEMILLAS
# ============================================================================
RANDOM_SEED = 42

def set_seed(seed=RANDOM_SEED):
    """Establece semillas para reproducibilidad"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ============================================================================
# CONFIGURACIÓN DE DATOS
# ============================================================================
DATA_CONFIG = {
    'classification': {
        'path': 'Dataset',  # Ruta al directorio con datos de clasificación
        'window_size': 64,  # Tamaño de ventana para secuencias
        'stride': 1,  # Paso para ventanas deslizantes
        'col_index': [0, 1, 2],  # Columnas a usar del CSV (3 fases del motor)
        'classes': {
            # Clase sana
            'SC_HLT': 0,          # Sano
            
            # Fase C: 4 niveles de falla
            'SC_A0_B0_C1': 1,     # Falla nivel 1 - Fase C
            'SC_A0_B0_C2': 2,     # Falla nivel 2 - Fase C
            'SC_A0_B0_C3': 3,     # Falla nivel 3 - Fase C
            'SC_A0_B0_C4': 4,     # Falla nivel 4 - Fase C
            
            # Fase B: 4 niveles de falla
            'SC_A0_B1_C0': 5,     # Falla nivel 1 - Fase B
            'SC_A0_B2_C0': 6,     # Falla nivel 2 - Fase B
            'SC_A0_B3_C0': 7,     # Falla nivel 3 - Fase B
            'SC_A0_B4_C0': 8,     # Falla nivel 4 - Fase B
            
            # Fase A: 4 niveles de falla
            'SC_A1_B0_C0': 9,     # Falla nivel 1 - Fase A
            'SC_A2_B0_C0': 10,    # Falla nivel 2 - Fase A
            'SC_A3_B0_C0': 11,    # Falla nivel 3 - Fase A
            'SC_A4_B0_C0': 12,    # Falla nivel 4 - Fase A
        },
        'num_classes': 13
    },
    'regression': {
        'window_size': 64,  # Tamaño de ventana de entrada
        'horizon': 1,       # Horizonte de predicción
        'T': 5000,          # Longitud de serie temporal a generar
        'train_split': 0.7,
        'val_split': 0.15,
        'test_split': 0.15
    }
}

# ============================================================================
# CONFIGURACIÓN DE ENTRENAMIENTO (PROTOCOLO COMÚN SEGÚN EL .TEX)
# ============================================================================
TRAIN_CONFIG = {
    'epochs': 80,           # Según documento: 80 épocas
    'batch_size': 64,       # Según documento: 64
    'learning_rate': 0.001, # Según documento: 0.001
    'weight_decay': 1e-5,   # Según documento: 1e-5
    'optimizer': 'Adam',    # Según documento: Adam
    'clip_norm': 1.0,       # Gradient clipping
    'patience': 15,         # Para early stopping (opcional)
}

# ============================================================================
# DIVISIÓN DE DATOS
# ============================================================================
SPLIT_CONFIG = {
    'train_ratio': 0.70,  # 70% entrenamiento
    'val_ratio': 0.15,    # 15% validación
    'test_ratio': 0.15,   # 15% test
}

# ============================================================================
# CONFIGURACIÓN DE MODELOS BASE
# ============================================================================
MODEL_CONFIG = {
    'rnn': {
        'base': {
            'hidden_size': 128,
            'num_layers': 1,
            'nonlinearity': 'tanh',
            'bidirectional': False,
            'dropout': 0.0
        },
        'variants': {
            'deep': {  # Variante 1: Mayor profundidad
                'hidden_size': 128,
                'num_layers': 2,
                'nonlinearity': 'tanh',
                'bidirectional': False,
                'dropout': 0.0
            },
            'relu': {  # Variante 2: Activación ReLU
                'hidden_size': 128,
                'num_layers': 1,
                'nonlinearity': 'relu',
                'bidirectional': False,
                'dropout': 0.0
            }
        }
    },
    'lstm': {
        'base': {
            'hidden_size': 64,
            'num_layers': 1,
            'bidirectional': False,
            'dropout': 0.0
        },
        'variants': {
            'bidirectional': {  # Variante 1: LSTM Bidireccional
                'hidden_size': 64,
                'num_layers': 1,
                'bidirectional': True,
                'dropout': 0.0
            },
            'stacked': {  # Variante 2: LSTM Apilada con dropout
                'hidden_size': 64,
                'num_layers': 2,
                'bidirectional': False,
                'dropout': 0.2
            }
        }
    },
    'gru': {
        'base': {
            'hidden_size': 64,
            'num_layers': 1,
            'bidirectional': False,
            'dropout': 0.0
        },
        'variants': {
            'bidirectional': {  # Variante 1: GRU Bidireccional
                'hidden_size': 64,
                'num_layers': 1,
                'bidirectional': True,
                'dropout': 0.0
            },
            'stacked': {  # Variante 2: GRU Apilada
                'hidden_size': 64,
                'num_layers': 2,
                'bidirectional': False,
                'dropout': 0.2
            }
        }
    }
}

# ============================================================================
# CONFIGURACIÓN DE CRITERIOS (LOSS FUNCTIONS)
# ============================================================================
CRITERION_CONFIG = {
    'classification': 'CrossEntropyLoss',  # Según documento
    'regression': 'MSELoss'                # Según documento
}

# ============================================================================
# RUTAS DE ARCHIVOS
# ============================================================================
PATHS = {
    'dataset': 'Dataset',
    'models': 'models',
    'utils': 'utils',
    'results': 'results',
    'checkpoints': 'checkpoints',
    'figures': 'figures',
    'logs': 'logs'
}

# ============================================================================
# CONFIGURACIÓN DE VISUALIZACIÓN
# ============================================================================
VIZ_CONFIG = {
    'dpi': 300,
    'figsize': (12, 8),
    'save_format': 'png',
    'style': 'seaborn-v0_8-darkgrid'
}

# ============================================================================
# MÉTRICAS A CALCULAR
# ============================================================================
METRICS_CONFIG = {
    'classification': ['accuracy', 'f1_macro', 'confusion_matrix', 'classification_report'],
    'regression': ['mse', 'rmse', 'mae', 'r2']
}

# ============================================================================
# HIPÓTESIS POR VARIANTE (SEGÚN REQUISITOS DEL DOCUMENTO)
# ============================================================================
HYPOTHESES = {
    'rnn': {
        'base': "La RNN simple con activación tanh capturará patrones básicos en las señales de motor.",
        'deep': "Incrementar la profundidad (2 capas) mejorará la capacidad de modelar dependencias temporales complejas, aumentando F1-Score en clasificación.",
        'relu': "Usar activación ReLU evitará el problema del gradiente desvaneciente mejor que tanh, mejorando la convergencia."
    },
    'lstm': {
        'base': "LSTM capturará dependencias a largo plazo mejor que RNN simple gracias a su mecanismo de compuertas.",
        'bidirectional': "La bidireccionalidad permitirá al modelo capturar contexto futuro y pasado, mejorando F1-Score especialmente en fallas incipientes.",
        'stacked': "Apilar capas LSTM con dropout regularizará el modelo y mejorará la generalización en el conjunto de test."
    },
    'gru': {
        'base': "GRU logrará rendimiento similar a LSTM pero con menor costo computacional (menos parámetros).",
        'bidirectional': "GRU bidireccional mejorará la detección de patrones en ambas direcciones temporales.",
        'stacked': "GRU apilada aumentará la capacidad de modelado manteniendo eficiencia computacional."
    }
}

# ============================================================================
# PREGUNTAS DE INVESTIGACIÓN
# ============================================================================
RESEARCH_QUESTIONS = [
    "¿Superará LSTM a la RNN simple en dependencias largas de las señales de motor?",
    "¿Mejorará la bidireccionalidad el F1-Score en la detección de fallas incipientes?",
    "¿GRU logrará un balance óptimo entre rendimiento y eficiencia computacional comparado con LSTM?",
    "¿Las variantes con mayor profundidad mejorarán las métricas a costa de tiempo de entrenamiento?",
    "¿Qué arquitectura es más robusta ante el overfitting en señales de motor?"
]

# ============================================================================
# INFORMACIÓN DE HARDWARE (PARA DOCUMENTACIÓN)
# ============================================================================
HARDWARE_INFO = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'cuda_available': torch.cuda.is_available(),
    'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
}

# Mostrar información al importar
if __name__ == '__main__':
    print("="*70)
    print("CONFIGURACIÓN DE LA PRÁCTICA 02")
    print("="*70)
    print(f"Dispositivo: {HARDWARE_INFO['device']}")
    print(f"Épocas: {TRAIN_CONFIG['epochs']}")
    print(f"Batch size: {TRAIN_CONFIG['batch_size']}")
    print(f"Learning rate: {TRAIN_CONFIG['learning_rate']}")
    print(f"Clases de clasificación: {DATA_CONFIG['classification']['num_classes']}")
    print(f"Tamaño de ventana: {DATA_CONFIG['classification']['window_size']}")
    print("="*70)
