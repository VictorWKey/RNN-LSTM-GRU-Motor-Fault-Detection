"""
Archivo de configuración global para el proyecto.
Ajusta estos parámetros según tus necesidades.
"""

# Configuración de datos
DATA_CONFIG = {
    'classification': {
        'path': 'Dataset',  # Ruta al directorio con datos de clasificación
        'window_size': 32,
        'col_index': [0, 1, 2],  # Columnas a usar del CSV
        'classes': {
            'SC_HLT': 0,          # Sano
            'SC_A0_B0_C1': 1,
            'SC_A0_B0_C2': 2,
            'SC_A0_B0_C3': 3,
            'SC_A0_B0_C4': 4,
            # Agregar más clases según sea necesario
        }
    },
    'regression': {
        'window_size': 32,
        'horizon': 1,
        'T': 5000  # Longitud de serie temporal
    }
}

# Configuración de entrenamiento
TRAIN_CONFIG = {
    'batch_size': 64,
    'epochs': 70,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'clip_norm': 1.0,
    'random_seed': 42
}

# Configuración de modelos
MODEL_CONFIG = {
    'rnn': {
        'hidden_size': 128,
        'num_layers': 2,
        'nonlinearity': 'tanh'
    },
    'lstm': {
        'hidden_size': 64,
        'num_layers': 2,
        'bidirectional': False,
        'dropout': 0.0
    },
    'gru': {
        'hidden_size': 64,
        'num_layers': 2,
        'bidirectional': False,
        'dropout': 0.0
    }
}

# División de datos
SPLIT_CONFIG = {
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15
}
