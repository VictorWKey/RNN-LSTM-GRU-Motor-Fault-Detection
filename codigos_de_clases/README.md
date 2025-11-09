# Proyecto de Modelos Recurrentes para Señales

Este proyecto implementa y compara arquitecturas de redes neuronales recurrentes (RNN, LSTM, GRU) para dos tareas:
1. **Clasificación de señales de motor** (13 clases de fallas)
2. **Regresión de series temporales**

## Estructura del Proyecto

```
codigos_de_clases/
├── models/
│   ├── __init__.py
│   └── rnn_models.py          # Definiciones de RNN, LSTM, GRU
├── utils/
│   ├── __init__.py
│   ├── data_utils.py          # Preparación de datos
│   ├── training_utils.py      # Funciones de entrenamiento
│   └── visualization.py       # Gráficos y visualización
├── train_classification.py    # Script para clasificación
├── train_regression.py        # Script para regresión
├── requirements.txt           # Dependencias
└── README.md                  # Este archivo
```

## Instalación

### 1. Crear entorno virtual (recomendado)

```bash
python -m venv venv
source venv/bin/activate  # En Linux/Mac
# o
venv\Scripts\activate  # En Windows
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

## Uso

### Clasificación de Señales de Motor

El script `train_classification.py` entrena múltiples modelos (RNN, LSTM, GRU) para clasificar señales de motor.

**Antes de ejecutar:**
1. Asegúrate de tener tus datos en la carpeta `Dataset/` con la siguiente estructura:
   ```
   Dataset/
   ├── SC_HLT/           # Clase 0 (sano)
   │   ├── file1.csv
   │   └── file2.csv
   ├── SC_A0_B0_C1/      # Clase 1
   ├── SC_A0_B0_C2/      # Clase 2
   └── ...
   ```
2. Ajusta las rutas y nombres de clases en el script según tus datos

**Ejecutar:**
```bash
python train_classification.py
```

**Salidas:**
- Modelos entrenados guardados como `.pth`
- Gráficas de entrenamiento (pérdida y accuracy)
- Matriz de confusión
- Reporte de clasificación con métricas

### Regresión de Series Temporales

El script `train_regression.py` entrena modelos para predicción de series temporales.

**Ejecutar:**
```bash
python train_regression.py
```

Este script genera automáticamente datos sintéticos para demostración. Puedes modificarlo para usar tus propios datos.

**Salidas:**
- Modelos entrenados guardados como `.pth`
- Gráficas de pérdida durante entrenamiento
- Gráficas de predicciones vs valores reales
- Métricas: MSE, RMSE, R²

## Modelos Disponibles

### Para Clasificación
- `BiRNNClassifier`: RNN bidireccional
- `LSTMClassifier`: LSTM (unidireccional o bidireccional)
- `GRUClassifier`: GRU (unidireccional o bidireccional)

### Para Regresión
- `RNNSimple`: RNN simple (tanh o relu)
- `LSTMRegressor`: LSTM para regresión
- `GRURegressor`: GRU para regresión

## Configuración

Los principales hiperparámetros se pueden ajustar en los scripts:

- `WINDOW_SIZE`: Tamaño de ventana deslizante (default: 32)
- `BATCH_SIZE`: Tamaño de batch (default: 64)
- `EPOCHS`: Número de épocas (default: 70)
- `LEARNING_RATE`: Tasa de aprendizaje (default: 0.001)
- `hidden_size`: Tamaño de capa oculta (default: 64 o 128)
- `num_layers`: Número de capas recurrentes (default: 2)

## Personalización

### Agregar un nuevo modelo

1. Define tu modelo en `models/rnn_models.py`
2. Agrégalo a `models/__init__.py`
3. Inclúyelo en el diccionario `models_to_train` del script correspondiente

### Usar tus propios datos

**Para clasificación:**
1. Organiza tus archivos CSV por clases
2. Actualiza `name_cls` y `path_db` en `train_classification.py`
3. Ajusta `col_index` según las columnas que quieras usar

**Para regresión:**
1. Carga tus datos en lugar de `generar_serie_temporal()`
2. Ajusta los parámetros de ventana y horizonte según tu problema

## Resultados Esperados

### Clasificación
- **Accuracy**: >90% en conjunto de prueba (depende de los datos)
- **F1-Score (macro)**: >0.85
- Matriz de confusión mostrando confusiones entre clases

### Regresión
- **RMSE**: Bajo error en datos normalizados
- **R²**: >0.90 indicando buen ajuste
- Predicciones cercanas a valores reales

## Troubleshooting

**Error: "No se encontraron archivos"**
- Verifica que la ruta `path_db` apunte correctamente a tus datos
- Asegúrate de que los archivos CSV estén en las subcarpetas correctas

**CUDA out of memory**
- Reduce `BATCH_SIZE`
- Reduce `hidden_size` o `num_layers`

**Underfitting (accuracy muy baja)**
- Aumenta `hidden_size` o `num_layers`
- Aumenta `EPOCHS`
- Prueba con `bidirectional=True`

**Overfitting (train accuracy >> val accuracy)**
- Agrega dropout: `dropout=0.2` al crear el modelo
- Reduce complejidad del modelo
- Aumenta `WEIGHT_DECAY`

## Referencias

- **Práctica #2**: Modelos Recurrentes para Señales
- **Instructor**: M.I. Juan José Cárdenas Cornejo
- **Curso**: Aprendizaje Profundo

## Licencia

Este código es material educativo para el curso de Aprendizaje Profundo.
