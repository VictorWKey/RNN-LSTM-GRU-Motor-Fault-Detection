# Práctica 02: Modelos Recurrentes para Señales de Motor

**Asignatura:** Aprendizaje Profundo  
**Institución:** Universidad de Guanajuato  
**Fecha:** Noviembre 2025

---

## 📋 Descripción

Esta práctica implementa y compara tres arquitecturas recurrentes (RNN, LSTM y GRU) para dos tareas principales:

1. **Clasificación:** Identificar el estado de salud del motor entre 13 clases (1 sano + 4 niveles de falla × 3 fases)
2. **Regresión:** Predicción de series temporales

Para cada arquitectura se implementan:
- **Modelo base:** Configuración estándar
- **Variante 1:** Modificación arquitectural (ej: bidireccionalidad)
- **Variante 2:** Modificación adicional (ej: apilamiento de capas, dropout)

---

## 🗂️ Estructura del Proyecto

```
practica02/
├── config.py                      # Configuración central del proyecto
├── train_classification.py        # Script principal para clasificación
├── train_regression.py            # Script principal para regresión
├── requirements.txt               # Dependencias de Python
├── README.md                      # Este archivo
│
├── models/                        # Modelos RNN, LSTM, GRU
│   ├── __init__.py
│   └── rnn_models.py
│
├── utils/                         # Utilidades
│   ├── __init__.py
│   ├── data_utils.py             # Carga y preparación de datos
│   ├── training_utils.py         # Funciones de entrenamiento
│   └── visualization.py          # Visualización de resultados
│
├── Dataset/                       # Datos de señales de motor
│   ├── SC_HLT/                   # Clase sana
│   ├── SC_A0_B0_C1-4/            # Falla nivel 1-4 - Fase C
│   ├── SC_A0_B1-4_C0/            # Falla nivel 1-4 - Fase B
│   └── SC_A1-4_B0_C0/            # Falla nivel 1-4 - Fase A
│
├── checkpoints/                   # Modelos guardados (generado)
├── figures/                       # Gráficas y visualizaciones (generado)
├── results/                       # Tablas de resultados (generado)
└── logs/                          # Logs de entrenamiento (generado)
```

---

## 🚀 Instalación y Configuración

### 1. Requisitos Previos

- Python 3.8+
- CUDA 11.8+ (opcional, para GPU)
- pip

### 2. Instalar Dependencias

```bash
# Navegar al directorio del proyecto
cd practica02

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Linux/Mac
# venv\Scripts\activate   # En Windows

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Verificar Instalación

```bash
# Verificar PyTorch y CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}')"
```

---

## 📊 Dataset

### Clasificación de Señales de Motor

- **13 clases totales:**
  - `SC_HLT`: Motor sano (healthy)
  
  **Fase C (4 niveles):**
  - `SC_A0_B0_C1`: Falla nivel 1 - Fase C
  - `SC_A0_B0_C2`: Falla nivel 2 - Fase C
  - `SC_A0_B0_C3`: Falla nivel 3 - Fase C
  - `SC_A0_B0_C4`: Falla nivel 4 - Fase C
  
  **Fase B (4 niveles):**
  - `SC_A0_B1_C0`: Falla nivel 1 - Fase B
  - `SC_A0_B2_C0`: Falla nivel 2 - Fase B
  - `SC_A0_B3_C0`: Falla nivel 3 - Fase B
  - `SC_A0_B4_C0`: Falla nivel 4 - Fase B
  
  **Fase A (4 niveles):**
  - `SC_A1_B0_C0`: Falla nivel 1 - Fase A
  - `SC_A2_B0_C0`: Falla nivel 2 - Fase A
  - `SC_A3_B0_C0`: Falla nivel 3 - Fase A
  - `SC_A4_B0_C0`: Falla nivel 4 - Fase A

- **Formato:** Archivos CSV con 3 columnas (señales de 3 fases del motor)
- **5 archivos por clase** con ~1000 muestras cada uno

### Regresión de Series Temporales

- **Serie sintética generada automáticamente**
- 5000 puntos temporales
- Componentes: tendencia + estacionalidad + ruido

---

## 🎯 Uso

### Entrenamiento de Clasificación

```bash
python train_classification.py
```

Este script:
1. Carga las señales de motor
2. Crea ventanas temporales de tamaño 64
3. Entrena 9 modelos (3 arquitecturas × 3 variantes)
4. Genera gráficas de curvas de entrenamiento y matrices de confusión
5. Guarda resultados en tablas CSV y LaTeX

**Parámetros principales** (en `config.py`):
- Épocas: 80
- Batch size: 64
- Learning rate: 0.001
- Window size: 64

### Entrenamiento de Regresión

```bash
python train_regression.py
```

Similar al de clasificación, pero para predicción de series temporales.

---

## 🏗️ Modelos Implementados

### 1. RNN (Vanilla/Elman)

**Base:**
- 1 capa RNN
- Hidden size: 128
- Activación: tanh

**Variantes:**
- **RNN_Deep:** 2 capas apiladas
- **RNN_ReLU:** Activación ReLU en lugar de tanh

### 2. LSTM (Long Short-Term Memory)

**Base:**
- 1 capa LSTM
- Hidden size: 64

**Variantes:**
- **LSTM_Bidirectional:** LSTM bidireccional
- **LSTM_Stacked:** 2 capas LSTM con dropout 0.2

### 3. GRU (Gated Recurrent Unit)

**Base:**
- 1 capa GRU
- Hidden size: 64

**Variantes:**
- **GRU_Bidirectional:** GRU bidireccional
- **GRU_Stacked:** 2 capas GRU con dropout 0.2

---

## 📈 Métricas Evaluadas

### Clasificación
- **Accuracy:** Precisión general
- **F1-Score (macro):** Media armónica de precisión y recall
- **Matriz de confusión:** Análisis detallado de errores
- **Classification report:** Métricas por clase

### Regresión
- **MSE:** Error cuadrático medio
- **RMSE:** Raíz del error cuadrático medio
- **MAE:** Error absoluto medio
- **R²:** Coeficiente de determinación

---

## 📁 Resultados Generados

Después de ejecutar los scripts, se generan:

### Checkpoints
```
checkpoints/
├── RNN_Base_classification_best.pth
├── LSTM_Bidirectional_classification_best.pth
├── GRU_Stacked_regression_best.pth
└── ...
```

### Figuras
```
figures/
├── RNN_Base_classification_history.png
├── LSTM_Bidirectional_classification_confusion_matrix.png
├── GRU_Stacked_regression_predictions.png
├── classification_model_comparison.png
└── regression_model_comparison.png
```

### Tablas de Resultados
```
results/
├── classification_results.csv
├── classification_results.tex
├── regression_results.csv
└── regression_results.tex
```

---

## 🔬 Hipótesis y Preguntas de Investigación

### Hipótesis por Variante

**RNN:**
- **Base:** Capturará patrones básicos en señales
- **Deep:** Mayor profundidad mejorará F1-Score
- **ReLU:** Evitará gradiente desvaneciente

**LSTM:**
- **Base:** Capturará dependencias a largo plazo
- **Bidirectional:** Contexto pasado/futuro mejorará detección de fallas
- **Stacked:** Dropout regularizará y mejorará generalización

**GRU:**
- **Base:** Rendimiento similar a LSTM con menos costo
- **Bidirectional:** Mejorará detección bidireccional
- **Stacked:** Aumentará capacidad manteniendo eficiencia

### Preguntas de Investigación

1. ¿Superará LSTM a RNN simple en dependencias largas?
2. ¿Mejorará la bidireccionalidad el F1-Score en fallas incipientes?
3. ¿GRU logrará balance óptimo entre rendimiento y eficiencia vs LSTM?
4. ¿Mayor profundidad mejorará métricas a costa de tiempo?
5. ¿Qué arquitectura es más robusta ante overfitting?

---

## 🛠️ Personalización

### Modificar Hiperparámetros

Editar `config.py`:

```python
TRAIN_CONFIG = {
    'epochs': 100,          # Cambiar número de épocas
    'batch_size': 128,      # Cambiar tamaño de batch
    'learning_rate': 0.0001 # Cambiar learning rate
}
```

### Agregar Nueva Variante

1. Editar `config.py` en `MODEL_CONFIG`
2. Agregar configuración en `HYPOTHESES`
3. Instanciar modelo en `train_classification.py` o `train_regression.py`

---

## 📊 Análisis de Resultados

Los resultados deben analizarse considerando:

1. **RNN vs LSTM/GRU:** ¿Problema de gradiente desvaneciente en RNN?
2. **LSTM vs GRU:** ¿Justifica LSTM su mayor costo computacional?
3. **Clasificación vs Regresión:** ¿Qué tarea fue más difícil?
4. **Errores:** Análisis de matriz de confusión y scatter plots
5. **Variantes:** ¿Se validaron las hipótesis?

---

## 🐛 Troubleshooting

### Error: CUDA out of memory
```bash
# Reducir batch size en config.py
TRAIN_CONFIG = {
    'batch_size': 32  # Era 64
}
```

### Error: No se encuentra el dataset
```bash
# Verificar que existe Dataset/ con las carpetas de clases
ls Dataset/
```

### Entrenamiento muy lento
```bash
# Verificar que está usando GPU
python -c "import torch; print(torch.cuda.is_available())"

# Reducir número de épocas para pruebas
TRAIN_CONFIG = {
    'epochs': 20  # En lugar de 80
}
```

---

## 📝 Reporte Académico

Los resultados de esta práctica deben documentarse siguiendo la filosofía **IMRA**:

1. **Introducción:** Contexto, problema, objetivos, hipótesis
2. **Metodología:** Datos, modelos, protocolo de entrenamiento
3. **Resultados:** Tablas, figuras, métricas
4. **Análisis:** Discusión de hipótesis, comparaciones, errores

---

## 👥 Autor

Práctica desarrollada para el curso de **Aprendizaje Profundo**  
Universidad de Guanajuato

---

## 📜 Licencia

Este proyecto es material educativo para uso académico.

---

## 🙏 Agradecimientos

- Código base inspirado en ejemplos de PyTorch
- Dataset de señales de motor (sintético para demostración)
- Instructor: M.I. Juan José Cárdenas Cornejo

---

**Última actualización:** Noviembre 2025
