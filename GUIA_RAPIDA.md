# 🚀 Guía de Inicio Rápido - Práctica 02

## Modelos Recurrentes para Señales de Motor

---

## ⚡ Instalación Rápida

### 1. Instalar dependencias

```bash
cd practica02
pip install -r requirements.txt
```

### 2. Verificar instalación de PyTorch

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 🎯 Ejecutar Entrenamientos

### Opción 1: Ejecutar TODO (Recomendado)

```bash
python run_all.py
```

Esto ejecutará secuencialmente:
1. Entrenamiento de clasificación (9 modelos)
2. Entrenamiento de regresión (9 modelos)

**Tiempo estimado:** 2-4 horas (dependiendo de GPU/CPU)

### Opción 2: Solo Clasificación

```bash
python train_classification.py
```

**Tiempo estimado:** 1-2 horas

### Opción 3: Solo Regresión

```bash
python train_regression.py
```

**Tiempo estimado:** 1-2 horas

### Opción 4: Ejecutar selectivamente

```bash
# Solo clasificación
python run_all.py --classification-only

# Solo regresión
python run_all.py --regression-only
```

---

## 📊 ¿Qué se genera?

Al finalizar, tendrás:

### Checkpoints (modelos entrenados)
```
checkpoints/
├── RNN_Base_classification_best.pth
├── RNN_Deep_classification_best.pth
├── RNN_ReLU_classification_best.pth
├── LSTM_Base_classification_best.pth
├── LSTM_Bidirectional_classification_best.pth
├── LSTM_Stacked_classification_best.pth
├── GRU_Base_classification_best.pth
├── GRU_Bidirectional_classification_best.pth
├── GRU_Stacked_classification_best.pth
└── ... (lo mismo para regression)
```

### Figuras
```
figures/
├── *_classification_history.png      (Curvas de entrenamiento)
├── *_classification_confusion_matrix.png (Matrices de confusión)
├── *_regression_predictions.png      (Scatter plots)
├── *_regression_timeseries.png       (Series temporales)
├── classification_model_comparison.png
└── regression_model_comparison.png
```

### Tablas de Resultados
```
results/
├── classification_results.csv  (Para Excel/Google Sheets)
├── classification_results.tex  (Para LaTeX)
├── regression_results.csv
└── regression_results.tex
```

---

## 🔧 Configuración Rápida

### Cambiar Hiperparámetros

Editar `config.py`:

```python
TRAIN_CONFIG = {
    'epochs': 80,           # Número de épocas
    'batch_size': 64,       # Tamaño de batch
    'learning_rate': 0.001, # Learning rate
}
```

### Cambiar Tamaño de Ventana

```python
DATA_CONFIG = {
    'classification': {
        'window_size': 64,  # Tamaño de ventana temporal
    }
}
```

---

## 📈 Monitoreo durante Entrenamiento

Durante el entrenamiento verás output como:

```
================================================================================
ENTRENANDO: LSTM_Bidirectional
================================================================================
Hipótesis: La bidireccionalidad permitirá al modelo capturar contexto...
--------------------------------------------------------------------------------

Época   1/80: Train Loss=1.4523, Train Acc=42.34% | Val Loss=1.2845, Val Acc=48.23%, Val F1=45.67% | Time=5.23s
Época  10/80: Train Loss=0.8234, Train Acc=68.45% | Val Loss=0.7912, Val Acc=72.34%, Val F1=70.12% | Time=5.18s
Época  20/80: Train Loss=0.5123, Train Acc=82.67% | Val Loss=0.5456, Val Acc=80.45%, Val F1=78.89% | Time=5.21s
...
Época  80/80: Train Loss=0.1234, Train Acc=95.23% | Val Loss=0.2345, Val Acc=91.34%, Val F1=89.67% | Time=5.19s

✅ Entrenamiento completado en 418.32s (6.97 min)
   Mejor val loss: 0.2156

================================================================================
EVALUACIÓN EN TEST: LSTM_Bidirectional
================================================================================

📊 Resultados en Test:
   Accuracy: 90.12%
   F1-Score (macro): 88.45%
```

---

## ⚙️ Optimización para Pruebas Rápidas

Si quieres hacer pruebas rápidas, edita `config.py`:

```python
TRAIN_CONFIG = {
    'epochs': 10,          # Reducir épocas
    'batch_size': 128,     # Aumentar batch size
}

DATA_CONFIG = {
    'regression': {
        'T': 1000,         # Menos puntos temporales
    }
}
```

---

## 🐛 Solución de Problemas Comunes

### Error: CUDA out of memory

```python
# En config.py, reducir batch size
TRAIN_CONFIG = {
    'batch_size': 32  # Era 64
}
```

### Entrenamiento muy lento (CPU)

```python
# Reducir complejidad
TRAIN_CONFIG = {
    'epochs': 40,  # En lugar de 80
}

MODEL_CONFIG = {
    'lstm': {
        'base': {
            'hidden_size': 32,  # En lugar de 64
        }
    }
}
```

### Dataset no encontrado

```bash
# Verificar estructura
ls Dataset/
# Debería mostrar: SC_HLT SC_A0_B0_C1 SC_A0_B0_C2 SC_A0_B0_C3 SC_A0_B0_C4
```

---

## 📝 Generar Reporte

### 1. Recopilar Resultados

Después de entrenar, los resultados están en:
- `results/classification_results.csv`
- `results/regression_results.csv`
- `figures/*.png`

### 2. Estructura del Reporte (IMRA)

#### I. Introducción
- Contexto de RNNs para señales
- Problema: 5 clases de motor
- Objetivos
- Hipótesis (ver `HIPOTESIS_Y_ANALISIS.md`)

#### II. Metodología
- Dataset (5 clases, 3 features)
- Preprocesamiento (ventanas, normalización)
- Modelos (RNN, LSTM, GRU + variantes)
- Protocolo de entrenamiento

#### III. Resultados
- **Tabla I:** Clasificación - copiar de `classification_results.csv`
- **Tabla II:** Regresión - copiar de `regression_results.csv`
- **Figuras:** Usar PNGs de `figures/`

#### IV. Análisis
- Validación de hipótesis
- Comparaciones RNN/LSTM/GRU
- Análisis de errores (matriz confusión)
- Conclusiones

---

## 🎓 Checklist de Entrega

- [ ] Código ejecutado completamente
- [ ] Checkpoints guardados en `checkpoints/`
- [ ] Todas las figuras generadas en `figures/`
- [ ] Tablas CSV y LaTeX en `results/`
- [ ] Reporte PDF con estructura IMRA
- [ ] Análisis de hipótesis completado
- [ ] Comparación de modelos documentada
- [ ] Código comentado y limpio

---

## 💡 Consejos para el Reporte

1. **No inventar resultados:** Usar los reales generados
2. **Analizar tendencias:** No solo reportar números
3. **Comparar con hipótesis:** ¿Se confirmaron?
4. **Explicar diferencias:** Si hipótesis falló, ¿por qué?
5. **Matrices de confusión:** Analizar patrones de error
6. **Figuras profesionales:** Ya están en alta resolución (300 DPI)

---

## 📚 Recursos Adicionales

- `README.md` - Documentación completa
- `HIPOTESIS_Y_ANALISIS.md` - Guía de análisis detallada
- `config.py` - Configuración con comentarios
- `models/rnn_models.py` - Arquitecturas con docstrings

---

## 🚀 Siguiente Nivel (Opcional)

### Agregar Nueva Variante

1. Definir configuración en `config.py`:
```python
MODEL_CONFIG = {
    'lstm': {
        'variants': {
            'attention': {
                'hidden_size': 64,
                'num_layers': 1,
                'use_attention': True  # Nueva feature
            }
        }
    }
}
```

2. Implementar en `models/rnn_models.py`
3. Agregar instancia en `train_classification.py`
4. Ejecutar entrenamiento

---

## ✅ Verificación Final

Antes de entregar, ejecuta:

```bash
# Verificar estructura
ls -R practica02/

# Verificar resultados
ls checkpoints/ figures/ results/

# Contar modelos entrenados
ls checkpoints/*.pth | wc -l
# Debería dar 18 (9 clasificación + 9 regresión)

# Verificar figuras
ls figures/*.png | wc -l
# Debería dar al menos 36
```

---

**¡Éxito en tu práctica! 🎉**

Si tienes dudas, revisa:
1. `README.md` para documentación completa
2. `HIPOTESIS_Y_ANALISIS.md` para análisis detallado
3. Comentarios en el código

---

**Última actualización:** Noviembre 2025
