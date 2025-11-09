# Hipótesis y Preguntas de Investigación - Práctica 02

## Modelos Recurrentes para Análisis de Señales de Motor

---

## 🎯 Objetivos de Aprendizaje

1. Implementar y entrenar RNN, LSTM y GRU para clasificación y regresión de señales
2. Aplicar técnicas de preprocesamiento de señales para RNNs (ventanas, normalización)
3. Diseñar y evaluar variantes arquitecturales (bidireccionalidad, apilamiento, dropout)
4. Analizar métricas (Accuracy, F1-Score, RMSE, R²) y curvas de entrenamiento

---

## ❓ Preguntas de Investigación

### Pregunta 1: Comparación RNN vs LSTM/GRU
**¿Superará LSTM a la RNN simple en la captura de dependencias largas de las señales de motor?**

- **Contexto:** Las RNN simples sufren del problema del gradiente desvaneciente en secuencias largas
- **Esperado:** LSTM y GRU deberían superar a RNN en F1-Score y Accuracy
- **Métricas a observar:** 
  - Convergencia durante entrenamiento
  - F1-Score final en test
  - Análisis de matriz de confusión

### Pregunta 2: Impacto de Bidireccionalidad
**¿Mejorará la bidireccionalidad el F1-Score en la detección de fallas incipientes?**

- **Contexto:** La bidireccionalidad permite capturar contexto pasado y futuro
- **Esperado:** Mejora en detección de patrones sutiles en fallas tempranas
- **Métricas a observar:**
  - F1-Score por clase (especialmente fallas leves)
  - Análisis de falsos negativos en fallas incipientes

### Pregunta 3: Eficiencia GRU vs LSTM
**¿Logrará GRU un balance óptimo entre rendimiento y eficiencia computacional comparado con LSTM?**

- **Contexto:** GRU tiene menos parámetros que LSTM pero rendimiento similar
- **Esperado:** GRU comparable en métricas pero más rápido
- **Métricas a observar:**
  - F1-Score / RMSE comparables
  - Tiempo por época
  - Número de parámetros

### Pregunta 4: Profundidad vs Costo
**¿Las variantes con mayor profundidad mejorarán las métricas a costa de tiempo de entrenamiento?**

- **Contexto:** Más capas = más capacidad pero mayor costo
- **Esperado:** Mejora marginal en métricas con aumento significativo en tiempo
- **Métricas a observar:**
  - Ganancia en F1-Score / RMSE
  - Incremento en tiempo de entrenamiento
  - Análisis costo-beneficio

### Pregunta 5: Robustez ante Overfitting
**¿Qué arquitectura es más robusta ante el overfitting en señales de motor?**

- **Contexto:** Dropout y regularización ayudan a generalizar
- **Esperado:** Variantes con dropout mostrarán menor brecha train-val
- **Métricas a observar:**
  - Gap entre train y validation accuracy/loss
  - Rendimiento en test
  - Curvas de entrenamiento

---

## 💡 Hipótesis por Modelo

### RNN (Elman/Vanilla)

#### Modelo Base
**Hipótesis:** *La RNN simple con activación tanh capturará patrones básicos temporales en las señales de motor, pero mostrará limitaciones en dependencias largas.*

**Justificación:**
- Tanh es estándar para RNN
- Suficiente para patrones locales
- Problema de gradiente desvaneciente en secuencias largas

**Predicción:**
- Accuracy: 60-75%
- F1-Score: 55-70%
- Convergencia lenta después de época 30

#### Variante 1: RNN Profunda (2 capas)
**Hipótesis:** *Incrementar la profundidad (2 capas) mejorará la capacidad de modelar dependencias temporales complejas, aumentando F1-Score en 5-10% respecto al modelo base.*

**Justificación:**
- Mayor capacidad de abstracción jerárquica
- Dos capas pueden capturar patrones en múltiples escalas temporales

**Predicción:**
- F1-Score: +5-10% vs base
- Tiempo por época: +30-50%
- Posible overfitting si no se regulariza

#### Variante 2: RNN con ReLU
**Hipótesis:** *Usar activación ReLU en lugar de tanh evitará el problema del gradiente desvaneciente, mejorando la convergencia y métricas finales.*

**Justificación:**
- ReLU no sufre saturación como tanh
- Gradientes más estables en secuencias largas

**Predicción:**
- Convergencia más rápida (alcanzar plateau en época 40 vs 50)
- F1-Score similar o +3-5% vs tanh
- Posible "dying ReLU" si learning rate es muy alto

---

### LSTM (Long Short-Term Memory)

#### Modelo Base
**Hipótesis:** *LSTM capturará dependencias a largo plazo mejor que RNN simple gracias a su mecanismo de compuertas (gates), superándola en 10-15% en F1-Score.*

**Justificación:**
- Gates (forget, input, output) previenen gradiente desvaneciente
- Cell state permite memoria a largo plazo
- Arquitectura probada en secuencias temporales

**Predicción:**
- Accuracy: 75-85%
- F1-Score: 70-80%
- Mejor detección de patrones complejos

#### Variante 1: LSTM Bidireccional
**Hipótesis:** *La bidireccionalidad permitirá al modelo capturar contexto futuro y pasado simultáneamente, mejorando F1-Score especialmente en fallas incipientes (+5-7% vs LSTM base).*

**Justificación:**
- Contexto bidireccional útil para detección de anomalías
- Fallas incipientes pueden tener precursores y consecuencias

**Predicción:**
- F1-Score: +5-7% vs LSTM base
- Mejor performance en clases con fallas leves
- Parámetros x2, tiempo por época +40-60%

#### Variante 2: LSTM Apilada con Dropout
**Hipótesis:** *Apilar capas LSTM con dropout (0.2) regularizará el modelo y mejorará la generalización en el conjunto de test, reduciendo el gap train-val en 3-5%.*

**Justificación:**
- Dropout previene co-adaptación de neuronas
- 2 capas aumentan capacidad de modelado
- Regularización mejora generalización

**Predicción:**
- Gap train-val: -3-5%
- F1-Score test similar o +2-3% vs base
- Menor overfitting en curvas de entrenamiento

---

### GRU (Gated Recurrent Unit)

#### Modelo Base
**Hipótesis:** *GRU logrará rendimiento similar a LSTM (diferencia < 2% en F1-Score) pero con menor costo computacional debido a menos parámetros (~25% menos).*

**Justificación:**
- GRU simplifica LSTM (2 gates vs 3)
- Performance similar en muchas tareas
- Más eficiente computacionalmente

**Predicción:**
- F1-Score: 68-78% (similar a LSTM ±2%)
- Parámetros: ~25% menos que LSTM
- Tiempo por época: -15-20% vs LSTM

#### Variante 1: GRU Bidireccional
**Hipótesis:** *GRU bidireccional mejorará la detección de patrones en ambas direcciones temporales, logrando F1-Score comparable a LSTM bidireccional pero con menor costo.*

**Justificación:**
- Bidireccionalidad útil independiente de arquitectura
- GRU mantiene eficiencia incluso bidireccional

**Predicción:**
- F1-Score: dentro de ±2% de LSTM bidireccional
- Tiempo: -10-15% vs LSTM bidireccional
- Mejor costo-beneficio

#### Variante 2: GRU Apilada
**Hipótesis:** *GRU apilada (2 capas con dropout) aumentará la capacidad de modelado manteniendo eficiencia computacional, logrando el mejor balance rendimiento-costo.*

**Justificación:**
- 2 capas GRU < parámetros que 2 capas LSTM
- Dropout regulariza
- Mantiene eficiencia de GRU

**Predicción:**
- F1-Score: +4-6% vs GRU base
- Tiempo: +25-35% vs GRU base (aún < LSTM apilada)
- **Mejor candidato para producción** (balance óptimo)

---

## 📊 Tabla de Predicciones Esperadas

### Clasificación

| Modelo | Accuracy (%) | F1-Score (%) | Params (M) | Tiempo/Época (s) | Observaciones |
|--------|--------------|--------------|------------|------------------|---------------|
| RNN Base | 60-75 | 55-70 | ~0.15 | 2-3 | Baseline, gradiente desvaneciente |
| RNN Deep | 65-80 | 60-75 | ~0.25 | 3-4 | +5-10% F1, más profundidad |
| RNN ReLU | 65-78 | 60-73 | ~0.15 | 2-3 | Mejor convergencia |
| LSTM Base | 75-85 | 70-80 | ~0.20 | 4-5 | Buenas dependencias largas |
| LSTM Bi | 78-88 | 75-85 | ~0.40 | 6-8 | +5-7% F1, contexto bidireccional |
| LSTM Stack | 76-87 | 72-82 | ~0.35 | 5-7 | Mejor generalización |
| GRU Base | 74-84 | 68-78 | ~0.15 | 3-4 | Similar a LSTM, más eficiente |
| GRU Bi | 77-87 | 73-83 | ~0.30 | 5-6 | Comparable a LSTM Bi |
| GRU Stack | 76-86 | 72-82 | ~0.25 | 4-5 | **Mejor balance** |

### Regresión

| Modelo | RMSE | R² | Params (M) | Tiempo/Época (s) | Observaciones |
|--------|------|-----|------------|------------------|---------------|
| RNN Base | 0.08-0.12 | 0.75-0.85 | ~0.10 | 2-3 | Baseline |
| RNN Deep | 0.06-0.10 | 0.80-0.90 | ~0.18 | 3-4 | Mejor que base |
| RNN ReLU | 0.07-0.11 | 0.78-0.88 | ~0.10 | 2-3 | Convergencia más rápida |
| LSTM Base | 0.05-0.08 | 0.85-0.92 | ~0.15 | 3-4 | Buena predicción |
| LSTM Bi | 0.04-0.07 | 0.88-0.94 | ~0.30 | 5-6 | Mejor contexto |
| LSTM Stack | 0.04-0.07 | 0.87-0.93 | ~0.25 | 4-5 | Regularizado |
| GRU Base | 0.05-0.09 | 0.84-0.91 | ~0.12 | 3-4 | Eficiente |
| GRU Bi | 0.04-0.08 | 0.86-0.93 | ~0.24 | 4-5 | Balance bueno |
| GRU Stack | 0.04-0.08 | 0.86-0.92 | ~0.20 | 3-4 | **Mejor opción** |

---

## 🔍 Aspectos Clave a Analizar

### 1. Gradiente Desvaneciente en RNN
**¿Cómo verificarlo?**
- Observar norma de gradientes durante entrenamiento
- Comparar convergencia RNN vs LSTM/GRU
- Analizar rendimiento en dependencias largas vs cortas

**Indicadores:**
- RNN plateau temprano en curvas
- LSTM/GRU continúan mejorando
- Gap en F1-Score significativo

### 2. Impacto de Bidireccionalidad
**¿Dónde se nota más?**
- Matriz de confusión: mejor en clases difíciles
- F1-Score por clase
- Análisis de errores (falsos positivos/negativos)

**Esperado:**
- Mejora mayor en fallas leves/incipientes
- Reducción de confusión entre clases similares

### 3. Eficiencia Computacional
**Métricas:**
- Parámetros totales
- Tiempo por época
- Memoria GPU utilizada
- Ratio (F1-Score / Tiempo)

**Esperado:**
- GRU más eficiente que LSTM
- Bidireccionalidad duplica tiempo ~50-70%
- Apilamiento aumenta tiempo ~30-50%

### 4. Overfitting
**Señales:**
- Gap train-val loss > 0.2
- Accuracy train > val por >10%
- Curvas de val comenzando a divergir

**Soluciones implementadas:**
- Dropout en variantes apiladas
- Weight decay (1e-5)
- Early stopping (opcional)

---

## 📝 Guía para el Análisis Final

### Sección de Resultados (III)
1. Presentar tablas con métricas
2. Mostrar curvas de entrenamiento
3. Matrices de confusión de mejores modelos
4. Gráficas de comparación

### Sección de Análisis (IV)

#### 4.1 Validación de Hipótesis
Para cada hipótesis:
- ✅ **Confirmada:** Si predicción ±5% de realidad
- ⚠️ **Parcialmente confirmada:** Si predicción ±10%
- ❌ **Rechazada:** Si predicción >10% errónea

Explicar **por qué** en cada caso.

#### 4.2 Comparación RNN vs LSTM/GRU
Analizar:
- Diferencias en F1-Score
- Curvas de convergencia
- Problema de gradiente desvaneciente (evidencia)

#### 4.3 Comparación LSTM vs GRU
Evaluar:
- Trade-off rendimiento-costo
- Casos donde uno supera al otro
- Recomendación para producción

#### 4.4 Análisis de Errores
Clasificación:
- Clases más confundidas (matriz confusión)
- ¿Fallas leves con sano?
- ¿Confusión entre niveles adyacentes?

Regresión:
- Scatter plot: ¿sesgo en rangos?
- ¿Subestimación/sobreestimación?
- Residuales: ¿patrones?

#### 4.5 Impacto de Variantes
Para cada variante:
- ¿Mejora significativa? (>3%)
- ¿Costo computacional justificado?
- ¿Cuándo usar cada una?

---

## 🎓 Lecciones Aprendidas Esperadas

1. **RNN simple:** Limitada para dependencias largas
2. **LSTM:** Excelente pero costosa
3. **GRU:** Mejor balance rendimiento-costo
4. **Bidireccionalidad:** Útil cuando hay contexto futuro disponible
5. **Profundidad:** Rendimiento decreciente, regularización crítica
6. **Dropout:** Esencial para generalización
7. **Normalización:** Crítica para convergencia
8. **Ventanas:** Tamaño importante (64 parece adecuado)

---

## 🚀 Próximos Pasos (Opcional)

1. **Attention mechanisms:** LSTM/GRU con attention
2. **Ensemble:** Combinar predicciones de múltiples modelos
3. **Arquitecturas híbridas:** CNN + LSTM para señales
4. **Transfer learning:** Pre-entrenar en señales similares
5. **Optimización:** Búsqueda de hiperparámetros automática

---

**Autor:** Implementación académica para Aprendizaje Profundo  
**Fecha:** Noviembre 2025
