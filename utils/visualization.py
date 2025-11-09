"""
Utilidades para visualización de resultados.

Este módulo contiene funciones para crear gráficas de:
- Curvas de entrenamiento (pérdida y métricas)
- Matrices de confusión
- Predicciones vs valores reales (regresión)
- Comparaciones entre modelos
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
import os


# Configurar estilo de matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# VISUALIZACIÓN DE CURVAS DE ENTRENAMIENTO
# ============================================================================

def plot_training_history(history, task='classification', save_path=None, model_name=''):
    """
    Grafica el historial de entrenamiento.
    
    Args:
        history: diccionario con 'train_loss', 'val_loss', 'train_metric', 'val_metric'
        task: 'classification' o 'regression'
        save_path: ruta para guardar la figura (opcional)
        model_name: nombre del modelo para el título
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    if task == 'classification':
        # Figura con 3 subplots para clasificación
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Pérdida
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Época', fontsize=12)
        axes[0].set_ylabel('Pérdida (Cross-Entropy)', fontsize=12)
        axes[0].set_title(f'{model_name} - Curva de Pérdida', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # 2. Accuracy
        axes[1].plot(epochs, history['train_metric'], 'b-', label='Train Acc', linewidth=2)
        axes[1].plot(epochs, history['val_metric'], 'r-', label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Época', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        # 3. F1-Score (si está disponible)
        if history['val_additional'] and 'f1_score' in history['val_additional'][0]:
            val_f1 = [item['f1_score'] for item in history['val_additional']]
            axes[2].plot(epochs, val_f1, 'g-', label='Val F1-Score', linewidth=2)
            axes[2].set_xlabel('Época', fontsize=12)
            axes[2].set_ylabel('F1-Score (%)', fontsize=12)
            axes[2].set_title(f'{model_name} - F1-Score', fontsize=14, fontweight='bold')
            axes[2].legend(fontsize=11)
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].axis('off')
        
    else:  # regression
        # Figura con 2 subplots para regresión
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. Pérdida (MSE)
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss (MSE)', linewidth=2)
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss (MSE)', linewidth=2)
        axes[0].set_xlabel('Época', fontsize=12)
        axes[0].set_ylabel('Pérdida (MSE)', fontsize=12)
        axes[0].set_title(f'{model_name} - Curva de Pérdida', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # 2. RMSE
        axes[1].plot(epochs, history['val_metric'], 'g-', label='Val RMSE', linewidth=2)
        axes[1].set_xlabel('Época', fontsize=12)
        axes[1].set_ylabel('RMSE', fontsize=12)
        axes[1].set_title(f'{model_name} - RMSE en Validación', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Gráfica guardada: {save_path}")
    
    plt.show()


# ============================================================================
# MATRIZ DE CONFUSIÓN
# ============================================================================

def plot_confusion_matrix(cm, class_names, save_path=None, model_name='', normalize=True):
    """
    Grafica la matriz de confusión.
    
    Args:
        cm: matriz de confusión (numpy array)
        class_names: nombres de las clases
        save_path: ruta para guardar la figura (opcional)
        model_name: nombre del modelo para el título
        normalize: si True, normaliza la matriz
    """
    plt.figure(figsize=(10, 8))
    
    if normalize:
        # Normalizar por filas (verdaderas clases)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # Reemplazar NaN con 0
        
        # Crear display
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_names)
        disp.plot(cmap='Blues', ax=plt.gca(), xticks_rotation=45, values_format='.2f')
        plt.title(f'{model_name} - Matriz de Confusión (Normalizada)', 
                 fontsize=14, fontweight='bold', pad=20)
    else:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap='Blues', ax=plt.gca(), xticks_rotation=45)
        plt.title(f'{model_name} - Matriz de Confusión', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Matriz de confusión guardada: {save_path}")
    
    plt.show()


# ============================================================================
# VISUALIZACIONES PARA REGRESIÓN
# ============================================================================

def plot_predictions_vs_actual(y_true, y_pred, save_path=None, model_name=''):
    """
    Grafica predicciones vs valores reales (scatter plot).
    
    Args:
        y_true: valores reales
        y_pred: valores predichos
        save_path: ruta para guardar la figura (opcional)
        model_name: nombre del modelo
    """
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
    
    # Línea de predicción perfecta (y = x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicción Perfecta')
    
    # Calcular R²
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    
    plt.xlabel('Valores Reales', fontsize=12)
    plt.ylabel('Valores Predichos', fontsize=12)
    plt.title(f'{model_name} - Predicciones vs Reales (R² = {r2:.4f})', 
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Gráfica de predicciones guardada: {save_path}")
    
    plt.show()


def plot_time_series_prediction(y_true, y_pred, n_samples=200, save_path=None, model_name=''):
    """
    Grafica una secuencia temporal de predicciones vs valores reales.
    
    Args:
        y_true: valores reales
        y_pred: valores predichos
        n_samples: número de muestras a graficar
        save_path: ruta para guardar la figura (opcional)
        model_name: nombre del modelo
    """
    n = min(n_samples, len(y_true))
    
    plt.figure(figsize=(15, 6))
    
    time_steps = range(n)
    plt.plot(time_steps, y_true[:n], 'b-', label='Real', linewidth=2, alpha=0.7)
    plt.plot(time_steps, y_pred[:n], 'r--', label='Predicho', linewidth=2, alpha=0.7)
    
    plt.xlabel('Paso Temporal', fontsize=12)
    plt.ylabel('Valor', fontsize=12)
    plt.title(f'{model_name} - Predicción de Serie Temporal', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Gráfica de serie temporal guardada: {save_path}")
    
    plt.show()


# ============================================================================
# COMPARACIÓN DE MODELOS
# ============================================================================

def plot_model_comparison(results_dict, task='classification', save_path=None):
    """
    Compara múltiples modelos en una gráfica de barras.
    
    Args:
        results_dict: diccionario {model_name: metrics_dict}
        task: 'classification' o 'regression'
        save_path: ruta para guardar la figura (opcional)
    """
    model_names = list(results_dict.keys())
    
    if task == 'classification':
        # Extraer métricas
        accuracies = [results_dict[name]['accuracy'] * 100 for name in model_names]
        f1_scores = [results_dict[name]['f1_macro'] * 100 for name in model_names]
        
        # Crear figura
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Accuracy
        bars1 = ax1.bar(model_names, accuracies, color='skyblue', alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_title('Comparación de Accuracy', fontsize=14, fontweight='bold')
        ax1.set_ylim([0, 100])
        ax1.grid(True, alpha=0.3, axis='y')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Añadir valores en las barras
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%', ha='center', va='bottom', fontsize=10)
        
        # F1-Score
        bars2 = ax2.bar(model_names, f1_scores, color='lightgreen', alpha=0.8, edgecolor='black')
        ax2.set_ylabel('F1-Score (macro) (%)', fontsize=12)
        ax2.set_title('Comparación de F1-Score', fontsize=14, fontweight='bold')
        ax2.set_ylim([0, 100])
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Añadir valores en las barras
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%', ha='center', va='bottom', fontsize=10)
    
    else:  # regression
        # Extraer métricas
        rmses = [results_dict[name]['rmse'] for name in model_names]
        r2s = [results_dict[name]['r2'] for name in model_names]
        
        # Crear figura
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # RMSE
        bars1 = ax1.bar(model_names, rmses, color='salmon', alpha=0.8, edgecolor='black')
        ax1.set_ylabel('RMSE', fontsize=12)
        ax1.set_title('Comparación de RMSE (menor es mejor)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)
        
        # R²
        bars2 = ax2.bar(model_names, r2s, color='lightblue', alpha=0.8, edgecolor='black')
        ax2.set_ylabel('R² Score', fontsize=12)
        ax2.set_title('Comparación de R² (mayor es mejor)', fontsize=14, fontweight='bold')
        ax2.set_ylim([-0.1, 1.1])
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Comparación de modelos guardada: {save_path}")
    
    plt.show()


# ============================================================================
# GUARDAR TABLAS DE RESULTADOS
# ============================================================================

def save_results_table(results_dict, save_path, task='classification'):
    """
    Guarda una tabla de resultados en formato CSV y LaTeX.
    
    Args:
        results_dict: diccionario {model_name: metrics_dict}
        save_path: ruta base para guardar (sin extensión)
        task: 'classification' o 'regression'
    """
    if task == 'classification':
        # Crear DataFrame
        data = []
        for model_name, metrics in results_dict.items():
            row = {
                'Modelo': model_name,
                'Accuracy (%)': metrics.get('accuracy', 0) * 100,
                'F1-Score (%)': metrics.get('f1_macro', 0) * 100,
                'Parámetros (M)': metrics.get('params_millions', 0),
                'Tiempo/Época (s)': metrics.get('time_per_epoch', 0)
            }
            data.append(row)
        
        df = pd.DataFrame(data)
    
    else:  # regression
        # Crear DataFrame
        data = []
        for model_name, metrics in results_dict.items():
            row = {
                'Modelo': model_name,
                'MSE': metrics.get('mse', 0),
                'RMSE': metrics.get('rmse', 0),
                'MAE': metrics.get('mae', 0),
                'R²': metrics.get('r2', 0),
                'Parámetros (M)': metrics.get('params_millions', 0),
                'Tiempo/Época (s)': metrics.get('time_per_epoch', 0)
            }
            data.append(row)
        
        df = pd.DataFrame(data)
    
    # Guardar CSV
    csv_path = f"{save_path}.csv"
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"📄 Tabla CSV guardada: {csv_path}")
    
    # Guardar LaTeX
    latex_path = f"{save_path}.tex"
    latex_str = df.to_latex(index=False, float_format='%.4f')
    with open(latex_path, 'w') as f:
        f.write(latex_str)
    print(f"📄 Tabla LaTeX guardada: {latex_path}")
    
    # Mostrar tabla
    print("\n" + "="*70)
    print("TABLA DE RESULTADOS")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70 + "\n")
    
    return df


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("TESTING VISUALIZATION")
    print("="*70)
    
    # Test de curvas de entrenamiento
    print("\nTest 1: Plot training history (clasificación)")
    history_classification = {
        'train_loss': np.random.rand(50) * 2,
        'val_loss': np.random.rand(50) * 2,
        'train_metric': 50 + np.random.rand(50) * 40,
        'val_metric': 50 + np.random.rand(50) * 40,
        'val_additional': [{'f1_score': 50 + np.random.rand() * 40} for _ in range(50)]
    }
    
    plot_training_history(history_classification, task='classification', model_name='LSTM Test')
    
    print("\n" + "="*70)
    print("TESTS COMPLETADOS")
    print("="*70)
