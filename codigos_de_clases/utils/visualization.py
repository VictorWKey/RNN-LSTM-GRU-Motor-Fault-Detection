"""
Utilidades para visualización de resultados.
"""
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np


def plot_training_history(history, save_path=None):
    """
    Grafica el historial de entrenamiento (pérdida y accuracy).
    
    Args:
        history: diccionario con 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: ruta para guardar la figura (opcional)
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Determinar si es clasificación o regresión
    is_classification = history['train_acc'][0] is not None
    
    if is_classification:
        # Crear figura con 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Pérdida
        axes[0].plot(epochs, history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(epochs, history['val_loss'], label='Val Loss', marker='s')
        axes[0].set_xlabel('Época')
        axes[0].set_ylabel('Pérdida')
        axes[0].set_title('Historial de Pérdida (Loss)')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy
        axes[1].plot(epochs, history['train_acc'], label='Train Acc', marker='o')
        axes[1].plot(epochs, history['val_acc'], label='Val Acc', marker='s')
        axes[1].set_xlabel('Época')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Historial de Precisión (Accuracy)')
        axes[1].legend()
        axes[1].grid(True)
    else:
        # Solo pérdida para regresión
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
        plt.plot(epochs, history['val_loss'], label='Val Loss', marker='s')
        plt.xlabel('Época')
        plt.ylabel('Pérdida (MSE)')
        plt.title('Historial de Pérdida durante el Entrenamiento')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Grafica la matriz de confusión.
    
    Args:
        cm: matriz de confusión
        class_names: nombres de las clases
        save_path: ruta para guardar la figura (opcional)
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    plt.title("Matriz de Confusión - Conjunto de Prueba")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_predictions_vs_actual(y_true, y_pred, save_path=None):
    """
    Grafica predicciones vs valores reales para regresión.
    
    Args:
        y_true: valores reales
        y_pred: valores predichos
        save_path: ruta para guardar la figura (opcional)
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Línea ideal (y=x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicción Perfecta')
    
    plt.xlabel('Valores Reales')
    plt.ylabel('Valores Predichos')
    plt.title('Predicciones vs Valores Reales')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_time_series_prediction(y_true, y_pred, n_samples=200, save_path=None):
    """
    Grafica una secuencia de predicciones vs valores reales.
    
    Args:
        y_true: valores reales
        y_pred: valores predichos
        n_samples: número de muestras a graficar
        save_path: ruta para guardar la figura (opcional)
    """
    n = min(n_samples, len(y_true))
    
    plt.figure(figsize=(14, 5))
    plt.plot(range(n), y_true[:n], label='Real', marker='o', markersize=3)
    plt.plot(range(n), y_pred[:n], label='Predicho', marker='x', markersize=3)
    plt.xlabel('Muestra')
    plt.ylabel('Valor')
    plt.title('Predicción de Series Temporales')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
