"""
Utilidades para entrenamiento y evaluación de modelos RNN.

Este módulo contiene funciones para:
- Entrenar modelos por época
- Validar modelos
- Calcular métricas de clasificación y regresión
- Guardar y cargar checkpoints
"""
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, mean_squared_error,
    mean_absolute_error, r2_score
)
import time
import json
import os


# ============================================================================
# FUNCIONES DE ENTRENAMIENTO
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, clip_norm=1.0, task='classification'):
    """
    Entrena el modelo por una época.
    
    Args:
        model: modelo de PyTorch
        dataloader: DataLoader de entrenamiento
        criterion: función de pérdida
        optimizer: optimizador
        device: dispositivo (cpu/cuda)
        clip_norm: valor para gradient clipping
        task: 'classification' o 'regression'
    
    Returns:
        avg_loss: pérdida promedio
        avg_metric: accuracy (clasificación) o None (regresión)
    """
    model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
        # Mover datos al dispositivo
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(X_batch)
        
        # Calcular pérdida
        if task == 'regression':
            # Para regresión, asegurar que las dimensiones coincidan
            if outputs.dim() == 2 and y_batch.dim() == 2:
                loss = criterion(outputs, y_batch)
            elif outputs.dim() == 2 and y_batch.dim() == 1:
                loss = criterion(outputs.squeeze(), y_batch)
            else:
                loss = criterion(outputs, y_batch)
        else:
            # Para clasificación
            loss = criterion(outputs, y_batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (evita explosión de gradientes)
        if clip_norm > 0:
            clip_grad_norm_(model.parameters(), clip_norm)
        
        # Actualizar parámetros
        optimizer.step()
        
        # Acumular pérdida
        total_loss += loss.item() * X_batch.size(0)
        
        # Calcular accuracy si es clasificación
        if task == 'classification':
            _, predicted = torch.max(outputs, 1)
            total_samples += y_batch.size(0)
            total_correct += (predicted == y_batch).sum().item()
    
    # Promedios
    avg_loss = total_loss / len(dataloader.dataset)
    avg_metric = (total_correct / total_samples * 100) if task == 'classification' else None
    
    return avg_loss, avg_metric


def validate_epoch(model, dataloader, criterion, device, task='classification'):
    """
    Evalúa el modelo en el conjunto de validación.
    
    Args:
        model: modelo de PyTorch
        dataloader: DataLoader de validación
        criterion: función de pérdida
        device: dispositivo (cpu/cuda)
        task: 'classification' o 'regression'
    
    Returns:
        avg_loss: pérdida promedio
        avg_metric: accuracy/F1 (clasificación) o RMSE (regresión)
        additional_metrics: métricas adicionales
    """
    model.eval()
    
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch)
            
            # Calcular pérdida
            if task == 'regression':
                if outputs.dim() == 2 and y_batch.dim() == 2:
                    loss = criterion(outputs, y_batch)
                elif outputs.dim() == 2 and y_batch.dim() == 1:
                    loss = criterion(outputs.squeeze(), y_batch)
                else:
                    loss = criterion(outputs, y_batch)
            else:
                loss = criterion(outputs, y_batch)
            
            total_loss += loss.item() * X_batch.size(0)
            
            # Guardar predicciones y targets
            if task == 'classification':
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
            else:
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
    
    # Calcular métricas
    avg_loss = total_loss / len(dataloader.dataset)
    
    if task == 'classification':
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        accuracy = accuracy_score(all_targets, all_predictions) * 100
        f1 = f1_score(all_targets, all_predictions, average='macro') * 100
        
        avg_metric = accuracy
        additional_metrics = {'f1_score': f1}
    else:
        all_predictions = np.array(all_predictions).flatten()
        all_targets = np.array(all_targets).flatten()
        
        mse = mean_squared_error(all_targets, all_predictions)
        rmse = np.sqrt(mse)
        
        avg_metric = rmse
        additional_metrics = {'mse': mse}
    
    return avg_loss, avg_metric, additional_metrics


def train_model(model, train_loader, val_loader, criterion, optimizer,
                epochs, device, clip_norm=1.0, task='classification',
                save_path=None, patience=None, verbose=True):
    """
    Entrena el modelo completo con validación y early stopping opcional.
    
    Args:
        model: modelo de PyTorch
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validación
        criterion: función de pérdida
        optimizer: optimizador
        epochs: número de épocas
        device: dispositivo (cpu/cuda)
        clip_norm: valor para gradient clipping
        task: 'classification' o 'regression'
        save_path: ruta para guardar el mejor modelo
        patience: épocas de paciencia para early stopping (None = sin early stopping)
        verbose: imprimir progreso
    
    Returns:
        history: diccionario con historial de entrenamiento
        best_model_state: estado del mejor modelo
        training_time: tiempo total de entrenamiento
    """
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metric': [],
        'val_metric': [],
        'val_additional': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Entrenar
        train_loss, train_metric = train_epoch(
            model, train_loader, criterion, optimizer, device, clip_norm, task
        )
        
        # Validar
        val_loss, val_metric, val_additional = validate_epoch(
            model, val_loader, criterion, device, task
        )
        
        # Guardar historial
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_metric'].append(train_metric)
        history['val_metric'].append(val_metric)
        history['val_additional'].append(val_additional)
        
        # Verificar si es el mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            
            # Guardar checkpoint si se especificó ruta
            if save_path:
                save_checkpoint(model, optimizer, epoch, val_loss, save_path, history)
        else:
            epochs_without_improvement += 1
        
        epoch_time = time.time() - epoch_start
        
        # Imprimir progreso
        if verbose and (epoch % 10 == 0 or epoch == 1):
            if task == 'classification':
                f1 = val_additional.get('f1_score', 0)
                print(f'Época {epoch:3d}/{epochs}: '
                      f'Train Loss={train_loss:.4f}, Train Acc={train_metric:.2f}% | '
                      f'Val Loss={val_loss:.4f}, Val Acc={val_metric:.2f}%, Val F1={f1:.2f}% | '
                      f'Time={epoch_time:.2f}s')
            else:
                mse = val_additional.get('mse', 0)
                print(f'Época {epoch:3d}/{epochs}: '
                      f'Train Loss={train_loss:.4f} | '
                      f'Val Loss={val_loss:.4f}, Val RMSE={val_metric:.4f}, Val MSE={mse:.4f} | '
                      f'Time={epoch_time:.2f}s')
        
        # Early stopping
        if patience is not None and epochs_without_improvement >= patience:
            if verbose:
                print(f'\n⚠️  Early stopping activado en época {epoch}')
                print(f'   No hubo mejora en {patience} épocas consecutivas')
            break
    
    total_time = time.time() - start_time
    
    if verbose:
        print(f'\n✅ Entrenamiento completado en {total_time:.2f}s ({total_time/60:.2f} min)')
        print(f'   Mejor val loss: {best_val_loss:.4f}')
    
    return history, best_model_state, total_time


# ============================================================================
# MÉTRICAS DE EVALUACIÓN
# ============================================================================

def compute_classification_metrics(model, dataloader, device, class_names=None):
    """
    Calcula métricas detalladas de clasificación.
    
    Args:
        model: modelo de PyTorch
        dataloader: DataLoader de evaluación
        device: dispositivo (cpu/cuda)
        class_names: nombres de las clases
    
    Returns:
        metrics: diccionario con todas las métricas
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Calcular métricas
    accuracy = accuracy_score(all_targets, all_predictions)
    f1_macro = f1_score(all_targets, all_predictions, average='macro')
    f1_weighted = f1_score(all_targets, all_predictions, average='weighted')
    conf_matrix = confusion_matrix(all_targets, all_predictions)
    
    if class_names is not None:
        report = classification_report(all_targets, all_predictions, target_names=class_names)
    else:
        report = classification_report(all_targets, all_predictions)
    
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': conf_matrix,
        'classification_report': report,
        'predictions': all_predictions,
        'targets': all_targets
    }
    
    return metrics


def compute_regression_metrics(model, dataloader, device):
    """
    Calcula métricas detalladas de regresión.
    
    Args:
        model: modelo de PyTorch
        dataloader: DataLoader de evaluación
        device: dispositivo (cpu/cuda)
    
    Returns:
        metrics: diccionario con todas las métricas
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    
    all_predictions = np.array(all_predictions).flatten()
    all_targets = np.array(all_targets).flatten()
    
    # Calcular métricas
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': all_predictions,
        'targets': all_targets
    }
    
    return metrics


# ============================================================================
# GUARDADO Y CARGA DE CHECKPOINTS
# ============================================================================

def save_checkpoint(model, optimizer, epoch, val_loss, save_path, history=None):
    """
    Guarda un checkpoint del modelo.
    
    Args:
        model: modelo de PyTorch
        optimizer: optimizador
        epoch: época actual
        val_loss: pérdida de validación
        save_path: ruta donde guardar el checkpoint
        history: historial de entrenamiento (opcional)
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'model_name': model.__class__.__name__
    }
    
    if history is not None:
        checkpoint['history'] = history
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save(checkpoint, save_path)
    

def load_checkpoint(model, checkpoint_path, optimizer=None, device='cpu'):
    """
    Carga un checkpoint del modelo.
    
    Args:
        model: modelo de PyTorch (arquitectura ya creada)
        checkpoint_path: ruta al checkpoint
        optimizer: optimizador (opcional)
        device: dispositivo donde cargar el modelo
    
    Returns:
        epoch: época del checkpoint
        val_loss: pérdida de validación del checkpoint
        history: historial de entrenamiento (si está disponible)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', None)
    val_loss = checkpoint.get('val_loss', None)
    history = checkpoint.get('history', None)
    
    print(f"✅ Checkpoint cargado desde: {checkpoint_path}")
    print(f"   Modelo: {checkpoint.get('model_name', 'Unknown')}")
    print(f"   Época: {epoch}")
    print(f"   Val Loss: {val_loss:.4f}")
    
    return epoch, val_loss, history


def count_parameters(model):
    """
    Cuenta el número de parámetros en el modelo.
    
    Args:
        model: modelo de PyTorch
    
    Returns:
        total_params: total de parámetros
        trainable_params: parámetros entrenables
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("TESTING TRAINING_UTILS")
    print("="*70)
    
    # Test de contador de parámetros
    print("\nTest 1: Contar parámetros")
    from models.rnn_models import LSTMClassifier
    
    model = LSTMClassifier(input_size=3, hidden_size=64, num_layers=2, num_classes=5)
    total, trainable = count_parameters(model)
    print(f"Total parámetros: {total:,}")
    print(f"Parámetros entrenables: {trainable:,}")
    print(f"Parámetros en millones: {total/1e6:.4f}M")
    
    print("\n" + "="*70)
    print("TESTS COMPLETADOS")
    print("="*70)
