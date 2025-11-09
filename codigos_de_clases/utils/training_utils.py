"""
Funciones de entrenamiento y evaluación de modelos.
"""
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


def train_epoch(model, dataloader, criterion, optimizer, device, clip_norm=1.0):
    """
    Entrena el modelo por una época.
    
    Args:
        model: modelo de PyTorch
        dataloader: DataLoader de entrenamiento
        criterion: función de pérdida
        optimizer: optimizador
        device: dispositivo (cpu/cuda)
        clip_norm: valor para gradient clipping
    
    Returns:
        loss_avg: pérdida promedio
        accuracy: precisión (para clasificación) o None (para regresión)
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    is_classification = isinstance(criterion, nn.CrossEntropyLoss)
    
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        
        if clip_norm > 0:
            clip_grad_norm_(model.parameters(), clip_norm)
        
        optimizer.step()
        
        total_loss += loss.item() * xb.size(0)
        
        if is_classification:
            _, predicted = torch.max(out, 1)
            total_samples += yb.size(0)
            total_correct += (predicted == yb).sum().item()
    
    loss_avg = total_loss / len(dataloader.dataset)
    accuracy = (total_correct / total_samples) * 100 if is_classification else None
    
    return loss_avg, accuracy


def evaluate(model, dataloader, criterion, device):
    """
    Evalúa el modelo en un conjunto de datos.
    
    Args:
        model: modelo de PyTorch
        dataloader: DataLoader de evaluación
        criterion: función de pérdida
        device: dispositivo (cpu/cuda)
    
    Returns:
        loss_avg: pérdida promedio
        accuracy: precisión (para clasificación) o None (para regresión)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    is_classification = isinstance(criterion, nn.CrossEntropyLoss)
    
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            
            total_loss += loss.item() * xb.size(0)
            
            if is_classification:
                _, predicted = torch.max(out, 1)
                total_samples += yb.size(0)
                total_correct += (predicted == yb).sum().item()
    
    loss_avg = total_loss / len(dataloader.dataset)
    accuracy = (total_correct / total_samples) * 100 if is_classification else None
    
    return loss_avg, accuracy


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                epochs, device, clip_norm=1.0, verbose=True):
    """
    Entrena el modelo completo con early stopping basado en pérdida de validación.
    
    Args:
        model: modelo de PyTorch
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validación
        criterion: función de pérdida
        optimizer: optimizador
        epochs: número de épocas
        device: dispositivo (cpu/cuda)
        clip_norm: valor para gradient clipping
        verbose: imprimir progreso
    
    Returns:
        history: diccionario con historial de entrenamiento
        best_state: mejor estado del modelo
    """
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(1, epochs + 1):
        # Entrenamiento
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, clip_norm
        )
        
        # Validación
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Guardar historial
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        # Imprimir progreso
        if verbose:
            if train_acc is not None:  # Clasificación
                print(f'Epoch {epoch}/{epochs}: '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            else:  # Regresión
                print(f'Epoch {epoch}/{epochs}: '
                      f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
    
    return history, best_state


def compute_classification_metrics(model, dataloader, device, class_names=None):
    """
    Calcula métricas de clasificación detalladas.
    
    Args:
        model: modelo de PyTorch
        dataloader: DataLoader de evaluación
        device: dispositivo (cpu/cuda)
        class_names: nombres de las clases
    
    Returns:
        metrics: diccionario con accuracy, f1, confusion_matrix y report
    """
    model.eval()
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1).cpu().numpy()
            all_y_pred.append(pred)
            all_y_true.append(yb.numpy())
    
    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is not None:
        report = classification_report(y_true, y_pred, target_names=class_names)
    else:
        report = classification_report(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': report
    }
