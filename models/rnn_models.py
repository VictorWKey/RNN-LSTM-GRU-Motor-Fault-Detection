"""
Implementación de modelos RNN, LSTM y GRU para clasificación y regresión de señales.

Arquitecturas base y variantes según los requisitos de la práctica.
"""
import torch
import torch.nn as nn


# ============================================================================
# RNN MODELS - CLASIFICACIÓN
# ============================================================================

class RNNClassifier(nn.Module):
    """
    RNN simple (Elman) para clasificación de señales de motor.
    
    Args:
        input_size: dimensión de entrada (número de features)
        hidden_size: tamaño del estado oculto
        num_layers: número de capas recurrentes
        num_classes: número de clases de salida
        nonlinearity: 'tanh' o 'relu'
        bidirectional: si es bidireccional
        dropout: probabilidad de dropout (solo si num_layers > 1)
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, 
                 nonlinearity='tanh', bidirectional=False, dropout=0.0):
        super(RNNClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Capa RNN
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Capa totalmente conectada
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: tensor de entrada (batch_size, seq_len, input_size)
        
        Returns:
            output: logits (batch_size, num_classes)
        """
        # x: (batch, seq_len, input_size)
        # out: (batch, seq_len, hidden_size * num_directions)
        # h_n: (num_layers * num_directions, batch, hidden_size)
        out, h_n = self.rnn(x)
        
        # Usar el último estado oculto
        if self.bidirectional:
            # Concatenar forward y backward del último layer
            h_final = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            h_final = h_n[-1, :, :]  # Último layer
        
        # Capa FC
        output = self.fc(h_final)
        
        return output


# ============================================================================
# RNN MODELS - REGRESIÓN
# ============================================================================

class RNNRegressor(nn.Module):
    """
    RNN simple para regresión de series temporales.
    
    Args:
        input_size: dimensión de entrada
        hidden_size: tamaño del estado oculto
        num_layers: número de capas recurrentes
        output_size: dimensión de salida (horizonte de predicción)
        nonlinearity: 'tanh' o 'relu'
        bidirectional: si es bidireccional
        dropout: probabilidad de dropout
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size=1,
                 nonlinearity='tanh', bidirectional=False, dropout=0.0):
        super(RNNRegressor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Capa RNN
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Capa totalmente conectada
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, output_size)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: tensor de entrada (batch_size, seq_len, input_size)
        
        Returns:
            output: predicción (batch_size, output_size)
        """
        out, h_n = self.rnn(x)
        
        # Usar el último estado oculto
        if self.bidirectional:
            h_final = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            h_final = h_n[-1, :, :]
        
        # Predicción
        output = self.fc(h_final)
        
        return output


# ============================================================================
# LSTM MODELS - CLASIFICACIÓN
# ============================================================================

class LSTMClassifier(nn.Module):
    """
    LSTM para clasificación de señales de motor.
    
    Args:
        input_size: dimensión de entrada
        hidden_size: tamaño del estado oculto
        num_layers: número de capas LSTM
        num_classes: número de clases
        bidirectional: si es bidireccional
        dropout: probabilidad de dropout recurrente
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 bidirectional=False, dropout=0.0):
        super(LSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Capa LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Capa totalmente conectada
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: tensor de entrada (batch_size, seq_len, input_size)
        
        Returns:
            output: logits (batch_size, num_classes)
        """
        # out: (batch, seq_len, hidden_size * num_directions)
        # h_n: (num_layers * num_directions, batch, hidden_size)
        # c_n: (num_layers * num_directions, batch, hidden_size)
        out, (h_n, c_n) = self.lstm(x)
        
        # Usar el último estado oculto
        if self.bidirectional:
            # Concatenar forward y backward del último layer
            h_final = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            h_final = h_n[-1, :, :]
        
        # Capa FC
        output = self.fc(h_final)
        
        return output


# ============================================================================
# LSTM MODELS - REGRESIÓN
# ============================================================================

class LSTMRegressor(nn.Module):
    """
    LSTM para regresión de series temporales.
    
    Args:
        input_size: dimensión de entrada
        hidden_size: tamaño del estado oculto
        num_layers: número de capas LSTM
        output_size: dimensión de salida
        bidirectional: si es bidireccional
        dropout: probabilidad de dropout recurrente
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size=1,
                 bidirectional=False, dropout=0.0):
        super(LSTMRegressor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Capa LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Capa totalmente conectada
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, output_size)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: tensor de entrada (batch_size, seq_len, input_size)
        
        Returns:
            output: predicción (batch_size, output_size)
        """
        out, (h_n, c_n) = self.lstm(x)
        
        # Usar el último estado oculto
        if self.bidirectional:
            h_final = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            h_final = h_n[-1, :, :]
        
        # Predicción
        output = self.fc(h_final)
        
        return output


# ============================================================================
# GRU MODELS - CLASIFICACIÓN
# ============================================================================

class GRUClassifier(nn.Module):
    """
    GRU para clasificación de señales de motor.
    
    Args:
        input_size: dimensión de entrada
        hidden_size: tamaño del estado oculto
        num_layers: número de capas GRU
        num_classes: número de clases
        bidirectional: si es bidireccional
        dropout: probabilidad de dropout recurrente
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 bidirectional=False, dropout=0.0):
        super(GRUClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Capa GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Capa totalmente conectada
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: tensor de entrada (batch_size, seq_len, input_size)
        
        Returns:
            output: logits (batch_size, num_classes)
        """
        # out: (batch, seq_len, hidden_size * num_directions)
        # h_n: (num_layers * num_directions, batch, hidden_size)
        out, h_n = self.gru(x)
        
        # Usar el último estado oculto
        if self.bidirectional:
            # Concatenar forward y backward del último layer
            h_final = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            h_final = h_n[-1, :, :]
        
        # Capa FC
        output = self.fc(h_final)
        
        return output


# ============================================================================
# GRU MODELS - REGRESIÓN
# ============================================================================

class GRURegressor(nn.Module):
    """
    GRU para regresión de series temporales.
    
    Args:
        input_size: dimensión de entrada
        hidden_size: tamaño del estado oculto
        num_layers: número de capas GRU
        output_size: dimensión de salida
        bidirectional: si es bidireccional
        dropout: probabilidad de dropout recurrente
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size=1,
                 bidirectional=False, dropout=0.0):
        super(GRURegressor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Capa GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Capa totalmente conectada
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, output_size)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: tensor de entrada (batch_size, seq_len, input_size)
        
        Returns:
            output: predicción (batch_size, output_size)
        """
        out, h_n = self.gru(x)
        
        # Usar el último estado oculto
        if self.bidirectional:
            h_final = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            h_final = h_n[-1, :, :]
        
        # Predicción
        output = self.fc(h_final)
        
        return output


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def count_parameters(model):
    """
    Cuenta el número total de parámetros entrenables en un modelo.
    
    Args:
        model: modelo de PyTorch
    
    Returns:
        total_params: número total de parámetros
        trainable_params: número de parámetros entrenables
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def get_model_info(model):
    """
    Obtiene información detallada del modelo.
    
    Args:
        model: modelo de PyTorch
    
    Returns:
        info: diccionario con información del modelo
    """
    total_params, trainable_params = count_parameters(model)
    
    info = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'params_millions': total_params / 1e6,
        'model_name': model.__class__.__name__
    }
    
    return info


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("TESTING MODELOS RNN")
    print("="*70)
    
    # Parámetros de prueba
    batch_size = 32
    seq_len = 64
    input_size = 3
    hidden_size = 64
    num_layers = 2
    num_classes = 5
    output_size = 1
    
    # Crear datos de prueba
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Probar RNN Classifier
    print("\n1. RNNClassifier (base)")
    model = RNNClassifier(input_size, hidden_size, num_layers, num_classes)
    output = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Parámetros: {count_parameters(model)[0]:,}")
    
    # Probar LSTM Classifier Bidireccional
    print("\n2. LSTMClassifier (bidirectional)")
    model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes, bidirectional=True)
    output = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Parámetros: {count_parameters(model)[0]:,}")
    
    # Probar GRU Regressor
    print("\n3. GRURegressor (stacked)")
    model = GRURegressor(input_size, hidden_size, num_layers, output_size, dropout=0.2)
    output = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Parámetros: {count_parameters(model)[0]:,}")
    
    print("\n" + "="*70)
    print("TESTS COMPLETADOS EXITOSAMENTE")
    print("="*70)
