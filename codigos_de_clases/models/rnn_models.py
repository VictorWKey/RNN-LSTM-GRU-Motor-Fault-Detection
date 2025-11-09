"""
Modelos RNN, LSTM y GRU para clasificación y regresión de señales.
"""
import torch
import torch.nn as nn


class RNNSimple(nn.Module):
    """RNN simple para regresión de series temporales."""
    
    def __init__(self, in_dim, hidden=128, layers=2, out_dim=1, nonlinearity='tanh'):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=layers,
            nonlinearity=nonlinearity,
            batch_first=True
        )
        self.fc = nn.Linear(hidden, out_dim)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, in_dim)
        Returns:
            out: (batch, out_dim)
        """
        out, h_n = self.rnn(x)
        # Usar el último estado oculto
        h_final = h_n[-1]  # (batch, hidden)
        yhat = self.fc(h_final)
        return yhat


class BiRNNClassifier(nn.Module):
    """RNN Bidireccional para clasificación."""
    
    def __init__(self, in_dim, hidden=128, layers=2, n_classes=5, nonlinearity='relu'):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=layers,
            nonlinearity=nonlinearity,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden * 2, n_classes)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, in_dim)
        Returns:
            out: (batch, n_classes)
        """
        out, h_n = self.rnn(x)
        # Concatenar los estados finales forward y backward
        h_final = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        yhat = self.fc(h_final)
        return yhat


class LSTMClassifier(nn.Module):
    """LSTM para clasificación de señales."""
    
    def __init__(self, in_dim, hidden=128, layers=2, n_classes=5, bidirectional=False, dropout=0.0):
        super().__init__()
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if layers > 1 else 0.0
        )
        
        fc_input_dim = hidden * 2 if bidirectional else hidden
        self.fc = nn.Linear(fc_input_dim, n_classes)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, in_dim)
        Returns:
            out: (batch, n_classes)
        """
        out, (h_n, c_n) = self.lstm(x)
        
        if self.bidirectional:
            # Concatenar forward y backward del último layer
            h_final = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            h_final = h_n[-1]  # Último layer
        
        yhat = self.fc(h_final)
        return yhat


class LSTMRegressor(nn.Module):
    """LSTM para regresión de series temporales."""
    
    def __init__(self, in_dim, hidden=128, layers=2, out_dim=1, bidirectional=False, dropout=0.0):
        super().__init__()
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if layers > 1 else 0.0
        )
        
        fc_input_dim = hidden * 2 if bidirectional else hidden
        self.fc = nn.Linear(fc_input_dim, out_dim)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, in_dim)
        Returns:
            out: (batch, out_dim)
        """
        out, (h_n, c_n) = self.lstm(x)
        
        if self.bidirectional:
            h_final = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            h_final = h_n[-1]
        
        yhat = self.fc(h_final)
        return yhat


class GRUClassifier(nn.Module):
    """GRU para clasificación de señales."""
    
    def __init__(self, in_dim, hidden=128, layers=2, n_classes=5, bidirectional=False, dropout=0.0):
        super().__init__()
        self.bidirectional = bidirectional
        
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if layers > 1 else 0.0
        )
        
        fc_input_dim = hidden * 2 if bidirectional else hidden
        self.fc = nn.Linear(fc_input_dim, n_classes)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, in_dim)
        Returns:
            out: (batch, n_classes)
        """
        out, h_n = self.gru(x)
        
        if self.bidirectional:
            h_final = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            h_final = h_n[-1]
        
        yhat = self.fc(h_final)
        return yhat


class GRURegressor(nn.Module):
    """GRU para regresión de series temporales."""
    
    def __init__(self, in_dim, hidden=128, layers=2, out_dim=1, bidirectional=False, dropout=0.0):
        super().__init__()
        self.bidirectional = bidirectional
        
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if layers > 1 else 0.0
        )
        
        fc_input_dim = hidden * 2 if bidirectional else hidden
        self.fc = nn.Linear(fc_input_dim, out_dim)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, in_dim)
        Returns:
            out: (batch, out_dim)
        """
        out, h_n = self.gru(x)
        
        if self.bidirectional:
            h_final = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            h_final = h_n[-1]
        
        yhat = self.fc(h_final)
        return yhat
