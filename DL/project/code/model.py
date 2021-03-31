from typing import Tuple

import torch
from torch import nn
from torch.nn.modules import dropout


class LSTMModel(nn.Module):
    """LSTM based model

    Reproduce the architecture describe in the article
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layer: int = 1, dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layer, batch_first=True, dropout=dropout*(n_layer > 1))
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] = None):
        """Forward pass

        Args:
            input (tensor[batch_size, seq_len, input_dim]): Inputs
            hidden (Tuple[tensor[n_layer, batch_size, hidden_dim]]): h_0 and c_0 for each layer and sequence.

        Returns:
            tensor[batch_size, seq_len, output_dim]: Outputs
            Tuple[tensor[n_layer, batch_size, hidden_dim]]: h_n and c_n for each layer.
        """
        x = self.dropout(x)
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden


class GRUModel(nn.Module):
    """GRU based model

    Reproduce the architecture describe in the article
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layer: int = 1, dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(input_dim, hidden_dim, n_layer, batch_first=True, dropout=dropout*(n_layer > 1))
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] = None):
        """Forward pass

        Args:
            input (tensor[batch_size, seq_len, input_dim]): Inputs
            hidden (Tuple[tensor[n_layer, batch_size, hidden_dim]]): h_0 and c_0 for each layer and sequence.

        Returns:
            tensor[batch_size, seq_len, output_dim]: Outputs
            Tuple[tensor[n_layer, batch_size, hidden_dim]]: h_n and c_n for each layer.
        """
        x = self.dropout(x)
        x, hidden = self.gru(x, hidden)
        x = self.fc(x)
        return x, hidden
