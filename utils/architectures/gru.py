import torch
import torch.nn as nn

class AttentionRNN(nn.Module):
    """
    AttentionRNN
    ------------
    Compact sequence classifier that demonstrates the core building blocks used
    in the larger ECG_CNN_RNN architecture.

    Components:
    - Bidirectional GRU: encodes sequence into per-timestep hidden vectors.
    - Attention scorer: a small MLP that produces a scalar score per timestep.
    - Skip connection: projects raw inputs into the GRU hidden space and adds
      them to the GRU outputs to preserve low-level information.
    - Classification head: projects the attention-weighted context to logits.

    Input shapes:
    - x: (batch, seq_len, input_channels)
    - Returns: (logits, attention_weights) where attention_weights is (batch, seq_len, 1)
    """
    def __init__(self, input_channels, hidden_size, num_layers, num_classes, dropout=0.2):
        super(AttentionRNN, self).__init__()
        self.model_name = 'GRU'
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional GRU produces hidden vectors of size hidden_size * 2
        self.gru = nn.GRU(input_channels, hidden_size, num_layers, 
                          batch_first=True, bidirectional=True, dropout=dropout)
        
        # Attention scorer: maps per-timestep hidden vector -> scalar score
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Skip connection: project raw input into the same dimensionality as GRU outputs
        self.skip_connection = nn.Linear(input_channels, hidden_size * 2)
        
        # Small classification head
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x: (batch, seq_len, input_channels)
        gru_out, _ = self.gru(x)  # gru_out: (batch, seq_len, 2*hidden_size)
        
        # Project inputs and add as a residual (preserve low-level features)
        skip = self.skip_connection(x)  # (batch, seq_len, 2*hidden_size)
        gru_out = gru_out + skip
        
        # Compute attention weights over timesteps
        attention_weights = self.attention(gru_out)          # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Weighted sum to get context vector
        context = torch.sum(attention_weights * gru_out, dim=1)  # (batch, 2*hidden_size)
        
        # Classification MLP
        out = self.fc1(context)               # (batch, hidden_size)
        out = self.batch_norm(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)                   # (batch, num_classes)
        
        return out, attention_weights