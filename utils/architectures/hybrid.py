import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.3):
        super(ImprovedResidualBlock, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout1d(dropout_rate),
            nn.Conv1d(out_channels, out_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(out_channels)
        )
        
        self.skip_connection = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else nn.Identity()
        
        self.dropout = nn.Dropout1d(dropout_rate)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = self.skip_connection(x)
        x = self.conv_block(x)
        x = self.dropout(x + residual)
        x = self.relu(x)
        return x
    
class ECG_CNN_RNN(nn.Module):
    """
    ECG_CNN_RNN
    -----------
    Hierarchical ECG classifier combining convolutional feature extraction with
    multi-scale temporal modeling and multi-head attention.

    High-level flow:
    1. CNN Backbone: 1D conv + residual blocks extract local waveform features.
    2. Short-term RNN: bidirectional LSTM over CNN features to capture local temporal patterns.
    3. Long-term RNN: bidirectional LSTM over short-term outputs to capture longer-range dependencies.
    4. Gated Fusion: learn per-timestep blending between short-term and long-term streams.
    5. Multi-head Attention: several independent attention scorers produce per-head contexts.
    6. Context Refinement RNN: an RNN over the stacked head contexts to model inter-head interactions.
    7. Classification Head: LayerNorm + MLP to produce final logits.
    """
    def __init__(self, input_channels=1, sequence_length=187, hidden_size=128, num_heads=4, num_classes=5, dropout=0.3):
        super(ECG_CNN_RNN, self).__init__()
        self.model_name = 'CNN-RNN'
        
        # --- CNN Backbone ---
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.res_block1 = ImprovedResidualBlock(32, 64, dropout_rate=dropout)
        self.res_block2 = ImprovedResidualBlock(64, 128, dropout_rate=dropout)
        self.res_block3 = ImprovedResidualBlock(128, 256, dropout_rate=dropout)

        # --- Short-term RNN ---
        self.short_rnn = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # --- Long-term RNN ---
        self.long_rnn = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # --- Gated Fusion ---
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.Sigmoid()
        )

        # --- Multi-head Attention ---
        self.attn_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1)
            ) for _ in range(num_heads)
        ])

        # --- Context Refinement RNN ---
        self.context_rnn = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # --- Classification Head ---
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

        # Calculate the size of flattened features
        self._feature_size = self._get_feature_size(input_channels, sequence_length)

    def _get_feature_size(self, input_channels, sequence_length):
        """Calculate the size of flattened features after convolutions"""
        # Create a dummy tensor to forward through the network
        x = torch.randn(1, input_channels, sequence_length)
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        # Return the flattened size
        return x.view(1, -1).size(1)

    def forward(self, x):
        # x: (batch, seq_len, channels) → (batch, channels, seq_len)
        x = x.transpose(1, 2)

        # --- CNN Feature Extraction ---
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        # --- Prepare for RNN: (batch, channels, seq_len) → (batch, seq_len, channels) ---
        x = x.transpose(1, 2)

        # --- Short-term RNN ---
        short_out, _ = self.short_rnn(x)

        # --- Long-term RNN ---
        long_out, _ = self.long_rnn(short_out)

        # --- Gated Fusion ---
        combined = torch.cat([short_out, long_out], dim=-1)
        gate = self.gate(combined)
        fused = gate * short_out + (1 - gate) * long_out

        # --- Multi-head Attention ---
        attn_outputs = []
        for head in self.attn_heads:
            weights = head(fused)               # (batch, seq_len, 1)
            weights = F.softmax(weights, dim=1)
            context = torch.sum(weights * fused, dim=1)  # (batch, 2H)
            attn_outputs.append(context)

        # --- Stack and Refine ---
        multihead_context = torch.stack(attn_outputs, dim=1)  # (batch, heads, 2H)
        refined, _ = self.context_rnn(multihead_context)      # (batch, heads, 2H)
        refined = torch.mean(refined, dim=1)                  # (batch, 2H)

        # --- Classification ---
        out = self.fc(refined)
        return out