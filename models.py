"""
transformer models used
TripleGRN is TE(k=3)
DoubleGRN is TE(k=2)
CovRev is TE(k=1)
"""

import torch
import torch.nn as nn
import numpy as np


class TripleGRN(nn.Module): # use inferred GRN (WENDY, etc.) and two covariance matrices to infer GRN
    def __init__(self, n_gene, d_model, nhead, num_layers, dropout=0.1):
        super(TripleGRN, self).__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4")
        
        # Separate embedding layers for each input
        self.embedding_x = nn.Linear(1, d_model)
        self.embedding_x_k0 = nn.Linear(1, d_model)
        self.embedding_x_ktstar = nn.Linear(1, d_model)

        # Segment embedding layer
        self.segment_embedding = nn.Embedding(3, d_model)  # 3 segment IDs: 0 for x, 1 for x', 2 for x''

        self.d_model_quarter = d_model // 4
        self.n_gene = n_gene

        # Transformer layers
        encoder_layers = nn.TransformerEncoderLayer(3 * d_model, nhead, dim_feedforward=4 * 3 * d_model, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output = nn.Linear(3 * d_model, 1)

    def positional_encoding(self, height, width, device):
        y_position = torch.arange(0, height, dtype=torch.float, device=device).unsqueeze(1).unsqueeze(2)
        x_position = torch.arange(0, width, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(2)
        
        div_term = torch.exp(torch.arange(0, self.d_model_quarter, 2, device=device).float() * (-np.log(10000.0) / self.d_model_quarter))
        pe = torch.zeros(height, width, self.d_model_quarter * 4, device=device)
        pe[:, :, 0:self.d_model_quarter:2] = torch.sin(y_position * div_term)
        pe[:, :, 1:self.d_model_quarter:2] = torch.cos(y_position * div_term)
        pe[:, :, self.d_model_quarter:2*self.d_model_quarter:2] = torch.sin(x_position * div_term)
        pe[:, :, self.d_model_quarter+1:2*self.d_model_quarter:2] = torch.cos(x_position * div_term)
        
        # Repeat the positional encoding to match the dimension of the combined embedding
        pe = pe.repeat(1, 1, 3)  # Repeat across the third dimension 3 times to match (height, width, 3 * d_model)
        
        return pe

    def forward(self, x, x_k0, x_ktstar):
        device = x.device  # Automatically determine the device
        batch_size, height, width = x.shape

        # Apply separate embedding layers to each input
        x_embedded = self.embedding_x(x.unsqueeze(-1))  # Shape: (batch_size, height, width, d_model)
        x_k0_embedded = self.embedding_x_k0(x_k0.unsqueeze(-1))  # Shape: (batch_size, height, width, d_model)
        x_ktstar_embedded = self.embedding_x_ktstar(x_ktstar.unsqueeze(-1))  # Shape: (batch_size, height, width, d_model)
        
        # Create segment IDs: 0 for x, 1 for x', 2 for x''
        segment_ids = torch.zeros_like(x, dtype=torch.long, device=device)  # Shape: (batch_size, height, width)
        segment_ids_k0 = torch.ones_like(x_k0, dtype=torch.long, device=device)  # Shape: (batch_size, height, width)
        segment_ids_ktstar = 2 * torch.ones_like(x_ktstar, dtype=torch.long, device=device)  # Shape: (batch_size, height, width)

        # Get segment embeddings
        segment_embedding_x = self.segment_embedding(segment_ids)
        segment_embedding_x_k0 = self.segment_embedding(segment_ids_k0)
        segment_embedding_x_ktstar = self.segment_embedding(segment_ids_ktstar)

        # Add segment embeddings to the respective inputs
        x_embedded += segment_embedding_x
        x_k0_embedded += segment_embedding_x_k0
        x_ktstar_embedded += segment_embedding_x_ktstar

        # Combine the embeddings by concatenation
        combined_embedding = torch.cat((x_embedded, x_k0_embedded, x_ktstar_embedded), dim=-1)  # Shape: (batch_size, height, width, 3 * d_model)
        
        # Add positional encoding
        pe = self.positional_encoding(height, width, device)
        src = combined_embedding + pe  # Ensure positional encoding is on the same device
        
        # Flatten for transformer input
        src = src.view(batch_size, height * width, -1).transpose(0, 1).contiguous()  # Shape: (seq_len, batch_size, 3 * d_model)
        
        # Apply transformer encoder
        output = self.transformer_encoder(src)
        
        # Reshape output and apply final linear layer
        output = output.transpose(0, 1).contiguous()  # Shape: (batch_size, seq_len, 3 * d_model)
        output = self.output(output)  # Shape: (batch_size, seq_len, 1)
        
        return output.view(batch_size, height, width)

class DoubleGRN(nn.Module): # use inferred GRN (WENDY, etc.) and two covariance matrices to infer GRN
    def __init__(self, n_gene, d_model, nhead, num_layers, dropout=0.1):
        super(DoubleGRN, self).__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4")
        
        # Separate embedding layers for each input
        self.embedding_x = nn.Linear(1, d_model)
        self.embedding_x_k = nn.Linear(1, d_model)

        # Segment embedding layer
        self.segment_embedding = nn.Embedding(2, d_model)  # 3 segment IDs: 0 for x, 1 for x', 2 for x''

        self.d_model_quarter = d_model // 4
        self.n_gene = n_gene

        # Transformer layers
        encoder_layers = nn.TransformerEncoderLayer(2 * d_model, nhead, dim_feedforward=4 * 2 * d_model, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output = nn.Linear(2 * d_model, 1)

    def positional_encoding(self, height, width, device):
        y_position = torch.arange(0, height, dtype=torch.float, device=device).unsqueeze(1).unsqueeze(2)
        x_position = torch.arange(0, width, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(2)
        
        div_term = torch.exp(torch.arange(0, self.d_model_quarter, 2, device=device).float() * (-np.log(10000.0) / self.d_model_quarter))
        pe = torch.zeros(height, width, self.d_model_quarter * 4, device=device)
        pe[:, :, 0:self.d_model_quarter:2] = torch.sin(y_position * div_term)
        pe[:, :, 1:self.d_model_quarter:2] = torch.cos(y_position * div_term)
        pe[:, :, self.d_model_quarter:2*self.d_model_quarter:2] = torch.sin(x_position * div_term)
        pe[:, :, self.d_model_quarter+1:2*self.d_model_quarter:2] = torch.cos(x_position * div_term)
        
        # Repeat the positional encoding to match the dimension of the combined embedding
        pe = pe.repeat(1, 1, 2)  # Repeat across the third dimension 3 times to match (height, width, 3 * d_model)
        
        return pe

    def forward(self, x, x_k):
        device = x.device  # Automatically determine the device
        batch_size, height, width = x.shape

        # Apply separate embedding layers to each input
        x_embedded = self.embedding_x(x.unsqueeze(-1))  # Shape: (batch_size, height, width, d_model)
        x_k_embedded = self.embedding_x_k(x_k.unsqueeze(-1))  # Shape: (batch_size, height, width, d_model)
        
        # Create segment IDs: 0 for x, 1 for x', 2 for x''
        segment_ids = torch.zeros_like(x, dtype=torch.long, device=device)  # Shape: (batch_size, height, width)
        segment_ids_k = torch.ones_like(x_k, dtype=torch.long, device=device)  # Shape: (batch_size, height, width)

        # Get segment embeddings
        segment_embedding_x = self.segment_embedding(segment_ids)
        segment_embedding_x_k = self.segment_embedding(segment_ids_k)

        # Add segment embeddings to the respective inputs
        x_embedded += segment_embedding_x
        x_k_embedded += segment_embedding_x_k

        # Combine the embeddings by concatenation
        combined_embedding = torch.cat((x_embedded, x_k_embedded), dim=-1)  # Shape: (batch_size, height, width, 3 * d_model)
        
        # Add positional encoding
        pe = self.positional_encoding(height, width, device)
        src = combined_embedding + pe  # Ensure positional encoding is on the same device
        
        # Flatten for transformer input
        src = src.view(batch_size, height * width, -1).transpose(0, 1).contiguous()  # Shape: (seq_len, batch_size, 3 * d_model)
        
        # Apply transformer encoder
        output = self.transformer_encoder(src)
        
        # Reshape output and apply final linear layer
        output = output.transpose(0, 1).contiguous()  # Shape: (batch_size, seq_len, 3 * d_model)
        output = self.output(output)  # Shape: (batch_size, seq_len, 1)
        
        return output.view(batch_size, height, width)


class CovRev(nn.Module):
    def __init__(self, n_gene, d_model, nhead, num_layers, dropout=0.1):
        super(CovRev, self).__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4")
        
        self.embedding = nn.Linear(1, d_model)
        self.d_model_quarter = d_model // 4
        self.n_gene = n_gene

        # Transformer layers
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output = nn.Linear(d_model, 1)

    def positional_encoding(self, height, width, device):
        y_position = torch.arange(0, height, dtype=torch.float, device=device).unsqueeze(1).unsqueeze(2)
        x_position = torch.arange(0, width, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(2)
        
        div_term = torch.exp(torch.arange(0, self.d_model_quarter, 2, device=device).float() * (-np.log(10000.0) / self.d_model_quarter))
        #print(div_term)
        pe = torch.zeros(height, width, self.d_model_quarter * 4, device=device)
        pe[:, :, 0:self.d_model_quarter:2] = torch.sin(y_position * div_term)
        pe[:, :, 1:self.d_model_quarter:2] = torch.cos(y_position * div_term)
        pe[:, :, self.d_model_quarter:2*self.d_model_quarter:2] = torch.sin(x_position * div_term)
        pe[:, :, self.d_model_quarter+1:2*self.d_model_quarter:2] = torch.cos(x_position * div_term)
        
        pe[:, :, 2*self.d_model_quarter:] = pe[:, :, :2*self.d_model_quarter]
        return pe

    def forward(self, src):
        device = src.device
        batch_size, height, width = src.shape
        src = src.view(batch_size, height, width, 1)  # Shape: (batch_size, height, width, 1)
        src = self.embedding(src)  # Shape: (batch_size, height, width, d_model)
        
        # Add positional encoding
        pe = self.positional_encoding(height, width, device)
        src = src + pe  # Ensure the positional encoding is on the same device as the input
        
        # Flatten for transformer input
        src = src.view(batch_size, height * width, -1).transpose(0, 1)  # Shape: (seq_len, batch_size, d_model)
        
        # Apply transformer encoder
        output = self.transformer_encoder(src)
        
        # Reshape output and apply final linear layer
        output = output.transpose(0, 1).contiguous()  # Shape: (batch_size, seq_len, d_model)
        output = self.output(output)  # Shape: (batch_size, seq_len, 1)
        
        return output.view(batch_size, height, width)