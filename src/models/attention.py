"""
Cross-level attention mechanisms for wavelet coefficients.

This module implements level-to-level attention where each wavelet level (as a whole)
attends to other levels, rather than individual coefficients attending to each other.
This creates a more balanced attention mechanism where levels with more coefficients
don't have disproportionate influence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import AdaLayerNorm


class CrossLevelAttention(nn.Module):
    """Cross-level attention mechanism for wavelet coefficients.
    
    This module implements level-to-level attention where each wavelet level (as a whole)
    attends to other levels, rather than individual coefficients attending to each other.
    This creates a more balanced attention mechanism where levels with more coefficients
    don't have disproportionate influence.
    """
    
    def __init__(self, level_embed_dims, common_dim=None, num_heads=8, dropout=0.1, 
                 time_embed_dim=64, attention_mode="all_to_all"):
        super().__init__()
        self.level_embed_dims = level_embed_dims
        self.num_levels = len(level_embed_dims)
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention_mode = attention_mode  # "all_to_all" or "cross_only"
        
        # Use the maximum embedding dimension as common dimension if not specified
        self.common_dim = common_dim if common_dim is not None else max(level_embed_dims)
        
        # Level aggregation layers - convert variable length coefficient sequences to fixed-size level representations
        self.level_aggregators = nn.ModuleList()
        for embed_dim in level_embed_dims:
            # Use attention-based pooling to aggregate coefficients within each level
            aggregator = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, 1),
                nn.Softmax(dim=1)
            )
            self.level_aggregators.append(aggregator)
        
        # Project aggregated level representations to common dimension
        self.level_projections = nn.ModuleList()
        for embed_dim in level_embed_dims:
            projector = nn.Linear(embed_dim, self.common_dim)
            self.level_projections.append(projector)
        
        # Cross-level attention layers (now operating on level representations)
        if self.attention_mode == "all_to_all":
            # Standard multi-head attention (each level attends to all levels including itself)
            self.cross_attention = nn.MultiheadAttention(
                self.common_dim, num_heads, dropout=dropout, batch_first=True
            )
        else:  # cross_only
            # Custom attention that prevents self-attention
            self.cross_attention_layers = nn.ModuleList()
            for i in range(self.num_levels):
                # Each level gets its own attention layer for attending to other levels
                attention_layer = nn.MultiheadAttention(
                    self.common_dim, num_heads, dropout=dropout, batch_first=True
                )
                self.cross_attention_layers.append(attention_layer)
        
        # Level expansion layers - distribute level information back to coefficients
        self.level_expanders = nn.ModuleList()
        for embed_dim in level_embed_dims:
            expander = nn.Linear(self.common_dim, embed_dim)
            self.level_expanders.append(expander)
        
        # Adaptive layer norm for cross-level attention
        self.cross_norm = nn.ModuleList()
        for embed_dim in level_embed_dims:
            norm = AdaLayerNorm(embed_dim, time_embed_dim)
            self.cross_norm.append(norm)
        
        # Learnable level position encodings to distinguish between levels
        self.level_position_embeddings = nn.Parameter(torch.randn(self.num_levels, self.common_dim))
        # Initialize level position encodings with more diverse values to help distinguish levels
        with torch.no_grad():
            self.level_position_embeddings.normal_(0, 0.1)
            # Ensure level position encodings are somewhat orthogonal
            for i in range(self.num_levels):
                self.level_position_embeddings[i] = F.normalize(self.level_position_embeddings[i], dim=0)
        
        # Gate to control how much cross-level information to use
        self.cross_level_gates = nn.ModuleList()
        for embed_dim in level_embed_dims:
            gate = nn.Sequential(
                nn.Linear(embed_dim + time_embed_dim, embed_dim),
                nn.Sigmoid()
            )
            self.cross_level_gates.append(gate)
    
    def get_cross_level_attention_weights(self, level_embeddings, time_embed):
        """
        Extract cross-level attention weights for visualization.
        This method mirrors the forward pass logic to ensure consistency.
        
        Args:
            level_embeddings: List of tensors, each of shape [batch_size, level_seq_len, level_embed_dim]
            time_embed: [batch_size, time_embed_dim]
        
        Returns:
            Attention weights tensor of shape [num_levels, num_levels]
        """
        batch_size = level_embeddings[0].shape[0]
        
        # Step 1: Aggregate each level's coefficient embeddings into a single level representation
        level_representations = []
        for i, (level_emb, aggregator, projector) in enumerate(zip(level_embeddings, self.level_aggregators, self.level_projections)):
            # level_emb shape: [batch_size, level_seq_len, level_embed_dim]
            
            # Self-attention pooling
            attention_weights = aggregator(level_emb)
            
            # Apply attention weights to aggregate coefficients into level representation
            level_repr = torch.sum(level_emb * attention_weights, dim=1)  # [batch_size, level_embed_dim]
            
            # Project to common dimension
            level_repr = projector(level_repr)  # [batch_size, common_dim]
            
            # Add level-specific position encoding
            level_repr = level_repr + self.level_position_embeddings[i].unsqueeze(0)
            
            level_representations.append(level_repr)
        
        # Step 2: Stack level representations for attention computation
        level_stack = torch.stack(level_representations, dim=1)  # [batch_size, num_levels, common_dim]
        
        # Step 3: Apply level-to-level attention and extract weights
        if self.attention_mode == "all_to_all":
            # Use the built-in attention mechanism
            _, attention_weights = self.cross_attention(
                level_stack, level_stack, level_stack, average_attn_weights=True, need_weights=True
            )
            # attention_weights shape: [batch_size, num_levels, num_levels]
            # Average over batch dimension for visualization
            attention_weights = attention_weights.mean(dim=0)  # [num_levels, num_levels]
            
        else:  # cross_only
            # For cross-only mode, construct attention matrix manually
            attention_matrix = torch.zeros(self.num_levels, self.num_levels, device=level_stack.device)
            
            for i in range(self.num_levels):
                # Create key-value tensor excluding the current level
                other_levels_indices = [j for j in range(self.num_levels) if j != i]
                if not other_levels_indices:
                    continue
                    
                other_levels = level_stack[:, other_levels_indices, :]  # [batch_size, num_other_levels, common_dim]
                query_level = level_stack[:, i:i+1, :]  # [batch_size, 1, common_dim]
                
                # Apply attention
                _, attn_weights = self.cross_attention_layers[i](
                    query_level, other_levels, other_levels
                )
                # attn_weights shape: [batch_size, 1, num_other_levels]
                
                # Average over batch and place in correct positions
                attn_weights_avg = attn_weights.mean(dim=0).squeeze(0)  # [num_other_levels]
                for k, j in enumerate(other_levels_indices):
                    attention_matrix[i, j] = attn_weights_avg[k]
            
            attention_weights = attention_matrix
        
        return attention_weights

    def forward(self, level_embeddings, time_embed):
        """
        Args:
            level_embeddings: List of tensors, each of shape [batch_size, level_seq_len, level_embed_dim]
            time_embed: [batch_size, time_embed_dim]
        
        Returns:
            List of tensors with same shapes as input, but with cross-level attention applied
        """
        batch_size = level_embeddings[0].shape[0]
        
        # Store original embeddings for residual connections
        original_embeddings = [emb.clone() for emb in level_embeddings]
        
        # Step 1: Aggregate each level's coefficient embeddings into a single level representation
        level_representations = []
        for i, (level_emb, aggregator, projector) in enumerate(zip(level_embeddings, self.level_aggregators, self.level_projections)):
            # level_emb shape: [batch_size, level_seq_len, level_embed_dim]
            
            # Self-attention pooling
            # Compute attention weights for aggregation
            attention_weights = aggregator(level_emb)  # [batch_size, level_seq_len, 1]
            
            # Apply attention weights to aggregate coefficients into level representation
            level_repr = torch.sum(level_emb * attention_weights, dim=1)  # [batch_size, level_embed_dim]
            
            # Project to common dimension
            level_repr = projector(level_repr)  # [batch_size, common_dim]
            
            # Add level-specific position encoding
            level_repr = level_repr + self.level_position_embeddings[i].unsqueeze(0)
            
            level_representations.append(level_repr)
        
        # Step 2: Stack level representations for attention computation
        # Shape: [batch_size, num_levels, common_dim]
        level_stack = torch.stack(level_representations, dim=1)
        
        # Step 3: Apply level-to-level attention
        if self.attention_mode == "all_to_all":
            # ALL-TO-ALL: Each level attends to all levels (including itself)
            cross_attended_levels, attention_weights = self.cross_attention(
                level_stack, level_stack, level_stack
            )
            # cross_attended_levels shape: [batch_size, num_levels, common_dim]
            # attention_weights shape: [batch_size, num_heads, num_levels, num_levels]
            
        else:  # cross_only
            # CROSS-ONLY: Each level only attends to other levels (not itself)
            cross_attended_levels = []
            
            for i in range(self.num_levels):
                # Create key-value tensor excluding the current level
                other_levels_indices = [j for j in range(self.num_levels) if j != i]
                if not other_levels_indices:
                    # If there's only one level, just return the original representation
                    cross_attended_levels.append(level_representations[i])
                    continue
                    
                other_levels = level_stack[:, other_levels_indices, :]  # [batch_size, num_other_levels, common_dim]
                query_level = level_stack[:, i:i+1, :]  # [batch_size, 1, common_dim]
                
                # Apply cross-attention (level i attends to all other levels)
                cross_attended_level, _ = self.cross_attention_layers[i](
                    query_level, other_levels, other_levels
                )
                # cross_attended_level shape: [batch_size, 1, common_dim]
                cross_attended_levels.append(cross_attended_level.squeeze(1))  # [batch_size, common_dim]
            
            # Convert to tensor for consistency
            cross_attended_levels = torch.stack(cross_attended_levels, dim=1)
            # cross_attended_levels shape: [batch_size, num_levels, common_dim]
        
        # Step 4: Expand level representations back to coefficient embeddings
        output_embeddings = []
        for i, (cross_attended_level, expander, original_emb) in enumerate(
            zip(cross_attended_levels.unbind(1), self.level_expanders, original_embeddings)
        ):
            # cross_attended_level shape: [batch_size, common_dim]
            # original_emb shape: [batch_size, level_seq_len, level_embed_dim]
            
            # Expand level representation to all coefficients in this level
            expanded_level = expander(cross_attended_level)  # [batch_size, level_embed_dim]
            expanded_level = expanded_level.unsqueeze(1).expand(-1, original_emb.shape[1], -1)
            # expanded_level shape: [batch_size, level_seq_len, level_embed_dim]
            
            # Apply adaptive layer norm
            cross_output_norm = self.cross_norm[i](expanded_level, time_embed)
            
            # Compute gate to control cross-level information
            time_embed_expanded = time_embed.unsqueeze(1).expand(-1, original_emb.shape[1], -1)
            gate_input = torch.cat([original_emb, time_embed_expanded], dim=-1)
            gate = self.cross_level_gates[i](gate_input)
            
            # Apply gated residual connection
            output = original_emb + gate * cross_output_norm
            output_embeddings.append(output)
        
        return output_embeddings



