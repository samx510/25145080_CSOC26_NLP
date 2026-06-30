import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalMultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads, max_seq_len):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divide by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        #  Projections from X to Q, K, V using basic Linear layers
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        #  lower-triangular mask: 1s on diagonal and below,0s above
        
        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len)))

    def forward(self, x):
        B, T, C = x.shape  # B=Batch size, T=Sequence length, C=d_model
        

        # projecting to Q,K,V
        q= self.q_proj(x)
        k=self.k_proj(x)
        v = self.v_proj(x)

        # Split of multiple heads and transpose
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        
        # Calculate raw scaled attention scores Q muliplied by K:
        #k.transpose(-2, -1) changes shape from [B, H, T, d_k] to [B, H, d_k, T]

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        #  Apply masking: replace 0s in the lower triangle mask with -inf
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        
        #  Softmax turns -inf into exactly 0 attention weight
        attn_weights = F.softmax(scores, dim=-1)
        
        # Final contextual output calculation
        out = attn_weights @ v  # [B, H, T, d_model]
        
        # Concatenate heads back together cleanly to reshape back to [B, T, d_model]
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        
        return self.out_proj(out)

class PositionWiseFeedForward(nn.Module):
    """
    Two-layer MLP processing positions identically across sequences using GELU.
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # Math calculation: GELU(xW1 + b1)W2 + b2
        return self.w2(F.gelu(self.w1(x)))

class TransformerBlock(nn.Module):
    """
    A single Transformer block implementing Pre-Layer Normalization (Pre-LN).
    """
    def __init__(self, d_model, n_heads, d_ff, max_seq_len):
        super().__init__()
        # Every block contains 2 sub-layers
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalMultiHeadAttention(d_model, n_heads, max_seq_len)
        
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PositionWiseFeedForward(d_model, d_ff)

    def forward(self, x):
        # Sub-Layer 1: Pre-LN -> Attention -> Residual Addition
        x = x + self.attn(self.ln1(x))
        # Sub-Layer 2: Pre-LN -> FFN -> Residual Addition
        x = x + self.ffn(self.ln2(x))
        return x

class DecoderTransformer(nn.Module):
    """
    Full Decoder-Only GPT-style Autoregressive Transformer with Weight Tying.
    """
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len
        
        # 1. Token and Learnable Position Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # 2. Stack of custom Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, max_seq_len) for _ in range(n_layers)
        ])
        
        # 3. Final Pre-output normalization
        self.ln_f = nn.LayerNorm(d_model)
        
        # 4. Language Model Head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # STRICT REQUIREMENT: Weight-Tied LM Head
        self.lm_head.weight = self.token_embedding.weight
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Cannot forward sequence of length {T}, max is {self.max_seq_len}"
        
        # Construct embeddings along the residual highway stream
        tok_emb = self.token_embedding(idx) # [B, T, d_model]
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.position_embedding(pos) # [T, d_model]
        
        x = tok_emb + pos_emb # [B, T, d_model]
        
        # Pass data forward through the stack of layers
        for layer in self.layers:
            x = layer(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x) # [B, T, vocab_size]
        
        loss = None
        if targets is not None:
            # Reshape logits and targets to meet PyTorch's cross_entropy shape expectations
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss
