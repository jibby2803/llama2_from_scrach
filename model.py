import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass 
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256 
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    
    # KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048 

    device: str = None
    

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 100000.0):
    assert head_dim % 2 == 0, "head_dim must be even"
    theta_numerator = torch.arange(0, head_dim, 2, device=device).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device) # (head_dim/2,)
    
    m = torch.arange(seq_len, device=device).float() # (seq_len,)
    freqs = torch.outer(m, theta) # (seq_len, head_dim/2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs) # (seq_len, head_dim/2) 
    return freqs_complex

def apply_rotary_embedding(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2)) # (batch, seq_len, n_heads, head_dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2) # (1, seq_len, 1, head_dim/2)
    x_rotated = torch.view_as_real(x_complex * freqs_complex) # (batch, seq_len, n_heads, head_dim/2, 2)
    x_out = x_rotated.flatten(-2) # (batch, seq_len, n_heads, head_dim)
    return x_out.to(device)
    

class RMSNorm(nn.Module):
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
     
    def forward(self, x: torch.Tensor):
        return self.weight * self._norm(x.float()).to(x.dtype)


class SelfAttention(nn.Module):
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim), device=args.device)
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim), device=args.device)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape
        
        xq = self.wq(x) # (B, S, n_heads * head_dim)
        xk = self.wk(x) # (B, S, n_kv_heads * head_dim)
        xv = self.wv(x) # (B, S, n_kv_heads * head_dim)
        
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim) # (B, S, n_heads_q, head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim) # (B, S, n_kv_heads, head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim) # (B, S, n_kv_heads, head_dim)
        
        xq = apply_rotary_embedding(xq, freqs_complex, x.device) # (B, S, n_heads_q, head_dim)
        xk = apply_rotary_embedding(xk, freqs_complex, x.device) # (B, S, n_kv_heads, head_dim)
        
        self.cache_k[: batch_size, start_pos: start_pos + seq_len] = xk
        self.cache_v[: batch_size, start_pos: start_pos + seq_len] = xv
        
        keys = self.cache_k[: batch_size, : start_pos + seq_len] # (B, S', n_kv_heads, head_dim)
        values = self.cache_v[: batch_size, : start_pos + seq_len] # (B, S', n_kv_heads, head_dim)
        
        keys = keys.repeat_interleave(self.n_rep, dim=2) # (B, S', n_heads_q, head_dim)
        values = values.repeat_interleave(self.n_rep, dim=2) # (B, S', n_heads_q, head_dim)
        
        xq = xq.transpose(1, 2) # (B, n_heads_q, S, head_dim)
        keys = keys.transpose(1, 2) # (B, n_heads_q, S', head_dim)
        values = values.transpose(1, 2) # (B, n_heads_q, S', head_dim)
        
        scores = torch.matmul(xq, keys.transpose(-2, -1)) / math.sqrt(self.head_dim) # (B, n_heads_q, S, S')
        scores = F.softmax(scores, dim=-1) # (B, n_heads_q, S, S')
        out = torch.matmul(scores, values) # (B, n_heads_q, S, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1) # (B, S, n_heads_q * head_dim)
        out = self.wo(out) # (B, S, dim)
        return out


class FeedForward(nn.Module):
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * args.dim)
        
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)
        
    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)
        return x
        
            

class EncoderBlock(nn.Module):
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads 
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)
        
        # norm before attn 
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # norm before ffn
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out        


class Transformer(nn.Module):
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        assert args.vocab_size != -1, "vocab_size must be specified"
        
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        self.layers = nn.ModuleList() 
        for _ in range(self.n_layers):
            self.layers.append(EncoderBlock(args))
        
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)
        
        self.freqs_complex = precompute_theta_pos_frequencies(
            self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device = self.args.device
        )
        
    
    def forward(self, tokens: torch.Tensor, start_pos: int):
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "only seq_len=1 is supported for now"
        
        h = self.tok_embeddings(tokens)
        freqs_complex = self.freqs_complex[start_pos: start_pos + seq_len]
        
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output