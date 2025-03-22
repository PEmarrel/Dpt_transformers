import torch
import torch.nn as nn
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads,
                 qkv_bias = False, device = 'cpu'):
        
        super().__init__()
        assert(d_out % num_heads == 0), "d_out should be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        
        # self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        # self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        # self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias) 
        # Don't use this because we want to have only one projection to optimize
        # To have only one projection we use the following line
        self.W_qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias).to(device)

        self.dropout = nn.Dropout(dropout).to(device)
        self.out_proj = nn.Linear(d_out, d_out).to(device)
        # If we want to see past
        mask = torch.triu(torch.ones(context_length,context_length), diagonal=1).to(device)
        
        # if we want to see future
        # mask = torch.tril(torch.ones(context_length,context_length), diagonal=-1).to(device)
        
        # If we want to see all the context
        # mask = torch.zeros(context_length, context_length, device=device)
        self.register_buffer('mask', mask)
        
    def forward(self,x: torch.Tensor):
        queries: torch.Tensor
        keys: torch.Tensor
        values: torch.Tensor
        b, num_tokens, d_in = x.shape # b, num_token, d_in

        # self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        # self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        # self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # Use one projection to get queries, keys and values
        # self.W_qkv(x) -> b, num_token, 3*d_out (is a tensor)
        # chunk(3, dim=-1) -> b, num_token, d_out we split the tensor in 3 parts
        queries, keys, values= self.W_qkv(x).chunk(3, dim=-1)
        
        # b, num_token, numheads, head_dim
        queries = queries.reshape(b,
                                num_tokens,
                                self.num_heads,
                                self.head_dim
                            ).transpose(1, 2)
        keys = keys.reshape(b,
                            num_tokens,
                            self.num_heads,
                            self.head_dim
                        ).transpose(1, 2)
        values = values.reshape(b,
                                num_tokens,
                                self.num_heads,
                                self.head_dim
                            ).transpose(1, 2)
        
        # b, num_heads, num_token, num_token
        attn_scores = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        
        attn_scores = attn_scores.masked_fill(
            self.mask[:num_tokens, :num_tokens].unsqueeze(0).unsqueeze(0).bool() == 1, 
            float('-inf')
        )

        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.einsum('bhqk, bhkd -> bhqd', 
                               attn_weights, 
                               values
                            ).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        
        
        return self.out_proj(context)
    
class FeedForward(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        # TODO fct FFN we can change the number of layers
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], cfg["emb_dim"] // 2, bias=cfg["qkv_bias"]),
            nn.GELU(),
            nn.Linear(cfg["emb_dim"] // 2, cfg["emb_dim"], bias=cfg["qkv_bias"])
        )
        
    def forward(self,x):
        return self.layers(x)
    

class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg["emb_dim"]).to(cfg["device"])
        self.att = MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            dropout = cfg["drop_rate"],
            num_heads = cfg["n_heads"],
            qkv_bias = cfg["qkv_bias"],
            device=cfg["device"]
        )
        self.ff = FeedForward(cfg).to(cfg["device"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"]).to(cfg["device"])
        self.dropout = nn.Dropout(cfg["drop_rate"]).to(cfg["device"])
        
    def forward(self,x):
        # print("we are in one transformer block")
        # print(x.shape)
        # print(x)
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        # print("after att", x.shape)
        # print(x)
        x = self.dropout(x)

        x = x + shortcut
        
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x+ shortcut
        # print(x.shape)
        # print(x)
        # print('end trs block')
        return x
