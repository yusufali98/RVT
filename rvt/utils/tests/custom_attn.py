import torch
import torch.nn as nn
import xformers.ops as xops

class MemoryEfficientSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p=0.1):
        super(MemoryEfficientSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x, mask):
        device = x.device
        dtype = torch.float16  # Use float16 as supported by the memory efficient attention

        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)  # (batch_size, seq_length, 3 * embed_dim)
        qkv = qkv.reshape(batch_size, seq_length, 3, self.num_heads, embed_dim // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_length, embed_dim // num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Create attention mask with -inf for padding positions
        attn_mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_length)
        attn_mask = attn_mask.expand(batch_size, self.num_heads, seq_length, seq_length)  # (batch_size, num_heads, seq_length, seq_length)
        attn_mask = attn_mask.to(dtype)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf'))
        attn_mask = attn_mask.masked_fill(attn_mask == 1, float(0.0))

        # Ensure the attn_bias is correctly broadcasted to match the batch size and num_heads
        attn_bias = attn_mask.reshape(batch_size * self.num_heads, seq_length, seq_length)

        # Cast q, k, v to the appropriate dtype
        q = q.to(device=device, dtype=dtype)
        k = k.to(device=device, dtype=dtype)
        v = v.to(device=device, dtype=dtype)
        attn_bias = attn_bias.to(device=device, dtype=dtype)

        attn_output = xops.memory_efficient_attention(
            query=q.reshape(batch_size * self.num_heads, seq_length, embed_dim // self.num_heads),
            key=k.reshape(batch_size * self.num_heads, seq_length, embed_dim // self.num_heads),
            value=v.reshape(batch_size * self.num_heads, seq_length, embed_dim // self.num_heads),
            p=self.dropout_p,
            attn_bias=attn_bias
        )  # (batch_size * num_heads, seq_length, embed_dim // num_heads)
        
        attn_output = attn_output.reshape(batch_size, self.num_heads, seq_length, embed_dim // self.num_heads)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_length, embed_dim)
        output = self.out_proj(attn_output)
        
        return output

# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 24
seq_length = 1984
embed_dim = 512
num_heads = 8

x = torch.randn(batch_size, seq_length, embed_dim, device=device).to(torch.float16)
mask = torch.randint(0, 2, (batch_size, seq_length), device=device, dtype=torch.bool)  # Example mask
attn_layer = MemoryEfficientSelfAttention(embed_dim, num_heads).to(device, dtype=torch.float16)
output = attn_layer(x, mask)

print(output.shape)
