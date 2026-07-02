import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


class MultiHeadAttention(nn.Module): 
  def __init__(self, config): 
    super().__init__()
    self.hidden_size = config["hidden_size"]
    self.num_attention_heads = config["num_attention_heads"]
    
    self.attention_head_size = self.hidden_size // self.num_attention_heads
    self.all_head_size = self.num_attention_heads * self.attention_head_size 

    self.qkv_bias = config["qkv_bias"]
    self.query = nn.Linear(self.hidden_size, self.all_head_size, bias=self.qkv_bias)
    self.key = nn.Linear(self.hidden_size, self.all_head_size, bias=self.qkv_bias)
    self.value = nn.Linear(self.hidden_size, self.all_head_size, bias=self.qkv_bias)

    self.attn_dropout_prob = config["attention_probs_dropout_prob"]
    self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
    self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    self.gated_attention = config.get("gated_attention", "none")
    if self.gated_attention == "elementwise":
      self.gate_proj = nn.Linear(self.hidden_size, self.all_head_size, bias=True)
      self.gate_proj.bias.data.fill_(4.0)
      self.gate_proj._is_gate = True
    elif self.gated_attention == "headwise":
      self.gate_proj = nn.Linear(self.hidden_size, self.num_attention_heads, bias=True)
      self.gate_proj.bias.data.fill_(4.0)
      self.gate_proj._is_gate = True

  def forward(self, x, output_attentions=False): 
    batch_size, seq_len, _ = x.size()

    q = self.query(x).view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
    k = self.key(x).view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
    v = self.value(x).view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

    if output_attentions:
      attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
      attention_probs = F.softmax(attention_scores, dim=-1)
      attention_probs_dropped = F.dropout(attention_probs, p=self.attn_dropout_prob, training=self.training)
      attention_output = torch.matmul(attention_probs_dropped, v)
    else:
      attention_output = F.scaled_dot_product_attention(
        q, k, v,
        dropout_p=self.attn_dropout_prob if self.training else 0.0,
      )
      attention_probs = None

    attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.all_head_size)

    if self.gated_attention == "elementwise":
      attention_output = attention_output * torch.sigmoid(self.gate_proj(x))
    elif self.gated_attention == "headwise":
      gate = torch.sigmoid(self.gate_proj(x))
      gate = gate.unsqueeze(-1).expand(-1, -1, -1, self.attention_head_size)
      gate = gate.contiguous().view(batch_size, seq_len, self.all_head_size)
      attention_output = attention_output * gate

    attention_output = self.output_projection(attention_output)
    attention_output = self.output_dropout(attention_output)

    return (attention_output, attention_probs)


class MLP(nn.Module): 
  def __init__(self, config): 
    super().__init__()
    self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
    self.activation = nn.GELU()
    self.dense_2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
    self.dropout = nn.Dropout(config["hidden_dropout_prob"])

  def forward(self, x): 
    x = self.dense_1(x)
    x = self.activation(x)
    x = self.dense_2(x)
    x = self.dropout(x)
    return x
  
  
class Block(nn.Module): 
  def __init__(self, config): 
    super().__init__()
    self.attention = MultiHeadAttention(config)
    self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
    self.mlp = MLP(config)
    self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

  def forward(self, x, output_attentions=False):
    attention_output, attention_probs = \
      self.attention(self.layernorm_1(x), output_attentions=output_attentions)
    
    x = x + attention_output
    mlp_output = self.mlp(self.layernorm_2(x))
    x = x + mlp_output

    if output_attentions:
      return (x, attention_probs)
    else: 
      return (x, None)    


class Encoder(nn.Module):
  def __init__(self, config, gradient_checkpointing=False):
    super().__init__()
    self.gradient_checkpointing = gradient_checkpointing
    self.blocks = nn.ModuleList([])
    for _ in range(config["num_hidden_layers"]):
      block = Block(config)
      self.blocks.append(block)

  def forward(self, x, output_attentions=False):
    all_attentions = []
    for block in self.blocks:
      if self.gradient_checkpointing and self.training and not output_attentions:
        x, attention_probs = grad_checkpoint(
            block, x, output_attentions, use_reentrant=False
        )
      else:
        x, attention_probs = block(x, output_attentions=output_attentions)
      if output_attentions:
        all_attentions.append(attention_probs)

    if output_attentions:
      return (x, all_attentions)
    else:
      return (x, None)
