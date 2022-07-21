import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch import Tensor
import copy
import mup
import numpy as np


def dot_product_attention(q: Tensor, k: Tensor, v: Tensor,
                          attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    B, Nt, E = q.shape
    q = q * (8.0 / E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    if attn_mask is not None:
        attn = torch.baddbmm(attn_mask, q, k.transpose(-2, -1))
    else:
        attn = torch.bmm(q, k.transpose(-2, -1))

    attn = F.softmax(attn, dim=-1)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn


def attn_mask_from_key_mask(key_padding_mask, num_heads, dtype):
    if key_padding_mask is None:
        return None

    bsz, n = key_padding_mask.shape
    attn_mask = key_padding_mask.view(bsz, 1, 1, n).   \
        expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, n)

    # convert mask to float
    new_attn_mask = torch.zeros_like(attn_mask, dtype=dtype)
    new_attn_mask.masked_fill_(attn_mask, float("-inf"))
    return new_attn_mask


class SelfAttention(nn.Module):
    __constants__ = ['batch_first']
    def __init__(self, embed_dim, num_heads, bias=True,
                 batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        self.use_bias = bias
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj = nn.Linear(embed_dim, 3*embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.use_bias is not None:
            nn.init.constant_(self.in_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None):
        if self.batch_first:
            x = x.transpose(1, 0)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        n, bsz, d = q.shape
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bsz, n)
        attn_mask = attn_mask_from_key_mask(key_padding_mask, self.num_heads, q.dtype)

        q = q.contiguous().view(n, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(n, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(n, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        output, att_weights = dot_product_attention(q, k, v, attn_mask)
        output = output.transpose(0, 1).contiguous().view(n * bsz, self.embed_dim)
        att_weights = att_weights.contiguous().view(bsz, self.num_heads, n, n)
        output = self.out_proj(output).view(n, bsz, self.embed_dim)

        if self.batch_first:
            return output.transpose(1, 0), att_weights
        else:
            return output, att_weights

    def mup_init(self):
        mup.init.xavier_uniform_(self.in_proj.weight)
        mup.init.xavier_uniform_(self.out_proj.weight)
        if self.use_bias is not None:
            nn.init.constant_(self.in_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)


class TransformerEncoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 activation_fn = nn.ReLU, layer_norm_eps: float = 1e-5,
                 batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = SelfAttention(d_model, nhead, bias=True, batch_first=batch_first,
                                       **factory_kwargs)

        # Implementation of Feedforward model
        self.ff_net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, **factory_kwargs),
            activation_fn(),
            nn.Linear(dim_feedforward, d_model, **factory_kwargs),
        )

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self.self_attn(self.norm1(x), key_padding_mask)[0]
            x = x + self.ff_net(self.norm2(x))
        else:
            x = self.norm1(x + self.self_attn(x, key_padding_mask)[0])
            x = self.norm2(x + self.ff_net(x))
        return x

    def mup_init(self):
        self.self_attn.mup_init()
        for name, param in self.ff_net.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.)
            else:
                mup.init.kaiming_uniform_(param, np.sqrt(5.))


class TransformerEncoder(nn.Module):
    def __init__(self, module, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(module) for i in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, key_padding_mask=key_padding_mask)
        return output

    def mup_init(self):
        for layer in self.layers:
            layer.mup_init()


if __name__ == '__main__':
    num_heads = 4
    d = 64
    batch_size = 2
    seq_len = 20

    sa = SelfAttention(d, num_heads, bias=True, batch_first=True)

    emb = torch.rand((batch_size, seq_len, d))
    key_padding_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    for i in range(batch_size):
        key_padding_mask[i, -i:] = True

    output, att_weights = sa(emb, key_padding_mask=key_padding_mask)
    print(output.shape, att_weights.shape)

    tel = TransformerEncoderLayer(d, num_heads, 2 * d, batch_first=True)
    encoder = TransformerEncoder(tel, 4)

    emb2 = encoder(emb)
    print(emb2.shape)
