import torch
import torch.nn as nn
import math
import gin

@gin.configurable(module='networks')
class NoPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(x)

    def index(self, inds):
        return None

    def get(self, size):
        return None


@gin.configurable(module='networks')
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, mult=1., dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) / mult)
        self.mult = mult

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.shape[1]] * self.mult)

    def index(self, inds):
        return self.pe[0][inds] * self.mult

    def get(self, size):
        return self.pe[:, :size] * self.mult


@gin.configurable(module='networks')
class SinePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.shape[1]])

    def index(self, inds):
        return self.pe[0][inds]

    def get(self, size):
        return self.pe[:, :size]


class TreePositionalEncoding(nn.Module):
    def __init__(self, pos_encoding_cls, d_model, max_len, dropout=0.1, concat=False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        if concat:
            self.pe = pos_encoding_cls(d_model // 2, max_len, dropout=0.0)
        else:
            self.pe = pos_encoding_cls(d_model, max_len, dropout=0.0)

        self.concat = concat

    def forward(self, x, parents):
        if self.concat:
            # concat encodings
            enc = x + torch.cat([self.pe.get(x.shape[1]).expand(x.shape[0], -1, -1),
                                 self.pe.index(parents)], dim=2)
            return self.dropout(enc)
        else:
            # sum encodings
            return self.dropout(self.pe(x) + self.pe.index(parents))


@gin.configurable(module='networks')
class ConcatSineTreePositionalEncoding(TreePositionalEncoding):
    def __init__(self, d_model, max_len, dropout=0.1):
        TreePositionalEncoding.__init__(self, SinePositionalEncoding, d_model, max_len,
                                        dropout=dropout, concat=True)


@gin.configurable(module='networks')
class ConcatLearnedTreePositionalEncoding(TreePositionalEncoding):
    def __init__(self, d_model, max_len, dropout=0.1):
        TreePositionalEncoding.__init__(self, LearnedPositionalEncoding, d_model, max_len,
                                        dropout=dropout, concat=True)


@gin.configurable(module='networks')
class SumSineTreePositionalEncoding(TreePositionalEncoding):
    def __init__(self, d_model, max_len, dropout=0.1):
        TreePositionalEncoding.__init__(self, SinePositionalEncoding, d_model, max_len,
                                        dropout=dropout, concat=False)


@gin.configurable(module='networks')
class SumLearnedTreePositionalEncoding(TreePositionalEncoding):
    def __init__(self, d_model, max_len, dropout=0.1):
        TreePositionalEncoding.__init__(self, LearnedPositionalEncoding, d_model, max_len,
                                        dropout=dropout, concat=False)
