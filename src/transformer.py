from .utils import Config

import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
import math
import copy

from typing import Optional, Tuple

T = torch.Tensor
M = nn.Module


def clones(module: M, N: int) -> nn.ModuleList:
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size: int) -> T:
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask_ = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask_ == 0


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features: int, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size: int, dropout: float):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: T, sublayer: M) -> T:
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: nn.Module = nn.ReLU,
    ):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation()

    def forward(self, x: T) -> T:
        o1 = self.activation(self.w_1(x))
        return self.w_2(self.dropout(o1))


def attention(
    query: T, key: T, value: T, mask: T = None, dropout: Optional[nn.Dropout] = None
) -> Tuple[T, T]:
    "Compute 'Scaled Dot Product Attention'"
    # query size: (batch_size, h, seq_len, d_k)
    # key size: (batch_size, h, seq_len, d_k)
    # value size: (batch_size, h, seq_len, d_k)
    # mask size: (batch_size, 1, seq_len, seq_len)
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
        d_k
    )  # batch_size, h, seq_len, seq_len
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # final attention size: (batch_size, h, seq_len, d_k)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: T, key: T, value: T, mask: Optional[T] = None) -> T:
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        del query
        del key
        del value
        return self.linears[-1](x)


class Embeddings(nn.Module):
    def __init__(self, d_model: int, vocab: int) -> None:
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x: T) -> T:
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: T) -> T:
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size: int, self_attn: M, feed_forward: M, dropout: float):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x: T, mask: T) -> T:
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer: M, N: int):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: T, mask: T) -> T:
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, source-attn, and feed forward (defined below)"

    def __init__(
        self, size: int, self_attn: M, source_attn: M, feed_forward: M, dropout: float
    ):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.source_attn = source_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x: T, memory: T, source_mask: T, target_mask: T) -> T:
        "Follow Figure 1 (right) for connections."
        m = memory  # the output of the encoder
        x = self.sublayer[0](
            x, lambda x: self.self_attn(x, x, x, target_mask)
        )  # self attention
        x = self.sublayer[1](
            x, lambda x: self.source_attn(x, m, m, source_mask)
        )  # encoder-decoder attention
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer: M, N: int):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: T, memory: T, source_mask: T, target_mask: T) -> T:
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model: int, vocab: int):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x: T) -> T:
        return log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    """A standard Encoder-Decoder architecture.

    Attributes
    ----------
    source_vocab_size : int
        The size of the source vocabulary.
    target_vocab_size : int
        The size of the target vocabulary.
    N_E : int
        The number of encoder layers.
    N_D : int
        The number of decoder layers.
    d_model : int
        The dimension of the model.
    d_ff : int
        The dimension of the feedforward network model.
    h : int
        The number of heads in the multiheadattention models.
    dropout : float
        The dropout value.
    activation : nn.Module
        The activation function.
    """

    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        N_E: int = 6,
        N_D: int = 6,
        d_model: int = 512,
        d_ff: int = 2048,
        h: int = 8,
        dropout: int = 0.1,
        activation: nn.Module = nn.ReLU,
    ):
        super(Transformer, self).__init__()
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout, activation)
        position = PositionalEncoding(d_model, dropout)

        encoder = EncoderLayer(d_model, c(attn), c(ff), dropout)
        self.encoder = Encoder(encoder, N_E)

        decoder = DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout)
        self.decoder = Decoder(decoder, N_D)

        self.source_embed = nn.Sequential(
            Embeddings(d_model, source_vocab_size), c(position)
        )
        self.target_embed = nn.Sequential(
            Embeddings(d_model, target_vocab_size), c(position)
        )
        self.generator = Generator(d_model, target_vocab_size)

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        source: T,
        target: T,
        source_mask: Optional[T] = None,
        target_mask: Optional[T] = None,
    ) -> T:
        """Forward pass of the transformer model.

        Parameters
        ----------
        source : T
            The source tensor.
        target : T
            The target tensor.
        source_mask : Optional[T], optional
            The source mask tensor, by default None.
        target_mask : Optional[T], optional
            The target mask tensor, by default None.
        """
        if source_mask is None:
            source_mask = torch.ones(source.size(0), source.size(1)).type(torch.uint8)

        if target_mask is None:
            target_mask = subsequent_mask(target.size(1)).type_as(target)

        encoded = self.encode(source, source_mask)
        decoded = self.decode(encoded, target, source_mask, target_mask)
        return decoded

    def encode(self, source: T, source_mask: Optional[T] = None) -> T:
        if source_mask is None:
            source_mask = torch.ones(source.size(0), source.size(1)).type(torch.uint8)

        return self.encoder(self.source_embed(source), source_mask)

    def decode(
        self,
        memory: T,
        target: T = None,
        source_mask: Optional[T] = None,
        target_mask: Optional[T] = None,
    ) -> T:

        if source_mask is None:
            source_mask = torch.ones(memory.size(0), memory.size(1)).type(torch.uint8)

        if target_mask is None:
            target_mask = subsequent_mask(target.size(1)).type_as(target)

        # print(target.dtype)
        return self.decoder(self.target_embed(target), memory, source_mask, target_mask)

    def generate(self, x: T) -> T:
        return self.generator(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransformerConfig(Config):
    ALLOWED_KEYS = [
        "source_vocab_size",
        "target_vocab_size",
        "N_E",
        "N_D",
        "d_model",
        "d_ff",
        "h",
        "dropout",
        "activation",
    ]
    ConfigFor = Transformer


class DecoderOnlyTransformer(nn.Module):
    """A standard Decoder architecture.

    Attributes
    ----------
    target_vocab_size : int
        The size of the target vocabulary.
    N : int
        The number of decoder layers.
    d_model : int
        The dimension of the model.
    d_ff : int
        The dimension of the feedforward network model.
    h : int
        The number of heads in the multiheadattention models.
    dropout : float
        The dropout value.
    activation : nn.Module
        The activation function.
    """

    def __init__(
        self,
        target_vocab_size: int,
        N: int = 6,
        d_model: int = 512,
        d_ff: int = 2048,
        h: int = 8,
        dropout: int = 0.1,
        activation: nn.Module = nn.ReLU,
    ):
        super(DecoderOnlyTransformer, self).__init__()
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout, activation)
        position = PositionalEncoding(d_model, dropout)

        decoder = EncoderLayer(d_model, c(attn), c(ff), dropout)
        self.decoder = Encoder(decoder, N)

        self.target_embed = nn.Sequential(
            Embeddings(d_model, target_vocab_size), c(position)
        )
        self.generator = Generator(d_model, target_vocab_size)

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, target: T, target_mask: Optional[T] = None) -> T:

        if target_mask is None:
            target_mask = subsequent_mask(target.size(1)).type_as(target)

        decoded = self.decoder(self.target_embed(target), target_mask)
        return decoded

    def decode(self, target: T, target_mask: Optional[T] = None) -> T:
        return self.forward(target, target_mask)

    def generate(self, x: T) -> T:
        return self.generator(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DecoderOnlyTransformerConfig(Config):
    ALLOWED_KEYS = [
        "target_vocab_size",
        "N",
        "d_model",
        "d_ff",
        "h",
        "dropout",
        "activation",
    ]
    ConfigFor = DecoderOnlyTransformer
