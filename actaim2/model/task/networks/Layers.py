"""
---------------------------------------------------------------------
-- Author: Jhosimar George Arias Figueroa
---------------------------------------------------------------------

Custom Layers

"""
import torch
from torch import nn
from torch.nn import functional as F
from monai.networks.blocks.mlp import MLPBlock
import math
import pdb

# Flatten layer
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


# Reshape layer
class Reshape(nn.Module):
    def __init__(self, outer_shape):
        super(Reshape, self).__init__()
        self.outer_shape = outer_shape

    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)


# Sample from the Gumbel-Softmax distribution and optionally discretize.
class GumbelSoftmax(nn.Module):

    def __init__(self, f_dim, c_dim):
        super(GumbelSoftmax, self).__init__()
        self.logits = nn.Linear(f_dim, c_dim)
        self.f_dim = f_dim
        self.c_dim = c_dim

    def sample_gumbel(self, shape, is_cuda=False, eps=1e-20):
        U = torch.rand(shape)
        if is_cuda:
            U = U.cuda()
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size(), logits.is_cuda)
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, hard=False):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        # categorical_dim = 10
        y = self.gumbel_softmax_sample(logits, temperature)

        if not hard:
            return y

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard

    def forward(self, x, temperature=1.0, hard=False):
        logits = self.logits(x).view(-1, self.c_dim)
        prob = F.softmax(logits, dim=-1)
        y = self.gumbel_softmax(logits, temperature, hard)
        return logits, prob, y


# Sample from a Gaussian distribution
class Gaussian(nn.Module):
    def __init__(self, in_dim, z_dim):
        super(Gaussian, self).__init__()
        self.mu = nn.Linear(in_dim, z_dim)
        self.var = nn.Linear(in_dim, z_dim)

    def reparameterize(self, mu, var):
        std = torch.sqrt(var + 1e-10)
        noise = torch.randn_like(std)
        z = mu + noise * std
        return z

    def forward(self, x):
        mu = self.mu(x)
        var = F.softplus(self.var(x))
        z = self.reparameterize(mu, var)
        return mu, var, z


# Create a Transformer block for cross attention layer
class TransformerBlock(nn.Module):
    """
    A transformer block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    Args:
        hidden_size: dimension of hidden layer.
        mlp_dim: dimension of feedforward layer.
        num_heads: number of attention heads.
        dropout_rate: faction of the input units to drop.
        qkv_bias: apply bias term for the qkv linear layer
        causal: whether to use causal attention.
        sequence_length: if causal is True, it is necessary to specify the sequence length.
        with_cross_attention: Whether to use cross attention for conditioning.
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        causal: bool = False,
        sequence_length = 5,
        with_cross_attention: bool = False,
    ) -> None:
        self.with_cross_attention = with_cross_attention
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SABlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            qkv_bias=qkv_bias,
            causal=causal,
            sequence_length=sequence_length,
        )

        self.norm2 = None
        self.cross_attn = None
        if self.with_cross_attention:
            self.norm2 = nn.LayerNorm(hidden_size)
            self.cross_attn = SABlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                qkv_bias=qkv_bias,
                with_cross_attention=with_cross_attention,
                causal=False,
            )

        self.norm3 = nn.LayerNorm(hidden_size)
        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)

    def forward(self, x: torch.Tensor, context=None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        if self.with_cross_attention:
            x = x + self.cross_attn(self.norm2(x), context=context)
        x = x + self.mlp(self.norm3(x))
        return x


class SABlock(nn.Module):
    """
    A self-attention block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    Args:
        hidden_size: dimension of hidden layer.
        num_heads: number of attention heads.
        dropout_rate: dropout ratio. Defaults to no dropout.
        qkv_bias: bias term for the qkv linear layer.
        causal: whether to use causal attention.
        sequence_length: if causal is True, it is necessary to specify the sequence length.
        with_cross_attention: Whether to use cross attention for conditioning.
    """

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            qkv_bias: bool = False,
            causal: bool = False,
            sequence_length = 5,
            with_cross_attention: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.causal = causal
        self.sequence_length = sequence_length
        self.with_cross_attention = with_cross_attention

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        self.dropout_rate = dropout_rate

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        if causal and sequence_length is None:
            raise ValueError("sequence_length is necessary for causal attention.")

        # key, query, value projections
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)

        # regularization
        self.drop_weights = nn.Dropout(dropout_rate)
        self.drop_output = nn.Dropout(dropout_rate)

        # output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        if causal and sequence_length is not None:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(sequence_length, sequence_length)).view(1, 1, sequence_length, sequence_length),
            )

    def forward(self, x: torch.Tensor, context=None) -> torch.Tensor:
        b, t, c = x.size()  # batch size, sequence length, embedding dimensionality (hidden_size)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query = self.to_q(x)

        kv = context if context is not None else x
        _, kv_t, _ = kv.size()
        key = self.to_k(kv)
        value = self.to_v(kv)

        query = query.view(b, t, self.num_heads, c // self.num_heads)  # (b, t, nh, hs)
        key = key.view(b, kv_t, self.num_heads, c // self.num_heads)  # (b, kv_t, nh, hs)
        value = value.view(b, kv_t, self.num_heads, c // self.num_heads)  # (b, kv_t, nh, hs)

        query = query.transpose(1, 2)  # (b, nh, t, hs)
        key = key.transpose(1, 2)  # (b, nh, kv_t, hs)
        value = value.transpose(1, 2)  # (b, nh, kv_t, hs)

        # manual implementation of attention
        query = query * self.scale
        attention_scores = query @ key.transpose(-2, -1)

        if self.causal:
            attention_scores = attention_scores.masked_fill(self.causal_mask[:, :, :t, :kv_t] == 0, float("-inf"))

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.drop_weights(attention_probs)
        y = attention_probs @ value  # (b, nh, t, kv_t) x (b, nh, kv_t, hs) -> (b, nh, t, hs)

        y = y.transpose(1, 2)  # (b, nh, t, hs) -> (b, t, nh, hs)

        y = y.contiguous().view(b, t, c)  # re-assemble all head outputs side by side

        y = self.out_proj(y)
        y = self.drop_output(y)
        return y