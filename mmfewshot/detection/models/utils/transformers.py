import numpy as np
import torch
import torch.nn as nn


'''
Create a symbolic link to this file in:
    miniconda3/envs/openmmlab/lib/python3.7/site-packages/mmdet/models/utils

Auxiliary classes for positional encodings taken from:
    https://github.com/tatp22/multidim-positional-encoding
'''
def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding1D(nn.Module):
    """Sinusoidal positional encoding, as described in Attention Is All You Need.

    """
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super().__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc


class MultiHeadAttention(nn.Module):
    """Multi-head attention. Based on https://arxiv.org/abs/1706.03762.

    Args:
        num_heads (int): Number of heads used in Multi-head attention.
        embed_size (int): dimension for the transformer embeddings.

    """
    def __init__(self,
                 num_heads: int,
                 embed_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.W_q = nn.Linear(in_features=self.embed_size, out_features=self.embed_size)
        self.W_k = nn.Linear(in_features=self.embed_size, out_features=self.embed_size)
        self.W_v = nn.Linear(in_features=self.embed_size, out_features=self.embed_size)
        self.W = nn.Linear(in_features=self.embed_size, out_features=self.embed_size)

    def forward(self, queries, keys, values):
        """"
        Args:
            queries (Tensor): Queries with shape (N, H * W, embed_size).
            keys (Tensor): Input support features with shape (N, H * W, embed_size).
            values (Tensor): Input support features with shape (N, H * W, embed_size).

        Returns:
            Tensor: Multi-head attention with shape (N, H * W, embed_size).

        """

        '''
        print("         Entering forward in MultiHeadAttention...")
        print(f"         queries.size() = {queries.size()}")
        print(f"         keys.size() = {keys.size()}")
        print(f"         values.size() = {values.size()}")
        '''

        queries = self.W_q(queries)
        keys = self.W_k(keys)
        values = self.W_v(values)

        '''
        print("           Sizes after transformation:")
        print(f"         queries.size() = {queries.size()}")
        print(f"         keys.size() = {keys.size()}")
        print(f"         values.size() = {values.size()}")
        '''

        # Split embeddings into the different heads
        queries = torch.reshape(queries, (queries.size()[0], queries.size()[1], self.num_heads, -1))
        keys = torch.reshape(keys, (keys.size()[0], keys.size()[1], self.num_heads, -1))
        values = torch.reshape(values, (values.size()[0], values.size()[1], self.num_heads, -1))

        '''
        print(f"         Splitting heads...")
        print(f"         queries.size() = {queries.size()}")
        print(f"         keys.size() = {keys.size()}")
        print(f"         values.size() = {values.size()}")

        print(f"         Computing einsum...")
        '''

        # Compute attention just like in the paper
        attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # (N, num_heads, queries_len, keys_len)
        attention = torch.softmax(attention / (self.embed_size ** 0.5), dim=3)
        attention = torch.einsum("nhqk,nkhd->nqhd", [attention, values])  # (N, queries_len, num_heads, head_dim)
        attention = torch.reshape(attention, (attention.size()[0], attention.size()[1], -1))
        attention = self.W(attention)
        return attention


class TransformerBlock(nn.Module):
    """Transformer block. Based on https://arxiv.org/abs/1706.03762.

    Args:
        num_heads (int): Number of heads used in Multi-head attention.
        embed_size (int): dimension for the transformer embeddings.
        forward_expansion (int): Factor of embed_size for the first FFN after
            multi-head attention.
        dropout_prob (float): Dropout probability.

    """
    def __init__(self,
                 num_heads: int,
                 embed_size: int,
                 forward_expansion: int,
                 dropout_prob: float):
        super().__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.forward_expansion = forward_expansion
        self.dropout_prob = dropout_prob
        self.attention_layer = MultiHeadAttention(num_heads=self.num_heads,
                                                  embed_size=self.embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(in_features=self.embed_size, out_features=self.embed_size * self.forward_expansion),
            nn.ReLU(),
            nn.Linear(in_features=self.embed_size * self.forward_expansion, out_features=self.embed_size),
        )

        self.dropout = nn.Dropout(self.dropout_prob)
        self.norm1 = nn.LayerNorm(self.embed_size)
        self.norm2 = nn.LayerNorm(self.embed_size)

    def forward(self, queries, keys, values):
        """Calculate a single Transformer layer.

        Args:
            queries (Tensor): Queries with shape (N, H * W, embed_size).
            keys (Tensor): Input support features with shape (N, H * W, embed_size).
            values (Tensor): Input support features with shape (N, H * W, embed_size).

        Returns:
            Tensor: Attention feature map with shape (N, H * W, embed_size).
        """
        # print("        Entering forward in TransformerBlock...")

        attention = self.attention_layer(queries=queries,
                                         keys=keys,
                                         values=values)
        # print("          MultiHeadAttention computed!")
        x = self.dropout(self.norm1(attention + queries))
        ffn = self.feed_forward(x)
        x = self.dropout(self.norm2(ffn + x))

        return x


class CrossAttentionTransformer(nn.Module):
    """A single CAT layer. Based on https://arxiv.org/abs/2104.14984.

    Args:
        num_heads (int): Number of heads used in each CAT layer for Multi-head
            attention.
        embed_size (int): dimension for the hidden embeddings.
        forward_expansion (int): Factor of embed_size for the first FFN after
            multi-head attention.
        pos_encoding (bool): Either add sinusoidal positional encodings or not.
        dropout_prob (float): Dropout probability.

    """
    def __init__(self,
                 num_heads: int,
                 embed_size: int,
                 forward_expansion: int,
                 pos_encoding: bool,
                 dropout_prob: float):
        super().__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.forward_expansion = forward_expansion
        self.pos_encoding = pos_encoding
        self.dropout_prob = dropout_prob
        if self.pos_encoding:
            self.query_pos_encoding = PositionalEncoding1D(self.embed_size)
            self.support_pos_encoding = PositionalEncoding1D(self.embed_size)
        self.transformer_block = TransformerBlock(num_heads=self.num_heads,
                                                  embed_size=self.embed_size,
                                                  forward_expansion=self.forward_expansion,
                                                  dropout_prob=self.dropout_prob)

    def forward(self, x_query, x_support):
        """Calculate a single layer of CAT. Based on https://arxiv.org/abs/2104.14984

        Args:
            x_query (Tensor): Input query features with shape (N, H_q * W_q, embed_size).
            x_support (Tensor): Input support features with shape (N, H_s * W_s, embed_size).

        Returns:
            Tensor: Aggregated features for query, its shape is (N, H_q * W_q, embed_size).
            Tensor: Aggregated features for support, its shape is (N, H_s * W_s, embed_size).

        """

        # print("    Entering forward in CrossAttentionTransformer...")

        if self.pos_encoding:
            x_query_pos = x_query + self.query_pos_encoding(x_query)
            x_support_pos = x_support + self.support_pos_encoding(x_support)

            # print("      Positional encodings computed!")
            # print("      Applying TransformerBlock...")

            x_query = self.transformer_block(queries=x_query_pos, keys=x_support_pos, values=x_support)
            # print(f"     TransformerBlock for x_query ready!")
            x_support = self.transformer_block(queries=x_support_pos, keys=x_query_pos, values=x_query)
        else:
            x_query = self.transformer_block(queries=x_query, keys=x_support, values=x_support)
            x_support = self.transformer_block(queries=x_support, keys=x_query, values=x_query)

        return x_query, x_support


class CompressionBlock(nn.Module):
    """ Change the number of channels from feature maps
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 padding='same'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.query_compression = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 3, padding='same'),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.Conv2d(self.in_channels, self.out_channels, 1, padding='same'),
        )
        self.support_compression = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 3, padding='same'),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.Conv2d(self.in_channels, self.out_channels, 1, padding='same'),
        )

    def forward(self, x_query, x_support):
        """

        Args:
            x_query (Tensor): Input query features with shape (N, in_channels, H_q, W_q)
            x_support (Tensor): Input support features with shape (N, in_channels, H_s, W_s)

        Returns:
            Tensor: Compressed feature map for query with shape (N, out_channels, H_q, W_q).
            Tensor: Compressed feature map for support with shape (N, out_channels, H_s, W_s).

        """

        x_query = self.query_compression(x_query)
        x_support = self.support_compression(x_support)

        return x_query, x_support


class CrossAttentionTransformerBlock(nn.Module):
    """Cross-Attention Transformer block. Based on https://arxiv.org/abs/2104.14984.

    Args:
        in_channels (int): Number of input features channels.
        num_layers (int): Number of CAT layers.
        num_heads (int): Number of heads used in each CAT layer for Multi-head
            attention.
        embed_size (int): Dimension for the hidden embeddings.
        forward_expansion (int): Factor of embed_size for the first FFN after
            multi-head attention.
        pos_encoding (bool): Either add sinusoidal positional encodings or not.
        dropout_prob (float): Dropout probability.

    """
    def __init__(self,
                 in_channels: int,
                 num_layers: int,
                 num_heads: int,
                 embed_size: int,
                 forward_expansion: int,
                 pos_encoding: bool,
                 dropout_prob: float):
        super().__init__()
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.forward_expansion = forward_expansion
        self.pos_encoding = pos_encoding
        self.dropout_prob = dropout_prob
        self.with_compression = False
        # if embed_size < in_channels:
        '''
        self.with_compression = True
        self.compression_layer = CompressionBlock(in_channels=self.in_channels,
                                                  out_channels=self.embed_size)
        self.expansion_layer = CompressionBlock(in_channels=self.embed_size,
                                                out_channels=self.in_channels)
        '''
        self.dropout = nn.Dropout(dropout_prob)
        self.layers = nn.ModuleList(
            [
                CrossAttentionTransformer(num_heads=self.num_heads,
                                          embed_size= self.embed_size,
                                          forward_expansion=self.forward_expansion,
                                          pos_encoding=self.pos_encoding,
                                          dropout_prob=self.dropout_prob,
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, x_query, x_support):
        """Calculate num_layers stacked layers of CAT. Based on https://arxiv.org/abs/2104.14984

        Args:
            x_query (Tensor): Input query features with shape (N, C, H_q, W_q).
            x_support (Tensor): Input support features with shape (N, C, H_s, W_s).

        Returns:
            Tensor: Aggregated features for query, its shape is (N, C, H_q, W_q).
            Tensor: Aggregated features for support, its shape is (N, C, H_s, W_s).

        """

        '''
        print("ENTERING forward in CrossAttentionTransformerBlock...")
        print(f"  x_query.size() = {x_query.size()}")
        print(f"  x_support.size() = {x_support.size()}")
        '''

        # if self.with_compression:
        # x_query, x_support = self.compression_layer(x_query, x_support)

        '''
        print("  Sizes after channel compression:")
        print(f"    x_query.size() = {x_query.size()}")
        print(f"    x_support.size() = {x_support.size()}")
        '''

        # Save original sizes to reshape the output from attention
        x_query_shape = x_query.size()
        x_support_shape = x_support.size()

        # Flatten spatial dimensions
        x_query = torch.flatten(x_query, start_dim=-2, end_dim=-1)      # (N, embed_size, H_q * W_q)
        x_support = torch.flatten(x_support, start_dim=-2, end_dim=-1)  # (N, embed_size, H_s * W_s)


        '''
        print("  Sizes after flatten of spatial dimensions:")
        print(f"    x_query.size() = {x_query.size()}")
        print(f"    x_support.size() = {x_support.size()}")
        '''

        # Permute last dimensions
        x_query = torch.permute(x_query, (0, 2, 1))      # (N, H_q * W_q, embed_size)
        x_support = torch.permute(x_support, (0, 2, 1))  # (N, H_s * W_s, embed_size)

        '''
        print("  Sizes after dims permutation:")
        print(f"    x_query.size() = {x_query.size()}")
        print(f"    x_support.size() = {x_support.size()}")
        '''

        x_query = self.dropout(x_query)
        x_support = self.dropout(x_support)

        for ix, layer in enumerate(self.layers):  # TODO: delete enumerate
            # print(f"  Layer {ix} of CrossAttentionTransformerBlock...")
            # print(f"    x_query.size() = {x_query.size()}")
            # print(f"    x_support.size() = {x_support.size()}")
            x_query, x_support = layer(x_query, x_support)

        # Reshape both query and support features to its original sizes
        # print("  RESHAPING!")
        # x_query_shape = torch.Size([x_support_shape[0], x_query_shape[1], x_query_shape[2], x_query_shape[3]])  # not needed if query was already repeated
        # print(f"  x_query.size() = {x_query.size()} --> x_query_shape = {x_query_shape}")
        # print(f"  x_support.size() = {x_support.size()} --> x_support_shape = {x_support_shape}")
        x_query = torch.reshape(x_query, x_query_shape)
        x_support = torch.reshape(x_support, x_support_shape)

        # if self.with_compression:
        # x_query, x_support = self.expansion_layer(x_query, x_support)

        '''
        print("  Sizes after channel expansion:")
        print(f"    x_query.size() = {x_query.size()}")
        print(f"    x_support.size() = {x_support.size()}")
        '''

        return x_query, x_support
